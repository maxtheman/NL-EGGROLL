"""
Minimal int8-only training loop on TinyStories tokens using eggroll updates.
Model: take last token embedding (int8) -> int8 linear to vocab logits.
Uses ES-style sign updates via apply_sign_update.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import mlx.core as mx

from eggroll_mlx import (
    NoiseConfig,
    apply_sign_update,
    calibrate_divisor,
    convert_fitnesses,
    do_mm,
    generate_big_rand,
    int8_matmul,
)


def load_tokens(path: Path):
    return np.memmap(path, mode="r", dtype=np.uint16)


def get_batch(memmap, seq_len, batch_size, vocab_size, rng):
    # memmap is flat; reshape to (-1, seq_len)
    total_tokens = memmap.shape[0]
    n_seq = total_tokens // seq_len
    idx = rng.integers(0, n_seq - batch_size)
    arr = np.array(memmap[idx * seq_len : (idx + batch_size) * seq_len], copy=False)
    arr = arr.reshape(batch_size, seq_len)
    x = arr[:, :-1]
    y = arr[:, -1]  # predict last token
    x = mx.array(x.astype(np.int32))
    y = mx.array(y.astype(np.int32))
    return x, y


def quantize(arr, fixed_point, scale=1.0):
    scaled = np.clip(np.round(arr * scale * (2**fixed_point)), -127, 127).astype(np.int8)
    return mx.array(scaled, dtype=mx.int8)


def cross_entropy(logits, targets):
    vocab = logits.shape[-1]
    one_hot = (targets[..., None] == mx.arange(vocab)).astype(mx.float32)
    max_logits = mx.max(logits, axis=-1, keepdims=True)
    logsumexp = max_logits + mx.log(mx.sum(mx.exp(logits - max_logits), axis=-1, keepdims=True))
    log_probs = logits - logsumexp
    return -mx.mean(mx.sum(one_hot * log_probs, axis=-1))


@dataclass
class Int8Linear:
    weight: mx.array  # (out, in)
    scale: int  # divisor after matmul


@dataclass
class TransformerWeights:
    tok_emb: mx.array  # (vocab, d_model)
    pos_emb: mx.array  # (seq_len, d_model)
    attn_q: Int8Linear
    attn_k: Int8Linear
    attn_v: Int8Linear
    attn_out: Int8Linear
    mlp_in: Int8Linear
    mlp_out: Int8Linear
    lm_head: Int8Linear


def calibrate_linear(x_sample: mx.array, w_int8: mx.array, target_max=32) -> int:
    div = calibrate_divisor(x_sample.astype(mx.int8), w_int8, target_max=target_max)
    return max(1, div)


def forward_int8(cfg: NoiseConfig, w: TransformerWeights, x_tokens: mx.array, big_rand, epoch, base_seed):
    # x_tokens: (batch, seq)
    B, S = x_tokens.shape
    d_model = w.tok_emb.shape[1]
    # embeddings
    tok = w.tok_emb[x_tokens]  # (B,S,d)
    pos = w.pos_emb[:S]
    h = mx.clip(tok + pos, -127, 127).astype(mx.int8)

    def proj(h_in, linear: Int8Linear, tid_offset):
        in_dim = linear.weight.shape[1]
        out = do_mm(cfg, h_in.reshape(-1, in_dim), linear.weight, big_rand, epoch, thread_id=tid_offset, base_seed=base_seed)
        out = (out // linear.scale).astype(mx.int8)
        out_dim = linear.weight.shape[0]
        return mx.reshape(out, (B, S, out_dim))

    # Self-attention with float softmax (float scores only, projections int8)
    q = proj(h, w.attn_q, tid_offset=0)
    k = proj(h, w.attn_k, tid_offset=1)
    v = proj(h, w.attn_v, tid_offset=2)
    d_head = q.shape[-1]
    scores = mx.matmul(q.astype(mx.float32), mx.swapaxes(k.astype(mx.float32), -1, -2)) / float(np.sqrt(d_head))
    attn = mx.softmax(scores, axis=-1)
    ctx = mx.matmul(attn, v.astype(mx.float32)).astype(mx.int8)
    attn_out = proj(ctx, w.attn_out, tid_offset=3)

    # MLP
    mlp_hidden = proj(attn_out, w.mlp_in, tid_offset=4)
    mlp_hidden = mx.maximum(mlp_hidden, mx.zeros_like(mlp_hidden))
    mlp_out = proj(mlp_hidden, w.mlp_out, tid_offset=5)

    # LM head on last token
    last = mlp_out[:, -1, :]
    logits_int = do_mm(cfg, last, w.lm_head.weight, big_rand, epoch, thread_id=6, base_seed=base_seed)
    logits = (logits_int // w.lm_head.scale).astype(mx.float32)
    return logits


def init_weights(cfg: NoiseConfig, vocab_size: int, seq_len: int, d_model: int, d_head: int, d_ff: int, rng, init_scale):
    def q(shape):
        return quantize(rng.normal(0, 0.05, size=shape), cfg.fixed_point, scale=init_scale)

    tok_emb = q((vocab_size, d_model))
    pos_emb = q((seq_len, d_model))
    attn_q_w = q((d_head, d_model))
    attn_k_w = q((d_head, d_model))
    attn_v_w = q((d_head, d_model))
    attn_out_w = q((d_model, d_head))
    mlp_in_w = q((d_ff, d_model))
    mlp_out_w = q((d_model, d_ff))
    lm_head_w = q((vocab_size, d_model))

    # simple calibration using small random batch
    dummy_ids = mx.array(rng.integers(0, vocab_size, size=(32,)), dtype=mx.int32)
    dummy_emb = tok_emb[dummy_ids]  # (32, d_model)
    target = 16
    scales = {
        "q": calibrate_linear(dummy_emb, attn_q_w, target_max=target),
        "k": calibrate_linear(dummy_emb, attn_k_w, target_max=target),
        "v": calibrate_linear(dummy_emb, attn_v_w, target_max=target),
        "o": calibrate_linear(dummy_emb.reshape(-1, d_head), attn_out_w, target_max=target),
        "mlp_in": calibrate_linear(dummy_emb, mlp_in_w, target_max=target),
        "mlp_out": calibrate_linear(dummy_emb.reshape(-1, d_ff), mlp_out_w, target_max=target),
        "lm": calibrate_linear(dummy_emb, lm_head_w, target_max=target),
    }

    return TransformerWeights(
        tok_emb=tok_emb,
        pos_emb=pos_emb,
        attn_q=Int8Linear(attn_q_w, scales["q"]),
        attn_k=Int8Linear(attn_k_w, scales["k"]),
        attn_v=Int8Linear(attn_v_w, scales["v"]),
        attn_out=Int8Linear(attn_out_w, scales["o"]),
        mlp_in=Int8Linear(mlp_in_w, scales["mlp_in"]),
        mlp_out=Int8Linear(mlp_out_w, scales["mlp_out"]),
        lm_head=Int8Linear(lm_head_w, scales["lm"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/tinystories")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_head", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--pop_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--fixed_point", type=int, default=2)
    parser.add_argument("--sigma_shift", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="checkpoints/int8_baseline", help="where to save emb/w_out")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--init_scale", type=float, default=1.0, help="multiplicative scale before quantize")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    train_path = Path(args.data_dir) / "train_tokens.npy"
    if not train_path.exists():
        train_path = Path(args.data_dir) / "train_tokens.uint16.memmap"
    memmap = load_tokens(train_path)

    cfg = NoiseConfig(fixed_point=args.fixed_point, sigma_shift=args.sigma_shift, rank=1)
    base_seed = args.seed
    big_rand = generate_big_rand(4_000_000, seed=args.seed, fixed_point=cfg.fixed_point, dtype=mx.int8)

    weights = init_weights(cfg, args.vocab_size, args.seq_len, args.d_model, args.d_head, args.d_ff, rng, args.init_scale)

    for step in range(args.steps):
        x, y = get_batch(memmap, args.seq_len, batch_size=args.batch_size, vocab_size=args.vocab_size, rng=rng)
        rewards = []
        for j in range(args.pop_size):
            logits = forward_int8(cfg, weights, x, big_rand, epoch=step, base_seed=base_seed)
            rewards.append(-cross_entropy(logits, y).item())
        rewards = mx.array(rewards)
        fitnesses = convert_fitnesses(cfg, rewards)
        # Update only lm_head for now (extend to all matrices iteratively)
        weights.lm_head.weight = apply_sign_update(cfg, weights.lm_head.weight, fitnesses, big_rand, step, base_seed)

        logits_eval = forward_int8(cfg, weights, x, big_rand=None, epoch=None, base_seed=None)
        loss = cross_entropy(logits_eval, y)
        lmax = float(np.array(logits_eval).max())
        print(f"step {step}: loss={loss.item():.4f}, logits_max={lmax:.1f}")

    # Save learned parameters
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "tok_emb.npy", np.array(weights.tok_emb))
    np.save(save_dir / "pos_emb.npy", np.array(weights.pos_emb))
    np.save(save_dir / "attn_q.npy", np.array(weights.attn_q.weight))
    np.save(save_dir / "attn_k.npy", np.array(weights.attn_k.weight))
    np.save(save_dir / "attn_v.npy", np.array(weights.attn_v.weight))
    np.save(save_dir / "attn_out.npy", np.array(weights.attn_out.weight))
    np.save(save_dir / "mlp_in.npy", np.array(weights.mlp_in.weight))
    np.save(save_dir / "mlp_out.npy", np.array(weights.mlp_out.weight))
    np.save(save_dir / "lm_head.npy", np.array(weights.lm_head.weight))
    np.save(save_dir / "scales.npy", np.array([weights.attn_q.scale, weights.attn_k.scale, weights.attn_v.scale, weights.attn_out.scale, weights.mlp_in.scale, weights.mlp_out.scale, weights.lm_head.scale]))
    print("Saved checkpoint to", save_dir)


if __name__ == "__main__":
    main()
