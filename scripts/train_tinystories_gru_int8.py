"""
TinyStories int8 GRU trained with Eggroll-style ES updates.
- Uses weight-only int8 + fixed-point scaling for matmuls (do_mm).
- Activations for gates are float for nonlinearities; hidden state is re-quantized to int8 each step.
- Updates all trainable matrices with sign updates from paired fitnesses.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import mlx.core as mx

from eggroll_api import EggrollContext, make_context
from eggroll_mlx import (
    NoiseConfig,
    apply_sign_update,
    calibrate_divisor,
    convert_fitnesses,
    do_mm,
    int8_matmul,
    _stack_lora_params,
)


def load_tokens(path: Path):
    return np.memmap(path, mode="r", dtype=np.uint16)


def get_batch(memmap, seq_len: int, batch_size: int, rng) -> Tuple[mx.array, mx.array]:
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
class Int8Mat:
    weight: mx.array  # (out, in)
    scale: int  # divisor after matmul


@dataclass
class GRUWeights:
    tok_emb: mx.array  # (vocab, d_model)
    pos_emb: mx.array  # (seq_len, d_model)
    gru_x: Int8Mat  # input projection (3H x D)
    gru_h: Int8Mat  # hidden projection (3H x H)
    lm_head: Int8Mat  # (vocab, H)


def calibrate_linear(x_sample: mx.array, w_int8: mx.array, target_max=256) -> int:
    # For now, keep divisor at 1 to avoid double-scaling on top of do_mm normalization.
    return 1


def forward_gru_int8(ctx: EggrollContext, w: GRUWeights, x_tokens: mx.array, thread_id: Optional[int]) -> mx.array:
    B, S = x_tokens.shape
    H = w.gru_h.weight.shape[1]

    tok = w.tok_emb[x_tokens]  # (B,S,D)
    pos = w.pos_emb[:S]
    x_emb = mx.clip(tok + pos, -127, 127).astype(mx.int8)  # (B,S,D)

    h = mx.zeros((B, H), dtype=mx.int8)

    def proj(mat: Int8Mat, inp: mx.array, tid_offset: int) -> mx.array:
        tid = None if thread_id is None else thread_id + tid_offset
        out = ctx.forward(inp, mat.weight, thread_id=tid)
        return (out // mat.scale).astype(mx.int8)

    for t in range(S):
        x_t = x_emb[:, t, :]  # (B,D)
        gates_x = proj(w.gru_x, x_t, tid_offset=0)  # (B,3H)
        gates_h = proj(w.gru_h, h, tid_offset=2)  # (B,3H) offset even to preserve parity
        gates = (gates_x.astype(mx.int32) + gates_h.astype(mx.int32)).astype(mx.float32)
        gates = mx.reshape(gates, (B, 3, H))
        r = mx.sigmoid(gates[:, 0, :])
        z = mx.sigmoid(gates[:, 1, :])
        n = mx.tanh(gates[:, 2, :] + r * gates[:, 0, :])
        h = mx.round((1 - z) * n + z * h.astype(mx.float32)).astype(mx.int8)

    logits_int = ctx.forward(h, w.lm_head.weight, thread_id=None if thread_id is None else thread_id + 4)
    logits = (logits_int // w.lm_head.scale).astype(mx.float32)
    return logits


def forward_gru_int8_batched(ctx: EggrollContext, w: GRUWeights, x_tokens: mx.array, thread_ids: list[int]) -> mx.array:
    """
    Vectorized over population dimension (thread_ids). Still loops over sequence length,
    but gate matmuls are vmapped across pop to avoid Python loops.
    """
    B, S = x_tokens.shape
    H = w.gru_h.weight.shape[1]
    P = len(thread_ids)

    tok = w.tok_emb[x_tokens]  # (B,S,D)
    pos = w.pos_emb[:S]
    x_emb = mx.clip(tok + pos, -127, 127).astype(mx.int8)  # (B,S,D)

    h = mx.zeros((P, B, H), dtype=mx.int8)
    x_broadcast = mx.broadcast_to(x_emb[None, ...], (P, B, S, x_emb.shape[-1]))

    def mm_pop(mat: Int8Mat, inp: mx.array, tids: list[int], offset: int) -> mx.array:
        # inp: (P,B,K)
        tids_off = [t + offset for t in tids]
        P, B, K = inp.shape
        x_flat = mx.reshape(inp, (P * B, K))
        base_out = int8_matmul(x_flat, mat.weight)  # (P*B, out)
        base_out = mx.reshape(base_out, (P, B, mat.weight.shape[0]))

        # low-rank delta per thread_id
        A_stack, B_stack = _stack_lora_params(ctx.cfg, ctx.big_rand, ctx.epoch, tids_off, mat.weight.shape, ctx.base_seed)
        x_int = inp.astype(mx.int32)
        proj = mx.sum(x_int[:, :, None, :] * B_stack[:, None, :, :], axis=-1)  # (P,B,rank)
        delta = mx.sum(proj[:, :, None, :] * A_stack[:, None, :, :], axis=-1).astype(mx.int32)  # (P,B,out)
        delta = delta >> (ctx.cfg.fixed_point + ctx.cfg.sigma_shift)

        out_int = base_out + delta
        denom = int(np.sqrt(mat.weight.shape[1]))
        scale = (2 ** ctx.cfg.fixed_point) * max(1, denom)
        out = out_int // scale
        return mx.clip(out // mat.scale, -127, 127).astype(mx.int8)

    for t in range(S):
        x_t = x_broadcast[:, :, t, :]  # (P,B,D)
        gates_x = mm_pop(w.gru_x, x_t, thread_ids, offset=0)  # (P,B,3H)
        gates_h = mm_pop(w.gru_h, h, thread_ids, offset=2)  # (P,B,3H)
        gates = (gates_x.astype(mx.int32) + gates_h.astype(mx.int32)).astype(mx.float32)
        gates = mx.reshape(gates, (P, B, 3, H))
        r = mx.sigmoid(gates[:, :, 0, :])
        z = mx.sigmoid(gates[:, :, 1, :])
        n = mx.tanh(gates[:, :, 2, :] + r * gates[:, :, 0, :])
        h = mx.round((1 - z) * n + z * h.astype(mx.float32)).astype(mx.int8)

    logits_int = mm_pop(w.lm_head, h, thread_ids, offset=4)  # (P,B,vocab) int8
    return logits_int.astype(mx.float32)


def init_weights(cfg: NoiseConfig, vocab_size: int, seq_len: int, d_model: int, d_hidden: int, rng, init_scale: float) -> GRUWeights:
    def q(shape):
        return quantize(rng.normal(0, 0.05, size=shape), cfg.fixed_point, scale=init_scale)

    tok_emb = q((vocab_size, d_model))
    pos_emb = q((seq_len, d_model))
    gru_x_w = q((3 * d_hidden, d_model))
    gru_h_w = q((3 * d_hidden, d_hidden))
    lm_head_w = q((vocab_size, d_hidden))

    dummy_ids = mx.array(rng.integers(0, vocab_size, size=(32,)), dtype=mx.int32)
    dummy_tok = tok_emb[dummy_ids]
    s_tok = calibrate_linear(dummy_tok, gru_x_w)
    s_h = calibrate_linear(mx.zeros((32, d_hidden), dtype=mx.int8), gru_h_w)
    s_lm = calibrate_linear(dummy_tok[:, :d_hidden], lm_head_w)

    return GRUWeights(
        tok_emb=tok_emb,
        pos_emb=pos_emb,
        gru_x=Int8Mat(gru_x_w, s_tok),
        gru_h=Int8Mat(gru_h_w, s_h),
        lm_head=Int8Mat(lm_head_w, s_lm),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/tinystories")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=10_000)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_hidden", type=int, default=256)
    p.add_argument("--pop_size", type=int, default=16)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--fixed_point", type=int, default=2)
    p.add_argument("--sigma_shift", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--init_scale", type=float, default=16.0)
    p.add_argument("--fast_fitness", type=int, default=1, help="1 for sign of paired diff, 0 for scaled diff")
    p.add_argument("--update_threshold", type=int, default=0, help="apply sign update only if |vote|>=threshold")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    train_path = Path(args.data_dir) / "train_tokens.npy"
    if not train_path.exists():
        train_path = Path(args.data_dir) / "train_tokens.uint16.memmap"
    memmap = load_tokens(train_path)

    cfg = NoiseConfig(
        fixed_point=args.fixed_point,
        sigma_shift=args.sigma_shift,
        rank=1,
        fast_fitness=bool(args.fast_fitness),
        update_threshold=args.update_threshold,
    )
    param_span = (3 * args.d_hidden + args.d_model) * cfg.rank * args.pop_size * 2
    ctx = make_context(cfg, param_span=param_span, seed=args.seed, safety_margin=4096)

    weights = init_weights(cfg, args.vocab_size, args.seq_len, args.d_model, args.d_hidden, rng, args.init_scale)

    if args.pop_size % 2 != 0:
        raise ValueError("pop_size must be even for paired fitnesses.")

    thread_ids = list(range(args.pop_size))
    thread_ids_for_update = [i * 2 for i in range(args.pop_size // 2)]

    for step in range(args.steps):
        ctx.epoch = step
        x, y = get_batch(memmap, args.seq_len, batch_size=args.batch_size, rng=rng)

        logits_pop = forward_gru_int8_batched(ctx, weights, x, thread_ids)  # (P,B,vocab)
        rewards = -mx.vmap(lambda l: cross_entropy(l, y))(logits_pop)  # (P,)
        rewards = rewards.reshape(-1)
        fitnesses = convert_fitnesses(cfg, rewards)

        weights.gru_x.weight, _ = apply_sign_update(cfg, weights.gru_x.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)
        weights.gru_h.weight, _ = apply_sign_update(cfg, weights.gru_h.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)
        weights.lm_head.weight, _ = apply_sign_update(cfg, weights.lm_head.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)

        logits_eval = forward_gru_int8(ctx, weights, x, thread_id=None)
        loss = cross_entropy(logits_eval, y)
        lmax = float(np.array(logits_eval).max())
        r_mean = float(mx.mean(rewards).item())
        r_std = float(mx.std(rewards).item())
        print(f"step {step}: loss={loss.item():.4f}, logits_max={lmax:.1f}, reward_mean={r_mean:.3f}, reward_std={r_std:.3f}")


if __name__ == "__main__":
    main()
