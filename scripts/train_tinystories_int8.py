"""
Minimal int8-only training loop on TinyStories tokens using eggroll updates.
Model: take last token embedding (int8) -> int8 linear to vocab logits.
Uses ES-style sign updates via apply_sign_update.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import mlx.core as mx

from eggroll_mlx import (
    NoiseConfig,
    apply_sign_update,
    convert_fitnesses,
    do_mm,
    generate_big_rand,
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
    x_last = x[:, -1]  # last input token
    # to int32 for indexing
    x_last = mx.array(x_last.astype(np.int32))
    y = mx.array(y.astype(np.int32))
    return x_last, y


def quantize(arr, fixed_point):
    scaled = np.clip(np.round(arr * (2**fixed_point)), -127, 127).astype(np.int8)
    return mx.array(scaled, dtype=mx.int8)


def cross_entropy(logits, targets):
    vocab = logits.shape[-1]
    one_hot = (targets[..., None] == mx.arange(vocab)).astype(mx.float32)
    max_logits = mx.max(logits, axis=-1, keepdims=True)
    logsumexp = max_logits + mx.log(mx.sum(mx.exp(logits - max_logits), axis=-1, keepdims=True))
    log_probs = logits - logsumexp
    return -mx.mean(mx.sum(one_hot * log_probs, axis=-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/tinystories")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--pop_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--fixed_point", type=int, default=4)
    parser.add_argument("--sigma_shift", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="checkpoints/int8_baseline", help="where to save emb/w_out")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    train_path = Path(args.data_dir) / "train_tokens.npy"
    if not train_path.exists():
        train_path = Path(args.data_dir) / "train_tokens.uint16.memmap"
    memmap = load_tokens(train_path)

    cfg = NoiseConfig(fixed_point=args.fixed_point, sigma_shift=args.sigma_shift, rank=1)
    base_seed = args.seed
    big_rand = generate_big_rand(2_000_000, seed=args.seed, fixed_point=cfg.fixed_point, dtype=mx.int8)

    # Parameters: embeddings (vocab x d_model), output (vocab x d_model)
    emb = quantize(rng.normal(0, 0.05, size=(args.vocab_size, args.d_model)), cfg.fixed_point)
    w_out = quantize(rng.normal(0, 0.05, size=(args.vocab_size, args.d_model)), cfg.fixed_point)

    for step in range(args.steps):
        x_last, y = get_batch(memmap, args.seq_len, batch_size=32, vocab_size=args.vocab_size, rng=rng)
        rewards = []
        for j in range(args.pop_size):
            thread_id = j
            # gather embeddings
            h_int = emb[x_last]  # (batch, d_model) int8
            logits_int = do_mm(cfg, h_int, w_out, big_rand, epoch=step, thread_id=thread_id, base_seed=base_seed)
            logits = logits_int.astype(mx.float32)
            rewards.append(-cross_entropy(logits, y).item())
        rewards = mx.array(rewards)
        fitnesses = convert_fitnesses(cfg, rewards)
        w_out = apply_sign_update(cfg, w_out, fitnesses, big_rand, step, base_seed)

        # eval on the same batch
        h_eval = emb[x_last]
        logits_eval = do_mm(cfg, h_eval, w_out, None, None, None, None).astype(mx.float32)
        loss = cross_entropy(logits_eval, y)
        print(f"step {step}: loss={loss.item():.4f}, logits_max={float(np.array(logits_eval).max()):.1f}")

    # Save learned parameters
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "emb.npy", np.array(emb))
    np.save(save_dir / "w_out.npy", np.array(w_out))
    print("Saved checkpoint to", save_dir)


if __name__ == "__main__":
    main()
