"""
FP16 TinyStories GRU trainer (simplified):
- Uses float16 weights/activations everywhere.
- No custom int8 kernel; relies on standard MLX matmul (NAX is used automatically when available).
- Keeps the data pipeline from the int8 script for convenience.
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from scripts.train_tinystories_gru_int8 import load_tokens, get_batch, cross_entropy


class TinyGRU(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_hidden: int, seq_len: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.gru = nn.GRUCell(input_size=d_model, hidden_size=d_hidden)
        self.out = nn.Linear(d_hidden, vocab_size)

    def __call__(self, x):
        # x: (B, T)
        B, T = x.shape
        tok = self.tok_emb(x)
        pos = self.pos_emb(mx.arange(T))[None, ...]
        h = mx.zeros((B, self.gru.hidden_size), dtype=mx.float16)
        for t in range(T):
            h = self.gru(tok[:, t, :] + pos[:, t, :], h)
        logits = self.out(h)
        return logits


def train(args):
    # Load data
    rng = np.random.default_rng(args.seed)
    train_path = Path(args.data_dir) / "train_tokens.npy"
    if not train_path.exists():
        train_path = Path(args.data_dir) / "train_tokens.uint16.memmap"
    tokens = load_tokens(train_path)

    model = TinyGRU(args.vocab, args.d_model, args.d_hidden, args.seq_len)
    # Cast params to fp16
    params = model.trainable_parameters()
    for p in params.values():
        p[:] = p.astype(mx.float16)

    opt = nn.optim.SGD(learning_rate=args.lr, momentum=0.9)

    def loss_fn(params, batch_x, batch_y):
        logits = model.apply(params, batch_x)
        return cross_entropy(logits, batch_y)

    @mx.jit
    def train_step(params, opt_state, batch_x, batch_y):
        loss, grads = mx.value_and_grad(loss_fn)(params, batch_x, batch_y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = nn.tree_apply(mx.add, params, updates)
        return params, opt_state, loss

    opt_state = opt.init(params)

    for step in range(args.steps):
        batch_x, batch_y = get_batch(tokens, args.seq_len, batch_size=args.batch_size, rng=rng)
        params, opt_state, loss = train_step(params, opt_state, batch_x, batch_y)
        mx.eval(params)
        if step % args.log_every == 0:
            print(f"step {step:04d} loss {loss.item():.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/tinystories")
    p.add_argument("--vocab", type=int, default=1024)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_hidden", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
