"""
Greedy inference for the int8 TinyStories baseline.
- Loads tokenizer, int8 embeddings, and output matrix from a checkpoint dir.
- Generates tokens by repeatedly applying int8 do_mm and softmax on float logits.
"""

import argparse
from pathlib import Path

import numpy as np
import mlx.core as mx
from tokenizers import Tokenizer

from eggroll_mlx import NoiseConfig, do_mm


def load_checkpoint(ckpt_dir: Path):
    emb = np.load(ckpt_dir / "emb.npy")
    w_out = np.load(ckpt_dir / "w_out.npy")
    return mx.array(emb, dtype=mx.int8), mx.array(w_out, dtype=mx.int8)


def softmax(logits):
    m = mx.max(logits, axis=-1, keepdims=True)
    exps = mx.exp(logits - m)
    return exps / mx.sum(exps, axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/int8_baseline")
    parser.add_argument("--tokenizer", type=str, default="data/tinystories/tokenizer.json")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--fixed_point", type=int, default=4)
    parser.add_argument("--sigma_shift", type=int, default=4)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    emb, w_out = load_checkpoint(Path(args.ckpt_dir))
    cfg = NoiseConfig(fixed_point=args.fixed_point, sigma_shift=args.sigma_shift, rank=1)

    ids = tokenizer.encode(args.prompt).ids
    for _ in range(args.max_new_tokens):
        last_id = ids[-1]
        x = mx.array([[last_id]], dtype=mx.int32)  # shape (1,1)
        h_int = emb[x]  # (1,1,d_model)
        h_int = mx.reshape(h_int, (1, -1))  # (batch, d_model)
        logits_int = do_mm(cfg, h_int, w_out, None, None, None, None)
        probs = softmax(logits_int.astype(mx.float32))
        next_id = int(mx.argmax(probs, axis=-1).item())
        ids.append(next_id)
    text = tokenizer.decode(ids)
    print(text)


if __name__ == "__main__":
    main()
