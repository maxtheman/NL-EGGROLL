"""
Generate a simple counting dataset for next-token prediction.
- Tokens cycle 0,1,2,...,(vocab-1),0,1,...
- Saved to a single .npy file expected by train_tinystories_gru_full.py.
"""

import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=16, help="vocabulary size")
    ap.add_argument("--total_tokens", type=int, default=200_000, help="total tokens to generate")
    ap.add_argument("--out_dir", type=str, default="data/count", help="output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.arange(args.total_tokens, dtype=np.uint16) % args.vocab
    out_path = out_dir / "train_tokens.npy"
    np.save(out_path, tokens)
    print(f"Saved {out_path} shape={tokens.shape} vocab={args.vocab}")


if __name__ == "__main__":
    main()
