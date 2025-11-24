"""
Prepare TinyStories dataset:
- download a subset of TinyStories shards
- train a BPE tokenizer (vocab ~10k)
- encode to token IDs with fixed seq_len and write train/val memmaps

Runs entirely locally; no training here.
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def train_tokenizer(texts, vocab_size=10_000, min_freq=2):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def iter_text(ds, max_samples=None):
    count = 0
    for item in ds:
        yield item["text"]
        count += 1
        if max_samples and count >= max_samples:
            break


def encode_split(tokenizer, texts, seq_len):
    encoded = []
    for t in texts:
        ids = tokenizer.encode(t).ids
        ids = ids[: seq_len - 1]  # leave room for EOS
        ids = [tokenizer.token_to_id("[BOS]")] + ids
        ids = ids + [tokenizer.token_to_id("[PAD]")] * (seq_len - len(ids))
        encoded.append(ids)
    return np.array(encoded, dtype=np.uint16)


def write_memmap(arr, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(path, mode="w+", dtype=arr.dtype, shape=arr.shape)
    mm[:] = arr[:]
    mm.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/tinystories")
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--train_samples", type=int, default=50_000, help="subset of train to use")
    parser.add_argument("--val_samples", type=int, default=5_000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load TinyStories (streaming to avoid full download)
    train_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    # Train tokenizer on a subset of text
    tokenizer = train_tokenizer(iter_text(train_ds, max_samples=args.train_samples), args.vocab_size, args.min_freq)
    tok_path = out_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))
    print("Saved tokenizer to", tok_path)

    # Reset iterator for encoding
    train_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    val_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True).skip(args.train_samples)

    train_arr = encode_split(tokenizer, iter_text(train_ds, args.train_samples), args.seq_len)
    val_arr = encode_split(tokenizer, iter_text(val_ds, args.val_samples), args.seq_len)

    write_memmap(train_arr, out_dir / "train_tokens.uint16.memmap")
    write_memmap(val_arr, out_dir / "val_tokens.uint16.memmap")
    np.save(out_dir / "train_tokens.npy", train_arr)
    np.save(out_dir / "val_tokens.npy", val_arr)
    print("Saved train/val token arrays to", out_dir)


if __name__ == "__main__":
    main()
