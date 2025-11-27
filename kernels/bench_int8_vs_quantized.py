import argparse
import time
from typing import List, Tuple

import mlx.core as mx
import numpy as np

from eggroll_mlx import int8_matmul


def parse_shapes(shapes_str: str) -> List[Tuple[int, int, int]]:
    shapes = []
    for part in shapes_str.split(";"):
        m, n, k = part.split(",")
        shapes.append((int(m), int(n), int(k)))
    return shapes


def bench_int8(M: int, N: int, K: int, iters: int, sync_each: bool) -> Tuple[float, float]:
    x = mx.array(np.random.randint(-128, 127, size=(M, K), dtype=np.int8))
    w = mx.array(np.random.randint(-128, 127, size=(N, K), dtype=np.int8))
    # warmup
    mx.eval(int8_matmul(x, w))
    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = int8_matmul(x, w)
        if sync_each:
            mx.eval(out)
    if not sync_each:
        mx.eval(out)
    elapsed = time.perf_counter() - t0
    return elapsed, elapsed / iters


def bench_quantized(
    M: int, N: int, K: int, iters: int, sync_each: bool, bits: int = 8, group_size: int = 64
) -> Tuple[float, float]:
    # quantized_matmul expects float activations; weights are quantized from float.
    x = mx.array(np.random.randn(M, K), dtype=mx.float16)
    w_float = mx.array(np.random.randn(N, K), dtype=mx.float32)
    q_w, scales, *biases = mx.quantize(w_float, group_size=group_size, bits=bits, mode="affine")
    bias = biases[0] if biases else None
    mx.eval(q_w, scales, x)
    # warmup
    mx.eval(
        mx.quantized_matmul(
            x, q_w, scales=scales, biases=bias, transpose=True, group_size=group_size, bits=bits, mode="affine"
        )
    )
    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = mx.quantized_matmul(
            x, q_w, scales=scales, biases=bias, transpose=True, group_size=group_size, bits=bits, mode="affine"
        )
        if sync_each:
            mx.eval(out)
    if not sync_each:
        mx.eval(out)
    elapsed = time.perf_counter() - t0
    return elapsed, elapsed / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--shapes", type=str, default="2048,2048,512;4096,4096,512")
    p.add_argument("--sync-each", action="store_true", help="sync every iteration (latency mode)")
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--group-size", type=int, default=64)
    args = p.parse_args()

    shapes = parse_shapes(args.shapes)
    print(f"Running head-to-head int8_matmul vs quantized_matmul | iters={args.iters} sync_each={args.sync_each}")
    for M, N, K in shapes:
        ops = 2 * M * N * K
        t_int8, avg_int8 = bench_int8(M, N, K, args.iters, args.sync_each)
        t_q, avg_q = bench_quantized(M, N, K, args.iters, args.sync_each, bits=args.bits, group_size=args.group_size)
        tops_int8 = ops / (avg_int8 * 1e12)
        tops_q = ops / (avg_q * 1e12)
        print(f"Shape M={M} N={N} K={K}:")
        print(f"  int8_matmul       : total={t_int8:.3f}s avg={avg_int8*1e3:.3f} ms TOPS={tops_int8:.3f}")
        print(f"  quantized_matmul  : total={t_q:.3f}s avg={avg_q*1e3:.3f} ms TOPS={tops_q:.3f}")


if __name__ == "__main__":
    main()
