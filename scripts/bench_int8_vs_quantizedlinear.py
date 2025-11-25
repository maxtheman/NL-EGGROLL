import argparse
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn


def bench_quantized_linear(M, N, K, iters):
    layer = nn.QuantizedLinear(input_dims=K, output_dims=N, bits=8)
    x = mx.random.normal((M, K))
    # warmup
    mx.eval(layer(x))
    start = time.perf_counter()
    for _ in range(iters):
        out = layer(x)
    mx.eval(out)
    elapsed = time.perf_counter() - start
    return elapsed / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    shapes = [
        (256, 256, 256),
        (512, 512, 256),
        (1024, 1024, 256),
        (4096, 4096, 512),
    ]
    for M, N, K in shapes:
        t_q = bench_quantized_linear(M, N, K, args.iters)
        ops = 2 * M * N * K
        tops_q = ops / (t_q * 1e12)
        print(f"Shape M={M} N={N} K={K}:")
        print(f"  QuantizedLinear avg={t_q*1e3:.3f} ms, TOPS={tops_q:.3f}")


if __name__ == "__main__":
    main()
