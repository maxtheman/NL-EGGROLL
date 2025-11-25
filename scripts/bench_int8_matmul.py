import argparse
import time

import numpy as np
import mlx.core as mx

from eggroll_mlx import int8_matmul


def benchmark_shape(M, N, K, iters=30):
    a = np.random.randint(-3, 4, size=(M, K), dtype=np.int8)
    b = np.random.randint(-3, 4, size=(N, K), dtype=np.int8)
    a_mx = mx.array(a)
    b_mx = mx.array(b)

    # correctness check
    ref = a.astype(np.int32) @ b.T.astype(np.int32)
    out = np.array(int8_matmul(a_mx, b_mx))
    assert np.array_equal(ref, out), "int8_matmul mismatch"

    # warmup
    mx.eval(int8_matmul(a_mx, b_mx))

    start = time.perf_counter()
    for _ in range(iters):
        res = int8_matmul(a_mx, b_mx)
    mx.eval(res)
    elapsed = time.perf_counter() - start
    avg = elapsed / iters
    ops = 2 * M * N * K
    tops = ops / (avg * 1e12)
    return avg, tops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    shapes = [
        (256, 256, 256),
        (512, 512, 256),
        (1024, 1024, 256),
        (1024, 1024, 512),
    ]

    for M, N, K in shapes:
        avg, tops = benchmark_shape(M, N, K, iters=args.iters)
        print(f"M={M} N={N} K={K}: avg={avg*1e3:.3f} ms, TOPS={tops:.3f}")


if __name__ == "__main__":
    main()
