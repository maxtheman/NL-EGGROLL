import argparse
import time

import mlx.core as mx


def benchmark_shape(M, N, K, iters, bits, group_size, dtype):
    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")

    # Activations in float16 (or chosen dtype); weights are quantized once.
    x = mx.random.normal((M, K)).astype(dtype)
    w_fp = mx.random.normal((N, K)).astype(mx.float32)
    q_w, scales, *biases = mx.quantize(
        w_fp, group_size=group_size, bits=bits, mode="affine"
    )
    bias = biases[0] if biases else None

    # Warmup
    mx.eval(
        mx.quantized_matmul(
            x,
            q_w,
            scales=scales,
            biases=bias,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
    )

    start = time.perf_counter()
    for _ in range(iters):
        out = mx.quantized_matmul(
            x,
            q_w,
            scales=scales,
            biases=bias,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
    mx.eval(out)
    elapsed = time.perf_counter() - start
    avg = elapsed / iters
    ops = 2 * M * N * K
    tops = ops / (avg * 1e12)
    return avg, tops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--bits", type=int, default=8, help="Quantized weight bits (2–8)")
    p.add_argument("--group-size", type=int, default=64, help="Per-group size for scales/bias")
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Activation dtype (weights are quantized regardless)",
    )
    p.add_argument(
        "--shapes",
        type=str,
        default="512,512,512;1024,1024,1024;2048,1024,512",
        help="Semicolon-separated M,N,K triples",
    )
    args = p.parse_args()

    dtype = getattr(mx, args.dtype)
    triples = []
    for part in args.shapes.split(";"):
        m, n, k = part.split(",")
        triples.append((int(m), int(n), int(k)))

    print(
        f"quantized_matmul benchmark | dtype={args.dtype} bits={args.bits} group_size={args.group_size} iters={args.iters}"
    )
    for M, N, K in triples:
        avg, tops = benchmark_shape(
            M, N, K, args.iters, args.bits, args.group_size, dtype
        )
        print(f"M={M} N={N} K={K}: avg={avg*1e3:.3f} ms, TOPS={tops:.3f}")


if __name__ == "__main__":
    main()
