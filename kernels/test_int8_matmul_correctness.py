import argparse
import numpy as np
import mlx.core as mx
import os

def run(shapes, use_tiled: bool):
    os.environ["INT8_KERNEL_TILE"] = "1" if use_tiled else "0"
    from eggroll_mlx import int8_matmul  # import after env var set
    all_ok = True
    for M, N, K in shapes:
        a = np.random.randint(-128, 127, size=(M, K), dtype=np.int8)
        b = np.random.randint(-128, 127, size=(N, K), dtype=np.int8)
        ref = a.astype(np.int32) @ b.T.astype(np.int32)
        res = int8_matmul(mx.array(a), mx.array(b))
        mx.eval(res)
        res_np = np.array(res)
        ok = np.array_equal(ref, res_np)
        max_diff = np.max(np.abs(ref - res_np))
        print(f"use_tiled={use_tiled} shape {M}x{N}x{K}: ok={ok} max_diff={max_diff}")
        all_ok = all_ok and ok
    return all_ok

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiled", action="store_true", help="use tiled kernel")
    args = p.parse_args()
    shapes = [(32, 32, 32), (64, 64, 64), (128, 64, 96), (128, 128, 128)]
    ok = run(shapes, use_tiled=args.tiled)
    if not ok:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
