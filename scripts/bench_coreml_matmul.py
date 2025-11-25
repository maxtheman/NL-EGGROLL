import argparse
import time
from pathlib import Path

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np


def build_coreml_matmul(M, N, K, out_dir):
    target = ct.target.iOS18

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(M, K), dtype=types.int8),
            mb.TensorSpec(shape=(K, N), dtype=types.int8),
        ],
        opset_version=target,
    )
    def prog(x, y):
        zp = np.int8(0)
        x_f = mb.dequantize(input=x, scale=1.0, zero_point=zp)
        y_f = mb.dequantize(input=y, scale=1.0, zero_point=zp)
        return mb.matmul(x=x_f, y=y_f, name="out")

    mlmodel = ct.convert(
        prog,
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"matmul_{M}_{N}_{K}.mlpackage"
    mlmodel.save(path)
    return path


def benchmark_coreml(model_path, M, N, K, iters):
    mlmodel = ct.models.MLModel(str(model_path))
    x = np.random.randint(-3, 4, size=(M, K), dtype=np.int8).astype(np.float16)
    y = np.random.randint(-3, 4, size=(K, N), dtype=np.int8).astype(np.float16)
    # warmup
    mlmodel.predict({"x": x, "y": y})
    start = time.perf_counter()
    for _ in range(iters):
        res = mlmodel.predict({"x": x, "y": y})
    elapsed = time.perf_counter() - start
    avg = elapsed / iters
    ops = 2 * M * N * K
    tops = ops / (avg * 1e12)
    return avg, tops, res["out"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--shapes", type=str, default="256,256,256")
    parser.add_argument("--out_dir", type=str, default="coreml_matmul_models")
    args = parser.parse_args()

    triples = []
    for part in args.shapes.split(";"):
        m, n, k = part.split(",")
        triples.append((int(m), int(n), int(k)))

    out_dir = Path(args.out_dir)
    for M, N, K in triples:
        path = out_dir / f"matmul_{M}_{N}_{K}.mlpackage"
        if not path.exists():
            path = build_coreml_matmul(M, N, K, out_dir)
        avg, tops, out = benchmark_coreml(path, M, N, K, args.iters)
        ref = np.matmul(np.zeros((M, K), dtype=np.float16), np.zeros((K, N), dtype=np.float16))
        print(f"M={M} N={N} K={K}: avg={avg*1e3:.3f} ms, TOPS={tops:.3f}, output shape={out.shape}")


if __name__ == "__main__":
    main()
