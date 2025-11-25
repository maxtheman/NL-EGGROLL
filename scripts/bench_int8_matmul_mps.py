"""
Reference int8 GEMM benchmark using MPSGraph.matrixMultiplication via PyObjC.
Requires pyobjc-framework-MetalPerformanceShadersGraph and a supported macOS GPU.
"""

import argparse
import time
from math import prod

import numpy as np

try:
    import objc  # type: ignore
    from MetalPerformanceShadersGraph import MPSGraph, MPSGraphTensor, MPSGraphTensorData  # type: ignore
    from Metal import MTLCreateSystemDefaultDevice  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "PyObjC MetalPerformanceShadersGraph not available. Install with "
        "`pip install pyobjc-framework-MetalPerformanceShadersGraph pyobjc-framework-Metal`"
    ) from e


def mps_int8_matmul(a: np.ndarray, b: np.ndarray):
    assert a.dtype == np.int8 and b.dtype == np.int8
    device = MTLCreateSystemDefaultDevice()
    graph = MPSGraph.alloc().init()
    pa = graph.placeholderWithShape_dtype_name_(a.shape, 0x13, "a")  # 0x13 = int8
    pb = graph.placeholderWithShape_dtype_name_(b.shape, 0x13, "b")
    # MPSGraph matmul expects primary [M,K], secondary [K,N]
    out = graph.matrixMultiplicationPrimary_secondary_name_(pa, pb, "mm")
    feeds = {pa: MPSGraphTensorData.tensorDataWithDevice_array_(device, a), pb: MPSGraphTensorData.tensorDataWithDevice_array_(device, b)}
    executable = graph.compileWithDevice_(device)
    return graph, executable, feeds, out


def benchmark_shape(M, N, K, iters=10):
    a = np.random.randint(-3, 4, size=(M, K), dtype=np.int8)
    b = np.random.randint(-3, 4, size=(K, N), dtype=np.int8)

    graph, exe, feeds, out = mps_int8_matmul(a, b)
    # warmup
    exe.runWithMTLCommandQueue_feeds_targets_returnDataTypes_(None, feeds, [out], None)

    start = time.perf_counter()
    for _ in range(iters):
        res = exe.runWithMTLCommandQueue_feeds_targets_returnDataTypes_(None, feeds, [out], None)
    elapsed = time.perf_counter() - start
    avg = elapsed / iters
    ops = 2 * M * N * K
    tops = ops / (avg * 1e12)
    return avg, tops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    shapes = [
        (256, 256, 256),
        (512, 512, 256),
    ]
    for M, N, K in shapes:
        avg, tops = benchmark_shape(M, N, K, args.iters)
        print(f"M={M} N={N} K={K}: avg={avg*1e3:.3f} ms, TOPS={tops:.3f}")


if __name__ == "__main__":
    main()
