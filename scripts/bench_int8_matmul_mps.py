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
    import Metal as Metal  # type: ignore
    import MetalPerformanceShaders as MPS  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "PyObjC MetalPerformanceShadersGraph not available. Install with "
        "`pip install pyobjc-framework-MetalPerformanceShadersGraph pyobjc-framework-Metal`"
    ) from e


def mps_int8_matmul(a: np.ndarray, b: np.ndarray):
    assert a.dtype == np.int8 and b.dtype == np.int8
    shape_a = tuple(int(x) for x in a.shape)
    shape_b = tuple(int(x) for x in b.shape)
    device = MTLCreateSystemDefaultDevice()
    print(f"  Device: {device}")
    if device is None:
        raise RuntimeError("No Metal device available")
    print(f"  Device name: {device.name()}")
    graph = MPSGraph.alloc().init()
    print(f"  Graph allocated")
    print(f"  Creating placeholders for shapes: {shape_a} x {shape_b}")
    int8_type = MPS.MPSDataTypeInt8  # correct enum, not a magic constant
    pa = graph.placeholderWithShape_dataType_name_(shape_a, int8_type, "a")
    pb = graph.placeholderWithShape_dataType_name_(shape_b, int8_type, "b")
    print(f"  Placeholders created")
    # MPSGraph matmul expects primary [M,K], secondary [K,N]
    out = graph.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name_(pa, pb, "mm")
    print(f"  Graph created with shapes: {a.shape} x {b.shape} -> {out.shape()}")
    # build tensor data via MTLBuffer
    def np_to_mps_data(np_arr):
        buf = device.newBufferWithBytes_length_options_(np_arr.tobytes(), np_arr.nbytes, Metal.MTLResourceStorageModeShared)
        shape = [int(x) for x in np_arr.shape]
        return MPSGraphTensorData.alloc().initWithMTLBuffer_shape_dataType_(buf, shape, int8_type)

    feeds = {pa: np_to_mps_data(a), pb: np_to_mps_data(b)}
    print(f"  Feeds created")
    return graph, feeds, out


def benchmark_shape(M, N, K, iters=10):
    print(f"  Creating arrays...")
    a = np.random.randint(-3, 4, size=(M, K), dtype=np.int8)
    b = np.random.randint(-3, 4, size=(K, N), dtype=np.int8)

    print(f"  Creating MPS graph...")
    graph, feeds, out = mps_int8_matmul(a, b)
    # warmup
    print("  Running warmup...")
    warm = graph.runWithMTLCommandQueue_feeds_targetTensors_targetOperations_(None, feeds, [out], None)

    print("  Running benchmark...")
    start = time.perf_counter()
    for _ in range(iters):
        res = graph.runWithMTLCommandQueue_feeds_targetTensors_targetOperations_(None, feeds, [out], None)
    elapsed = time.perf_counter() - start
    avg = elapsed / iters
    # fetch result for correctness
    out_np = res[out].mpsndarray().to_numpy()
    ref = a.astype(np.int32) @ b.astype(np.int32)
    ok = np.array_equal(out_np.astype(np.int32), ref)
    if not ok:
        raise ValueError("MPSGraph int8 matmul mismatch")
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
        print(f"Testing shape: M={M}, N={N}, K={K}")
        try:
            print("  Running benchmark...")
            avg, tops = benchmark_shape(M, N, K, args.iters)
            print(f"M={M} N={N} K={K}: avg={avg*1e3:.3f} ms, TOPS={tops:.3f}")
        except Exception as e:
            print(f"M={M} N={N} K={K}: ERROR - {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
