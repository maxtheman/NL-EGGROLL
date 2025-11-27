#include <metal_stdlib>
using namespace metal;

// Naive half GEMM (no NAX). Replace with NAX-tiled matmul if desired.
kernel void nax_gemm(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C      [[buffer(2)]],
    constant uint& M     [[buffer(3)]],
    constant uint& N     [[buffer(4)]],
    constant uint& K     [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {

    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    uint a_off = row * K;
    uint b_off = col * K;
    for (uint k = 0; k < K; ++k) {
        acc += float(A[a_off + k]) * float(B[b_off + k]);
    }
    C[row * N + col] = acc;
}
