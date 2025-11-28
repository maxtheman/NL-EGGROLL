
import mlx.core as mx
import numpy as np
import os

# Force enable the tiled kernel for testing
os.environ["INT8_KERNEL_TILE"] = "1"

from eggroll_mlx import int8_matmul, _int8_mm_kernel_vec, _int8_mm_kernel_tiled

def test_kernel_correctness():
    # Define shapes that are likely to trigger boundary conditions
    # M, N not multiples of 16
    M = 33
    N = 33
    K = 64
    
    key = mx.random.key(0)
    a = mx.random.randint(-127, 127, (M, K), key=key).astype(mx.int8)
    b = mx.random.randint(-127, 127, (N, K), key=key).astype(mx.int8)
    
    # Golden reference (using the vectorized kernel directly, or numpy)
    # We can use the vectorized kernel as reference since user said it works
    # Or just simple numpy
    a_np = np.array(a).astype(np.int32)
    b_np = np.array(b).astype(np.int32)
    expected_np = a_np @ b_np.T
    
    # Run tiled kernel via int8_matmul (env var set)
    # Note: int8_matmul computes a @ b.T
    out_tiled = int8_matmul(a, b)
    out_tiled_np = np.array(out_tiled)
    
    # Run vectorized kernel explicitly for comparison
    # We need to manually call it    print(f"a[0,0] = {a_np[0,0]}")
    print(f"a[0,1] = {a_np[0,1]}")
    print(f"a[1,0] = {a_np[1,0]}")
    
    print(f"a == b: {np.array_equal(a_np, b_np)}")
    print(f"Norm^2 of a[0]: {np.linalg.norm(a_np[0].astype(float))**2}")
    print(f"Expected[0,0]: {expected_np[0,0]}")
    
    diff = np.abs(out_tiled_np - expected_np)
    max_diff = np.max(diff)
    
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"Max diff: {max_diff}")
    
    if max_diff > 0:
        print("FAIL: Mismatch found!")
        # Print a small patch to see pattern
        print("Expected slice:")
        print(expected_np[:4, :4])
        print("Actual slice:")
        print(out_tiled_np[:4, :4])
    else:
        print("PASS: Exact match.")

if __name__ == "__main__":
    test_kernel_correctness()
