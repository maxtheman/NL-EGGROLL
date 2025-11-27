import mlx.core as mx
import time
import eggroll_mlx

# Enable timing
eggroll_mlx.TIMING_ENABLED = True
from eggroll_mlx import int8_matmul

def profile():
    # Shapes from training loop: (128*16, 256) x (256, 256)
    M = 2048
    K = 256
    N = 256
    
    a = mx.random.randint(-127, 127, (M, K)).astype(mx.int8)
    b = mx.random.randint(-127, 127, (N, K)).astype(mx.int8)
    
    print("\n=== TEST 1: Standard Path ===")
    eggroll_mlx.MAX_GRID_OVERRIDE = None # Default
    # Warmup
    print("Warmup...")
    for _ in range(3):
        out = int8_matmul(a, b)
        mx.eval(out)
        
    print("Running profile...")
    for i in range(3):
        print(f"Iter {i}:")
        out = int8_matmul(a, b)
        mx.eval(out)

    print("\n=== TEST 2: Chunked Path (Forced) ===")
    # Force chunking by setting MAX_GRID to something small
    # M*N = 2048*256 = 524288
    # Let's set MAX_GRID = 100000, so we get ~6 chunks
    eggroll_mlx.MAX_GRID_OVERRIDE = 100000
    
    print("Warmup...")
    for _ in range(3):
        out = int8_matmul(a, b)
        mx.eval(out)
        
    print("Running profile...")
    for i in range(3):
        print(f"Iter {i}:")
        out = int8_matmul(a, b)
        mx.eval(out)

if __name__ == "__main__":
    profile()
