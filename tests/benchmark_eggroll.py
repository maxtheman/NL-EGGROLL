
import time
import mlx.core as mx
import numpy as np
from eggroll_mlx import NoiseConfig, _stack_lora_params, generate_big_rand

def benchmark_stack_lora(pop_size, rank=1, rows=256, cols=256):
    print(f"Benchmarking pop_size={pop_size}...")
    cfg = NoiseConfig(rank=rank)
    # Generate a dummy big_rand
    big_rand_size = (rows + cols) * rank * pop_size * 2 + 10000
    big_rand = generate_big_rand(big_rand_size)
    
    thread_ids = list(range(pop_size))
    base_seed = 42
    epoch = 0
    
    # Warmup
    mx.eval(big_rand)
    
    start_time = time.time()
    # We want to measure the graph construction + execution time
    A_stack, B_stack = _stack_lora_params(cfg, big_rand, epoch, thread_ids, (rows, cols), base_seed)
    mx.eval(A_stack, B_stack)
    end_time = time.time()
    
    print(f"Time: {end_time - start_time:.4f}s")
    return end_time - start_time

if __name__ == "__main__":
    # Test small first
    benchmark_stack_lora(10)
    # Test medium
    benchmark_stack_lora(100)
    # Test large (might fail or be very slow before fix)
    try:
        benchmark_stack_lora(1000)
    except Exception as e:
        print(f"Failed at 1000: {e}")
