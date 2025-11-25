
import time
import numpy as np
import mlx.core as mx
from pathlib import Path
from eggroll_mlx import NoiseConfig
from eggroll_api import make_context
from scripts.train_tinystories_gru_full import (
    init_model,
    load_tokens,
    get_batch,
    forward_model,
    convert_fitnesses,
    apply_sign_update,
    cross_entropy
)

def run_benchmark():
    # Configuration
    data_dir = "data/tinystories"
    seq_len = 16
    vocab_size = 1024
    d_model = 256
    d_hidden = 256
    layers = 1
    batch_size = 1
    steps = 2  # Warmup + measure
    
    # Population sizes to test
    pop_sizes = [10240]
    
    print(f"{'Pop Size':<10} | {'Time/Step (s)':<15} | {'Tokens/Sec':<15} | {'Speedup':<10}")
    print("-" * 60)
    
    base_throughput = None

    # Load data once
    rng = np.random.default_rng(42)
    train_path = Path(data_dir) / "train_tokens.npy"
    if not train_path.exists():
        train_path = Path(data_dir) / "train_tokens.uint16.memmap"
    memmap = load_tokens(train_path)

    for pop_size in pop_sizes:
        # Setup
        cfg = NoiseConfig(
            fixed_point=4,
            sigma_shift=4,
            rank=1,
            fast_fitness=True,
            fitness_alpha=0.01,
            update_threshold=512,
            noise_reuse=1
        )
        
        # Calculate param span for context
        # Cap param_span to avoid OOM during generation (2**28 = 268M elements = 268MB int8)
        # Throughput is unaffected by wrapping noise
        calculated_span = (d_model + 3 * d_hidden + 4 * d_hidden + vocab_size) * cfg.rank * pop_size * 2
        param_span = min(calculated_span, 2**28)
        ctx = make_context(cfg, param_span=param_span, seed=42, safety_margin=4096)
        
        weights = init_model(cfg, vocab_size, seq_len, d_model, d_hidden, layers, rng, init_scale=16.0)
        
        thread_ids = list(range(pop_size))
        thread_ids_for_update = [i * 2 for i in range(pop_size // 2)]
        
        # JIT compile the step function for fair benchmarking
        # We'll just run the loop in python as the ops are heavy enough
        
        times = []
        
        # Run steps
        for step in range(steps):
            # mx.cuda.synchronize() if hasattr(mx, 'cuda') else None
            t0 = time.time()
            
            # Forward
            # We use group_size = pop_size (one batch shared across all)
            # This matches the "pure batch inference" claim logic where we want max throughput
            # But typically we might split. For this benchmark let's assume 1 group for max memory pressure test
            # actually, let's stick to the script default which defaults group_size=pop_size
            
            x, y = get_batch(memmap, seq_len, batch_size=batch_size, rng=rng)
            logits, states_out = forward_model(ctx, weights, x, thread_ids, None)
            
            # Force eval
            mx.eval(logits)
            
            # Backward / Update (simplified for benchmark, just doing the heavy lifting)
            rewards = -mx.vmap(lambda l: cross_entropy(l, y))(logits).reshape(-1)
            fitnesses = convert_fitnesses(cfg, rewards)
            
            # Just update one matrix to test update overhead
            weights.head.weight, _ = apply_sign_update(cfg, weights.head.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)
            mx.eval(weights.head.weight)
            
            # mx.cuda.synchronize() if hasattr(mx, 'cuda') else None
            t1 = time.time()
            
            if step > 0: # Skip warmup
                times.append(t1 - t0)
                
        avg_time = np.mean(times)
        tokens_per_step = batch_size * seq_len * pop_size
        throughput = tokens_per_step / avg_time
        
        if base_throughput is None:
            base_throughput = throughput
            speedup = 1.0
        else:
            speedup = throughput / base_throughput
            
        print(f"{pop_size:<10} | {avg_time:<15.4f} | {throughput:<15.2e} | {speedup:<10.2f}x")

if __name__ == "__main__":
    run_benchmark()
