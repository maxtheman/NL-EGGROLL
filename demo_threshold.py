import mlx.core as mx
import numpy as np
from eggroll_mlx import NoiseConfig, apply_sign_update, generate_big_rand

def run_demo():
    print(f"{'Pop Size':<10} | {'Threshold':<10} | {'Eff. Thresh':<12} | {'Max Signal':<10} | {'Update Rate':<12}")
    print("-" * 65)
    
    fixed_point = 4
    cfg = NoiseConfig(fixed_point=fixed_point, use_clt=False, rank=1)
    
    # Create a dummy parameter
    rows, cols = 128, 128
    param = mx.zeros((rows, cols), dtype=mx.int8)
    base_seed = 0
    
    # Generate big rand once
    big_rand = generate_big_rand(1024*1024, seed=0, fixed_point=fixed_point)
    
    # Intermediate values between 1024 and 32k
    pop_sizes = [1024, 2048, 4096, 8192, 16384, 32768]
    
    for pop_size in pop_sizes:
        pop_pairs = pop_size // 2
        
        # Simulate fitnesses with small correlation to noise
        # In real training, fitness is correlated with the "good" direction.
        # Let's say there is a "true gradient" direction G.
        # Fitness ~ <Noise, G>.
        # Let's simulate this by making fitness slightly correlated with the first noise component.
        
        # We need to know what the noise "is" to correlate with it.
        # But here we just want to show that IF there is signal, it passes.
        # Let's just force a signal Z of magnitude X * sigma.
        # Z ~ N(correlation * pop_pairs, pop_pairs) roughly.
        
        # Instead of simulating complex correlation, let's just manually set Z distribution
        # or easier: use the "all ones" fitness again to show the THEORETICAL MAX capacity.
        # The user asked "what's the cutoff".
        # The cutoff is when Max Signal > Eff Thresh.
        # Max Signal = Pop Pairs.
        # Eff Thresh = 32 * sqrt(Pop Pairs).
        # Cutoff: P/2 > 32 * sqrt(P/2)  => sqrt(P/2) > 32 => P/2 > 1024 => P > 2048.
        
        # Let's revert to "all ones" (Max Signal) to show the *capacity* for updates.
        fitnesses = mx.ones((pop_pairs,), dtype=mx.int8)
        
        for thresh in [512]: # Focus on default threshold
            cfg.update_threshold = thresh
            
            # Calculate effective threshold
            # abs(Z) < thresh * sqrt(pop) / 2^FP
            scale_factor = np.sqrt(pop_pairs) / (2 ** fixed_point)
            eff_thresh = thresh * scale_factor
            
            # Run actual update
            thread_ids = list(range(pop_pairs))
            
            # To show capacity, we need fitness to align with noise direction.
            # We need to peek at the noise first.
            # This is a bit circular but valid for demonstrating "what if the gradient was perfect".
            
            # We can't easily peek inside apply_sign_update without refactoring.
            # But we can just use a trick:
            # Z = sum(fitness * sign(A*B))
            # If we want Z to be max, we need fitness = sign(A*B).
            # But A*B varies per parameter!
            # Fitness is a scalar per individual.
            # We can't make fitness align with EVERY parameter simultaneously.
            # That's impossible.
            # "Max Signal" for a *single* parameter is pop_pairs.
            # But for the whole model?
            # If we have 1 parameter, we can align it.
            # The demo uses a (128, 128) matrix.
            # We can't align fitness to all 128*128 params.
            # However, the "Update Rate" is the % of params that update.
            # With random fitness, Z is random walk.
            # With "perfect" fitness for *one* param, that param updates.
            # But we want to see *any* updates.
            
            # Actually, the "Max Signal" column in my table assumes perfect alignment for *that* param.
            # The question is: for a given param, can Z exceed Threshold?
            # Z ~ N(0, pop_pairs).
            # We want P(|Z| > Thresh).
            # If Thresh > 3 * sqrt(pop_pairs), prob is ~0.
            # Eff Thresh = 32 * sqrt(pop_pairs).
            # So Thresh is 32 sigma!
            # Probability is basically zero.
            
            # CONCLUSION: With `threshold=512` and `use_clt=False`, updates are IMPOSSIBLE for random noise, regardless of population size (up to huge numbers).
            # 32 sigma is huge.
            # Even for 262k pop, Eff Thresh is 11k. Sigma is sqrt(131k) ~ 362.
            # 11k / 362 = 30 sigma.
            # So... `threshold=512` is WAY too high for `use_clt=False`?
            # Let's check the reference again.
            # Reference uses `update_threshold=512`?
            # In `run.py`: `update_threshold` default is 128?
            # Let's check `run.py`.
            pass
            
            updated, num_changed = apply_sign_update(cfg, param, fitnesses, big_rand, 0, base_seed, thread_ids)
            rate = (num_changed / param.size) * 100
            rate_str = f"{rate:.4f}%"
                
            print(f"{pop_size:<10} | {thresh:<10} | {eff_thresh:<12.1f} | {pop_pairs:<10} | {rate_str:<12}")

if __name__ == "__main__":
    run_demo()
