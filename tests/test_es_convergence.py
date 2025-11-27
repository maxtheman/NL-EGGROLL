
import mlx.core as mx
import numpy as np
import sys
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

# Add parent dir to sys.path to import eggroll modules
sys.path.append(str(Path(__file__).parent.parent))

from eggroll_api import EggrollContext, make_context
from eggroll_mlx import NoiseConfig, apply_sign_update, convert_fitnesses, get_lora_update_params

# --- Copied from scripts/train_tinystories_gru_full.py ---

@dataclass
class Int8Mat:
    weight: mx.array            # float16 weights for compute
    weight_unpacked: mx.array   # float16 weights for updates/noise (alias of weight)
    scale: Optional[mx.array] = None
    bias: Optional[mx.array] = None

def matmul_with_noise(ctx: EggrollContext, mat: Int8Mat, x: mx.array, tid: Optional[int], seed_offset: int = 0, noise: Optional[Tuple[mx.array, mx.array]] = None) -> mx.array:
    from eggroll_mlx import quantized_matmul_wrapper
    
    # x is float16
    denom = mat.weight_unpacked.shape[1]
    denom_scale = max(1.0, math.sqrt(denom))
    if tid is not None:
        # vectorized thread ids -> use batched matmul
        if noise is not None:
            # Optimized path: use pre-gathered noise
            A, B = noise
            # x: (P, B, K) or (B, K) broadcasted
            if x.ndim == 2:
                x = mx.broadcast_to(x[None, ...], (A.shape[0], x.shape[0], x.shape[1]))
            
            # Base matmul (FP16)
            P, B_dim, K = x.shape
            x_flat = mx.reshape(x, (P * B_dim, K))
            base_out = quantized_matmul_wrapper(x_flat, mat.weight, mat.scale, mat.bias) # (P*B, out)
            base_out = mx.reshape(base_out, (P, B_dim, mat.weight.shape[0]))
            base_out = base_out / denom_scale
            
            # Delta
            # A: (P, out, r), B: (P, in, r)
            # Cast to float32 for matmul to avoid overflow (delta can be ~500k)
            x_f = x.astype(mx.float32)
            B_f = B.astype(mx.float32)
            proj = mx.matmul(x_f, B_f) # (P, B, r)
            
            # delta = proj * A^T
            A_f = A.astype(mx.float32)
            A_T = mx.transpose(A_f, (0, 2, 1))
            delta = mx.matmul(proj, A_T) # (P, B, out)
            
            # Scaling: delta is raw noise.
            # A and B are scaled by 2^fixed_point.
            # delta = x @ B @ A.T
            # x is unscaled float. A, B are scaled.
            # delta is scaled by 2^(2*fixed_point).
            # We also want to apply sigma_shift (divide by 2^sigma_shift).
            # So total scale factor is 2^(-2*fixed_point - sigma_shift).
            scale_factor = 2.0 ** (-2 * ctx.cfg.fixed_point - ctx.cfg.sigma_shift)
            
            if seed_offset == 1 and ctx.epoch < 5:
                 print(f"DEBUG: scale_factor={scale_factor:.6f}")
                 print(f"DEBUG: base_out mean={mx.mean(mx.abs(base_out)).item():.4f}, max={mx.max(mx.abs(base_out)).item():.4f}")
                 print(f"DEBUG: delta mean={mx.mean(mx.abs(delta)).item():.4f}, max={mx.max(mx.abs(delta)).item():.4f}")
                 print(f"DEBUG: noise contribution mean={mx.mean(mx.abs(delta * scale_factor)).item():.4f}")
            
            out = base_out + (delta * scale_factor).astype(mx.float16)
            return out

        # Fallback for non-noise path (not implemented for batched yet)
        if noise is None:
            # Just run base matmul
            orig_shape = x.shape
            if x.ndim == 3:
                x_flat = mx.reshape(x, (-1, x.shape[-1]))
                base = quantized_matmul_wrapper(x_flat, mat.weight, mat.scale, mat.bias)
                base = mx.reshape(base, (orig_shape[0], orig_shape[1], -1))
            else:
                base = quantized_matmul_wrapper(x, mat.weight, mat.scale, mat.bias)
            base = base / denom_scale
            return base
        pass
    else:
        # Inference path
        orig_shape = x.shape
        if x.ndim == 3:
            x_flat = mx.reshape(x, (-1, x.shape[-1]))
            base = quantized_matmul_wrapper(x_flat, mat.weight, mat.scale, mat.bias)
            base = mx.reshape(base, (orig_shape[0], orig_shape[1], -1))
        else:
            base = quantized_matmul_wrapper(x, mat.weight, mat.scale, mat.bias)
        base = base / denom_scale
        return base
    return x # Should not reach here

# --- End Copied Code ---

def test_linear_regression_convergence():
    print("Running Linear Regression Convergence Test...")
    
    # 1. Setup Problem: Y = W_target @ X
    input_dim = 64
    output_dim = 16
    batch_size = 256
    pop_size = 2048
    steps = 2000
    
    # Target weights (random)
    key = mx.random.key(42)
    k1, k2, k3 = mx.random.split(key, 3)
    W_target = mx.random.normal((output_dim, input_dim), key=k1).astype(mx.float32)
    
    # Learned weights (random init)
    W_init = mx.random.normal((output_dim, input_dim), key=k2).astype(mx.float32) * 0.1
    mat = Int8Mat(weight=W_init, weight_unpacked=W_init)
    
    # Config
    cfg = NoiseConfig(
        rank=4,
        sigma_shift=0,
        fixed_point=4,
        learning_rate=0.01, 
        weight_clip=3.0,
        fast_fitness=False,
        update_threshold=0,
        use_clt=True
    )
    
    # Context
    param_span = (output_dim + input_dim) * cfg.rank * pop_size * 2 + 10000
    ctx = make_context(cfg, param_span=param_span, seed=1337)
    
    thread_ids = list(range(pop_size))
    thread_ids_for_update = [i * 2 for i in range(pop_size // 2)]
    
    losses = []
    
    for step in range(steps):
        ctx.epoch = step
        
        # Data batch
        X = mx.random.normal((batch_size, input_dim)).astype(mx.float32)
        Y_true = mx.matmul(X, W_target.T) # (B, Out)
        
        # Gather noise
        noise = get_lora_update_params(ctx.big_rand, ctx.cfg, ctx.epoch, thread_ids, mat.weight.shape, ctx.base_seed + 1)
        
        # Forward Pass (Population)
        # X: (B, In) -> Broadcast to (Pop, B, In)
        X_pop = mx.broadcast_to(X[None, ...], (pop_size, batch_size, input_dim))
        
        # Y_pred: (Pop, B, Out)
        Y_pred = matmul_with_noise(ctx, mat, X_pop, thread_ids, seed_offset=1, noise=noise)
        
        # Loss: MSE
        # Y_true: (B, Out) -> Broadcast to (Pop, B, Out)
        Y_true_pop = mx.broadcast_to(Y_true[None, ...], (pop_size, batch_size, output_dim))
        
        diff = Y_pred - Y_true_pop
        mse = mx.mean(diff * diff, axis=(1, 2)) # (Pop,)
        
        if step % 50 == 0:
             print(f"DEBUG: mse mean={mx.mean(mse).item():.4f}, std={mx.std(mse).item():.4f}, min={mx.min(mse).item():.4f}, max={mx.max(mse).item():.4f}")
        
        # Calculate True Gradient
        # Y_mean = X @ W.T
        Y_mean = mx.matmul(X, mat.weight.T)
        error = Y_mean - Y_true # (B, Out)
        # Cast to float32 to avoid overflow
        grad = mx.matmul(error.astype(mx.float32).T, X.astype(mx.float32)) # (Out, In)

        # Unpack noise for perfect fitness
        A_fwd, B_fwd = noise

        # Perfect Fitness Sanity Check
        # fitness = <Noise, -Grad>
        # Noise_i = A_i @ B_i.T
        # Grad = grad
        # Score_i = sum(Noise_i * -Grad)
        
        # A_fwd: (Pop, Rows, Rank), B_fwd: (Pop, Cols, Rank)
        # Grad: (Rows, Cols)
        # Noise_i . Grad = tr(B_i @ A_i.T @ Grad) = tr(A_i.T @ Grad @ B_i)
        # = sum(A_i * (Grad @ B_i))
        
        # Grad @ B_i: (Rows, Cols) @ (Cols, Rank) -> (Rows, Rank)
        # B_fwd is (Pop, Cols, Rank)
        # GB = matmul(Grad, B_fwd) -> (Pop, Rows, Rank)
        
        # We need to broadcast Grad to (Pop, Rows, Cols)? No, matmul handles (Rows,Cols) x (Pop,Cols,Rank) -> (Pop,Rows,Rank) if we transpose properly?
        # mlx.matmul(A, B). If A is (M, K), B is (P, K, N) -> (P, M, N).
        # Here Grad is (Rows, Cols). B_fwd is (Pop, Cols, Rank).
        # We want Grad @ B_fwd[i].
        # mx.matmul(grad, B_fwd) ? No, dims mismatch for broadcast.
        # mx.matmul(grad[None, ...], B_fwd) ?
        # grad: (Rows, Cols). B_fwd: (Pop, Cols, Rank).
        # We want (Pop, Rows, Rank).
        # Let's use einsum.
        # grad: ij. B: pjr. -> pir.
        GB = mx.einsum("ij,pjr->pir", grad, B_fwd)
        
        # Score = sum(A * GB)
        scores = mx.sum(A_fwd * GB, axis=(1, 2)) # (Pop,)
        
        # We want to minimize Loss, so maximize -Score.
        # But wait, Score is <Noise, Grad>.
        # If we move in direction of Noise, we change W by +Noise.
        # Change in Loss approx <Grad, Noise>.
        # If <Grad, Noise> is positive, Loss increases.
        # So we want to discourage this noise.
        # So fitness should be -Score.
        
        perfect_rewards = -scores
        # fitnesses = convert_fitnesses(cfg, perfect_rewards)
        # mx.eval(fitnesses)
        
        # Use perfect fitness for update
        fitnesses = convert_fitnesses(cfg, -mse) # Uncomment to revert
        
        # DEBUG: Check correlation between perfect fitness and MSE fitness
        if step % 50 == 0:
             # perfect_rewards and -mse should be correlated
             # We need to compute correlation on the population
             # fitnesses is already transformed, let's compare raw rewards
             
             x = perfect_rewards
             y = -mse
             vx = x - mx.mean(x)
             vy = y - mx.mean(y)
             corr = mx.sum(vx * vy) / (mx.sqrt(mx.sum(vx ** 2)) * mx.sqrt(mx.sum(vy ** 2)) + 1e-8)
             
             print(f"DEBUG: Fitness correlation (Perfect vs MSE): {corr.item():.4f}")
        
        if step % 50 == 0:
            print(f"DEBUG: fitness mean={mx.mean(mx.abs(fitnesses)).item():.4f}, max={mx.max(mx.abs(fitnesses)).item():.4f}")
        
        # Update
        # Verify noise consistency
        A_upd, B_upd = get_lora_update_params(ctx.big_rand, ctx.cfg, ctx.epoch, thread_ids_for_update, mat.weight.shape, ctx.base_seed + 1)
        
        # A_fwd: (128, ...), A_upd: (64, ...)
        # A_upd should match A_fwd[0::2] (since thread_ids_for_update are evens)
        # Note: get_lora_update_params applies anti_sign. 
        # For even threads (0, 2...), anti_sign is 1.
        # So A_upd should exactly match A_fwd[0::2].
        if not mx.array_equal(A_upd, A_fwd[0::2]):
            print("CRITICAL ERROR: Noise mismatch (A) between Forward and Update!")
            exit(1)
        if not mx.array_equal(B_upd, B_fwd[0::2]):
            print("CRITICAL ERROR: Noise mismatch (B) between Forward and Update!")
            exit(1)
            
        # Revert sign flip
        mat.weight_unpacked, diff_count = apply_sign_update(
            cfg, mat.weight_unpacked, fitnesses, ctx.big_rand, step, ctx.base_seed, 
            thread_ids=thread_ids_for_update, seed_offset=1
        )
        
        # ES Update Direction
        # updated = old + lr * delta
        # delta = (updated - old) / lr
        delta_es = (mat.weight_unpacked - mat.weight) / cfg.learning_rate
        
        # Cosine Similarity
        # flatten
        g_flat = grad.reshape(-1)
        d_flat = delta_es.reshape(-1)
        
        g_norm = mx.linalg.norm(g_flat)
        d_norm = mx.linalg.norm(d_flat)
        
        cos_sim = mx.sum(g_flat * d_flat) / (g_norm * d_norm + 1e-8)
        
        # Distance to target
        dist = mx.linalg.norm(mat.weight_unpacked - W_target)
        
        if step % 50 == 0:
            print(f"DEBUG: Grad Norm={g_norm.item():.4f}, Delta Norm={d_norm.item():.4f}, Cos Sim={cos_sim.item():.4f}, Dist={dist.item():.4f}")
        
        mat.weight = mat.weight_unpacked
        mx.eval(mat.weight)
        
        # Log
        mean_loss = mx.mean(mse).item()
        losses.append(mean_loss)
        if step % 50 == 0:
            print(f"Step {step}: Loss = {mean_loss:.4f}, Updates = {diff_count.item()}")
            
    print(f"Final Loss: {losses[-1]:.4f}")
    
    # Check convergence
    if losses[-1] < losses[0] * 0.5:
        print("SUCCESS: Loss decreased significantly.")
    else:
        print("FAILURE: Loss did not decrease significantly.")
        exit(1)

if __name__ == "__main__":
    test_linear_regression_convergence()
