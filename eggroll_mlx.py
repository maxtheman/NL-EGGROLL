"""
Minimal Eggroll implementation in MLX, inspired by the nano-egg reference.

Goals:
- Low-rank perturbations (LoRA-style) for ES updates.
- Deterministic noise slicing via a big noise table and (epoch, thread_id) addressing.
- Optional fixed-point style shift (sigma_shift) and sign-step updates for parity tests.

This module is designed to be testable against the nano-egg reference (third_party_references/repos/nano-egg/run.py)
on small shapes. It favors clarity over speed; the batched forward dedup lives in scripts/exp_3_batched_eggroll.py.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import os

import numpy as np
import mlx.core as mx

# Require the Metal fast path explicitly.
try:
    from mlx.core import fast as mxfast  # type: ignore
    _HAS_MXFAST = True
except Exception as e:  # pragma: no cover
    _HAS_MXFAST = False
    raise ImportError(
        "mlx.core.fast (Metal fast path) is required for int8 eggroll; install MLX with fast kernels."
    ) from e


@dataclass
class NoiseConfig:
    rank: int = 1
    sigma_shift: int = 4  # matches nano-egg default
    noise_reuse: int = 1
    use_clt: bool = True  # matches nano-egg default
    fast_fitness: bool = True
    fixed_point: int = 4  # 2^4 scale for int8 path
    use_quantized_base: bool = False  # optional fast path via mlx.quantized_matmul
    quant_bits: int = 8
    quant_group_size: int = 32
    update_threshold: int = 0  # optional magnitude threshold before applying sign update
    fitness_alpha: float = 0.01  # scale factor for CLT fitness normalization
    debug_perturbations: bool = False  # print noise stats (WARNING: significantly slows down training due to CPU sync)
    learning_rate: float = 0.01  # step size multiplier for sign updates (fp path)
    weight_clip: float = 5.0     # optional clip after updates to prevent divergence


def fold_in(base_key_int32: Tuple[int, int], new_int32: int | mx.array) -> int | mx.array:
    """
    Matches the fold_in logic in nano-egg (see run.py:424-439).
    base_key_int32 is a tuple/list of two uint32 ints (jax key data style).
    Supports both scalar int and mx.array for new_int32.
    """
    x = new_int32
    if isinstance(x, mx.array):
        # Vectorized path
        x = ((x >> 16) ^ x) * 0x45D9F3B
        # base_key_int32[0] is scalar, x is array
        return mx.array(base_key_int32[0], dtype=x.dtype) ^ x
    else:
        # Scalar path
        x = ((x >> 16) ^ x) * 0x45D9F3B
        return base_key_int32[0] ^ x


def get_common_start_idx(
    cfg: NoiseConfig, epoch: int, thread_id: int | mx.array, base_seed: int
) -> Tuple[int | mx.array, int | mx.array]:
    """
    Compute start index into BIG_RAND and antithetic sign.
    Mirrors nano-egg get_common_start_idx.
    Supports vectorized thread_id (mx.array).
    """
    # Note: We skip the jax/nano import check for the vectorized path to keep it simple
    # and because we are optimizing for MLX speed.
    
    true_epoch = 0 if cfg.noise_reuse == 0 else epoch // cfg.noise_reuse
    
    if isinstance(thread_id, mx.array):
        true_thread_idx = (thread_id >> 1)
        # fold_in handles array broadcasting
        actual_key = fold_in((int(base_seed), 0), true_thread_idx + true_epoch)
        start_idx = actual_key & (2**30 - 1)
        # Vectorized sign: 1 if even, -1 if odd
        # (thread_id % 2) is 0 or 1. 
        # 1 - 2*(thread_id % 2) -> 1 if 0, -1 if 1.
        # Wait, nano-egg: sign = 1 if (thread_id % 2 == 0) else -1
        # If thread_id % 2 == 0 -> 1
        # If thread_id % 2 == 1 -> -1
        # Formula: 1 - 2 * (thread_id % 2)
        sign = 1 - 2 * (thread_id % 2)
        return start_idx, sign
    else:
        true_thread_idx = (thread_id >> 1)
        actual_key = fold_in((int(base_seed), 0), true_thread_idx + true_epoch)
        start_idx = actual_key & (2**30 - 1)
        sign = 1 if (thread_id % 2 == 0) else -1
        return start_idx, sign


def get_lora_update_params(
    big_rand: mx.array,
    cfg: NoiseConfig,
    epoch: int,
    thread_id: int | mx.array | List[int],
    param_shape: Tuple[int, int],
    base_seed: int,
) -> Tuple[mx.array, mx.array]:
    """
    Slice BIG_RAND into A (rows x rank) and B (cols x rank).
    Order matches nano-egg: first B (cols), then A (rows).
    Supports vectorized thread_id (mx.array or list).
    """
    rows, cols = param_shape
    r = cfg.rank
    span = (rows + cols) * r
    
    # Handle list input
    if isinstance(thread_id, list):
        thread_id = mx.array(thread_id, dtype=mx.int32)
        
    start_idx, anti_sign = get_common_start_idx(cfg, epoch, thread_id, base_seed)
    
    limit = max(1, int(big_rand.shape[0] - span))
    
    if isinstance(start_idx, mx.array):
        # Vectorized path
        start_idx = start_idx % limit
        # Gather: (pop, span)
        indices = start_idx[:, None] + mx.arange(span)[None, :]
        slice_ = big_rand[indices]
        slice_ = mx.reshape(slice_, (start_idx.shape[0], rows + cols, r))
        
        B = slice_[:, :cols]
        A = slice_[:, cols:]
        
        # Apply sign (pop, 1, 1) broadcast
        return A * anti_sign[:, None, None], B
    else:
        # Scalar path
        start_idx = int(start_idx) % limit
        slice_ = mx.reshape(big_rand[start_idx : start_idx + span], (rows + cols, r))
        B = slice_[:cols]
        A = slice_[cols:]
        return A * anti_sign, B


def get_nonlora_update_params(
    big_rand: mx.array,
    cfg: NoiseConfig,
    epoch: int,
    thread_id: int,
    param_shape: Tuple[int, ...],
    base_seed: int,
) -> mx.array:
    """
    Full-parameter perturbation: product of two int8 values (matches nano-egg sign logic).
    """
    span = int(np.prod(param_shape)) * 2
    start_idx, anti_sign = get_common_start_idx(cfg, epoch, thread_id, base_seed)
    start_idx = int(start_idx) % max(1, int(big_rand.shape[0] - span))
    updates = mx.reshape(big_rand[start_idx : start_idx + span], param_shape + (2,))
    return mx.prod(updates.astype(mx.int32), axis=-1) * anti_sign


def convert_fitnesses(cfg: NoiseConfig, raw_scores: mx.array) -> mx.array:
    """
    Paired scores -> sign (fast fitness). raw_scores shape: (pop,)
    """
    paired = mx.reshape(raw_scores, (-1, 2))
    if cfg.fast_fitness:
        return mx.sign(paired[:, 0] - paired[:, 1]).astype(mx.int8)
    diff = paired[:, 0] - paired[:, 1]
    rms = mx.sqrt(mx.mean(diff * diff) + 1e-8)
    return (diff / rms * cfg.fitness_alpha).astype(mx.float32)


def summarize(name: str, arr: mx.array) -> dict:
    """Return basic stats for an array."""
    a = np.array(arr)
    return {
        "name": name,
        "shape": a.shape,
        "dtype": str(a.dtype),
        "mean": float(a.mean()) if a.size else 0.0,
        "std": float(a.std()) if a.size else 0.0,
        "min": float(a.min()) if a.size else 0.0,
        "max": float(a.max()) if a.size else 0.0,
    }


def _stack_lora_params(
    cfg: NoiseConfig,
    big_rand: mx.array,
    epoch: int,
    thread_ids: List[int] | mx.array,
    param_shape: Tuple[int, int],
    base_seed: int,
) -> Tuple[mx.array, mx.array]:
    """Batch slice A/B for a list of thread_ids."""
    # Now fully vectorized via get_lora_update_params
    return get_lora_update_params(big_rand, cfg, epoch, thread_ids, param_shape, base_seed)


def _quantized_base_mm(
    cfg: NoiseConfig, x: mx.array, w: mx.array
) -> Optional[mx.array]:
    """
    Fast base matmul using mlx.quantized_matmul (float activations, quantized weights).
    Returns float32 output or None if unsupported for the given shapes/config.
    """
    # quantized_matmul needs last dim divisible by group_size
    if w.shape[1] % cfg.quant_group_size != 0:
        return None
    x_float = x.astype(mx.float32) if x.dtype != mx.float32 else x
    w_float = w.astype(mx.float32)
    try:
        q_weight, scales, *biases = mx.quantize(
            w_float, cfg.quant_group_size, cfg.quant_bits, mode="affine"
        )
        bias = biases[0] if biases else None
        out = mx.quantized_matmul(
            x_float,
            q_weight,
            scales=scales,
            biases=bias,
            transpose=True,
            group_size=cfg.quant_group_size,
            bits=cfg.quant_bits,
            mode="affine",
        )
        return out
    except Exception:
        return None


def do_mm(
    cfg: NoiseConfig,
    x: mx.array,
    w: mx.array,
    big_rand: Optional[mx.array],
    epoch: Optional[int],
    thread_id: Optional[int],
    base_seed: Optional[int],
):
    """
    Matrix multiply with optional low-rank perturbation injected.
    - Default path: int8 inputs/weights with custom Metal kernel (parity path).
    - Optional fast path: QuantizedLinear-style base matmul (float activations, quantized weights),
      enabled by cfg.use_quantized_base when shapes allow (group_size divides K).
    - If epoch/thread_id is None, returns base matmul / sqrt(d) (no noise).
    - If provided, applies low-rank delta using get_lora_update_params.
    """
    base_out: Optional[mx.array] = None

    if cfg.use_quantized_base:
        base_out_float = _quantized_base_mm(cfg, x, w)
        if base_out_float is not None:
            base_out = mx.round(base_out_float * (2 ** cfg.fixed_point)).astype(mx.int32)

    if base_out is None:
        if x.dtype != mx.int8 or w.dtype != mx.int8:
            raise ValueError("do_mm expects int8 inputs/weights when not using quantized base; quantize first.")
        x_int = x.astype(mx.int32)
        base_out = int8_matmul(x, w)  # int32
    else:
        # Build int representation of activations for delta path
        if x.dtype == mx.int8:
            x_int = x.astype(mx.int32)
        else:
            x_int = mx.round(x.astype(mx.float32) * (2 ** cfg.fixed_point)).astype(mx.int32)

    if epoch is not None and thread_id is not None and big_rand is not None and base_seed is not None:
        A, B = get_lora_update_params(big_rand, cfg, epoch, thread_id, w.shape, base_seed)
        A_int = A.astype(mx.int32)
        B_int = B.astype(mx.int32)
        # delta = (x @ B) @ A.T with int32 ops
        proj = mx.sum(x_int[:, :, None] * B_int[None, :, :], axis=1)  # (batch, rank)
        delta = mx.sum(proj[:, :, None] * A_int[None, :, :], axis=1)  # (batch, out)
        delta = delta >> cfg.sigma_shift
        base_out = base_out + delta

    denom = w.shape[1]
    scale = (2 ** cfg.fixed_point) * max(1, denom)
    out = base_out // scale
    out = mx.clip(out, -127, 127)
    return out.astype(mx.int8)


def do_mm_batched(
    cfg: NoiseConfig,
    x: mx.array,
    w: mx.array,
    big_rand: mx.array,
    epoch: int,
    thread_ids: List[int],
    base_seed: int,
) -> mx.array:
    """
    Batched variant: apply per-thread low-rank deltas.
    If x has shape (batch, k), activations are shared across pop.
    If x has shape (pop, batch, k), activations are per-pop.
    Returns shape (pop, batch, out_dim), int8.
    """
    if len(thread_ids) == 0:
        raise ValueError("thread_ids must be non-empty for batched matmul.")

    pop_dim = len(thread_ids) if x.ndim == 2 else x.shape[0]
    if x.ndim == 2:
        batch_x = x
        x_int = x.astype(mx.int32)
        base_out = int8_matmul(batch_x, w)
        base_out = mx.broadcast_to(base_out[None, :, :], (pop_dim, base_out.shape[0], base_out.shape[1]))
        x_int = mx.broadcast_to(x_int[None, :, :], (pop_dim, x_int.shape[0], x_int.shape[1]))
    else:
        P, B, K = x.shape
        batch_x = mx.reshape(x, (P * B, K))
        base_out = int8_matmul(batch_x, w)  # (P*B, out)
        base_out = mx.reshape(base_out, (P, B, w.shape[0]))
        x_int = x.astype(mx.int32)

    rows, cols = w.shape
    A_stack, B_stack = _stack_lora_params(cfg, big_rand, epoch, thread_ids, (rows, cols), base_seed)
    x_exp = x_int[:, :, :, None]  # (pop, batch, k, 1)
    B_exp = B_stack[:, None, :, :]  # (pop, 1, k, rank)
    proj = mx.sum(x_exp * B_exp, axis=2).astype(mx.int32)  # (pop, batch, rank)
    proj_exp = proj[:, :, :, None]  # (pop, batch, rank, 1)
    A_exp = A_stack[:, None, :, :]  # (pop, 1, rows, rank)
    delta = mx.sum(proj_exp * A_exp, axis=2).astype(mx.int32)  # (pop, batch, rows)
    delta = delta >> cfg.sigma_shift

    if cfg.debug_perturbations:
        d_float = delta.astype(mx.float32)
        print(f"DEBUG: do_mm_batched noise: mean={d_float.mean().item():.4f}, std={d_float.std().item():.4f}, max={d_float.max().item():.4f}, min={d_float.min().item():.4f}")

    out_int = base_out + delta
    denom = w.shape[1]
    scale = (2 ** cfg.fixed_point) * max(1, denom)
    out = out_int // scale
    out = mx.clip(out, -127, 127)
    return out.astype(mx.int8)


def apply_sign_update(
    cfg: NoiseConfig,
    param: mx.array,
    fitnesses: mx.array,
    big_rand: mx.array,
    epoch: int,
    base_seed: int,
    thread_ids: Optional[List[int]] = None,
    seed_offset: int = 0,
) -> Tuple[mx.array, int]:
    """
    Apply sign update using low-rank perturbations, paired antithetic.
    Returns (updated_param, num_changed_params).
    """
    # try:
    #     import jax
    #     import jax.numpy as jnp
    #     import run as nano  # type: ignore
    #     ...
    # except Exception:
    #     pass
    assert param.ndim == 2, "Only matrix params handled here"
    rows, cols = param.shape
    pop_pairs = fitnesses.shape[0]
    thread_ids = thread_ids or list(range(pop_pairs))
    if len(thread_ids) != pop_pairs:
        raise ValueError(f"thread_ids length {len(thread_ids)} must match fitnesses length {pop_pairs}")
    A_stack, B_stack = _stack_lora_params(cfg, big_rand, epoch, thread_ids, (rows, cols), base_seed + seed_offset)

    if cfg.use_clt:
        A_scaled = A_stack.astype(mx.float32) * fitnesses.reshape(-1, 1, 1).astype(mx.float32)
        B_scaled = B_stack.astype(mx.float32)
    else:
        A_scaled = mx.sign(A_stack).astype(mx.float32) * fitnesses.reshape(-1, 1, 1).astype(mx.float32)
        B_scaled = mx.sign(B_stack).astype(mx.float32)

    Z = mx.einsum("nir,njr->ij", A_scaled, B_scaled)
    Z = mx.einsum("nir,njr->ij", A_scaled, B_scaled)
    thresh = cfg.update_threshold
    if thresh > 0:
        # Reference logic:
        # abs(Z) * 2^FP < thresh * sqrt(pop) * (4^FP if clt else 1)
        # So: abs(Z) < thresh * sqrt(pop) * (2^FP if clt else 2^-FP)
        
        scale_factor = np.sqrt(pop_pairs)
        if cfg.use_clt:
            # thresh * sqrt(pop) * 4^FP / 2^FP = thresh * sqrt(pop) * 2^FP
            scale_factor *= (2 ** cfg.fixed_point)
        else:
            # thresh * sqrt(pop) / 2^FP
            scale_factor /= (2 ** cfg.fixed_point)
            
        thresh_val = thresh * scale_factor
        Z_mask = mx.where(mx.abs(Z) >= thresh_val, mx.sign(Z), mx.zeros_like(Z))
        delta = Z_mask.astype(mx.int32)
    else:
        delta = mx.sign(Z).astype(mx.int32)
    updated = param.astype(mx.float32) + cfg.learning_rate * delta.astype(mx.float32)
    if cfg.weight_clip is not None:
        updated = mx.clip(updated, -cfg.weight_clip, cfg.weight_clip)
    num_changed = mx.sum(mx.abs(delta))
    return updated.astype(param.dtype), num_changed


def apply_full_update(
    cfg: NoiseConfig,
    param: mx.array,
    fitnesses: mx.array,
    big_rand: mx.array,
    epoch: int,
    base_seed: int,
) -> mx.array:
    """
    Full-parameter perturbation sign update (not low-rank), paired antithetic.
    """
    pop_pairs = fitnesses.shape[0]
    accumulator = mx.zeros_like(param.astype(mx.int32))
    for pair_idx in range(pop_pairs):
        score = fitnesses[pair_idx]
        for sign_idx, thread_sign in enumerate([0, 1]):
            thread_id = pair_idx * 2 + thread_sign
            noise = get_nonlora_update_params(big_rand, cfg, epoch, thread_id, param.shape, base_seed)
            if cfg.use_clt:
                accumulator = accumulator + noise.astype(mx.int32) * score
            else:
                accumulator = accumulator + mx.sign(noise).astype(mx.int32) * score
    updated = param.astype(mx.float32) + cfg.learning_rate * mx.sign(accumulator).astype(mx.float32)
    if cfg.weight_clip is not None:
        updated = mx.clip(updated, -cfg.weight_clip, cfg.weight_clip)
    return updated.astype(param.dtype)


def generate_big_rand(numel: int, seed: int = 0, fixed_point: int = 4, dtype=mx.int8) -> mx.array:
    """
    Generate a BIG_RAND array similar to nano-egg (normal * 2^fixed_point).
    """
    key = mx.random.key(seed)
    scale = 2**fixed_point
    arr = mx.random.normal((numel,), key=key) * scale
    return arr.astype(dtype)


USE_TILED_INT8_KERNEL = os.getenv("INT8_KERNEL_TILE", "0") == "1"
_int8_mm_kernel_vec = None
_int8_mm_kernel_tiled = None

if _HAS_MXFAST:
    # Baseline vectorized (one thread per output)
    _int8_mm_kernel_vec = mxfast.metal_kernel(
        name="int8_mm_vec",
        input_names=["a", "b", "M", "N", "K"],
        output_names=["out"],
        source=r"""
            uint elem = thread_position_in_grid.x;
            int Mv = int(M[0]);
            int Nv = int(N[0]);
            int Kv = int(K[0]);
            int row = int(elem / Nv);
            int col = int(elem - row * Nv);
            if (row >= Mv || col >= Nv) return;
            int base_a = row * Kv;
            int base_b = col * Kv;
            int32_t acc = 0;
            int kv4 = Kv >> 2;
            const device char4* a4 = reinterpret_cast<const device char4*>(a + base_a);
            const device char4* b4 = reinterpret_cast<const device char4*>(b + base_b);
            for (int i = 0; i < kv4; ++i) {
                char4 va = a4[i];
                char4 vb = b4[i];
                acc += int32_t(va.x) * int32_t(vb.x);
                acc += int32_t(va.y) * int32_t(vb.y);
                acc += int32_t(va.z) * int32_t(vb.z);
                acc += int32_t(va.w) * int32_t(vb.w);
            }
            int tail = kv4 << 2;
            for (int t = tail; t < Kv; ++t) {
                acc += int32_t(a[base_a + t]) * int32_t(b[base_b + t]);
            }
            out[elem] = acc;
        """,
        ensure_row_contiguous=True,
    )

    # Experimental tiled kernel (16x16 tile, TK=32) for tuning
    _int8_mm_kernel_tiled = mxfast.metal_kernel(
        name="int8_mm_tiled",
        input_names=["a", "b", "M", "N", "K"],
        output_names=["out"],
        source=r"""
            constexpr ushort TM = 16;
            constexpr ushort TN = 16;
            constexpr ushort TK = 32;

            int Mv = int(M[0]);
            int Nv = int(N[0]);
            int Kv = int(K[0]);
            
            // 2D Grid Dispatch
            uint tg_x = threadgroup_position_in_grid.x;
            uint tg_y = threadgroup_position_in_grid.y;
            ushort lx = thread_position_in_threadgroup.x; // 0..15 (col in tile)
            ushort ly = thread_position_in_threadgroup.y; // 0..15 (row in tile)
            
            uint row = tg_y * TM + ly;
            uint col = tg_x * TN + lx;
            uint tid = ly * TN + lx; // Linear thread ID 0..255

            // Shared Memory
            threadgroup char Asub[TM * TK];
            threadgroup char Bsub[TN * TK];

            int32_t acc = 0;

            // Loop over K in chunks of TK
            for (int k0 = 0; k0 < Kv; k0 += TK) {
                
                // Load A tile
                for (uint i = tid; i < TM * TK; i += 256) {
                    uint r = i / TK;      // row in tile
                    uint k = i % TK;      // k in tile
                    uint global_r = tg_y * TM + r;
                    uint global_k = k0 + k;
                    
                    char val = 0;
                    if (global_r < (uint)Mv && global_k < (uint)Kv) {
                        val = a[global_r * Kv + global_k];
                    }
                    Asub[i] = val;
                }

                // Load B tile
                for (uint i = tid; i < TN * TK; i += 256) {
                    uint c = i / TK;      // col in tile (row in b)
                    uint k = i % TK;      // k in tile
                    uint global_c = tg_x * TN + c;
                    uint global_k = k0 + k;
                    
                    char val = 0;
                    if (global_c < (uint)Nv && global_k < (uint)Kv) {
                        val = b[global_c * Kv + global_k];
                    }
                    Bsub[i] = val;
                }
                
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Compute
                const threadgroup char* A_row = Asub + ly * TK;
                const threadgroup char* B_col = Bsub + lx * TK;
                
                for (int k = 0; k < TK; ++k) {
                    // Check boundary for partial K tiles
                    if (k0 + k < Kv) {
                        acc += int32_t(A_row[k]) * int32_t(B_col[k]);
                    }
                }
                
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (row < (uint)Mv && col < (uint)Nv) {
                out[row * Nv + col] = acc;
            }
        """,
        ensure_row_contiguous=True,
    )


def int8_matmul(a: mx.array, b: mx.array) -> mx.array:
    """
    Compute a @ b^T for int8 inputs, returning int32.
    a: (m, k), b: (n, k). Requires Metal kernel.
    """
    m, k = a.shape
    n = b.shape[0]
    kernel = _int8_mm_kernel_tiled if (USE_TILED_INT8_KERNEL and _int8_mm_kernel_tiled is not None) else _int8_mm_kernel_vec
    if kernel is None:
        raise RuntimeError("int8 matmul kernel missing; mlx.fast must be available.")

    if kernel is _int8_mm_kernel_tiled:
        tile_m = 16
        tile_n = 16
        grid_x = (n + tile_n - 1) // tile_n
        grid_y = (m + tile_m - 1) // tile_m
        out_flat = kernel(
            inputs=[a, b, mx.array([m], dtype=mx.int32), mx.array([n], dtype=mx.int32), mx.array([k], dtype=mx.int32)],
            output_shapes=[(int(m * n),)],
            output_dtypes=[mx.int32],
            grid=(int(grid_x * 16), int(grid_y * 16), 1),
            threadgroup=(16, 16, 1),
        )[0]
    else:
        out_flat = kernel(
            inputs=[a, b, mx.array([m], dtype=mx.int32), mx.array([n], dtype=mx.int32), mx.array([k], dtype=mx.int32)],
            output_shapes=[(int(m * n),)],
            output_dtypes=[mx.int32],
            grid=(int(m * n), 1, 1),
            threadgroup=(256, 1, 1),
        )[0]
    return mx.reshape(out_flat, (m, n))


def quantized_matmul_wrapper(x: mx.array, w: mx.array, scales: mx.array, biases: Optional[mx.array] = None, group_size: int = 64) -> mx.array:
    """
    Float matmul wrapper (fp16 -> fp16).
    x: (..., K) float16/32
    w: (N, K) float16/32
    """
    if x.dtype != mx.float16:
        x = x.astype(mx.float16)
    return mx.matmul(x, mx.transpose(w))



def calibrate_divisor(x: mx.array, w: mx.array, target_max: int = 32) -> int:
    """
    Compute an integer divisor so that max(|x @ w.T|) / divisor is near target_max.
    Does not apply fixed_point or sqrt(d); meant to guide choosing scale factors.
    """
    if x.dtype != mx.int8 or w.dtype != mx.int8:
        raise ValueError("calibrate_divisor expects int8 inputs/weights.")
    base_out = int8_matmul(x, w)
    max_abs = int(mx.max(mx.abs(base_out)).item())
    if max_abs == 0:
        return 1
    return max(1, int(np.ceil(max_abs / target_max)))
