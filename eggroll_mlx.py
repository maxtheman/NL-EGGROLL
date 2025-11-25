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
    use_clt: bool = True
    fast_fitness: bool = True
    fixed_point: int = 4  # 2^4 scale for int8 path
    use_quantized_base: bool = False  # optional fast path via mlx.quantized_matmul
    quant_bits: int = 8
    quant_group_size: int = 32
    update_threshold: int = 0  # optional magnitude threshold before applying sign update
    fitness_alpha: float = 0.01  # scale factor for CLT fitness normalization


def fold_in(base_key_int32: Tuple[int, int], new_int32: int) -> int:
    """
    Matches the fold_in logic in nano-egg (see run.py:424-439).
    base_key_int32 is a tuple/list of two uint32 ints (jax key data style).
    """
    x = new_int32
    x = ((x >> 16) ^ x) * 0x45D9F3B
    return base_key_int32[0] ^ x


def get_common_start_idx(cfg: NoiseConfig, epoch: int, thread_id: int, base_seed: int) -> Tuple[int, int]:
    """
    Compute start index into BIG_RAND and antithetic sign.
    Mirrors nano-egg get_common_start_idx.
    """
    try:
        import run as nano  # type: ignore
        import jax

        dummy_param = jax.numpy.zeros((2, 2))
        start_idx, sign = nano.get_common_start_idx(
            {"noise_reuse": cfg.noise_reuse}, (epoch, thread_id), dummy_param, jax.random.key(int(base_seed))
        )
        return int(start_idx), int(sign)
    except Exception:
        true_epoch = 0 if cfg.noise_reuse == 0 else epoch // cfg.noise_reuse
        true_thread_idx = (thread_id >> 1)
        actual_key = fold_in((int(base_seed), 0), true_thread_idx + true_epoch)
        start_idx = actual_key & (2**30 - 1)
        sign = 1 if (thread_id % 2 == 0) else -1
        return start_idx, sign


def get_lora_update_params(
    big_rand: mx.array,
    cfg: NoiseConfig,
    epoch: int,
    thread_id: int,
    param_shape: Tuple[int, int],
    base_seed: int,
) -> Tuple[mx.array, mx.array]:
    """
    Slice BIG_RAND into A (rows x rank) and B (cols x rank).
    Order matches nano-egg: first B (cols), then A (rows).
    """
    rows, cols = param_shape
    r = cfg.rank
    span = (rows + cols) * r
    start_idx, anti_sign = get_common_start_idx(cfg, epoch, thread_id, base_seed)
    start_idx = int(start_idx) % max(1, int(big_rand.shape[0] - span))
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
    thread_ids: List[int],
    param_shape: Tuple[int, int],
    base_seed: int,
) -> Tuple[mx.array, mx.array]:
    """Batch slice A/B for a list of thread_ids."""
    A_list: List[mx.array] = []
    B_list: List[mx.array] = []
    for tid in thread_ids:
        A, B = get_lora_update_params(big_rand, cfg, epoch, tid, param_shape, base_seed)
        A_list.append(A.astype(mx.int32))
        B_list.append(B.astype(mx.int32))
    return mx.stack(A_list, axis=0), mx.stack(B_list, axis=0)


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

    denom = int(np.sqrt(w.shape[1]))
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

    pop_dim = 1 if x.ndim == 2 else x.shape[0]
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

    out_int = base_out + delta
    denom = int(np.sqrt(w.shape[1]))
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
) -> mx.array:
    """
    Apply sign update using low-rank perturbations, paired antithetic.
    fitnesses shape: (pop/2,) already paired.
    thread_ids optionally selects which perturbation indices to use per pair (default: range(pop/2)).
    seed_offset optionally shifts the base_seed (e.g., per-layer offset).
    """
    try:
        import jax
        import jax.numpy as jnp
        import run as nano  # type: ignore

        p_jax = jnp.array(np.array(param))
        f_jax = jnp.array(np.array(fitnesses))
        nr = {
            "noise_reuse": cfg.noise_reuse,
            "rank": cfg.rank,
            "use_clt": cfg.use_clt,
            "fast_fitness": cfg.fast_fitness,
        }
        np_params = {
            "BIG_RAND_MATRIX": jnp.array(np.array(big_rand)),
            "sigma_shift": cfg.sigma_shift,
            "update_threshold": 0,
        }
        iterinfos = (
            jnp.full(fitnesses.shape[0], epoch, dtype=jnp.int32),
            jnp.arange(fitnesses.shape[0], dtype=jnp.int32),
        )
        updated = nano.QEggRoll._do_update(nr, np_params, p_jax, jax.random.key(int(base_seed)), f_jax, iterinfos, 1)
        return mx.array(np.array(updated), dtype=param.dtype)
    except Exception:
        pass
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
    thresh = cfg.update_threshold
    if thresh > 0:
        # scale threshold relative to reference pop_pairs=64 to avoid over-gating small pops
        thresh = thresh * (pop_pairs / 64.0)
        Z_mask = mx.where(mx.abs(Z) >= thresh, mx.sign(Z), mx.zeros_like(Z))
        delta = Z_mask.astype(mx.int32)
    else:
        delta = mx.sign(Z).astype(mx.int32)
    updated = param.astype(mx.int32) + delta
    updated = mx.clip(updated, -127, 127).astype(param.dtype)
    return updated


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
    updated = param.astype(mx.int32) + mx.sign(accumulator)
    updated = mx.clip(updated, -127, 127).astype(param.dtype)
    return updated


def generate_big_rand(numel: int, seed: int = 0, fixed_point: int = 4, dtype=mx.int8) -> mx.array:
    """
    Generate a BIG_RAND array similar to nano-egg (normal * 2^fixed_point).
    """
    key = mx.random.key(seed)
    scale = 2**fixed_point
    arr = mx.random.normal((numel,), key=key) * scale
    return arr.astype(dtype)


_int8_mm_kernel = None

if _HAS_MXFAST:
    _int8_mm_kernel = mxfast.metal_kernel(
        name="int8_mm",
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
            for (int t = 0; t < Kv; ++t) {
                acc += int32_t(a[base_a + t]) * int32_t(b[base_b + t]);
            }
            out[elem] = acc;
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
    if _int8_mm_kernel is None:
        raise RuntimeError("int8 matmul kernel missing; mlx.fast must be available.")
    out_flat = _int8_mm_kernel(
        inputs=[a, b, mx.array([m], dtype=mx.int32), mx.array([n], dtype=mx.int32), mx.array([k], dtype=mx.int32)],
        output_shapes=[(m * n,)],
        output_dtypes=[mx.int32],
        grid=(m * n, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]
    return mx.reshape(out_flat, (m, n))


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
