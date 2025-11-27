"""
Closer-to-reference TinyStories trainer:
- Multi-layer GRU + MLP blocks with layernorm and residuals (float LN).
- All linear weights are int8 with low-rank perturbations (do_mm).
- Per-layer seed offsets for noise; distinct thread_id offsets per matmul.
Note: LN and MLP are float-heavy to keep code simple; fixed-point LN like the C ref is not implemented.
"""

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from eggroll_api import EggrollContext, make_context
from eggroll_mlx import (
    NoiseConfig,
    apply_sign_update,
    calibrate_divisor,
    convert_fitnesses,
    do_mm,
    do_mm_batched,
    int8_matmul,
    _stack_lora_params,
)
from scripts.train_tinystories_gru_int8 import load_tokens, quantize, get_batch

def cross_entropy(logits, targets, reduction='mean'):
    # logits: (..., V), targets: (...,) int
    # Gather-based implementation to avoid one-hot allocation
    if targets.ndim < logits.ndim - 1:
        targets = mx.broadcast_to(targets, logits.shape[:-1])
    max_logits = mx.max(logits, axis=-1, keepdims=True)
    logsumexp = max_logits + mx.log(mx.sum(mx.exp(logits - max_logits), axis=-1, keepdims=True))
    target_logits = mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    loss = logsumexp[..., 0] - target_logits
    if reduction == 'mean':
        return mx.mean(loss)
    elif reduction == 'none':
        return loss
    return loss




@dataclass
class Int8Mat:
    weight: mx.array            # float16 weights for compute
    weight_unpacked: mx.array   # float16 weights for updates/noise (alias of weight)
    scale: Optional[mx.array] = None
    bias: Optional[mx.array] = None


@dataclass
class BlockWeights:
    ln1_scale: mx.array  # float16
    ln1_bias: mx.array   # float16
    ln2_scale: mx.array
    ln2_bias: mx.array
    gru_x: Int8Mat  # 3H x D
    gru_h: Int8Mat  # 3H x H
    mlp_in: Int8Mat  # 4H x H
    mlp_out: Int8Mat  # H x 4H
    gru_b1: Optional[mx.array] = None  # gate biases
    gru_b2: Optional[mx.array] = None
    mlp_b1: Optional[mx.array] = None
    mlp_b2: Optional[mx.array] = None


@dataclass
class ModelWeights:
    tok_emb: mx.array
    pos_emb: mx.array
    blocks: List[BlockWeights]
    ln_out_scale: mx.array
    ln_out_bias: mx.array
    head: Int8Mat
    head_bias: Optional[mx.array] = None


def init_model(cfg: NoiseConfig, vocab: int, seq_len: int, d_model: int, d_hidden: int, n_layers: int, rng, init_scale: float) -> ModelWeights:
    def q(shape):
        # Initialize float weights.
        w_float = rng.normal(0, 0.05, size=shape).astype(np.float16)
        w_mx = mx.array(w_float)
        return Int8Mat(weight=w_mx, weight_unpacked=w_mx, scale=None, bias=None)

    tok_emb = mx.array(rng.normal(0, 0.05, size=(vocab, d_model)).astype(np.float16))
    pos_emb = mx.array(rng.normal(0, 0.05, size=(seq_len, d_model)).astype(np.float16))
    
    blocks: List[BlockWeights] = []
    for _ in range(n_layers):
        ln1_scale = mx.ones((d_model,), dtype=mx.float16)
        ln1_bias = mx.zeros((d_model,), dtype=mx.float16)
        ln2_scale = mx.ones((d_model,), dtype=mx.float16)
        ln2_bias = mx.zeros((d_model,), dtype=mx.float16)
        
        gru_x = q((3 * d_hidden, d_model))
        gru_h = q((3 * d_hidden, d_hidden))
        mlp_in = q((4 * d_hidden, d_hidden))
        mlp_out = q((d_hidden, 4 * d_hidden))
        
        gru_b1 = mx.zeros((3 * d_hidden,), dtype=mx.float16)
        gru_b2 = mx.zeros((3 * d_hidden,), dtype=mx.float16)
        mlp_b1 = mx.zeros((4 * d_hidden,), dtype=mx.float16)
        mlp_b2 = mx.zeros((d_hidden,), dtype=mx.float16)
        
        blocks.append(
            BlockWeights(
                ln1_scale=ln1_scale, ln1_bias=ln1_bias,
                ln2_scale=ln2_scale, ln2_bias=ln2_bias,
                gru_x=gru_x, gru_h=gru_h, mlp_in=mlp_in, mlp_out=mlp_out,
                gru_b1=gru_b1, gru_b2=gru_b2, mlp_b1=mlp_b1, mlp_b2=mlp_b2,
            )
        )
    ln_out_scale = mx.ones((d_model,), dtype=mx.float16)
    ln_out_bias = mx.zeros((d_model,), dtype=mx.float16)
    head = q((vocab, d_model))
    head_bias = mx.zeros((vocab,), dtype=mx.float16)
    
    return ModelWeights(tok_emb, pos_emb, blocks, ln_out_scale, ln_out_bias, head, head_bias)


def pack_int8_to_uint32(w_float: mx.array):
    """
    Placeholder for compatibility; returns the float weights unchanged.
    """
    return w_float, None, None

def save_checkpoint(weights: ModelWeights, path: str):
    print(f"Saving checkpoint to {path}...")
    tensors = {}
    
    # Save embeddings
    tensors["tok_emb"] = weights.tok_emb
    tensors["pos_emb"] = weights.pos_emb
    
    # Save Head
    tensors["head.weight"] = weights.head.weight
    tensors["head.weight_unpacked"] = weights.head.weight_unpacked
    if weights.head_bias is not None:
        tensors["head_bias"] = weights.head_bias
        
    # Save Output LN
    tensors["ln_out_scale"] = weights.ln_out_scale
    tensors["ln_out_bias"] = weights.ln_out_bias
    
    # Save Blocks
    for i, blk in enumerate(weights.blocks):
        prefix = f"blocks.{i}"
        
        # Layer Norms
        tensors[f"{prefix}.ln1_scale"] = blk.ln1_scale
        tensors[f"{prefix}.ln1_bias"] = blk.ln1_bias
        tensors[f"{prefix}.ln2_scale"] = blk.ln2_scale
        tensors[f"{prefix}.ln2_bias"] = blk.ln2_bias
        
        # GRU/MLP Matrices (Save both packed and unpacked for resumption)
        for name, mat in [("gru_x", blk.gru_x), ("gru_h", blk.gru_h), ("mlp_in", blk.mlp_in), ("mlp_out", blk.mlp_out)]:
            tensors[f"{prefix}.{name}.weight"] = mat.weight
            tensors[f"{prefix}.{name}.weight_unpacked"] = mat.weight_unpacked
            
        # Biases
        if blk.gru_b1 is not None: tensors[f"{prefix}.gru_b1"] = blk.gru_b1
        if blk.gru_b2 is not None: tensors[f"{prefix}.gru_b2"] = blk.gru_b2
        if blk.mlp_b1 is not None: tensors[f"{prefix}.mlp_b1"] = blk.mlp_b1
        if blk.mlp_b2 is not None: tensors[f"{prefix}.mlp_b2"] = blk.mlp_b2

    mx.save_safetensors(path, tensors)
    print("Checkpoint saved.")

def load_checkpoint(path: str, cfg: NoiseConfig, vocab: int, seq_len: int, d_model: int, d_hidden: int, n_layers: int) -> ModelWeights:
    print(f"Loading checkpoint from {path}...")
    tensors = mx.load_safetensors(path)
    
    tok_emb = tensors["tok_emb"]
    pos_emb = tensors["pos_emb"]
    
    # Head
    head = Int8Mat(
        weight=tensors["head.weight"],
        weight_unpacked=tensors["head.weight_unpacked"],
        scale=None,
        bias=None
    )
    head_bias = tensors.get("head_bias")
    
    ln_out_scale = tensors["ln_out_scale"]
    ln_out_bias = tensors["ln_out_bias"]
    
    blocks = []
    for i in range(n_layers):
        prefix = f"blocks.{i}"
        
        ln1_scale = tensors[f"{prefix}.ln1_scale"]
        ln1_bias = tensors[f"{prefix}.ln1_bias"]
        ln2_scale = tensors[f"{prefix}.ln2_scale"]
        ln2_bias = tensors[f"{prefix}.ln2_bias"]
        
        def load_mat(name):
            return Int8Mat(
                weight=tensors[f"{prefix}.{name}.weight"],
                weight_unpacked=tensors[f"{prefix}.{name}.weight_unpacked"],
                scale=None,
                bias=None
            )
            
        gru_x = load_mat("gru_x")
        gru_h = load_mat("gru_h")
        mlp_in = load_mat("mlp_in")
        mlp_out = load_mat("mlp_out")
        
        gru_b1 = tensors.get(f"{prefix}.gru_b1")
        gru_b2 = tensors.get(f"{prefix}.gru_b2")
        mlp_b1 = tensors.get(f"{prefix}.mlp_b1")
        mlp_b2 = tensors.get(f"{prefix}.mlp_b2")
        
        blocks.append(BlockWeights(
            ln1_scale, ln1_bias, ln2_scale, ln2_bias,
            gru_x, gru_h, mlp_in, mlp_out,
            gru_b1, gru_b2, mlp_b1, mlp_b2
        ))
        
    return ModelWeights(tok_emb, pos_emb, blocks, ln_out_scale, ln_out_bias, head, head_bias)

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
            
            # if ctx.epoch == 0 and seed_offset == 1: # Print once for first block/step
            #      print(f"DEBUG: matmul_with_noise: delta_mean={mx.mean(mx.abs(delta)).item():.4f}, scale={scale_factor:.6f}, max_delta={mx.max(mx.abs(delta)).item()}")
            
            out = base_out + (delta * scale_factor).astype(mx.float16)
            # out = base_out # Disable noise for debugging
            
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



def forward_block(
    ctx: EggrollContext,
    blk: BlockWeights,
    x_in: mx.array,  # (P,B,D) float16
    h_state: mx.array,  # (P,B,H) float16
    tid_base: Optional[int],
    seed_offset: int,
    noise: Optional[Tuple] = None,
) -> mx.array:
    # Unpack noise if present
    n_gx, n_gh, n_mi, n_mo = (None, None, None, None)
    if noise is not None:
        n_gx, n_gh, n_mi, n_mo = noise

    # Layernorm on input (use float32 to avoid overflow from large activations)
    x_ln = mx.fast.layer_norm(x_in.astype(mx.float32), blk.ln1_scale.astype(mx.float32), blk.ln1_bias.astype(mx.float32), eps=1e-5).astype(mx.float16)
    
    gx = matmul_with_noise(ctx, blk.gru_x, x_ln, tid_base if isinstance(tid_base, list) else (None if tid_base is None else tid_base), seed_offset=seed_offset + 1, noise=n_gx)
    gh_tid = [t + 2 for t in tid_base] if isinstance(tid_base, list) else (None if tid_base is None else tid_base + 2)
    gh = matmul_with_noise(ctx, blk.gru_h, h_state, gh_tid, seed_offset=seed_offset + 2, noise=n_gh)
    
    H3 = gx.shape[-1]
    H = H3 // 3
    gx_split = mx.reshape(gx, (x_in.shape[0], x_in.shape[1], 3, H))
    gh_split = mx.reshape(gh, (x_in.shape[0], x_in.shape[1], 3, H))

    if blk.gru_b1 is not None:
        gx_split = gx_split + blk.gru_b1.reshape(1, 1, 3, H)
    if blk.gru_b2 is not None:
        gh_split = gh_split + blk.gru_b2.reshape(1, 1, 3, H)

    r = mx.sigmoid(gx_split[:, :, 0, :] + gh_split[:, :, 0, :])
    z = mx.sigmoid(gx_split[:, :, 1, :] + gh_split[:, :, 1, :])
    n = mx.tanh(gx_split[:, :, 2, :] + r * gh_split[:, :, 2, :])
    h_new = (1 - z) * n + z * h_state
    h_res = h_new + x_in

    # MLP
    h_ln2 = mx.fast.layer_norm(h_res.astype(mx.float32), blk.ln2_scale.astype(mx.float32), blk.ln2_bias.astype(mx.float32), eps=1e-5).astype(mx.float16)
    
    mlp_hidden_tid = [t + 4 for t in tid_base] if isinstance(tid_base, list) else (None if tid_base is None else tid_base + 4)
    mlp_hidden = matmul_with_noise(ctx, blk.mlp_in, h_ln2, mlp_hidden_tid, seed_offset=seed_offset + 3, noise=n_mi)
    mlp_hidden = mx.maximum(mlp_hidden, 0) # ReLU
    if blk.mlp_b1 is not None:
        mlp_hidden = mlp_hidden + blk.mlp_b1
        
    mlp_out_tid = [t + 6 for t in tid_base] if isinstance(tid_base, list) else (None if tid_base is None else tid_base + 6)
    mlp_out = matmul_with_noise(ctx, blk.mlp_out, mlp_hidden, mlp_out_tid, seed_offset=seed_offset + 4, noise=n_mo)
    if blk.mlp_b2 is not None:
        mlp_out = mlp_out + blk.mlp_b2
    h_out = mlp_out + h_res
    return h_out


def gather_noise_for_block(ctx, blk, thread_ids, base_offset):
    # offsets: gru_x=1, gru_h=2, mlp_in=3, mlp_out=4
    from eggroll_mlx import get_lora_update_params

    tids = list(thread_ids) if isinstance(thread_ids, mx.array) else thread_ids
    n_gx = get_lora_update_params(ctx.big_rand, ctx.cfg, ctx.epoch, tids, blk.gru_x.weight_unpacked.shape, ctx.base_seed + base_offset + 1)
    n_gh = get_lora_update_params(ctx.big_rand, ctx.cfg, ctx.epoch, [t + 2 for t in tids], blk.gru_h.weight_unpacked.shape, ctx.base_seed + base_offset + 2)
    n_mi = get_lora_update_params(ctx.big_rand, ctx.cfg, ctx.epoch, [t + 4 for t in tids], blk.mlp_in.weight_unpacked.shape, ctx.base_seed + base_offset + 3)
    n_mo = get_lora_update_params(ctx.big_rand, ctx.cfg, ctx.epoch, [t + 6 for t in tids], blk.mlp_out.weight_unpacked.shape, ctx.base_seed + base_offset + 4)
    return (n_gx, n_gh, n_mi, n_mo)


def forward_model(
    ctx: EggrollContext,
    w: ModelWeights,
    x_tokens: mx.array,
    thread_ids: List[int] | mx.array,
    h_states: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    # Vectorized over population dimension.
    # x_tokens: (B,S), thread_ids: (P,), h_states: (P, B, H) or None.
    # Returns logits_float (P,B,V) and new_states (P,B,H).
    
    # Pre-gather noise if thread_ids is list (population mode)
    block_noises = []
    head_noise = None
    if isinstance(thread_ids, list) or (isinstance(thread_ids, mx.array) and thread_ids.ndim == 1):
        tids = list(thread_ids) if isinstance(thread_ids, mx.array) else thread_ids
        for blk_idx, blk in enumerate(w.blocks):
            base = blk_idx * 100
            block_noises.append(gather_noise_for_block(ctx, blk, tids, base))
        
        # Head noise (offset 0)
        from eggroll_mlx import get_lora_update_params
        head_noise = get_lora_update_params(ctx.big_rand, ctx.cfg, ctx.epoch, tids, w.head.weight_unpacked.shape, ctx.base_seed + 0)

    B, S = x_tokens.shape
    P = len(thread_ids) if isinstance(thread_ids, list) else (thread_ids.shape[0] if isinstance(thread_ids, mx.array) else 1)
    
    tok = w.tok_emb[x_tokens]  # (B,S,D)
    pos = w.pos_emb[:S]
    x_emb = (tok + pos).astype(mx.float16)  # (B,S,D)

    hidden_dim = w.blocks[0].gru_h.weight_unpacked.shape[1]
    num_layers = len(w.blocks)

    if h_states is None:
        states_list = [mx.zeros((P, B, hidden_dim), dtype=mx.float16) for _ in range(num_layers)]
    else:
        if h_states.ndim == 3:
            h0 = h_states
            zeros = [mx.zeros((P, B, hidden_dim), dtype=h0.dtype) for _ in range(num_layers - 1)]
            states_list = [h0] + zeros
        else:
            states_list = [h_states[i] for i in range(num_layers)]

    for t in range(S):
        x_shared = x_emb[:, t, :]  # (B,D)
        x_t = mx.broadcast_to(x_shared[None, ...], (P, B, x_shared.shape[-1]))  # (P,B,D)
        for blk_idx, blk in enumerate(w.blocks):
            tid_base = [tid + blk_idx * 10 for tid in thread_ids] if isinstance(thread_ids, list) else (thread_ids + blk_idx * 10 if isinstance(thread_ids, mx.array) else thread_ids)
            h_state = states_list[blk_idx]

            # Use pre-gathered noise if available
            noise = block_noises[blk_idx] if block_noises else None
            h_new = forward_block(ctx, blk, x_t, h_state, tid_base, seed_offset=blk_idx * 100, noise=noise)
            states_list[blk_idx] = h_new
            x_t = h_new
        
    # Head
    last_ln = mx.fast.layer_norm(x_t, w.ln_out_scale, w.ln_out_bias, eps=1e-5)
    logits = matmul_with_noise(ctx, w.head, last_ln, thread_ids, seed_offset=0, noise=head_noise)
    if w.head_bias is not None:
        logits = logits + w.head_bias

    return logits.astype(mx.float32), mx.stack(states_list, axis=0)


def main():
    print("DEBUG: Starting main...")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/tinystories")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--vocab_size", type=int, default=10000)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--d_hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--pop_size", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--fixed_point", type=int, default=4)
    ap.add_argument("--sigma_shift", type=int, default=4)
    ap.add_argument("--init_scale", type=float, default=16.0)
    ap.add_argument("--fast_fitness", type=int, default=1)
    ap.add_argument("--fitness_alpha", type=float, default=0.01, help="scale factor for CLT fitness normalization")
    ap.add_argument("--update_threshold", type=int, default=32, help="initial update threshold; set >0 to gate small votes")
    ap.add_argument("--update_threshold_final", type=int, default=None, help="optional final threshold for linear decay")
    ap.add_argument("--group_size", type=int, default=None, help="reuse the same batch across this many population members")
    ap.add_argument("--noise_reuse", type=int, default=1, help="reuse noise every N epochs (matches reference get_common_start_idx)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--carry_state", action="store_true", help="carry pop hidden state across batches")
    ap.add_argument("--save_checkpoint", type=str, default=None, help="path to save checkpoint")
    ap.add_argument("--load_checkpoint", type=str, default=None, help="path to load checkpoint")
    ap.add_argument("--prompt", type=str, default=None, help="prompt for generation (skips training if set)")
    ap.add_argument("--save_every", type=int, default=100, help="save checkpoint every N steps")
    ap.add_argument("--debug_perturbations", action="store_true", help="enable verbose logging of perturbation stats")
    ap.add_argument("--learning_rate", type=float, default=0.01, help="step size multiplier for sign updates")
    ap.add_argument("--weight_clip", type=float, default=5.0, help="clip weights to [-C, C] after update")
    args = ap.parse_args()

    if args.pop_size % 2 != 0:
        raise ValueError("pop_size must be even for paired fitnesses.")
    
    if args.group_size is None:
        args.group_size = args.pop_size

    print("DEBUG: Loading tokens...")
    rng = np.random.default_rng(args.seed)
    train_path = Path(args.data_dir) / "train_tokens.npy"
    if not train_path.exists():
        train_path = Path(args.data_dir) / "train_tokens.uint16.memmap"
    memmap = load_tokens(train_path)
    print(f"DEBUG: Loaded tokens from {train_path}, shape={memmap.shape}")

    cfg = NoiseConfig(
        fixed_point=args.fixed_point,
        sigma_shift=args.sigma_shift,
        rank=1,
        fast_fitness=bool(args.fast_fitness),
        fitness_alpha=args.fitness_alpha,
        update_threshold=args.update_threshold,
        noise_reuse=args.noise_reuse,
        debug_perturbations=args.debug_perturbations,
        learning_rate=args.learning_rate,
        weight_clip=args.weight_clip,
    )
    param_span = (args.d_model + 3 * args.d_hidden + 4 * args.d_hidden + args.vocab_size) * cfg.rank * args.pop_size * 2
    print(f"DEBUG: Creating context with param_span={param_span}...")
    ctx = make_context(cfg, param_span=param_span, seed=args.seed, safety_margin=4096)
    print(f"DEBUG: Context created. big_rand stats: mean={mx.mean(ctx.big_rand.astype(mx.float32)).item():.4f}, max={mx.max(ctx.big_rand).item()}, min={mx.min(ctx.big_rand).item()}, nonzeros={mx.sum(ctx.big_rand != 0).item()}")
    
    if args.load_checkpoint:
        print(f"Loading checkpoint from {args.load_checkpoint}...")
        weights = load_checkpoint(args.load_checkpoint, cfg, args.vocab_size, args.seq_len, args.d_model, args.d_hidden, args.layers)
    else:
        weights = init_model(cfg, args.vocab_size, args.seq_len, args.d_model, args.d_hidden, args.layers, rng, args.init_scale)

    if args.prompt:
        # Inference mode
        print(f"Generating from prompt: '{args.prompt}'")
        
        # Simple tokenizer fallback
        class AsciiTokenizer:
            def encode(self, text):
                return [ord(c) for c in text]
            def decode(self, ids):
                return "".join([chr(i) if 0 <= i < 128 else "?" for i in ids])

        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.data_dir)
        except Exception as e:
            print(f"Tokenizer load failed ({e}), using simple ASCII tokenizer...")
            tokenizer = AsciiTokenizer()

        input_ids = tokenizer.encode(args.prompt)
        output_ids = generate(ctx, weights, input_ids, max_new_tokens=50)
        text = tokenizer.decode(output_ids)
        print(f"Generated: {text}")
        return

    thread_ids = list(range(args.pop_size))
    thread_ids_for_update = [i * 2 for i in range(args.pop_size // 2)]
    pop_states = None

    import time
    
    for step in range(args.steps):
        t0 = time.time()
        
        # threshold schedule (linear decay if final provided)
        if args.update_threshold_final is not None:
            t0_thresh = args.update_threshold
            t1_thresh = args.update_threshold_final
            frac = min(1.0, step / max(1, args.steps - 1))
            cfg.update_threshold = int(round(t0_thresh + (t1_thresh - t0_thresh) * frac))
        ctx.epoch = step
        
        t_batch_start = time.time()
        # group size reuse: same batch for each group; vectorized over pop
        rewards_chunks = []
        state_chunks = []
        
        # Get batch ONCE for the whole population
        x, y = get_batch(memmap, args.seq_len, batch_size=args.batch_size, rng=rng)
        t_batch_end = time.time()
        
        t_forward_start = time.time()
        num_chunks = 0
        for g_start in range(0, args.pop_size, args.group_size):
            num_chunks += 1
            g_tids = thread_ids[g_start : g_start + args.group_size]
            states_slice = None
            if args.carry_state and pop_states is not None:
                states_slice = pop_states[:, g_start : g_start + args.group_size]
            
            # Forward
            logits, states_out = forward_model(ctx, weights, x, g_tids, states_slice)
            
            # Compute rewards immediately and discard logits
            # Broadcast targets: (Batch,) -> (Group, Batch)
            y_broad = mx.broadcast_to(y[None, ...], (logits.shape[0], y.shape[0]))
            # Loss per example: (Group, Batch)
            loss = cross_entropy(logits, y_broad, reduction='none')
            # Mean over batch -> (Group,)
            chunk_rewards = -mx.mean(loss, axis=1)
            
            rewards_chunks.append(chunk_rewards)
            mx.eval(chunk_rewards) # Ensure logits are freed
            
            if args.carry_state:
                state_chunks.append(states_out)
        t_forward_end = time.time()
                
        t_concat_start = time.time()
        rewards = mx.concatenate(rewards_chunks, axis=0).reshape(-1)
        if args.carry_state:
            pop_states = mx.concatenate(state_chunks, axis=1)
        fitnesses = convert_fitnesses(cfg, rewards)
        mx.eval(fitnesses) # Ensure fitnesses are ready for update
        t_concat_end = time.time()

        # Update weights
        t_update_start = time.time()
        total_updates = 0
        total_params = 0
        
        def update_vec(arr, seed_offset):
            updated, diff = apply_sign_update(cfg, arr.reshape(-1, 1), fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=seed_offset)
            return updated.reshape(arr.shape), diff

        def update_mat(mat: Int8Mat, seed_offset):
            # Update float weights
            updated_float, diff = apply_sign_update(cfg, mat.weight_unpacked, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=seed_offset)
            mat.weight_unpacked = updated_float
            mat.weight = updated_float
            return mat, diff

        weights.head.weight_unpacked, diff = apply_sign_update(cfg, weights.head.weight_unpacked, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=0)
        weights.head.weight = weights.head.weight_unpacked
        total_updates += diff
        total_params += weights.head.weight_unpacked.size
        
        for blk_idx, blk in enumerate(weights.blocks):
            block_base = blk_idx * 100

            blk.gru_x, diff = update_mat(blk.gru_x, seed_offset=block_base + 1)
            total_updates += diff
            total_params += blk.gru_x.weight_unpacked.size
            
            blk.gru_h, diff = update_mat(blk.gru_h, seed_offset=block_base + 2)
            total_updates += diff
            total_params += blk.gru_h.weight_unpacked.size
            
            blk.mlp_in, diff = update_mat(blk.mlp_in, seed_offset=block_base + 3)
            total_updates += diff
            total_params += blk.mlp_in.weight_unpacked.size
            
            blk.mlp_out, diff = update_mat(blk.mlp_out, seed_offset=block_base + 4)
            total_updates += diff
            total_params += blk.mlp_out.weight_unpacked.size

            if blk.gru_b1 is not None:
                blk.gru_b1, diff = update_vec(blk.gru_b1, seed_offset=block_base + 5)
                total_updates += diff
                total_params += blk.gru_b1.size
            if blk.gru_b2 is not None:
                blk.gru_b2, diff = update_vec(blk.gru_b2, seed_offset=block_base + 6)
                total_updates += diff
                total_params += blk.gru_b2.size
            if blk.mlp_b1 is not None:
                blk.mlp_b1, diff = update_vec(blk.mlp_b1, seed_offset=block_base + 7)
                total_updates += diff
                total_params += blk.mlp_b1.size
            if blk.mlp_b2 is not None:
                blk.mlp_b2, diff = update_vec(blk.mlp_b2, seed_offset=block_base + 7)
                total_updates += diff
                total_params += blk.mlp_b2.size
            
            blk.ln1_scale, diff_scale = update_vec(blk.ln1_scale, seed_offset=block_base + 8)
            total_updates += diff_scale
            total_params += blk.ln1_scale.size
            blk.ln1_bias, diff_bias = update_vec(blk.ln1_bias, seed_offset=block_base + 8 + 1000)
            total_updates += diff_bias
            total_params += blk.ln1_bias.size

            blk.ln2_scale, diff_scale = update_vec(blk.ln2_scale, seed_offset=block_base + 9)
            total_updates += diff_scale
            total_params += blk.ln2_scale.size
            blk.ln2_bias, diff_bias = update_vec(blk.ln2_bias, seed_offset=block_base + 9 + 1000)
            total_updates += diff_bias
            total_params += blk.ln2_bias.size

        if weights.head_bias is not None:
            weights.head_bias, diff = update_vec(weights.head_bias, seed_offset=1000)
            total_updates += diff
            total_params += weights.head_bias.size
        
        weights.ln_out_scale, diff_scale = update_vec(weights.ln_out_scale, seed_offset=1001)
        total_updates += diff_scale
        total_params += weights.ln_out_scale.size
        weights.ln_out_bias, diff_bias = update_vec(weights.ln_out_bias, seed_offset=1001 + 1000)
        total_updates += diff_bias
        total_params += weights.ln_out_bias.size

        weights.tok_emb, diff = apply_sign_update(cfg, weights.tok_emb, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=1002)
        total_updates += diff
        total_params += weights.tok_emb.size
        weights.pos_emb, diff = apply_sign_update(cfg, weights.pos_emb, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=1003)
        total_updates += diff
        total_params += weights.pos_emb.size
        
        # Force update completion
        mx.eval(total_updates)
        
        t_update_end = time.time()
        
        # Stats
        update_rate = (total_updates.item() / total_params) * 100
        
        # Logging
        t_log_start = time.time()
        # Single thread eval for logging
        logits_eval, _ = forward_model(ctx, weights, x, thread_ids=mx.array([0], dtype=mx.int32), h_states=None) 
        loss = cross_entropy(logits_eval, y)
        lmax = float(np.array(logits_eval).max())
        
        print(f"step {step:04d} | loss {loss.item():.4f} | reward {mx.mean(rewards).item():.4f} | update_rate {update_rate:.1f}% | "
              f"fwd {t_forward_end - t_forward_start:.3f}s | "
              f"upd {t_update_end - t_update_start:.3f}s")
        r_mean = float(mx.mean(rewards).item())
        r_std = float(mx.std(rewards).item())
        
        t_log_end = time.time()
        
        print(f"step {step}: loss={loss.item():.4f}, logits_max={lmax:.1f}, reward_mean={r_mean:.3f}, reward_std={r_std:.3f}")
        print(f"  reward_range=[{mx.min(rewards).item():.3f}, {mx.max(rewards).item():.3f}], update_rate={update_rate:.4f}% ({total_updates.item()}/{total_params})")
        print(f"  update_threshold={cfg.update_threshold}")
        
        # Timing stats
        print(f"  TIMING: Batch={t_batch_end-t_batch_start:.3f}s, Forward={t_forward_end-t_forward_start:.3f}s ({num_chunks} chunks, {(t_forward_end-t_forward_start)/num_chunks:.3f}s/chunk), Concat={t_concat_end-t_concat_start:.3f}s, Update={t_update_end-t_update_start:.3f}s, Log={t_log_end-t_log_start:.3f}s, Total={time.time()-t0:.3f}s")

        if args.save_checkpoint and (step + 1) % args.save_every == 0:
            save_checkpoint(weights, args.save_checkpoint)
            
    if args.save_checkpoint:
        save_checkpoint(weights, args.save_checkpoint)


if __name__ == "__main__":
    main()
