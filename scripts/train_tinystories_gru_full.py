"""
Closer-to-reference TinyStories trainer:
- Multi-layer GRU + MLP blocks with layernorm and residuals (float LN).
- All linear weights are int8 with low-rank perturbations (do_mm).
- Per-layer seed offsets for noise; distinct thread_id offsets per matmul.
Note: LN and MLP are float-heavy to keep code simple; fixed-point LN like the C ref is not implemented.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import mlx.core as mx

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
from scripts.train_tinystories_gru_int8 import load_tokens, quantize, cross_entropy, get_batch




@dataclass
class Int8Mat:
    weight: mx.array
    scale: int
    bias: Optional[mx.array] = None


@dataclass
class BlockWeights:
    ln1: mx.array  # layernorm scale (int8)
    ln2: mx.array
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
    ln_out: mx.array
    head: Int8Mat
    head_bias: Optional[mx.array] = None


def layernorm(x: mx.array, scale: mx.array, eps: float = 1e-5) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * mx.rsqrt(var + eps) * scale


def l1_layernorm_int8(x_int8: mx.array, ln_scale: mx.array, fixed_point: int) -> mx.array:
    """
    L1 LayerNorm matching reference EGG_LN:
    - Normalizes by mean(|x|) instead of std(x).
    - x_int8 is treated as having `fixed_point` fractional bits.
    - Output is rescaled to int8 range.
    """
    # Reference: abs_sum = (sum(|x|) / size)
    # We do this in float for simplicity, but it matches the logic.
    x_f = x_int8.astype(mx.float32)
    mean_abs = mx.mean(mx.abs(x_f), axis=-1, keepdims=True)
    
    # Avoid division by zero
    denom = mx.maximum(mean_abs, 1e-5)
    
    # Normalize and scale
    # Reference: numerator = (x * weight)
    # result = numerator / abs_sum
    # Here we do: (x / mean_abs) * scale
    out = (x_f / denom) * ln_scale
    
    # Clip and cast
    out_int = mx.clip(mx.round(out), -127, 127).astype(mx.int8)
    return out_int


def init_model(cfg: NoiseConfig, vocab: int, seq_len: int, d_model: int, d_hidden: int, n_layers: int, rng, init_scale: float) -> ModelWeights:
    def q(shape):
        return quantize(rng.normal(0, 0.05, size=shape), cfg.fixed_point, scale=init_scale)

    tok_emb = q((vocab, d_model))
    pos_emb = q((seq_len, d_model))
    blocks: List[BlockWeights] = []
    for _ in range(n_layers):
        # Reference EGG_LN init: ones * 2^fixed_point
        # But our quantize/matmul handles fixed point scaling implicitly in some places.
        # Let's stick to the previous convention: ln scale is just int8 values.
        # If fixed_point=4, 1.0 is represented as 16.
        scale_val = int(2 ** cfg.fixed_point)
        ln1 = mx.full((d_model,), scale_val, dtype=mx.int8)
        ln2 = mx.full((d_model,), scale_val, dtype=mx.int8)
        gru_x = q((3 * d_hidden, d_model))
        gru_h = q((3 * d_hidden, d_hidden))
        mlp_in = q((4 * d_hidden, d_hidden))
        mlp_out = q((d_hidden, 4 * d_hidden))
        gru_b1 = mx.zeros((3 * d_hidden,), dtype=mx.int8)
        gru_b2 = mx.zeros((3 * d_hidden,), dtype=mx.int8)
        mlp_b1 = mx.zeros((4 * d_hidden,), dtype=mx.int8)
        mlp_b2 = mx.zeros((d_hidden,), dtype=mx.int8)
        blocks.append(
            BlockWeights(
                ln1=ln1,
                ln2=ln2,
                gru_x=Int8Mat(gru_x, scale=1, bias=None),
                gru_h=Int8Mat(gru_h, scale=1, bias=None),
                mlp_in=Int8Mat(mlp_in, scale=1, bias=None),
                mlp_out=Int8Mat(mlp_out, scale=1, bias=None),
                gru_b1=gru_b1,
                gru_b2=gru_b2,
                mlp_b1=mlp_b1,
                mlp_b2=mlp_b2,
            )
        )
    ln_out = mx.full((d_model,), int(2 ** cfg.fixed_point), dtype=mx.int8)
    head = Int8Mat(q((vocab, d_model)), scale=1, bias=None)
    head_bias = mx.zeros((vocab,), dtype=mx.int8)
    return ModelWeights(tok_emb=tok_emb, pos_emb=pos_emb, blocks=blocks, ln_out=ln_out, head=head, head_bias=head_bias)

def save_checkpoint(weights: ModelWeights, path: str):
    flat = {}
    flat["tok_emb"] = weights.tok_emb
    flat["pos_emb"] = weights.pos_emb
    flat["ln_out"] = weights.ln_out
    flat["head.weight"] = weights.head.weight
    flat["head.scale"] = mx.array(weights.head.scale)
    if weights.head_bias is not None:
        flat["head_bias"] = weights.head_bias
    
    for i, blk in enumerate(weights.blocks):
        prefix = f"blocks.{i}"
        flat[f"{prefix}.ln1"] = blk.ln1
        flat[f"{prefix}.ln2"] = blk.ln2
        
        for name, mat in [("gru_x", blk.gru_x), ("gru_h", blk.gru_h), ("mlp_in", blk.mlp_in), ("mlp_out", blk.mlp_out)]:
            flat[f"{prefix}.{name}.weight"] = mat.weight
            flat[f"{prefix}.{name}.scale"] = mx.array(mat.scale)
            
        for name, vec in [("gru_b1", blk.gru_b1), ("gru_b2", blk.gru_b2), ("mlp_b1", blk.mlp_b1), ("mlp_b2", blk.mlp_b2)]:
            if vec is not None:
                flat[f"{prefix}.{name}"] = vec
                
    mx.savez(path, **flat)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(path: str, cfg: NoiseConfig, vocab: int, seq_len: int, d_model: int, d_hidden: int, n_layers: int) -> ModelWeights:
    data = mx.load(path)
    
    tok_emb = data["tok_emb"]
    pos_emb = data["pos_emb"]
    ln_out = data["ln_out"]
    head = Int8Mat(data["head.weight"], int(data["head.scale"].item()), bias=None)
    head_bias = data.get("head_bias")
    
    blocks = []
    for i in range(n_layers):
        prefix = f"blocks.{i}"
        ln1 = data[f"{prefix}.ln1"]
        ln2 = data[f"{prefix}.ln2"]
        
        gru_x = Int8Mat(data[f"{prefix}.gru_x.weight"], int(data[f"{prefix}.gru_x.scale"].item()))
        gru_h = Int8Mat(data[f"{prefix}.gru_h.weight"], int(data[f"{prefix}.gru_h.scale"].item()))
        mlp_in = Int8Mat(data[f"{prefix}.mlp_in.weight"], int(data[f"{prefix}.mlp_in.scale"].item()))
        mlp_out = Int8Mat(data[f"{prefix}.mlp_out.weight"], int(data[f"{prefix}.mlp_out.scale"].item()))
        
        gru_b1 = data.get(f"{prefix}.gru_b1")
        gru_b2 = data.get(f"{prefix}.gru_b2")
        mlp_b1 = data.get(f"{prefix}.mlp_b1")
        mlp_b2 = data.get(f"{prefix}.mlp_b2")
        
        blocks.append(BlockWeights(ln1, ln2, gru_x, gru_h, mlp_in, mlp_out, gru_b1, gru_b2, mlp_b1, mlp_b2))
        
    return ModelWeights(tok_emb, pos_emb, blocks, ln_out, head, head_bias)

def generate(ctx: EggrollContext, weights: ModelWeights, prompt: List[int], max_new_tokens: int) -> List[int]:
    # Simple greedy generation
    # We reuse forward_model but with thread_ids=None (handled by matmul_with_noise)
    # But forward_model expects thread_ids list/array.
    # Let's make a minimal inference loop.
    
    curr_tokens = mx.array(prompt)[None, :] # (1, S)
    
    for _ in range(max_new_tokens):
        # Forward pass
        # We can pass thread_ids=[0] but we want NO noise.
        # matmul_with_noise: if tid is None -> clean int8 matmul.
        # forward_model: expects thread_ids list.
        # Let's hack forward_model to accept None for thread_ids to mean "inference mode"
        # Or just call forward_block manually.
        
        # Actually, forward_model is vectorized. Let's write a clean single-sequence forward.
        B, S = curr_tokens.shape
        h = mx.zeros((1, B, weights.blocks[0].gru_h.weight.shape[1]), dtype=mx.int8) # (P=1, B, H)
        
        tok = weights.tok_emb[curr_tokens]
        pos = weights.pos_emb[:S]
        x_emb = mx.clip(tok + pos, -127, 127).astype(mx.int8)
        
        # Run full sequence (inefficient but simple for now)
        # Ideally we'd cache state, but GRU state is small.
        
        states = h
        for t in range(S):
            x_t = x_emb[:, t, :] # (B, D)
            x_t = x_t[None, ...] # (P=1, B, D)
            
            for blk in weights.blocks:
                # tid_base=None -> clean matmul
                states = forward_block(ctx, blk, x_t if blk == weights.blocks[0] else states, states, tid_base=None, seed_offset=0)
                x_t = states
                
        last_ln = l1_layernorm_int8(states, weights.ln_out, ctx.cfg.fixed_point)
        logits = matmul_with_noise(ctx, weights.head, last_ln, tid=None, seed_offset=0)
        if weights.head_bias is not None:
            logits = logits + weights.head_bias
            
        next_token = int(mx.argmax(logits[0, 0, :]).item())
        curr_tokens = mx.concatenate([curr_tokens, mx.array([[next_token]])], axis=1)
        
    return curr_tokens[0].tolist()


def matmul_with_noise(ctx: EggrollContext, mat: Int8Mat, x: mx.array, tid: Optional[int], seed_offset: int = 0) -> mx.array:
    if tid is not None:
        # vectorized thread ids -> use batched matmul
        if isinstance(tid, mx.array):
            tids_list = list(np.array(tid))
            out = do_mm_batched(ctx.cfg, x, mat.weight, ctx.big_rand, ctx.epoch, tids_list, ctx.base_seed + seed_offset)
        elif isinstance(tid, list):
            out = do_mm_batched(ctx.cfg, x, mat.weight, ctx.big_rand, ctx.epoch, tid, ctx.base_seed + seed_offset)
        else:
            tid_int = int(tid)
            out = do_mm(ctx.cfg, x, mat.weight, ctx.big_rand, ctx.epoch, tid_int, ctx.base_seed + seed_offset)
    else:
        # match do_mm scaling: divide by (2^fixed_point * sqrt(k)) then mat.scale
        # Handle 3D input (P, B, D) for inference
        orig_shape = x.shape
        if x.ndim == 3:
            x_flat = mx.reshape(x, (-1, x.shape[-1]))
            base = int8_matmul(x_flat, mat.weight)
            base = mx.reshape(base, (orig_shape[0], orig_shape[1], -1))
        else:
            base = int8_matmul(x, mat.weight)  # int32
            
        denom = (2 ** ctx.cfg.fixed_point) * max(1, mat.weight.shape[1])
        out = (base // denom) // mat.scale
        out = mx.clip(out, -127, 127).astype(mx.int8)
        return out
    # do_mm/do_mm_batched already apply fixed_point/sqrt scaling and return int8
    if mat.scale != 1:
        out = (out // mat.scale).astype(mx.int8)
    if mat.bias is not None:
        out = out + mat.bias
    return mx.clip(out, -127, 127).astype(mx.int8)


def forward_block(
    ctx: EggrollContext,
    blk: BlockWeights,
    x_in: mx.array,  # (P,B,D)
    h_state: mx.array,  # (P,B,H)
    tid_base: Optional[int],
    seed_offset: int,
) -> mx.array:
    # Layernorm on input
    x_ln = l1_layernorm_int8(x_in, blk.ln1, ctx.cfg.fixed_point)
    gx = matmul_with_noise(ctx, blk.gru_x, x_ln, tid_base if isinstance(tid_base, list) else (None if tid_base is None else tid_base), seed_offset=seed_offset + 1)
    gh_tid = [t + 2 for t in tid_base] if isinstance(tid_base, list) else (None if tid_base is None else tid_base + 2)
    gh = matmul_with_noise(ctx, blk.gru_h, h_state, gh_tid, seed_offset=seed_offset + 2)
    gates = (gx.astype(mx.int32) + gh.astype(mx.int32)).astype(mx.float32)
    if blk.gru_b1 is not None:
        gates = gates + blk.gru_b1.astype(mx.float32)
    H3 = gx.shape[-1]
    H = H3 // 3
    gates = mx.reshape(gates, (x_in.shape[0], x_in.shape[1], 3, H))
    r = mx.sigmoid(gates[:, :, 0, :])
    z = mx.sigmoid(gates[:, :, 1, :])
    n = mx.tanh(gates[:, :, 2, :] + r * gates[:, :, 0, :])
    h_new = mx.round((1 - z) * n + z * h_state.astype(mx.float32)).astype(mx.int8)
    h_res = mx.clip(h_new + x_in, -127, 127)

    # MLP
    h_ln2 = l1_layernorm_int8(h_res, blk.ln2, ctx.cfg.fixed_point)
    mlp_hidden_tid = [t + 4 for t in tid_base] if isinstance(tid_base, list) else (None if tid_base is None else tid_base + 4)
    mlp_hidden = matmul_with_noise(ctx, blk.mlp_in, h_ln2, mlp_hidden_tid, seed_offset=seed_offset + 3)
    mlp_hidden = mx.maximum(mlp_hidden, mx.zeros_like(mlp_hidden))
    if blk.mlp_b1 is not None:
        mlp_hidden = mlp_hidden + blk.mlp_b1
    mlp_out_tid = [t + 6 for t in tid_base] if isinstance(tid_base, list) else (None if tid_base is None else tid_base + 6)
    mlp_out = matmul_with_noise(ctx, blk.mlp_out, mlp_hidden, mlp_out_tid, seed_offset=seed_offset + 4)
    if blk.mlp_b2 is not None:
        mlp_out = mlp_out + blk.mlp_b2
    h_out = mx.clip(mlp_out + h_res, -127, 127)
    return h_out


def forward_model(
    ctx: EggrollContext,
    w: ModelWeights,
    x_tokens: mx.array,
    thread_ids: List[int] | mx.array,
    h_states: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Vectorized over population dimension.
    x_tokens: (B,S), thread_ids: (P,), h_states: (P, B, H) or None.
    Returns logits_float (P,B,V) and new_states (P,B,H).
    """
    B, S = x_tokens.shape
    if isinstance(thread_ids, mx.array):
        thread_ids = list(map(int, np.array(thread_ids)))
    P = len(thread_ids)
    H = w.blocks[0].gru_h.weight.shape[1]
    if h_states is None:
        h_states = mx.zeros((P, B, H), dtype=mx.int8)

    tok = w.tok_emb[x_tokens]  # (B,S,D)
    pos = w.pos_emb[:S]
    x_emb = mx.clip(tok + pos, -127, 127).astype(mx.int8)

    states = h_states
    for t in range(S):
        x_shared = x_emb[:, t, :]  # (B,D)
        x_t = mx.broadcast_to(x_shared[None, ...], (P, B, x_shared.shape[-1]))  # (P,B,D)
        for blk_idx, blk in enumerate(w.blocks):
            tid_base = [tid + blk_idx * 10 for tid in thread_ids]
            inp = x_t if blk_idx == 0 else states
            states = forward_block(ctx, blk, inp, states, tid_base, seed_offset=blk_idx * 100)
            x_t = states
        
        # ES optimization: eval states to free graph memory since we don't need gradients
        mx.eval(states)

    last = states  # (P,B,H)
    last_ln = l1_layernorm_int8(last, w.ln_out, ctx.cfg.fixed_point)
    logits_int = matmul_with_noise(ctx, w.head, last_ln, [tid + 999 for tid in thread_ids], seed_offset=999)
    if w.head_bias is not None:
        logits_int = logits_int + w.head_bias
    return logits_int.astype(mx.float32), states


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
    ap.add_argument("--update_threshold", type=int, default=512, help="initial update threshold; set >0 to gate small votes")
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
    )
    param_span = (args.d_model + 3 * args.d_hidden + 4 * args.d_hidden + args.vocab_size) * cfg.rank * args.pop_size * 2
    print(f"DEBUG: Creating context with param_span={param_span}...")
    ctx = make_context(cfg, param_span=param_span, seed=args.seed, safety_margin=4096)
    print("DEBUG: Context created.")
    
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

    for step in range(args.steps):
        # threshold schedule (linear decay if final provided)
        if args.update_threshold_final is not None:
            t0 = args.update_threshold
            t1 = args.update_threshold_final
            frac = min(1.0, step / max(1, args.steps - 1))
            cfg.update_threshold = int(round(t0 + (t1 - t0) * frac))
        ctx.epoch = step
        # group size reuse: same batch for each group; vectorized over pop
        rewards_chunks = []
        state_chunks = []
        
        # Get batch ONCE for the whole population
        x, y = get_batch(memmap, args.seq_len, batch_size=args.batch_size, rng=rng)
        
        for g_start in range(0, args.pop_size, args.group_size):
            g_tids = thread_ids[g_start : g_start + args.group_size]
            states_slice = None
            if args.carry_state and pop_states is not None:
                states_slice = pop_states[g_start : g_start + args.group_size]
            logits, states_out = forward_model(ctx, weights, x, g_tids, states_slice)
            
            # Compute rewards immediately and discard logits
            chunk_rewards = -mx.vmap(lambda l: cross_entropy(l, y))(logits)
            rewards_chunks.append(chunk_rewards)
            mx.eval(chunk_rewards) # Ensure logits are freed
            
            if args.carry_state:
                state_chunks.append(states_out)
                
        rewards = mx.concatenate(rewards_chunks, axis=0).reshape(-1)
        if args.carry_state:
            pop_states = mx.concatenate(state_chunks, axis=0)
        fitnesses = convert_fitnesses(cfg, rewards)

        # Update weights
        total_updates = 0
        total_params = 0
        
        def update_vec(arr, seed_offset):
            updated, diff = apply_sign_update(cfg, arr.reshape(-1, 1), fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=seed_offset)
            return updated.reshape(arr.shape), diff

        weights.head.weight, diff = apply_sign_update(cfg, weights.head.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=0)
        total_updates += diff
        total_params += weights.head.weight.size
        
        for i, blk in enumerate(weights.blocks):
            # We use different seed offsets for each layer/matrix to ensure independent noise
            # Offsets: 100*i + [0, 1, 2, 3]
            base = 100 * (i + 1)
            blk.gru_x.weight, diff = apply_sign_update(cfg, blk.gru_x.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=base + 0)
            total_updates += diff
            total_params += blk.gru_x.weight.size
            
            blk.gru_h.weight, diff = apply_sign_update(cfg, blk.gru_h.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=base + 1)
            total_updates += diff
            total_params += blk.gru_h.weight.size
            
            blk.mlp_in.weight, diff = apply_sign_update(cfg, blk.mlp_in.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=base + 2)
            total_updates += diff
            total_params += blk.mlp_in.weight.size
            
            blk.mlp_out.weight, diff = apply_sign_update(cfg, blk.mlp_out.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=base + 3)
            total_updates += diff
            total_params += blk.mlp_out.weight.size

            if blk.gru_b1 is not None:
                blk.gru_b1, diff = update_vec(blk.gru_b1, seed_offset=base + 4)
                total_updates += diff
                total_params += blk.gru_b1.size
            if blk.gru_b2 is not None:
                blk.gru_b2, diff = update_vec(blk.gru_b2, seed_offset=base + 5)
                total_updates += diff
                total_params += blk.gru_b2.size
            if blk.mlp_b1 is not None:
                blk.mlp_b1, diff = update_vec(blk.mlp_b1, seed_offset=base + 6)
                total_updates += diff
                total_params += blk.mlp_b1.size
            if blk.mlp_b2 is not None:
                blk.mlp_b2, diff = update_vec(blk.mlp_b2, seed_offset=base + 7)
                total_updates += diff
                total_params += blk.mlp_b2.size
            blk.ln1, diff = update_vec(blk.ln1, seed_offset=base + 8)
            total_updates += diff
            total_params += blk.ln1.size
            blk.ln2, diff = update_vec(blk.ln2, seed_offset=base + 9)
            total_updates += diff
            total_params += blk.ln2.size

        if weights.head_bias is not None:
            weights.head_bias, diff = update_vec(weights.head_bias, seed_offset=1000)
            total_updates += diff
            total_params += weights.head_bias.size
        weights.ln_out, diff = update_vec(weights.ln_out, seed_offset=1001)
        total_updates += diff
        total_params += weights.ln_out.size
        weights.tok_emb, diff = apply_sign_update(cfg, weights.tok_emb, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=1002)
        total_updates += diff
        total_params += weights.tok_emb.size
        weights.pos_emb, diff = apply_sign_update(cfg, weights.pos_emb, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=1003)
        total_updates += diff
        total_params += weights.pos_emb.size

        # Logging
        logits_eval, _ = forward_model(ctx, weights, x, thread_ids=mx.array([0], dtype=mx.int32), h_states=None) # Single thread eval
        loss = cross_entropy(logits_eval, y)
        lmax = float(mx.max(logits_eval).item())
        r_mean = float(mx.mean(rewards).item())
        r_std = float(mx.std(rewards).item())
        r_min = float(mx.min(rewards).item())
        r_max = float(mx.max(rewards).item())
        
        update_rate = (total_updates / total_params) * 100.0
        
        print(f"step {step}: loss={loss.item():.4f}, logits_max={lmax:.1f}, reward_mean={r_mean:.3f}, reward_std={r_std:.3f}")
        print(f"  reward_range=[{r_min:.3f}, {r_max:.3f}], update_rate={update_rate:.4f}% ({total_updates}/{total_params})")
        print(f"  update_threshold={cfg.update_threshold}")
        
        if args.save_checkpoint and (step + 1) % args.save_every == 0:
            save_checkpoint(weights, args.save_checkpoint)
            
    if args.save_checkpoint:
        save_checkpoint(weights, args.save_checkpoint)


if __name__ == "__main__":
    main()
