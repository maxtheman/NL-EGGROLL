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


def fixed_layernorm_int8(x_int8: mx.array, ln_scale: mx.array, fixed_point: int) -> mx.array:
    """
    Approximate fixed-point layernorm:
    - Treat int8 as fixed_point fractional bits.
    - Compute ln in float, then rescale back to int8 with 2^fixed_point.
    """
    scale = float(2 ** fixed_point)
    x_f = x_int8.astype(mx.float32) / scale
    mean = mx.mean(x_f, axis=-1, keepdims=True)
    var = mx.mean((x_f - mean) ** 2, axis=-1, keepdims=True)
    ln = (x_f - mean) * mx.rsqrt(var + 1e-5) * ln_scale
    ln_int = mx.clip(mx.round(ln * scale), -127, 127).astype(mx.int8)
    return ln_int


def init_model(cfg: NoiseConfig, vocab: int, seq_len: int, d_model: int, d_hidden: int, n_layers: int, rng, init_scale: float) -> ModelWeights:
    def q(shape):
        return quantize(rng.normal(0, 0.05, size=shape), cfg.fixed_point, scale=init_scale)

    tok_emb = q((vocab, d_model))
    pos_emb = q((seq_len, d_model))
    blocks: List[BlockWeights] = []
    for _ in range(n_layers):
        ln1 = mx.ones((d_model,), dtype=mx.int8)
        ln2 = mx.ones((d_model,), dtype=mx.int8)
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
    ln_out = mx.ones((d_model,), dtype=mx.int8)
    head = Int8Mat(q((vocab, d_model)), scale=1, bias=None)
    head_bias = mx.zeros((vocab,), dtype=mx.int8)
    return ModelWeights(tok_emb=tok_emb, pos_emb=pos_emb, blocks=blocks, ln_out=ln_out, head=head, head_bias=head_bias)


def matmul_with_noise(ctx: EggrollContext, mat: Int8Mat, x: mx.array, tid: Optional[int], seed_offset: int = 0) -> mx.array:
    if tid is not None:
        # vectorized thread ids -> use batched matmul
        if isinstance(tid, mx.array):
            tids_list = list(np.array(tid))
            out = do_mm_batched(ctx.cfg, x, mat.weight, ctx.big_rand, ctx.epoch, tids_list, ctx.base_seed + seed_offset)
        else:
            tid_int = int(tid)
            out = do_mm(ctx.cfg, x, mat.weight, ctx.big_rand, ctx.epoch, tid_int, ctx.base_seed + seed_offset)
    else:
        # match do_mm scaling: divide by (2^fixed_point * sqrt(k)) then mat.scale
        base = int8_matmul(x, mat.weight)  # int32
        denom = (2 ** ctx.cfg.fixed_point) * max(1, int(np.sqrt(mat.weight.shape[1])))
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
    x_ln = fixed_layernorm_int8(x_in, blk.ln1, ctx.cfg.fixed_point)
    gx = matmul_with_noise(ctx, blk.gru_x, x_ln, None if tid_base is None else tid_base + 0, seed_offset=seed_offset + 1)
    gh = matmul_with_noise(ctx, blk.gru_h, h_state, None if tid_base is None else tid_base + 2, seed_offset=seed_offset + 2)
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
    h_ln2 = fixed_layernorm_int8(h_res, blk.ln2, ctx.cfg.fixed_point)
    mlp_hidden = matmul_with_noise(ctx, blk.mlp_in, h_ln2, None if tid_base is None else tid_base + 4, seed_offset=seed_offset + 3)
    mlp_hidden = mx.maximum(mlp_hidden, mx.zeros_like(mlp_hidden))
    if blk.mlp_b1 is not None:
        mlp_hidden = mlp_hidden + blk.mlp_b1
    mlp_out = matmul_with_noise(ctx, blk.mlp_out, mlp_hidden, None if tid_base is None else tid_base + 6, seed_offset=seed_offset + 4)
    if blk.mlp_b2 is not None:
        mlp_out = mlp_out + blk.mlp_b2
    h_out = mx.clip(mlp_out + h_res, -127, 127)
    return h_out


def forward_model(
    ctx: EggrollContext,
    w: ModelWeights,
    x_tokens: mx.array,
    thread_ids: mx.array,
    h_states: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Vectorized over population dimension.
    x_tokens: (B,S), thread_ids: (P,), h_states: (P, B, H) or None.
    Returns logits_float (P,B,V) and new_states (P,B,H).
    """
    B, S = x_tokens.shape
    P = thread_ids.shape[0]
    H = w.blocks[0].gru_h.weight.shape[1]
    if h_states is None:
        h_states = mx.zeros((P, B, H), dtype=mx.int8)

    tok = w.tok_emb[x_tokens]  # (B,S,D)
    pos = w.pos_emb[:S]
    x_emb = mx.clip(tok + pos, -127, 127).astype(mx.int8)
    x_emb = mx.broadcast_to(x_emb[None, ...], (P, B, S, x_emb.shape[-1]))  # (P,B,S,D)

    states = h_states
    for t in range(S):
        x_t = x_emb[:, :, t, :]  # (P,B,D)
        for blk_idx, blk in enumerate(w.blocks):
            tid_base = thread_ids + blk_idx * 10
            inp = x_t if blk_idx == 0 else states
            states = forward_block(ctx, blk, inp, states, tid_base, seed_offset=blk_idx * 100)
            x_t = states

    last = states  # (P,B,H)
    last_ln = fixed_layernorm_int8(last, w.ln_out, ctx.cfg.fixed_point)
    logits_int = matmul_with_noise(ctx, w.head, last_ln, thread_ids + 999, seed_offset=999)
    if w.head_bias is not None:
        logits_int = logits_int + w.head_bias
    return logits_int.astype(mx.float32), states


def main():
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
    ap.add_argument("--fixed_point", type=int, default=2)
    ap.add_argument("--sigma_shift", type=int, default=4)
    ap.add_argument("--init_scale", type=float, default=12.0)
    ap.add_argument("--fast_fitness", type=int, default=0)
    ap.add_argument("--fitness_alpha", type=float, default=0.01, help="scale factor for CLT fitness normalization")
    ap.add_argument("--update_threshold", type=int, default=0, help="initial update threshold; set >0 to gate small votes")
    ap.add_argument("--update_threshold_final", type=int, default=None, help="optional final threshold for linear decay")
    ap.add_argument("--group_size", type=int, default=1, help="reuse the same batch across this many population members")
    ap.add_argument("--noise_reuse", type=int, default=1, help="reuse noise every N epochs (matches reference get_common_start_idx)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--carry_state", action="store_true", help="carry pop hidden state across batches")
    args = ap.parse_args()

    if args.pop_size % 2 != 0:
        raise ValueError("pop_size must be even for paired fitnesses.")

    rng = np.random.default_rng(args.seed)
    train_path = Path(args.data_dir) / "train_tokens.npy"
    if not train_path.exists():
        train_path = Path(args.data_dir) / "train_tokens.uint16.memmap"
    memmap = load_tokens(train_path)

    cfg = NoiseConfig(
        fixed_point=args.fixed_point,
        sigma_shift=args.sigma_shift,
        rank=1,
        fast_fitness=bool(args.fast_fitness),
        fitness_alpha=args.fitness_alpha,
        update_threshold=args.update_threshold,
        noise_reuse=args.noise_reuse,
    )
    param_span = (args.d_model + 3 * args.d_hidden + 4 * args.d_hidden + args.vocab_size) * cfg.rank * args.pop_size * 2
    ctx = make_context(cfg, param_span=param_span, seed=args.seed, safety_margin=4096)
    weights = init_model(cfg, args.vocab_size, args.seq_len, args.d_model, args.d_hidden, args.layers, rng, args.init_scale)

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
        logits_chunks = []
        state_chunks = []
        for g_start in range(0, args.pop_size, args.group_size):
            g_tids = mx.array(thread_ids[g_start : g_start + args.group_size], dtype=mx.int32)
            x, y = get_batch(memmap, args.seq_len, batch_size=args.batch_size, rng=rng)
            states_slice = None
            if args.carry_state and pop_states is not None:
                states_slice = pop_states[g_start : g_start + args.group_size]
            logits, states_out = forward_model(ctx, weights, x, g_tids, states_slice)
            logits_chunks.append(logits)
            state_chunks.append(states_out)
        logits_pop = mx.concatenate(logits_chunks, axis=0)
        if args.carry_state:
            pop_states = mx.concatenate(state_chunks, axis=0)
        rewards = -mx.vmap(lambda l: cross_entropy(l, y))(logits_pop).reshape(-1)
        fitnesses = convert_fitnesses(cfg, rewards)

        def update_vec(arr):
            updated = apply_sign_update(cfg, arr.reshape(-1, 1), fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)
            return updated.reshape(arr.shape)

        # Update all matrices
        head_before = weights.head.weight
        for blk_idx, blk in enumerate(weights.blocks):
            blk.gru_x.weight = apply_sign_update(cfg, blk.gru_x.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)
            blk.gru_h.weight = apply_sign_update(cfg, blk.gru_h.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)
            blk.mlp_in.weight = apply_sign_update(cfg, blk.mlp_in.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)
            blk.mlp_out.weight = apply_sign_update(cfg, blk.mlp_out.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)
            if blk.gru_b1 is not None:
                blk.gru_b1 = update_vec(blk.gru_b1)
            if blk.gru_b2 is not None:
                blk.gru_b2 = update_vec(blk.gru_b2)
            if blk.mlp_b1 is not None:
                blk.mlp_b1 = update_vec(blk.mlp_b1)
            if blk.mlp_b2 is not None:
                blk.mlp_b2 = update_vec(blk.mlp_b2)
            blk.ln1 = update_vec(blk.ln1)
            blk.ln2 = update_vec(blk.ln2)
        weights.head.weight = apply_sign_update(cfg, weights.head.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)
        if weights.head_bias is not None:
            weights.head_bias = update_vec(weights.head_bias)
        weights.ln_out = update_vec(weights.ln_out)
        weights.tok_emb = apply_sign_update(cfg, weights.tok_emb, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)
        weights.pos_emb = apply_sign_update(cfg, weights.pos_emb, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)

        logits_eval, _ = forward_model(ctx, weights, x, thread_ids=mx.array([0], dtype=mx.int32), h_states=None)
        loss = cross_entropy(logits_eval, y)
        lmax = float(mx.max(logits_eval).item())
        r_std = float(mx.std(rewards).item())
        head_delta = float(mx.max(mx.abs(weights.head.weight - head_before)).item())
        print(f"step {step}: loss={loss.item():.4f}, logits_max={lmax:.1f}, reward_std={r_std:.3f}")
        print(f"  update_threshold={cfg.update_threshold}, head_delta_max={head_delta}")


if __name__ == "__main__":
    main()
