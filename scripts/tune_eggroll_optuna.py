"""
Lightweight Optuna tuner for the Eggroll ES trainer on the count dataset.

It tunes a few hyperparameters over a small number of steps to keep runs fast.
Requires optuna (`pip install optuna`).
"""

import argparse
from pathlib import Path
import numpy as np
import mlx.core as mx

try:
    import optuna
except ImportError as e:  # pragma: no cover
    raise SystemExit("optuna not installed. Install with: pip install optuna") from e

from eggroll_api import make_context
from eggroll_mlx import NoiseConfig, apply_sign_update, convert_fitnesses
from scripts.train_tinystories_gru_full import (
    init_model,
    forward_model,
    cross_entropy,
)


def load_tokens(path: Path):
    return np.memmap(path, mode="r", dtype=np.uint16)


def get_batch(memmap, seq_len: int, batch_size: int, rng):
    total_tokens = memmap.shape[0]
    n_seq = total_tokens // seq_len
    idx = rng.integers(0, n_seq - batch_size)
    arr = np.array(memmap[idx * seq_len : (idx + batch_size) * seq_len], copy=False)
    arr = arr.reshape(batch_size, seq_len)
    x = mx.array(arr.astype(np.int32))
    # Targets are next tokens (shifted by 1), last token repeats last input
    y_seq = mx.concatenate([x[:, 1:], x[:, -1:]], axis=1)
    return x, y_seq


def objective(trial, args, memmap):
    rng = np.random.default_rng(args.seed + trial.number)

    # Suggest hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    fitness_alpha = trial.suggest_float("fitness_alpha", 0.005, 0.05, log=True)
    sigma_shift = trial.suggest_int("sigma_shift", 3, 6)
    update_threshold = trial.suggest_int("update_threshold", 0, 2)
    rank = trial.suggest_int("rank", 1, 4)

    cfg = NoiseConfig(
        fixed_point=args.fixed_point,
        sigma_shift=sigma_shift,
        rank=rank,
        fast_fitness=bool(args.fast_fitness),
        fitness_alpha=fitness_alpha,
        update_threshold=update_threshold,
        noise_reuse=args.noise_reuse,
        debug_perturbations=False,
        learning_rate=lr,
        weight_clip=args.weight_clip,
        use_clt=True,
    )

    param_span = (args.d_model + 3 * args.d_hidden + 4 * args.d_hidden + args.vocab_size) * cfg.rank * args.pop_size * 2
    ctx = make_context(cfg, param_span=param_span, seed=args.seed + trial.number, safety_margin=4096)

    weights = init_model(cfg, args.vocab_size, args.seq_len, args.d_model, args.d_hidden, args.layers, rng, args.init_scale)

    thread_ids = list(range(args.pop_size))
    thread_ids_for_update = [i * 2 for i in range(args.pop_size // 2)]

    loss_and_reward = []

    for step in range(args.steps):
        ctx.epoch = step
        x, y = get_batch(memmap, args.seq_len, args.batch_size, rng)
        logits, _ = forward_model(ctx, weights, x, thread_ids, h_states=None, return_seq_logits=True)
        y_broad = mx.broadcast_to(y[None, ...], (logits.shape[0],) + y.shape)
        loss_pop = cross_entropy(logits, y_broad, reduction="none")
        # mean over batch and seq for rewards
        rewards = -mx.mean(loss_pop, axis=(1, 2)).reshape(-1)
        fitnesses = convert_fitnesses(cfg, rewards)
        mx.eval(fitnesses)

        # Updates
        def update_vec(arr, seed_offset):
            updated, _ = apply_sign_update(cfg, arr.reshape(-1, 1), fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=seed_offset)
            return updated.reshape(arr.shape)

        def update_mat(mat, seed_offset):
            updated_float, _ = apply_sign_update(cfg, mat.weight_unpacked, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=seed_offset)
            mat.weight_unpacked = updated_float
            mat.weight = updated_float
            return mat

        weights.head = update_mat(weights.head, seed_offset=0)
        base = 10
        for blk in weights.blocks:
            blk.gru_x = update_mat(blk.gru_x, seed_offset=base)
            blk.gru_h = update_mat(blk.gru_h, seed_offset=base + 1)
            blk.mlp_in = update_mat(blk.mlp_in, seed_offset=base + 2)
            blk.mlp_out = update_mat(blk.mlp_out, seed_offset=base + 4)
            blk.ln1_scale = update_vec(blk.ln1_scale, seed_offset=base + 8)
            blk.ln1_bias = update_vec(blk.ln1_bias, seed_offset=base + 8 + 1000)
            blk.ln2_scale = update_vec(blk.ln2_scale, seed_offset=base + 9)
            blk.ln2_bias = update_vec(blk.ln2_bias, seed_offset=base + 9 + 1000)
            base += 100

        weights.ln_out_scale = update_vec(weights.ln_out_scale, seed_offset=1001)
        weights.ln_out_bias = update_vec(weights.ln_out_bias, seed_offset=1001 + 1000)
        weights.tok_emb = apply_sign_update(cfg, weights.tok_emb, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=1002)[0]
        weights.pos_emb = apply_sign_update(cfg, weights.pos_emb, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update, seed_offset=1003)[0]

        # Evaluate single-pop loss for reporting
        logits_eval, _ = forward_model(ctx, weights, x, thread_ids=None, h_states=None, return_seq_logits=True)
        loss_eval = cross_entropy(logits_eval, y, reduction="mean")
        loss_and_reward.append(loss_eval.item())

    final_loss = float(np.mean(loss_and_reward[-5:])) if loss_and_reward else float("inf")
    trial.report(final_loss, step=args.steps)
    return final_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/count")
    ap.add_argument("--vocab_size", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--d_hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--pop_size", type=int, default=256)
    ap.add_argument("--group_size", type=int, default=256)
    ap.add_argument("--init_scale", type=float, default=16.0)
    ap.add_argument("--fast_fitness", type=int, default=0)
    ap.add_argument("--noise_reuse", type=int, default=1)
    ap.add_argument("--fixed_point", type=int, default=4)
    ap.add_argument("--weight_clip", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    train_path = Path(args.data_dir) / "train_tokens.npy"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found. Generate with scripts/make_count_data.py")
    memmap = load_tokens(train_path)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args, memmap), n_trials=args.trials)

    print("Best trial:", study.best_trial.number)
    print("  loss:", study.best_value)
    print("  params:", study.best_params)


if __name__ == "__main__":
    main()
