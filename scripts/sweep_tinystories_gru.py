"""
Parameter sweep harness for TinyStories int8 GRU ES training.
- Imports the training primitives from train_tinystories_gru_int8.py and runs short trials.
- Reports final loss, reward std, and avg step time per config.

Example:
PYTHONPATH=. uv run python scripts/sweep_tinystories_gru.py \
  --data_dir data/tinystories \
  --configs "fixed_point=2,sigma_shift=2,init_scale=16.0,pop_size=10,fast_fitness=0" \
            "fixed_point=2,sigma_shift=3,init_scale=16.0,pop_size=10,fast_fitness=1"
"""

import argparse
import time
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import mlx.core as mx

from scripts.train_tinystories_gru_int8 import (
    NoiseConfig,
    apply_sign_update,
    convert_fitnesses,
    forward_gru_int8,
    forward_gru_int8_batched,
    get_batch,
    init_weights,
    load_tokens,
    make_context,
    cross_entropy,
)


def parse_kv_list(items: List[str]) -> List[Dict[str, str]]:
    parsed = []
    for item in items:
        cfg = {}
        for kv in item.split(","):
            k, v = kv.split("=")
            cfg[k.strip()] = v.strip()
        parsed.append(cfg)
    return parsed


def expand_grid(grid: Dict[str, List[str]]) -> List[Dict[str, str]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    combos = []
    for prod_vals in product(*vals):
        combos.append({k: v for k, v in zip(keys, prod_vals)})
    return combos


def run_trial(
    cfg_dict: Dict[str, str],
    data_dir: Path,
    seq_len: int,
    vocab_size: int,
    d_model: int,
    d_hidden: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> Dict[str, float]:
    # typed config
    cfg = NoiseConfig(
        fixed_point=int(cfg_dict.get("fixed_point", 2)),
        sigma_shift=int(cfg_dict.get("sigma_shift", 2)),
        rank=1,
        fast_fitness=bool(int(cfg_dict.get("fast_fitness", 0))),
    )
    init_scale = float(cfg_dict.get("init_scale", 16.0))
    pop_size = int(cfg_dict.get("pop_size", 10))
    if pop_size % 2 != 0:
        raise ValueError("pop_size must be even for paired fitnesses.")

    rng = np.random.default_rng(seed)
    train_path = data_dir / "train_tokens.npy"
    if not train_path.exists():
        train_path = data_dir / "train_tokens.uint16.memmap"
    memmap = load_tokens(train_path)

    param_span = (3 * d_hidden + d_model) * cfg.rank * pop_size * 2
    ctx = make_context(cfg, param_span=param_span, seed=seed, safety_margin=4096)
    weights = init_weights(cfg, vocab_size, seq_len, d_model, d_hidden, rng, init_scale)

    thread_ids = list(range(pop_size))
    thread_ids_for_update = [i * 2 for i in range(pop_size // 2)]

    step_times: List[float] = []
    final_loss = None
    final_reward_std = None
    for step in range(steps):
        ctx.epoch = step
        x, y = get_batch(memmap, seq_len, batch_size=batch_size, rng=rng)

        t0 = time.perf_counter()
        logits_pop = forward_gru_int8_batched(ctx, weights, x, thread_ids)  # (P,B,vocab)
        rewards = -mx.vmap(lambda l: cross_entropy(l, y))(logits_pop).reshape(-1)
        fitnesses = convert_fitnesses(cfg, rewards)
        weights.gru_x.weight = apply_sign_update(cfg, weights.gru_x.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)
        weights.gru_h.weight = apply_sign_update(cfg, weights.gru_h.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)
        weights.lm_head.weight = apply_sign_update(cfg, weights.lm_head.weight, fitnesses, ctx.big_rand, step, ctx.base_seed, thread_ids=thread_ids_for_update)
        step_times.append(time.perf_counter() - t0)

        logits_eval = forward_gru_int8(ctx, weights, x, thread_id=None)
        final_loss = float(cross_entropy(logits_eval, y).item())
        final_reward_std = float(mx.std(rewards).item())

    return {
        "fixed_point": cfg.fixed_point,
        "sigma_shift": cfg.sigma_shift,
        "init_scale": init_scale,
        "pop_size": pop_size,
        "fast_fitness": int(cfg.fast_fitness),
        "loss": final_loss if final_loss is not None else float("nan"),
        "reward_std": final_reward_std if final_reward_std is not None else float("nan"),
        "avg_step_time": float(np.mean(step_times)) if step_times else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/tinystories")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_hidden", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=10, help="steps per trial")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--configs",
        nargs="*",
        default=[],
        help='List like "fixed_point=2,sigma_shift=2,init_scale=16.0,pop_size=10,fast_fitness=0"',
    )
    p.add_argument(
        "--grid",
        nargs="*",
        default=[],
        help='Grid spec like "sigma_shift=2|3 fixed_point=2|3 init_scale=8|16 pop_size=8|12 fast_fitness=0|1"',
    )
    args = p.parse_args()

    configs: List[Dict[str, str]] = []
    if args.configs:
        configs.extend(parse_kv_list(args.configs))
    if args.grid:
        grid = {}
        for item in args.grid:
            if "=" not in item:
                continue
            k, v = item.split("=")
            grid[k.strip()] = v.split("|")
        configs.extend(expand_grid(grid))
    if not configs:
        configs = [
            {"fixed_point": "2", "sigma_shift": "2", "init_scale": "16.0", "pop_size": "10", "fast_fitness": "0"},
            {"fixed_point": "2", "sigma_shift": "3", "init_scale": "16.0", "pop_size": "10", "fast_fitness": "1"},
        ]

    results = []
    for idx, cfg_dict in enumerate(configs):
        print(f"[{idx+1}/{len(configs)}] running {cfg_dict}")
        res = run_trial(
            cfg_dict,
            data_dir=Path(args.data_dir),
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + idx,
        )
        results.append(res)
        print(res)

    print("=== sweep summary ===")
    for res in results:
        print(res)


if __name__ == "__main__":
    main()
