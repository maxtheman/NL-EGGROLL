"""
Scripted EGGROLL vs ES vs SGD benchmarks (no marimo).
Refined to match nano-egg accumulation logic (aggregate then update).

Usage examples:
  uv run python scripts/eggroll_benchmark_v2.py --task linear --input_dim 128 --output_dim 8
  uv run python scripts/eggroll_benchmark_v2.py --task ill_cond --eval_budget 3200 --pop_size 16
  uv run python scripts/eggroll_benchmark_v2.py --task step --pop_size 32 --rank 2 --sigma 0.05
"""

import argparse
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Tuple, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np


@dataclass
class ExperimentConfig:
    task: str = "linear"  # linear | ill_cond | step
    input_dim: int = 256
    output_dim: int = 16
    samples: int = 2048  # for token task, acts as num sequences
    steps: int = 200
    eval_budget: int = 6400  # total forward evals allowed
    pop_size: int = 32
    sigma: float = 0.05
    lr_es: float = 0.05
    lr_sgd: float = 0.1
    rank: int = 2
    antithetic: bool = True
    seed: int = 42
    fast_fitness: bool = True
    update_threshold: float = (
        0.0  # if >0, gate individual scores (rarely used if Z-threshold is used)
    )
    noise_reuse: int = 1  # how many epochs to reuse same noise slice
    group_size: int = 2  # must be even when antithetic
    dtype: str = "float32"
    mode_map: str = "full"  # full | lora; affects es_map
    use_sign_updates: bool = True  # mimic nano-egg sign/threshold updates
    param_clip: float = 1.0  # clip magnitude when applying sign updates (float mode)
    int_mode: bool = False  # store params as int8 and clip
    z_threshold: float = 0.0  # Z-quantile gating (e.g., 1.96)
    stride: int = 4096  # offset stride into BIG_RAND per thread/epoch


class HighDimLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.lin = nn.Linear(input_dim, output_dim, bias=False)

    def __call__(self, x):
        return self.lin(x)


@dataclass
class TaskSpec:
    X: mx.array
    Y: mx.array
    loss_fn: Callable[[HighDimLinear], mx.array]
    allows_backprop: bool
    name: str
    dims: List[Tuple[int, int]]
    info: str
    es_map: List[int]  # 0 = full, 1 = lora, 2 = noop
    thread_ids: List[int]  # thread id per population member baseline


def make_task(cfg: ExperimentConfig) -> TaskSpec:
    key = mx.random.key(cfg.seed)
    k1, k2, k3 = mx.random.split(key, 3)

    if cfg.task == "linear":
        X = mx.random.normal((cfg.samples, cfg.input_dim), key=k1)
        true_W = mx.random.normal((cfg.input_dim, cfg.output_dim), key=k2)
        noise = mx.random.normal((cfg.samples, cfg.output_dim), key=k3) * 0.1
        Y = X @ true_W + noise

        def loss_fn(model: HighDimLinear):
            return mx.mean((model(X) - Y) ** 2)

        # es_map can switch behavior: full=0, lora=1, noop=2
        mode_map = cfg.mode_map.lower()
        if mode_map == "lora":
            es_map = [1]
        elif mode_map == "noop":
            es_map = [2]
        else:
            es_map = [0]
        thread_ids = list(range(cfg.pop_size))
        return TaskSpec(
            X,
            Y,
            loss_fn,
            True,
            "High-dim Linear",
            [(cfg.input_dim, cfg.output_dim)],
            "MSE regression",
            es_map,
            thread_ids,
        )

    if cfg.task == "ill_cond":
        scales = mx.array(np.logspace(0, 2, cfg.input_dim))
        X_raw = mx.random.normal((cfg.samples, cfg.input_dim), key=k1)
        X = X_raw * scales
        true_W = mx.random.normal((cfg.input_dim, cfg.output_dim), key=k2)
        noise = mx.random.normal((cfg.samples, cfg.output_dim), key=k3) * 0.1
        Y = X @ true_W + noise

        def loss_fn(model: HighDimLinear):
            return mx.mean((model(X) - Y) ** 2)

        mode_map = cfg.mode_map.lower()
        if mode_map == "lora":
            es_map = [1]
        elif mode_map == "noop":
            es_map = [2]
        else:
            es_map = [0]
        thread_ids = list(range(cfg.pop_size))
        return TaskSpec(
            X,
            Y,
            loss_fn,
            True,
            "Ill-conditioned Linear",
            [(cfg.input_dim, cfg.output_dim)],
            "logspace feature scales (1..100)",
            es_map,
            thread_ids,
        )

    if cfg.task == "step":
        X = mx.random.normal((cfg.samples, cfg.input_dim), key=k1)
        true_W = mx.random.normal((cfg.input_dim, cfg.output_dim), key=k2)
        logits = X @ true_W
        Y = (logits > 0).astype(mx.int32)

        def loss_fn(model: HighDimLinear):
            preds = model(X)
            # Fix type error: cast comparison to float/int before mean
            # correct = (preds > 0).astype(mx.int32) == Y
            correct = mx.equal((preds > 0).astype(mx.int32), Y)
            acc = mx.mean(correct.astype(mx.float32))
            return 1.0 - acc  # non-diff; ES only

        mode_map = cfg.mode_map.lower()
        if mode_map == "lora":
            es_map = [1]
        elif mode_map == "noop":
            es_map = [2]
        else:
            es_map = [0]
        thread_ids = list(range(cfg.pop_size))
        return TaskSpec(
            X,
            Y,
            loss_fn,
            False,
            "Non-diff Step",
            [(cfg.input_dim, cfg.output_dim)],
            "sign accuracy (ES only)",
            es_map,
            thread_ids,
        )

    raise ValueError(f"Unknown task {cfg.task}")


def flatten_params(tree):
    flat = mlx.utils.tree_flatten(tree)
    paths = [p for p, _ in flat]
    vals = [v for _, v in flat]
    return paths, vals


def unflatten(paths, vals):
    return mlx.utils.tree_unflatten(list(zip(paths, vals)))


def bandwidth_proxy(dims: List[Tuple[int, int]], rank: int) -> float:
    full = 0.0
    low = 0.0
    for m, n in dims:
        full += m * n
        low += rank * (m + n)
    return low / full if full > 0 else 0.0


# ----------------------------------------------------------------------
# Deterministic noise (nano-egg style): precompute BIG_RAND and slice by (epoch, thread)
# ----------------------------------------------------------------------
def make_big_rand_matrix(max_size: int, dtype: str, seed: int):
    key = mx.random.key(seed)
    big = mx.random.normal((max_size,), key=key).astype(getattr(mx, dtype))
    return big


def slice_noise(big_rand: mx.array, start: int, length: int):
    return big_rand[start : start + length]


def run_sgd(task: TaskSpec, cfg: ExperimentConfig):
    model = HighDimLinear(task.dims[0][0], task.dims[0][1])
    mx.eval(model.parameters())
    loss_and_grad = nn.value_and_grad(model, lambda m: task.loss_fn(m))

    losses: List[float] = []
    evals = 0
    start = time.perf_counter()
    for _ in range(cfg.steps):
        if evals >= cfg.eval_budget:
            break
        loss, grads = loss_and_grad(model)
        new_params = mlx.utils.tree_map(
            lambda p, g: p - cfg.lr_sgd * g, model.trainable_parameters(), grads
        )
        model.update(new_params)
        mx.eval(model.parameters())
        losses.append(loss.item())
        evals += 1
    duration = time.perf_counter() - start
    return {"losses": losses, "time": duration, "evals": evals}


def _generate_full_noise(
    shapes,
    pop,
    sigma,
    key,
    antithetic,
    big_rand=None,
    start_idx=0,
    start_indices=None,
    stride=1024,
):
    k = key
    noises = [mx.zeros((pop, *shape)) for shape in shapes]
    if start_indices is None:
        start_indices = [start_idx + j * stride for j in range(pop)]
    for worker in range(pop):
        w_start = start_indices[worker]
        for idx_shape, shape in enumerate(shapes):
            total = int(np.prod(shape))
            if big_rand is not None:
                base = slice_noise(big_rand, w_start, total)
                w_start += total
                base = mx.reshape(base, shape)
            else:
                k, sub = mx.random.split(k)
                base = mx.random.normal(shape, key=sub)
            base = base * sigma
            sign = 1 if (not antithetic or worker % 2 == 0) else -1
            noises[idx_shape][worker] = base * sign
    return noises


def _generate_low_rank_noise(
    shapes, pop, sigma, rank, key, antithetic, big_rand=None, start_idx=0, stride=1024
):
    k = key
    noises = [mx.zeros((pop, *shape)) for shape in shapes]
    start_indices = [start_idx + j * stride for j in range(pop)]
    for worker in range(pop):
        w_start = start_indices[worker]
        for idx_shape, shape in enumerate(shapes):
            if len(shape) < 2:
                total = int(np.prod(shape))
                if big_rand is not None:
                    base = slice_noise(big_rand, w_start, total)
                    w_start += total
                    base = mx.reshape(base, shape)
                else:
                    k, sub = mx.random.split(k)
                    base = mx.random.normal(shape, key=sub)
                base = base * sigma
                sign = 1 if (not antithetic or worker % 2 == 0) else -1
                noises[idx_shape][worker] = base * sign
                continue
            rows, cols = shape
            total_A = rows * rank
            total_B = cols * rank
            if big_rand is not None:
                A_slice = slice_noise(big_rand, w_start, total_A)
                w_start += total_A
                B_slice = slice_noise(big_rand, w_start, total_B)
                w_start += total_B
                A = mx.reshape(A_slice, (rows, rank))
                B = mx.reshape(B_slice, (cols, rank))
            else:
                k, subA = mx.random.split(k)
                k, subB = mx.random.split(k)
                A = mx.random.normal((rows, rank), key=subA)
                B = mx.random.normal((cols, rank), key=subB)
            scale = sigma / mx.sqrt(mx.array(rank, dtype=mx.float32))
            pert = scale * mx.einsum("ir,jr->ij", A, B)
            sign = 1 if (not antithetic or worker % 2 == 0) else -1
            noises[idx_shape][worker] = pert * sign
    return noises


def _pairwise_scores(rewards: np.ndarray, fast_fitness: bool = True):
    rewards = rewards.reshape(-1, 2)
    diff = rewards[:, 0] - rewards[:, 1]
    if fast_fitness:
        return np.sign(diff)
    # CLT-like normalization
    rms = np.sqrt(np.mean(diff**2) + 1e-8)
    return diff / rms


def _apply_update_final(param: mx.array, accumulator: mx.array, cfg: ExperimentConfig):
    if cfg.int_mode:
        # Nano-egg style integer update: step by +/- 1 based on aggregate signal
        # Accumulator 'Z' is sum of (score * sign(noise)) usually

        step = mx.zeros((accumulator.shape), dtype=accumulator.dtype)
        if cfg.z_threshold > 0:
            mask = mx.abs(accumulator) >= cfg.z_threshold
            step = mx.where(mask, mx.sign(accumulator), step)
        else:
            step = mx.sign(accumulator)

        updated = param + step
        updated = mx.clip(updated, -1, 1).astype(mx.int8)
        return updated
    else:
        # Float update - batch averaging
        if cfg.z_threshold > 0:
            mask = mx.abs(accumulator) >= cfg.z_threshold
            accumulator = mx.where(mask, accumulator, 0)

        updated = param + cfg.lr_es * accumulator
        updated = mx.clip(updated, -cfg.param_clip, cfg.param_clip)
        return updated


def run_es(task: TaskSpec, cfg: ExperimentConfig, mode: str = "full"):
    model = HighDimLinear(task.dims[0][0], task.dims[0][1])
    mx.eval(model.parameters())
    params_paths, params_vals_any = flatten_params(model.trainable_parameters())
    # Ensure items are arrays
    params_vals = [p for p in params_vals_any if isinstance(p, mx.array)]

    # Initialize int8 if needed
    if cfg.int_mode:
        params_vals = [mx.clip(p, -1, 1).astype(mx.int8) for p in params_vals]
        model.update(unflatten(params_paths, params_vals))

    pop = (
        cfg.pop_size
        if cfg.pop_size % 2 == 0 or not cfg.antithetic
        else cfg.pop_size + 1
    )

    # precompute big rand for deterministic slicing
    big_rand = make_big_rand_matrix(max_size=10_000_000, dtype=cfg.dtype, seed=cfg.seed)

    losses: List[float] = []
    evals = 0
    key = mx.random.key(cfg.seed + (123 if mode == "low_rank" else 321))
    start = time.perf_counter()

    for _ in range(cfg.steps):
        if evals >= cfg.eval_budget:
            break
        current_loss = task.loss_fn(model).item()
        losses.append(current_loss)

        k, key = mx.random.split(key)
        # deterministic start index based on epoch (step) and noise_reuse
        reuse_block = _ // max(1, cfg.noise_reuse)
        start_idx = (reuse_block * cfg.group_size * cfg.stride) % big_rand.size

        # Generate perturbations
        if mode == "low_rank":
            perturbations = _generate_low_rank_noise(
                [p.shape for p in params_vals],
                pop,
                cfg.sigma,
                cfg.rank,
                k,
                cfg.antithetic,
                big_rand=big_rand,
                start_idx=start_idx,
            )
        else:
            perturbations = _generate_full_noise(
                [p.shape for p in params_vals],
                pop,
                cfg.sigma,
                k,
                cfg.antithetic,
                big_rand=big_rand,
                start_idx=start_idx,
            )

        rewards = []
        for j in range(pop):
            # Apply individual perturbation
            perturbed = [p + perturbations[i][j] for i, p in enumerate(params_vals)]
            model.update(unflatten(params_paths, perturbed))
            l = task.loss_fn(model)
            rewards.append(-l.item())

        # Restore center
        model.update(unflatten(params_paths, params_vals))

        rewards = np.array(rewards)

        if cfg.antithetic:
            paired_scores = _pairwise_scores(rewards, fast_fitness=cfg.fast_fitness)
            pair_indices = list(range(0, pop, 2))
        else:
            paired_scores = rewards
            pair_indices = list(range(pop))

        updates = []
        # Calculate update per parameter
        for idx_param, param in enumerate(params_vals):
            map_class = task.es_map[idx_param] if idx_param < len(task.es_map) else 0
            if map_class == 2:
                updates.append(param)
                continue

            # Accumulate "gradient" Z
            accumulator = mx.zeros((param.shape), dtype=mx.float32)

            if cfg.antithetic:
                for idx, score in zip(pair_indices, paired_scores):
                    if cfg.update_threshold > 0 and abs(score) < cfg.update_threshold:
                        continue

                    # In antithetic with +/- pairs, we use the positive perturbation
                    pert = perturbations[idx_param][idx]
                    step = mx.sign(pert) if cfg.use_sign_updates else pert
                    accumulator = accumulator + step * score
            else:
                mean_reward = paired_scores.mean() if cfg.fast_fitness else 0.0
                for j in range(pop):
                    score = paired_scores[j] - mean_reward
                    if cfg.update_threshold > 0 and abs(score) < cfg.update_threshold:
                        continue

                    pert = perturbations[idx_param][j]
                    step = mx.sign(pert) if cfg.use_sign_updates else pert
                    accumulator = accumulator + step * score

            # Apply aggregated update (with Z-thresholding if enabled)
            updated = _apply_update_final(param, accumulator, cfg)
            updates.append(updated)

        params_vals = updates
        model.update(unflatten(params_paths, params_vals))
        mx.eval(model.parameters())
        evals += pop

    duration = time.perf_counter() - start
    return {"losses": losses, "time": duration, "evals": evals, "pop": pop}


def run_all(cfg: ExperimentConfig) -> Dict[str, Dict]:
    task = make_task(cfg)
    results: Dict[str, Dict] = {}
    if task.allows_backprop:
        results["sgd"] = run_sgd(task, cfg)
    results["es_full"] = run_es(task, cfg, mode="full")
    results["eggroll_low_rank"] = run_es(task, cfg, mode="low_rank")
    bw_ratio = bandwidth_proxy(task.dims, cfg.rank)
    for v in results.values():
        v["bandwidth_ratio"] = bw_ratio
        v["task"] = task.name
        v["info"] = task.info
    return results


def render_table(results: Dict[str, Dict], cfg: ExperimentConfig) -> str:
    lines = [
        f"Task: {list(results.values())[0]['task']} ({list(results.values())[0]['info']})",
        f"Config: {asdict(cfg)}",
        "",
        "Optimizer\tFinal Loss\tEvals\tTime (s)\tBandwidth r",
    ]
    for name, res in results.items():
        final_loss = res["losses"][-1] if res["losses"] else float("nan")
        lines.append(
            f"{name}\t{final_loss:.4f}\t{res['evals']}\t{res['time']:.3f}\t{res['bandwidth_ratio']:.3f}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="EGGROLL vs ES vs SGD harness (MLX, script) V2"
    )
    parser.add_argument(
        "--task", type=str, default="linear", choices=["linear", "ill_cond", "step"]
    )
    parser.add_argument("--input_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=16)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval_budget", type=int, default=6400)
    parser.add_argument("--pop_size", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lr_es", type=float, default=0.05)
    parser.add_argument("--lr_sgd", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--antithetic", action="store_true", default=True)
    parser.add_argument("--no-antithetic", action="store_false", dest="antithetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast_fitness", action="store_true", default=True)
    parser.add_argument("--update_threshold", type=float, default=0.0)
    parser.add_argument("--noise_reuse", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--mode_map", type=str, default="full")
    parser.add_argument("--use_sign_updates", action="store_true", default=True)
    parser.add_argument(
        "--no_use_sign_updates", action="store_false", dest="use_sign_updates"
    )
    parser.add_argument("--param_clip", type=float, default=1.0)
    parser.add_argument("--int_mode", action="store_true", default=False)
    parser.add_argument("--no_int_mode", action="store_false", dest="int_mode")
    parser.add_argument("--z_threshold", type=float, default=0.0)
    parser.add_argument("--stride", type=int, default=4096)
    args = parser.parse_args()
    cfg = ExperimentConfig(**vars(args))
    results = run_all(cfg)
    print(render_table(results, cfg))


if __name__ == "__main__":
    main()
