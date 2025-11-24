import argparse
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple

import marimo
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np

__generated_with = "0.18.0"
app = marimo.App()

"""
The short answer is yes, the primary benefit is engineering scalability (bandwidth), but there is a secondary theoretical benefit regarding intrinsic dimensionality.
In a single-machine simulation (like these scripts), Eggroll is effectively "Full ES with a handicap." You are forcing the optimizer to search in a smaller, constrained subspace (Rank 2) rather than the full parameter space.
1. The Engineering "Win" (Bandwidth)
In these small experiments, the benefit is invisible (or negative due to overhead).
*   Small MLP (Current): Parameters $\approx 350$. Sending full updates is trivial.
*   LLM (Target): Parameters $\approx 70,000,000,000$.
    *   Full ES: Every worker must communicate a dense gradient vector (280 GB per step). Impossible over standard networks.
    *   Eggroll: Workers communicate only the random seed + low-rank factors ($A, B$). For rank=16, this might be $<0.1\%$ of the data.
The Bandwidth r column in your table estimates this compression.
*   In exp_2 (MLP), r=0.562 (56% of original size). Not impressive.
*   In a real LLM setting, r would be ~0.001 (99.9% reduction).
2. The Algorithmic "Win" (Intrinsic Dimension)
There is a hypothesis (popularized by LoRA) that the "intrinsic dimension" of learning is low.
*   Hypothesis: You don't need to adjust all 70B parameters independently to improve the loss. A random slice through the parameter space often contains a good descent direction.
*   Result: If this hypothesis holds, Eggroll (Low-Rank ES) should converge almost as fast as Full ES, despite ignoring 99% of the directions.
*   Your Data: In exp_2, Eggroll reached 84.88% accuracy vs Full ES 87.80%. It performed nearly as well while searching a much smaller space. This validates that the "Low Rank Hypothesis" holds reasonably well even for this small MLP.
Summary
If you can afford Full ES (small models), use Full ES (or SGD). It will always be more sample-efficient because it sees the true gradient.
Eggroll is for when the model is so large that Full ES is physically impossible, and it bets that a low-rank approximation is "good enough" to keep training.
3. The "Intrinsic Dimension" Bet
This is the theoretical catch.
*   Full ES: Unbiased. If run forever, it finds the true gradient.
*   Eggroll: Biased. It assumes the steepest descent direction lies (mostly) within a random rank-16 subspace.
For a tiny MLP (two-moons), the parameter space is so small (~300 params) that "Full" access is cheap, and restricting to Rank-2 cuts off useful directions, hurting accuracy (as seen in your 97% vs 93% result).
For an LLM (70B params), the "Intrinsic Dimension" hypothesis (supported by LoRA's success) states that you only need to optimize a tiny subspace to get 99% of the performance. Eggroll bets on this property to gain the 100x speedup.
Summary Table
| Feature | Small MLP (Current Exp) | LLM (70B Params) |
| :--- | :--- | :--- |
| Parameter Count ($d$) | ~350 | ~70,000,000,000 |
| Population ($P$) | 32 | 1,024 |
| Full ES Data/Step | 44 KB (Trivial) | 286 TB (Impossible) |
| Eggroll Data/Step | 20 KB (Trivial) | ~2 TB (Doable) |
| Bottleneck | Python Overhead | RNG & Memory Bandwidth |
| Accuracy Winner | Full ES (Unbiased) | Eggroll (Runs 100x faster) |
You are seeing Full ES win now because the "tax" (RNG/Memory) is negligible at this scale, so you only see the "benefit" (better gradients). At scale, the tax kills Full ES completely.
"""
# -------------------------------
# Config and task definitions
# -------------------------------
@dataclass
class ExperimentConfig:
    task: str = "linear"  # linear | ill_cond | step
    input_dim: int = 256
    output_dim: int = 16
    samples: int = 2048
    steps: int = 200
    eval_budget: int = 6400  # total forward evals allowed
    pop_size: int = 32
    sigma: float = 0.05
    lr_es: float = 0.05
    lr_sgd: float = 0.1
    rank: int = 2
    antithetic: bool = True
    seed: int = 42


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

        return TaskSpec(X, Y, loss_fn, True, "High-dim Linear", [(cfg.input_dim, cfg.output_dim)], "MSE regression")

    if cfg.task == "ill_cond":
        scales = mx.array(np.logspace(0, 2, cfg.input_dim))
        X_raw = mx.random.normal((cfg.samples, cfg.input_dim), key=k1)
        X = X_raw * scales
        true_W = mx.random.normal((cfg.input_dim, cfg.output_dim), key=k2)
        noise = mx.random.normal((cfg.samples, cfg.output_dim), key=k3) * 0.1
        Y = X @ true_W + noise

        def loss_fn(model: HighDimLinear):
            return mx.mean((model(X) - Y) ** 2)

        return TaskSpec(X, Y, loss_fn, True, "Ill-conditioned Linear", [(cfg.input_dim, cfg.output_dim)], "logspace feature scales (1..100)")

    if cfg.task == "step":
        X = mx.random.normal((cfg.samples, cfg.input_dim), key=k1)
        true_W = mx.random.normal((cfg.input_dim, cfg.output_dim), key=k2)
        logits = X @ true_W
        Y = (logits > 0).astype(mx.int32)

        def loss_fn(model: HighDimLinear):
            preds = model(X)
            acc = mx.mean((preds > 0).astype(mx.int32) == Y)
            return 1.0 - acc  # non-diff; ES only

        return TaskSpec(X, Y, loss_fn, False, "Non-diff Step", [(cfg.input_dim, cfg.output_dim)], "sign accuracy (ES only)")

    raise ValueError(f"Unknown task {cfg.task}")


# -------------------------------
# Helpers
# -------------------------------
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


# -------------------------------
# Optimizers
# -------------------------------
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
        new_params = mlx.utils.tree_map(lambda p, g: p - cfg.lr_sgd * g, model.trainable_parameters(), grads)
        model.update(new_params)
        mx.eval(model.parameters())
        losses.append(loss.item())
        evals += 1
    duration = time.perf_counter() - start
    return {"losses": losses, "time": duration, "evals": evals}


def _generate_full_noise(shapes, pop, sigma, key, antithetic):
    k = key
    half = pop // 2 if antithetic else pop
    noises = []
    for shape in shapes:
        k, sub = mx.random.split(k)
        base = mx.random.normal((half, *shape), key=sub) * sigma
        if antithetic:
            noises.append(mx.concatenate([base, -base], axis=0))
        else:
            noises.append(base)
    return noises


def _generate_low_rank_noise(shapes, pop, sigma, rank, key, antithetic):
    k = key
    half = pop // 2 if antithetic else pop
    noises = []
    for shape in shapes:
        if len(shape) < 2:
            k, sub = mx.random.split(k)
            base = mx.random.normal((half, *shape), key=sub) * sigma
            perturb = mx.concatenate([base, -base], axis=0) if antithetic else base
            noises.append(perturb)
            continue
        rows, cols = shape
        k, subA = mx.random.split(k)
        k, subB = mx.random.split(k)
        A = mx.random.normal((half, rows, rank), key=subA)
        B = mx.random.normal((half, cols, rank), key=subB)
        pert = (sigma / np.sqrt(rank)) * mx.einsum("pir,pjr->pij", A, B)
        if antithetic:
            pert = mx.concatenate([pert, -pert], axis=0)
        noises.append(pert)
    return noises


def run_es(task: TaskSpec, cfg: ExperimentConfig, mode: str = "full"):
    model = HighDimLinear(task.dims[0][0], task.dims[0][1])
    mx.eval(model.parameters())
    params_paths, params_vals = flatten_params(model.trainable_parameters())
    pop = cfg.pop_size if cfg.pop_size % 2 == 0 or not cfg.antithetic else cfg.pop_size + 1

    losses: List[float] = []
    evals = 0
    key = mx.random.key(cfg.seed + 123 if mode == "low_rank" else cfg.seed + 321)
    start = time.perf_counter()
    for _ in range(cfg.steps):
        if evals >= cfg.eval_budget:
            break
        current_loss = task.loss_fn(model).item()
        losses.append(current_loss)

        k, key = mx.random.split(key)
        if mode == "low_rank":
            perturbations = _generate_low_rank_noise([p.shape for p in params_vals], pop, cfg.sigma, cfg.rank, k, cfg.antithetic)
        else:
            perturbations = _generate_full_noise([p.shape for p in params_vals], pop, cfg.sigma, k, cfg.antithetic)

        rewards = []
        for j in range(pop):
            perturbed = [p + perturbations[i][j] for i, p in enumerate(params_vals)]
            model.update(unflatten(params_paths, perturbed))
            l = task.loss_fn(model)
            rewards.append(-l.item())

        # reset to base params
        model.update(unflatten(params_paths, params_vals))

        rewards = np.array(rewards)
        centered = rewards - rewards.mean()

        updates = []
        for i, param in enumerate(params_vals):
            grad_est = mx.zeros_like(param)
            for j in range(pop):
                grad_est += perturbations[i][j] * centered[j]
            grad_est /= (pop * cfg.sigma)
            updates.append(param + cfg.lr_es * grad_est)

        params_vals = updates
        model.update(unflatten(params_paths, params_vals))
        mx.eval(model.parameters())
        evals += pop

    duration = time.perf_counter() - start
    return {"losses": losses, "time": duration, "evals": evals, "pop": pop}


# -------------------------------
# Runner and reporting
# -------------------------------
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


def render_markdown(results: Dict[str, Dict], cfg: ExperimentConfig):
    lines = [
        f"### Task: {results['es_full']['task']} ({results['es_full']['info']})",
        f"Config: `{asdict(cfg)}`",
        "",
        "| Optimizer | Final Loss | Evals | Time (s) | Bandwidth r |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, res in results.items():
        final_loss = res["losses"][-1] if res["losses"] else float("nan")
        lines.append(
            f"| {name} | {final_loss:.4f} | {res['evals']} | {res['time']:.3f} | {res['bandwidth_ratio']:.3f} |"
        )
    return "\n".join(lines)


# -------------------------------
# Marimo cell for quick view
# -------------------------------
@app.cell
def _(mo=marimo, ExperimentConfig=ExperimentConfig, run_all=run_all, render_markdown=render_markdown):
    cfg = ExperimentConfig()
    results = run_all(cfg)
    mo.md(render_markdown(results, cfg))
    return cfg, results


# -------------------------------
# CLI entry
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="EGGROLL vs ES vs SGD harness (MLX)")
    parser.add_argument("--task", type=str, default="linear", choices=["linear", "ill_cond", "step"])
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
    args = parser.parse_args()
    cfg = ExperimentConfig(**vars(args))
    results = run_all(cfg)
    print(render_markdown(results, cfg))


if __name__ == "__main__":
    main()
