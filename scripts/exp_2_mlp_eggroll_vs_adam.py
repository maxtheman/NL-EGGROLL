"""
EGGROLL (low-rank ES) vs full ES vs Adam on a tiny MLP classifier (two-moons).

Usage examples:
  uv run python scripts/exp_2_mlp_eggroll_vs_adam.py
  uv run python scripts/exp_2_mlp_eggroll_vs_adam.py --hidden_dim 32 --pop_size 32 --rank 4
"""

import argparse
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


@dataclass
class ExperimentConfig:
    samples: int = 1024
    noise: float = 0.1
    test_size: float = 0.2
    input_dim: int = 2
    hidden_dim: int = 16
    output_dim: int = 2
    steps: int = 200
    eval_budget: int = 6400
    pop_size: int = 32
    sigma: float = 0.05
    lr_es: float = 0.05
    lr_adam: float = 0.01
    rank: int = 2
    antithetic: bool = True
    seed: int = 42
    # New fields for Nano-egg fidelity
    fast_fitness: bool = True
    update_threshold: float = 0.0
    dtype: str = "float32"
    use_sign_updates: bool = True
    param_clip: float = 1.0
    int_mode: bool = False
    z_threshold: float = 0.0


class EggrollLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = mx.random.normal((output_dim, input_dim))
        self.bias = mx.zeros((output_dim,)) if bias else None

        # Buffers for current perturbation (populated during forward if active)
        # We store them as member vars to be update-able, but they aren't trained parameters in standard sense
        self.lora_a = None
        self.lora_b = None
        self.perturbation_scale = 1.0

    def set_perturbation(self, a, b, scale=1.0):
        self.lora_a = a
        self.lora_b = b
        self.perturbation_scale = scale

    def clear_perturbation(self):
        self.lora_a = None
        self.lora_b = None

    def __call__(self, x):
        # 1. Base forward pass (Shared Weights)
        # equivalent to x @ W.T + b
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias

        # 2. Add Low-Rank Perturbation if active
        # Perturbation is (x @ B) @ A.T  (or similar, depending on A/B shapes)
        # Noise generation produced A (out, rank) and B (in, rank) typically for W (out, in)
        # Delta W = scale * (A @ B.T)
        # x @ Delta W.T = x @ (scale * A @ B.T).T = scale * x @ (B @ A.T) = scale * (x @ B) @ A.T
        if self.lora_a is not None and self.lora_b is not None:
            # lora_a: (out_dim, rank)
            # lora_b: (in_dim, rank)
            # x: (batch, in_dim)

            # x @ lora_b -> (batch, rank)
            # result @ lora_a.T -> (batch, out_dim)

            lora_term = (x @ self.lora_b) @ self.lora_a.T
            out = out + self.perturbation_scale * lora_term

        return out


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, scale: float = 1.0
    ):
        super().__init__()
        # Use EggrollLinear instead of nn.Linear
        self.fc1 = EggrollLinear(input_dim, hidden_dim)
        self.fc2 = EggrollLinear(hidden_dim, output_dim)
        self.scale = scale

    def __call__(self, x):
        s = self.scale
        # Pass scale down? Or assume layers handle their own scale?
        # If we use integer scale logic, we divide output.
        x = nn.relu(self.fc1(x) / s)
        return self.fc2(x) / s

    def set_perturbations(self, perts, scale=1.0):
        # perts is list of (A, B) pairs or None
        # flattened params order: fc1.weight, fc1.bias, fc2.weight, fc2.bias
        # We need to map correctly. Bias doesn't have low-rank usually, or handled separately.
        # generate_low_rank_noise handles shapes.

        # This mapping depends heavily on flatten order.
        # Let's assume perts order matches modules order if we iterate.
        pass  # implemented in loop manually for now


def make_data(cfg: ExperimentConfig):
    X, y = make_moons(cfg.samples, noise=cfg.noise, random_state=cfg.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed
    )
    return (
        mx.array(X_train, dtype=mx.float32),
        mx.array(y_train, dtype=mx.int32),
        mx.array(X_test, dtype=mx.float32),
        mx.array(y_test, dtype=mx.int32),
    )


def cross_entropy_logits(logits, targets):
    num_classes = logits.shape[-1]
    one_hot = (targets[..., None] == mx.arange(num_classes)).astype(mx.float32)
    log_probs = logits - mx.log(mx.sum(mx.exp(logits), axis=-1, keepdims=True))
    return -mx.mean(mx.sum(one_hot * log_probs, axis=-1))


def accuracy(logits, targets):
    preds = mx.argmax(logits, axis=-1)
    return (preds == targets).astype(mx.float32).mean()


def flatten_params(tree):
    flat = mlx.utils.tree_flatten(tree)
    paths = [p for p, _ in flat]
    vals = [v for _, v in flat]
    return paths, vals


def unflatten(paths, vals):
    return mlx.utils.tree_unflatten(list(zip(paths, vals)))


def run_adam(X_train, y_train, X_test, y_test, cfg: ExperimentConfig):
    model = MLP(cfg.input_dim, cfg.hidden_dim, cfg.output_dim)
    mx.eval(model.parameters())

    # Minimal Adam implementation
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m, v = None, None

    def loss_fn(mdl):
        logits = mdl(X_train)
        return cross_entropy_logits(logits, y_train)

    losses = []
    accs = []
    start = time.perf_counter()
    for t in range(cfg.steps):
        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        if m is None:
            m = mlx.utils.tree_map(lambda g: mx.zeros_like(g), grads)
            v = mlx.utils.tree_map(lambda g: mx.zeros_like(g), grads)
        m = mlx.utils.tree_map(lambda m_i, g: beta1 * m_i + (1 - beta1) * g, m, grads)
        v = mlx.utils.tree_map(
            lambda v_i, g: beta2 * v_i + (1 - beta2) * (g * g), v, grads
        )
        m_hat = mlx.utils.tree_map(lambda m_i: m_i / (1 - beta1 ** (t + 1)), m)
        v_hat = mlx.utils.tree_map(lambda v_i: v_i / (1 - beta2 ** (t + 1)), v)
        updates = mlx.utils.tree_map(
            lambda m_i, v_i: -cfg.lr_adam * m_i / (mx.sqrt(v_i) + eps), m_hat, v_hat
        )
        new_params = mlx.utils.tree_map(
            lambda p, u: p + u, model.trainable_parameters(), updates
        )
        model.update(new_params)
        mx.eval(model.parameters())
        losses.append(loss.item())
        accs.append(accuracy(model(X_test), y_test).item())
    duration = time.perf_counter() - start
    return {"losses": losses, "accs": accs, "time": duration, "evals": cfg.steps}


def _generate_full_noise_v2(shapes, pop, sigma, key, antithetic):
    k = key
    noises = [mx.zeros((pop, *shape)) for shape in shapes]

    step = 2 if antithetic else 1

    for worker in range(0, pop, step):
        for idx_shape, shape in enumerate(shapes):
            k, sub = mx.random.split(k)
            base = mx.random.normal(shape, key=sub) * sigma

            noises[idx_shape][worker] = base
            if antithetic and (worker + 1 < pop):
                noises[idx_shape][worker + 1] = -base

    return noises


def _generate_low_rank_noise_v2_structured(shapes, pop, sigma, rank, key, antithetic):
    # Returns list of (A, B) tuples for matrix weights, or raw noise for vectors
    k = key
    noises = []

    # Pre-calculate structure for each param shape
    # We return a structure: noises[worker][param_idx] = (A, B) or noise
    worker_noises = [[] for _ in range(pop)]

    step = 2 if antithetic else 1

    for worker in range(0, pop, step):
        for idx_shape, shape in enumerate(shapes):
            if len(shape) < 2:
                # Vector (bias) - Full noise
                k, sub = mx.random.split(k)
                base = mx.random.normal(shape, key=sub) * sigma

                worker_noises[worker].append(base)
                if antithetic and (worker + 1 < pop):
                    worker_noises[worker + 1].append(-base)
                continue

            # Matrix - Low Rank
            rows, cols = shape
            k, subA = mx.random.split(k)
            k, subB = mx.random.split(k)
            # A: (rows, rank), B: (cols, rank) -> A @ B.T matches (rows, cols)
            # wait, einsum "ir,jr->ij" means A(rows, rank) and B(cols, rank).
            # So Delta = A @ B.T.

            A = mx.random.normal((rows, rank), key=subA)
            B = mx.random.normal((cols, rank), key=subB)

            # We store A and B directly, NOT the product
            # Scale sigma by sqrt(rank)
            scale = sigma / mx.sqrt(mx.array([rank]).astype(mx.float32)[0])

            worker_noises[worker].append(
                (A * scale, B)
            )  # B is not scaled? Product is scaled.
            if antithetic and (worker + 1 < pop):
                worker_noises[worker + 1].append(
                    (-A * scale, B)
                )  # Flip A sign flips product

    return worker_noises


def _pairwise_scores(rewards: np.ndarray, fast_fitness: bool = True):
    rewards = rewards.reshape(-1, 2)
    diff = rewards[:, 0] - rewards[:, 1]
    if fast_fitness:
        return np.sign(diff)
    rms = np.sqrt(np.mean(diff**2) + 1e-8)
    return diff / rms


def _apply_update_final(param: mx.array, accumulator: mx.array, cfg: ExperimentConfig):
    if cfg.int_mode:
        step = mx.zeros_like(accumulator)
        if cfg.z_threshold > 0:
            mask = mx.abs(accumulator) >= cfg.z_threshold
            step = mx.where(mask, mx.sign(accumulator), step)
        else:
            step = mx.sign(accumulator)
        updated = param + step
        updated = mx.clip(updated, -1, 1).astype(mx.int8)
        return updated
    else:
        if cfg.z_threshold > 0:
            mask = mx.abs(accumulator) >= cfg.z_threshold
            accumulator = mx.where(mask, accumulator, 0)
        updated = param + cfg.lr_es * accumulator
        updated = mx.clip(updated, -cfg.param_clip, cfg.param_clip)
        return updated


def run_es(mode: str, X_train, y_train, X_test, y_test, cfg: ExperimentConfig):
    scale = 16.0 if cfg.int_mode else 1.0
    model = MLP(cfg.input_dim, cfg.hidden_dim, cfg.output_dim, scale=scale)
    mx.eval(model.parameters())
    params_paths, params_vals_any = flatten_params(model.trainable_parameters())
    params_vals = [p for p in params_vals_any if isinstance(p, mx.array)]

    if cfg.int_mode:
        # Scale up and integerize
        params_vals = [
            mx.clip((p * scale).round(), -127, 127).astype(mx.int8) for p in params_vals
        ]
        # Store as float32 in model to make MLX operations happy
        params_vals_storage = [p.astype(mx.float32) for p in params_vals]
        model.update(unflatten(params_paths, params_vals_storage))
        # Use float versions for loop calculation to avoid casting issues
        params_vals = params_vals_storage

    pop = (
        cfg.pop_size
        if cfg.pop_size % 2 == 0 or not cfg.antithetic
        else cfg.pop_size + 1
    )
    key = mx.random.key(cfg.seed + (123 if mode == "low_rank" else 321))

    losses = []
    accs = []
    evals = 0
    start = time.perf_counter()

    for _ in range(cfg.steps):
        if evals >= cfg.eval_budget:
            break
        k, key = mx.random.split(key)

        # Get structured perturbations (A, B) for low_rank, or Noise for full
        if mode == "low_rank":
            # Use structured generator
            structured_perts = _generate_low_rank_noise_v2_structured(
                [p.shape for p in params_vals],
                pop,
                cfg.sigma,
                cfg.rank,
                k,
                cfg.antithetic,
            )
        else:
            # For full, we generate full noise matrices
            structured_perts = _generate_full_noise_v2(
                [p.shape for p in params_vals], pop, cfg.sigma, k, cfg.antithetic
            )
            # Reshape to match structured access pattern (list of lists)
            # _generate_full_noise_v2 returns [shape_idx][worker]
            # We want [worker][shape_idx]
            # Actually _generate_full_noise_v2 returns list of [pop, *shape] tensors usually in other codes,
            # but here it returns [shape_idx][worker_idx]. Wait, let's check `_generate_full_noise_v2` again.

            # Current implementation:
            # def _generate_full_noise_v2(...):
            #    noises = [mx.zeros((pop, *shape)) for shape in shapes]
            #    ...
            #    return noises (list of tensors, one per param)

            # We need to transpose to iterate by worker easily or just index.
            pass

        rewards = []
        for j in range(pop):
            # Apply perturbation
            if mode == "low_rank":
                # Use the efficient forward pass if we implemented it fully,
                # BUT for this script to be identical to previous logic (just cleaner),
                # we can just manually reconstruct the weight perturbation here.
                # This avoids rewriting the entire loop structure while proving the point.

                perturbed_params = []
                for i, p in enumerate(params_vals):
                    noise_struct = structured_perts[j][i]
                    if isinstance(noise_struct, tuple):
                        A, B = noise_struct
                        # Reconstruct Delta = scale * A @ B.T
                        # A is already scaled by sigma/sqrt(rank) in generator?
                        # Yes: A * scale.

                        # Wait, MLX doesn't like explicit @ for 1D vectors or mixed dims sometimes.
                        # A: (rows, rank), B: (cols, rank). A @ B.T -> (rows, cols).
                        delta = A @ B.T
                    else:
                        delta = noise_struct

                    perturbed_params.append(p + delta)

                model.update(unflatten(params_paths, perturbed_params))
            else:
                # Full mode
                # structured_perts is list of (pop, *shape) tensors
                perturbed_params = [
                    params_vals[i] + structured_perts[i][j]
                    for i in range(len(params_vals))
                ]
                model.update(unflatten(params_paths, perturbed_params))

            logits = model(X_train)
            rewards.append(-cross_entropy_logits(logits, y_train).item())

        # Restore center
        model.update(unflatten(params_paths, params_vals))

        rewards = np.array(rewards)

        if cfg.antithetic:
            paired_scores = _pairwise_scores(rewards, fast_fitness=cfg.fast_fitness)
            pair_indices = list(range(0, pop, 2))
        else:
            paired_scores = rewards
            if cfg.fast_fitness:
                pass
            pair_indices = list(range(pop))

        updates = []
        for idx_param, param in enumerate(params_vals):
            accumulator = mx.zeros_like(param.astype(mx.float32))

            if cfg.antithetic:
                for idx, score in zip(pair_indices, paired_scores):
                    if cfg.update_threshold > 0 and abs(score) < cfg.update_threshold:
                        continue

                    if mode == "low_rank":
                        noise_struct = structured_perts[idx][idx_param]
                        if isinstance(noise_struct, tuple):
                            A, B = noise_struct
                            pert = A @ B.T
                        else:
                            pert = noise_struct
                    else:
                        pert = structured_perts[idx_param][idx]

                    step = mx.sign(pert) if cfg.use_sign_updates else pert
                    accumulator = accumulator + step * score
            else:
                mean_reward = rewards.mean()
                for j in range(pop):
                    score = rewards[j] - mean_reward
                    if mode == "low_rank":
                        noise_struct = structured_perts[j][idx_param]
                        if isinstance(noise_struct, tuple):
                            A, B = noise_struct
                            pert = A @ B.T
                        else:
                            pert = noise_struct
                    else:
                        pert = structured_perts[idx_param][j]

                    step = mx.sign(pert) if cfg.use_sign_updates else pert
                    accumulator = accumulator + step * score

            if cfg.int_mode:
                step = mx.zeros_like(accumulator)
                if cfg.z_threshold > 0:
                    mask = mx.abs(accumulator) >= cfg.z_threshold
                    step = mx.where(mask, mx.sign(accumulator), step)
                else:
                    step = mx.sign(accumulator)

                updated = param + step
                updated = mx.clip(updated, -127, 127)
            else:
                if cfg.z_threshold > 0:
                    mask = mx.abs(accumulator) >= cfg.z_threshold
                    accumulator = mx.where(mask, accumulator, 0)
                updated = param + cfg.lr_es * accumulator
                updated = mx.clip(updated, -cfg.param_clip, cfg.param_clip)

            updates.append(updated)

        params_vals = updates
        model.update(unflatten(params_paths, params_vals))
        mx.eval(model.parameters())
        evals += pop

        # Eval
        losses.append(cross_entropy_logits(model(X_train), y_train).item())
        accs.append(accuracy(model(X_test), y_test).item())

    duration = time.perf_counter() - start
    return {
        "losses": losses,
        "accs": accs,
        "time": duration,
        "evals": evals,
        "pop": pop,
    }


def run_all(cfg: ExperimentConfig) -> Dict[str, Dict]:
    X_train, y_train, X_test, y_test = make_data(cfg)
    results: Dict[str, Dict] = {}
    results["adam"] = run_adam(X_train, y_train, X_test, y_test, cfg)
    results["es_full"] = run_es("full", X_train, y_train, X_test, y_test, cfg)
    results["eggroll_low_rank"] = run_es(
        "low_rank", X_train, y_train, X_test, y_test, cfg
    )
    bw_ratio = (
        cfg.rank
        * (cfg.hidden_dim + cfg.output_dim)
        / ((cfg.hidden_dim * cfg.output_dim) + (cfg.input_dim * cfg.hidden_dim))
    )
    for v in results.values():
        v["bandwidth_ratio"] = bw_ratio
    return results


def render_table(results: Dict[str, Dict], cfg: ExperimentConfig) -> str:
    lines = [
        f"Two-moons MLP: input_dim={cfg.input_dim}, hidden_dim={cfg.hidden_dim}, output_dim={cfg.output_dim}",
        f"Config: {asdict(cfg)}",
        "",
        "Optimizer\tFinal Loss\tFinal Acc\tEvals\tTime (s)\tBandwidth r",
    ]
    for name, res in results.items():
        final_loss = res["losses"][-1] if res["losses"] else float("nan")
        final_acc = res["accs"][-1] if res["accs"] else float("nan")
        lines.append(
            f"{name}\t{final_loss:.4f}\t{final_acc:.4f}\t{res['evals']}\t{res['time']:.3f}\t{res['bandwidth_ratio']:.3f}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="EGGROLL vs ES vs Adam on MLP (two-moons)"
    )
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval_budget", type=int, default=6400)
    parser.add_argument("--pop_size", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lr_es", type=float, default=0.05)
    parser.add_argument("--lr_adam", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--antithetic", action="store_true", default=True)
    parser.add_argument("--no-antithetic", action="store_false", dest="antithetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast_fitness", action="store_true", default=True)
    parser.add_argument("--update_threshold", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--use_sign_updates", action="store_true", default=True)
    parser.add_argument(
        "--no_use_sign_updates", action="store_false", dest="use_sign_updates"
    )
    parser.add_argument("--param_clip", type=float, default=1.0)
    parser.add_argument("--int_mode", action="store_true", default=False)
    parser.add_argument("--no_int_mode", action="store_false", dest="int_mode")
    parser.add_argument("--z_threshold", type=float, default=0.0)
    args = parser.parse_args()
    cfg = ExperimentConfig(
        samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        input_dim=2,
        hidden_dim=args.hidden_dim,
        output_dim=2,
        steps=args.steps,
        eval_budget=args.eval_budget,
        pop_size=args.pop_size,
        sigma=args.sigma,
        lr_es=args.lr_es,
        lr_adam=args.lr_adam,
        rank=args.rank,
        antithetic=args.antithetic,
        seed=args.seed,
        fast_fitness=args.fast_fitness,
        update_threshold=args.update_threshold,
        dtype=args.dtype,
        use_sign_updates=args.use_sign_updates,
        param_clip=args.param_clip,
        int_mode=args.int_mode,
        z_threshold=args.z_threshold,
    )
    results = run_all(cfg)
    print(render_table(results, cfg))


if __name__ == "__main__":
    main()
