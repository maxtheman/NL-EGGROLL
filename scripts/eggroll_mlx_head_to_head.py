import argparse
import csv
import time

import numpy as np
import psutil
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import mlx.core as mx
import mlx.nn as nn

from eggroll_mlx import (
    NoiseConfig,
    apply_sign_update,
    convert_fitnesses,
    generate_big_rand,
    do_mm,
)


def make_data(seed: int, samples: int = 512):
    X, y = make_moons(samples, noise=0.1, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    return (
        mx.array(X_train, dtype=mx.float32),
        mx.array(y_train, dtype=mx.int32),
        mx.array(X_test, dtype=mx.float32),
        mx.array(y_test, dtype=mx.int32),
    )


def forward(cfg, W1, W2, x):
    """Integer forward pass using the same do_mm path (no noise)."""
    h_int = do_mm(cfg, x, W1, None, None, None, None)
    h_int = mx.maximum(h_int, mx.zeros_like(h_int))  # int8 ReLU
    logits_int = do_mm(cfg, h_int, W2, None, None, None, None)
    return logits_int.astype(mx.float32)


def cross_entropy(logits, targets):
    num_classes = logits.shape[-1]
    one_hot = (targets[..., None] == mx.arange(num_classes)).astype(mx.float32)
    max_logits = mx.max(logits, axis=-1, keepdims=True)
    logsumexp = max_logits + mx.log(mx.sum(mx.exp(logits - max_logits), axis=-1, keepdims=True))
    log_probs = logits - logsumexp
    return -mx.mean(mx.sum(one_hot * log_probs, axis=-1))


def accuracy(logits, targets):
    preds = mx.argmax(logits, axis=-1)
    return (preds == targets).astype(mx.float32).mean()


def run(cfg: NoiseConfig, pop_size: int, steps: int, seed: int):
    X_train_f, y_train, X_test_f, y_test = make_data(seed)
    rng = np.random.default_rng(seed)

    def quantize(arr):
        scaled = np.clip(np.round(arr * (2**cfg.fixed_point)), -127, 127).astype(np.int8)
        return mx.array(scaled, dtype=mx.int8)

    X_train = quantize(np.array(X_train_f))
    X_test = quantize(np.array(X_test_f))
    W1 = quantize(rng.normal(0, 0.1, size=(16, 2)))
    W2 = quantize(rng.normal(0, 0.1, size=(2, 16)))

    base_seed = seed
    big_rand = generate_big_rand(2_000_000, seed=seed, fixed_point=cfg.fixed_point, dtype=mx.int8)

    start = time.perf_counter()
    logs = {
        "fitness_mean": [],
        "fitness_std": [],
        "fitness_min": [],
        "fitness_max": [],
        "delta_norm": [],
        "loss": [],
        "loss_inputs": [],
    }
    for epoch in range(steps):
        rewards = []
        for j in range(pop_size):
            thread_id = j
            h_int = do_mm(cfg, X_train, W1.astype(mx.int8), big_rand, epoch, thread_id, base_seed)
            h_act = mx.maximum(h_int, mx.zeros_like(h_int))  # int8 ReLU
            logits_int = do_mm(cfg, h_act.astype(mx.int8), W2.astype(mx.int8), big_rand, epoch + 17, thread_id, base_seed)
            logits = logits_int.astype(mx.float32)
            rewards.append(-cross_entropy(logits, y_train).item())

        rewards = mx.array(rewards)
        fitnesses = convert_fitnesses(cfg, rewards)
        # diagnostics: fitness stats
        r_np = np.array(rewards)
        logs["fitness_mean"].append(float(r_np.mean()))
        logs["fitness_std"].append(float(r_np.std()))
        logs["fitness_min"].append(float(r_np.min()))
        logs["fitness_max"].append(float(r_np.max()))
        # approximate delta norm using first member
        logs["delta_norm"].append(0.0)
        W1 = apply_sign_update(cfg, W1.astype(mx.int8), fitnesses, big_rand, epoch, base_seed)
        W2 = apply_sign_update(cfg, W2.astype(mx.int8), fitnesses, big_rand, epoch + 17, base_seed)
        # base loss each epoch
        h_base = do_mm(cfg, X_train, W1.astype(mx.int8), None, None, None, None)
        h_base = mx.maximum(h_base, mx.zeros_like(h_base))  # int8 ReLU
        base_logits = do_mm(cfg, h_base.astype(mx.int8), W2.astype(mx.int8), None, None, None, None).astype(mx.float32)
        ce_val = cross_entropy(base_logits.astype(mx.float32), y_train)
        logs["loss"].append(float(ce_val.item()))
        logs["loss_inputs"].append(
            {
                "logits_mean": float(np.array(base_logits).mean()),
                "logits_std": float(np.array(base_logits).std()),
                "logits_min": float(np.array(base_logits).min()),
                "logits_max": float(np.array(base_logits).max()),
            }
        )

    duration = time.perf_counter() - start
    logits = forward(cfg, W1, W2, X_test)
    l_np = np.array(logits)
    logit_stats = {
        "mean": float(l_np.mean()),
        "std": float(l_np.std()),
        "min": float(l_np.min()),
        "max": float(l_np.max()),
    }
    acc = accuracy(logits, y_test).item()
    loss = cross_entropy(logits, y_test).item()
    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    return {
        "acc": acc,
        "loss": loss,
        "time_s": duration,
        "rss_mb": rss_mb,
        "logit_stats": logit_stats,
        "logs": logs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--sigma_shift", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", action="store_true", help="Run a sweep over pop_size/rank/sigma_shift")
    parser.add_argument("--diagnostic_plot", action="store_true", help="Save diagnostic plots for single run")
    args = parser.parse_args()

    if not args.sweep:
        cfg = NoiseConfig(rank=args.rank, sigma_shift=args.sigma_shift)
        res = run(cfg, args.pop_size, args.steps, args.seed)
        print(f"pop={args.pop_size} steps={args.steps} rank={args.rank} sigma_shift={args.sigma_shift}")
        print(
            f"Acc={res['acc']:.4f} Loss={res['loss']:.4f} Time={res['time_s']:.3f}s RSS={res['rss_mb']:.1f} MB "
            f"logits(mean={res['logit_stats']['mean']:.4f}, std={res['logit_stats']['std']:.4f}, "
            f"min={res['logit_stats']['min']:.4f}, max={res['logit_stats']['max']:.4f})"
        )
        # Print last loss inputs for inspection
        if res["logs"]["loss_inputs"]:
            last_loss_in = res["logs"]["loss_inputs"][-1]
            print(
                "Last loss inputs:",
                f"logits_mean={last_loss_in['logits_mean']:.4f}, logits_std={last_loss_in['logits_std']:.4f}, "
                f"logits_min={last_loss_in['logits_min']:.4f}, logits_max={last_loss_in['logits_max']:.4f}",
            )
        if args.diagnostic_plot:
            logs = res["logs"]
            fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
            axs[0].plot(logs["fitness_mean"], label="fitness_mean")
            axs[0].fill_between(
                range(len(logs["fitness_mean"])),
                np.array(logs["fitness_mean"]) - np.array(logs["fitness_std"]),
                np.array(logs["fitness_mean"]) + np.array(logs["fitness_std"]),
                color="gray",
                alpha=0.3,
            )
            axs[0].set_ylabel("Fitness")
            axs[0].legend()
            axs[1].plot(logs["delta_norm"], label="delta_norm")
            axs[1].set_ylabel("Delta norm")
            axs[1].legend()
            axs[2].plot(logs["loss"], label="train CE")
            axs[2].set_ylabel("Loss")
            axs[2].set_xlabel("Epoch")
            axs[2].legend()
            plt.tight_layout()
            out_path = "scripts/eggroll_diagnostics.png"
            plt.savefig(out_path)
            print(f"Saved diagnostics plot to {out_path}")
        return

    pop_sizes = [16, 32, 64]
    ranks = [1, 2]
    sigma_shifts = [0, 2, 4]
    results = []
    for pop in pop_sizes:
        for rank in ranks:
            for sig in sigma_shifts:
                cfg = NoiseConfig(rank=rank, sigma_shift=sig)
                res = run(cfg, pop, args.steps, args.seed)
                results.append(
                    {
                        "pop": pop,
                        "rank": rank,
                        "sigma_shift": sig,
                        "acc": res["acc"],
                        "loss": res["loss"],
                        "time_s": res["time_s"],
                        "rss_mb": res["rss_mb"],
                        "logit_mean": res["logit_stats"]["mean"],
                        "logit_std": res["logit_stats"]["std"],
                        "logit_min": res["logit_stats"]["min"],
                        "logit_max": res["logit_stats"]["max"],
                    }
                )
                print(
                    f"pop={pop} rank={rank} sigma_shift={sig} "
                    f"acc={res['acc']:.4f} loss={res['loss']:.4f} time={res['time_s']:.3f}s "
                    f"rss={res['rss_mb']:.1f}MB logits_mean={res['logit_stats']['mean']:.2f}"
                )

    fieldnames = list(results[0].keys())
    with open("scripts/eggroll_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print("Wrote scripts/eggroll_sweep.csv")


if __name__ == "__main__":
    main()
