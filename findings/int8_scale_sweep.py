import csv
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import mlx.core as mx

from eggroll_mlx import NoiseConfig, do_mm, generate_big_rand


def run_sweep():
    mags = [1, 2, 4, 8, 16, 32]
    fixed_points = [2, 3, 4]
    sigma_shifts = [0, 2, 4]

    rows = []
    proc = psutil.Process()
    for mag, fp, ss in itertools.product(mags, fixed_points, sigma_shifts):
        cfg = NoiseConfig(fixed_point=fp, sigma_shift=ss, rank=1)
        x = mx.full((1, 2), mag, dtype=mx.int8)
        w = mx.full((1, 2), mag, dtype=mx.int8)
        big_rand = generate_big_rand(1024, seed=0, fixed_point=cfg.fixed_point, dtype=mx.int8)
        out = do_mm(cfg, x, w, big_rand, epoch=0, thread_id=0, base_seed=0)
        rows.append(
            {
                "magnitude": mag,
                "fixed_point": fp,
                "sigma_shift": ss,
                "max_output": int(mx.max(out).item()),
                "rss_mb": proc.memory_info().rss / (1024 * 1024),
            }
        )
    return rows


def write_csv(rows, path: Path):
    fieldnames = ["magnitude", "fixed_point", "sigma_shift", "max_output", "rss_mb"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, path: Path):
    mags = sorted(set(r["magnitude"] for r in rows))
    sigma_shifts = sorted(set(r["sigma_shift"] for r in rows))
    fixed_points = sorted(set(r["fixed_point"] for r in rows))

    fig, axs = plt.subplots(1, len(fixed_points), figsize=(12, 4), sharey=True)
    for ax, fp in zip(axs, fixed_points):
        grid = np.zeros((len(mags), len(sigma_shifts)), dtype=np.int32)
        for r in rows:
            if r["fixed_point"] != fp:
                continue
            i = mags.index(r["magnitude"])
            j = sigma_shifts.index(r["sigma_shift"])
            grid[i, j] = r["max_output"]
        im = ax.imshow(grid, origin="lower", cmap="viridis", vmin=0, vmax=127)
        for i, mag in enumerate(mags):
            for j, ss in enumerate(sigma_shifts):
                ax.text(j, i, f"{grid[i, j]}", ha="center", va="center", color="white" if grid[i, j] > 64 else "black")
        ax.set_title(f"fixed_point={fp}")
        ax.set_xticks(range(len(sigma_shifts)), sigma_shifts)
        ax.set_xlabel("sigma_shift")
        ax.set_yticks(range(len(mags)), mags)
        ax.set_ylabel("input magnitude")
    fig.colorbar(im, ax=axs, shrink=0.8, label="max output (post-scale int8)")
    fig.suptitle("Max output vs input magnitude (int8 path). Annotated values; 127 = clamp.")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def plot_bars(rows, path: Path):
    mags = sorted(set(r["magnitude"] for r in rows))
    sigma_shifts = sorted(set(r["sigma_shift"] for r in rows))
    fixed_points = sorted(set(r["fixed_point"] for r in rows))
    colors = {0: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c"}

    fig, axs = plt.subplots(1, len(fixed_points), figsize=(12, 4), sharey=True)
    width = 0.2
    for ax, fp in zip(axs, fixed_points):
        for idx, ss in enumerate(sigma_shifts):
            vals = []
            for mag in mags:
                match = next(r for r in rows if r["fixed_point"] == fp and r["sigma_shift"] == ss and r["magnitude"] == mag)
                vals.append(match["max_output"])
            offsets = np.array(range(len(mags))) + (idx - len(sigma_shifts) / 2) * width + width / 2
            ax.bar(offsets, vals, width=width, color=colors[ss], label=f"sigma_shift={ss}")
        ax.set_xticks(range(len(mags)))
        ax.set_xticklabels(mags)
        ax.set_xlabel("input magnitude")
        ax.set_title(f"fixed_point={fp}")
    axs[0].set_ylabel("max output (post-scale int8)")
    axs[-1].legend()
    fig.suptitle("Max output by magnitude (grouped bars by sigma_shift)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    out_dir = Path(__file__).parent
    rows = run_sweep()
    write_csv(rows, out_dir / "int8_scale_sweep.csv")
    plot(rows, out_dir / "int8_scale_sweep.png")
    plot_bars(rows, out_dir / "int8_scale_bars.png")
    print("Wrote:", out_dir / "int8_scale_sweep.csv")
    print("Saved plot:", out_dir / "int8_scale_sweep.png")
    print("Saved bars plot:", out_dir / "int8_scale_bars.png")


if __name__ == "__main__":
    main()
