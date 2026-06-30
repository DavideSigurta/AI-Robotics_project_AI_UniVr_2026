"""Plot training curves: reward comparison between PPO and Dreamer.

Generates a two-panel PNG comparing PPO and Dreamer reward curves
over the full training horizon and zoomed to 0-50k env steps.

Usage:
    python analysis/plot_training.py --dreamer-csv path --ppo-csv path --output path
"""

import argparse
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rolling_average(data, window: int = 20):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def load_csv(path: str):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    env_steps = np.array([int(r["envSteps"]) for r in rows], dtype=float)
    rewards = np.array([float(r["totalReward"]) for r in rows], dtype=float)
    return env_steps, rewards


def make_plots(dreamer_csv: str, ppo_csv: str | None, output: str):
    d_steps, d_rewards = load_csv(dreamer_csv)
    p_steps, p_rewards = None, None
    if ppo_csv and os.path.isfile(ppo_csv):
        p_steps, p_rewards = load_csv(ppo_csv)
        print(f"PPO: {len(p_steps)} points")
    print(f"Dreamer: {len(d_steps)} points, env_steps=[{d_steps[0]:.0f}, {d_steps[-1]:.0f}]")

    window = 20
    d_smooth = rolling_average(d_rewards, window)
    d_smooth_steps = d_steps[:len(d_smooth)]
    if p_steps is not None:
        p_smooth = rolling_average(p_rewards, window)
        p_smooth_steps = p_steps[:len(p_smooth)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Panel 1: Full
    if p_steps is not None:
        ax1.plot(p_steps, p_rewards, alpha=0.12, color="C0", linewidth=0.5)
        ax1.plot(p_smooth_steps, p_smooth, color="C0", linewidth=1.5, label=f"PPO (smooth w={window})")
    ax1.plot(d_steps, d_rewards, alpha=0.15, color="C1", linewidth=0.5)
    ax1.plot(d_smooth_steps, d_smooth, color="C1", linewidth=1.5, label=f"Dreamer (smooth w={window})")
    ax1.set_xlabel("Environment steps"); ax1.set_ylabel("Episode reward")
    ax1.set_title("Training reward comparison (full)")
    ax1.legend(fontsize=11); ax1.grid(True, alpha=0.3)
    xmax = max(d_steps[-1], p_steps[-1] if p_steps is not None else d_steps[-1])
    ax1.set_xlim(0, xmax)

    # Panel 2: Zoom
    zoom_max = 50000
    if p_steps is not None:
        mask = p_steps <= zoom_max
        if mask.any():
            ax2.plot(p_steps[mask], p_rewards[mask], alpha=0.2, color="C0", linewidth=0.5)
            z = rolling_average(p_rewards[mask], window)
            ax2.plot(p_steps[mask][:len(z)], z, color="C0", linewidth=1.5, label="PPO")
    mask = d_steps <= zoom_max
    if mask.any():
        ax2.plot(d_steps[mask], d_rewards[mask], alpha=0.2, color="C1", linewidth=0.5)
        z = rolling_average(d_rewards[mask], window)
        ax2.plot(d_steps[mask][:len(z)], z, color="C1", linewidth=1.5, label="Dreamer")
    ax2.set_xlabel("Environment steps"); ax2.set_ylabel("Episode reward")
    ax2.set_title(f"Zoom: first {zoom_max//1000}k env steps")
    ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, zoom_max)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved to {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreamer-csv", required=True)
    parser.add_argument("--ppo-csv", default=None)
    parser.add_argument("--output", default="results/figures/training_comparison.png")
    args = parser.parse_args()
    make_plots(args.dreamer_csv, args.ppo_csv, args.output)


if __name__ == "__main__":
    main()
