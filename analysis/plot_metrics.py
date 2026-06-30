"""Comparative metrics bar charts: PPO vs Dreamer.

Generates a PNG figure with side-by-side bar charts comparing goal rate
and crash rate across curriculum stages, plus a summary table.

Usage:
    python analysis/plot_metrics.py --output results/figures/evaluation_comparison.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Evaluation results (100 episodes per stage, seed=123, deterministic)
EVAL_DATA = {
    "PPO": {
        "goal_rate":  [0.91, 0.77, 0.69],
        "crash_rate": [0.09, 0.23, 0.27],
        "timeout_rate": [0.00, 0.00, 0.04],
        "mean_reward": [None, None, None],
    },
    "Dreamer": {
        "goal_rate":  [0.85, 0.71, 0.66],
        "crash_rate": [0.15, 0.29, 0.34],
        "timeout_rate": [0.00, 0.00, 0.00],
        "mean_reward": [7.378, 3.972, 2.391],
    },
}

STAGES = ["Stage 0\n(easy)", "Stage 1\n(medium)", "Stage 2\n(hard)"]
COLORS = {"PPO": "#4C72B0", "Dreamer": "#DD8452"}


def make_plots(output: str):
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30)

    x = np.arange(len(STAGES))
    bar_width = 0.30

    # ── Panel 1: Goal rate ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (model, data) in enumerate(EVAL_DATA.items()):
        offset = (i - 0.5) * bar_width
        bars = ax1.bar(x + offset, data["goal_rate"], bar_width,
                       color=COLORS[model], label=model, alpha=0.85)
        for bar, val in zip(bars, data["goal_rate"]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(STAGES)
    ax1.set_ylabel("Goal rate")
    ax1.set_title("Goal rate by curriculum stage")
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, axis="y", alpha=0.3)

    # ── Panel 2: Crash rate ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (model, data) in enumerate(EVAL_DATA.items()):
        offset = (i - 0.5) * bar_width
        bars = ax2.bar(x + offset, data["crash_rate"], bar_width,
                       color=COLORS[model], label=model, alpha=0.85)
        for bar, val in zip(bars, data["crash_rate"]):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(STAGES)
    ax2.set_ylabel("Crash rate")
    ax2.set_title("Crash rate by curriculum stage")
    ax2.legend()
    ax2.set_ylim(0, 0.45)
    ax2.grid(True, axis="y", alpha=0.3)

    # ── Panel 3: Summary table (bottom, spans full width) ──────────────
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")

    col_labels = ["Model", "Stage 0", "Stage 1", "Stage 2", "Avg goal", "Env steps"]
    cell_text = []
    for model, data in EVAL_DATA.items():
        avg_goal = np.mean(data["goal_rate"])
        env_steps = "250k" if model == "PPO" else "~48k"
        row = [
            model,
            f'{data["goal_rate"][0]:.0%}',
            f'{data["goal_rate"][1]:.0%}',
            f'{data["goal_rate"][2]:.0%}',
            f"{avg_goal:.0%}",
            env_steps,
        ]
        cell_text.append(row)

    table = ax3.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f0f0f0" if i % 2 == 1 else "white")

    ax3.set_title("Summary comparison", fontsize=13, pad=20)

    # ── Save ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot comparative metrics bar charts.")
    parser.add_argument("--output", default="results/figures/evaluation_comparison.png",
                        help="Output PNG path.")
    args = parser.parse_args()
    make_plots(args.output)


if __name__ == "__main__":
    main()


def plot_success_rate(log_path, output_path="plots/success_rate.html"):
    """Plot the episode success rate."""
    # TODO: implement
    pass
