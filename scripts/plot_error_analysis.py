#!/usr/bin/env python3
"""Plot BenchPress-style pathology predictability-factor panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAGENTA = "#D33682"
BLUE = "#268BD2"
TEAL = "#2AA198"
VIOLET = "#6C71C4"
CHARCOAL = "#002B36"


def _panel(ax, rows, x_key, y_key, title, x_label, color):
    x = np.asarray([float(row[x_key]) for row in rows])
    y = np.asarray([float(row[y_key]) for row in rows])
    valid = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[valid], y[valid], s=22, color=color, alpha=0.72, edgecolors="white", linewidths=0.3)
    if valid.sum() >= 2 and np.std(x[valid]) > 1e-12:
        slope, intercept = np.polyfit(x[valid], y[valid], 1)
        grid = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(grid, slope * grid + intercept, color=CHARCOAL, linewidth=1.4)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("MedAE (normalized points)")
    ax.grid(color="#EEF1F4", linewidth=0.8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "experiments" / "error_analysis_rank1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "predictability_factors_rank1")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    evaluation = data["evaluation_analysis"]["rows"]
    model = data["model_analysis"]["rows"]
    plt.rcParams.update({"font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    specs = [
        (evaluation, "rank1_r2", "A  Evaluation low-rank fit", "Rank-1 reconstruction R²", MAGENTA),
        (evaluation, "median_score", "B  Evaluation score level", "Median observed score", BLUE),
        (evaluation, "score_std", "C  Evaluation score spread", "Observed score SD", TEAL),
        (model, "rank1_r2", "D  Model low-rank fit", "Rank-1 reconstruction R²", VIOLET),
        (model, "n_obs", "E  Model evidence", "Observed evaluation count", MAGENTA),
        (model, "best_peer_abs_r", "F  Best model peer", "Best peer |Pearson r|", BLUE),
    ]
    for ax, (rows, x_key, title, label, color) in zip(axes.flat, specs):
        _panel(ax, rows, x_key, "medae", title, label, color)
    fig.suptitle("What makes pathology scores predictable?", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
