#!/usr/bin/env python3
"""Render the primary cell-level interaction-rank validation figure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pathopress-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAGENTA = "#D33682"
BLUE = "#268BD2"
GRAY = "#7C8790"
CHARCOAL = "#333333"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def validation_plot_data(result: dict[str, object]) -> dict[str, object]:
    """Extract the explicitly cell-level rank-selection quantities."""

    configuration = result["configuration"]
    matrix = result["matrix"]
    by_rank = result["by_rank"]
    ranks = [int(rank) for rank in configuration["ranks"]]
    selected_rank = int(configuration["prediction_rank"])
    selected = by_rank[str(selected_rank)]
    pooled_n = int(selected["pooled"]["n"])
    return {
        "ranks": ranks,
        "pooled_medae": [float(by_rank[str(rank)]["pooled"]["medae"]) for rank in ranks],
        "fold_q1": [float(by_rank[str(rank)]["fold_medae"]["q1"]) for rank in ranks],
        "fold_q3": [float(by_rank[str(rank)]["fold_medae"]["q3"]) for rank in ranks],
        "selected_rank": selected_rank,
        "selected_pooled_medae": float(selected["pooled"]["medae"]),
        "baseline_medae": float(result["column_median_baseline"]["pooled"]["medae"]),
        "unique_cells": int(matrix["n_observed"]),
        "prediction_instances": pooled_n,
        "n_models": int(matrix["n_models"]),
        "n_evaluations": int(matrix["n_evaluations"]),
        "n_seeds": int(configuration["n_seeds"]),
        "n_folds": int(configuration["n_folds"]),
    }


def build_rank_selection_figure(result: dict[str, object]):
    values = validation_plot_data(result)
    ranks = np.asarray(values["ranks"], dtype=int)
    pooled = np.asarray(values["pooled_medae"], dtype=float)
    q1 = np.asarray(values["fold_q1"], dtype=float)
    q3 = np.asarray(values["fold_q3"], dtype=float)
    selected_rank = int(values["selected_rank"])
    selected_error = float(values["selected_pooled_medae"])
    baseline = float(values["baseline_medae"])

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.fill_between(ranks, q1, q3, color=BLUE, alpha=0.16, label="Fold MedAE IQR")
    ax.plot(ranks, pooled, "o-", color=BLUE, lw=2.4, ms=5.5, label="Pooled OOF MedAE")
    ax.axhline(
        baseline,
        color=GRAY,
        ls="--",
        lw=1.7,
        label=f"Column-median baseline ({baseline:.2f})",
    )
    ax.scatter(
        [selected_rank],
        [selected_error],
        marker="*",
        s=230,
        color=MAGENTA,
        edgecolor=CHARCOAL,
        linewidth=0.7,
        zorder=5,
    )
    ax.annotate(
        f"Selected rank {selected_rank}\nMedAE {selected_error:.3f}",
        (selected_rank, selected_error),
        xytext=(18, 20),
        textcoords="offset points",
        color=MAGENTA,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": MAGENTA, "lw": 1.0},
    )
    ax.set(
        xlabel="Latent interaction rank",
        ylabel="Median absolute error (normalized-score points)",
        xticks=ranks,
        xlim=(ranks.min() - 0.25, ranks.max() + 0.25),
    )
    ax.set_ylim(min(q1) - 0.08, baseline + 0.10)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Cell-level cross-validation selects interaction rank 1", fontweight="bold")

    semantics = (
        f"{values['n_models']} models × {values['n_evaluations']} protocols; "
        f"{values['unique_cells']:,} unique reported cells; "
        f"{values['prediction_instances']:,} repeated held-out predictions from "
        f"{values['n_seeds']} seeds × {values['n_folds']} folds. "
        "Cell-level validation: other scores from the same model may remain visible."
    )
    fig.text(0.5, 0.015, semantics, ha="center", va="bottom", fontsize=8.3, color=CHARCOAL)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.22, top=0.88)
    return fig, ax


def save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "benchpress_style_results.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    apply_style()
    fig, _ = build_rank_selection_figure(result)
    prediction_rank = int(result["configuration"]["prediction_rank"])
    save(fig, args.output_dir, f"benchpress_style_validation_rank{prediction_rank}")
    print(args.output_dir / f"benchpress_style_validation_rank{prediction_rank}.{{png,pdf}}")


if __name__ == "__main__":
    main()
