#!/usr/bin/env python3
"""Plot one clear model-level temporal-deployment trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DISPLAY_K = (1, 5, 10)
TARGET_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
)


def model_trajectories(payload: dict, metric: str = "medae") -> dict[str, list[float]]:
    """Return target-level seed-median errors in the artifact's fixed order."""

    return {
        target: [
            float(payload["summary_by_target"][target]["by_k"][str(k)][metric]["median"])
            for k in DISPLAY_K
        ]
        for target in payload["config"]["target_model_ids"]
    }


def _nonoverlapping_label_positions(values: list[float], minimum_gap: float = 0.10) -> list[float]:
    """Separate direct labels while preserving their vertical ordering."""

    positions = [0.0] * len(values)
    previous = -np.inf
    for index in np.argsort(values):
        position = max(values[index], previous + minimum_gap)
        positions[index] = float(position)
        previous = position
    return positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "experiments" / "temporal_deployment_rank1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "temporal_deployment_rank1.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    config = payload["config"]
    targets = list(config["target_model_ids"])
    trajectories = model_trajectories(payload)
    if len(targets) != 7 or int(config["n_seeds"]) != 10:
        raise ValueError("temporal figure requires the pinned seven-target, ten-seed artifact")
    if tuple(config["k_values"]) != DISPLAY_K:
        raise ValueError(f"temporal figure requires k={DISPLAY_K}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
        }
    )
    fig, ax = plt.subplots(figsize=(9.6, 5.7))
    x = np.asarray(DISPLAY_K, dtype=float)

    final_values = []
    for target, color in zip(targets, TARGET_COLORS):
        values = np.asarray(trajectories[target], dtype=float)
        final_values.append(float(values[-1]))
        ax.plot(x, values, "o-", color=color, lw=1.8, ms=5.5, alpha=0.88)

    label_y = _nonoverlapping_label_positions(final_values)
    for target, color, actual_y, text_y in zip(targets, TARGET_COLORS, final_values, label_y):
        ax.plot([10.0, 10.35], [actual_y, text_y], color=color, lw=0.9, alpha=0.75)
        ax.text(10.42, text_y, target, color=color, va="center", fontsize=9)

    matrix = np.asarray([trajectories[target] for target in targets], dtype=float)
    cohort_median = np.median(matrix, axis=0)
    ax.plot(
        x,
        cohort_median,
        "D--",
        color="#303030",
        lw=1.6,
        ms=5,
        label="Cohort median",
        zorder=5,
    )

    ax.set(
        xlim=(0.6, 12.65),
        ylim=(0, max(2.3, max(label_y) + 0.15)),
        xticks=DISPLAY_K,
        xlabel="Revealed target evaluations (k)",
        ylabel="Median absolute error (normalized-score points)",
    )
    ax.grid(axis="y", color="#D7D7D7", alpha=0.65, lw=0.8)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Temporal deployment with prior-only training: seven 2025 pathology targets",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.035,
        "Each trajectory is the target-model median across ten probe seeds. "
        "MedAE includes k exact revealed cells plus all supported hidden predictions.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.11, right=0.80, bottom=0.18, top=0.86)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
