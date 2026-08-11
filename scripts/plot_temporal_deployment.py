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
# Darkened from the pastel Vega set: these hues double as the direct-label text
# colour, and the light pink/teal originals fell below readable contrast on white.
TARGET_COLORS = (
    "#2F5D8C",
    "#C05A00",
    "#2E7D32",
    "#C62828",
    "#00796B",
    "#7B4E9E",
    "#B0306B",
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


def build_temporal_figure(payload: dict):
    """Build the single-panel, target-labelled temporal figure."""

    config = payload["config"]
    targets = list(config["target_model_ids"])
    trajectories = model_trajectories(payload)
    if len(targets) != 7 or int(config["n_seeds"]) != 10:
        raise ValueError("temporal figure requires the pinned seven-target, ten-seed artifact")
    if tuple(config["k_values"]) != DISPLAY_K:
        raise ValueError(f"temporal figure requires k={DISPLAY_K}")

    plt.rcParams.update(
        {
            # Serif, matching the other three public figures.
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    x = np.asarray(DISPLAY_K, dtype=float)

    final_values = []
    for target, color in zip(targets, TARGET_COLORS):
        values = np.asarray(trajectories[target], dtype=float)
        final_values.append(float(values[-1]))
        ax.plot(x, values, "o-", color=color, lw=2.2, ms=6.5, alpha=0.95)

    label_y = _nonoverlapping_label_positions(final_values)
    for target, color, actual_y, text_y in zip(targets, TARGET_COLORS, final_values, label_y):
        ax.plot([10.0, 10.35], [actual_y, text_y], color=color, lw=1.1, alpha=0.8)
        ax.text(10.45, text_y, target, color=color, va="center", fontsize=12.5)

    ax.set(
        xlim=(0.6, 13.6),
        ylim=(0, max(2.3, max(label_y) + 0.15)),
        xticks=DISPLAY_K,
        xlabel="Revealed target evaluations (k)",
        ylabel=(
            "Parity/reconstruction MedAE\n"
            "(includes exact revealed cells; normalized points)"
        ),
    )
    # Stop the axis spine at the last measured k so it does not run on under the
    # direct-label gutter, which read as an axis extending past the data.
    ax.spines["bottom"].set_bounds(min(DISPLAY_K), max(DISPLAY_K))
    ax.grid(axis="y", color="#D7D7D7", alpha=0.65, lw=0.8)
    fig.suptitle(
        "Prior-only temporal reconstruction: seven 2025 pathology targets",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    # Identical wording, rewrapped onto two lines: as one line it was wider than
    # the axes and therefore dictated the saved figure width.
    fig.text(
        0.5,
        0.03,
        "Each trajectory is the target-model median across ten probe seeds.\n"
        "MedAE includes k exact revealed cells plus all supported hidden predictions.",
        ha="center",
        va="bottom",
        fontsize=11,
        linespacing=1.3,
        color="#444444",
    )
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.20, top=0.88)
    return fig, ax


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    fig, _ = build_temporal_figure(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    fig.savefig(
        args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06, facecolor="white"
    )
    plt.close(fig)
    print(f"wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
