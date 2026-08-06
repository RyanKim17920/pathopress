#!/usr/bin/env python3
"""BenchPress-style box/strip plot for pathology temporal deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DISPLAY_K = (1, 5, 10)
COLORS = ("#4C78A8", "#D45087", "#6F4C9B")


def metric_values(payload: dict, metric: str) -> list[list[float]]:
    return [
        [
            float(payload["summary_by_target"][target]["by_k"][str(k)][metric]["median"])
            for target in payload["config"]["target_model_ids"]
        ]
        for k in DISPLAY_K
    ]


def plot_metric(ax, values: list[list[float]], ylabel: str) -> None:
    positions = np.arange(len(DISPLAY_K))
    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#242424", "linewidth": 1.4},
        boxprops={"edgecolor": "#242424", "linewidth": 0.9},
        whiskerprops={"color": "#242424", "linewidth": 0.8},
        capprops={"color": "#242424", "linewidth": 0.8},
    )
    for color, box in zip(COLORS, boxes["boxes"]):
        box.set_facecolor(color)
        box.set_alpha(0.16)
    rng = np.random.RandomState(42)
    for index, ys in enumerate(values):
        jitter = rng.uniform(-0.11, 0.11, size=len(ys))
        ax.scatter(
            np.full(len(ys), positions[index]) + jitter,
            ys,
            s=20,
            color=COLORS[index],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
        median = float(np.median(ys))
        ax.text(positions[index], median, f"{median:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(positions, [str(k) for k in DISPLAY_K])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#A0A0A0", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "experiments" / "temporal_deployment_rank1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "temporal_deployment_rank1.png")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0), sharex=True)
    plot_metric(axes[0], metric_values(payload, "medae"), "MedAE (score points)")
    plot_metric(axes[1], metric_values(payload, "medape"), "MedAPE (%)")
    for ax in axes:
        ax.set_xlabel("Revealed target scores k")
    fig.suptitle(f"Temporal deployment ({len(payload['config']['target_model_ids'])} pathology models)", fontsize=10)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
