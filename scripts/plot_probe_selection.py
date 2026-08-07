#!/usr/bin/env python3
"""Render the one-probe informativeness figure."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pathopress-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHARCOAL = "#263238"
GRAY = "#8A9299"
MAGENTA = "#D81B60"
BLUE = "#2878B5"
TEAL = "#00897B"
GRID = "#E5E1D8"
ORANGE = "#E67E22"
VIOLET = "#6C5CE7"
SUITE_COLORS = {
    "pathobench": ORANGE,
    "eva": VIOLET,
    "hest": TEAL,
    "thunder": MAGENTA,
    "pathorob": BLUE,
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )


def _short_name(value: str) -> str:
    parts = value.split(".")
    if value.startswith("thunder."):
        return parts[1].replace("_", " ").upper()
    if value.startswith("hest."):
        return f"HEST {parts[1].upper()}"
    if value.startswith("pathorob."):
        return f"PathoROB {parts[1].replace('_', ' ')}"
    if value.startswith("eva.leaderboard."):
        dataset = parts[2].replace("camelyon16_small", "CAM16-S")
        dataset = dataset.replace("patch_camelyon", "PCam").replace("_", " ")
        return f"EVA {dataset} {parts[-1]}"
    if value.startswith("pathobench.threads2025."):
        task = parts[-1].replace("-mutation", "").replace("-", " ").upper()
        return f"THREADS {task}"
    if value.startswith("pathobench.exaone2025."):
        task = parts[-1].replace("-mutation", "").replace("-", " ").upper()
        return f"EXAONE {task}"
    return value


def build_informativeness_figure(csv_path: Path, top_n: int = 15):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:top_n]
    rows.reverse()
    names = [_short_name(row["evaluation_id"]) for row in rows]
    improvements = [float(row["improvement_over_column_median"]) for row in rows]
    colors = [SUITE_COLORS.get(row["suite_id"], GRAY) for row in rows]
    coverages = [100.0 * float(row["model_coverage"]) for row in rows]

    y = np.arange(len(rows))
    fig, (ax, coverage_ax) = plt.subplots(
        1,
        2,
        figsize=(10.6, 7.8),
        sharey=True,
        gridspec_kw={"width_ratios": (4.7, 1.15), "wspace": 0.04},
    )
    ax.barh(y, improvements, color=colors, alpha=0.9)
    ax.set_yticks(y, names)
    ax.axvline(0, color=CHARCOAL, lw=0.8)
    ax.set_xlabel("Reduction in all-known scorecard MedAE vs column-median baseline")
    ax.set_title(
        "Single-evaluation probe informativeness",
        fontsize=14, fontweight="bold", color=CHARCOAL,
    )
    ax.grid(axis="x", color=GRID, alpha=0.75, lw=0.7)

    coverage_ax.barh(y, coverages, color="#CFD8DC", alpha=0.95)
    coverage_ax.set_xlim(0, 100)
    coverage_ax.set_xlabel("Model coverage (%)", fontsize=9)
    coverage_ax.set_xticks([0, 50, 100])
    coverage_ax.tick_params(axis="y", left=False, labelleft=False)
    coverage_ax.grid(axis="x", color=GRID, alpha=0.75, lw=0.7)
    coverage_ax.set_axisbelow(True)
    for row_y, coverage in zip(y, coverages):
        coverage_ax.text(
            min(coverage + 3.0, 97.0),
            row_y,
            f"{coverage:.0f}%",
            va="center",
            ha="left" if coverage <= 90 else "right",
            fontsize=7.5,
            color=CHARCOAL,
        )
    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=7, label=suite.upper())
        for suite, color in SUITE_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.62, 0.005),
        ncol=len(legend_handles),
        fontsize=8,
        handlelength=1.5,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.27, right=0.98, top=0.91, bottom=0.16)
    return fig, ax, coverage_ax


def render_informativeness(csv_path: Path, output_base: Path, top_n: int = 15) -> None:
    fig, _, _ = build_informativeness_figure(csv_path, top_n=top_n)
    for suffix in ("png", "pdf"):
        path = output_base.with_suffix(f".{suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--informativeness",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "probe_informativeness_rank1.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _style()
    render_informativeness(
        args.informativeness,
        PROJECT_ROOT / "figures" / "probe_informativeness_rank1",
    )
    print("wrote figures/probe_informativeness_rank1.{png,pdf}")


if __name__ == "__main__":
    main()
