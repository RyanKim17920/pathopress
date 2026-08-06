#!/usr/bin/env python3
"""Render BenchPress-style per-evaluation and per-model predictability charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "pathobench": "#D33682",
    "eva": "#268BD2",
    "thunder": "#6C71C4",
    "hest": "#2AA198",
    "pathorob": "#F06AA6",
}
SUITE_LABELS = {
    "pathobench": "Patho-Bench",
    "eva": "EVA",
    "thunder": "THUNDER",
    "hest": "HEST",
    "pathorob": "PathoROB",
}
GRAY = "#93A1A1"
CHARCOAL = "#002B36"


def _short(value: str, limit: int = 35) -> str:
    value = value.replace("pathobench.exaone2025.", "EXAONE · ")
    value = value.replace("pathobench.threads2025.", "THREADS · ")
    value = value.replace("eva.leaderboard.", "EVA · ")
    value = value.replace("thunder.", "THUNDER · ")
    value = value.replace("hest.", "HEST · ")
    value = value.replace("pathorob.", "PathoROB · ")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _bar_panels(rows: list[dict], *, key: str, output_stem: Path, title: str) -> None:
    rows = sorted(rows, key=lambda row: float(row["medape"]))
    midpoint = (len(rows) + 1) // 2
    height = max(4.0, 0.25 * midpoint + 1.0)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, height))
    for ax, chunk in zip(axes, (rows[:midpoint], rows[midpoint:])):
        names = [_short(str(row[key])) for row in chunk]
        values = [float(row["medape"]) for row in chunk]
        colors = [COLORS.get(str(row.get("suite_id", "")), GRAY) for row in chunk]
        y = np.arange(len(chunk))
        ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.3, height=0.78)
        ax.set_yticks(y, labels=names, fontsize=7.4)
        ax.invert_yaxis()
        ax.set_xlabel("MedAPE (%)")
        ax.grid(axis="x", color="#EEF1F4", linewidth=0.8)
        if values:
            ax.set_xlim(0, max(values) * 1.12)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    suites = [suite for suite in COLORS if any(row.get("suite_id") == suite for row in rows)]
    if suites:
        fig.legend(
            handles=[Patch(color=COLORS[suite], label=SUITE_LABELS[suite]) for suite in suites],
            loc="upper center",
            ncol=len(suites),
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
        )
    if key == "evaluation_id":
        fig.subplots_adjust(top=0.90, left=0.24, right=0.98, wspace=0.58)
    else:
        fig.subplots_adjust(top=0.90, left=0.19, right=0.98, wspace=0.48)
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "predictability_results_rank1.json",
    )
    parser.add_argument("--figure-dir", type=Path, default=PROJECT_ROOT / "figures")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    _bar_panels(
        data["by_evaluation"],
        key="evaluation_id",
        output_stem=args.figure_dir / "benchmark_predictability_rank1",
        title="Pathology benchmark predictability",
    )
    _bar_panels(
        data["by_model"],
        key="model_id",
        output_stem=args.figure_dir / "model_predictability_rank1",
        title="Pathology model predictability",
    )


if __name__ == "__main__":
    main()
