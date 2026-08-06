#!/usr/bin/env python3
"""Plot BenchPress-style pairwise ranking and shortlist recovery curves."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pathopress-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAGENTA = "#D33682"
BLUE = "#268BD2"
VIOLET = "#6C71C4"
TEAL = "#2AA198"
ORANGE = "#CB6D1D"
GRAY = "#93A1A1"
CHARCOAL = "#333333"
COLORS = {"pathobench": ORANGE, "eva": VIOLET, "thunder": BLUE, "hest": MAGENTA, "pathorob": TEAL}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "experiments/ranking_preservation_rank1.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures")
    args = parser.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    pairwise = result["summary"]["pairwise_by_margin"]
    top = result["summary"]["top_by_fraction"]
    margins = result["metadata"]["margins"]
    fractions = result["metadata"]["top_fractions"]
    suites = sorted(next(iter(pairwise.values()))["by_suite"])

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    ax = axes[0]
    for suite in suites:
        values = [100 * pairwise[str(float(margin))]["by_suite"].get(suite, {}).get("median_accuracy", float("nan")) for margin in margins]
        ax.plot(margins, values, marker="o", ms=3.5, lw=1.15, alpha=0.58, color=COLORS[suite], label=suite.upper())
    overall = [100 * pairwise[str(float(margin))]["median_accuracy"] for margin in margins]
    ax.plot(margins, overall, marker="o", ms=6, lw=2.8, color=CHARCOAL, label="Overall median")
    for x, y in zip(margins, overall):
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(margins)
    ax.set_xlabel("Minimum true score gap (normalized points)")
    ax.set_ylabel("Pairwise ordering accuracy (%)")
    ax.set_title("A  Meaningful pairwise ordering")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2, fontsize=8.5)

    ax = axes[1]
    x_values = [100 * fraction for fraction in fractions]
    for suite in suites:
        values = [100 * top[str(float(fraction))]["by_suite"].get(suite, {}).get("median_recovery", float("nan")) for fraction in fractions]
        ax.plot(x_values, values, marker="s", ms=3.5, lw=1.15, alpha=0.58, color=COLORS[suite])
    overall = [100 * top[str(float(fraction))]["median_recovery"] for fraction in fractions]
    ax.plot(x_values, overall, marker="s", ms=6, lw=2.8, color=CHARCOAL)
    for x, y in zip(x_values, overall):
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x_values, [f"Top {value:.0f}%" for value in x_values])
    ax.set_xlabel("True/completed shortlist size")
    ax.set_ylabel("Top-set recovery (%)")
    ax.set_title("B  Shortlist recovery")
    ax.grid(axis="y", alpha=0.22)
    fig.suptitle("PathoPress ranking preservation — rank-1 out-of-fold completion", fontsize=14)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / f"ranking_preservation_rank1.{extension}", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
