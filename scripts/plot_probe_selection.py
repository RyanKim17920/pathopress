#!/usr/bin/env python3
"""Render BenchPress-style probe and informativeness figures."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
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


def _random_band(rows: list[dict[str, object]], key: str, metric: str) -> tuple[np.ndarray, ...]:
    by_k: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row[key][metric]  # type: ignore[index]
        if value is not None:
            by_k[int(row["k"])].append(float(value))
    x = np.asarray(sorted(by_k))
    median = np.asarray([np.median(by_k[int(k)]) for k in x])
    q1 = np.asarray([np.percentile(by_k[int(k)], 25) for k in x])
    q3 = np.asarray([np.percentile(by_k[int(k)], 75) for k in x])
    return x, median, q1, q3


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


def render_curves(payload: dict[str, object], output_base: Path) -> None:
    baseline = payload["baseline"]
    greedy = payload["all_known_greedy"]
    random_rows = payload["random_global_prefixes"]
    heldout = payload["heldout_model"]["validation"]  # type: ignore[index]
    max_k = len(greedy)  # type: ignore[arg-type]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.7))
    specifications = [
        (
            axes[0],
            "parity", "medae", "hidden_only", "medae",
            "Scorecard reconstruction", "Median absolute cell error",
        ),
        (
            axes[1],
            "model_average", "mae", "model_average", "mae",
            "Literal model-average prediction", "Mean absolute row-average error",
        ),
    ]
    for ax, all_key, all_metric, held_key, held_metric, title, ylabel in specifications:
        base = float(baseline[all_key][all_metric])  # type: ignore[index]
        random_x, random_y, random_q1, random_q3 = _random_band(
            random_rows, all_key, all_metric  # type: ignore[arg-type]
        )
        random_x = np.concatenate([[0], random_x])
        random_y = np.concatenate([[base], random_y])
        random_q1 = np.concatenate([[base], random_q1])
        random_q3 = np.concatenate([[base], random_q3])
        greedy_x = np.arange(0, max_k + 1)
        greedy_y = np.asarray(
            [base]
            + [float(row[all_key][all_metric]) for row in greedy]  # type: ignore[index,union-attr]
        )
        heldout_x = np.arange(1, len(heldout) + 1)  # type: ignore[arg-type]
        heldout_y = np.asarray(
            [float(row[held_key][held_metric]) for row in heldout]  # type: ignore[index,union-attr]
        )

        ax.fill_between(random_x, random_q1, random_q3, color=GRAY, alpha=0.14, lw=0)
        ax.plot(random_x, random_y, "o--", color=GRAY, lw=2.0, ms=4.5, label="Random probe set")
        ax.plot(greedy_x, greedy_y, "o-", color=MAGENTA, lw=2.2, ms=4.7, label="Greedy, all-known")
        ax.plot(heldout_x, heldout_y, "s-", color=BLUE, lw=2.2, ms=4.5, label="70/30 held-out models")
        ax.plot([0], [base], marker="D", color="white", markeredgecolor=CHARCOAL, ms=5.3, zorder=5)
        if ax is axes[0]:
            for row in greedy:  # type: ignore[union-attr]
                step = int(row["step"])
                name = _short_name(str(row["added_evaluation_id"]))
                ax.annotate(
                    name,
                    (step, float(row[all_key][all_metric])),  # type: ignore[index]
                    xytext=(0, 10 if step % 2 else 22), textcoords="offset points",
                    rotation=36, ha="left", va="bottom", fontsize=6.8,
                    color=MAGENTA,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.78),
                )
        ax.set_xlim(-0.4, max_k + 0.4)
        ax.set_xticks(range(0, max_k + 1))
        ax.set_xlabel("# Top pathology evaluations", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", color=CHARCOAL)
        ax.grid(axis="y", color=GRID, alpha=0.75, lw=0.7)
        ax.legend(frameon=False, fontsize=8.5)

    fig.suptitle(
        "PathoPress probe policies (rank-1 Bias-ALS)",
        fontsize=15, fontweight="bold", color=CHARCOAL, y=1.01,
    )
    fig.text(
        0.5,
        -0.005,
        "All-known matches BenchPress and counts measured probes as zero error; held-out models exclude probe cells.",
        ha="center", fontsize=8.5, color=CHARCOAL,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output_base.with_suffix(f".{suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
        "--results",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "probe_selection_results_rank1.json",
    )
    parser.add_argument(
        "--informativeness",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "probe_informativeness_rank1.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    _style()
    render_curves(payload, PROJECT_ROOT / "figures" / "probe_selection_rank1")
    render_informativeness(
        args.informativeness,
        PROJECT_ROOT / "figures" / "probe_informativeness_rank1",
    )
    print("wrote figures/probe_selection_rank1.{png,pdf}")
    print("wrote figures/probe_informativeness_rank1.{png,pdf}")


if __name__ == "__main__":
    main()
