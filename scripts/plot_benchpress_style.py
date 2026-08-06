#!/usr/bin/env python3
"""Render BenchPress-style PathoPress matrix and validation figures."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pathopress-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathopress.completion import complete  # noqa: E402
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402


MAGENTA = "#D33682"
BLUE = "#268BD2"
VIOLET = "#6C71C4"
TEAL = "#2AA198"
ORANGE = "#CB6D1D"
GREEN = "#4C956C"
GRAY = "#93A1A1"
CHARCOAL = "#333333"
SUITE_COLORS = {
    "pathobench": ORANGE,
    "eva": VIOLET,
    "hest": MAGENTA,
    "thunder": BLUE,
    "pathorob": TEAL,
    "hoptimus1_report": GREEN,
}
SUITE_LABELS = {
    "pathobench": "PATHO-BENCH",
    "eva": "EVA",
    "thunder": "THUNDER",
    "hoptimus1_report": "H-OPT1",
    "hest": "HEST",
    "pathorob": "PATHOROB",
}


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


def save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def load_matrix(scores_path: Path):
    scores = load_scores(scores_path)
    matrix, models, evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(matrix, models, evaluations)
    metadata: dict[str, tuple[str, str]] = {}
    for score in scores:
        if score.evaluation_id in evaluations:
            metadata.setdefault(score.evaluation_id, (score.suite_id, score.metric))
    return matrix, models, evaluations, metadata


def matrix_order(matrix, models, evaluations, metadata):
    row_order = np.argsort(-np.sum(np.isfinite(matrix), axis=1), kind="stable")
    suite_rank = {
        "pathobench": 0,
        "eva": 1,
        "thunder": 2,
        "hoptimus1_report": 3,
        "hest": 4,
        "pathorob": 5,
    }
    col_order = np.asarray(
        sorted(
            range(len(evaluations)),
            key=lambda j: (
                suite_rank.get(metadata[evaluations[j]][0], 99),
                -int(np.sum(np.isfinite(matrix[:, j]))),
                evaluations[j],
            ),
        )
    )
    return row_order, col_order


def suite_groups(ordered_evaluations, metadata) -> list[tuple[str, int, int]]:
    groups: list[tuple[str, int, int]] = []
    start = 0
    while start < len(ordered_evaluations):
        suite = metadata[ordered_evaluations[start]][0]
        end = start + 1
        while end < len(ordered_evaluations) and metadata[ordered_evaluations[end]][0] == suite:
            end += 1
        groups.append((suite, start, end))
        start = end
    return groups


def suite_legend_handles(groups: list[tuple[str, int, int]]) -> list[Patch]:
    return [
        Patch(
            facecolor=SUITE_COLORS.get(suite, GRAY),
            edgecolor="none",
            label=SUITE_LABELS.get(suite, suite.upper()),
        )
        for suite, _, _ in groups
    ]


def add_suite_axis(ax, ordered_evaluations, metadata) -> list[tuple[str, int, int]]:
    groups = suite_groups(ordered_evaluations, metadata)
    # Narrow pathology suites can occupy only a handful of pixels in this
    # Wide matrix. A shared color legend is legible; centered suite-name
    # ticks are not, so the band carries the grouping and the legend names it.
    ax.set_xticks([])
    for suite, start, end in groups:
        ax.plot([start - 0.5, end - 0.5], [-1.15, -1.15], color=SUITE_COLORS.get(suite, GRAY), lw=5, clip_on=False)
        if end < len(ordered_evaluations):
            ax.axvline(end - 0.5, color="white", lw=1.5)
    return groups


def plot_observation_pattern(matrix, models, evaluations, metadata, output_dir):
    row_order, col_order = matrix_order(matrix, models, evaluations, metadata)
    observed = np.isfinite(matrix[np.ix_(row_order, col_order)])
    ordered_evaluations = [evaluations[j] for j in col_order]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    cmap = colors.ListedColormap(["white", BLUE])
    ax.imshow(observed, cmap=cmap, interpolation="nearest", aspect="auto")
    groups = add_suite_axis(ax, ordered_evaluations, metadata)
    ax.set_yticks([])
    ax.set_ylabel(f"{len(models)} models")
    ax.set_xlabel(f"{len(evaluations)} scored evaluations")
    ax.set_title(
        f"Published score coverage: {int(observed.sum())}/{observed.size} cells ({observed.mean():.1%})"
    )
    fig.legend(
        handles=suite_legend_handles(groups),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=len(groups),
        frameon=False,
        fontsize=7.5,
        handlelength=1.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(bottom=0.17)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, output_dir, "matrix_observation_pattern")


def plot_completed_matrix(matrix, models, evaluations, metadata, output_dir, rank):
    completed = complete(matrix, rank=rank)
    row_order, col_order = matrix_order(matrix, models, evaluations, metadata)
    original = matrix[np.ix_(row_order, col_order)]
    filled = completed[np.ix_(row_order, col_order)]
    ordered_evaluations = [evaluations[j] for j in col_order]
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)
    image = axes[0].imshow(original, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    axes[0].set_title("Published normalized scores")
    axes[0].set_ylabel(f"{len(models)} models (sorted by coverage)")
    rgba = cmap(np.clip(filled / 100.0, 0.0, 1.0))
    rgba[..., 3] = np.where(np.isfinite(original), 1.0, 0.48)
    axes[1].imshow(rgba, aspect="auto")
    axes[1].set_title(f"Completed matrix (latent interaction rank {rank})")
    groups = []
    for ax in axes:
        groups = add_suite_axis(ax, ordered_evaluations, metadata)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_xlabel("White = unreported")
    axes[1].set_xlabel("Solid = reported; translucent = imputed")
    fig.legend(
        handles=suite_legend_handles(groups),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=len(groups),
        frameon=False,
        fontsize=8.0,
        handlelength=1.5,
        columnspacing=1.2,
    )
    fig.subplots_adjust(bottom=0.17)
    cbar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Normalized score (0–100)")
    save(fig, output_dir, f"matrix_completed_rank{rank}")


def plot_validation(results_path: Path, predictions_path: Path, output_dir: Path) -> None:
    result = json.loads(results_path.read_text(encoding="utf-8"))
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    ranks = result["configuration"]["ranks"]
    prediction_rank = int(result["configuration"]["prediction_rank"])
    pooled_medae = [result["by_rank"][str(rank)]["pooled"]["medae"] for rank in ranks]
    fold_medae = [result["by_rank"][str(rank)]["fold_medae"]["median"] for rank in ranks]
    fold_q1 = [result["by_rank"][str(rank)]["fold_medae"]["q1"] for rank in ranks]
    fold_q3 = [result["by_rank"][str(rank)]["fold_medae"]["q3"] for rank in ranks]
    baseline_medae = result["column_median_baseline"]["pooled"]["medae"]

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.4))
    ax = axes[0, 0]
    ax.plot(ranks, pooled_medae, "o-", color=MAGENTA, lw=2.2, label="Pooled MedAE")
    ax.plot(ranks, fold_medae, "s-", color=BLUE, lw=2.2, label="Median fold MedAE")
    ax.fill_between(ranks, fold_q1, fold_q3, color=BLUE, alpha=0.14, label="Fold IQR")
    ax.axhline(baseline_medae, color=GRAY, ls="--", lw=1.6, label="Task-median baseline")
    ax.set_xlabel("Latent interaction rank")
    ax.set_ylabel("Absolute error (normalized points)")
    ax.set_xticks(ranks)
    ax.set_title("A  Rank sweep on identical folds")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    suites = sorted(
        result["by_rank"][str(prediction_rank)]["by_suite"],
        key=lambda suite: {
            "pathobench": 0,
            "eva": 1,
            "thunder": 2,
            "hoptimus1_report": 3,
            "hest": 4,
            "pathorob": 5,
        }.get(suite, 99),
    )
    for suite in suites:
        rows = [row for row in predictions if row["suite_id"] == suite]
        actual = np.asarray([float(row["actual_normalized_score"]) for row in rows])
        predicted = np.asarray([float(row["predicted_normalized_score"]) for row in rows])
        ax.scatter(
            actual,
            predicted,
            s=9,
            alpha=0.18,
            color=SUITE_COLORS[suite],
            label=SUITE_LABELS.get(suite, suite.upper()),
            edgecolors="none",
        )
    low, high = 0, 100
    ax.plot([low, high], [low, high], color=CHARCOAL, ls="--", lw=1.2)
    ax.set_xlim(35, 100)
    ax.set_ylim(35, 100)
    ax.set_xlabel("Held-out reported score")
    ax.set_ylabel("Out-of-fold imputation")
    ax.set_title(f"B  Rank-{prediction_rank} parity")
    ax.legend(frameon=False, markerscale=2)

    ax = axes[1, 0]
    selected = result["by_rank"][str(prediction_rank)]
    mae = [selected["by_suite"][suite]["mae"] for suite in suites]
    medae = [selected["by_suite"][suite]["medae"] for suite in suites]
    x = np.arange(len(suites))
    width = 0.36
    ax.bar(x - width / 2, mae, width, color=[SUITE_COLORS[s] for s in suites], alpha=0.6, label="MAE")
    ax.bar(x + width / 2, medae, width, color=[SUITE_COLORS[s] for s in suites], label="MedAE")
    ax.set_xticks(x)
    ax.set_xticklabels([SUITE_LABELS.get(s, s.upper()) for s in suites])
    ax.set_ylabel("Absolute error (normalized points)")
    ax.set_title("C  Error varies by benchmark suite")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    errors = np.sort(np.asarray([float(row["absolute_error"]) for row in predictions]))
    cumulative = np.arange(1, len(errors) + 1) / len(errors)
    ax.plot(errors, cumulative * 100, color=VIOLET, lw=2.4)
    for threshold, style in ((1, ":"), (3, "--"), (5, "-.")):
        coverage = 100 * float(np.mean(errors <= threshold))
        ax.axvline(threshold, color=GRAY, ls=style, lw=1.1)
        ax.text(threshold + 0.15, coverage, f"{coverage:.0f}% ≤ {threshold}", fontsize=9, va="bottom")
    ax.set_xlim(0, min(20, float(np.percentile(errors, 99.5))))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Absolute error (normalized points)")
    ax.set_ylabel("Predictions covered (%)")
    ax.set_title(f"D  Rank-{prediction_rank} error distribution")
    ax.grid(alpha=0.25)
    fig.suptitle("PathoPress BenchPress-style 10-seed × 3-fold validation", fontsize=15, y=1.01)
    fig.tight_layout()
    save(fig, output_dir, f"benchpress_style_validation_rank{prediction_rank}")


def plot_soft_impute_rank_sweep(results_path: Path, output_dir: Path) -> None:
    result = json.loads(results_path.read_text(encoding="utf-8"))
    ranks = result["configuration"]["ranks"]
    specifications = (
        ("identity", "Raw space", BLUE, "o"),
        ("logit", "Logit space", MAGENTA, "s"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    for ax, metric, label in (
        (axes[0], "medape", "MedAPE (%)"),
        (axes[1], "medae", "MedAE (normalized points)"),
    ):
        for key, name, color, marker in specifications:
            values = [result["results"][key][str(rank)]["pooled"][metric] for rank in ranks]
            best = int(np.argmin(values))
            ax.plot(ranks, values, marker=marker, color=color, lw=2.2, label=name)
            ax.scatter(
                [ranks[best]], [values[best]], marker="*", s=180,
                color=VIOLET, edgecolor=color, linewidth=0.8, zorder=4,
            )
        ax.set_xticks(ranks)
        ax.set_xlabel("Truncated-SVD rank")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("A  BenchPress paper metric")
    axes[1].set_title("B  Absolute-error view")
    axes[0].legend(frameon=False)
    fig.suptitle("Exact BenchPress raw/logit Soft-Impute rank sweep", fontsize=14)
    fig.tight_layout()
    save(fig, output_dir, "soft_impute_rank_sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "data" / "scores.csv")
    parser.add_argument(
        "--results", type=Path, default=PROJECT_ROOT / "experiments" / "benchpress_style_results.json"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "benchpress_style_predictions_rank1.csv",
    )
    parser.add_argument(
        "--soft-impute-results",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "soft_impute_rank_sweep_results.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures")
    parser.add_argument("--rank", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    matrix, models, evaluations, metadata = load_matrix(args.scores)
    plot_observation_pattern(matrix, models, evaluations, metadata, args.output_dir)
    plot_completed_matrix(matrix, models, evaluations, metadata, args.output_dir, args.rank)
    plot_validation(args.results, args.predictions, args.output_dir)
    plot_soft_impute_rank_sweep(args.soft_impute_results, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
