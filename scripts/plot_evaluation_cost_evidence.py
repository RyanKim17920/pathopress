#!/usr/bin/env python3
"""Plot PathoPress cost-evidence coverage and pre-error feasibility strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


FIELD_LABELS = {
    "sample_count": "Sample count",
    "acquisition_scale": "MPP / scale",
    "stain": "Stain",
    "compute_configuration": "Compute config",
    "dataset_license": "Dataset license",
    "hardware_model": "Hardware model",
    "observed_runtime": "Observed runtime",
    "annotation_hours": "Annotation hours",
    "dollar_cost": "Dollar cost",
}

TIER_LABELS = {
    "tier_1_direct_small_labeled": "1: direct, ≤10k",
    "tier_2_direct_labeled": "2: direct labeled",
    "tier_3_aggregated_or_wsi": "3: case / WSI",
    "tier_4_specialized_protocol": "4: specialized",
}


def denominator_copy(total: int) -> dict[str, str]:
    """Return every figure label that states the retained-protocol denominator.

    Keeping this copy in one small, testable function prevents a regenerated
    registry and a previously hard-coded plot annotation from drifting apart.
    """

    return {
        "coverage_title": f"A. Evidence coverage (n={total:,})",
        "missingness_footer": (
            "Observed runtime, hardware make/model, annotation hours, and dollar cost: "
            f"0/{total:,}. A numeric cost curve is therefore unsupported."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/evaluation_cost_evidence.json")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "figures/evaluation_cost_evidence_coverage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = payload["summary"]
    total = int(summary["n_evaluations"])
    copy = denominator_copy(total)
    fields = list(FIELD_LABELS)
    any_counts = summary["field_coverage_count"]
    direct_counts = summary["field_direct_evaluation_coverage_count"]

    fig = plt.figure(figsize=(13.2, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.12), width_ratios=(1.55, 1.0))

    ax = fig.add_subplot(grid[0, 0])
    positions = np.arange(len(fields))
    any_values = np.array([100 * any_counts[field] / total for field in fields])
    direct_values = np.array([100 * direct_counts[field] / total for field in fields])
    ax.barh(positions, any_values, color="#b8c7d9", label="Any sourced context")
    ax.barh(positions, direct_values, color="#2468a2", label="Evaluation-specific evidence")
    ax.set_yticks(positions, [FIELD_LABELS[field] for field in fields])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Retained protocols with evidence (%)")
    ax.set_title(copy["coverage_title"], loc="left", fontweight="bold")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    for y, value, count in zip(positions, any_values, [any_counts[field] for field in fields]):
        ax.text(min(value + 1.2, 101), y, str(count), va="center", fontsize=8)

    ax = fig.add_subplot(grid[0, 1])
    tier_counts = summary["pre_error_feasibility_tier_counts"]
    tier_ids = list(TIER_LABELS)
    values = [tier_counts.get(tier, 0) for tier in tier_ids]
    colors = ["#2b8cbe", "#7bccc4", "#fdbb84", "#d7301f"]
    ax.barh(np.arange(len(tier_ids)), values, color=colors)
    ax.set_yticks(np.arange(len(tier_ids)), [TIER_LABELS[tier] for tier in tier_ids])
    ax.invert_yaxis()
    ax.set_xlabel("Retained protocols")
    ax.set_title("B. Pre-error feasibility strata", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for y, value in enumerate(values):
        ax.text(value + 1, y, str(value), va="center", fontsize=9)
    ax.text(
        0,
        -0.23,
        "Strata use unit, task type, and reported count only; they are not measured cost tiers.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )

    ax = fig.add_subplot(grid[1, :])
    suites = list(summary["field_coverage_by_suite"])
    heat_fields = [
        "sample_count",
        "acquisition_scale",
        "stain",
        "dataset_license",
        "hardware_model",
        "observed_runtime",
        "annotation_hours",
        "dollar_cost",
    ]
    matrix = np.zeros((len(suites), len(heat_fields)), dtype=float)
    labels = np.empty_like(matrix, dtype=object)
    for i, suite in enumerate(suites):
        row = summary["field_coverage_by_suite"][suite]
        denominator = row["n_evaluations"]
        for j, field in enumerate(heat_fields):
            matrix[i, j] = row[field] / denominator
            labels[i, j] = f"{row[field]}/{denominator}"
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_yticks(np.arange(len(suites)), suites)
    ax.set_xticks(np.arange(len(heat_fields)), [FIELD_LABELS[field] for field in heat_fields])
    ax.tick_params(axis="x", rotation=25)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                labels[i, j],
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[i, j] > 0.55 else "#222222",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.02, pad=0.015)
    colorbar.set_label("Evidence coverage fraction")
    ax.set_title("C. Any source-backed evidence by suite", loc="left", fontweight="bold")

    fig.suptitle(
        "PathoPress evaluation cost evidence: coverage, not imputed cost",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        copy["missingness_footer"],
        ha="center",
        fontsize=10,
        color="#8c2d04",
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {args.output_prefix.with_suffix('.png')}")
    print(f"wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
