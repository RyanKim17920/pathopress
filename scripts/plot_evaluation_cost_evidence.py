#!/usr/bin/env python3
"""Plot source coverage without implying an unsupported numeric cost model."""

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

def denominator_copy(total: int) -> dict[str, str]:
    """Return every figure label that states the retained-protocol denominator.

    Keeping this copy in one small, testable function prevents a regenerated
    registry and a previously hard-coded plot annotation from drifting apart.
    """

    return {
        "coverage_title": f"A. Evidence coverage (n={total:,})",
        "callout_title": "B. Numeric burden curve",
        "callout_headline": "NOT MEASURABLE YET",
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

    fig = plt.figure(figsize=(12.6, 5.7), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.75, 1.0))

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
    for y, value, count, direct_count in zip(
        positions,
        any_values,
        [any_counts[field] for field in fields],
        [direct_counts[field] for field in fields],
    ):
        ax.text(
            min(value + 1.2, 101),
            y,
            f"{direct_count}/{count}",
            va="center",
            fontsize=8,
        )
    ax.text(
        0,
        -0.13,
        "Labels show evaluation-specific / any sourced context. Family defaults are not measurements.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )

    ax = fig.add_subplot(grid[0, 1])
    ax.set_axis_off()
    ax.set_title(copy["callout_title"], loc="left", fontweight="bold")
    ax.text(
        0.5,
        0.72,
        copy["callout_headline"],
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color="#8c2d04",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.51,
        f"0 / {total:,} protocols report\nobserved runtime and dollar cost",
        ha="center",
        va="center",
        fontsize=13,
        linespacing=1.4,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.25,
        "Sample count, batch size, step limits, and\nqualitative access notes are evidence—but not cost.\nUnknown values remain unknown, never zero.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
        linespacing=1.5,
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#f7f4ef", "edgecolor": "#d7cfc3"},
    )

    fig.suptitle(
        "PathoPress evaluation burden evidence: coverage, not imputed cost",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.025,
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
