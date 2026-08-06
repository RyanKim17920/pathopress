#!/usr/bin/env python3
"""Plot unseen-model empirical risk/coverage and interval width diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"1": "#7b2cbf", "3": "#2a9d8f", "5": "#e76f51", "10": "#264653"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "experiments" / "new_model_confidence_rank1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "new_model_confidence_rank1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    metrics = payload["crossfit_metrics"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), constrained_layout=True)

    ax = axes[0, 0]
    for k, row in sorted(metrics["by_k"].items(), key=lambda item: int(item[0])):
        curve = row["risk_coverage_curve"]
        ax.plot([100 * item["kept_fraction"] for item in curve], [item["medae"] for item in curve], "o-", lw=2, label=f"k={k}", color=COLORS[k])
    ax.invert_xaxis()
    ax.set(xlabel="Predictions retained by lowest empirical risk (%)", ylabel="Held-out MedAE (points)")
    ax.set_title("A  Risk–coverage", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    offsets = {"1": (6, 5), "3": (6, 5), "5": (8, 20), "10": (8, -22)}
    for k, row in sorted(metrics["by_k"].items(), key=lambda item: int(item[0])):
        ax.scatter(row["median_interval_width"], 100 * row["interval_coverage"], s=90, color=COLORS[k], label=f"k={k}")
        ax.annotate(f"k={k}\nn={row['n_calibrated']:,}", (row["median_interval_width"], 100 * row["interval_coverage"]), xytext=offsets[k], textcoords="offset points", fontsize=9)
    ax.axhline(90, color="#333333", ls="--", lw=1.4, label="nominal 90%")
    ax.set(xlabel="Median interval width (points)", ylabel="Empirical held-out coverage (%)")
    ax.set_title("B  Coverage–width", loc="left", fontweight="bold")

    ax = axes[1, 0]
    all_suite_rows = metrics["by_suite"]
    suite_rows = {
        name: row for name, row in all_suite_rows.items()
        if row["interval_coverage"] is not None
    }
    suites = list(suite_rows)
    coverage = [100 * suite_rows[name]["interval_coverage"] for name in suites]
    bars = ax.bar(suites, coverage, color="#4c9f9b")
    ax.axhline(90, color="#333333", ls="--", lw=1.4)
    for bar, value in zip(bars, coverage):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .5, f"{value:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 102)
    ax.tick_params(axis="x", rotation=20)
    ax.set(ylabel="Empirical held-out coverage (%)")
    n_undefined = len(all_suite_rows) - len(suite_rows)
    suffix = f" ({n_undefined} unsupported omitted)" if n_undefined else ""
    ax.set_title("C  Coverage by suite" + suffix, loc="left", fontweight="bold")

    ax = axes[1, 1]
    entries = [item for evaluation in payload["by_evaluation"].values() for item in evaluation["by_k"].values()]
    counts = np.asarray([item["n_models"] for item in entries])
    ax.hist(counts, bins=np.arange(counts.min() - .5, counts.max() + 1.5), color="#e9c46a", edgecolor="white")
    threshold = payload["minimum_support"]["evaluation_models"]
    ax.axvline(threshold, color="#9b2226", ls="--", lw=1.5, label=f"abstain below {threshold} groups")
    ax.set(xlabel="Distinct held-out target-model groups per evaluation × k", ylabel="Calibration contexts")
    ax.set_title("D  Protocol support and abstention threshold", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    overall = metrics["overall"]
    fig.suptitle(
        "Unseen pathology-model uncertainty: empirical held-out coverage, not a clinical guarantee\n"
        f"nominal 90%; observed {100 * overall['interval_coverage']:.2f}% across {overall['n_calibrated']:,} predictions; median width {overall['median_interval_width']:.2f}",
        fontsize=15,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {args.output}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
