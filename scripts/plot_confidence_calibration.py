#!/usr/bin/env python3
"""Render the full PathoPress confidence-calibration diagnostic composite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STYLES = {
    "disagreement": ("Ensemble-spread", "#7b8ba3", "--"),
    "structural_support": ("Matrix-support", "#268bd2", "-."),
    "combined_risk_model": ("Hybrid uncertainty", "#d33682", "-"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "experiments" / "confidence_calibration_rank1.json",
    )
    parser.add_argument(
        "--cells", type=Path,
        default=ROOT / "experiments" / "confidence_cells_rank1.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "figures" / "confidence_calibration_rank1",
    )
    return parser.parse_args()


def _cell_arrays(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    errors: list[float] = []
    risks = {method: [] for method in STYLES}
    trust = {method: [] for method in STYLES}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            errors.append(float(row["absolute_error"]))
            for method in STYLES:
                risks[method].append(float(row[f"{method}_risk"]))
                trust[method].append(float(row[f"{method}_trust_probability"]))
    return np.asarray(errors), {
        method: np.asarray(values) for method, values in risks.items()
    }, {
        method: np.asarray(values) for method, values in trust.items()
    }


def _quantile_calibration(error: np.ndarray, log_risk: np.ndarray, bins: int = 10):
    order = np.argsort(log_risk, kind="stable")
    groups = np.array_split(order, bins)
    observed = [float(np.median(error[group])) for group in groups]
    percentile = 100.0 * (np.arange(bins, dtype=float) + 0.5) / bins
    return percentile, np.asarray(observed)


def _style_axis(ax, panel: str, title: str) -> None:
    ax.set_title(f"{panel}  {title}", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9dee7", linewidth=0.8, alpha=0.72)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    absolute_error, risk, trust = _cell_arrays(args.cells)
    methods = payload["confidence_methods"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8.5,
    })
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.0))

    # A: the upstream main diagnostic: retain the lowest-risk predictions.
    ax = axes[0, 0]
    for method, (label, color, line) in STYLES.items():
        rows = methods[method]["risk_coverage_curve"]
        ax.plot(
            [100.0 * float(row["kept_fraction"]) for row in rows],
            [float(row["medae"]) for row in rows],
            marker="o", linewidth=2.0, markersize=4.8,
            linestyle=line, color=color, label=label,
        )
    ax.set_xlabel("Lowest-risk predictions kept (%)")
    ax.set_ylabel("MedAE (normalized points)")
    ax.set_xlim(15, 105)
    ax.legend(frameon=False)
    _style_axis(ax, "A", "Risk–coverage")

    # B: predicted risk versus realized error, binned only for legibility.
    ax = axes[0, 1]
    for method, (label, color, line) in STYLES.items():
        percentile, observed = _quantile_calibration(absolute_error, risk[method])
        rho = float(methods[method]["spearman_uncertainty_abs_error"])
        ax.plot(
            percentile, observed, marker="o", markersize=4.5,
            linewidth=1.8, linestyle=line, color=color,
            label=f"{label} ($\\rho$={rho:.2f})",
        )
    ax.set_xlabel("Predicted-risk percentile (decile)")
    ax.set_ylabel("Observed absolute error (decile median)")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False)
    _style_axis(ax, "B", "Error–risk calibration")

    # C: the reliability strata reported in the JSON artifact.
    ax = axes[1, 0]
    bins = ["low_uncertainty", "medium_uncertainty", "high_uncertainty"]
    labels = ["Low", "Medium", "High"]
    centers = np.arange(len(bins), dtype=float)
    width = 0.23
    for offset, (method, (label, color, _)) in enumerate(STYLES.items()):
        rows = {row["bin"]: row for row in methods[method]["uncertainty_terciles"]}
        values = [float(rows[name]["medae"]) for name in bins]
        ax.bar(centers + (offset - 1) * width, values, width=width,
               color=color, alpha=0.88, label=label)
    ax.set_xticks(centers, labels)
    ax.set_xlabel("Predicted-uncertainty tercile")
    ax.set_ylabel("MedAE (normalized points)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    _style_axis(ax, "C", "Uncertainty strata")

    # D: interval efficiency and empirical coverage together.
    ax = axes[1, 1]
    names = list(STYLES)
    y = np.arange(len(names))
    widths = [float(methods[name]["conformal_90_interval"]["median_width"])
              for name in names]
    coverages = [100.0 * float(methods[name]["conformal_90_interval"]["coverage"])
                 for name in names]
    colors = [STYLES[name][1] for name in names]
    display = [STYLES[name][0] for name in names]
    ax.barh(y, widths, color=colors, alpha=0.88, height=0.55)
    for index, (value, coverage) in enumerate(zip(widths, coverages)):
        ax.text(value + 0.10, index, f"{value:.2f}  ({coverage:.1f}% covered)",
                va="center", fontsize=9, color=colors[index])
    ax.axvline(0, color="#002b36", linewidth=0.8)
    ax.set_yticks(y, display)
    ax.set_xlabel("Median conformal 90% interval width")
    ax.set_xlim(0, max(widths) * 1.55)
    _style_axis(ax, "D", "Interval coverage and efficiency")

    # E: the requested probabilistic interpretation of hybrid uncertainty.
    ax = axes[0, 2]
    reliability = methods["combined_risk_model"]["trust_probability"]["reliability_curve"]
    observed_probability = [100.0 * float(row["empirical_probability"]) for row in reliability]
    predicted_probability = [100.0 * float(row["mean_predicted_probability"]) for row in reliability]
    ax.plot([0, 100], [0, 100], color="#7b8ba3", linestyle="--", linewidth=1.2,
            label="Perfect calibration")
    ax.plot(predicted_probability, observed_probability, "o-", color="#d33682",
            linewidth=2.0, markersize=4.8, label="Hybrid trust")
    ax.set_xlim(0, 102); ax.set_ylim(0, 102)
    ax.set_xlabel("Predicted P(|error| ≤10) (%)")
    ax.set_ylabel("Observed event rate (%)")
    ax.legend(frameon=False)
    _style_axis(ax, "E", "Cross-fitted trust calibration")

    # F: show that trust is serialized cell-by-cell, not inferred from intervals.
    ax = axes[1, 2]
    event = absolute_error <= 10.0
    bins = np.linspace(0.0, 1.0, 16)
    ax.hist(trust["combined_risk_model"][event], bins=bins, color="#268bd2",
            alpha=0.68, label="|error| ≤10")
    ax.hist(trust["combined_risk_model"][~event], bins=bins, color="#d33682",
            alpha=0.68, label="|error| >10")
    ax.set_xlabel("Cross-fitted trust probability")
    ax.set_ylabel("Held-out prediction instances")
    ax.legend(frameon=False)
    _style_axis(ax, "F", "Trust-probability separation")

    fig.suptitle("Pathology matrix-completion confidence diagnostics",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.2, w_pad=1.7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
