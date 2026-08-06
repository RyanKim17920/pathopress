#!/usr/bin/env python3
"""Render main and appendix-style Section 6 pathology factor figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAGENTA = "#d33682"
BLUE = "#268bd2"
TEAL = "#2aa198"
VIOLET = "#6c71c4"
CHARCOAL = "#002b36"
GRID = "#e5e9ef"


def _finite(rows, x_key, y_key="medae"):
    x = np.asarray([
        np.nan if row.get(x_key) is None else float(row[x_key]) for row in rows
    ])
    y = np.asarray([float(row[y_key]) for row in rows])
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def _scatter(ax, rows, x_key, title, xlabel, color, test):
    x, y = _finite(rows, x_key)
    ax.scatter(x, y, s=25, color=color, alpha=0.72, edgecolor="white", linewidth=0.3)
    if len(x) >= 2 and np.std(x) > 1e-12:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(grid, slope * grid + intercept, color=CHARCOAL, linewidth=1.3)
    rho, p_value, n = float(test["rho"]), float(test["p"]), int(test["n"])
    ax.text(0.03, 0.97, f"$\\rho$={rho:+.2f}, p={p_value:.3g}, n={n}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel("MedAE (normalized points)")


def _stat_annotation(test):
    """Compact two-line label avoids glyph collisions in narrow panels."""
    return (
        f"$\\rho$={float(test['rho']):+.2f}, n={int(test['n'])}\n"
        f"p={float(test['p']):.3g}"
    )


def _style(axes):
    for ax in np.asarray(axes).flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=240, bbox_inches="tight")


def _effect(data, hypothesis, metric="medae", setting=None):
    block = data["interventions"][hypothesis]
    if setting is not None:
        return block["by_setting"][str(setting)][metric]
    return block["tests"][metric]


def benchmark_main(data, output: Path):
    import matplotlib.pyplot as plt

    block = data["correlational"]["benchmark"]
    rows, tests = block["rows"], block["tests"]["medae"]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    _scatter(axes[0, 0], rows, "rank2_r2", "A  Low-rank fit", "Rank-2 reconstruction $R^2$", MAGENTA, tests["rank2_r2"])
    _scatter(axes[0, 1], rows, "median_score", "B  Score level", "Median observed score", BLUE, tests["median_score"])
    _scatter(axes[1, 0], rows, "score_std", "C  Score dispersion", "Observed-score SD", TEAL, tests["score_std"])
    ax = axes[1, 1]
    hypotheses = ["benchmark_h4", "benchmark_h5", "benchmark_h6", "benchmark_h7"]
    labels = ["Target\ncoverage", "Neighbor\npresence", "Neighbor\nsupport", "Same task\nfamily"]
    values = [float(_effect(data, name)["median_delta"]) for name in hypotheses]
    colors = [MAGENTA, BLUE, TEAL, VIOLET]
    ax.bar(np.arange(4), values, color=colors, alpha=0.86)
    ax.axhline(0, color=CHARCOAL, linewidth=0.8)
    for i, (value, name) in enumerate(zip(values, hypotheses)):
        test = _effect(data, name)
        large_bar = value > 0.3
        ax.text(
            i, value - 0.08 if large_bar else value + 0.025,
            f"p={float(test['p_value']):.3g}\nn={int(test['n'])}",
            ha="center", va="top" if large_bar else "bottom", fontsize=8,
            color="white" if large_bar else CHARCOAL,
        )
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("Median $\\Delta$MedAE")
    ax.set_title("D  Paired evidence interventions", loc="left", fontweight="bold")
    _style(axes)
    fig.suptitle("Benchmark-side pathology prediction-error factors", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
    _save(fig, output / "predictability_factors_benchmark_rank1")
    plt.close(fig)


def model_main(data, output: Path):
    import matplotlib.pyplot as plt

    block = data["correlational"]["model"]
    rows, tests = block["rows"], block["tests"]["medae"]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    _scatter(axes[0, 0], rows, "log10_parameter_count", "A  Encoder size", "$\\log_{10}$(parameters)", MAGENTA, tests["log10_parameter_count"])

    ax = axes[0, 1]
    tile = [float(row["medae"]) for row in rows if int(row["is_slide_model"]) == 0]
    slide = [float(row["medae"]) for row in rows if int(row["is_slide_model"]) == 1]
    boxes = ax.boxplot(
        [tile, slide],
        tick_labels=[f"Tile\n(n={len(tile)})", f"Slide\n(n={len(slide)})"],
        patch_artist=True,
    )
    for patch, color in zip(boxes["boxes"], [BLUE, MAGENTA]):
        patch.set_facecolor(color); patch.set_alpha(0.58)
    test = tests["is_slide_model"]
    ax.text(
        0.03, 0.97, _stat_annotation(test), transform=ax.transAxes,
        ha="left", va="top", fontsize=9, linespacing=1.15,
        zorder=10,
        # The first Tile outlier sits directly behind this label.  A generous,
        # opaque pad prevents its circular edge from reading as punctuation.
        bbox={"boxstyle": "round,pad=0.7", "fc": "white", "ec": "none", "alpha": 1.0},
    )
    ax.set_ylabel("MedAE (normalized points)")
    ax.set_title("B  Model type", loc="left", fontweight="bold")

    _scatter(axes[1, 0], rows, "median_score", "C  Score level", "Median observed score", TEAL, tests["median_score"])
    _scatter(axes[1, 1], rows, "rank2_r2", "D  Low-rank fit", "Rank-2 reconstruction $R^2$", VIOLET, tests["rank2_r2"])
    _style(axes)
    fig.suptitle("Model-side pathology prediction-error factors", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
    _save(fig, output / "predictability_factors_model_rank1")
    plt.close(fig)


def benchmark_appendix(data, output: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    specs = [
        ("benchmark_h4", [0.25, 0.5, 0.75], "Target observations removed", MAGENTA),
        ("benchmark_h5", [0.95, 0.9, 0.85], "Neighbor |r| threshold", BLUE),
        ("benchmark_h6", [0.25, 0.5, 0.75], "Best-neighbor evidence removed", TEAL),
    ]
    for panel, (hypothesis, settings, xlabel, color) in zip("ABC", specs):
        ax = axes[ord(panel) - ord("A")]
        for metric, line in (("medae", "-"), ("medape", "--")):
            values = [float(_effect(data, hypothesis, metric, setting)["median_delta"]) for setting in settings]
            ax.plot(settings, values, marker="o", color=color, linestyle=line,
                    label="$\\Delta$MedAE" if metric == "medae" else "$\\Delta$MedAPE")
        ax.axhline(0, color=CHARCOAL, linewidth=0.8)
        ax.set_xlabel(xlabel); ax.set_ylabel("Median paired error delta")
        ax.set_title(f"{panel}  {hypothesis.replace('_', ' ').title()}", loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
    _style(axes)
    fig.tight_layout()
    _save(fig, output / "predictability_factors_benchmark_appendix_rank1")
    plt.close(fig)


def model_appendix(data, output: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    ax = axes[0, 0]
    names = ["model_h5", "model_h6", "model_h7"]
    labels = ["Peer presence", "Peer support", "Same provider"]
    values = [float(_effect(data, name)["median_delta"]) for name in names]
    ax.bar(np.arange(3), values, color=[MAGENTA, BLUE, TEAL], alpha=0.86)
    ax.axhline(0, color=CHARCOAL, linewidth=0.8); ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Median $\\Delta$MedAE"); ax.set_title("A  Peer-evidence ablations", loc="left", fontweight="bold")

    ax = axes[0, 1]
    settings = [0.25, 0.5, 0.75]
    for metric, color in (("medae", BLUE), ("medape", MAGENTA)):
        values = [float(_effect(data, "model_h6", metric, setting)["median_delta"]) for setting in settings]
        ax.plot(settings, values, marker="o", color=color, label=f"$\\Delta${metric.upper()}")
    ax.axhline(0, color=CHARCOAL, linewidth=0.8); ax.set_xlabel("Strongest-peer evidence removed")
    ax.set_ylabel("Median paired error delta"); ax.legend(frameon=False)
    ax.set_title("B  Peer-support dose response", loc="left", fontweight="bold")

    ax = axes[1, 0]
    h8 = data["interventions"]["model_h8"]["by_condition"]
    labels = ["Hide 25%", "Hide 75%"]
    values = [float(h8[name]["tests"]["medae"]["median_delta"]) for name in ("hide_25pct", "hide_75pct")]
    ax.bar(np.arange(2), values, color=[TEAL, VIOLET], alpha=0.86)
    ax.axhline(0, color=CHARCOAL, linewidth=0.8); ax.set_xticks(np.arange(2), labels)
    ax.set_ylabel("Median $\\Delta$MedAE vs hide 50%")
    ax.set_title("C  Target observation count", loc="left", fontweight="bold")

    ax = axes[1, 1]
    temporal = data["interventions"]["model_h9"]["comparison_A_vs_B"]
    k_values = [1, 3, 5, 8, 10, 15]
    a = [float(temporal[str(k)]["medae"]["median_A"]) for k in k_values]
    b = [float(temporal[str(k)]["medae"]["median_B"]) for k in k_values]
    ax.plot(k_values, a, "o-", color=CHARCOAL, label="Oldest-third context")
    ax.plot(k_values, b, "s-", color=MAGENTA, label="Middle-third context")
    ax.set_xlabel("Revealed evaluations ($k$)"); ax.set_ylabel("MedAE")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("D  Training-anchor recency", loc="left", fontweight="bold")
    _style(axes)
    fig.tight_layout(h_pad=2.0)
    _save(fig, output / "predictability_factors_model_appendix_rank1")
    plt.close(fig)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "experiments" / "prediction_error_factors_rank1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.titlesize": 12})
    benchmark_main(data, args.output_dir)
    model_main(data, args.output_dir)
    benchmark_appendix(data, args.output_dir)
    model_appendix(data, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
