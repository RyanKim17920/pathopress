#!/usr/bin/env python3
"""Plot current-score ranking preservation from probe-compression tracks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pathopress-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAGENTA = "#D33682"
BLUE = "#268BD2"
GRAY = "#93A1A1"
CHARCOAL = "#333333"


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


def _values(rows: list[dict[str, object]], metrics_key: str) -> tuple[list[int], np.ndarray]:
    return (
        [int(row["k"]) for row in rows],
        100
        * np.asarray(
            [float(row[metrics_key]["pairwise_median_accuracy"]) for row in rows]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        default=PROJECT_ROOT / "experiments/ranking_preservation_rank1.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures")
    args = parser.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    if result.get("schema_version") != 2:
        raise ValueError("ranking plot requires current compression-derived schema v2")
    tracks = result["tracks"]

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.15), sharex=True, sharey=True)

    ax = axes[0]
    random = tracks["any_candidate"]["all_known_random_summary"]
    random_x = np.asarray([int(row["k"]) for row in random])
    random_median = 100 * np.asarray([float(row["pairwise_median"]) for row in random])
    random_q1 = 100 * np.asarray([float(row["pairwise_q1"]) for row in random])
    random_q3 = 100 * np.asarray([float(row["pairwise_q3"]) for row in random])
    ax.fill_between(random_x, random_q1, random_q3, color=GRAY, alpha=0.18, lw=0)
    ax.plot(random_x, random_median, "o--", color=GRAY, lw=2, ms=4, label="Random any-evaluation")
    for mode, color, marker in (
        ("any_candidate", MAGENTA, "o"),
        ("pre_error_low_friction_allowlist", BLUE, "s"),
    ):
        rows = tracks[mode]["all_known_greedy"]
        x, y = _values(rows, "metrics")
        ax.plot(np.r_[0, x], np.r_[0, y], marker=marker, color=color, lw=2.5,
                label=f"{tracks[mode]['label']} — greedy")
    ax.set_title("A  All-known margin-5 ranking")
    ax.set_ylabel("Median pairwise accuracy (%)")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    for mode, color, marker in (
        ("any_candidate", MAGENTA, "o"),
        ("pre_error_low_friction_allowlist", BLUE, "s"),
    ):
        rows = tracks[mode]["heldout_greedy"]
        x, hidden = _values(rows, "validation_non_probe")
        _, with_probe = _values(rows, "validation_with_probe_zero")
        ax.plot(x, hidden, marker=marker, color=color, lw=2.5,
                label=f"{tracks[mode]['label']} — hidden only")
        ax.plot(x, with_probe, marker=marker, color=color, lw=1.5, ls="--", alpha=.75,
                label=f"{tracks[mode]['label']} — probes exact")
    ax.set_title("B  Held-out-model validation")
    ax.legend(frameon=False, fontsize=7.8)

    for ax in axes:
        ax.set(xlim=(0, 10.25), ylim=(0, 100), xticks=range(0, 11))
        ax.set_xlabel("Number of measured probe evaluations (k)")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle(
        "PathoPress ranking preservation — current 59×187 rank-1 probe compression",
        fontsize=14,
    )
    fig.text(
        .5,
        .005,
        "True normalized-score gap ≥5. All-known includes exact probes; held-out hidden-only excludes them. Feasibility proxy ≠ measured cost.",
        ha="center",
        fontsize=8.2,
    )
    fig.subplots_adjust(bottom=.17, top=.84, wspace=.16)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            args.output_dir / f"ranking_preservation_rank1.{extension}",
            bbox_inches="tight",
            pad_inches=0.08,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
