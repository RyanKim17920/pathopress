#!/usr/bin/env python3
"""Plot unrestricted and feasibility-allowlisted probe-compression curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"any_candidate": "#1368CE", "pre_error_low_friction_allowlist": "#D1495B"}
LABELS = {"any_candidate": "Any evaluation", "pre_error_low_friction_allowlist": "Pre-error feasibility proxy"}


def probe_ticks(max_k: int) -> list[int]:
    """Return readable ticks for the upstream-equivalent k<=30 random range."""

    if max_k <= 10:
        return list(range(1, max_k + 1))
    return [value for value in (1, 5, 10, 15, 20, 25, 30) if value <= max_k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "experiments/probe_compression_rank1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/probe_compression_curves_rank1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    score_random_max_k = max(
        entry["k"]
        for mode in ("any_candidate", "pre_error_low_friction_allowlist")
        for protocol in ("all_known", "heldout")
        for entry in payload["curves"][mode][f"{protocol}_random"]
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), sharex=True)
    for column, metric in enumerate(("medae", "medape")):
        for row_index, protocol in enumerate(("all_known", "heldout")):
            ax = axes[row_index, column]
            for candidate_mode in ("any_candidate", "pre_error_low_friction_allowlist"):
                curves = payload["curves"][candidate_mode]
                greedy = curves[f"{protocol}_greedy_{metric}"]
                greedy_values = [
                    step["selection_metrics"][metric]
                    if protocol == "all_known"
                    else step["validation_metrics"][metric]
                    for step in greedy
                ]
                ks = [step["k"] for step in greedy]
                color = COLORS[candidate_mode]
                ax.plot(ks, greedy_values, color=color, marker="o", linewidth=2.2,
                        label=f"{LABELS[candidate_mode]} — greedy")
                random_rows = curves[f"{protocol}_random"]
                random_ks = sorted({entry["k"] for entry in random_rows})
                medians, lows, highs = [], [], []
                for k in random_ks:
                    values = [entry["metrics"][metric] for entry in random_rows if entry["k"] == k]
                    medians.append(float(np.median(values)))
                    lows.append(float(np.quantile(values, 0.25)))
                    highs.append(float(np.quantile(values, 0.75)))
                ax.plot(random_ks, medians, color=color, linestyle="--", linewidth=1.5,
                        label=f"{LABELS[candidate_mode]} — random")
                ax.fill_between(random_ks, lows, highs, color=color, alpha=0.10)
            ax.grid(alpha=0.22)
            ax.set_title(f"{'All-known parity' if protocol == 'all_known' else 'Held-out models'} — {metric.upper()}")
            ax.set_ylabel("Normalized-score points" if metric == "medae" else "Absolute percentage error (%)")
            if row_index == 1:
                ax.set_xlabel("Number of measured probe evaluations (k)")
            ax.set_xticks(probe_ticks(score_random_max_k))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("PathoPress probe compression under pinned BenchPress masking semantics", fontsize=14)
    fig.text(0.5, 0.073, "Shading: random-prefix interquartile range. Feasibility allowlist is a protocol proxy, not measured cost.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    print(f"wrote {args.output}.png and {args.output}.pdf")


if __name__ == "__main__":
    main()
