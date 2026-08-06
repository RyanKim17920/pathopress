#!/usr/bin/env python3
"""Plot validated real LLM metrics or an explicit unrun status panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-status", type=Path, default=ROOT / "experiments/llm_baseline/real_run_status.json")
    parser.add_argument("--real-metrics", type=Path, default=ROOT / "experiments/llm_baseline/real_metrics.json")
    parser.add_argument("--mock-metrics", type=Path, default=ROOT / "experiments/llm_baseline_smoke/mock_metrics.json")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/llm_baseline_status")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = json.loads(args.real_metrics.read_text()) if args.real_metrics.exists() else None
    status = json.loads(args.real_status.read_text())
    mock = json.loads(args.mock_metrics.read_text()) if args.mock_metrics.exists() else None
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    if real and real.get("headline_eligible"):
        labels = [row["condition"].replace("_", "\n") for row in real["summary"]]
        axes[0].bar(labels, [row["medae"] for row in real["summary"]], color="#1368CE")
        axes[1].bar(labels, [row["medape"] for row in real["summary"]], color="#D1495B")
        comparator = real.get("rank1_comparator", {})
        if comparator:
            axes[0].axhline(comparator["medae"], color="#222222", linestyle="--", label="rank-1 comparator")
            axes[1].axhline(comparator["medape"], color="#222222", linestyle="--", label="rank-1 comparator")
            axes[0].legend(frameon=False)
            axes[1].legend(frameon=False)
        axes[0].set_ylabel("MedAE (normalized-score points)")
        axes[1].set_ylabel("MedAPE (%)")
        title = "Validated real-provider LLM baselines"
    else:
        for ax in axes:
            ax.axis("off")
        axes[0].text(.5, .58, "REAL LLM RESULTS\nUNRUN", ha="center", va="center", fontsize=24, fontweight="bold", color="#D1495B")
        axes[0].text(.5, .27, status["reason"], ha="center", va="center", wrap=True, fontsize=9)
        mock_n = sum(row["n"] for row in mock["summary"]) if mock else 0
        axes[1].text(.5, .58, "Artifact contract validated", ha="center", va="center", fontsize=17, fontweight="bold", color="#1368CE")
        axes[1].text(
            .5, .36,
            f"Full pack: {status.get('request_count', 0):,} requests / "
            f"{status.get('target_prediction_count', 0):,} targets\n"
            f"Smoke mock predictions: {mock_n} · headline eligible: no",
            ha="center", va="center", fontsize=10,
        )
        axes[1].text(.5, .16, "Mock accuracy is intentionally not plotted or compared.", ha="center", va="center", fontsize=9)
        title = "PathoPress LLM-baseline execution status"
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    print(f"wrote {args.output}.png/.pdf")


if __name__ == "__main__":
    main()
