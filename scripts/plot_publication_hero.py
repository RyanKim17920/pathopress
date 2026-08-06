#!/usr/bin/env python3
"""Build the BenchPress-style composite PathoPress hero figure."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.publication import hero_target_cells, select_hero_target  # noqa: E402


BLUE = "#1368CE"
RED = "#D1495B"
GOLD = "#D89C1D"


def _random_band(rows: list[dict], metric: str) -> tuple[list[int], list[float], list[float], list[float]]:
    ks = sorted({int(row["k"]) for row in rows})
    values = [[float(row["metrics"][metric]) for row in rows if int(row["k"]) == k] for k in ks]
    return ks, [float(np.median(v)) for v in values], [float(np.quantile(v, .25)) for v in values], [float(np.quantile(v, .75)) for v in values]


def _model_average_curve(raw: list[dict[str, str]], candidate_mode: str) -> tuple[list[int], list[float]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        if row["protocol"] == "all_known" and row["candidate_mode"] == candidate_mode and row["selection_objective"] == "medae":
            groups[(int(row["k"]), row["model_id"])].append(row)
    ks = sorted({key[0] for key in groups})
    medians = []
    for k in ks:
        errors = []
        for (group_k, _), rows in groups.items():
            if group_k != k:
                continue
            actual = np.mean([float(row["actual_normalized_score"]) for row in rows])
            predicted = np.mean([float(row["predicted_normalized_score"]) for row in rows])
            errors.append(abs(actual - predicted))
        medians.append(float(np.median(errors)))
    return ks, medians


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compression", type=Path, default=ROOT / "experiments/probe_compression_rank1.json")
    parser.add_argument("--raw", type=Path, default=ROOT / "outputs/probe_compression_selected_raw_rank1.csv")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "experiments/publication_hero_summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/pathopress_hero_rank1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.compression.read_text(encoding="utf-8"))
    with args.raw.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    target = select_hero_target(raw)
    target_cells = hero_target_cells(raw, target)
    complete = target_cells[10]
    positions = np.unique(np.linspace(0, len(complete) - 1, min(28, len(complete)), dtype=int))
    ids = [complete[index]["evaluation_id"] for index in positions]
    by_k = {k: {row["evaluation_id"]: row for row in rows} for k, rows in target_cells.items()}
    by_k[0] = by_k[10]
    available_by_k = {
        k: sum(row["is_revealed_probe_cell"] == "True" for row in target_cells[k])
        for k in (1, 3, 10)
    }
    heat = np.asarray(
        [[float(by_k[k][evaluation]["actual_normalized_score"] if k == 0 else by_k[k][evaluation]["predicted_normalized_score"]) for evaluation in ids]
         for k in (0, 1, 3, 10)], dtype=float
    )

    fig = plt.figure(figsize=(14.2, 10.2))
    grid = fig.add_gridspec(3, 6, height_ratios=(1.12, 1, 1), hspace=.58, wspace=.62)
    ax_heat = fig.add_subplot(grid[0, :])
    image = ax_heat.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax_heat.set_yticks(
        range(4),
        ["Actual"] + [f"Keep {k} global ({available_by_k[k]} available)" for k in (1, 3, 10)],
    )
    ax_heat.set_xticks(range(len(ids)), [value.split(".", 1)[-1][:16] for value in ids], rotation=62, ha="right", fontsize=6.8)
    for y, k in enumerate((0, 1, 3, 10)):
        if k == 0:
            continue
        for x, evaluation in enumerate(ids):
            if by_k[k][evaluation]["is_revealed_probe_cell"] == "True":
                ax_heat.add_patch(plt.Rectangle((x - .48, y - .48), .96, .96, fill=False, edgecolor="white", linewidth=1.8))
    ax_heat.set_title(f"A  Concrete target row: {target} ({len(complete)} observed evaluation cells); white boxes are measured probes", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=.012, pad=.012)
    colorbar.set_label("Normalized score")

    for column, metric in enumerate(("medae", "medape")):
        ax = fig.add_subplot(grid[1, column * 3:(column + 1) * 3])
        for mode, color, label in (("any_candidate", BLUE, "Any evaluation"), ("pre_error_low_friction_allowlist", RED, "Pre-error feasibility proxy")):
            curves = payload["curves"][mode]
            greedy = curves[f"all_known_greedy_{metric}"]
            xs = [int(row["k"]) for row in greedy]
            ys = [float(row["selection_metrics"][metric]) for row in greedy]
            ax.plot(xs, ys, color=color, marker="o", linewidth=2.2, label=f"{label} — greedy")
            rxs, med, low, high = _random_band(curves["all_known_random"], metric)
            ax.plot(rxs, med, color=color, linestyle="--", linewidth=1.4, label=f"{label} — random")
            ax.fill_between(rxs, low, high, color=color, alpha=.11)
        ax.grid(alpha=.2)
        ax.set_xlabel("Measured evaluations (k)")
        ax.set_ylabel("Normalized-score points" if metric == "medae" else "Absolute percentage error (%)")
        ax.set_title(f"{'B' if column == 0 else 'C'}  Score reconstruction — {metric.upper()}", loc="left", fontweight="bold")
        ax.set_xticks(range(1, 11))
        if column == 0:
            ax.legend(fontsize=7.4, frameon=False, ncol=2)

    ax_avg = fig.add_subplot(grid[2, :3])
    for mode, color, label in (("any_candidate", BLUE, "Any evaluation"), ("pre_error_low_friction_allowlist", RED, "Pre-error feasibility proxy")):
        xs, ys = _model_average_curve(raw, mode)
        ax_avg.plot(xs, ys, marker="o", linewidth=2.2, color=color, label=label)
    ax_avg.set_title("D  Error predicting each model's average observed score", loc="left", fontweight="bold")
    ax_avg.set_xlabel("Measured evaluations (k)")
    ax_avg.set_ylabel("Median absolute average error")
    ax_avg.grid(alpha=.2)
    ax_avg.legend(frameon=False, fontsize=8)

    ax_rank = fig.add_subplot(grid[2, 3:])
    for mode, color, label in (("error_informed_pruned", GOLD, "Error-informed pruned"), ("pre_error_low_friction_allowlist", RED, "Pre-error feasibility proxy")):
        rank = payload["ranking_aware"][mode]
        pair = rank["pairwise_margin_error"]
        top = rank["top_fraction_error"]
        ax_rank.plot([row["k"] for row in pair], [row["selection_metrics"]["pairwise_median_accuracy"] for row in pair], color=color, marker="o", label=f"{label}: pairwise Δ≥2")
        ax_rank.plot([row["k"] for row in top], [row["selection_metrics"]["top_median_recovery"] for row in top], color=color, marker="s", linestyle="--", label=f"{label}: top 20%")
    exact = payload["ranking_aware"]["any_candidate_exact_k1"]
    ax_rank.scatter([1], [exact["pairwise_margin_error"][0]["selection_metrics"]["pairwise_median_accuracy"]], color=BLUE, marker="*", s=105, zorder=5, label="Any evaluation exact k=1")
    ax_rank.set_ylim(-.03, 1.03)
    ax_rank.set_xlabel("Measured evaluations (k)")
    ax_rank.set_ylabel("Median evaluation-level recovery")
    ax_rank.set_title("E  Ranking-oriented probe objectives", loc="left", fontweight="bold")
    ax_rank.grid(alpha=.2)
    ax_rank.legend(frameon=False, fontsize=7.2, ncol=2)

    summary = {
        "schema_version": 1,
        "target_model_id": target,
        "target_n_observed": len(complete),
        "displayed_evaluation_ids": ids,
        "keep_k": [1, 3, 10],
        "target_available_probe_cells": available_by_k,
        "unrestricted_curve_lengths": {
            metric: len(payload["curves"]["any_candidate"][f"all_known_greedy_{metric}"])
            for metric in ("medae", "medape")
        },
        "allowlist_curve_lengths": {
            metric: len(payload["curves"]["pre_error_low_friction_allowlist"][f"all_known_greedy_{metric}"])
            for metric in ("medae", "medape")
        },
        "pathology_adaptations": [
            "rank-1 completion selected by pathology validation rather than upstream rank-2",
            "low-friction curve uses a pre-error protocol-metadata proxy, not measured cost",
            "ranking continuation uses separately labelled error-informed pruning; unrestricted ranking is exact at k=1",
        ],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig.suptitle("PathoPress: compressing pathology foundation-model evaluation", fontsize=17, fontweight="bold", y=.995)
    fig.text(.5, .012, "Pinned BenchPress all-known masking; rank-1 pathology completion. Feasibility is a metadata proxy, not measured monetary cost; pruned ranking curves are error-informed.", ha="center", fontsize=8.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    print(f"target={target}; wrote {args.summary_output} and {args.output}.png/.pdf")


if __name__ == "__main__":
    main()
