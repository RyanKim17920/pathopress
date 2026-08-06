#!/usr/bin/env python3
"""Plot PathoPress pairwise-correlation and classical-MDS diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "structure_analysis"
COLORS = {
    "eva": "#7b2cbf", "hest": "#e76f51", "pathobench": "#277da1",
    "pathorob": "#43aa8b", "thunder": "#f9c74f",
}


def _save(figure, stem: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(ROOT / "figures" / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_correlation_summary(stats: dict) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    suites = sorted({row["suite_id"] for row in stats.values()})
    bins = np.linspace(0.0, 1.0, 21)
    for suite in suites:
        values = [row["max_abs_r"] for row in stats.values() if row["suite_id"] == suite]
        axis.hist(values, bins=bins, histtype="step", linewidth=2.2, color=COLORS[suite], label=f"{suite} (n={len(values)})")
    axis.axvline(0.85, color="#333333", linestyle="--", linewidth=1.2, label="|r| = 0.85")
    axis.set(xlabel="Best-neighbor absolute correlation", ylabel="Evaluation count", xlim=(0.0, 1.0))
    axis.set_title("Redundancy among pathology evaluation protocols", fontweight="bold")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    _save(figure, "benchmark_best_neighbor_correlations")


def _base_mds(axis, coordinates, evaluation_ids, suite_by_evaluation, *, show_centroids=True):
    for suite in sorted(set(suite_by_evaluation.values())):
        indices = [index for index, evaluation in enumerate(evaluation_ids) if suite_by_evaluation[evaluation] == suite]
        axis.scatter(
            coordinates[indices, 0], coordinates[indices, 1], s=38,
            color=COLORS[suite], alpha=0.62, edgecolor="white", linewidth=0.35,
            label=f"{suite} (n={len(indices)})",
        )
        if show_centroids:
            centroid = coordinates[indices].mean(axis=0)
            axis.text(
                centroid[0], centroid[1], suite.upper(), fontsize=9, fontweight="bold",
                ha="center", va="center", bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.75, "ec": "none"},
            )
    axis.set(xlabel="MDS dimension 1", ylabel="MDS dimension 2")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")


def plot_mds(stats: dict) -> None:
    with np.load(DATA / "correlation_mds.npz", allow_pickle=False) as data:
        coordinates = data["coordinates"]
        evaluation_ids = data["evaluation_ids"].astype(str).tolist()
    suite_by_evaluation = {evaluation: row["suite_id"] for evaluation, row in stats.items()}

    probe_path = ROOT / "experiments" / "probe_selection_results_rank1.json"
    probes = json.loads(probe_path.read_text(encoding="utf-8"))["all_known_greedy"][:10]
    id_to_index = {evaluation: index for index, evaluation in enumerate(evaluation_ids)}
    figure, axis = plt.subplots(figsize=(9.6, 6.7), constrained_layout=True)
    _base_mds(axis, coordinates, evaluation_ids, suite_by_evaluation, show_centroids=False)
    probe_key = []
    annotation_rows = []
    # Fixed screen-space offsets keep the dense near-origin probe cluster
    # readable while leader lines preserve the exact MDS locations.
    offsets = [(-26, 22), (-10, 30), (8, 30), (25, 22), (-30, 5),
               (-12, 12), (8, 12), (27, 5), (-18, -16), (18, -16)]
    for rank, probe in enumerate(probes, 1):
        evaluation = probe["added_evaluation_id"]
        coordinate = coordinates[id_to_index[evaluation]]
        axis.scatter(
            coordinate[0], coordinate[1], s=30, marker="o", color="#d00070",
            edgecolor="#222222", linewidth=0.45, zorder=5,
        )
        offset = offsets[rank - 1]
        axis.annotate(
            str(rank), xy=(coordinate[0], coordinate[1]), xytext=offset,
            textcoords="offset points", ha="center", va="center", fontsize=7.5,
            fontweight="bold", color="white", zorder=6,
            bbox={"boxstyle": "circle,pad=0.24", "fc": "#d00070", "ec": "#222222", "lw": 0.55},
            arrowprops={"arrowstyle": "-", "color": "#7a0040", "lw": 0.55},
        )
        probe_key.append(f"{rank}. {evaluation}")
        annotation_rows.append({
            "rank": rank, "evaluation_id": evaluation,
            "mds_x": float(coordinate[0]), "mds_y": float(coordinate[1]),
            "offset_points": list(offset),
        })
    axis.text(
        1.02, 0.03, "Selected probes\n" + "\n".join(probe_key),
        transform=axis.transAxes, ha="left", va="bottom", fontsize=6.8,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cccccc"},
    )
    axis.set_title("Correlation MDS with top ten greedy pathology probes", fontweight="bold")
    _save(figure, "benchmark_correlation_mds_probes")
    (DATA / "probe_mds_annotations.json").write_text(
        json.dumps({"schema_version": 1, "annotations": annotation_rows}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    stats = json.loads((DATA / "pairwise_ols_stats.json").read_text(encoding="utf-8"))
    plot_correlation_summary(stats)
    plot_mds(stats)


if __name__ == "__main__":
    main()
