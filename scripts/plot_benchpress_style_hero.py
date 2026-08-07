#!/usr/bin/env python3
"""Render the result-first, single-panel PathoPress compression overview."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pathopress-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAGENTA = "#D33682"
BLUE = "#268BD2"
GRAY = "#7C8790"
CHARCOAL = "#333333"


def _random_band(rows: list[dict[str, object]]) -> tuple[np.ndarray, ...]:
    by_k: dict[int, list[float]] = {}
    for row in rows:
        k = int(row["k"])
        if 1 <= k <= 10:
            by_k.setdefault(k, []).append(float(row["metrics"]["medae"]))
    if sorted(by_k) != list(range(1, 11)):
        raise ValueError("hero requires random all-known MedAE controls for k=1..10")
    x = np.asarray(sorted(by_k), dtype=int)
    return (
        x,
        np.asarray([np.median(by_k[k]) for k in x]),
        np.asarray([np.quantile(by_k[k], 0.25) for k in x]),
        np.asarray([np.quantile(by_k[k], 0.75) for k in x]),
    )


def hero_plot_data(compression: dict[str, object], selection: dict[str, object]) -> dict[str, object]:
    """Extract only committed retrospective all-known trajectories."""

    configuration = compression["configuration"]
    provenance = selection["provenance"]
    if configuration["scores_sha256"] != provenance["scores_sha256"]:
        raise ValueError("compression and selection artifacts use different score snapshots")
    if int(configuration["prediction_rank"]) != 1:
        raise ValueError("hero requires the pathology-selected interaction rank 1")

    curves = compression["curves"]
    any_rows = curves["any_candidate"]["all_known_greedy_medae"]
    if [int(row["k"]) for row in any_rows] != list(range(1, 11)):
        raise ValueError("hero requires selected all-known MedAE results for k=1..10")

    proxy_rows = curves.get("pre_error_low_friction_allowlist", {}).get(
        "all_known_greedy_medae", []
    )
    proxy_supported = [int(row["k"]) for row in proxy_rows] == list(range(1, 11))
    random_x, random_median, random_q1, random_q3 = _random_band(
        curves["any_candidate"]["all_known_random"]
    )
    baseline = float(selection["baseline"]["parity"]["medae"])
    return {
        "source_shape": list(configuration["matrix_shape"]),
        "n_observed": int(configuration["n_observed"]),
        "scores_sha256": str(configuration["scores_sha256"]),
        "baseline": baseline,
        "random_x": random_x,
        "random_median": random_median,
        "random_q1": random_q1,
        "random_q3": random_q3,
        "selected_x": np.asarray([int(row["k"]) for row in any_rows]),
        "selected_medae": np.asarray(
            [float(row["selection_metrics"]["medae"]) for row in any_rows]
        ),
        "proxy_supported": proxy_supported,
        "proxy_x": np.asarray([int(row["k"]) for row in proxy_rows]) if proxy_supported else np.asarray([]),
        "proxy_medae": (
            np.asarray([float(row["selection_metrics"]["medae"]) for row in proxy_rows])
            if proxy_supported
            else np.asarray([])
        ),
    }


def build_hero_figure(values: dict[str, object]):
    baseline = float(values["baseline"])
    random_x = np.asarray(values["random_x"])
    random_median = np.asarray(values["random_median"])
    random_q1 = np.asarray(values["random_q1"])
    random_q3 = np.asarray(values["random_q3"])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(8.1, 5.1))
    x_with_zero = np.r_[0, random_x]
    ax.fill_between(
        x_with_zero,
        np.r_[baseline, random_q1],
        np.r_[baseline, random_q3],
        color=GRAY,
        alpha=0.16,
        lw=0,
    )
    ax.plot(
        x_with_zero,
        np.r_[baseline, random_median],
        "o--",
        color=GRAY,
        lw=2.0,
        ms=4.5,
        label="Random probes (median; IQR)",
    )
    ax.plot(
        np.r_[0, values["selected_x"]],
        np.r_[baseline, values["selected_medae"]],
        "o-",
        color=MAGENTA,
        lw=2.6,
        ms=5.2,
        label="Selected from any protocol",
    )
    if bool(values["proxy_supported"]):
        ax.plot(
            np.r_[0, values["proxy_x"]],
            np.r_[baseline, values["proxy_medae"]],
            "s-",
            color=BLUE,
            lw=2.4,
            ms=4.8,
            label="25-task feasibility pool",
        )
    ax.plot(
        [0],
        [baseline],
        marker="D",
        mfc="white",
        mec=CHARCOAL,
        ms=6.5,
        linestyle="none",
        label=f"Column-median baseline ({baseline:.2f})",
        zorder=5,
    )
    ax.set(
        xlabel="Number of revealed probe protocols (k)",
        ylabel="Median absolute cell error (normalized-score points)",
        xlim=(-0.25, 10.25),
        xticks=range(0, 11),
    )
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Retrospective all-known matrix reconstruction", fontweight="bold")
    semantics = (
        f"{values['source_shape'][0]} models × {values['source_shape'][1]} protocols; "
        f"{values['n_observed']:,} reported cells. Revealed probes are scored as exact.\n"
        "Probe selection and evaluation use the same model population; this is not model-level holdout. "
        "The 25-task feasibility pool is not measured cost."
    )
    fig.text(
        0.5,
        0.015,
        semantics,
        ha="center",
        va="bottom",
        fontsize=9.2,
        linespacing=1.35,
        color=CHARCOAL,
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.27, top=0.89)
    return fig, ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compression",
        type=Path,
        default=ROOT / "experiments/probe_compression_rank1.json",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "experiments/probe_selection_results_rank1.json",
    )
    parser.add_argument(
        "--hero-output",
        type=Path,
        default=ROOT / "figures/pathopress_benchpress_hero_rank1",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "experiments/benchpress_style_hero_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compression = json.loads(args.compression.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    values = hero_plot_data(compression, selection)
    fig, _ = build_hero_figure(values)

    args.hero_output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.hero_output.with_suffix(f".{suffix}"), dpi=240, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "schema_version": 2,
        "source_shape": values["source_shape"],
        "n_observed": values["n_observed"],
        "inputs": {
            "scores_sha256": values["scores_sha256"],
            "compression_sha256": hashlib.sha256(args.compression.read_bytes()).hexdigest(),
            "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        },
        "tracks": {
            "column_median_baseline": True,
            "random_any_candidate_median_iqr": True,
            "selected_any_candidate": True,
            "selected_25_task_low_friction_proxy": bool(values["proxy_supported"]),
        },
        "semantics": {
            "evaluation_scope": "retrospective_all_known_same_model_population",
            "revealed_probe_cells": "exact_zero_error",
            "model_level_holdout": False,
            "low_friction_proxy_is_measured_cost": False,
            "task_annotations": "omitted",
            "outcome_selected_examples": "omitted",
        },
        "contract_status": {
            "masking_and_k_budget": "exact",
            "rank_and_domain": "pathology_adapted",
            "exhaustive_25C5_30C5": "not_claimed_in_hero",
        },
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.hero_output}.{{png,pdf}} and {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
