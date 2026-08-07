#!/usr/bin/env python3
"""Plot task utility and held-out-model mean-score prediction in one figure."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pathopress-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


CHARCOAL = "#263238"
GRAY = "#8A9299"
MAGENTA = "#D81B60"
BLUE = "#2878B5"
TEAL = "#00897B"
GRID = "#E5E1D8"
ORANGE = "#E67E22"
VIOLET = "#6C5CE7"
SUITE_COLORS = {
    "pathobench": ORANGE,
    "eva": VIOLET,
    "hest": TEAL,
    "thunder": MAGENTA,
    "pathorob": BLUE,
}
MODE_LABELS = {
    "any_candidate": "All 187 candidates",
    "pre_error_low_friction_allowlist": "25-task feasibility pool",
}
MODE_COLORS = {
    "any_candidate": MAGENTA,
    "pre_error_low_friction_allowlist": BLUE,
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
        }
    )


def _short_name(value: str) -> str:
    parts = value.split(".")
    if value.startswith("thunder."):
        return parts[1].replace("_", " ").upper()
    if value.startswith("hest."):
        return f"HEST {parts[1].upper()}"
    if value.startswith("pathorob."):
        return f"PathoROB {parts[1].replace('_', ' ')}"
    if value.startswith("eva.leaderboard."):
        dataset = parts[2].replace("camelyon16_small", "CAM16-S")
        dataset = dataset.replace("patch_camelyon", "PCam").replace("_", " ")
        return f"EVA {dataset} {parts[-1]}"
    if value.startswith("pathobench.threads2025."):
        task = parts[-1].replace("-mutation", "").replace("-", " ").upper()
        return f"THREADS {task}"
    if value.startswith("pathobench.exaone2025."):
        task = parts[-1].replace("-mutation", "").replace("-", " ").upper()
        return f"EXAONE {task}"
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def top_utility_rows(
    informativeness_path: Path, *, top_n: int = 8
) -> list[dict[str, Any]]:
    """Return positive single-probe utilities from the transductive analysis."""

    rows: list[dict[str, Any]] = []
    for source in _read_csv(informativeness_path):
        improvement = float(source["improvement_over_column_median"])
        if improvement <= 0:
            continue
        rows.append(
            {
                "evaluation_id": source["evaluation_id"],
                "label": _short_name(source["evaluation_id"]),
                "suite_id": source["suite_id"],
                "improvement": improvement,
                "coverage": float(source["model_coverage"]),
            }
        )
    rows.sort(key=lambda row: (-row["improvement"], row["evaluation_id"]))
    return rows[:top_n]


def _model_mean_errors(rows: Iterable[dict[str, str]]) -> np.ndarray:
    by_model: dict[str, tuple[list[float], list[float]]] = defaultdict(
        lambda: ([], [])
    )
    for row in rows:
        actual, predicted = by_model[row["model_id"]]
        actual.append(float(row["actual_normalized_score"]))
        predicted.append(float(row["predicted_normalized_score"]))
    errors = [
        abs(float(np.mean(actual)) - float(np.mean(predicted)))
        for actual, predicted in by_model.values()
        if actual and predicted
    ]
    return np.asarray(errors, dtype=float)


def _record(
    *,
    mode: str,
    k: int,
    errors: np.ndarray,
    probe_ids: list[str],
    added_evaluation_id: str,
    n_train: int,
    n_validation: int,
) -> dict[str, Any]:
    if errors.size != n_validation or not np.isfinite(errors).all():
        raise ValueError(
            f"expected {n_validation} finite held-out model errors for {mode} k={k}; "
            f"found {errors.size}"
        )
    return {
        "protocol": "heldout_model",
        "candidate_mode": mode,
        "candidate_label": MODE_LABELS[mode],
        "k": k,
        "added_evaluation_id": added_evaluation_id,
        "probe_ids": json.dumps(probe_ids, separators=(",", ":")),
        "model_average_medae": float(np.median(errors)),
        "model_average_q25": float(np.quantile(errors, 0.25)),
        "model_average_q75": float(np.quantile(errors, 0.75)),
        "model_average_mae": float(np.mean(errors)),
        "n_train_models": n_train,
        "n_heldout_models": n_validation,
        "selection_objective": "none" if k == 0 else "training_scorecard_medae",
        "selection_scope": "nested prefixes selected on training models only",
        "target": "mean of each held-out model's reported normalized scores",
        "random_control_available": False,
    }


def build_heldout_mean_records(
    compression_path: Path,
    raw_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build model-average records using only committed held-out predictions."""

    payload = json.loads(compression_path.read_text(encoding="utf-8"))
    semantics = payload.get("configuration", {}).get("heldout_semantics", "")
    if "selected on training rows only" not in semantics:
        raise ValueError("compression artifact does not declare training-only probe selection")

    split = payload["split"]
    train_indices = tuple(int(value) for value in split["train_model_indices"])
    validation_indices = tuple(int(value) for value in split["validation_model_indices"])
    n_train = len(train_indices)
    n_validation = len(validation_indices)

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(raw_path):
        if (
            row["protocol"] == "heldout"
            and row["method"] == "greedy"
            and row["selection_objective"] == "medae"
        ):
            grouped[(row["candidate_mode"], int(row["k"]))].append(row)

    expected_validation_ids = split["validation_model_ids"]

    records: list[dict[str, Any]] = []
    for mode in MODE_LABELS:
        steps = payload["curves"][mode]["heldout_greedy_medae"]
        for step in steps:
            k = int(step["k"])
            rows = grouped.get((mode, k), [])
            row_models = {row["model_id"] for row in rows}
            if row_models != set(expected_validation_ids):
                raise ValueError(
                    f"held-out raw rows for {mode} k={k} do not cover the declared split"
                )
            records.append(
                _record(
                    mode=mode,
                    k=k,
                    errors=_model_mean_errors(rows),
                    probe_ids=list(step["probe_ids"]),
                    added_evaluation_id=step["added_evaluation_id"],
                    n_train=n_train,
                    n_validation=n_validation,
                )
            )
    return records, {
        "n_train": n_train,
        "n_validation": n_validation,
        "matrix_shape": payload["configuration"]["matrix_shape"],
        "zero_probe_available": False,
        "zero_probe_missing_reason": (
            "no committed held-out zero-probe predictions or model-average summary"
        ),
        "random_control_available": False,
        "random_control_missing_reason": (
            "the current held-out random artifact stores pooled cell metrics, not "
            "per-model predictions or model-average errors"
        ),
    }


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def build_figure(
    utility: list[dict[str, Any]],
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
):
    fig, (utility_ax, prediction_ax) = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.7),
        gridspec_kw={"width_ratios": (1.05, 1.15), "wspace": 0.35},
    )

    ordered = list(reversed(utility))
    y = np.arange(len(ordered))
    utility_ax.barh(
        y,
        [row["improvement"] for row in ordered],
        color=[SUITE_COLORS.get(row["suite_id"], GRAY) for row in ordered],
        alpha=0.92,
    )
    utility_ax.set_yticks(y, [row["label"] for row in ordered], fontsize=8.5)
    utility_ax.set_xlabel("All-known MedAE reduction (points; higher is better)")
    utility_ax.set_title(
        "A   Retrospective single-task utility", loc="left", fontsize=12, pad=28
    )
    coverage_values = {round(float(row["coverage"]), 12) for row in ordered}
    common_coverage = next(iter(coverage_values)) if len(coverage_values) == 1 else None
    displayed_suites = {str(row["suite_id"]).upper() for row in ordered}
    common_suite = next(iter(displayed_suites)) if len(displayed_suites) == 1 else None
    common_display_scope = common_coverage is not None and common_suite is not None
    coverage_note = (
        f"; all {len(ordered)}: {common_suite}, "
        f"{round(common_coverage * metadata['matrix_shape'][0])}/"
        f"{metadata['matrix_shape'][0]} models each ({100 * common_coverage:.0f}%)"
        if common_display_scope
        else ""
    )
    utility_ax.text(
        0,
        1.01,
        "Transductive; exact probe cells included" + coverage_note,
        transform=utility_ax.transAxes,
        fontsize=9.1,
        color=CHARCOAL,
    )
    utility_ax.grid(axis="x", color=GRID, alpha=0.8, lw=0.7)
    utility_ax.set_axisbelow(True)
    max_improvement = max(row["improvement"] for row in ordered)
    utility_ax.set_xlim(0, max_improvement * (1.08 if common_display_scope else 1.28))
    if not common_display_scope:
        for position, row in zip(y, ordered):
            utility_ax.text(
                row["improvement"] + max_improvement * 0.025,
                position,
                f"{100 * row['coverage']:.0f}% coverage",
                va="center",
                fontsize=7.5,
                color=CHARCOAL,
            )

    for mode in MODE_LABELS:
        rows = sorted(
            (row for row in records if row["candidate_mode"] == mode),
            key=lambda row: int(row["k"]),
        )
        x = np.asarray([int(row["k"]) for row in rows])
        median = np.asarray([float(row["model_average_medae"]) for row in rows])
        q25 = np.asarray([float(row["model_average_q25"]) for row in rows])
        q75 = np.asarray([float(row["model_average_q75"]) for row in rows])
        color = MODE_COLORS[mode]
        prediction_ax.fill_between(x, q25, q75, color=color, alpha=0.10, linewidth=0)
        prediction_ax.plot(
            x,
            median,
            "o-",
            color=color,
            lw=2.2,
            ms=5,
            label=MODE_LABELS[mode],
        )
    prediction_ax.set(
        xlabel="Measured evaluations (k)",
        ylabel="Median absolute error (points)",
        xticks=range(1, 11),
    )
    prediction_ax.set_ylim(bottom=0)
    prediction_ax.set_title(
        "B   Error predicting mean reported score", loc="left", fontsize=12, pad=28
    )
    prediction_ax.text(
        0,
        1.01,
        f"Nested prefixes: {metadata['n_train']} selection models → "
        f"{metadata['n_validation']} held-out models",
        transform=prediction_ax.transAxes,
        fontsize=9.1,
        color=CHARCOAL,
    )
    prediction_ax.grid(color=GRID, alpha=0.8, lw=0.7)
    prediction_ax.set_axisbelow(True)
    prediction_ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    prediction_ax.text(
        0.02,
        0.03,
        "Shading: held-out-model IQR\n"
        "No k=0/random model-mean controls in current artifacts",
        transform=prediction_ax.transAxes,
        fontsize=8.4,
        color=CHARCOAL,
        va="bottom",
    )

    fig.suptitle(
        "Which pathology evaluations matter, and how much is enough?",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.018,
        "A ranks retrospective utility—not causal task importance. B predicts each held-out "
        "model's mean over its reported normalized scores.\n"
        "The 25-task pool is a feasibility proxy—not measured cost.",
        ha="center",
        fontsize=8.8,
        color=CHARCOAL,
    )
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.20, top=0.84)
    return fig, utility_ax, prediction_ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compression",
        type=Path,
        default=ROOT / "experiments/probe_compression_rank1.json",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "outputs/probe_compression_selected_raw_rank1.csv",
    )
    parser.add_argument(
        "--informativeness",
        type=Path,
        default=ROOT / "outputs/probe_informativeness_rank1.csv",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "outputs/probe_dual_objective_rank1.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures/probe_dual_objective_rank1",
    )
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _style()
    utility = top_utility_rows(args.informativeness, top_n=args.top_n)
    records, metadata = build_heldout_mean_records(args.compression, args.raw)
    write_records(args.table, records)
    fig, _, _ = build_figure(utility, records, metadata)
    for suffix in ("png", "pdf"):
        path = args.output.with_suffix(f".{suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.table} and {args.output}.{{png,pdf}}")
    print(
        "k=0 omitted: " + metadata["zero_probe_missing_reason"]
    )
    print(
        "random control omitted: " + metadata["random_control_missing_reason"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
