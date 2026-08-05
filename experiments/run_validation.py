#!/usr/bin/env python3
"""Reproducible retrospective validation of PathoPress matrix completion.

This script deliberately builds every split before evaluating any rank.  Thus
the rank sweep compares ranks 1--6 on exactly the same held-out cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathopress.completion import complete  # noqa: E402
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402


RANKS = tuple(range(1, 7))


@dataclass(frozen=True)
class Split:
    protocol: str
    split_id: str
    train: np.ndarray
    held_cells: tuple[tuple[int, int], ...]
    metadata: dict[str, Any]


def _summary(errors: Iterable[float]) -> dict[str, float | int | None]:
    values = np.asarray(tuple(errors), dtype=float)
    if not values.size:
        return {"n_predictions": 0, "mae": None, "medae": None}
    return {
        "n_predictions": int(values.size),
        "mae": round(float(np.mean(values)), 6),
        "medae": round(float(np.median(values)), 6),
    }


def _cell_signature(
    cells: Iterable[tuple[int, int]], models: list[str], evaluations: list[str]
) -> str:
    labels = sorted(f"{models[i]}\t{evaluations[j]}" for i, j in cells)
    return hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest()


def _supported_prediction(
    train: np.ndarray,
    original: np.ndarray,
    held_cells: tuple[tuple[int, int], ...],
    rank: int,
) -> tuple[list[float], dict[str, int]]:
    """Complete the train-supported submatrix and score supported holdouts.

    Whole-block holdouts can create all-missing rows or columns.  The core API
    correctly rejects those, so this adapter removes them and explicitly
    counts held-out cells that cannot be estimated.
    """
    row_supported = np.any(np.isfinite(train), axis=1)
    col_supported = np.any(np.isfinite(train), axis=0)
    kept = tuple(
        (i, j) for i, j in held_cells if row_supported[i] and col_supported[j]
    )
    dropped_row = sum(not row_supported[i] for i, _ in held_cells)
    dropped_col = sum(
        row_supported[i] and not col_supported[j] for i, j in held_cells
    )
    if not kept:
        return [], {
            "n_held_out": len(held_cells),
            "n_unsupported_row": dropped_row,
            "n_unsupported_column": dropped_col,
        }

    row_ids = np.flatnonzero(row_supported)
    col_ids = np.flatnonzero(col_supported)
    row_map = {int(old): new for new, old in enumerate(row_ids)}
    col_map = {int(old): new for new, old in enumerate(col_ids)}
    prediction = complete(train[np.ix_(row_ids, col_ids)], rank=rank)
    errors = [
        abs(float(prediction[row_map[i], col_map[j]] - original[i, j]))
        for i, j in kept
    ]
    return errors, {
        "n_held_out": len(held_cells),
        "n_unsupported_row": dropped_row,
        "n_unsupported_column": dropped_col,
    }


def _evaluate_job(job: tuple[int, Split, np.ndarray]) -> dict[str, Any]:
    rank, split, original = job
    errors, support = _supported_prediction(
        split.train, original, split.held_cells, rank
    )
    return {
        "rank": rank,
        "protocol": split.protocol,
        "split_id": split.split_id,
        "metadata": split.metadata,
        "support": support,
        "errors": errors,
    }


def _random_cell_splits(
    matrix: np.ndarray,
    models: list[str],
    evaluations: list[str],
    *,
    fraction: float,
    repeats: int,
    seed: int,
) -> list[Split]:
    rng = np.random.RandomState(seed)
    observed = np.argwhere(np.isfinite(matrix))
    requested = max(1, int(len(observed) * fraction))
    splits: list[Split] = []
    for repeat in range(repeats):
        train = matrix.copy()
        held: list[tuple[int, int]] = []
        for raw_i, raw_j in observed[rng.permutation(len(observed))]:
            i, j = int(raw_i), int(raw_j)
            # Match the core smoke validator's support policy: after removal,
            # retain >=2 cells in every row and >=3 in every column.
            if np.sum(np.isfinite(train[i])) <= 2:
                continue
            if np.sum(np.isfinite(train[:, j])) <= 3:
                continue
            train[i, j] = np.nan
            held.append((i, j))
            if len(held) >= requested:
                break
        cells = tuple(held)
        splits.append(
            Split(
                protocol="random_cell",
                split_id=f"repeat_{repeat:02d}",
                train=train,
                held_cells=cells,
                metadata={
                    "repeat": repeat,
                    "requested_cells": requested,
                    "cell_signature_sha256": _cell_signature(
                        cells, models, evaluations
                    ),
                },
            )
        )
    return splits


def _suite_block_splits(
    matrix: np.ndarray,
    models: list[str],
    evaluations: list[str],
    evaluation_suites: dict[str, str],
) -> list[Split]:
    suites = sorted(set(evaluation_suites.values()))
    suite_for_col = np.asarray([evaluation_suites[name] for name in evaluations])
    splits: list[Split] = []
    for suite in suites:
        target_cols = np.flatnonzero(suite_for_col == suite)
        other_cols = np.flatnonzero(suite_for_col != suite)
        # A bridge model has at least one observed target-suite result and at
        # least one observed result in another suite.
        bridge_rows = np.flatnonzero(
            np.any(np.isfinite(matrix[:, target_cols]), axis=1)
            & np.any(np.isfinite(matrix[:, other_cols]), axis=1)
        )
        held = tuple(
            (int(i), int(j))
            for i in bridge_rows
            for j in target_cols
            if np.isfinite(matrix[i, j])
        )
        train = matrix.copy()
        for i, j in held:
            train[i, j] = np.nan
        splits.append(
            Split(
                protocol="leave_one_suite_block_out",
                split_id=suite,
                train=train,
                held_cells=held,
                metadata={
                    "suite": suite,
                    "n_bridge_models": int(len(bridge_rows)),
                    "n_suite_columns": int(len(target_cols)),
                    "cell_signature_sha256": _cell_signature(
                        held, models, evaluations
                    ),
                },
            )
        )
    return splits


def _sparse_new_model_splits(
    matrix: np.ndarray,
    models: list[str],
    evaluations: list[str],
    *,
    probe_counts: tuple[int, ...],
    seed: int,
) -> list[Split]:
    rng = np.random.RandomState(seed)
    splits: list[Split] = []
    for k in probe_counts:
        for i, model in enumerate(models):
            observed_cols = np.flatnonzero(np.isfinite(matrix[i]))
            if len(observed_cols) <= k:
                continue
            probe_cols = np.sort(rng.choice(observed_cols, size=k, replace=False))
            probe_set = {int(j) for j in probe_cols}
            held = tuple(
                (i, int(j)) for j in observed_cols if int(j) not in probe_set
            )
            train = matrix.copy()
            for row, col in held:
                train[row, col] = np.nan
            splits.append(
                Split(
                    protocol="sparse_new_model_probe",
                    split_id=f"k{k}:{model}",
                    train=train,
                    held_cells=held,
                    metadata={
                        "k": k,
                        "model": model,
                        "n_observed_model_cells": int(len(observed_cols)),
                        "probe_evaluations": [evaluations[j] for j in probe_cols],
                        "held_cell_signature_sha256": _cell_signature(
                            held, models, evaluations
                        ),
                    },
                )
            )
    return splits


def _aggregate(
    evaluated: list[dict[str, Any]], ranks: tuple[int, ...]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for rank in ranks:
        rank_rows = [row for row in evaluated if row["rank"] == rank]
        errors = [error for row in rank_rows for error in row["errors"]]
        unsupported_rows = sum(
            row["support"]["n_unsupported_row"] for row in rank_rows
        )
        unsupported_cols = sum(
            row["support"]["n_unsupported_column"] for row in rank_rows
        )
        result[str(rank)] = {
            **_summary(errors),
            "n_splits": len(rank_rows),
            "n_unsupported_row": unsupported_rows,
            "n_unsupported_column": unsupported_cols,
        }
    return result


def _grouped_aggregate(
    evaluated: list[dict[str, Any]],
    ranks: tuple[int, ...],
    group_key: str,
) -> dict[str, Any]:
    groups = sorted({str(row["metadata"][group_key]) for row in evaluated})
    return {
        group: _aggregate(
            [row for row in evaluated if str(row["metadata"][group_key]) == group],
            ranks,
        )
        for group in groups
    }


def _split_manifest(splits: list[Split]) -> list[dict[str, Any]]:
    return [
        {
            "split_id": split.split_id,
            "n_held_out": len(split.held_cells),
            **split.metadata,
        }
        for split in splits
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores", type=Path, default=PROJECT_ROOT / "data" / "scores.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "experiments" / "results.json"
    )
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores, verified_only=True)
    matrix, models, evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(
        matrix,
        models,
        evaluations,
        min_scores_per_model=3,
        min_models_per_evaluation=5,
    )

    suite_by_evaluation: dict[str, str] = {}
    for score in scores:
        if score.evaluation_id in evaluations:
            previous = suite_by_evaluation.setdefault(
                score.evaluation_id, score.suite_id
            )
            if previous != score.suite_id:
                raise ValueError(
                    f"evaluation assigned to multiple suites: {score.evaluation_id}"
                )

    random_splits = _random_cell_splits(
        matrix,
        models,
        evaluations,
        fraction=args.holdout_fraction,
        repeats=args.random_repeats,
        seed=args.seed,
    )
    suite_splits = _suite_block_splits(
        matrix, models, evaluations, suite_by_evaluation
    )
    sparse_splits = _sparse_new_model_splits(
        matrix,
        models,
        evaluations,
        probe_counts=(3, 5, 10),
        seed=args.seed + 1,
    )
    all_splits = random_splits + suite_splits + sparse_splits
    # Include the small source matrix explicitly so workers are portable across
    # both fork- and spawn-based multiprocessing implementations.
    jobs = [(rank, split, matrix) for rank in RANKS for split in all_splits]

    if args.workers == 1:
        evaluated = [_evaluate_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            evaluated = list(executor.map(_evaluate_job, jobs, chunksize=1))

    random_rows = [r for r in evaluated if r["protocol"] == "random_cell"]
    suite_rows = [
        r for r in evaluated if r["protocol"] == "leave_one_suite_block_out"
    ]
    sparse_rows = [
        r for r in evaluated if r["protocol"] == "sparse_new_model_probe"
    ]
    input_sha = hashlib.sha256(args.scores.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "input": {
            "scores_path": str(args.scores.resolve().relative_to(PROJECT_ROOT)),
            "scores_sha256": input_sha,
            "verified_only": True,
            "support_filter": {
                "min_scores_per_model": 3,
                "min_models_per_evaluation": 5,
            },
        },
        "matrix": {
            "n_models": len(models),
            "n_evaluations": len(evaluations),
            "n_observed": int(np.sum(np.isfinite(matrix))),
            "density": round(float(np.mean(np.isfinite(matrix))), 6),
            "suites": {
                suite: {
                    "n_evaluations": sum(
                        suite_by_evaluation[evaluation] == suite
                        for evaluation in evaluations
                    )
                }
                for suite in sorted(set(suite_by_evaluation.values()))
            },
        },
        "configuration": {
            "seed": args.seed,
            "ranks": list(RANKS),
            "completion_regularization": 0.1,
            "completion_internal_iterations": 40,
            "completion_internal_ensembles": 10,
            "error_unit": "normalized-score points on a 0-100 scale",
        },
        "runtime": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "protocols": {
            "random_cell": {
                "description": (
                    "Greedy fixed 20% observed-cell holdouts retaining at least "
                    "two training scores per row and three per column."
                ),
                "split_manifest": _split_manifest(random_splits),
                "by_rank": _aggregate(random_rows, RANKS),
            },
            "leave_one_suite_block_out": {
                "description": (
                    "For each suite, hide every suite score of each model that "
                    "also has a score in another suite."
                ),
                "split_manifest": _split_manifest(suite_splits),
                "overall_by_rank": _aggregate(suite_rows, RANKS),
                "by_suite": _grouped_aggregate(suite_rows, RANKS, "suite"),
            },
            "sparse_new_model_probe": {
                "description": (
                    "For each eligible model independently, reveal a fixed "
                    "random k of its scores and hide all its other known scores; "
                    "all other models remain fully observed."
                ),
                "split_manifest": _split_manifest(sparse_splits),
                "overall_by_rank": _aggregate(sparse_rows, RANKS),
                "by_probe_count": _grouped_aggregate(sparse_rows, RANKS, "k"),
            },
        },
        "caveats": [
            "Random-cell holdout leaks model- and suite-level context: nearby tasks from the same suite often remain observed.",
            "Suite-block holdout is harder but still learns each target column from other models and each target model from other suites; it is not a new-institution, temporal, or model-family holdout.",
            "Sparse-probe simulations evaluate one existing model at a time and leave all other models observed, so related-model and publication-selection leakage can remain.",
            "normalized_score maps heterogeneous, differently scaled metrics into 0-100 with direction corrected. An error point is a normalized-score point, not one percentage point of accuracy and not a shared clinical utility unit.",
            "Only published primary-source-parsed cells are evaluated; they have not received dual human review, missingness is not random, and retrospective error can be optimistic for genuinely novel models or tasks.",
            "Reported n values are pooled prediction instances. Random repeats can hold out the same source cell more than once, and sparse k settings can target the same cell in separate simulations.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "protocols": {
        key: value.get("by_rank", value.get("overall_by_rank"))
        for key, value in payload["protocols"].items()
    }}, indent=2))


if __name__ == "__main__":
    main()
