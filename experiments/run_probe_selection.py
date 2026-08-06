#!/usr/bin/env python3
"""Reproduce BenchPress's hero probe policy on the pathology score matrix.

The primary curve intentionally matches the optimistic all-known protocol.
We additionally report hidden-only error, literal row-average error, and an
isolated 70/30 held-out-model validation curve.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathopress.completion import complete  # noqa: E402
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.probes import (  # noqa: E402
    ProbeEvaluation,
    evaluate_column_median_baseline,
    evaluate_global_probes,
    random_global_probe_prefixes,
)


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _summary(result: ProbeEvaluation) -> dict[str, object]:
    return {
        "n_target_cells": result.n_target_cells,
        "n_revealed_cells": result.n_revealed_cells,
        "n_hidden_cells": result.n_hidden_cells,
        "parity": {
            "n": result.parity.n,
            "medae": _finite(result.parity.median_absolute_error),
            "mae": _finite(result.parity.mean_absolute_error),
        },
        "hidden_only": {
            "n": result.hidden_only.n,
            "medae": _finite(result.hidden_only.median_absolute_error),
            "mae": _finite(result.hidden_only.mean_absolute_error),
        },
        "model_average": {
            "n": result.model_average.n,
            "medae": _finite(result.model_average.median_absolute_error),
            "mae": _finite(result.model_average.mean_absolute_error),
        },
    }


def _suite_coverage_summary(
    matrix: np.ndarray,
    models: list[str],
    evaluations: list[str],
    suite_by_evaluation: dict[str, str],
) -> dict[str, object]:
    """Summarize suite support from the filtered matrix actually being analyzed."""

    if matrix.shape != (len(models), len(evaluations)):
        raise ValueError("matrix shape must match model and evaluation labels")
    missing = [evaluation for evaluation in evaluations if evaluation not in suite_by_evaluation]
    if missing:
        raise ValueError(f"missing suite metadata for evaluations: {missing}")

    represented_suites = sorted({suite_by_evaluation[value] for value in evaluations})
    columns_by_suite = {
        suite: np.asarray(
            [suite_by_evaluation[evaluation] == suite for evaluation in evaluations],
            dtype=bool,
        )
        for suite in represented_suites
    }
    fully_represented_models = [
        model
        for row, model in enumerate(models)
        if represented_suites
        and all(np.isfinite(matrix[row, columns]).any() for columns in columns_by_suite.values())
    ]
    return {
        "represented_suites": represented_suites,
        "n_represented_suites": len(represented_suites),
        "models_with_all_represented_suites": fully_represented_models,
        "n_models_with_all_represented_suites": len(fully_represented_models),
    }


def _eval_task(matrix: np.ndarray, probes: tuple[int, ...], rank: int) -> ProbeEvaluation:
    return evaluate_global_probes(matrix, probes, rank=rank)


def _parallel_evaluate(
    executor: ProcessPoolExecutor,
    matrix: np.ndarray,
    probe_sets: list[tuple[int, ...]],
    rank: int,
) -> list[ProbeEvaluation]:
    futures = [executor.submit(_eval_task, matrix, probes, rank) for probes in probe_sets]
    return [future.result() for future in futures]


def _greedy(
    matrix: np.ndarray,
    *,
    rank: int,
    max_probes: int,
    executor: ProcessPoolExecutor,
    evaluations: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected: list[int] = []
    remaining = list(range(matrix.shape[1]))
    trajectory: list[dict[str, object]] = []
    first_step: list[dict[str, object]] = []
    for step in range(1, max_probes + 1):
        probe_sets = [tuple([*selected, candidate]) for candidate in remaining]
        results = _parallel_evaluate(executor, matrix, probe_sets, rank)
        scores = [result.parity.median_absolute_error for result in results]
        best_position = min(range(len(remaining)), key=lambda pos: (scores[pos], pos))
        if step == 1:
            first_step = [
                {
                    "evaluation_index": candidate,
                    "evaluation_id": evaluations[candidate],
                    **_summary(result),
                }
                for candidate, result in zip(remaining, results)
            ]
        candidate_table = [
            {
                "evaluation_index": candidate,
                "evaluation_id": evaluations[candidate],
                "parity_medae": _finite(result.parity.median_absolute_error),
            }
            for candidate, result in zip(remaining, results)
        ]
        selected.append(remaining.pop(best_position))
        best = results[best_position]
        record = {
            "step": step,
            "added_evaluation_index": selected[-1],
            "added_evaluation_id": evaluations[selected[-1]],
            "probe_indices": selected.copy(),
            "probe_ids": [evaluations[index] for index in selected],
            **_summary(best),
            "candidate_results": candidate_table,
        }
        trajectory.append(record)
        print(
            f"all-known step {step}: {record['added_evaluation_id']} "
            f"MedAE={best.parity.median_absolute_error:.4f}",
            flush=True,
        )
    return trajectory, first_step


def _heldout_evaluate(
    matrix: np.ndarray,
    probes: tuple[int, ...],
    train_indices: tuple[int, ...],
    validation_indices: tuple[int, ...],
    rank: int,
) -> dict[str, object]:
    observed = np.isfinite(matrix)
    probe_set = set(probes)
    hidden_errors: list[float] = []
    parity_errors: list[float] = []
    average_errors: list[float] = []
    revealed = 0
    for target in validation_indices:
        target_columns = np.flatnonzero(observed[target])
        hidden = [int(j) for j in target_columns if int(j) not in probe_set]
        known = [int(j) for j in target_columns if int(j) in probe_set]
        revealed += len(known)
        parity_errors.extend([0.0] * len(known))
        predicted_by_column = {j: float(matrix[target, j]) for j in known}
        if hidden:
            train = np.full_like(matrix, np.nan)
            train[list(train_indices), :] = matrix[list(train_indices), :]
            if known:
                train[target, known] = matrix[target, known]
            completed = complete(train, rank=rank, allow_empty_rows=True)
            for column in hidden:
                error = abs(float(completed[target, column] - matrix[target, column]))
                hidden_errors.append(error)
                parity_errors.append(error)
                predicted_by_column[column] = float(completed[target, column])
        if predicted_by_column:
            actual_average = float(np.mean(matrix[target, target_columns]))
            predicted_average = float(
                np.mean([predicted_by_column[int(j)] for j in target_columns])
            )
            average_errors.append(abs(predicted_average - actual_average))

    def stats(values: list[float]) -> dict[str, object]:
        array = np.asarray(values, dtype=float)
        return {
            "n": int(array.size),
            "medae": _finite(float(np.median(array))) if array.size else None,
            "mae": _finite(float(np.mean(array))) if array.size else None,
        }

    return {
        "probe_indices": list(probes),
        "n_revealed_cells": revealed,
        "parity": stats(parity_errors),
        "hidden_only": stats(hidden_errors),
        "model_average": stats(average_errors),
    }


def _heldout_task(args: tuple[object, ...]) -> dict[str, object]:
    return _heldout_evaluate(*args)  # type: ignore[arg-type]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "data" / "scores.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "probe_selection_results_rank1.json",
    )
    parser.add_argument(
        "--informativeness-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "probe_informativeness_rank1.csv",
    )
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--max-probes", type=int, default=10)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, min(28, (os.cpu_count() or 2) - 1)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores)
    matrix, models, evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(matrix, models, evaluations)
    metadata: dict[str, tuple[str, str]] = {}
    for score in scores:
        if score.evaluation_id in evaluations:
            metadata[score.evaluation_id] = (score.suite_id, score.metric)
    suite_coverage = _suite_coverage_summary(
        matrix,
        models,
        evaluations,
        {evaluation: suite for evaluation, (suite, _) in metadata.items()},
    )

    baseline = evaluate_column_median_baseline(matrix)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        trajectory, one_probe = _greedy(
            matrix,
            rank=args.rank,
            max_probes=args.max_probes,
            executor=executor,
            evaluations=evaluations,
        )
        random_prefixes = random_global_probe_prefixes(
            len(evaluations),
            max_probes=args.max_probes,
            repeats=args.random_repeats,
            seed=args.seed,
        )
        random_sets = [probes for repeat in random_prefixes for probes in repeat]
        random_results = _parallel_evaluate(executor, matrix, random_sets, args.rank)

        rng = np.random.RandomState(args.seed)
        permutation = rng.permutation(len(models))
        n_train = min(max(1, round(0.7 * len(models))), len(models) - 1)
        train_indices = tuple(sorted(int(i) for i in permutation[:n_train]))
        validation_indices = tuple(sorted(int(i) for i in permutation[n_train:]))
        train_matrix = matrix[list(train_indices), :]
        heldout_train_trajectory, _ = _greedy(
            train_matrix,
            rank=args.rank,
            max_probes=args.max_probes,
            executor=executor,
            evaluations=evaluations,
        )
        heldout_args = [
            (
                matrix,
                tuple(int(i) for i in step["probe_indices"]),
                train_indices,
                validation_indices,
                args.rank,
            )
            for step in heldout_train_trajectory
        ]
        heldout_results = list(executor.map(_heldout_task, heldout_args))

    random_rows: list[dict[str, object]] = []
    cursor = 0
    for repeat, prefixes in enumerate(random_prefixes):
        for k, probes in enumerate(prefixes, start=1):
            random_rows.append(
                {
                    "repeat": repeat,
                    "k": k,
                    "probe_indices": list(probes),
                    "probe_ids": [evaluations[index] for index in probes],
                    **_summary(random_results[cursor]),
                }
            )
            cursor += 1

    baseline_medae = baseline.parity.median_absolute_error
    informativeness = []
    for row in one_probe:
        index = int(row["evaluation_index"])
        evaluation_id = evaluations[index]
        parity_medae = float(row["parity"]["medae"])  # type: ignore[index]
        suite, metric = metadata[evaluation_id]
        informativeness.append(
            {
                "evaluation_index": index,
                "evaluation_id": evaluation_id,
                "suite_id": suite,
                "metric": metric,
                "models_with_score": int(np.sum(np.isfinite(matrix[:, index]))),
                "model_coverage": float(np.mean(np.isfinite(matrix[:, index]))),
                "parity_medae": parity_medae,
                "hidden_only_medae": row["hidden_only"]["medae"],  # type: ignore[index]
                "model_average_mae": row["model_average"]["mae"],  # type: ignore[index]
                "improvement_over_column_median": baseline_medae - parity_medae,
            }
        )
    informativeness.sort(key=lambda row: (float(row["parity_medae"]), int(row["evaluation_index"])))
    for position, row in enumerate(informativeness, start=1):
        row["informativeness_rank"] = position

    args.informativeness_output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "informativeness_rank", "evaluation_id", "suite_id", "metric",
        "models_with_score", "model_coverage", "parity_medae",
        "hidden_only_medae", "model_average_mae",
        "improvement_over_column_median",
    ]
    with args.informativeness_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(informativeness)

    payload = {
        "schema_version": 1,
        "protocol": "benchpress_all_known_probe_cells_zero_error_plus_strict_diagnostics_v1",
        "matrix": {
            "n_models": len(models),
            "n_evaluations": len(evaluations),
            "n_observed": int(np.sum(np.isfinite(matrix))),
            "density": float(np.mean(np.isfinite(matrix))),
            "suite_coverage": suite_coverage,
        },
        "configuration": {
            "rank": args.rank,
            "max_probes": args.max_probes,
            "random_repeats": args.random_repeats,
            "seed": args.seed,
            "workers": args.workers,
            "greedy_objective": "pooled all-known-cell MedAE including exact probe cells",
            "random_seed_formula": "RandomState((seed + repeat) * 100000)",
        },
        "baseline": _summary(baseline),
        "all_known_greedy": trajectory,
        "random_global_prefixes": random_rows,
        "heldout_model": {
            "train_fraction": 0.7,
            "train_models": [models[i] for i in train_indices],
            "validation_models": [models[i] for i in validation_indices],
            "train_selected_trajectory": heldout_train_trajectory,
            "validation": [
                {
                    "step": step["step"],
                    "added_evaluation_id": step["added_evaluation_id"],
                    "probe_ids": step["probe_ids"],
                    **result,
                }
                for step, result in zip(heldout_train_trajectory, heldout_results)
            ],
        },
        "informativeness": informativeness,
        "provenance": {
            "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "benchpress_commit_audited": "0a684b63ee0e4a401cb907a3827a82ea997d74c4",
        },
        "caveats": [
            "The hero-parity curve is transductive and includes revealed probes as zero-error targets.",
            "The held-out-model primary curve excludes revealed probes and isolates each validation row.",
            (
                "The current matrix is suite-blocked across "
                f"{suite_coverage['n_represented_suites']} represented suites; "
                f"{suite_coverage['n_models_with_all_represented_suites']} of {len(models)} "
                "supported models have at least one observed score in every represented suite."
            ),
            "Normalized F1, Pearson r, and robustness index are not a validated common clinical utility scale.",
            "No low-cost curve is claimed because pathology acquisition/runtime costs have not been audited.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "informativeness": str(args.informativeness_output)}, indent=2))


if __name__ == "__main__":
    main()
