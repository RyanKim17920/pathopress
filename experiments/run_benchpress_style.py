#!/usr/bin/env python3
"""BenchPress-style 10-seed x 3-fold evaluation for PathoPress.

Every observed cell is assigned to one within-model fold per seed. Aggregate
rank results and fold summaries are written to JSON; selected-rank out-of-fold
point predictions are written to CSV for diagnostic plots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathopress.completion import complete  # noqa: E402
from pathopress.artifacts import load_fold_artifact, sha256_file  # noqa: E402
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.metrics import median_absolute_percentage_error  # noqa: E402


def make_folds(matrix: np.ndarray, *, n_folds: int, seed: int):
    """Match BenchPress's within-model fold construction."""
    rng = np.random.RandomState(seed)
    assignments: list[list[int]] = []
    for row in range(matrix.shape[0]):
        observed = list(np.flatnonzero(np.isfinite(matrix[row])))
        rng.shuffle(observed)
        assignments.append(observed)

    folds = []
    for fold in range(n_folds):
        train = matrix.copy()
        held: list[tuple[int, int]] = []
        for row, observed in enumerate(assignments):
            if not observed:
                continue
            fold_size = max(1, len(observed) // n_folds)
            start = fold * fold_size
            end = start + fold_size if fold < n_folds - 1 else len(observed)
            for column in observed[start:end]:
                train[row, column] = np.nan
                held.append((row, column))
        folds.append((train, held))
    return folds


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float | int]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    error = np.abs(p - a)
    return {
        "n": int(error.size),
        "mae": round(float(np.mean(error)), 6),
        "medae": round(float(np.median(error)), 6),
        "medape": round(median_absolute_percentage_error(a, p), 6),
        "within_1": round(float(np.mean(error <= 1.0)), 6),
        "within_3": round(float(np.mean(error <= 3.0)), 6),
        "within_5": round(float(np.mean(error <= 5.0)), 6),
        "within_10": round(float(np.mean(error <= 10.0)), 6),
    }


def _fit_fold(job: tuple[np.ndarray, np.ndarray, list[tuple[int, int]], int]):
    """Fit one rank/fold job; kept top-level for process-pool portability."""
    matrix, train, held, rank = job
    row_supported = np.any(np.isfinite(train), axis=1)
    col_supported = np.any(np.isfinite(train), axis=0)
    row_ids = np.flatnonzero(row_supported)
    col_ids = np.flatnonzero(col_supported)
    row_map = {int(old): new for new, old in enumerate(row_ids)}
    col_map = {int(old): new for new, old in enumerate(col_ids)}
    predicted = complete(train[np.ix_(row_ids, col_ids)], rank=rank)
    baseline = np.nanmedian(train, axis=0)
    rows: list[tuple[int, int, float, float, float]] = []
    for i, j in held:
        if not row_supported[i] or not col_supported[j]:
            continue
        rows.append(
            (
                i,
                j,
                float(matrix[i, j]),
                float(predicted[row_map[i], col_map[j]]),
                float(baseline[j]),
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "data" / "scores.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "benchpress_style_results.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "benchpress_style_predictions_rank1.csv",
    )
    parser.add_argument("--prediction-rank", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, min(28, (os.cpu_count() or 2) - 1)))
    parser.add_argument(
        "--folds-artifact",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "folds_s10_f3_bs42.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores)
    matrix, models, evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(matrix, models, evaluations)
    metadata: dict[str, tuple[str, str]] = {}
    for score in scores:
        if score.evaluation_id in evaluations:
            previous = metadata.setdefault(
                score.evaluation_id, (score.suite_id, score.metric)
            )
            if previous != (score.suite_id, score.metric):
                raise ValueError(f"inconsistent metadata for {score.evaluation_id}")

    ranks = tuple(range(0, 11))
    by_rank: dict[str, dict[str, object]] = {}
    prediction_rows: list[dict[str, object]] = []
    if args.folds_artifact.exists():
        fold_inputs = load_fold_artifact(
            args.folds_artifact, matrix, models, evaluations
        )
        expected_records = args.n_seeds * args.n_folds
        if len(fold_inputs) != expected_records:
            raise ValueError(
                f"fold artifact has {len(fold_inputs)} records, expected {expected_records}"
            )
    else:
        fold_inputs = [
            (args.base_seed + seed_offset, fold, train, held)
            for seed_offset in range(args.n_seeds)
            for fold, (train, held) in enumerate(
                make_folds(
                    matrix,
                    n_folds=args.n_folds,
                    seed=args.base_seed + seed_offset,
                )
            )
        ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for rank in ranks:
            pooled_actual: list[float] = []
            pooled_prediction: list[float] = []
            fold_metrics: list[dict[str, object]] = []
            suite_values: dict[str, tuple[list[float], list[float]]] = {}
            jobs = [(matrix, train, held, rank) for _, _, train, held in fold_inputs]
            fitted = executor.map(_fit_fold, jobs)
            for (seed, fold, _train, _held), fitted_rows in zip(fold_inputs, fitted):
                fold_actual: list[float] = []
                fold_prediction: list[float] = []
                for i, j, actual, estimate, baseline in fitted_rows:
                    suite, metric = metadata[evaluations[j]]
                    fold_actual.append(actual)
                    fold_prediction.append(estimate)
                    pooled_actual.append(actual)
                    pooled_prediction.append(estimate)
                    suite_actual, suite_pred = suite_values.setdefault(suite, ([], []))
                    suite_actual.append(actual)
                    suite_pred.append(estimate)
                    if rank == args.prediction_rank:
                        prediction_rows.append(
                            {
                                "seed": seed,
                                "fold": fold,
                                "model_id": models[i],
                                "evaluation_id": evaluations[j],
                                "suite_id": suite,
                                "metric": metric,
                                "actual_normalized_score": f"{actual:.6f}",
                                "predicted_normalized_score": f"{estimate:.6f}",
                                "absolute_error": f"{abs(estimate - actual):.6f}",
                                "column_median_baseline": f"{baseline:.6f}",
                            }
                        )
                fold_metrics.append(
                    {"seed": seed, "fold": fold, **metrics(fold_actual, fold_prediction)}
                )
            pooled = metrics(pooled_actual, pooled_prediction)
            fold_medae = np.asarray([float(row["medae"]) for row in fold_metrics])
            by_rank[str(rank)] = {
                "pooled": pooled,
                "fold_medae": {
                    "median": round(float(np.median(fold_medae)), 6),
                    "q1": round(float(np.percentile(fold_medae, 25)), 6),
                    "q3": round(float(np.percentile(fold_medae, 75)), 6),
                },
                "by_suite": {
                    suite: metrics(actual, predicted)
                    for suite, (actual, predicted) in sorted(suite_values.items())
                },
                "folds": fold_metrics,
            }

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    prediction_fields = (
        "seed",
        "fold",
        "model_id",
        "evaluation_id",
        "suite_id",
        "metric",
        "actual_normalized_score",
        "predicted_normalized_score",
        "absolute_error",
        "column_median_baseline",
    )
    baseline_actual = [float(row["actual_normalized_score"]) for row in prediction_rows]
    baseline_prediction = [float(row["column_median_baseline"]) for row in prediction_rows]
    baseline_by_suite: dict[str, tuple[list[float], list[float]]] = {}
    for row in prediction_rows:
        actuals, predictions = baseline_by_suite.setdefault(
            str(row["suite_id"]), ([], [])
        )
        actuals.append(float(row["actual_normalized_score"]))
        predictions.append(float(row["column_median_baseline"]))
    with args.predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=prediction_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    payload = {
        "schema_version": 1,
        "description": "BenchPress-style within-model cross-validation on normalized pathology scores.",
        "matrix": {
            "n_models": len(models),
            "n_evaluations": len(evaluations),
            "n_observed": int(np.sum(np.isfinite(matrix))),
            "density": round(float(np.mean(np.isfinite(matrix))), 6),
        },
        "configuration": {
            "n_seeds": args.n_seeds,
            "n_folds": args.n_folds,
            "base_seed": args.base_seed,
            "ranks": list(ranks),
            "prediction_rank": args.prediction_rank,
            "aggregation_note": (
                "pooled metrics include every held-out prediction instance; fold_medae.median "
                "matches the style of BenchPress's headline fold aggregation"
            ),
        },
        "input": {
            "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
            "predictions_path": str(args.predictions.resolve().relative_to(PROJECT_ROOT)),
            "folds_artifact": str(args.folds_artifact.resolve().relative_to(PROJECT_ROOT))
            if args.folds_artifact.exists()
            else None,
            "folds_sha256": sha256_file(args.folds_artifact)
            if args.folds_artifact.exists()
            else None,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "by_rank": by_rank,
        "column_median_baseline": {
            "pooled": metrics(baseline_actual, baseline_prediction),
            "by_suite": {
                suite: metrics(actual, predicted)
                for suite, (actual, predicted) in sorted(baseline_by_suite.items())
            },
        },
        "caveats": [
            "Errors are normalized-score points, not a common pathology utility unit.",
            "The 10 seeds alter fold assignment; the ALS ensemble itself is optimization stabilization, not uncertainty estimation.",
            "Ranks were compared on these same folds without nested model selection.",
            "The matrix contains Patho-Bench, EVA, HEST, THUNDER, and PathoROB score columns.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "prediction_rank": args.prediction_rank,
                "selected": by_rank[str(args.prediction_rank)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
