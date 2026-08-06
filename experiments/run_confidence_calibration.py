#!/usr/bin/env python3
"""BenchPress-style cross-fitted confidence calibration for PathoPress.

The target point predictor is the pathology-selected rank-1 Logit Bias ALS.
As in BenchPress, three leave-fold-out risk models use ensemble disagreement,
training-matrix support, or both, and predict log1p absolute held-out error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Matrix fits are parallelized across folds. Prevent each worker's BLAS from
# starting another full thread pool and oversubscribing the host.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.run_benchpress_style import make_folds  # noqa: E402
from pathopress.completion import complete, complete_soft_impute  # noqa: E402
from pathopress.confidence import (  # noqa: E402
    confidence_feature_sets,
    conformal_interval,
    crossfit_error_risk,
    stack_features,
    structural_support_features_for_cells,
    summarize_confidence_method,
)
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402


METHODS = ("disagreement", "structural_support", "combined_risk_model")


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _supported_completion(
    training: np.ndarray, *, rank: int, regularization: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_ids = np.flatnonzero(np.any(np.isfinite(training), axis=1))
    column_ids = np.flatnonzero(np.any(np.isfinite(training), axis=0))
    completed = complete(
        training[np.ix_(row_ids, column_ids)],
        rank=rank,
        regularization=regularization,
    )
    return completed, row_ids, column_ids


def _supported_soft_impute(
    training: np.ndarray, *, rank: int, transform: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_ids = np.flatnonzero(np.any(np.isfinite(training), axis=1))
    column_ids = np.flatnonzero(np.any(np.isfinite(training), axis=0))
    completed = complete_soft_impute(
        training[np.ix_(row_ids, column_ids)], rank=rank, transform=transform
    )
    return completed, row_ids, column_ids


def _take_cells(
    result: tuple[np.ndarray, np.ndarray, np.ndarray], cells: list[tuple[int, int]]
) -> np.ndarray:
    completed, row_ids, column_ids = result
    row_map = {int(old): new for new, old in enumerate(row_ids)}
    column_map = {int(old): new for new, old in enumerate(column_ids)}
    return np.asarray(
        [completed[row_map[row], column_map[column]] for row, column in cells], dtype=float
    )


def _fold_features(
    job: tuple[int, int, int, np.ndarray, np.ndarray, list[tuple[int, int]]]
) -> dict[str, object]:
    fold_id, seed, fold, full, training, held = job
    supported_rows = np.any(np.isfinite(training), axis=1)
    supported_columns = np.any(np.isfinite(training), axis=0)
    cells = [
        (int(row), int(column))
        for row, column in held
        if supported_rows[row] and supported_columns[column]
    ]
    target_result = _supported_completion(training, rank=1, regularization=0.1)
    target = _take_cells(target_result, cells)

    # BenchPress uses same-family regularization variants for local sensitivity.
    hp_stack = np.stack(
        [
            _take_cells(_supported_completion(training, rank=1, regularization=value), cells)
            if value != 0.1
            else target
            for value in (0.01, 0.1, 1.0)
        ]
    )
    # Pathology adaptation: the upstream transform/method grid is replaced by
    # every strong full-coverage alternative already implemented locally.
    strong_stack = np.stack(
        [
            _take_cells(_supported_completion(training, rank=0), cells),
            _take_cells(_supported_completion(training, rank=2), cells),
            _take_cells(_supported_soft_impute(training, rank=1, transform="logit"), cells),
            _take_cells(_supported_soft_impute(training, rank=2, transform="logit"), cells),
            _take_cells(_supported_soft_impute(training, rank=1, transform="identity"), cells),
            _take_cells(_supported_soft_impute(training, rank=2, transform="identity"), cells),
        ]
    )
    features = confidence_feature_sets(
        stack_features(hp_stack, target),
        stack_features(strong_stack, target),
        structural_support_features_for_cells(training, cells),
    )
    return {
        "fold_id": fold_id,
        "seed": seed,
        "fold": fold,
        "rows": np.asarray([row for row, _ in cells], dtype=int),
        "columns": np.asarray([column for _, column in cells], dtype=int),
        "actual": np.asarray([full[row, column] for row, column in cells], dtype=float),
        "predicted": target,
        "features": features,
    }


def _concatenate_features(
    records: list[dict[str, object]], method: str
) -> dict[str, np.ndarray]:
    first = records[0]["features"][method]  # type: ignore[index]
    return {
        name: np.concatenate(
            [record["features"][method][name] for record in records]  # type: ignore[index]
        )
        for name in sorted(first)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=ROOT / "data" / "scores.csv")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments" / "confidence_calibration_rank1.json"
    )
    parser.add_argument(
        "--cells", type=Path, default=ROOT / "experiments" / "confidence_cells_rank1.csv"
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores)
    matrix, models, evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(matrix, models, evaluations)
    metadata = {}
    for score in scores:
        if score.evaluation_id in evaluations:
            metadata.setdefault(score.evaluation_id, (score.suite_id, score.metric))

    jobs = []
    fold_id = 0
    for seed_offset in range(args.n_seeds):
        seed = args.base_seed + seed_offset
        for fold, (training, held) in enumerate(make_folds(matrix, n_folds=args.n_folds, seed=seed)):
            jobs.append((fold_id, seed, fold, matrix, training, held))
            fold_id += 1
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_fold_features, job): job[0] for job in jobs}
        records = []
        for completed, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            print(f"fold feature stacks: {completed}/{len(jobs)} complete", flush=True)
    records.sort(key=lambda record: int(record["fold_id"]))

    fold_ids = np.concatenate(
        [np.full(len(record["actual"]), int(record["fold_id"]), dtype=int) for record in records]
    )
    seeds = np.concatenate(
        [np.full(len(record["actual"]), int(record["seed"]), dtype=int) for record in records]
    )
    folds = np.concatenate(
        [np.full(len(record["actual"]), int(record["fold"]), dtype=int) for record in records]
    )
    rows = np.concatenate([record["rows"] for record in records])
    columns = np.concatenate([record["columns"] for record in records])
    actual = np.concatenate([record["actual"] for record in records])
    predicted = np.concatenate([record["predicted"] for record in records])

    all_features = {method: _concatenate_features(records, method) for method in METHODS}
    uncertainties = {}
    risk_metadata = {}
    intervals = {}
    summaries = {}
    for index, method in enumerate(METHODS):
        uncertainty, feature_names, selected = crossfit_error_risk(
            actual,
            predicted,
            fold_ids,
            all_features[method],
            seed=args.base_seed + 100 * index,
            label=method,
            verbose=True,
        )
        if not np.all(np.isfinite(uncertainty)):
            raise ValueError(f"cross-fit uncertainty is incomplete for {method}")
        lower, upper, scale = conformal_interval(actual, predicted, uncertainty, fold_ids)
        uncertainties[method] = uncertainty
        intervals[method] = (lower, upper, scale)
        risk_metadata[method] = {
            "feature_names": feature_names,
            "selected_risk_model_by_fold": selected,
        }
        summaries[method] = summarize_confidence_method(
            actual, predicted, fold_ids, uncertainty
        )

    raw_feature_columns = {}
    for method in ("disagreement", "structural_support"):
        for name, values in all_features[method].items():
            raw_feature_columns[f"{method}_{name}"] = values
    cell_fields = [
        "seed",
        "fold",
        "crossfit_fold_id",
        "model_id",
        "evaluation_id",
        "suite_id",
        "metric",
        "actual_normalized_score",
        "predicted_normalized_score",
        "absolute_error",
        *sorted(raw_feature_columns),
    ]
    for method in METHODS:
        cell_fields.extend(
            [f"{method}_risk", f"{method}_lower_90", f"{method}_upper_90", f"{method}_conformal_scale"]
        )
    args.cells.parent.mkdir(parents=True, exist_ok=True)
    with args.cells.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cell_fields, lineterminator="\n")
        writer.writeheader()
        for index in range(len(actual)):
            evaluation = evaluations[int(columns[index])]
            suite, metric = metadata[evaluation]
            row = {
                "seed": int(seeds[index]),
                "fold": int(folds[index]),
                "crossfit_fold_id": int(fold_ids[index]),
                "model_id": models[int(rows[index])],
                "evaluation_id": evaluation,
                "suite_id": suite,
                "metric": metric,
                "actual_normalized_score": f"{actual[index]:.6f}",
                "predicted_normalized_score": f"{predicted[index]:.6f}",
                "absolute_error": f"{abs(predicted[index] - actual[index]):.6f}",
                **{name: f"{values[index]:.6f}" for name, values in raw_feature_columns.items()},
            }
            for method in METHODS:
                lower, upper, scale = intervals[method]
                row.update(
                    {
                        f"{method}_risk": f"{uncertainties[method][index]:.6f}",
                        f"{method}_lower_90": f"{lower[index]:.6f}",
                        f"{method}_upper_90": f"{upper[index]:.6f}",
                        f"{method}_conformal_scale": f"{scale[index]:.6f}",
                    }
                )
            writer.writerow(row)

    payload = {
        "schema_version": 1,
        "description": "BenchPress-style cross-fitted confidence calibration for pathology matrix completion.",
        "matrix": {
            "n_models": len(models),
            "n_evaluations": len(evaluations),
            "n_observed": int(np.isfinite(matrix).sum()),
            "density": float(np.isfinite(matrix).mean()),
        },
        "configuration": {
            "target_predictor": "logit bias ALS rank=1 regularization=0.1",
            "n_seeds": args.n_seeds,
            "n_folds": args.n_folds,
            "base_seed": args.base_seed,
            "n_prediction_instances": len(actual),
            "risk_target": "log1p(abs(predicted_normalized_score - actual_normalized_score))",
            "confidence_methods": list(METHODS),
            "hp_disagreement_variants": [
                "rank1_regularization_0.01",
                "rank1_regularization_0.1",
                "rank1_regularization_1.0",
            ],
            "strong_method_variants": [
                "bias_als_rank0",
                "bias_als_rank2",
                "soft_impute_logit_rank1",
                "soft_impute_logit_rank2",
                "soft_impute_identity_rank1",
                "soft_impute_identity_rank2",
            ],
            "conformal_protocol": "leave the target point-prediction fold out when fitting the 90% scale",
        },
        "risk_models": risk_metadata,
        "confidence_methods": summaries,
        "input": {
            "scores_path": _display_path(args.scores),
            "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
            "cells_path": _display_path(args.cells),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "pathology_adaptations": [
            "The target rank is 1 because pathology cross-validation selected rank 1; BenchPress uses rank 2.",
            "BenchPress's twelve strong transform/KNN alternatives are replaced by the six full-coverage Bias ALS and Soft-Impute alternatives implemented in PathoPress.",
            "Errors and interval widths are normalized-score points across heterogeneous pathology endpoints, not clinical utility units.",
            "The same observed cell appears once per seed, matching BenchPress's repeated within-model fold experiment; this is not temporal or external-institution validation.",
            "The risk models are evaluation-only cross-fits. A deploy-time confidence artifact for genuinely new cells is intentionally not trained here.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "methods": summaries}, indent=2))


if __name__ == "__main__":
    main()
