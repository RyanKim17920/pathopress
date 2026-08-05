#!/usr/bin/env python3
"""Exact BenchPress-style raw/logit Soft-Impute rank sweep."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pathopress.completion import complete_soft_impute  # noqa: E402
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from run_benchpress_style import make_folds, metrics  # noqa: E402


def main() -> None:
    scores_path = PROJECT_ROOT / "data" / "scores.csv"
    output_path = PROJECT_ROOT / "experiments" / "soft_impute_rank_sweep_results.json"
    scores = load_scores(scores_path)
    matrix, models, evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(matrix, models, evaluations)
    ranks = tuple(range(1, 11))
    transforms = ("identity", "logit")
    folds = [
        (seed, fold, train, held)
        for seed in range(42, 52)
        for fold, (train, held) in enumerate(make_folds(matrix, n_folds=3, seed=seed))
    ]
    results: dict[str, dict[str, object]] = {}
    for transform in transforms:
        results[transform] = {}
        for rank in ranks:
            actual: list[float] = []
            predicted: list[float] = []
            fold_medae: list[float] = []
            for seed, fold, train, held in folds:
                estimate = complete_soft_impute(train, rank=rank, transform=transform)
                fold_actual = [float(matrix[i, j]) for i, j in held]
                fold_prediction = [float(estimate[i, j]) for i, j in held]
                actual.extend(fold_actual)
                predicted.extend(fold_prediction)
                fold_medae.append(float(metrics(fold_actual, fold_prediction)["medae"]))
            results[transform][str(rank)] = {
                "pooled": metrics(actual, predicted),
                "fold_medae_median": round(float(np.median(fold_medae)), 6),
                "fold_medae_q1": round(float(np.percentile(fold_medae, 25)), 6),
                "fold_medae_q3": round(float(np.percentile(fold_medae, 75)), 6),
            }
    payload = {
        "schema_version": 1,
        "description": "BenchPress Section 3 raw/logit iterative truncated-SVD rank sweep on PathoPress.",
        "matrix": {
            "n_models": len(models),
            "n_evaluations": len(evaluations),
            "n_observed": int(np.sum(np.isfinite(matrix))),
        },
        "configuration": {
            "ranks": list(ranks),
            "transforms": list(transforms),
            "n_seeds": 10,
            "n_folds": 3,
            "base_seed": 42,
            "soft_impute_max_iterations": 100,
            "soft_impute_tolerance": 1e-4,
        },
        "input": {"scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest()},
        "results": results,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
