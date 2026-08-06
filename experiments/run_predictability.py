#!/usr/bin/env python3
"""Run BenchPress's ten-seed hide-half predictability experiment on pathology."""

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
from pathopress.predictability import (  # noqa: E402
    aggregate_raw_predictions,
    holdout_half_per_model,
)


def _run_seed(job: tuple[np.ndarray, int, int]) -> list[tuple[int, int, float, float]]:
    matrix, seed, rank = job
    train, heldout = holdout_half_per_model(matrix, np.random.RandomState(seed))
    supported_rows = np.any(np.isfinite(train), axis=1)
    supported_columns = np.any(np.isfinite(train), axis=0)
    row_ids = np.flatnonzero(supported_rows)
    column_ids = np.flatnonzero(supported_columns)
    row_map = {int(old): new for new, old in enumerate(row_ids)}
    column_map = {int(old): new for new, old in enumerate(column_ids)}
    predicted = complete(train[np.ix_(row_ids, column_ids)], rank=rank)
    rows = []
    for i, columns in heldout.items():
        if i not in row_map:
            continue
        for j in columns:
            if j not in column_map:
                continue
            estimate = float(predicted[row_map[i], column_map[j]])
            actual = float(matrix[i, j])
            if np.isfinite(actual) and np.isfinite(estimate):
                rows.append((i, j, actual, estimate))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "data" / "scores.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "predictability_results_rank1.json",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "predictability_predictions_rank1.csv",
    )
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) - 1)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores)
    matrix, models, evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(matrix, models, evaluations)
    evaluation_suite = {}
    for score in scores:
        if score.evaluation_id in evaluations:
            evaluation_suite.setdefault(score.evaluation_id, score.suite_id)

    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))
    jobs = [(matrix, seed, args.rank) for seed in seeds]
    raw = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for seed, seed_rows in zip(seeds, executor.map(_run_seed, jobs)):
            for i, j, actual, predicted in seed_rows:
                raw.append(
                    {
                        "seed": seed,
                        "model_id": models[i],
                        "evaluation_id": evaluations[j],
                        "suite_id": evaluation_suite[evaluations[j]],
                        "actual": actual,
                        "predicted": predicted,
                        "absolute_error": abs(predicted - actual),
                    }
                )

    by_evaluation = aggregate_raw_predictions(
        raw, group_key="evaluation_id", group_ids=evaluations
    )
    for row in by_evaluation:
        row["suite_id"] = evaluation_suite[str(row["evaluation_id"])]
    by_model = aggregate_raw_predictions(raw, group_key="model_id", group_ids=models)
    dominant_suite = {}
    for model_id in models:
        counts = {}
        for row in raw:
            if row["model_id"] == model_id:
                suite = str(row["suite_id"])
                counts[suite] = counts.get(suite, 0) + 1
        if counts:
            dominant_suite[model_id] = max(sorted(counts), key=counts.get)
    for row in by_model:
        row["suite_id"] = dominant_suite.get(str(row["model_id"]), "")
    by_evaluation.sort(key=lambda row: float(row["medape"]))
    by_model.sort(key=lambda row: float(row["medape"]))

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "seed",
        "model_id",
        "evaluation_id",
        "suite_id",
        "actual",
        "predicted",
        "absolute_error",
    )
    with args.raw_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in raw:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    payload = {
        "schema_version": 1,
        "experiment": "benchpress_hide_half_predictability",
        "method": f"PathoPress logit bias ALS rank {args.rank}, lambda 0.1",
        "protocol": {
            "split": "For each seed and each model with >=4 observations, shuffle observed columns and hide floor(n/2).",
            "seeds": seeds,
            "aggregation": "Pool raw held-out predictions over seeds, separately by evaluation and model.",
            "upstream_mapping": "appendix_e_sec6_trust/prediction_error_analysis/per_benchmark_predictability",
        },
        "matrix": {
            "n_models": len(models),
            "n_evaluations": len(evaluations),
            "n_observed": int(np.isfinite(matrix).sum()),
        },
        "input": {
            "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
            "raw_predictions": str(args.raw_output.relative_to(PROJECT_ROOT)),
        },
        "n_raw_predictions": len(raw),
        "by_evaluation": by_evaluation,
        "by_model": by_model,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "n_raw": len(raw)}, indent=2))


if __name__ == "__main__":
    main()
