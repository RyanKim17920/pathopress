#!/usr/bin/env python3
"""BenchPress-style benchmark/model predictability-factor correlations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.error_analysis import (  # noqa: E402
    best_neighbor_rows,
    low_rank_r2,
    pairwise_abs_correlation,
    spearman_test,
)
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402


def _task_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["evaluation_id"]: row for row in csv.DictReader(handle)}


def _correlations(rows: list[dict], features: list[str]) -> dict[str, dict]:
    return {
        target: {
            feature: spearman_test(
                np.asarray([float(row[feature]) for row in rows]),
                np.asarray([float(row[target]) for row in rows]),
            )
            for feature in features
        }
        for target in ("medape", "medae")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=ROOT / "data" / "scores.csv")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data" / "tasks.csv")
    parser.add_argument(
        "--predictability",
        type=Path,
        default=ROOT / "experiments" / "predictability_results_rank1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "error_analysis_rank1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores)
    matrix, models, evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(matrix, models, evaluations)
    prediction = json.loads(args.predictability.read_text(encoding="utf-8"))
    evaluation_error = {row["evaluation_id"]: row for row in prediction["by_evaluation"]}
    model_error = {row["model_id"]: row for row in prediction["by_model"]}
    tasks = _task_metadata(args.tasks)

    evaluation_r1 = low_rank_r2(matrix, rank=1, axis=0)
    evaluation_r2 = low_rank_r2(matrix, rank=2, axis=0)
    eval_corr, eval_shared = pairwise_abs_correlation(matrix, axis=0)
    eval_neighbors = best_neighbor_rows(eval_corr, eval_shared)
    family_counts = Counter(
        tasks.get(evaluation, {}).get("task_family", "unknown") for evaluation in evaluations
    )
    evaluation_rows = []
    for j, evaluation in enumerate(evaluations):
        if evaluation not in evaluation_error:
            continue
        observed = matrix[np.isfinite(matrix[:, j]), j]
        task = tasks.get(evaluation, {})
        neighbor_index = eval_neighbors[j]["best_neighbor_index"]
        evaluation_rows.append(
            {
                "evaluation_id": evaluation,
                "suite_id": task.get("suite_id", evaluation_error[evaluation].get("suite_id", "")),
                "task_family": task.get("task_family", "unknown"),
                "medape": float(evaluation_error[evaluation]["medape"]),
                "medae": float(evaluation_error[evaluation]["medae"]),
                "rank1_r2": float(evaluation_r1[j]),
                "rank2_r2": float(evaluation_r2[j]),
                "n_obs": int(len(observed)),
                "best_neighbor_id": evaluations[int(neighbor_index)] if neighbor_index is not None else None,
                "best_neighbor_abs_r": float(eval_neighbors[j]["best_neighbor_abs_r"]),
                "best_neighbor_shared": int(eval_neighbors[j]["best_neighbor_shared"]),
                "median_score": float(np.median(observed)),
                "score_std": float(np.std(observed, ddof=1)) if len(observed) > 1 else float("nan"),
                "n_same_task_family": family_counts[task.get("task_family", "unknown")] - 1,
            }
        )

    model_r1 = low_rank_r2(matrix, rank=1, axis=1)
    model_r2 = low_rank_r2(matrix, rank=2, axis=1)
    model_corr, model_shared = pairwise_abs_correlation(matrix, axis=1, min_shared=3)
    model_neighbors = best_neighbor_rows(model_corr, model_shared)
    model_rows = []
    for i, model in enumerate(models):
        if model not in model_error:
            continue
        observed = matrix[i, np.isfinite(matrix[i])]
        neighbor_index = model_neighbors[i]["best_neighbor_index"]
        model_rows.append(
            {
                "model_id": model,
                "dominant_suite_id": model_error[model].get("suite_id", ""),
                "medape": float(model_error[model]["medape"]),
                "medae": float(model_error[model]["medae"]),
                "rank1_r2": float(model_r1[i]),
                "rank2_r2": float(model_r2[i]),
                "n_obs": int(len(observed)),
                "best_peer_id": models[int(neighbor_index)] if neighbor_index is not None else None,
                "best_peer_abs_r": float(model_neighbors[i]["best_neighbor_abs_r"]),
                "best_peer_shared": int(model_neighbors[i]["best_neighbor_shared"]),
                "median_score": float(np.median(observed)),
                "score_std": float(np.std(observed, ddof=1)) if len(observed) > 1 else float("nan"),
            }
        )

    evaluation_features = [
        "rank1_r2",
        "rank2_r2",
        "n_obs",
        "best_neighbor_abs_r",
        "best_neighbor_shared",
        "median_score",
        "score_std",
        "n_same_task_family",
    ]
    model_features = [
        "rank1_r2",
        "rank2_r2",
        "n_obs",
        "best_peer_abs_r",
        "best_peer_shared",
        "median_score",
        "score_std",
    ]
    payload = {
        "schema_version": 1,
        "experiment": "benchpress_prediction_error_correlates",
        "protocol": {
            "errors": "Ten-seed hide-half-per-model predictions from predictability_results_rank1.json.",
            "low_rank_fit": "Column z-score, zero-impute missing values, truncated SVD, R2 by row/column.",
            "neighbor": "Strongest absolute Pearson correlation with >=5 shared cells for evaluations and >=3 for models.",
            "tests": "Two-sided Spearman rank correlation after finite-pair filtering.",
        },
        "evaluation_analysis": {
            "features": evaluation_features,
            "rows": evaluation_rows,
            "spearman": _correlations(evaluation_rows, evaluation_features),
        },
        "model_analysis": {
            "features": model_features,
            "rows": model_rows,
            "spearman": _correlations(model_rows, model_features),
        },
        "pathology_adaptations": [
            "Rank-1 expressibility is primary because matched pathology validation selected rank 1; rank-2 is retained for direct BenchPress comparability.",
            "Task family replaces BenchPress's benchmark category metadata.",
            "Model size/type/provider hypotheses require a separately audited metadata table and are not inferred from model names.",
            "This file implements BenchPress's correlational hypotheses; intervention ablations are separate experiments.",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "n_evaluations": len(evaluation_rows), "n_models": len(model_rows)}, indent=2))


if __name__ == "__main__":
    main()
