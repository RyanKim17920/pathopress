#!/usr/bin/env python3
"""Run the BenchPress hard-rule temporal-deployment protocol on pathology."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.matrix import filter_matrix, load_scores, make_matrix
from pathopress.temporal import (
    PROTOCOL_VERSION,
    aggregate_metric,
    load_release_metadata,
    run_unit,
    select_targets,
    training_models,
    validate_metadata_coverage,
)

SELECTION_START_DATE = date(2025, 1, 1)
SELECTION_END_DATE = date(2025, 12, 31)
MIN_OBSERVED_SCORES = 20
K_VALUES = (1, 5, 10)
N_SEEDS = 10
BASE_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=ROOT / "data" / "scores.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data" / "model_release_dates.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "temporal_deployment_rank1.json")
    parser.add_argument("--raw-csv", type=Path, default=ROOT / "outputs" / "temporal_deployment_raw_rank1.csv")
    parser.add_argument("--summary-csv", type=Path, default=ROOT / "outputs" / "temporal_deployment_summary_rank1.csv")
    parser.add_argument("--rank", type=int, default=1)
    args = parser.parse_args()

    matrix, models, evaluations = make_matrix(load_scores(args.scores))
    matrix, models, evaluations = filter_matrix(matrix, models, evaluations)
    metadata = load_release_metadata(args.metadata)
    validate_metadata_coverage(models, metadata)
    targets = select_targets(
        matrix,
        models,
        metadata,
        start_date=SELECTION_START_DATE,
        end_date=SELECTION_END_DATE,
        observed_score_count_gt=MIN_OBSERVED_SCORES,
    )
    if not targets:
        raise RuntimeError("hard-rule target selection returned no models")

    units = []
    raw_predictions = []
    for target in targets:
        for k in K_VALUES:
            for seed in range(N_SEEDS):
                unit = run_unit(
                    matrix,
                    models,
                    evaluations,
                    metadata,
                    target_model_id=target,
                    k=k,
                    seed=seed,
                    base_seed=BASE_SEED,
                    rank=args.rank,
                )
                units.append(
                    {
                        "target_model_id": target,
                        "cutoff_date": unit["config"]["cutoff_date"],
                        "k": k,
                        "seed": seed,
                        **unit["metrics"],
                        **{
                            key: unit["config"][key]
                            for key in (
                                "n_eval_cells",
                                "n_metric_cells",
                                "n_revealed_cells",
                                "n_hidden_cells",
                                "n_not_predictable_cells",
                            )
                        },
                        "train_model_ids": unit["config"]["train_model_ids"],
                        "revealed_evaluation_ids": unit["config"]["revealed_evaluation_ids"],
                    }
                )
                raw_predictions.extend(unit["raw_predictions"])

    summary: dict[str, object] = {}
    summary_rows = []
    for target in targets:
        by_k = {}
        for k in K_VALUES:
            rows = [row for row in units if row["target_model_id"] == target and row["k"] == k]
            entry = {
                "n_seeds": len(rows),
                "n_eval_cells_median": int(np.median([row["n_eval_cells"] for row in rows])),
                "n_metric_cells_median": int(np.median([row["n_metric_cells"] for row in rows])),
                "n_revealed_cells_median": int(np.median([row["n_revealed_cells"] for row in rows])),
                "n_hidden_cells_median": int(np.median([row["n_hidden_cells"] for row in rows])),
                "n_not_predictable_cells_median": int(np.median([row["n_not_predictable_cells"] for row in rows])),
                "medae": aggregate_metric(row["medae"] for row in rows),
                "medape": aggregate_metric(row["medape"] for row in rows),
            }
            by_k[str(k)] = entry
            summary_rows.append(
                {
                    "target_model_id": target,
                    "cutoff_date": metadata[target].release_date.isoformat(),
                    "target_date_is_proxy": metadata[target].is_proxy,
                    "observed_score_count": int(np.isfinite(matrix[models.index(target)]).sum()),
                    "n_train_models": len(training_models(models, metadata, target)),
                    "k": k,
                    "n_seeds": len(rows),
                    "median_medae": entry["medae"]["median"],
                    "iqr_medae": entry["medae"]["iqr"],
                    "median_medape": entry["medape"]["median"],
                    "iqr_medape": entry["medape"]["iqr"],
                    "n_metric_cells_median": entry["n_metric_cells_median"],
                    "n_not_predictable_cells_median": entry["n_not_predictable_cells_median"],
                }
            )
        summary[target] = {
            "cutoff_date": metadata[target].release_date.isoformat(),
            "target_date_is_proxy": metadata[target].is_proxy,
            "primary_source_url": metadata[target].primary_source_url,
            "target_observed_count": int(np.isfinite(matrix[models.index(target)]).sum()),
            "n_train_models": len(training_models(models, metadata, target)),
            "by_k": by_k,
        }

    payload = {
        "config": {
            "protocol_version": PROTOCOL_VERSION,
            "matrix_shape": list(matrix.shape),
            "n_observed": int(np.isfinite(matrix).sum()),
            "predictor": f"pathology-selected logit bias ALS rank={args.rank} regularization=0.1",
            "upstream_predictor_difference": "BenchPress uses rank 2; pathology matched validation selected rank 1.",
            "base_seed": BASE_SEED,
            "k_values": list(K_VALUES),
            "n_seeds": N_SEEDS,
            "selection_rule": {
                "release_date_min": SELECTION_START_DATE.isoformat(),
                "release_date_max": SELECTION_END_DATE.isoformat(),
                "observed_score_count_gt": MIN_OBSERVED_SCORES,
                "verified_dates_only": True,
                "selection_basis": "release metadata and observed-score coverage only; prediction errors were not inspected",
                "window_rationale": "the complete calendar year 2025 is the first pathology release cohort with multiple high-coverage new models and enough strictly earlier context models",
            },
            "date_rule": "earliest verified official checkpoint/model-card/repository release or primary paper date that explicitly identifies the exact model; family/checkpoint ambiguity is flagged is_proxy",
            "evaluation_universe": "every observed target cell is recorded; metric cells are revealed exact cells plus hidden finite predictions",
            "aggregation": "medians across seed-level MedAE and MedAPE",
            "target_model_ids": targets,
        },
        "landmarks": [
            {
                "target_model_id": target,
                "cutoff_date": metadata[target].release_date.isoformat(),
                "target_date_is_proxy": metadata[target].is_proxy,
                "target_observed_count": int(np.isfinite(matrix[models.index(target)]).sum()),
                "train_model_ids": training_models(models, metadata, target),
            }
            for target in targets
        ],
        "summary_by_target": summary,
        "summary_by_target_k_seed": units,
        "raw_predictions": raw_predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    args.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.raw_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(raw_predictions[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(raw_predictions)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"targets={targets}")
    print(f"units={len(units)} raw_predictions={len(raw_predictions)}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
