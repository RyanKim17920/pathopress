#!/usr/bin/env python3
"""Calibrate confidence for a genuinely unseen pathology model row.

The primary simulation removes one existing model completely, appends it as a
new row with only k measured probes, and predicts every other reported cell.
Pinned temporal-release predictions provide a second, stricter calibration
population.  The resulting raw CSV contains leave-target-model-out risk and
conformal interval fields for every prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.completion import complete  # noqa: E402
from pathopress.new_model_confidence import (  # noqa: E402
    SUPPORTED_PROBE_COUNTS,
    build_new_model_confidence_artifact,
)
from pathopress.prediction import load_prediction_dataset  # noqa: E402


def _simulate_job(job: tuple[np.ndarray, list[str], list[str], dict[str, str], int, int, int]):
    matrix, models, evaluations, suites, target_index, k, seed = job
    observed = np.flatnonzero(np.isfinite(matrix[target_index]))
    if observed.size <= k:
        return []
    rng = np.random.RandomState(42 + 1009 * seed + 97 * target_index + 13 * k)
    probes = np.sort(rng.choice(observed, size=k, replace=False))
    probe_ids = [evaluations[int(column)] for column in probes]
    known = np.full(matrix.shape[1], np.nan, dtype=float)
    known[probes] = matrix[target_index, probes]
    training = np.delete(matrix, target_index, axis=0)
    prediction = complete(np.vstack([training, known]), rank=1, regularization=0.1)[-1]
    probe_suites = [suites[evaluation] for evaluation in probe_ids]
    rows = []
    for column in observed:
        column = int(column)
        if column in probes:
            continue
        evaluation = evaluations[column]
        actual = float(matrix[target_index, column])
        predicted = float(prediction[column])
        rows.append({
            "target_model_id": models[target_index],
            "evaluation_id": evaluation,
            "suite_id": suites[evaluation],
            "k": k,
            "seed": seed,
            "source": "leave_one_model_out_probe",
            "actual": actual,
            "predicted": predicted,
            "absolute_error": abs(predicted - actual),
            "probe_evaluations": json.dumps(probe_ids, separators=(",", ":")),
            "same_suite_probe_count": sum(value == suites[evaluation] for value in probe_suites),
            "column_training_support": int(np.isfinite(training[:, column]).sum()),
            "cutoff_date": "",
        })
    return rows


def _temporal_rows(path: Path, suites: dict[str, str]) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    probe_lookup = {
        (str(row["target_model_id"]), int(row["k"]), int(row["seed"])):
            [str(value) for value in row["revealed_evaluation_ids"]]
        for row in payload["summary_by_target_k_seed"]
    }
    rows = []
    for row in payload["raw_predictions"]:
        k = int(row["k"])
        if k not in SUPPORTED_PROBE_COUNTS or bool(row["is_revealed"]) or not bool(row["is_metric_cell"]):
            continue
        evaluation = str(row["evaluation_id"])
        if evaluation not in suites:
            continue
        key = (str(row["target_model_id"]), k, int(row["seed"]))
        probes = probe_lookup[key]
        predicted, actual = float(row["pred"]), float(row["actual"])
        rows.append({
            "target_model_id": str(row["target_model_id"]),
            "evaluation_id": evaluation,
            "suite_id": suites[evaluation],
            "k": k,
            "seed": int(row["seed"]),
            "source": "temporal_release",
            "actual": actual,
            "predicted": predicted,
            "absolute_error": abs(predicted - actual),
            "probe_evaluations": json.dumps(probes, separators=(",", ":")),
            "same_suite_probe_count": sum(suites.get(value) == suites[evaluation] for value in probes),
            "column_training_support": "",
            "cutoff_date": str(row["cutoff_date"]),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "data" / "scores.csv")
    parser.add_argument("--temporal", type=Path, default=ROOT / "experiments" / "temporal_deployment_rank1.json")
    parser.add_argument("--raw-output", type=Path, default=ROOT / "experiments" / "new_model_confidence_predictions_rank1.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "new_model_confidence_rank1.json")
    parser.add_argument("--probe-seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--reuse-raw", action="store_true",
        help="Rebuild calibration fields from the existing raw CSV without rerunning point predictions",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_prediction_dataset(args.scores)
    suite_by_evaluation = {}
    for score in dataset.scores:
        suite_by_evaluation.setdefault(score.evaluation_id, score.suite_id)
    jobs = [
        (dataset.matrix, dataset.models, dataset.evaluations, suite_by_evaluation, target, k, seed)
        for target in range(len(dataset.models))
        for k in SUPPORTED_PROBE_COUNTS
        for seed in range(args.probe_seeds)
        if int(np.isfinite(dataset.matrix[target]).sum()) > k
    ]
    if args.reuse_raw:
        with args.raw_output.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        for row in records:
            for field in ("k", "seed", "same_suite_probe_count"):
                row[field] = int(row[field])
            for field in ("actual", "predicted", "absolute_error"):
                row[field] = float(row[field])
    elif args.workers == 1:
        parts = [_simulate_job(job) for job in jobs]
        records = [row for part in parts for row in part]
        records.extend(_temporal_rows(args.temporal, suite_by_evaluation))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            parts = list(executor.map(_simulate_job, jobs, chunksize=1))
        records = [row for part in parts for row in part]
        records.extend(_temporal_rows(args.temporal, suite_by_evaluation))
    artifact, audited = build_new_model_confidence_artifact(records, args.scores)
    artifact["inputs"] = {
        "temporal_path": str(args.temporal.relative_to(ROOT)),
        "temporal_sha256": __import__("hashlib").sha256(args.temporal.read_bytes()).hexdigest(),
        "probe_seeds": args.probe_seeds,
        "simulation_jobs": len(jobs),
    }

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "target_model_id", "evaluation_id", "suite_id", "k", "seed", "source",
        "actual", "predicted", "absolute_error", "probe_evaluations",
        "same_suite_probe_count", "column_training_support", "cutoff_date",
        "crossfit_risk", "crossfit_conformal_scale", "crossfit_lower_90",
        "crossfit_upper_90", "confidence_status", "calibration_scope",
        "calibration_evaluation_models", "calibration_excluded_target_model",
    ]
    with args.raw_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in audited)
    artifact["raw_predictions"] = {
        "path": str(args.raw_output.relative_to(ROOT)),
        "sha256": __import__("hashlib").sha256(args.raw_output.read_bytes()).hexdigest(),
        "rows": len(audited),
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    overall = artifact["crossfit_metrics"]["overall"]
    print(f"artifact={args.output}")
    print(f"raw={args.raw_output} rows={len(audited)}")
    print(f"calibrated={overall['n_calibrated']} abstained={overall['n_abstained']}")
    print(f"coverage={overall['interval_coverage']:.4f} median_width={overall['median_interval_width']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
