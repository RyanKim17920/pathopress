#!/usr/bin/env python3
"""Scalar-recheck exhaustive-search finalists and certify winner stability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_probe_exhaustive as runner  # noqa: E402
from pathopress.probe_compression import predict_all_known, score_predictions  # noqa: E402


def _scalar_metrics(job: tuple[int, tuple[int, ...]]) -> dict[str, Any]:
    combo_index, probe_indices = job
    if runner._WORKER_MATRIX is None:
        raise RuntimeError("worker matrix is uninitialized")
    metrics = score_predictions(
        predict_all_known(
            runner._WORKER_MATRIX,
            probe_indices,
            rank=runner.PREDICTOR_RANK,
            regularization=runner.PREDICTOR_REGULARIZATION,
        )
    )
    return {
        "combo_index": int(combo_index),
        "medae": float(metrics["medae"]),
        "medape": float(metrics["medape"]),
        "n": int(metrics["n_target"]),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--scores", type=Path, default=ROOT / "data" / "scores.csv")
    parser.add_argument("--validate-top", type=int, default=100)
    parser.add_argument("--spread-count", type=int, default=32)
    parser.add_argument(
        "--integrity-manifest",
        type=Path,
        default=ROOT / "experiments/probe_exhaustive_integrity_manifest.json",
    )
    parser.add_argument(
        "--merged-validation",
        type=Path,
        default=ROOT / "experiments/probe_exhaustive_merged_validation.json",
    )
    parser.add_argument(
        "--equivalence",
        type=Path,
        default=ROOT / "experiments" / "probe_exhaustive_fast_equivalence.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "probe_exhaustive_scalar_top_validation.json",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(20, (os.cpu_count() or 2) - 1))
    )
    return parser.parse_args()


def validate_run(
    run_dir: Path,
    matrix: np.ndarray,
    evaluation_ids: list[str],
    validate_top: int,
    spread_count: int,
    workers: int,
    metric_tolerance: float,
) -> dict[str, Any]:
    merged_path = run_dir / "merged_summary.json.gz"
    merged = runner._load_json(merged_path)
    config = merged["config"]
    if not merged.get("complete") or int(merged["n_records"]) != int(
        config["total_combinations"]
    ):
        raise RuntimeError(f"run is not strictly complete: {run_dir}")
    top = list(merged["top"])
    if len(top) < validate_top + 1:
        raise RuntimeError(
            f"merge must retain at least {validate_top + 1} rows for certification"
        )
    index = {evaluation_id: position for position, evaluation_id in enumerate(evaluation_ids)}

    def raw_record(combo_index: int) -> dict[str, Any]:
        num_waves = int(config["num_waves"])
        modulus = int(config["assignment_modulus"])
        residue = combo_index % modulus
        wave = residue % num_waves
        shard = residue // num_waves
        offset = (combo_index - residue) // modulus
        chunk_index = offset // int(config["chunk_size"])
        record_offset = offset % int(config["chunk_size"])
        payload = runner._load_json(
            runner._chunk_path(run_dir, wave, shard, chunk_index)
        )
        record = payload["records"][record_offset]
        if int(record["combo_index"]) != combo_index:
            raise RuntimeError(f"raw spread record mismatch at {combo_index}")
        return record

    spread_indices = sorted(
        {
            int(value)
            for value in np.linspace(
                0,
                int(config["total_combinations"]) - 1,
                num=spread_count,
                dtype=np.int64,
            )
        }
    )
    spread_rows = [raw_record(combo_index) for combo_index in spread_indices]
    accelerated_rows = {
        int(row["combo_index"]): row
        for row in [*top[:validate_top], *spread_rows]
    }
    jobs = [
        (
            combo_index,
            tuple(index[evaluation_id] for evaluation_id in row["probe_set"]),
        )
        for combo_index, row in accelerated_rows.items()
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=runner._init_worker,
        initargs=(matrix, tuple(evaluation_ids), runner.SEED, None),
    ) as pool:
        scalar_rows = list(pool.map(_scalar_metrics, jobs, chunksize=1))
    scalar_by_combo = {int(row["combo_index"]): row for row in scalar_rows}
    accelerated = {int(row["combo_index"]): row for row in top[:validate_top]}
    comparisons = []
    for combo_index, prior in accelerated.items():
        scalar = scalar_by_combo[combo_index]
        comparisons.append(
            {
                **scalar,
                "probe_set": prior["probe_set"],
                "accelerated_medae": prior["medae"],
                "accelerated_medape": prior["medape"],
                "absolute_medae_delta": abs(
                    float(scalar["medae"]) - float(prior["medae"])
                ),
                "absolute_medape_delta": abs(
                    float(scalar["medape"]) - float(prior["medape"])
                ),
            }
        )
    comparisons.sort(key=lambda row: (float(row["medae"]), int(row["combo_index"])))
    best = comparisons[0]
    runner_up = comparisons[1]
    first_unvalidated_accelerated = float(top[validate_top]["medae"])
    max_delta = max(
        max(row["absolute_medae_delta"], row["absolute_medape_delta"])
        for row in comparisons
    )
    spread_comparisons = []
    for prior in spread_rows:
        scalar = scalar_by_combo[int(prior["combo_index"])]
        spread_comparisons.append(
            {
                **scalar,
                "probe_set": prior["probe_set"],
                "accelerated_medae": prior["medae"],
                "accelerated_medape": prior["medape"],
                "absolute_medae_delta": abs(
                    float(scalar["medae"]) - float(prior["medae"])
                ),
                "absolute_medape_delta": abs(
                    float(scalar["medape"]) - float(prior["medape"])
                ),
            }
        )
    spread_max_delta = max(
        max(row["absolute_medae_delta"], row["absolute_medape_delta"])
        for row in spread_comparisons
    )
    certified = (
        max_delta <= metric_tolerance
        and spread_max_delta <= metric_tolerance
        and float(best["medae"]) + metric_tolerance
        < first_unvalidated_accelerated - metric_tolerance
    )
    if not certified:
        raise RuntimeError(
            f"scalar winner is not separated from the unvalidated boundary: {run_dir}"
        )
    return {
        "run_dir": runner._display_path(run_dir),
        "config_sha256": sha256(run_dir / "config.json"),
        "merged_summary_sha256": sha256(merged_path),
        "total_combinations": int(config["total_combinations"]),
        "scalar_validated_top": validate_top,
        "max_absolute_metric_delta": max_delta,
        "deterministic_spread": {
            "rule": (
                "unique integer points from linspace(0,total_combinations-1,count); "
                "scalar MedAE/MedAPE metric recomputation covers the full combo-index "
                "range, not only finalists; it is not a full cell-array comparison"
            ),
            "requested_count": spread_count,
            "validated_count": len(spread_comparisons),
            "combo_indices": spread_indices,
            "max_absolute_metric_delta": spread_max_delta,
            "comparisons": spread_comparisons,
        },
        "metric_tolerance": metric_tolerance,
        "winner_certified": certified,
        "best": best,
        "runner_up": runner_up,
        "scalar_best_runner_up_gap": float(runner_up["medae"] - best["medae"]),
        "first_unvalidated_accelerated_medae": first_unvalidated_accelerated,
        "best_to_unvalidated_boundary_gap": float(
            first_unvalidated_accelerated - best["medae"]
        ),
        "comparisons": comparisons,
    }


def main() -> int:
    args = parse_args()
    if args.validate_top < 2 or args.spread_count < 3 or args.workers < 1:
        raise ValueError(
            "--validate-top must be >=2, --spread-count >=3, and --workers positive"
        )
    equivalence = runner._load_json(args.equivalence)
    if (
        float(equivalence["tolerances"]["max_absolute_cell_delta"]) > 1e-10
        or float(equivalence["tolerances"]["max_absolute_metric_delta"]) > 1e-11
        or len(equivalence.get("comparisons", [])) < 8
        or int(equivalence.get("observed", {}).get("sample_combinations", -1))
        != len(equivalence.get("comparisons", []))
    ):
        raise RuntimeError("equivalence evidence exceeds hard-coded scalar caps")
    integrity = runner._load_json(args.integrity_manifest)
    merged_validation = runner._load_json(args.merged_validation)
    if integrity.get("status") != "passed" or merged_validation.get("status") != "passed":
        raise RuntimeError("integrity and merged-order validations must pass first")
    runner._validate_fast_equivalence(
        args.scores,
        Path(equivalence["inputs"]["library_path"]),
        args.equivalence,
    )
    metric_tolerance = float(
        equivalence["tolerances"]["max_absolute_metric_delta"]
    )
    matrix, _, evaluation_ids = runner._load_matrix(args.scores)
    results = [
        validate_run(
            run_dir.resolve(),
            matrix,
            evaluation_ids,
            args.validate_top,
            args.spread_count,
            args.workers,
            metric_tolerance,
        )
        for run_dir in args.run_dirs
    ]
    payload = {
        "schema_version": 1,
        "status": "passed",
        "scores_sha256": sha256(args.scores),
        "equivalence_sha256": sha256(args.equivalence),
        "integrity_manifest_sha256": sha256(args.integrity_manifest),
        "merged_validation_sha256": sha256(args.merged_validation),
        "scientific_engine": equivalence["scientific_engine"],
        "validation_rule": (
            "Recompute the accelerated top-N with the scalar Python predictor; "
            "require observed deltas within the hash-bound tolerance and the "
            "scalar winner to remain separated from accelerated rank N+1. Also "
            f"recompute MedAE/MedAPE only (not full cell arrays) at {args.spread_count} "
            "deterministic indices spread across each full combination space."
        ),
        "runs": results,
    }
    runner._write_json_atomic(args.output, payload, indent=2)
    print(json.dumps({"status": payload["status"], "runs": results}, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
