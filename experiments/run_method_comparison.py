#!/usr/bin/env python3
"""Prediction-first BenchPress transform-by-method grid for PathoPress."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.artifacts import (  # noqa: E402
    load_fold_artifact,
    write_fold_artifact,
)
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.method_comparison import (  # noqa: E402
    HP_GRIDS,
    METHODS,
    TRANSFORMS,
    UnsupportedMethodError,
    predict_scores,
)


DEFAULT_OUTPUT = ROOT / "experiments" / "method_comparison"
DEFAULT_FOLDS = ROOT / "experiments" / "folds_s10_f3_bs42.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_identity(matrix: np.ndarray, models: list[str], evaluations: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(models, separators=(",", ":")).encode())
    digest.update(json.dumps(evaluations, separators=(",", ":")).encode())
    digest.update(np.asarray(np.isfinite(matrix), dtype=np.uint8).tobytes())
    digest.update(np.nan_to_num(matrix, nan=-1.23456789e300).astype("<f8").tobytes())
    return digest.hexdigest()


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _hp_hash(hyperparameters: dict[str, Any]) -> str:
    encoded = json.dumps(hyperparameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode()).hexdigest()[:10]


def all_shards(output_dir: Path) -> list[dict[str, Any]]:
    shards = []
    for transform in TRANSFORMS:
        for method in METHODS:
            for hp_index, hyperparameters in enumerate(HP_GRIDS[method]):
                index = len(shards)
                shard_id = (
                    f"{index:04d}__{_slug(transform)}__{_slug(method)}__"
                    f"hp{hp_index:02d}_{_hp_hash(hyperparameters)}"
                )
                shards.append(
                    {
                        "shard_index": index,
                        "shard_id": shard_id,
                        "transform": transform,
                        "method": method,
                        "hp_index": hp_index,
                        "hp": hyperparameters,
                        "path": output_dir / "predictions" / f"{shard_id}.npz",
                    }
                )
    return shards


def _load_matrix(scores_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    scores = load_scores(scores_path)
    matrix, models, evaluations = make_matrix(scores)
    return filter_matrix(matrix, models, evaluations)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".tmp_", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _expected_metadata(
    shard: dict[str, Any], matrix: np.ndarray, models: list[str], evaluations: list[str],
    scores_path: Path, folds_path: Path,
) -> dict[str, Any]:
    implementation_digest = hashlib.sha256()
    for implementation_path in (
        Path(__file__),
        ROOT / "src" / "pathopress" / "method_comparison.py",
        ROOT / "src" / "pathopress" / "artifacts.py",
    ):
        implementation_digest.update(implementation_path.read_bytes())
    return {
        "schema_version": 1,
        "shard_index": shard["shard_index"],
        "shard_id": shard["shard_id"],
        "transform": shard["transform"],
        "method": shard["method"],
        "hp_index": shard["hp_index"],
        "hp": shard["hp"],
        "matrix_shape": list(matrix.shape),
        "matrix_identity_sha256": _matrix_identity(matrix, models, evaluations),
        "scores_sha256": _sha256(scores_path),
        "folds_sha256": _sha256(folds_path),
        "implementation_sha256": implementation_digest.hexdigest(),
        "fold_protocol": {"n_seeds": 10, "n_folds": 3, "base_seed": 42, "min_scores": 1},
        "upstream_commit": "0a684b63ee0e4a401cb907a3827a82ea997d74c4",
        "rank_adaptation": (
            "Pathology-selected rank 1 replaces upstream fixed rank 2; direct rank-2 "
            "sensitivity shards are retained for Soft-Impute and Bias ALS."
        ),
    }


def _read_shard(
    path: Path, expected: dict[str, Any]
) -> tuple[str, dict[str, np.ndarray], dict[str, Any]] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"M_pred_by_fold", "fold_id", "test_i", "test_j", "actual", "predicted", "metadata_json"}
            if not required.issubset(data.files):
                return None
            arrays = {key: data[key] for key in data.files}
            metadata = json.loads(str(arrays["metadata_json"]))
        actual = {key: metadata.get(key) for key in expected}
        if actual != expected:
            return None
        status = str(metadata.get("status", "completed"))
        if status not in {"completed", "unsupported"}:
            return None
        return status, arrays, metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def prepare_folds(scores_path: Path, folds_path: Path, *, force: bool = False) -> Path:
    matrix, models, evaluations = _load_matrix(scores_path)
    if folds_path.exists() and not force:
        load_fold_artifact(folds_path, matrix, models, evaluations)
        return folds_path
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    write_fold_artifact(folds_path, matrix, models, evaluations)
    print(f"WROTE 30 persisted folds: {folds_path}")
    return folds_path


def run_shard(
    index: int, scores_path: Path, output_dir: Path, folds_path: Path, *, force: bool = False
) -> Path:
    matrix, models, evaluations = _load_matrix(scores_path)
    prepare_folds(scores_path, folds_path)
    shards = all_shards(output_dir)
    if index < 0 or index >= len(shards):
        raise ValueError(f"shard index must be in [0, {len(shards) - 1}]")
    shard = shards[index]
    expected = _expected_metadata(shard, matrix, models, evaluations, scores_path, folds_path)
    if shard["path"].exists() and not force:
        current = _read_shard(shard["path"], expected)
        if current is not None:
            print(f"SKIP {current[0]} shard {index}: {shard['path']}")
            return shard["path"]
        print(f"INVALIDATE metadata-mismatched shard {index}: {shard['path']}")

    folds = load_fold_artifact(folds_path, matrix, models, evaluations)
    started = time.perf_counter()
    matrices, fold_ids, test_i, test_j, actual, predicted = [], [], [], [], [], []
    status, reason = "completed", None
    try:
        for fold_id, (_seed, _fold, training, held) in enumerate(folds):
            prediction = predict_scores(training, shard["transform"], shard["method"], shard["hp"])
            matrices.append(prediction.astype(np.float64, copy=False))
            for row, column in held:
                fold_ids.append(fold_id)
                test_i.append(row)
                test_j.append(column)
                actual.append(matrix[row, column])
                predicted.append(prediction[row, column])
    except UnsupportedMethodError as exc:
        status, reason = "unsupported", str(exc)
        matrices, fold_ids, test_i, test_j, actual, predicted = [], [], [], [], [], []

    metadata = {
        **expected,
        "status": status,
        "unsupported_reason": reason,
        "elapsed_seconds": time.perf_counter() - started,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    matrix_array = (
        np.stack(matrices) if matrices else np.empty((0, *matrix.shape), dtype=np.float64)
    )
    _atomic_npz(
        shard["path"],
        M_pred_by_fold=matrix_array,
        fold_id=np.asarray(fold_ids, dtype=np.int16),
        test_i=np.asarray(test_i, dtype=np.int16),
        test_j=np.asarray(test_j, dtype=np.int16),
        actual=np.asarray(actual, dtype=np.float64),
        predicted=np.asarray(predicted, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"WROTE {status} shard {index}: {shard['path']} ({metadata['elapsed_seconds']:.2f}s)")
    return shard["path"]


def _run_shard_job(job: tuple[int, Path, Path, Path, bool]) -> str:
    index, scores_path, output_dir, folds_path, force = job
    return str(run_shard(index, scores_path, output_dir, folds_path, force=force))


def run_shards(
    indices: list[int], scores_path: Path, output_dir: Path, folds_path: Path,
    *, force: bool = False, workers: int = 1,
) -> None:
    """Run independent shards with bounded process-level parallelism."""
    jobs = [(index, scores_path, output_dir, folds_path, force) for index in indices]
    if workers <= 1:
        for job in jobs:
            _run_shard_job(job)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for completed, path in enumerate(executor.map(_run_shard_job, jobs), 1):
            print(f"PROGRESS {completed}/{len(jobs)} {path}", flush=True)


def _metrics(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    fold_ids = arrays["fold_id"].astype(int)
    actual = arrays["actual"].astype(float)
    predicted = arrays["predicted"].astype(float)
    fold_rows = []
    for fold_id in np.unique(fold_ids):
        selected = fold_ids == fold_id
        valid = selected & np.isfinite(actual) & np.isfinite(predicted)
        absolute = np.abs(predicted[valid] - actual[valid])
        nonzero = np.abs(actual[valid]) > 1e-12
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "n": int(valid.sum()),
                "medae": float(np.median(absolute)) if len(absolute) else None,
                "medape": float(np.median(absolute[nonzero] / np.abs(actual[valid][nonzero])) * 100.0)
                if nonzero.any() else None,
            }
        )
    medae = [row["medae"] for row in fold_rows if row["medae"] is not None]
    medape = [row["medape"] for row in fold_rows if row["medape"] is not None]
    covered = sum(row["n"] for row in fold_rows)
    return {
        "medae_median": float(np.median(medae)) if medae else None,
        "medape_median": float(np.median(medape)) if medape else None,
        "coverage": covered / len(actual) if len(actual) else 0.0,
        "n_predictions": covered,
        "n_expected_predictions": len(actual),
        "per_fold": fold_rows,
    }


def _write_top_tables(output_dir: Path, rows: list[dict[str, Any]], top_n: int = 15) -> None:
    usable = [row for row in rows if row["status"] == "completed" and row["medape_median"] is not None]
    fields = ("rank", "metric", "transform", "method", "hyperparameters", "value", "coverage", "shard_id")
    table_rows = []
    for metric, field in (("MedAPE", "medape_median"), ("MedAE", "medae_median")):
        for rank, row in enumerate(sorted(usable, key=lambda item: item[field])[:top_n], 1):
            table_rows.append(
                {
                    "rank": rank,
                    "metric": metric,
                    "transform": row["transform"],
                    "method": row["method"],
                    "hyperparameters": json.dumps(row["hp"], sort_keys=True),
                    "value": row[field],
                    "coverage": row["coverage"],
                    "shard_id": row["shard_id"],
                }
            )
    csv_path = output_dir / "top_methods.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table_rows)
    markdown = ["| Rank | Metric | Transform | Method | Hyperparameters | Value | Coverage |", "|---:|---|---|---|---|---:|---:|"]
    for row in table_rows:
        markdown.append(
            f"| {row['rank']} | {row['metric']} | {row['transform']} | {row['method']} | "
            f"`{row['hyperparameters']}` | {row['value']:.4f} | {row['coverage']:.1%} |"
        )
    (output_dir / "top_methods.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def merge_results(scores_path: Path, output_dir: Path, folds_path: Path) -> dict[str, Any]:
    matrix, models, evaluations = _load_matrix(scores_path)
    prepare_folds(scores_path, folds_path)
    completed, unsupported, missing = [], [], []
    results: dict[str, dict[str, Any]] = {transform: {} for transform in TRANSFORMS}
    for shard in all_shards(output_dir):
        expected = _expected_metadata(shard, matrix, models, evaluations, scores_path, folds_path)
        loaded = _read_shard(shard["path"], expected) if shard["path"].exists() else None
        basic = {key: value for key, value in shard.items() if key != "path"}
        basic["prediction_file"] = str(shard["path"].relative_to(ROOT))
        if loaded is None:
            missing.append(basic)
            continue
        status, arrays, metadata = loaded
        if status == "unsupported":
            unsupported.append({**basic, "status": status, "reason": metadata["unsupported_reason"]})
            continue
        row = {**basic, "status": status, "elapsed_seconds": metadata["elapsed_seconds"], **_metrics(arrays)}
        completed.append(row)
        current = results[row["transform"]].get(row["method"])
        if current is None or row["medape_median"] < current["medape_median"]:
            results[row["transform"]][row["method"]] = row

    expected_shards = [
        {**{key: value for key, value in shard.items() if key != "path"}, "prediction_file": str(shard["path"].relative_to(ROOT))}
        for shard in all_shards(output_dir)
    ]
    manifest = {
        "schema_version": 1,
        "description": "PathoPress port of BenchPress Section 4 transform-by-method comparison",
        "upstream_commit": "0a684b63ee0e4a401cb907a3827a82ea997d74c4",
        "configuration": {
            "n_transforms": len(TRANSFORMS),
            "n_methods": len(METHODS),
            "n_expected_shards": len(expected_shards),
            "folds": {"n_seeds": 10, "n_folds": 3, "base_seed": 42, "min_scores": 1},
            "primary_metric": "median across fold-level MedAPE",
            "rank_adaptation": "rank 1 primary with direct rank-2 Soft-Impute and Bias-ALS sensitivity rows",
        },
        "matrix": {"n_models": len(models), "n_evaluations": len(evaluations), "n_observed": int(np.isfinite(matrix).sum())},
        "inputs": {"scores_sha256": _sha256(scores_path), "folds_sha256": _sha256(folds_path)},
        "counts": {"completed": len(completed), "unsupported": len(unsupported), "missing": len(missing)},
        "completed": completed,
        "unsupported": unsupported,
        "missing": missing,
        "expected": expected_shards,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "manifest.json", manifest)
    _atomic_json(output_dir / "results.json", results)
    _write_top_tables(output_dir, completed)
    print(f"MERGED completed={len(completed)} unsupported={len(unsupported)} missing={len(missing)}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "data" / "scores.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-folds", action="store_true")
    action.add_argument("--list-shards", action="store_true")
    action.add_argument("--shard-index", type=int)
    action.add_argument("--shard-indices", nargs="+", type=int)
    action.add_argument("--run-range", nargs=2, type=int, metavar=("START", "END"))
    action.add_argument("--merge", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds_path = args.folds or DEFAULT_FOLDS
    if args.prepare_folds:
        prepare_folds(args.scores, folds_path, force=args.force)
    elif args.list_shards:
        for shard in all_shards(args.output_dir):
            print(json.dumps({**{key: value for key, value in shard.items() if key != "path"}, "done": shard["path"].exists()}, sort_keys=True))
    elif args.shard_index is not None:
        run_shard(args.shard_index, args.scores, args.output_dir, folds_path, force=args.force)
    elif args.shard_indices is not None:
        run_shards(
            args.shard_indices, args.scores, args.output_dir, folds_path,
            force=args.force, workers=args.workers,
        )
    elif args.run_range is not None:
        start, end = args.run_range
        run_shards(
            list(range(start, end)), args.scores, args.output_dir, folds_path,
            force=args.force, workers=args.workers,
        )
    elif args.merge:
        merge_results(args.scores, args.output_dir, folds_path)


if __name__ == "__main__":
    main()
