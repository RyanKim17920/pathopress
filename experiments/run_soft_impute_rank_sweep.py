#!/usr/bin/env python3
"""Exact BenchPress-style raw/logit Soft-Impute rank sweep.

Each transform/rank/fold result is durably checkpointed.  Checkpoints are
content-addressed by the input matrix, folds, numerical configuration, and
relevant implementation files, so interrupted sweeps can resume without
mixing evidence from incompatible runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pathopress.completion import complete_soft_impute  # noqa: E402
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from run_benchpress_style import make_folds, metrics  # noqa: E402


CHECKPOINT_SCHEMA_VERSION = 1
RANKS = tuple(range(1, 11))
TRANSFORMS = ("identity", "logit")
SEEDS = tuple(range(42, 52))
N_FOLDS = 3
SOFT_IMPUTE_MAX_ITERATIONS = 100
SOFT_IMPUTE_TOLERANCE = 1e-4
_WORKER_MATRIX: np.ndarray | None = None
_WORKER_FOLDS: Sequence[tuple[int, int, np.ndarray, list[tuple[int, int]]]] | None = None
_WORKER_THREAD_LIMITER: Any = None


def _soft_job(job):
    transform, rank, train, held, matrix = job
    row_supported = np.any(np.isfinite(train), axis=1)
    column_supported = np.any(np.isfinite(train), axis=0)
    row_ids = np.flatnonzero(row_supported)
    column_ids = np.flatnonzero(column_supported)
    row_map = {int(old): new for new, old in enumerate(row_ids)}
    column_map = {int(old): new for new, old in enumerate(column_ids)}
    estimate = complete_soft_impute(
        train[np.ix_(row_ids, column_ids)], rank=rank, transform=transform
    )
    supported = [
        (i, j) for i, j in held if row_supported[i] and column_supported[j]
    ]
    actual = [float(matrix[i, j]) for i, j in supported]
    predicted = [float(estimate[row_map[i], column_map[j]]) for i, j in supported]
    unsupported_rows = sum(not row_supported[i] for i, _ in held)
    unsupported_columns = sum(
        row_supported[i] and not column_supported[j] for i, j in held
    )
    return (
        transform,
        rank,
        actual,
        predicted,
        float(metrics(actual, predicted)["medae"]),
        unsupported_rows,
        unsupported_columns,
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _matrix_sha256(
    matrix: np.ndarray, models: Sequence[str], evaluations: Sequence[str]
) -> str:
    contiguous = np.ascontiguousarray(matrix, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(_canonical_bytes({"shape": contiguous.shape, "dtype": "float64"}))
    digest.update(contiguous.tobytes(order="C"))
    digest.update(_canonical_bytes({"models": list(models), "evaluations": list(evaluations)}))
    return digest.hexdigest()


def _folds_sha256(
    folds: Sequence[tuple[int, int, np.ndarray, list[tuple[int, int]]]],
) -> str:
    # The training matrix is exactly the input matrix with ``held`` cells masked,
    # so the ordered held coordinates plus the matrix identity fully specify it.
    return _sha256(
        [
            {
                "seed": int(seed),
                "fold": int(fold),
                "held": [[int(i), int(j)] for i, j in held],
            }
            for seed, fold, _train, held in folds
        ]
    )


def _code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        PROJECT_ROOT / "src" / "pathopress" / "completion.py",
        PROJECT_ROOT / "experiments" / "run_benchpress_style.py",
    ):
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _job_spec(
    transform: str, rank: int, fold_index: int, seed: int, fold: int
) -> dict[str, object]:
    return {
        "transform": transform,
        "rank": rank,
        "fold_index": fold_index,
        "seed": seed,
        "fold": fold,
    }


def _job_name(spec: dict[str, object]) -> str:
    return (
        f"{spec['transform']}_rank{int(spec['rank']):02d}_"
        f"seed{int(spec['seed'])}_fold{int(spec['fold'])}.json"
    )


def _result_payload(result: tuple[object, ...]) -> dict[str, object]:
    (
        transform,
        rank,
        actual,
        predicted,
        fold_medae,
        unsupported_rows,
        unsupported_columns,
    ) = result
    return {
        "transform": transform,
        "rank": rank,
        "actual": actual,
        "predicted": predicted,
        "fold_medae": fold_medae,
        "unsupported_rows": unsupported_rows,
        "unsupported_columns": unsupported_columns,
    }


def _result_tuple(value: object, spec: dict[str, object]) -> tuple[object, ...]:
    if not isinstance(value, dict) or set(value) != {
        "transform",
        "rank",
        "actual",
        "predicted",
        "fold_medae",
        "unsupported_rows",
        "unsupported_columns",
    }:
        raise ValueError("checkpoint result has an invalid schema")
    if value["transform"] != spec["transform"] or value["rank"] != spec["rank"]:
        raise ValueError("checkpoint result does not match its job")
    actual = value["actual"]
    predicted = value["predicted"]
    if not isinstance(actual, list) or not isinstance(predicted, list):
        raise ValueError("checkpoint predictions must be lists")
    if len(actual) != len(predicted) or not actual:
        raise ValueError("checkpoint predictions are empty or misaligned")
    numeric = actual + predicted + [value["fold_medae"]]
    if not all(isinstance(item, (int, float)) and np.isfinite(item) for item in numeric):
        raise ValueError("checkpoint predictions must be finite numbers")
    unsupported_rows = value["unsupported_rows"]
    unsupported_columns = value["unsupported_columns"]
    if (
        not isinstance(unsupported_rows, int)
        or isinstance(unsupported_rows, bool)
        or unsupported_rows < 0
        or not isinstance(unsupported_columns, int)
        or isinstance(unsupported_columns, bool)
        or unsupported_columns < 0
    ):
        raise ValueError("checkpoint unsupported-cell counts are invalid")
    return (
        value["transform"],
        value["rank"],
        [float(item) for item in actual],
        [float(item) for item in predicted],
        float(value["fold_medae"]),
        unsupported_rows,
        unsupported_columns,
    )


def _write_checkpoint(
    path: Path,
    *,
    run_identity: dict[str, object],
    run_identity_sha256: str,
    spec: dict[str, object],
    result: tuple[object, ...],
) -> None:
    result_value = _result_payload(result)
    _atomic_write_json(
        path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_identity": run_identity,
            "run_identity_sha256": run_identity_sha256,
            "job": spec,
            "result": result_value,
            "result_sha256": _sha256(result_value),
        },
    )


def _load_checkpoint(
    path: Path,
    *,
    run_identity: dict[str, object],
    run_identity_sha256: str,
    spec: dict[str, object],
) -> tuple[object, ...] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "run_identity",
            "run_identity_sha256",
            "job",
            "result",
            "result_sha256",
        }:
            raise ValueError("checkpoint has an invalid envelope")
        if value["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint schema version mismatch")
        if value["run_identity"] != run_identity:
            raise ValueError("checkpoint run components mismatch")
        if _sha256(value["run_identity"]) != value["run_identity_sha256"]:
            raise ValueError("checkpoint run identity hash mismatch")
        if value["run_identity_sha256"] != run_identity_sha256:
            raise ValueError("checkpoint run identity mismatch")
        if value["job"] != spec:
            raise ValueError("checkpoint job identity mismatch")
        if value["result_sha256"] != _sha256(value["result"]):
            raise ValueError("checkpoint result hash mismatch")
        return _result_tuple(value["result"], spec)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def _init_worker(
    matrix: np.ndarray,
    folds: Sequence[tuple[int, int, np.ndarray, list[tuple[int, int]]]],
    blas_threads: int,
) -> None:
    global _WORKER_MATRIX, _WORKER_FOLDS, _WORKER_THREAD_LIMITER
    _WORKER_MATRIX = matrix
    _WORKER_FOLDS = folds
    # Environment variables protect spawn-based workers before NumPy loads its
    # BLAS.  threadpoolctl also covers platforms that import NumPy earlier.
    try:
        from threadpoolctl import threadpool_limits

        _WORKER_THREAD_LIMITER = threadpool_limits(limits=blas_threads, user_api="blas")
    except ImportError:
        _WORKER_THREAD_LIMITER = None


def _soft_job_indexed(spec: dict[str, object]) -> tuple[object, ...]:
    if _WORKER_MATRIX is None or _WORKER_FOLDS is None:
        raise RuntimeError("rank-sweep worker was not initialized")
    _seed, _fold, train, held = _WORKER_FOLDS[int(spec["fold_index"])]
    return _soft_job(
        (spec["transform"], spec["rank"], train, held, _WORKER_MATRIX)
    )


def _set_blas_environment(blas_threads: int) -> dict[str, str | None]:
    value = str(blas_threads)
    variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    previous = {variable: os.environ.get(variable) for variable in variables}
    for variable in variables:
        os.environ[variable] = value
    return previous


def _restore_blas_environment(previous: dict[str, str | None]) -> None:
    for variable, value in previous.items():
        if value is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = value


def _progress(
    *, completed: int, total: int, cached: int, started_at: float, detail: str
) -> None:
    elapsed = time.monotonic() - started_at
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else float("inf")
    eta = f"{remaining:.1f}s" if np.isfinite(remaining) else "unknown"
    print(
        f"[soft-impute {completed}/{total}] cached={cached} "
        f"computed={completed - cached} elapsed={elapsed:.1f}s eta={eta} {detail}",
        flush=True,
    )


def run_sweep(
    *,
    matrix: np.ndarray,
    models: Sequence[str],
    evaluations: Sequence[str],
    scores_sha256: str,
    output_path: Path,
    checkpoint_root: Path,
    ranks: Sequence[int] = RANKS,
    transforms: Sequence[str] = TRANSFORMS,
    seeds: Sequence[int] = SEEDS,
    n_folds: int = N_FOLDS,
    workers: int,
    blas_threads: int = 1,
    resume: bool = True,
    merge_only: bool = False,
    progress_every: int = 10,
) -> dict[str, object]:
    """Run or resume the exact sweep and return its unchanged result payload."""
    ranks = tuple(ranks)
    transforms = tuple(transforms)
    seeds = tuple(seeds)
    if not ranks or not transforms or not seeds or n_folds < 1:
        raise ValueError("ranks, transforms, seeds, and folds must be non-empty")
    if workers < 1 or blas_threads < 1 or progress_every < 1:
        raise ValueError("workers, BLAS threads, and progress interval must be positive")
    folds = [
        (seed, fold, train, held)
        for seed in seeds
        for fold, (train, held) in enumerate(
            make_folds(matrix, n_folds=n_folds, seed=seed)
        )
    ]
    configuration_identity = {
        "ranks": list(ranks),
        "transforms": list(transforms),
        "seeds": list(seeds),
        "n_folds": n_folds,
        "soft_impute_max_iterations": SOFT_IMPUTE_MAX_ITERATIONS,
        "soft_impute_tolerance": SOFT_IMPUTE_TOLERANCE,
    }
    code_sha256 = _code_sha256()
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "scores_sha256": scores_sha256,
        "matrix_sha256": _matrix_sha256(matrix, models, evaluations),
        "folds_sha256": _folds_sha256(folds),
        "configuration_sha256": _sha256(configuration_identity),
        "code_sha256": code_sha256,
    }
    run_identity_sha256 = _sha256(identity)
    run_directory = checkpoint_root / run_identity_sha256
    specs = [
        _job_spec(transform, rank, fold_index, seed, fold)
        for transform in transforms
        for rank in ranks
        for fold_index, (seed, fold, _train, _held) in enumerate(folds)
    ]
    results_by_job: dict[str, tuple[object, ...]] = {}
    missing: list[dict[str, object]] = []
    if resume or merge_only:
        for spec in specs:
            name = _job_name(spec)
            result = _load_checkpoint(
                run_directory / name,
                run_identity=identity,
                run_identity_sha256=run_identity_sha256,
                spec=spec,
            )
            if result is None:
                missing.append(spec)
            else:
                results_by_job[name] = result
    else:
        missing = list(specs)
    if merge_only and missing:
        raise RuntimeError(
            f"merge-only requested but {len(missing)} of {len(specs)} checkpoints are missing or invalid"
        )

    started_at = time.monotonic()
    cached = len(results_by_job)
    _progress(
        completed=cached,
        total=len(specs),
        cached=cached,
        started_at=started_at,
        detail="resume scan complete",
    )
    if missing and workers == 1:
        global _WORKER_MATRIX, _WORKER_FOLDS
        _WORKER_MATRIX, _WORKER_FOLDS = matrix, folds
        for offset, spec in enumerate(missing, start=1):
            result = _soft_job_indexed(spec)
            name = _job_name(spec)
            _write_checkpoint(
                run_directory / name,
                run_identity=identity,
                run_identity_sha256=run_identity_sha256,
                spec=spec,
                result=result,
            )
            results_by_job[name] = result
            completed = cached + offset
            if completed % progress_every == 0 or completed == len(specs):
                _progress(
                    completed=completed,
                    total=len(specs),
                    cached=cached,
                    started_at=started_at,
                    detail=name,
                )
    elif missing:
        previous_blas_environment = _set_blas_environment(blas_threads)
        context = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_init_worker,
                initargs=(matrix, folds, blas_threads),
            ) as executor:
                future_to_spec = {
                    executor.submit(_soft_job_indexed, spec): spec for spec in missing
                }
                for offset, future in enumerate(
                    as_completed(future_to_spec), start=1
                ):
                    spec = future_to_spec[future]
                    result = future.result()
                    name = _job_name(spec)
                    _write_checkpoint(
                        run_directory / name,
                        run_identity=identity,
                        run_identity_sha256=run_identity_sha256,
                        spec=spec,
                        result=result,
                    )
                    results_by_job[name] = result
                    completed = cached + offset
                    if completed % progress_every == 0 or completed == len(specs):
                        _progress(
                            completed=completed,
                            total=len(specs),
                            cached=cached,
                            started_at=started_at,
                            detail=name,
                        )
        finally:
            _restore_blas_environment(previous_blas_environment)

    if _code_sha256() != code_sha256:
        raise RuntimeError(
            "rank-sweep implementation changed during execution; checkpoints were "
            "kept, but no mixed-provenance result was written"
        )

    aggregate = {
        (transform, rank): ([], [], [], 0, 0)
        for transform in transforms
        for rank in ranks
    }
    # Never merge in filesystem or completion order: canonical job order is part
    # of numerical reproducibility for pooled floating-point metrics.
    for spec in specs:
        (
            transform,
            rank,
            fold_actual,
            fold_prediction,
            medae,
            unsupported_rows,
            unsupported_columns,
        ) = results_by_job[_job_name(spec)]
        actual, predicted, fold_medae, dropped_rows, dropped_columns = aggregate[
            (transform, rank)
        ]
        actual.extend(fold_actual)
        predicted.extend(fold_prediction)
        fold_medae.append(medae)
        aggregate[(transform, rank)] = (
            actual,
            predicted,
            fold_medae,
            dropped_rows + unsupported_rows,
            dropped_columns + unsupported_columns,
        )

    results: dict[str, dict[str, object]] = {}
    for transform in transforms:
        results[transform] = {}
        for rank in ranks:
            actual, predicted, fold_medae, unsupported_rows, unsupported_columns = aggregate[
                (transform, rank)
            ]
            results[transform][str(rank)] = {
                "pooled": metrics(actual, predicted),
                "n_unsupported_row": unsupported_rows,
                "n_unsupported_column": unsupported_columns,
                "fold_medae_median": round(float(np.median(fold_medae)), 6),
                "fold_medae_q1": round(float(np.percentile(fold_medae, 25)), 6),
                "fold_medae_q3": round(float(np.percentile(fold_medae, 75)), 6),
            }
    payload: dict[str, object] = {
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
            "n_seeds": len(seeds),
            "n_folds": n_folds,
            "base_seed": seeds[0],
            "soft_impute_max_iterations": SOFT_IMPUTE_MAX_ITERATIONS,
            "soft_impute_tolerance": SOFT_IMPUTE_TOLERANCE,
            "workers": workers,
        },
        "input": {"scores_sha256": scores_sha256},
        "results": results,
        "script_sha256": script_sha256,
    }
    _atomic_write_json(output_path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    default_workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "soft_impute_rank_sweep_results.json",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "soft_impute_rank_sweep_checkpoints",
    )
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="BLAS threads per worker (default: 1 prevents process/thread oversubscription)",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="write the final result only if every compatible checkpoint already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scores_path = PROJECT_ROOT / "data" / "scores.csv"
    scores = load_scores(scores_path)
    matrix, models, evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(matrix, models, evaluations)
    payload = run_sweep(
        matrix=matrix,
        models=models,
        evaluations=evaluations,
        scores_sha256=hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        output_path=args.output,
        checkpoint_root=args.checkpoint_dir,
        workers=args.workers,
        blas_threads=args.blas_threads,
        resume=not args.no_resume,
        merge_only=args.merge_only,
        progress_every=args.progress_every,
    )
    print(json.dumps(payload["results"], indent=2))


if __name__ == "__main__":
    main()
