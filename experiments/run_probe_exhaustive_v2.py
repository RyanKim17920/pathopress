#!/usr/bin/env python3
"""Backend-bound schema-v2 exhaustive search using the BenchPress contract.

This is the pathology adaptation of BenchPress's
``all_known_probe_bruteforce_v1`` runner.  Each invocation evaluates one
wave/shard residue.  ``merge`` is deliberately complete-by-default: a partial
run is rejected unless ``--allow-incomplete`` is explicitly requested.

The sibling ``run_probe_exhaustive.py`` is retained byte-for-byte as the
legacy-v1 generator referenced by frozen audit manifests. New searches must use
this schema-v2 runner.
"""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import inspect
import itertools
import json
import math
import multiprocessing
import os
import platform
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

# Avoid multiplying BLAS threads inside the process pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.probe_compression import (  # noqa: E402
    ProbePredictions,
    predict_all_known,
    score_predictions,
)


SEED = 42
EVAL_PROTOCOL = "all_known_probe_bruteforce_v1"
UPSTREAM_REFERENCE_COMMIT = "0a684b63ee0e4a401cb907a3827a82ea997d74c4"
DEFAULT_CHUNK_SIZE = 256
PREDICTOR_RANK = 1
PREDICTOR_REGULARIZATION = 0.1
FAST_SOURCE = ROOT / "experiments" / "fast_rank1_v2.cpp"
DEFAULT_FAST_EQUIVALENCE = (
    ROOT / "experiments" / "probe_exhaustive_fast_equivalence_v2.json"
)
STALE_RUN_REGISTRY = ROOT / "experiments" / "probe_exhaustive_stale_runs.json"
CONFIG_SCHEMA_VERSION = 2
FAST_EQUIVALENCE_SCHEMA_VERSION = 2
FAST_CELL_DELTA_CAP = 1e-10
FAST_METRIC_DELTA_CAP = 1e-11
FAST_MIN_COMPARISONS = 32
FAST_COMPILE_FLAGS = (
    "-O3",
    "-std=c++17",
    "-fPIC",
    "-shared",
    "-ffp-contract=off",
)

_WORKER_MATRIX: np.ndarray | None = None
_WORKER_EVALUATION_IDS: tuple[str, ...] = ()
_WORKER_FAST_FUNCTION: Any | None = None
_WORKER_FAST_INITIAL_ROWS: np.ndarray | None = None
_WORKER_FAST_INITIAL_COLUMNS: np.ndarray | None = None


def _open_text(path: Path, mode: str):
    opener = gzip.open if path.name.endswith(".gz") else open
    return opener(path, mode, encoding="utf-8")


def _load_json(path: Path) -> Any:
    with _open_text(path, "rt") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, payload: Any, *, indent: int | None = None) -> None:
    """Write plain or gzip JSON atomically next to its final destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if path.name.endswith(".gz") else ".tmp"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with _open_text(temporary, "wt") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=False)
            if indent is not None:
                handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_strings(values: Sequence[str]) -> str:
    encoded = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _function_sha256(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _platform_identity() -> dict[str, Any]:
    libc_name, libc_version = platform.libc_ver()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_abi": getattr(sys.implementation, "cache_tag", None),
        "libc": [libc_name, libc_version],
    }


def _read_regular_file_no_follow(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"Fast library must be a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), metadata
    finally:
        os.close(descriptor)


def _stage_fast_library(
    path: Path, expected_sha256: str
) -> tuple[int, Path, Path, str]:
    """Copy verified bytes to a private path and return an inherited read FD.

    Workers load ``/proc/self/fd/<fd>`` under an explicit Linux/fork contract,
    so replacing either the user path or staged pathname cannot change the
    bytes loaded after validation.
    """

    if not sys.platform.startswith("linux") or not Path("/proc/self/fd").is_dir():
        raise RuntimeError("The fast backend requires Linux /proc/self/fd support")
    payload, _ = _read_regular_file_no_follow(path)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise RuntimeError("Fast library changed while equivalence evidence was checked")
    directory = Path(tempfile.mkdtemp(prefix="pathopress-fast-"))
    os.chmod(directory, 0o700)
    staged = directory / f"libfast_rank1-{observed_sha256}.so"
    write_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(staged, write_flags, 0o500)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(staged, 0o500)
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    read_descriptor = os.open(staged, read_flags)
    if hashlib.sha256(os.read(read_descriptor, len(payload) + 1)).hexdigest() != observed_sha256:
        os.close(read_descriptor)
        shutil.rmtree(directory)
        raise RuntimeError("Staged fast library failed its content hash check")
    os.lseek(read_descriptor, 0, os.SEEK_SET)
    return read_descriptor, directory, staged, observed_sha256


def _cleanup_staged_fast_library(descriptor: int | None, directory: Path | None) -> None:
    if descriptor is not None:
        os.close(descriptor)
    if directory is not None:
        shutil.rmtree(directory)


def _short_text_hash(values: Sequence[str]) -> str:
    return hashlib.sha1("\n".join(values).encode("utf-8")).hexdigest()[:12]


def _safe_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _assert_run_is_active(out_dir: Path) -> None:
    """Fail closed for stopped score-matrix snapshots retained for audit."""

    if not STALE_RUN_REGISTRY.exists():
        return
    registry = _load_json(STALE_RUN_REGISTRY)
    relative = _display_path(out_dir)
    for run in registry.get("runs", []):
        if run.get("out_dir") == relative and run.get("status") != "active":
            raise RuntimeError(
                f"Run directory {relative} is marked {run.get('status')!r} in "
                f"{STALE_RUN_REGISTRY}. Use a new hash-bound --out-dir after "
                "rebuilding the matrix and candidate allowlists; retained chunks "
                "are audit evidence only."
            )


def _validate_fast_equivalence(
    scores: Path, library: Path, equivalence_path: Path
) -> dict[str, Any]:
    if not equivalence_path.is_file():
        raise FileNotFoundError(
            f"Missing fast-backend equivalence evidence: {equivalence_path}. "
            "Run experiments/verify_fast_rank1.py first."
        )
    payload = _load_json(equivalence_path)
    inputs = payload.get("inputs", {})
    engine = payload.get("scientific_engine", {})
    observed = payload.get("observed", {})
    hard_caps = payload.get("hard_caps", {})
    comparisons = payload.get("comparisons")
    compiler = inputs.get("compiler", {})
    compiler_path = Path(str(compiler.get("path", "")))
    compiler_version = None
    if compiler_path.is_file():
        compiler_version = subprocess.run(
            [str(compiler_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
    execution_hashes = {
        "_init_worker": _function_sha256(_init_worker),
        "_predict_all_known_fast": _function_sha256(_predict_all_known_fast),
        "_evaluate_combo": _function_sha256(_evaluate_combo),
    }
    comparison_rows_valid = isinstance(comparisons, list)
    comparison_probe_sets: list[tuple[str, ...]] = []
    cell_deltas: list[float] = []
    metric_deltas: list[float] = []
    if comparison_rows_valid:
        for row in comparisons:
            if not isinstance(row, dict) or set(row) != {
                "combo_index",
                "probe_set",
                "max_absolute_cell_delta",
                "absolute_medae_delta",
                "absolute_medape_delta",
            }:
                comparison_rows_valid = False
                break
            probes = row.get("probe_set")
            if (
                not isinstance(row.get("combo_index"), int)
                or isinstance(row.get("combo_index"), bool)
                or not isinstance(probes, list)
                or len(probes) != 5
                or len(set(probes)) != 5
                or not all(isinstance(value, str) for value in probes)
                or not _finite_nonnegative(row.get("max_absolute_cell_delta"))
                or not _finite_nonnegative(row.get("absolute_medae_delta"))
                or not _finite_nonnegative(row.get("absolute_medape_delta"))
            ):
                comparison_rows_valid = False
                break
            comparison_probe_sets.append(tuple(probes))
            cell_deltas.append(float(row["max_absolute_cell_delta"]))
            metric_deltas.append(
                max(
                    float(row["absolute_medae_delta"]),
                    float(row["absolute_medape_delta"]),
                )
            )
    recomputed_cell = max(cell_deltas, default=float("inf"))
    recomputed_metric = max(metric_deltas, default=float("inf"))
    checks = {
        "schema_version": payload.get("schema_version")
        == FAST_EQUIVALENCE_SCHEMA_VERSION,
        "status": payload.get("status") == "passed",
        "scores_sha256": inputs.get("scores_sha256") == _sha256_bytes(scores),
        "source_sha256": inputs.get("source_sha256") == _sha256_bytes(FAST_SOURCE),
        "library_sha256": inputs.get("library_sha256") == _sha256_bytes(library),
        "runner_sha256": inputs.get("runner_sha256") == _sha256_bytes(Path(__file__)),
        "execution_function_sha256": inputs.get("execution_function_sha256")
        == execution_hashes,
        "compile_flags": inputs.get("compile_flags") == list(FAST_COMPILE_FLAGS),
        "platform": inputs.get("platform") == _platform_identity(),
        "compiler_regular": compiler_path.is_file() and not compiler_path.is_symlink(),
        "compiler_sha256": compiler.get("sha256")
        == (_sha256_bytes(compiler_path) if compiler_path.is_file() else None),
        "compiler_version": compiler.get("version") == compiler_version,
        "rank": engine.get("rank") == PREDICTOR_RANK,
        "regularization": engine.get("regularization") == PREDICTOR_REGULARIZATION,
        "iterations": engine.get("iterations") == 40,
        "ensembles": engine.get("ensembles") == 10,
        "seeds": engine.get("seeds") == list(range(42, 52)),
        "hard_caps": hard_caps
        == {
            "max_absolute_cell_delta": FAST_CELL_DELTA_CAP,
            "max_absolute_metric_delta": FAST_METRIC_DELTA_CAP,
            "minimum_comparisons": FAST_MIN_COMPARISONS,
        },
        "comparison_rows": comparison_rows_valid,
        "comparison_count": len(comparison_probe_sets) >= FAST_MIN_COMPARISONS
        and observed.get("sample_combinations") == len(comparison_probe_sets),
        "comparison_unique": len(set(comparison_probe_sets))
        == len(comparison_probe_sets),
        "cell_max_recomputed": _finite_nonnegative(
            observed.get("max_absolute_cell_delta")
        )
        and float(observed["max_absolute_cell_delta"]) == recomputed_cell,
        "metric_max_recomputed": _finite_nonnegative(
            observed.get("max_absolute_metric_delta")
        )
        and float(observed["max_absolute_metric_delta"]) == recomputed_metric,
        "cell_hard_cap": recomputed_cell <= FAST_CELL_DELTA_CAP,
        "metric_hard_cap": recomputed_metric <= FAST_METRIC_DELTA_CAP,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"Fast-backend equivalence evidence is stale or failed checks: {failed}"
        )
    return payload


def _execution_backend_config(
    equivalence: dict[str, Any] | None,
    equivalence_path: Path | None,
    library_sha256: str | None,
) -> dict[str, Any]:
    runner_hash = _sha256_bytes(Path(__file__))
    if equivalence is None:
        return {
            "kind": "python_scalar",
            "runner_sha256": runner_hash,
            "execution_function_sha256": {
                "predict_all_known": _function_sha256(predict_all_known),
                "_evaluate_combo": _function_sha256(_evaluate_combo),
            },
            "platform": _platform_identity(),
        }
    inputs = equivalence["inputs"]
    if library_sha256 is None or equivalence_path is None:
        raise RuntimeError("Fast backend identity is incomplete")
    return {
        "kind": "native_rank1",
        "contract_schema_version": FAST_EQUIVALENCE_SCHEMA_VERSION,
        "runner_sha256": runner_hash,
        "source_sha256": inputs["source_sha256"],
        "library_sha256": library_sha256,
        "equivalence_path": _display_path(equivalence_path),
        "equivalence_sha256": _sha256_bytes(equivalence_path),
        "compiler": inputs["compiler"],
        "compile_flags": inputs["compile_flags"],
        "platform": inputs["platform"],
        "execution_function_sha256": inputs["execution_function_sha256"],
        "hard_caps": equivalence["hard_caps"],
        "observed": {
            "sample_combinations": equivalence["observed"]["sample_combinations"],
            "max_absolute_cell_delta": equivalence["observed"][
                "max_absolute_cell_delta"
            ],
            "max_absolute_metric_delta": equivalence["observed"][
                "max_absolute_metric_delta"
            ],
        },
    }


def _parse_csv_ids(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_unique_known(ids: Sequence[str], known: Sequence[str], label: str) -> None:
    unknown = [value for value in ids if value not in known]
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}")
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {label}: {duplicates}")


def _load_matrix(scores_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    scores = load_scores(scores_path)
    matrix, model_ids, evaluation_ids = filter_matrix(*make_matrix(scores))
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("Score matrix contains no finite observations")
    if np.any(finite < 0.0) or np.any(finite > 100.0):
        raise ValueError("Finite score values must lie in the closed interval [0, 100]")
    return matrix, model_ids, evaluation_ids


def _allowlist_ids(payload: Any, path: Path) -> list[str]:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        # ``evaluation_ids`` is the PathoPress spelling.  Accept the upstream
        # spelling as well so allowlists can be mechanically translated.
        values = payload.get("evaluation_ids", payload.get("benchmark_ids"))
    else:
        values = None
    if not isinstance(values, list) or not values or not all(isinstance(x, str) for x in values):
        raise ValueError(
            f"Candidate allowlist {path} must contain a non-empty evaluation_ids list"
        )
    return list(values)


def _load_candidates(
    evaluation_ids: list[str],
    candidate_allowlist: Path | None,
    candidate_limit: int | None,
) -> tuple[list[int], list[str], str | None]:
    if candidate_allowlist is None:
        candidate_ids = list(evaluation_ids)
        allowlist_sha256 = None
    else:
        candidate_ids = _allowlist_ids(_load_json(candidate_allowlist), candidate_allowlist)
        allowlist_sha256 = _sha256_bytes(candidate_allowlist)
    _validate_unique_known(candidate_ids, evaluation_ids, "candidate evaluation IDs")
    if candidate_limit is not None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        candidate_ids = candidate_ids[:candidate_limit]
    if not candidate_ids:
        raise ValueError("Candidate set is empty")
    return [evaluation_ids.index(value) for value in candidate_ids], candidate_ids, allowlist_sha256


def _assignment_residue(
    wave_index: int, num_waves: int, shard_index: int, num_shards: int
) -> int:
    return int(wave_index) + int(num_waves) * int(shard_index)


def _assigned_count(total: int, residue: int, modulus: int) -> int:
    if residue >= total:
        return 0
    return ((total - 1 - residue) // modulus) + 1


def _candidate_source_label(path: Path | None) -> str:
    if path is None:
        return "all"
    name = path.name
    for suffix in (".json.gz", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return _safe_token(name)


def _default_out_dir(args: argparse.Namespace, candidate_label: str) -> Path:
    fixed = _parse_csv_ids(args.fixed_probe)
    fixed_part = "" if not fixed else "_fixed-" + _safe_token("-".join(fixed))
    limit_part = "" if args.candidate_limit is None else f"_limit{args.candidate_limit}"
    return ROOT / "experiments" / "probe_exhaustive_runs" / (
        f"exhaustive_{_safe_token(args.metric)}_k{args.k}_"
        f"candidates-{candidate_label}{limit_part}{fixed_part}"
    )


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    absolute = path.resolve()
    try:
        return str(absolute.relative_to(ROOT))
    except ValueError:
        return str(absolute)


def _build_config(
    args: argparse.Namespace,
    matrix: np.ndarray,
    model_ids: list[str],
    evaluation_ids: list[str],
    candidate_indices: list[int],
    candidate_ids: list[str],
    allowlist_sha256: str | None,
    execution_backend: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[int], list[int]]:
    fixed_ids = _parse_csv_ids(args.fixed_probe)
    _validate_unique_known(fixed_ids, evaluation_ids, "fixed evaluation IDs")
    fixed_indices = [evaluation_ids.index(value) for value in fixed_ids]
    fixed_set = set(fixed_indices)
    remaining_candidates = [value for value in candidate_indices if value not in fixed_set]
    remaining_ids = [evaluation_ids[value] for value in remaining_candidates]

    choose_size = int(args.k) - len(fixed_indices)
    if choose_size < 0:
        raise ValueError(
            f"k={args.k} is smaller than the number of fixed probes ({len(fixed_indices)})"
        )
    if choose_size > len(remaining_candidates):
        raise ValueError(
            f"Cannot choose {choose_size} additional probes from "
            f"{len(remaining_candidates)} remaining candidates"
        )
    if args.num_waves < 1 or args.num_shards < 1:
        raise ValueError("num_waves and num_shards must be positive")
    if not 0 <= args.wave_index < args.num_waves:
        raise ValueError(f"wave_index must be in [0, {args.num_waves})")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"shard_index must be in [0, {args.num_shards})")
    if args.chunk_size < 1 or args.workers < 1:
        raise ValueError("chunk_size and workers must be positive")

    total = math.comb(len(remaining_candidates), choose_size)
    modulus = int(args.num_waves) * int(args.num_shards)
    residue = _assignment_residue(
        args.wave_index, args.num_waves, args.shard_index, args.num_shards
    )
    observed = np.isfinite(matrix)
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "config_schema": "pathopress.probe_exhaustive.run.v2",
        "eval_protocol": EVAL_PROTOCOL,
        "upstream_reference_commit": UPSTREAM_REFERENCE_COMMIT,
        "k": int(args.k),
        "metric": args.metric,
        "seed": SEED,
        "n_models": int(matrix.shape[0]),
        "n_evaluations": int(matrix.shape[1]),
        "n_observed": int(observed.sum()),
        "n_target_cells": int(observed.sum()),
        "eval_scope": "all_observed_cells",
        "predictor_rank": PREDICTOR_RANK,
        "predictor_regularization": PREDICTOR_REGULARIZATION,
        "prediction_engine": (
            "pathopress.complete (masked low-rank completion, "
            f"rank={PREDICTOR_RANK}, regularization={PREDICTOR_REGULARIZATION})"
        ),
        "scores_path": _display_path(args.scores),
        "scores_sha256": _sha256_bytes(args.scores),
        "model_ids": model_ids,
        "model_ids_hash": _short_text_hash(model_ids),
        "model_ids_sha256": _sha256_strings(model_ids),
        "evaluation_ids": evaluation_ids,
        "evaluation_ids_hash": _short_text_hash(evaluation_ids),
        "evaluation_ids_sha256": _sha256_strings(evaluation_ids),
        "candidate_allowlist_path": _display_path(args.candidate_allowlist),
        "candidate_allowlist_sha256": allowlist_sha256,
        "candidate_limit": args.candidate_limit,
        "candidate_ids": candidate_ids,
        "candidate_hash": _short_text_hash(candidate_ids),
        "candidate_ids_sha256": _sha256_strings(candidate_ids),
        "fixed_probe_ids": fixed_ids,
        "fixed_probe_hash": _short_text_hash(fixed_ids) if fixed_ids else None,
        "fixed_probe_ids_sha256": _sha256_strings(fixed_ids),
        "remaining_candidate_ids": remaining_ids,
        "remaining_candidate_hash": _short_text_hash(remaining_ids),
        "remaining_candidate_ids_sha256": _sha256_strings(remaining_ids),
        "execution_backend": execution_backend
        if execution_backend is not None
        else _execution_backend_config(None, None, None),
        "choose_size_after_fixed": choose_size,
        "total_combinations": total,
        "num_waves": int(args.num_waves),
        "num_shards": int(args.num_shards),
        "wave_index": int(args.wave_index),
        "shard_index": int(args.shard_index),
        "assignment_residue": residue,
        "assignment_modulus": modulus,
        "assigned_combinations": _assigned_count(total, residue, modulus),
        "chunk_size": int(args.chunk_size),
        "cell_masking": (
            "For model i, keep observed probe cells visible and mask observed "
            "non-probe cells. Probe target cells are stored with pred=true; "
            "non-probe target cells are rank-1 predictions. The evaluation "
            "universe is fixed to all observed cells."
        ),
    }, fixed_indices, remaining_candidates


_SHARD_KEYS = (
    "wave_index",
    "shard_index",
    "assignment_residue",
    "assigned_combinations",
)


def _base_run_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in _SHARD_KEYS}


def _config_for_chunk_v1(config: dict[str, Any]) -> dict[str, Any]:
    """Preserve the exact historical chunk identity for frozen legacy runs."""

    keys = (
        "eval_protocol",
        "upstream_reference_commit",
        "k",
        "metric",
        "seed",
        "n_models",
        "n_evaluations",
        "n_observed",
        "n_target_cells",
        "predictor_rank",
        "predictor_regularization",
        "scores_sha256",
        "model_ids_hash",
        "evaluation_ids_hash",
        "candidate_hash",
        "fixed_probe_hash",
        "remaining_candidate_hash",
        "choose_size_after_fixed",
        "total_combinations",
        "num_waves",
        "num_shards",
        "wave_index",
        "shard_index",
        "assignment_residue",
        "assignment_modulus",
        "chunk_size",
    )
    return {key: config.get(key) for key in keys}


def _config_for_chunk_v2(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "config_schema",
        "eval_protocol",
        "upstream_reference_commit",
        "k",
        "metric",
        "seed",
        "n_models",
        "n_evaluations",
        "n_observed",
        "n_target_cells",
        "predictor_rank",
        "predictor_regularization",
        "scores_sha256",
        "model_ids_sha256",
        "evaluation_ids_sha256",
        "candidate_ids_sha256",
        "fixed_probe_ids_sha256",
        "remaining_candidate_ids_sha256",
        "choose_size_after_fixed",
        "total_combinations",
        "num_waves",
        "num_shards",
        "wave_index",
        "shard_index",
        "assignment_residue",
        "assignment_modulus",
        "chunk_size",
        "execution_backend",
    )
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"schema-v2 config is missing required keys: {missing}")
    return {key: config[key] for key in keys}


def _config_for_chunk(config: dict[str, Any]) -> dict[str, Any]:
    schema_version = config.get("schema_version", 1)
    if schema_version == 1:
        return _config_for_chunk_v1(config)
    if schema_version == CONFIG_SCHEMA_VERSION:
        return _config_for_chunk_v2(config)
    raise ValueError(f"Unsupported run config schema_version={schema_version!r}")


def _config_at_location(
    base_config: dict[str, Any], wave_index: int, shard_index: int
) -> dict[str, Any]:
    config = dict(base_config)
    num_waves = int(config["num_waves"])
    num_shards = int(config["num_shards"])
    total = int(config["total_combinations"])
    modulus = num_waves * num_shards
    residue = _assignment_residue(wave_index, num_waves, shard_index, num_shards)
    config.update(
        wave_index=wave_index,
        shard_index=shard_index,
        assignment_residue=residue,
        assignment_modulus=modulus,
        assigned_combinations=_assigned_count(total, residue, modulus),
    )
    return config


def _chunk_path(out_dir: Path, wave_index: int, shard_index: int, chunk_index: int) -> Path:
    return (
        out_dir
        / "shards"
        / f"wave_{wave_index:02d}"
        / f"shard_{shard_index:03d}"
        / f"chunk_{chunk_index:06d}.json.gz"
    )


def _expected_combo_indices(
    total: int, residue: int, modulus: int, chunk_size: int, chunk_index: int
) -> list[int]:
    assigned = _assigned_count(total, residue, modulus)
    start = chunk_index * chunk_size
    stop = min(start + chunk_size, assigned)
    return [residue + modulus * offset for offset in range(start, stop)]


def _unrank_combination(n: int, k: int, rank: int) -> tuple[int, ...]:
    if rank < 0 or rank >= math.comb(n, k):
        raise ValueError(f"combination rank out of bounds: {rank}")
    result: list[int] = []
    start = 0
    remaining_rank = rank
    for position in range(k):
        slots_after = k - position - 1
        for candidate in range(start, n):
            count = math.comb(n - candidate - 1, slots_after)
            if remaining_rank < count:
                result.append(candidate)
                start = candidate + 1
                break
            remaining_rank -= count
        else:  # pragma: no cover - defensive combinatorial invariant
            raise RuntimeError("Could not unrank combination")
    return tuple(result)


def _valid_v2_record(
    record: Any, config: dict[str, Any], combo_index: int
) -> tuple[bool, str]:
    required = {
        "combo_index",
        "probe_set",
        "probe_names",
        "score",
        "medape",
        "medae",
        "n",
        "elapsed_s",
        "predictions",
    }
    if not isinstance(record, dict) or set(record) != required:
        return False, f"record keys/type mismatch at combo {combo_index}"
    if record["combo_index"] != combo_index:
        return False, f"record combo_index mismatch at combo {combo_index}"
    remaining = config["remaining_candidate_ids"]
    positions = _unrank_combination(
        len(remaining), int(config["choose_size_after_fixed"]), combo_index
    )
    expected_probes = list(config["fixed_probe_ids"]) + [
        remaining[position] for position in positions
    ]
    if record["probe_set"] != expected_probes or record["probe_names"] != expected_probes:
        return False, f"probe identity/order mismatch at combo {combo_index}"
    if (
        not isinstance(record["n"], int)
        or isinstance(record["n"], bool)
        or record["n"] != int(config["n_target_cells"])
        or not _finite_nonnegative(record["elapsed_s"])
    ):
        return False, f"record count/elapsed mismatch at combo {combo_index}"
    predictions = record["predictions"]
    if not isinstance(predictions, dict) or set(predictions) != {"i", "j", "true", "pred"}:
        return False, f"prediction keys/type mismatch at combo {combo_index}"
    n = record["n"]
    values = [predictions[key] for key in ("i", "j", "true", "pred")]
    if not all(isinstance(value, list) and len(value) == n for value in values):
        return False, f"prediction length mismatch at combo {combo_index}"
    rows = predictions["i"]
    columns = predictions["j"]
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and 0 <= value < config["n_models"]
        for value in rows
    ) or not all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < config["n_evaluations"]
        for value in columns
    ):
        return False, f"prediction coordinate mismatch at combo {combo_index}"
    coordinates = list(zip(rows, columns))
    if coordinates != sorted(coordinates) or len(set(coordinates)) != n:
        return False, f"prediction coordinate order/uniqueness mismatch at combo {combo_index}"
    truth = predictions["true"]
    predicted = predictions["pred"]
    if not all(_finite_nonnegative(value) and float(value) <= 100.0 for value in truth):
        return False, f"truth range mismatch at combo {combo_index}"
    if not all(_finite_nonnegative(value) and float(value) <= 100.0 for value in predicted):
        return False, f"prediction range mismatch at combo {combo_index}"
    evaluation_index = {
        evaluation_id: index for index, evaluation_id in enumerate(config["evaluation_ids"])
    }
    probe_columns = {evaluation_index[value] for value in expected_probes}
    if any(
        column in probe_columns and float(actual) != float(candidate)
        for column, actual, candidate in zip(columns, truth, predicted)
    ):
        return False, f"revealed probe prediction mismatch at combo {combo_index}"
    actual_array = np.asarray(truth, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    medae = float(np.median(np.abs(predicted_array - actual_array)))
    denominator = np.abs(actual_array) > 1e-6
    medape = float(
        np.median(
            np.abs((predicted_array[denominator] - actual_array[denominator])
                   / actual_array[denominator])
            * 100.0
        )
    )
    metrics = {"medae": medae, "medape": medape}
    for key, expected in metrics.items():
        if not _finite_nonnegative(record[key]) or abs(float(record[key]) - expected) > 1e-11:
            return False, f"{key} mismatch at combo {combo_index}"
    if (
        not _finite_nonnegative(record["score"])
        or abs(float(record["score"]) - metrics[config["metric"]]) > 1e-11
    ):
        return False, f"objective mismatch at combo {combo_index}"
    return True, "ok"


def _valid_chunk_payload(
    payload: Any, config: dict[str, Any], combo_indices: list[int]
) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload is not an object"
    if payload.get("config") != _config_for_chunk(config):
        return False, "chunk config mismatch"
    if payload.get("combo_indices") != combo_indices:
        return False, "combo_indices mismatch"
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(combo_indices):
        return False, "record count mismatch"
    if config.get("schema_version", 1) == CONFIG_SCHEMA_VERSION:
        for record, combo_index in zip(records, combo_indices):
            valid, reason = _valid_v2_record(record, config, combo_index)
            if not valid:
                return False, reason
        return True, "ok"
    record_indices = [record.get("combo_index") for record in records if isinstance(record, dict)]
    if record_indices != combo_indices:
        return False, "record combo_index mismatch"
    return True, "ok"


def _is_valid_existing_chunk(
    path: Path, config: dict[str, Any], combo_indices: list[int]
) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        payload = _load_json(path)
    except (OSError, EOFError, json.JSONDecodeError):
        return False
    valid, _ = _valid_chunk_payload(payload, config, combo_indices)
    return valid


def _init_worker(
    matrix: np.ndarray,
    evaluation_ids: tuple[str, ...],
    seed: int,
    fast_library: str | None = None,
) -> None:
    global _WORKER_MATRIX, _WORKER_EVALUATION_IDS
    global _WORKER_FAST_FUNCTION, _WORKER_FAST_INITIAL_ROWS, _WORKER_FAST_INITIAL_COLUMNS
    _WORKER_MATRIX = matrix
    _WORKER_EVALUATION_IDS = evaluation_ids
    _WORKER_FAST_FUNCTION = None
    _WORKER_FAST_INITIAL_ROWS = None
    _WORKER_FAST_INITIAL_COLUMNS = None
    np.random.seed(seed)
    random.seed(seed)
    if fast_library is not None:
        library = ctypes.CDLL(fast_library)
        function = library.complete_target_rank1
        vector = np.ctypeslib.ndpointer(
            dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
        )
        function.argtypes = [
            vector,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            vector,
            vector,
            vector,
        ]
        function.restype = ctypes.c_int
        initial_rows = []
        initial_columns = []
        for offset in range(10):
            rng = np.random.RandomState(seed + offset)
            initial_rows.append(rng.normal(0.0, 0.01, size=matrix.shape[0]))
            initial_columns.append(rng.normal(0.0, 0.01, size=matrix.shape[1]))
        _WORKER_FAST_FUNCTION = function
        _WORKER_FAST_INITIAL_ROWS = np.ascontiguousarray(initial_rows).ravel()
        _WORKER_FAST_INITIAL_COLUMNS = np.ascontiguousarray(initial_columns).ravel()


def _pack_predictions(result: Any) -> dict[str, list[int] | list[float]]:
    rows, columns = np.where(result.target_mask)
    return {
        "i": [int(value) for value in rows],
        "j": [int(value) for value in columns],
        "true": [float(result.actual[i, j]) for i, j in zip(rows, columns)],
        "pred": [float(result.predicted[i, j]) for i, j in zip(rows, columns)],
    }


def _predict_all_known_fast(
    matrix: np.ndarray, probe_indices: tuple[int, ...]
) -> ProbePredictions:
    if (
        _WORKER_FAST_FUNCTION is None
        or _WORKER_FAST_INITIAL_ROWS is None
        or _WORKER_FAST_INITIAL_COLUMNS is None
    ):
        raise RuntimeError("fast worker was not initialized")
    observed = np.isfinite(matrix)
    probe_columns = np.zeros(matrix.shape[1], dtype=bool)
    probe_columns[list(probe_indices)] = True
    revealed = observed & probe_columns[None, :]
    heldout = observed & ~probe_columns[None, :]
    predicted = np.full_like(matrix, np.nan)
    predicted[revealed] = matrix[revealed]
    for row in range(matrix.shape[0]):
        hidden = heldout[row]
        if not hidden.any():
            continue
        train = matrix.copy()
        train[row, ~probe_columns] = np.nan
        target_predictions = np.empty(matrix.shape[1], dtype=np.float64)
        return_code = _WORKER_FAST_FUNCTION(
            np.ascontiguousarray(train).ravel(),
            matrix.shape[0],
            matrix.shape[1],
            row,
            _WORKER_FAST_INITIAL_ROWS,
            _WORKER_FAST_INITIAL_COLUMNS,
            target_predictions,
        )
        if return_code != 0:
            raise RuntimeError(f"fast rank-1 completion failed with code {return_code}")
        predicted[row, hidden] = target_predictions[hidden]
    return ProbePredictions(
        tuple(probe_indices), matrix, predicted, observed, revealed, heldout
    )


def _evaluate_combo(job: tuple[int, tuple[int, ...], str]) -> dict[str, Any]:
    combo_index, probe_indices, metric = job
    if _WORKER_MATRIX is None:
        raise RuntimeError("worker was not initialized")
    started = time.time()
    if _WORKER_FAST_FUNCTION is None:
        predictions = predict_all_known(
            _WORKER_MATRIX,
            probe_indices,
            rank=PREDICTOR_RANK,
            regularization=PREDICTOR_REGULARIZATION,
        )
    else:
        predictions = _predict_all_known_fast(_WORKER_MATRIX, probe_indices)
    metrics = score_predictions(predictions)
    probe_ids = [_WORKER_EVALUATION_IDS[index] for index in probe_indices]

    def finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "combo_index": int(combo_index),
        "probe_set": probe_ids,
        # PathoPress evaluation IDs are already the canonical display labels.
        "probe_names": list(probe_ids),
        "score": finite_or_none(float(metrics[metric])),
        "medape": finite_or_none(float(metrics["medape"])),
        "medae": finite_or_none(float(metrics["medae"])),
        "n": int(metrics["n_target"]),
        "elapsed_s": time.time() - started,
        "predictions": _pack_predictions(predictions),
    }


def _write_config(out_dir: Path, config: dict[str, Any]) -> None:
    run_config = _base_run_config(config)
    path = out_dir / "config.json"
    if path.exists():
        existing = _load_json(path)
        if existing != run_config:
            raise RuntimeError(
                f"Existing {path} has an incompatible config. Use a different "
                "--out-dir or move the stale run directory."
            )
        return
    _write_json_atomic(path, run_config, indent=2)


def iter_assigned_combos(
    fixed_indices: list[int],
    remaining_candidates: list[int],
    choose_size: int,
    residue: int,
    modulus: int,
) -> Iterable[tuple[int, tuple[int, ...]]]:
    for combo_index, combo in enumerate(itertools.combinations(remaining_candidates, choose_size)):
        if combo_index % modulus == residue:
            yield combo_index, tuple(fixed_indices) + tuple(combo)


def run_shard(args: argparse.Namespace) -> None:
    fast_descriptor: int | None = None
    staged_directory: Path | None = None
    fast_worker_path: str | None = None
    equivalence: dict[str, Any] | None = None
    library_sha256: str | None = None
    try:
        if args.fast_library is not None:
            equivalence = _validate_fast_equivalence(
                args.scores, args.fast_library, args.fast_equivalence
            )
            expected_sha256 = str(equivalence["inputs"]["library_sha256"])
            fast_descriptor, staged_directory, _, library_sha256 = _stage_fast_library(
                args.fast_library, expected_sha256
            )
            fast_worker_path = f"/proc/self/fd/{fast_descriptor}"
        backend = _execution_backend_config(
            equivalence,
            args.fast_equivalence if equivalence is not None else None,
            library_sha256,
        )
        _run_shard_with_backend(args, fast_worker_path, backend)
    finally:
        _cleanup_staged_fast_library(fast_descriptor, staged_directory)


def _run_shard_with_backend(
    args: argparse.Namespace,
    fast_worker_path: str | None,
    execution_backend: dict[str, Any],
) -> None:
    matrix, model_ids, evaluation_ids = _load_matrix(args.scores)
    candidate_indices, candidate_ids, allowlist_sha256 = _load_candidates(
        evaluation_ids, args.candidate_allowlist, args.candidate_limit
    )
    out_dir = (args.out_dir or _default_out_dir(
        args, _candidate_source_label(args.candidate_allowlist)
    )).resolve()
    _assert_run_is_active(out_dir)
    config, fixed_indices, remaining_candidates = _build_config(
        args,
        matrix,
        model_ids,
        evaluation_ids,
        candidate_indices,
        candidate_ids,
        allowlist_sha256,
        execution_backend,
    )
    _write_config(out_dir, config)

    print("=== Exhaustive pathology probe shard ===", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(
        f"k={config['k']} fixed={config['fixed_probe_ids']} "
        f"choose_size={config['choose_size_after_fixed']}",
        flush=True,
    )
    print(
        f"total={config['total_combinations']} assigned={config['assigned_combinations']} "
        f"wave={config['wave_index']}/{config['num_waves']} "
        f"shard={config['shard_index']}/{config['num_shards']} workers={args.workers}",
        flush=True,
    )

    assigned_seen = 0
    evaluated = 0
    skipped_chunks = 0
    started = time.time()
    chunk_jobs: list[tuple[int, int, tuple[int, ...]]] = []
    pool = ProcessPoolExecutor(
        max_workers=int(args.workers),
        mp_context=multiprocessing.get_context("fork"),
        initializer=_init_worker,
        initargs=(matrix, tuple(evaluation_ids), SEED, fast_worker_path),
    )

    def flush_chunk(jobs: list[tuple[int, int, tuple[int, ...]]]) -> None:
        nonlocal evaluated, skipped_chunks
        if not jobs:
            return
        chunk_index = jobs[0][0]
        combo_indices = [job[1] for job in jobs]
        path = _chunk_path(
            out_dir, config["wave_index"], config["shard_index"], chunk_index
        )
        if _is_valid_existing_chunk(path, config, combo_indices):
            skipped_chunks += 1
            print(f"skip chunk {chunk_index:06d} ({len(jobs)} combos)", flush=True)
            return

        print(f"run chunk {chunk_index:06d} ({len(jobs)} combos) -> {path}", flush=True)
        chunk_started = time.time()
        worker_jobs = [
            (combo_index, probe_indices, args.metric)
            for _, combo_index, probe_indices in jobs
        ]
        records: list[dict[str, Any]] = []
        futures = [pool.submit(_evaluate_combo, job) for job in worker_jobs]
        for future in as_completed(futures):
            records.append(future.result())
        records.sort(key=lambda row: int(row["combo_index"]))
        payload = {
            "config": _config_for_chunk(config),
            "combo_indices": combo_indices,
            "records": records,
            "elapsed_s": time.time() - chunk_started,
        }
        valid, reason = _valid_chunk_payload(payload, config, combo_indices)
        if not valid:
            raise RuntimeError(f"Refusing to write invalid chunk: {reason}")
        _write_json_atomic(path, payload)
        evaluated += len(records)
        print(f"wrote chunk {chunk_index:06d}: {len(records)} records", flush=True)

    choose_size = int(config["choose_size_after_fixed"])
    residue = int(config["assignment_residue"])
    modulus = int(config["assignment_modulus"])
    try:
        for combo_index, probe_indices in iter_assigned_combos(
            fixed_indices, remaining_candidates, choose_size, residue, modulus
        ):
            if args.max_subsets is not None and assigned_seen >= args.max_subsets:
                break
            chunk_index = assigned_seen // args.chunk_size
            chunk_jobs.append((chunk_index, combo_index, probe_indices))
            assigned_seen += 1
            if len(chunk_jobs) >= args.chunk_size:
                flush_chunk(chunk_jobs)
                chunk_jobs = []
        flush_chunk(chunk_jobs)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    print(
        f"done: assigned_seen={assigned_seen} evaluated={evaluated} "
        f"skipped_chunks={skipped_chunks} elapsed_s={time.time() - started:.1f}",
        flush=True,
    )


def _expected_chunk_count(assigned_count: int, chunk_size: int) -> int:
    return math.ceil(assigned_count / chunk_size) if assigned_count else 0


def merge(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    _assert_run_is_active(out_dir)
    config_path = out_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    base_config = _load_json(config_path)
    num_waves = int(base_config["num_waves"])
    num_shards = int(base_config["num_shards"])
    total = int(base_config["total_combinations"])
    chunk_size = int(base_config["chunk_size"])
    modulus = num_waves * num_shards

    integrity_chunk_hashes: dict[str, str] | None = None
    integrity_provenance: dict[str, str] | None = None
    integrity_argument = getattr(args, "integrity_manifest", None)
    if integrity_argument is not None:
        integrity_path = integrity_argument.resolve()
        integrity = _load_json(integrity_path)
        if integrity.get("status") != "passed":
            raise RuntimeError("Integrity manifest did not pass")
        config_sha256 = _sha256_bytes(config_path)
        matches = [
            row
            for row in integrity.get("runs", [])
            if row.get("config_sha256") == config_sha256
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Integrity manifest must contain exactly one run for {config_sha256}"
            )
        integrity_run = matches[0]
        if (
            int(integrity_run.get("validated_records", -1)) != total
            or int(integrity_run.get("validated_chunks", -1))
            != int(integrity_run.get("expected_chunks", -2))
        ):
            raise RuntimeError("Integrity manifest does not certify a complete run")
        integrity_chunk_hashes = {
            str(row["path"]): str(row["sha256"])
            for row in integrity_run.get("chunks", [])
        }
        if len(integrity_chunk_hashes) != int(integrity_run["validated_chunks"]):
            raise RuntimeError("Integrity manifest has duplicate/missing chunk entries")
        integrity_provenance = {
            "path": _display_path(integrity_path),
            "sha256": _sha256_bytes(integrity_path),
            "config_sha256": config_sha256,
            "chunk_digest_aggregate_sha256": str(
                integrity_run["chunk_digest_aggregate_sha256"]
            ),
        }

    summaries: list[dict[str, Any]] = []
    missing_chunks: list[str] = []
    invalid_chunks: list[dict[str, str]] = []
    seen: set[int] = set()
    for wave_index in range(num_waves):
        for shard_index in range(num_shards):
            location_config = _config_at_location(base_config, wave_index, shard_index)
            residue = int(location_config["assignment_residue"])
            assigned = _assigned_count(total, residue, modulus)
            for chunk_index in range(_expected_chunk_count(assigned, chunk_size)):
                path = _chunk_path(out_dir, wave_index, shard_index, chunk_index)
                expected_indices = _expected_combo_indices(
                    total, residue, modulus, chunk_size, chunk_index
                )
                if not path.exists():
                    missing_chunks.append(str(path))
                    continue
                if integrity_chunk_hashes is not None:
                    key = str(_display_path(path))
                    expected_hash = integrity_chunk_hashes.get(key)
                    if expected_hash is None:
                        invalid_chunks.append(
                            {"path": str(path), "reason": "absent from integrity manifest"}
                        )
                        continue
                    if _sha256_bytes(path) != expected_hash:
                        invalid_chunks.append(
                            {"path": str(path), "reason": "integrity SHA256 mismatch"}
                        )
                        continue
                try:
                    payload = _load_json(path)
                except (OSError, EOFError, json.JSONDecodeError) as error:
                    invalid_chunks.append({"path": str(path), "reason": str(error)})
                    continue
                valid, reason = _valid_chunk_payload(
                    payload, location_config, expected_indices
                )
                if not valid:
                    invalid_chunks.append({"path": str(path), "reason": reason})
                    continue
                for record in payload["records"]:
                    combo_index = int(record["combo_index"])
                    if combo_index in seen:
                        raise RuntimeError(f"Duplicate combo_index {combo_index} in {path}")
                    seen.add(combo_index)
                    summaries.append(
                        {
                            "combo_index": combo_index,
                            "probe_set": record["probe_set"],
                            "score": record["score"],
                            "medape": record["medape"],
                            "medae": record["medae"],
                            "n": record["n"],
                            "elapsed_s": record.get("elapsed_s"),
                        }
                    )

    problems = len(missing_chunks) + len(invalid_chunks)
    if problems and not args.allow_incomplete:
        first = missing_chunks[0] if missing_chunks else invalid_chunks[0]
        raise RuntimeError(
            f"Run is incomplete: {len(missing_chunks)} missing and "
            f"{len(invalid_chunks)} invalid chunks. First problem: {first}. "
            "Use --allow-incomplete only for diagnostics."
        )
    if not args.allow_incomplete and seen != set(range(total)):
        missing_indices = sorted(set(range(total)) - seen)
        raise RuntimeError(
            f"Expected {total} unique combo records, found {len(seen)}; "
            f"first missing combo_index={missing_indices[0] if missing_indices else None}"
        )

    summaries.sort(
        key=lambda row: (
            float("inf") if row["score"] is None else float(row["score"]),
            int(row["combo_index"]),
        )
    )
    complete = not problems and len(seen) == total
    payload = {
        "config": base_config,
        "complete": complete,
        "n_records": len(seen),
        "missing_chunks": missing_chunks,
        "invalid_chunks": invalid_chunks,
        "integrity_manifest": integrity_provenance,
        "best": summaries[0] if summaries else None,
        "top": summaries[: args.top_n],
    }
    output = out_dir / "merged_summary.json.gz"
    _write_json_atomic(output, payload, indent=2)
    print(f"merged -> {output}", flush=True)
    if payload["best"] is not None:
        print(
            f"best score={payload['best']['score']} "
            f"probe_set={payload['best']['probe_set']}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-shard", help="evaluate one wave/shard residue")
    run.add_argument("--scores", type=Path, default=ROOT / "data" / "scores.csv")
    run.add_argument("--candidate-allowlist", type=Path, default=None)
    run.add_argument("--candidate-limit", type=int, default=None)
    run.add_argument("--fixed-probe", default=None, help="comma-separated evaluation IDs")
    run.add_argument("--k", type=int, default=5)
    run.add_argument("--metric", choices=("medae", "medape"), default="medae")
    run.add_argument("--num-waves", type=int, default=10)
    run.add_argument("--wave-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    run.add_argument(
        "--fast-library",
        type=Path,
        default=None,
        help=(
            "optional compiled experiments/fast_rank1.cpp execution backend; "
            "scientific equivalence must be verified before production use"
        ),
    )
    run.add_argument(
        "--fast-equivalence",
        type=Path,
        default=DEFAULT_FAST_EQUIVALENCE,
        help="hash-bound equivalence evidence emitted by verify_fast_rank1.py",
    )
    run.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    run.add_argument(
        "--max-subsets",
        type=int,
        default=None,
        help="diagnostic bound only; a bounded run cannot pass complete merge",
    )
    run.add_argument("--out-dir", type=Path, default=None)
    run.set_defaults(func=run_shard)

    merge_parser = subparsers.add_parser("merge", help="validate and compact completed shards")
    merge_parser.add_argument("--out-dir", type=Path, required=True)
    merge_parser.add_argument("--top-n", type=int, default=100)
    merge_parser.add_argument("--allow-incomplete", action="store_true")
    merge_parser.add_argument(
        "--integrity-manifest",
        type=Path,
        default=None,
        help=(
            "full-record validation manifest; when supplied, every raw chunk "
            "must still match its certified SHA256 before merge"
        ),
    )
    merge_parser.set_defaults(func=merge)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "max_subsets", None) is not None and args.max_subsets < 0:
        parser.error("--max-subsets must be non-negative")
    if getattr(args, "top_n", 1) < 0:
        parser.error("--top-n must be non-negative")
    args.func(args)


if __name__ == "__main__":
    main()
