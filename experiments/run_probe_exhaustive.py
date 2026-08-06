#!/usr/bin/env python3
"""Resumable exhaustive all-known probe search using the BenchPress contract.

This is the pathology adaptation of BenchPress's
``all_known_probe_bruteforce_v1`` runner.  Each invocation evaluates one
wave/shard residue.  ``merge`` is deliberately complete-by-default: a partial
run is rejected unless ``--allow-incomplete`` is explicitly requested.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import random
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
    predict_all_known,
    score_predictions,
)


SEED = 42
EVAL_PROTOCOL = "all_known_probe_bruteforce_v1"
UPSTREAM_REFERENCE_COMMIT = "0a684b63ee0e4a401cb907a3827a82ea997d74c4"
DEFAULT_CHUNK_SIZE = 256
PREDICTOR_RANK = 1
PREDICTOR_REGULARIZATION = 0.1

_WORKER_MATRIX: np.ndarray | None = None
_WORKER_EVALUATION_IDS: tuple[str, ...] = ()


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


def _short_text_hash(values: Sequence[str]) -> str:
    return hashlib.sha1("\n".join(values).encode("utf-8")).hexdigest()[:12]


def _safe_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


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
    return filter_matrix(*make_matrix(scores))


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
        "model_ids_hash": _short_text_hash(model_ids),
        "evaluation_ids_hash": _short_text_hash(evaluation_ids),
        "candidate_allowlist_path": _display_path(args.candidate_allowlist),
        "candidate_allowlist_sha256": allowlist_sha256,
        "candidate_limit": args.candidate_limit,
        "candidate_ids": candidate_ids,
        "candidate_hash": _short_text_hash(candidate_ids),
        "fixed_probe_ids": fixed_ids,
        "fixed_probe_hash": _short_text_hash(fixed_ids) if fixed_ids else None,
        "remaining_candidate_ids": remaining_ids,
        "remaining_candidate_hash": _short_text_hash(remaining_ids),
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


def _config_for_chunk(config: dict[str, Any]) -> dict[str, Any]:
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


def _init_worker(matrix: np.ndarray, evaluation_ids: tuple[str, ...], seed: int) -> None:
    global _WORKER_MATRIX, _WORKER_EVALUATION_IDS
    _WORKER_MATRIX = matrix
    _WORKER_EVALUATION_IDS = evaluation_ids
    np.random.seed(seed)
    random.seed(seed)


def _pack_predictions(result: Any) -> dict[str, list[int] | list[float]]:
    rows, columns = np.where(result.target_mask)
    return {
        "i": [int(value) for value in rows],
        "j": [int(value) for value in columns],
        "true": [float(result.actual[i, j]) for i, j in zip(rows, columns)],
        "pred": [float(result.predicted[i, j]) for i, j in zip(rows, columns)],
    }


def _evaluate_combo(job: tuple[int, tuple[int, ...], str]) -> dict[str, Any]:
    combo_index, probe_indices, metric = job
    if _WORKER_MATRIX is None:
        raise RuntimeError("worker was not initialized")
    started = time.time()
    predictions = predict_all_known(
        _WORKER_MATRIX,
        probe_indices,
        rank=PREDICTOR_RANK,
        regularization=PREDICTOR_REGULARIZATION,
    )
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
    matrix, model_ids, evaluation_ids = _load_matrix(args.scores)
    candidate_indices, candidate_ids, allowlist_sha256 = _load_candidates(
        evaluation_ids, args.candidate_allowlist, args.candidate_limit
    )
    out_dir = (args.out_dir or _default_out_dir(
        args, _candidate_source_label(args.candidate_allowlist)
    )).resolve()
    config, fixed_indices, remaining_candidates = _build_config(
        args,
        matrix,
        model_ids,
        evaluation_ids,
        candidate_indices,
        candidate_ids,
        allowlist_sha256,
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
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_init_worker,
            initargs=(matrix, tuple(evaluation_ids), SEED),
        ) as pool:
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
    print(
        f"done: assigned_seen={assigned_seen} evaluated={evaluated} "
        f"skipped_chunks={skipped_chunks} elapsed_s={time.time() - started:.1f}",
        flush=True,
    )


def _expected_chunk_count(assigned_count: int, chunk_size: int) -> int:
    return math.ceil(assigned_count / chunk_size) if assigned_count else 0


def merge(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    config_path = out_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    base_config = _load_json(config_path)
    num_waves = int(base_config["num_waves"])
    num_shards = int(base_config["num_shards"])
    total = int(base_config["total_combinations"])
    chunk_size = int(base_config["chunk_size"])
    modulus = num_waves * num_shards

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
