#!/usr/bin/env python3
"""Deep-validate every raw exact-search record and emit a provenance manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import inspect
import itertools
import json
import math
import os
import platform
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import run_probe_exhaustive as runner  # noqa: E402


METRIC_TOLERANCE = 1e-11
CELL_EQUIVALENCE_CAP = 1e-10
METRIC_EQUIVALENCE_CAP = 1e-11
MIN_EQUIVALENCE_COMPARISONS = 8
PREDICTION_MIN = 0.0
PREDICTION_MAX = 100.0
MAX_COMPRESSED_CHUNK_BYTES = 64 * 1024 * 1024
MAX_DECOMPRESSED_CHUNK_BYTES = 256 * 1024 * 1024
EXPECTED_TOP_COUNT = 1001
COMPILE_COMMAND = (
    "g++ -O3 -std=c++17 -fPIC -shared -ffp-contract=off "
    "experiments/fast_rank1.cpp -o /tmp/libpathopress_fast_rank1.so"
)
EXPECTED_CONFIG_KEYS = {
    "eval_protocol", "upstream_reference_commit", "k", "metric", "seed",
    "n_models", "n_evaluations", "n_observed", "n_target_cells", "eval_scope",
    "predictor_rank", "predictor_regularization", "prediction_engine",
    "scores_path", "scores_sha256", "model_ids_hash", "evaluation_ids_hash",
    "candidate_allowlist_path", "candidate_allowlist_sha256", "candidate_limit",
    "candidate_ids", "candidate_hash", "fixed_probe_ids", "fixed_probe_hash",
    "remaining_candidate_ids", "remaining_candidate_hash",
    "choose_size_after_fixed", "total_combinations", "num_waves", "num_shards",
    "assignment_modulus", "chunk_size", "cell_masking",
}
EXPECTED_MASKING_TEXT = (
    "For model i, keep observed probe cells visible and mask observed "
    "non-probe cells. Probe target cells are stored with pred=true; "
    "non-probe target cells are rank-1 predictions. The evaluation "
    "universe is fixed to all observed cells."
)
EXPECTED_ENGINE_TEXT = (
    "pathopress.complete (masked low-rank completion, rank=1, regularization=0.1)"
)

_MATRIX: np.ndarray | None = None
_EVALUATION_IDS: tuple[str, ...] = ()
_TARGET_I: np.ndarray | None = None
_TARGET_J: np.ndarray | None = None
_TARGET_TRUE: np.ndarray | None = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_reject_json_constant)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant rejected: {value}")


def _require_regular(path: Path, label: str) -> None:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a non-symlink regular file: {path}")


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _init_worker(matrix: np.ndarray, evaluation_ids: tuple[str, ...]) -> None:
    global _MATRIX, _EVALUATION_IDS, _TARGET_I, _TARGET_J, _TARGET_TRUE
    _MATRIX = matrix
    _EVALUATION_IDS = evaluation_ids
    observed = np.isfinite(matrix)
    _TARGET_I, _TARGET_J = np.where(observed)
    _TARGET_TRUE = matrix[_TARGET_I, _TARGET_J]


def _unrank_combination(n: int, k: int, rank: int) -> tuple[int, ...]:
    """Return itertools.combinations(range(n), k)[rank] without materializing it."""

    if rank < 0 or rank >= math.comb(n, k):
        raise ValueError(f"combination rank out of bounds: {rank}")
    output: list[int] = []
    start = 0
    remaining_rank = rank
    for position in range(k):
        slots_after = k - position - 1
        for candidate in range(start, n):
            count = math.comb(n - candidate - 1, slots_after)
            if remaining_rank < count:
                output.append(candidate)
                start = candidate + 1
                break
            remaining_rank -= count
        else:  # pragma: no cover - defensive proof guard
            raise RuntimeError("could not unrank combination")
    return tuple(output)


def _same_float(actual: Any, expected: float, tolerance: float = METRIC_TOLERANCE) -> bool:
    return (
        _finite_number(actual)
        and abs(float(actual) - expected) <= tolerance
    )


def _validate_record(
    record: Any,
    combo_index: int,
    config: dict[str, Any],
) -> tuple[float, float, float, float]:
    if _MATRIX is None or _TARGET_I is None or _TARGET_J is None or _TARGET_TRUE is None:
        raise RuntimeError("validator worker is uninitialized")
    required_keys = {
        "combo_index", "probe_set", "probe_names", "score", "medape", "medae",
        "n", "elapsed_s", "predictions",
    }
    if not isinstance(record, dict) or set(record) != required_keys:
        raise ValueError(f"record keys/type mismatch at combo {combo_index}")
    if record["combo_index"] != combo_index:
        raise ValueError(f"combo_index mismatch at combo {combo_index}")
    fixed_ids = tuple(config["fixed_probe_ids"])
    remaining_ids = tuple(config["remaining_candidate_ids"])
    choose_size = int(config["choose_size_after_fixed"])
    positions = _unrank_combination(len(remaining_ids), choose_size, combo_index)
    expected_probe_ids = fixed_ids + tuple(remaining_ids[value] for value in positions)
    if record["probe_set"] != list(expected_probe_ids):
        raise ValueError(f"probe identity/order mismatch at combo {combo_index}")
    if record["probe_names"] != list(expected_probe_ids):
        raise ValueError(f"probe_names mismatch at combo {combo_index}")
    if not (
        isinstance(record["n"], int)
        and not isinstance(record["n"], bool)
        and record["n"] == int(config["n_target_cells"])
    ):
        raise ValueError(f"n mismatch at combo {combo_index}")
    if not _finite_number(record["elapsed_s"]) or float(record["elapsed_s"]) < 0:
        raise ValueError(f"invalid elapsed_s at combo {combo_index}")

    packed = record["predictions"]
    if not isinstance(packed, dict) or set(packed) != {"i", "j", "true", "pred"}:
        raise ValueError(f"packed prediction schema mismatch at combo {combo_index}")
    count = int(config["n_target_cells"])
    if any(not isinstance(packed[key], list) or len(packed[key]) != count for key in packed):
        raise ValueError(f"packed prediction length mismatch at combo {combo_index}")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in packed["i"]):
        raise ValueError(f"non-integer row indices at combo {combo_index}")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in packed["j"]):
        raise ValueError(f"non-integer column indices at combo {combo_index}")
    for key in ("true", "pred"):
        if any(not _finite_number(value) for value in packed[key]):
            raise ValueError(
                f"non-numeric/non-finite {key} values at combo {combo_index}"
            )
    rows = np.asarray(packed["i"], dtype=np.int64)
    columns = np.asarray(packed["j"], dtype=np.int64)
    if not np.array_equal(rows, _TARGET_I) or not np.array_equal(columns, _TARGET_J):
        raise ValueError(f"target indices/order mismatch at combo {combo_index}")
    true = np.asarray(packed["true"], dtype=np.float64)
    predicted = np.asarray(packed["pred"], dtype=np.float64)
    if not np.isfinite(true).all() or not np.array_equal(true, _TARGET_TRUE):
        raise ValueError(f"target truth mismatch at combo {combo_index}")
    if not np.isfinite(predicted).all():
        raise ValueError(f"non-finite predictions at combo {combo_index}")
    if predicted.min() < PREDICTION_MIN or predicted.max() > PREDICTION_MAX:
        raise ValueError(f"prediction outside [0,100] at combo {combo_index}")

    probe_columns = {_EVALUATION_IDS.index(value) for value in expected_probe_ids}
    revealed = np.fromiter((int(value) in probe_columns for value in columns), bool, count=count)
    if not np.array_equal(predicted[revealed], true[revealed]):
        raise ValueError(f"revealed probe value mismatch at combo {combo_index}")
    errors = np.abs(predicted - true)
    medae = float(np.median(errors))
    percent_valid = np.abs(true) > 1e-6
    if not percent_valid.any():
        raise ValueError(f"no valid MedAPE targets at combo {combo_index}")
    medape = float(
        np.median(100.0 * errors[percent_valid] / np.abs(true[percent_valid]))
    )
    if not _same_float(record["medae"], medae):
        raise ValueError(f"recomputed MedAE mismatch at combo {combo_index}")
    if not _same_float(record["medape"], medape):
        raise ValueError(f"recomputed MedAPE mismatch at combo {combo_index}")
    expected_score = medae if config["metric"] == "medae" else medape
    if not _same_float(record["score"], expected_score):
        raise ValueError(f"recomputed objective mismatch at combo {combo_index}")
    return (
        abs(float(record["medae"]) - medae),
        abs(float(record["medape"]) - medape),
        float(predicted.min()),
        float(predicted.max()),
    )


def _validate_chunk(job: tuple[str, dict[str, Any], list[int]]) -> dict[str, Any]:
    path_text, location_config, expected_indices = job
    path = Path(path_text)
    _require_regular(path, "raw chunk")
    raw = path.read_bytes()
    if not 0 < len(raw) <= MAX_COMPRESSED_CHUNK_BYTES:
        raise ValueError(f"compressed chunk size outside bound: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as handle:
            decoded = handle.read(MAX_DECOMPRESSED_CHUNK_BYTES + 1)
        if len(decoded) > MAX_DECOMPRESSED_CHUNK_BYTES:
            raise ValueError(f"decompressed chunk exceeds bound: {path}")
        payload = json.loads(decoded, parse_constant=_reject_json_constant)
    except (OSError, EOFError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot decode {path}: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {
        "config", "combo_indices", "records", "elapsed_s"
    }:
        raise ValueError(f"chunk payload schema mismatch: {path}")
    if payload["config"] != runner._config_for_chunk(location_config):
        raise ValueError(f"chunk config mismatch: {path}")
    if payload["combo_indices"] != expected_indices:
        raise ValueError(f"chunk combo_indices mismatch: {path}")
    records = payload["records"]
    if not isinstance(records, list) or len(records) != len(expected_indices):
        raise ValueError(f"chunk record count mismatch: {path}")
    if not _finite_number(payload["elapsed_s"]) or float(payload["elapsed_s"]) < 0:
        raise ValueError(f"invalid chunk elapsed_s: {path}")
    max_medae_delta = 0.0
    max_medape_delta = 0.0
    prediction_min = float("inf")
    prediction_max = float("-inf")
    score_rows: list[dict[str, Any]] = []
    for record, combo_index in zip(records, expected_indices):
        medae_delta, medape_delta, low, high = _validate_record(
            record, combo_index, location_config
        )
        max_medae_delta = max(max_medae_delta, medae_delta)
        max_medape_delta = max(max_medape_delta, medape_delta)
        prediction_min = min(prediction_min, low)
        prediction_max = max(prediction_max, high)
        score_rows.append(
            {
                "combo_index": combo_index,
                "probe_set": record["probe_set"],
                "score": record["score"],
                "medape": record["medape"],
                "medae": record["medae"],
                "n": record["n"],
                "elapsed_s": record["elapsed_s"],
            }
        )
    return {
        "path": display(path),
        "sha256": digest,
        "bytes": len(raw),
        "records": len(records),
        "first_combo_index": expected_indices[0] if expected_indices else None,
        "last_combo_index": expected_indices[-1] if expected_indices else None,
        "max_absolute_medae_delta": max_medae_delta,
        "max_absolute_medape_delta": max_medape_delta,
        "prediction_min": prediction_min,
        "prediction_max": prediction_max,
        "score_rows": score_rows,
    }


def validate_run(
    run_dir: Path,
    matrix: np.ndarray,
    model_ids: list[str],
    evaluation_ids: list[str],
    cli_scores_path: Path,
    workers: int,
) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    _require_regular(config_path, "run config")
    config = load_json(config_path)
    if set(config) != EXPECTED_CONFIG_KEYS:
        raise RuntimeError(
            f"config key set mismatch: missing={sorted(EXPECTED_CONFIG_KEYS-set(config))}, "
            f"extra={sorted(set(config)-EXPECTED_CONFIG_KEYS)}"
        )
    observed = np.isfinite(matrix)
    scores_path = Path(config["scores_path"])
    if not scores_path.is_absolute():
        scores_path = ROOT / scores_path
    _require_regular(scores_path, "scores input")
    if (
        scores_path.resolve() != cli_scores_path.resolve()
        or config["scores_sha256"] != sha256(scores_path)
        or config["scores_sha256"] != sha256(cli_scores_path)
    ):
        raise RuntimeError(f"scores hash mismatch: {run_dir}")
    if [config["n_models"], config["n_evaluations"]] != list(matrix.shape):
        raise RuntimeError(f"matrix shape mismatch: {run_dir}")
    if config["n_observed"] != int(observed.sum()) or config["n_target_cells"] != int(observed.sum()):
        raise RuntimeError(f"observed count mismatch: {run_dir}")
    if config["evaluation_ids_hash"] != runner._short_text_hash(evaluation_ids):
        raise RuntimeError(f"evaluation ID hash mismatch: {run_dir}")
    if config["model_ids_hash"] != runner._short_text_hash(model_ids):
        raise RuntimeError(f"model ID hash mismatch: {run_dir}")
    allowlist = Path(config["candidate_allowlist_path"])
    if not allowlist.is_absolute():
        allowlist = ROOT / allowlist
    _require_regular(allowlist, "candidate allowlist")
    if config["candidate_allowlist_sha256"] != sha256(allowlist):
        raise RuntimeError(f"allowlist hash mismatch: {run_dir}")
    if config["candidate_ids"] != runner._allowlist_ids(load_json(allowlist), allowlist):
        raise RuntimeError(f"allowlist candidate mismatch: {run_dir}")
    if len(config["candidate_ids"]) != len(set(config["candidate_ids"])):
        raise RuntimeError(f"duplicate candidates: {run_dir}")
    if config["candidate_hash"] != runner._short_text_hash(config["candidate_ids"]):
        raise RuntimeError(f"candidate hash mismatch: {run_dir}")
    if config["remaining_candidate_hash"] != runner._short_text_hash(config["remaining_candidate_ids"]):
        raise RuntimeError(f"remaining candidate hash mismatch: {run_dir}")
    if (
        config["fixed_probe_ids"]
        or config["fixed_probe_hash"] is not None
        or config["candidate_limit"] is not None
    ):
        raise RuntimeError(f"this release expects no fixed probes: {run_dir}")
    if config["remaining_candidate_ids"] != config["candidate_ids"]:
        raise RuntimeError(f"remaining/candidate identity mismatch: {run_dir}")
    if any(value not in evaluation_ids for value in config["candidate_ids"]):
        raise RuntimeError(f"candidate outside evaluation universe: {run_dir}")
    total = math.comb(
        len(config["remaining_candidate_ids"]),
        int(config["choose_size_after_fixed"]),
    )
    if total != int(config["total_combinations"]):
        raise RuntimeError(f"combination total mismatch: {run_dir}")
    candidate_count = len(config["candidate_ids"])
    expected_partition = {25: (10, 8, 53130), 30: (20, 1, 142506)}.get(
        candidate_count
    )
    if expected_partition is None or (
        int(config["num_waves"]),
        int(config["num_shards"]),
        total,
    ) != expected_partition:
        raise RuntimeError(f"candidate-space/partition contract mismatch: {run_dir}")
    if (
        config["eval_protocol"] != runner.EVAL_PROTOCOL
        or config["upstream_reference_commit"] != runner.UPSTREAM_REFERENCE_COMMIT
        or config["predictor_rank"] != runner.PREDICTOR_RANK
        or config["predictor_regularization"] != runner.PREDICTOR_REGULARIZATION
        or config["metric"] != "medae"
        or config["k"] != 5
        or config["seed"] != runner.SEED
        or config["eval_scope"] != "all_observed_cells"
        or config["prediction_engine"] != EXPECTED_ENGINE_TEXT
        or config["choose_size_after_fixed"] != 5
        or config["assignment_modulus"]
        != int(config["num_waves"]) * int(config["num_shards"])
        or config["chunk_size"] != runner.DEFAULT_CHUNK_SIZE
        or config["cell_masking"] != EXPECTED_MASKING_TEXT
    ):
        raise RuntimeError(f"scientific contract mismatch: {run_dir}")

    jobs: list[tuple[str, dict[str, Any], list[int]]] = []
    expected_paths: set[Path] = set()
    for wave in range(int(config["num_waves"])):
        for shard in range(int(config["num_shards"])):
            location = runner._config_at_location(config, wave, shard)
            assigned = runner._assigned_count(
                total,
                int(location["assignment_residue"]),
                int(location["assignment_modulus"]),
            )
            for chunk in range(runner._expected_chunk_count(assigned, int(config["chunk_size"]))):
                path = runner._chunk_path(run_dir, wave, shard, chunk)
                expected_paths.add(path.resolve())
                expected_indices = runner._expected_combo_indices(
                    total,
                    int(location["assignment_residue"]),
                    int(location["assignment_modulus"]),
                    int(config["chunk_size"]),
                    chunk,
                )
                jobs.append((str(path), location, expected_indices))
    observed_paths = {
        path.resolve() for path in (run_dir / "shards").glob("**/chunk_*.json.gz")
    }
    missing = sorted(str(path) for path in expected_paths - observed_paths)
    unexpected = sorted(str(path) for path in observed_paths - expected_paths)
    temporary = sorted(str(path) for path in (run_dir / "shards").glob("**/*.tmp*"))
    if missing or unexpected or temporary:
        raise RuntimeError(
            f"chunk inventory mismatch for {run_dir}: missing={missing[:1]}, "
            f"unexpected={unexpected[:1]}, temporary={temporary[:1]}"
        )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(matrix, tuple(evaluation_ids)),
    ) as pool:
        chunks = list(pool.map(_validate_chunk, jobs, chunksize=1))
    chunks.sort(key=lambda row: row["path"])
    score_rows = [
        row
        for chunk in chunks
        for row in chunk.pop("score_rows")
    ]
    score_rows.sort(key=lambda row: (float(row["score"]), int(row["combo_index"])))
    expected_top = score_rows[: min(EXPECTED_TOP_COUNT, total)]
    combo_total = sum(int(row["records"]) for row in chunks)
    if combo_total != total:
        raise RuntimeError(f"validated record total mismatch: {run_dir}")
    digest_lines = [
        f"{row['path']}\0{row['sha256']}\0{row['bytes']}\0{row['records']}"
        for row in chunks
    ]
    return {
        "run_dir": display(run_dir),
        "config": display(config_path),
        "config_sha256": sha256(config_path),
        "total_combinations": total,
        "expected_chunks": len(jobs),
        "validated_chunks": len(chunks),
        "validated_records": combo_total,
        "raw_bytes": sum(int(row["bytes"]) for row in chunks),
        "chunk_digest_aggregate_sha256": hashlib.sha256(
            "\n".join(digest_lines).encode("utf-8")
        ).hexdigest(),
        "max_absolute_medae_delta": max(row["max_absolute_medae_delta"] for row in chunks),
        "max_absolute_medape_delta": max(row["max_absolute_medape_delta"] for row in chunks),
        "prediction_min": min(row["prediction_min"] for row in chunks),
        "prediction_max": max(row["prediction_max"] for row in chunks),
        "expected_top": expected_top,
        "expected_top_count": len(expected_top),
        "checks": [
            "exact expected chunk inventory with no temporary or extra chunk files",
            "gzip and JSON decode",
            "config and deterministic residue/chunk combo indices",
            "lexicographically unranked probe identity and order",
            "exact packed target row/column order and matrix truth values",
            "finite predictions bounded to [0,100]",
            "revealed probe predictions equal truth exactly",
            "recomputed MedAE, MedAPE, objective, and target n",
            "unique complete combo coverage follows from exact deterministic chunk indices",
        ],
        "chunks": chunks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--scores", type=Path, default=ROOT / "data/scores.csv")
    parser.add_argument(
        "--library", type=Path, default=Path("/tmp/libpathopress_fast_rank1.so")
    )
    parser.add_argument(
        "--equivalence",
        type=Path,
        default=ROOT / "experiments/probe_exhaustive_fast_equivalence.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/probe_exhaustive_integrity_manifest.json",
    )
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 20:
        raise ValueError("--workers must be in [1,20]")
    for path, label in (
        (args.scores, "scores input"),
        (args.library, "fast library"),
        (args.equivalence, "equivalence evidence"),
        (runner.FAST_SOURCE, "fast source"),
        (Path(runner.__file__), "runner source"),
        (Path(__file__), "validator source"),
    ):
        _require_regular(path, label)
    equivalence = load_json(args.equivalence)
    runner._validate_fast_equivalence(args.scores, args.library, args.equivalence)
    comparisons = equivalence.get("comparisons", [])
    tolerances = equivalence.get("tolerances", {})
    observed_equivalence = equivalence.get("observed", {})
    recomputed_cell_max = max(
        (float(row.get("max_absolute_cell_delta", float("inf"))) for row in comparisons),
        default=float("inf"),
    )
    recomputed_metric_max = max(
        (
            max(
                float(row.get("absolute_medae_delta", float("inf"))),
                float(row.get("absolute_medape_delta", float("inf"))),
            )
            for row in comparisons
        ),
        default=float("inf"),
    )
    unique_combo_indices = {row.get("combo_index") for row in comparisons}
    unique_probe_sets = {
        tuple(row.get("probe_set", [])) for row in comparisons
    }
    if (
        float(tolerances.get("max_absolute_cell_delta", float("inf")))
        > CELL_EQUIVALENCE_CAP
        or float(tolerances.get("max_absolute_metric_delta", float("inf")))
        > METRIC_EQUIVALENCE_CAP
        or len(comparisons) < MIN_EQUIVALENCE_COMPARISONS
        or int(observed_equivalence.get("sample_combinations", -1))
        != len(comparisons)
        or len(unique_combo_indices) != len(comparisons)
        or len(unique_probe_sets) != len(comparisons)
        or recomputed_cell_max
        != float(observed_equivalence.get("max_absolute_cell_delta", float("inf")))
        or recomputed_metric_max
        != float(observed_equivalence.get("max_absolute_metric_delta", float("inf")))
        or any(
            not _finite_number(row.get("max_absolute_cell_delta"))
            or float(row["max_absolute_cell_delta"]) > CELL_EQUIVALENCE_CAP
            or not _finite_number(row.get("absolute_medae_delta"))
            or float(row["absolute_medae_delta"]) > METRIC_EQUIVALENCE_CAP
            or not _finite_number(row.get("absolute_medape_delta"))
            or float(row["absolute_medape_delta"]) > METRIC_EQUIVALENCE_CAP
            for row in comparisons
        )
    ):
        raise RuntimeError("equivalence evidence exceeds hard-coded release caps")
    library_hash = sha256(args.library)
    source_hash = sha256(runner.FAST_SOURCE)
    if equivalence["inputs"]["library_sha256"] != library_hash:
        raise RuntimeError("library/equivalence hash mismatch")
    if equivalence["inputs"]["source_sha256"] != source_hash:
        raise RuntimeError("source/equivalence hash mismatch")
    mode = stat.S_IMODE(args.library.stat().st_mode)
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError("fast library must be frozen read-only before validation")
    matrix, model_ids, evaluation_ids = runner._load_matrix(args.scores)
    runs = [
        validate_run(
            run_dir.resolve(),
            matrix,
            model_ids,
            evaluation_ids,
            args.scores.resolve(),
            args.workers,
        )
        for run_dir in args.run_dirs
    ]
    if len({load_json(ROOT / run["config"])["scores_sha256"] for run in runs}) != 1:
        raise RuntimeError("validated runs do not share one score matrix")
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is unavailable for compiler provenance")
    compiler_path = str(Path(compiler).resolve())
    _require_regular(Path(compiler_path), "compiler binary")
    compiler_version = subprocess.check_output(
        [compiler_path, "--version"], text=True
    ).splitlines()[0]
    payload = {
        "schema_version": 1,
        "status": "passed",
        "validation_scope": "every raw chunk and every raw combination record",
        "metric_tolerance": METRIC_TOLERANCE,
        "inputs": {
            "scores_path": display(args.scores),
            "scores_sha256": sha256(args.scores),
            "runner_path": display(Path(runner.__file__)),
            "runner_sha256": sha256(Path(runner.__file__)),
            "execution_function_sha256": {
                function.__name__: hashlib.sha256(
                    inspect.getsource(function).encode("utf-8")
                ).hexdigest()
                for function in (
                    runner._build_config,
                    runner._config_for_chunk,
                    runner._config_at_location,
                    runner._expected_combo_indices,
                    runner.iter_assigned_combos,
                    runner._init_worker,
                    runner._predict_all_known_fast,
                    runner._evaluate_combo,
                    runner._pack_predictions,
                )
            },
            "validator_path": display(Path(__file__)),
            "validator_sha256": sha256(Path(__file__)),
            "fast_source_path": display(runner.FAST_SOURCE),
            "fast_source_sha256": source_hash,
            "fast_library_path": str(args.library.resolve()),
            "fast_library_sha256": library_hash,
            "fast_library_mode_octal": oct(mode),
            "equivalence_path": display(args.equivalence),
            "equivalence_sha256": sha256(args.equivalence),
            "expected_compile_command_not_recovered": COMPILE_COMMAND,
            "current_host_compiler_path": compiler_path,
            "current_host_compiler_sha256": sha256(Path(compiler_path)),
            "current_host_compiler_version": compiler_version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_abi": sys.implementation.cache_tag,
            "libc": list(platform.libc_ver()),
            "equivalence_hard_caps": {
                "max_absolute_cell_delta": CELL_EQUIVALENCE_CAP,
                "max_absolute_metric_delta": METRIC_EQUIVALENCE_CAP,
                "minimum_comparisons": MIN_EQUIVALENCE_COMPARISONS,
            },
            "equivalence_comparisons_validated": len(comparisons),
        },
        "backend_attribution_limitation": (
            "Legacy chunk configs do not contain backend, library, or equivalence "
            "hashes; this posthoc audit cannot prove which binary generated them. "
            "The separately preserved shared-object bytes are hash-bound to the "
            "hard-capped numerical equivalence suite, but that establishes numerical "
            "compatibility rather than generator attribution. The compile command is "
            "the expected command and compiler/platform fields describe the current "
            "validation host; neither is recovered build provenance for the legacy "
            "binary. Execution-critical Python function hashes document the unchanged "
            "generation path without retroactively changing legacy-v1 chunk configs."
        ),
        "runs": runs,
    }
    runner._write_json_atomic(args.output, payload, indent=2)
    print(json.dumps({
        "status": payload["status"],
        "runs": [
            {
                key: run[key]
                for key in (
                    "run_dir", "validated_chunks", "validated_records", "raw_bytes",
                    "max_absolute_medae_delta", "max_absolute_medape_delta",
                )
            }
            for run in runs
        ],
    }, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
