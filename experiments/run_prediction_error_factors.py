#!/usr/bin/env python3
"""Run BenchPress Section 6 benchmark/model prediction-error hypotheses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Avoid nested BLAS oversubscription inside the process pool.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.error_analysis import (  # noqa: E402
    pairwise_abs_correlation,
    spearman_test,
)
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.model_metadata import load_model_metadata  # noqa: E402
from pathopress.predictability import prediction_error  # noqa: E402
from pathopress.provenance import (  # noqa: E402
    BENCHPRESS_PINNED_COMMIT,
    BENCHPRESS_REPOSITORY,
    validate_benchpress_pin,
)
from pathopress.section6_factors import (  # noqa: E402
    grouped_wilcoxon,
    holdout_half_per_benchmark,
    paired_error_record,
    paired_model_wilcoxon,
    pooled_model_metrics,
    supported_complete,
    wilcoxon_signed_rank,
)
from pathopress.temporal import load_release_metadata  # noqa: E402


BENCHMARK_HYPOTHESES = ("benchmark_h4", "benchmark_h5", "benchmark_h6", "benchmark_h7")
MODEL_HYPOTHESES = ("model_h5", "model_h6", "model_h7", "model_h8", "model_h9")
ALL_HYPOTHESES = BENCHMARK_HYPOTHESES + MODEL_HYPOTHESES
DROP_RATES = (0.25, 0.50, 0.75)
NEIGHBOR_THRESHOLDS = (0.95, 0.90, 0.85)
N_BENCHMARK_SEEDS = 5
N_MODEL_SEEDS = 10
BASE_SEED = 42


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _matrix_identity(matrix: np.ndarray, models: list[str], evaluations: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(np.nan_to_num(matrix, nan=-999.0).astype("<f8").tobytes())
    digest.update("\n".join(models).encode())
    digest.update("\n".join(evaluations).encode())
    return digest.hexdigest()


def _task_families(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["evaluation_id"]: row["task_family"] for row in csv.DictReader(handle)}


def _raw_pairs(
    indices: np.ndarray,
    fixed_index: int,
    actual: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
    *,
    side: str,
) -> list[list[float | int]]:
    if side == "benchmark":
        return [
            [int(indices[k]), int(fixed_index), float(actual[k]), float(predicted[k])]
            for k in range(len(indices)) if valid[k]
        ]
    return [
        [int(fixed_index), int(indices[k]), float(actual[k]), float(predicted[k])]
        for k in range(len(indices)) if valid[k]
    ]


def _paired_record(
    actual: np.ndarray,
    base: np.ndarray,
    treat: np.ndarray,
    indices: np.ndarray,
    fixed_index: int,
    *,
    side: str,
    min_predictions: int = 2,
) -> dict[str, object] | None:
    metrics = paired_error_record(actual, base, treat, min_predictions=min_predictions)
    if metrics is None:
        return None
    valid = np.isfinite(actual) & np.isfinite(base) & np.isfinite(treat)
    return {
        **metrics,
        "raw_base": _raw_pairs(indices, fixed_index, actual, base, valid, side=side),
        "raw_treat": _raw_pairs(indices, fixed_index, actual, treat, valid, side=side),
    }


def _benchmark_worker(job: dict[str, object]) -> dict[str, object]:
    matrix = np.asarray(job["matrix"], dtype=float)
    hypothesis = str(job["hypothesis"])
    j = int(job["target_index"])
    test = np.asarray(job["test_indices"], dtype=int)
    actual = matrix[test, j]
    records: list[dict[str, object]] = []

    if hypothesis == "benchmark_h4":
        train = np.asarray(job["train_indices"], dtype=int)
        keep_by_drop = dict(job["keep_by_drop"])
        base_matrix = matrix.copy()
        base_matrix[:, j] = np.nan
        base_matrix[train, j] = matrix[train, j]
        base = supported_complete(base_matrix)[test, j]
        for drop_rate in DROP_RATES:
            keep_positions = np.asarray(keep_by_drop[str(drop_rate)], dtype=int)
            kept = train[keep_positions]
            treatment_matrix = matrix.copy()
            treatment_matrix[:, j] = np.nan
            treatment_matrix[kept, j] = matrix[kept, j]
            treatment = supported_complete(treatment_matrix)[test, j]
            paired = _paired_record(actual, base, treatment, test, j, side="benchmark")
            if paired is not None:
                records.append({
                    **paired,
                    "setting": float(drop_rate),
                    "n_observed_original": int(np.isfinite(matrix[:, j]).sum()),
                    "n_train_kept": int(len(kept)),
                })
    else:
        base_matrix = matrix.copy()
        base_matrix[test, j] = np.nan
        base = supported_complete(base_matrix)[test, j]
        if hypothesis == "benchmark_h5":
            correlation = np.asarray(job["correlation"], dtype=float)
            for threshold in NEIGHBOR_THRESHOLDS:
                neighbors = np.flatnonzero(correlation[j] >= threshold)
                neighbors = neighbors[neighbors != j]
                treatment_matrix = base_matrix.copy()
                treatment_matrix[:, neighbors] = np.nan
                treatment = supported_complete(treatment_matrix)[test, j]
                paired = _paired_record(actual, base, treatment, test, j, side="benchmark")
                if paired is not None:
                    records.append({
                        **paired,
                        "setting": float(threshold),
                        "n_neighbors_removed": int(len(neighbors)),
                        "removed_indices": neighbors.astype(int).tolist(),
                    })
        elif hypothesis == "benchmark_h6":
            neighbor = int(job["neighbor_index"])
            overlap = np.asarray(job["overlap_permutation"], dtype=int)
            for drop_rate in DROP_RATES:
                n_drop = int(round(len(overlap) * drop_rate))
                dropped = overlap[:n_drop]
                treatment_matrix = base_matrix.copy()
                treatment_matrix[dropped, neighbor] = np.nan
                treatment = supported_complete(treatment_matrix)[test, j]
                paired = _paired_record(actual, base, treatment, test, j, side="benchmark")
                if paired is not None:
                    records.append({
                        **paired,
                        "setting": float(drop_rate),
                        "neighbor_index": neighbor,
                        "neighbor_abs_r": float(job["neighbor_abs_r"]),
                        "n_overlap_original": int(len(overlap)),
                        "removed_indices": dropped.astype(int).tolist(),
                    })
        elif hypothesis == "benchmark_h7":
            same_family = np.asarray(job["same_family_indices"], dtype=int)
            treatment_matrix = base_matrix.copy()
            treatment_matrix[:, same_family] = np.nan
            treatment = supported_complete(treatment_matrix)[test, j]
            paired = _paired_record(
                actual, base, treatment, test, j, side="benchmark", min_predictions=3
            )
            if paired is not None:
                records.append({
                    **paired,
                    "setting": "same_task_family",
                    "task_family": str(job["task_family"]),
                    "n_neighbors_removed": int(len(same_family)),
                    "removed_indices": same_family.astype(int).tolist(),
                })
        else:
            raise ValueError(hypothesis)
    return {"unit_id": job["unit_id"], "records": records}


def _model_worker(job: dict[str, object]) -> dict[str, object]:
    matrix = np.asarray(job["matrix"], dtype=float)
    hypothesis = str(job["hypothesis"])
    i = int(job["target_index"])
    test = np.asarray(job["test_indices"], dtype=int)
    actual = matrix[i, test]
    base_matrix = matrix.copy()
    base_matrix[i, test] = np.nan
    base = supported_complete(base_matrix)[i, test]
    records = []

    if hypothesis == "model_h5":
        peers = np.asarray(job["peer_indices"], dtype=int)
        treatment_matrix = base_matrix.copy()
        treatment_matrix[peers, :] = np.nan
        treatment = supported_complete(treatment_matrix)[i, test]
        paired = _paired_record(actual, base, treatment, test, i, side="model")
        if paired is not None:
            records.append({
                **paired, "setting": 0.95,
                "n_peers_removed": int(len(peers)),
                "removed_indices": peers.astype(int).tolist(),
            })
    elif hypothesis == "model_h6":
        peer = int(job["peer_index"])
        overlap = np.asarray(job["overlap_permutation"], dtype=int)
        for drop_rate in DROP_RATES:
            n_drop = max(1, int(round(len(overlap) * drop_rate)))
            dropped = overlap[:n_drop]
            treatment_matrix = base_matrix.copy()
            treatment_matrix[peer, dropped] = np.nan
            treatment = supported_complete(treatment_matrix)[i, test]
            paired = _paired_record(actual, base, treatment, test, i, side="model")
            if paired is not None:
                records.append({
                    **paired, "setting": float(drop_rate),
                    "peer_index": peer,
                    "peer_abs_r": float(job["peer_abs_r"]),
                    "n_overlap_original": int(len(overlap)),
                    "removed_indices": dropped.astype(int).tolist(),
                })
    elif hypothesis == "model_h7":
        peers = np.asarray(job["peer_indices"], dtype=int)
        treatment_matrix = base_matrix.copy()
        treatment_matrix[peers, :] = np.nan
        treatment = supported_complete(treatment_matrix)[i, test]
        paired = _paired_record(actual, base, treatment, test, i, side="model")
        if paired is not None:
            records.append({
                **paired, "setting": "same_provider",
                "provider": str(job["provider"]),
                "n_peers_removed": int(len(peers)),
                "removed_indices": peers.astype(int).tolist(),
            })
    else:
        raise ValueError(hypothesis)
    return {"unit_id": job["unit_id"], "records": records}


def _model_h8_worker(job: dict[str, object]) -> dict[str, object]:
    matrix = np.asarray(job["matrix"], dtype=float)
    raw = []
    for condition, fraction in (("hide_25pct", 0.25), ("baseline", 0.50), ("hide_75pct", 0.75)):
        train = matrix.copy()
        test_by_model = dict(job["test_by_condition"])[condition]
        for i_string, indices in test_by_model.items():
            train[int(i_string), np.asarray(indices, dtype=int)] = np.nan
        predicted = supported_complete(train)
        for i_string, indices in test_by_model.items():
            i = int(i_string)
            for j in indices:
                actual = float(matrix[i, int(j)])
                estimate = float(predicted[i, int(j)])
                if np.isfinite(actual) and np.isfinite(estimate):
                    raw.append({
                        "condition": condition,
                        "model_index": i,
                        "evaluation_index": int(j),
                        "actual": actual,
                        "predicted": estimate,
                        "fraction_hidden": fraction,
                    })
    return {"unit_id": job["unit_id"], "raw": raw}


def _model_h9_worker(job: dict[str, object]) -> dict[str, object]:
    matrix = np.asarray(job["matrix"], dtype=float)
    train = np.full_like(matrix, np.nan)
    train_indices = np.asarray(job["train_indices"], dtype=int)
    train[train_indices] = matrix[train_indices]
    target_reveals = dict(job["target_reveals"])
    hidden_by_target = {}
    for i_string, reveal_hidden in target_reveals.items():
        i = int(i_string)
        reveal = np.asarray(reveal_hidden["revealed"], dtype=int)
        hidden = np.asarray(reveal_hidden["hidden"], dtype=int)
        train[i, reveal] = matrix[i, reveal]
        hidden_by_target[i] = hidden
    predicted = supported_complete(train)
    raw = []
    actual_values = []
    predicted_values = []
    for i, hidden in hidden_by_target.items():
        for j in hidden:
            actual = float(matrix[i, j])
            estimate = float(predicted[i, j])
            if np.isfinite(actual) and np.isfinite(estimate):
                actual_values.append(actual)
                predicted_values.append(estimate)
                raw.append({
                    "condition": str(job["condition"]),
                    "model_index": i,
                    "evaluation_index": int(j),
                    "actual": actual,
                    "predicted": estimate,
                })
    metrics = prediction_error(actual_values, predicted_values)
    return {"unit_id": job["unit_id"], "raw": raw, "metrics": metrics}


def _benchmark_units(
    hypothesis: str,
    matrix: np.ndarray,
    evaluations: list[str],
    task_family: dict[str, str],
) -> list[dict[str, object]]:
    units = []
    if hypothesis == "benchmark_h4":
        rng = np.random.RandomState(BASE_SEED)
        for j in range(matrix.shape[1]):
            if int(np.isfinite(matrix[:, j]).sum()) < 6:
                continue
            for seed in range(N_BENCHMARK_SEEDS):
                test, train = holdout_half_per_benchmark(matrix, j, rng, min_test=3)
                keep_by_drop = {}
                for drop_rate in (0.0,) + DROP_RATES:
                    n_keep = max(1, int(len(train) * (1.0 - drop_rate)))
                    keep_by_drop[str(drop_rate)] = rng.permutation(len(train))[:n_keep].astype(int).tolist()
                units.append({
                    "hypothesis": hypothesis, "target_index": j, "seed": seed,
                    "test_indices": test.astype(int).tolist(),
                    "train_indices": train.astype(int).tolist(),
                    "keep_by_drop": keep_by_drop,
                })
    elif hypothesis == "benchmark_h5":
        correlation, _ = pairwise_abs_correlation(matrix, axis=0, min_shared=5)
        rng = np.random.RandomState(BASE_SEED + 6000)
        for j in range(matrix.shape[1]):
            if int(np.isfinite(matrix[:, j]).sum()) < 6:
                continue
            for seed in range(N_BENCHMARK_SEEDS):
                test, _ = holdout_half_per_benchmark(matrix, j, rng, min_test=3)
                units.append({
                    "hypothesis": hypothesis, "target_index": j, "seed": seed,
                    "test_indices": test.astype(int).tolist(), "correlation": correlation,
                })
    elif hypothesis == "benchmark_h6":
        correlation, _ = pairwise_abs_correlation(matrix, axis=0, min_shared=5)
        for j in range(matrix.shape[1]):
            if int(np.isfinite(matrix[:, j]).sum()) < 6:
                continue
            row = correlation[j].copy(); row[j] = np.nan
            if not np.any(np.isfinite(row)):
                continue
            neighbor = int(np.nanargmax(row))
            overlap = np.flatnonzero(np.isfinite(matrix[:, j]) & np.isfinite(matrix[:, neighbor]))
            if len(overlap) < 4:
                continue
            for seed in range(N_BENCHMARK_SEEDS):
                split_rng = np.random.RandomState(BASE_SEED + 10_000 + 1_000 * seed + j)
                mask_rng = np.random.RandomState(BASE_SEED + 20_000 + 1_000 * seed + j)
                test, _ = holdout_half_per_benchmark(matrix, j, split_rng, min_test=3)
                overlap_permutation = overlap[mask_rng.permutation(len(overlap))]
                units.append({
                    "hypothesis": hypothesis, "target_index": j, "seed": seed,
                    "test_indices": test.astype(int).tolist(), "neighbor_index": neighbor,
                    "neighbor_abs_r": float(row[neighbor]),
                    "overlap_permutation": overlap_permutation.astype(int).tolist(),
                })
    elif hypothesis == "benchmark_h7":
        families = [task_family.get(evaluation, "") for evaluation in evaluations]
        rng = np.random.RandomState(BASE_SEED)
        for j in range(matrix.shape[1]):
            if int(np.isfinite(matrix[:, j]).sum()) < 6:
                continue
            same = [k for k, family in enumerate(families) if k != j and family == families[j]]
            if not same:
                continue
            for seed in range(N_MODEL_SEEDS):
                test, _ = holdout_half_per_benchmark(matrix, j, rng, min_test=0)
                units.append({
                    "hypothesis": hypothesis, "target_index": j, "seed": seed,
                    "test_indices": test.astype(int).tolist(),
                    "same_family_indices": same, "task_family": families[j],
                })
    else:
        raise ValueError(hypothesis)
    for unit in units:
        j = int(unit["target_index"])
        unit["target_id"] = evaluations[j]
        unit["unit_id"] = f"{hypothesis}:{evaluations[j]}:{unit['seed']}"
        unit["matrix"] = matrix
    return units


def _standard_model_tests(matrix: np.ndarray) -> dict[int, dict[int, list[int]]]:
    result = {}
    for seed in range(N_MODEL_SEEDS):
        rng = np.random.RandomState(seed * 2000)
        per_model = {}
        for i in range(matrix.shape[0]):
            observed = np.flatnonzero(np.isfinite(matrix[i]))
            if len(observed) < 4:
                continue
            rng.shuffle(observed)
            per_model[i] = observed[: len(observed) // 2].astype(int).tolist()
        result[seed] = per_model
    return result


def _model_units(
    hypothesis: str,
    matrix: np.ndarray,
    models: list[str],
    metadata: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    correlation, _ = pairwise_abs_correlation(matrix, axis=1, min_shared=3)
    standard = _standard_model_tests(matrix)
    units = []
    if hypothesis in {"model_h5", "model_h7"}:
        providers = [metadata[model]["provider"] for model in models]
        for seed, tests in standard.items():
            for i, test in tests.items():
                if hypothesis == "model_h5":
                    peers = np.flatnonzero(correlation[i] >= 0.95)
                    peers = peers[peers != i].astype(int).tolist()
                    extra = {"peer_indices": peers}
                else:
                    provider = providers[i]
                    if not provider:
                        continue
                    peers = [k for k, value in enumerate(providers) if k != i and provider and value == provider]
                    extra = {"peer_indices": peers, "provider": provider}
                units.append({
                    "hypothesis": hypothesis, "target_index": i, "seed": seed,
                    "test_indices": test, **extra,
                })
    elif hypothesis == "model_h6":
        for i in range(matrix.shape[0]):
            observed = np.flatnonzero(np.isfinite(matrix[i]))
            if len(observed) < 4:
                continue
            row = correlation[i].copy(); row[i] = np.nan
            eligible = np.flatnonzero(row >= 0.95)
            if not len(eligible):
                continue
            peer = int(eligible[np.nanargmax(row[eligible])])
            overlap = np.flatnonzero(np.isfinite(matrix[i]) & np.isfinite(matrix[peer]))
            if not len(overlap):
                continue
            for seed in range(N_MODEL_SEEDS):
                split_rng = np.random.RandomState(BASE_SEED + 10_000 + seed * 1_000 + i)
                mask_rng = np.random.RandomState(BASE_SEED + 20_000 + seed * 1_000 + i)
                shuffled = observed.copy(); split_rng.shuffle(shuffled)
                test = shuffled[: len(shuffled) // 2]
                overlap_permutation = overlap[mask_rng.permutation(len(overlap))]
                units.append({
                    "hypothesis": hypothesis, "target_index": i, "seed": seed,
                    "test_indices": test.astype(int).tolist(), "peer_index": peer,
                    "peer_abs_r": float(row[peer]),
                    "overlap_permutation": overlap_permutation.astype(int).tolist(),
                })
    elif hypothesis == "model_h8":
        for seed in range(N_MODEL_SEEDS):
            by_condition = {}
            for condition, fraction in (("hide_25pct", 0.25), ("baseline", 0.50), ("hide_75pct", 0.75)):
                rng = np.random.RandomState(seed * 2000)
                per_model = {}
                for i in range(matrix.shape[0]):
                    observed = np.flatnonzero(np.isfinite(matrix[i]))
                    if len(observed) < 4:
                        continue
                    rng.shuffle(observed)
                    count = max(1, int(len(observed) * fraction))
                    per_model[str(i)] = observed[:count].astype(int).tolist()
                by_condition[condition] = per_model
            units.append({
                "hypothesis": hypothesis, "seed": seed,
                "test_by_condition": by_condition,
                "unit_id": f"{hypothesis}:seed:{seed}", "matrix": matrix,
            })
    else:
        raise ValueError(hypothesis)
    if hypothesis != "model_h8":
        for unit in units:
            i = int(unit["target_index"])
            unit["target_id"] = models[i]
            unit["unit_id"] = f"{hypothesis}:{models[i]}:{unit['seed']}"
            unit["matrix"] = matrix
    return units


def _model_h9_units(
    matrix: np.ndarray,
    models: list[str],
    releases_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    releases = load_release_metadata(releases_path)
    dated = sorted(
        [(model, releases[model].release_date) for model in models
         if releases[model].release_date is not None and releases[model].verification_status == "verified"],
        key=lambda pair: (pair[1], pair[0]),
    )
    cut1, cut2 = len(dated) // 3, 2 * len(dated) // 3
    groups = {
        "A_oldest": [model for model, _ in dated[:cut1]],
        "B_middle": [model for model, _ in dated[cut1:cut2]],
        "C_newest": [model for model, _ in dated[cut2:]],
    }
    index = {model: i for i, model in enumerate(models)}
    offsets = {
        name: int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 10000
        for name in ("A_oldest", "B_middle")
    }
    units = []
    for condition in ("A_oldest", "B_middle"):
        for seed in range(N_MODEL_SEEDS):
            rng = np.random.RandomState(seed * 500 + offsets[condition])
            for k in (1, 3, 5, 8, 10, 15):
                target_reveals = {}
                for model in groups["C_newest"]:
                    i = index[model]
                    observed = np.flatnonzero(np.isfinite(matrix[i]))
                    if len(observed) < k + 2:
                        continue
                    shuffled = observed.copy(); rng.shuffle(shuffled)
                    target_reveals[str(i)] = {
                        "revealed": shuffled[:k].astype(int).tolist(),
                        "hidden": shuffled[k:].astype(int).tolist(),
                    }
                units.append({
                    "hypothesis": "model_h9", "condition": condition,
                    "seed": seed, "k": k,
                    "train_indices": [index[model] for model in groups[condition]],
                    "target_reveals": target_reveals,
                    "unit_id": f"model_h9:{condition}:k{k}:seed{seed}",
                    "matrix": matrix,
                })
    metadata = {
        "design": "A_vs_B_release_date_thirds",
        "groups": groups,
        "date_ranges": {
            "A_oldest": [dated[0][1].isoformat(), dated[cut1 - 1][1].isoformat()],
            "B_middle": [dated[cut1][1].isoformat(), dated[cut2 - 1][1].isoformat()],
            "C_newest": [dated[cut2][1].isoformat(), dated[-1][1].isoformat()],
        },
        "n_dated_verified": len(dated),
    }
    return units, metadata


def _cache_path(directory: Path, hypothesis: str, unit_id: str) -> Path:
    short = hashlib.sha256(unit_id.encode()).hexdigest()[:16]
    return directory / hypothesis / f"unit_{short}.json"


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_cached(
    hypothesis: str,
    units: list[dict[str, object]],
    worker,
    *,
    cache_dir: Path,
    matrix_sha: str,
    workers: int,
    num_shards: int,
    shard_id: int,
    merge_only: bool,
    limit_units: int | None,
) -> list[dict[str, object]]:
    expected_units = units[:limit_units] if limit_units is not None else units
    selected = [unit for position, unit in enumerate(expected_units) if position % num_shards == shard_id]
    pending = []
    for unit in selected:
        path = _cache_path(cache_dir, hypothesis, str(unit["unit_id"]))
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("matrix_sha256") != matrix_sha or cached.get("unit_id") != unit["unit_id"]:
                raise ValueError(f"stale cache shard: {path}")
        else:
            pending.append(unit)
    print(f"[{hypothesis}] units={len(selected)} pending={len(pending)} workers={workers}", flush=True)
    if pending and merge_only:
        raise FileNotFoundError(f"{hypothesis}: {len(pending)} selected cache units are missing")
    started = time.time()
    if workers == 1:
        iterator = ((unit, worker(unit)) for unit in pending)
        for done, (unit, result) in enumerate(iterator, start=1):
            _write_atomic(_cache_path(cache_dir, hypothesis, str(unit["unit_id"])), {
                **result, "matrix_sha256": matrix_sha,
                "hypothesis": hypothesis, "seed": int(unit["seed"]),
                "target_id": unit.get("target_id"),
            })
            print(f"  {hypothesis} {done}/{len(pending)} ({time.time()-started:.0f}s)", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker, unit): unit for unit in pending}
            for done, future in enumerate(as_completed(futures), start=1):
                unit = futures[future]
                result = future.result()
                _write_atomic(_cache_path(cache_dir, hypothesis, str(unit["unit_id"])), {
                    **result, "matrix_sha256": matrix_sha,
                    "hypothesis": hypothesis, "seed": int(unit["seed"]),
                    "target_id": unit.get("target_id"),
                })
                if done % 10 == 0 or done == len(pending):
                    print(f"  {hypothesis} {done}/{len(pending)} ({time.time()-started:.0f}s)", flush=True)
    outputs = []
    for unit in expected_units:
        path = _cache_path(cache_dir, hypothesis, str(unit["unit_id"]))
        if path.exists():
            outputs.append(json.loads(path.read_text(encoding="utf-8")))
    if num_shards == 1 and len(outputs) != len(expected_units):
        raise RuntimeError(f"{hypothesis}: expected {len(expected_units)} outputs, found {len(outputs)}")
    return outputs


def _correlational_payload(
    error_path: Path,
    metadata: dict[str, dict[str, str]],
) -> dict[str, object]:
    source = json.loads(error_path.read_text(encoding="utf-8"))
    evaluation_rows = source["evaluation_analysis"]["rows"]
    model_rows = []
    for row in source["model_analysis"]["rows"]:
        meta = metadata[row["model_id"]]
        count = float(meta["parameter_count"]) if meta["parameter_count"] else float("nan")
        model_rows.append({
            **row,
            "provider": meta["provider"],
            "family": meta["family"],
            "parameter_count": None if not np.isfinite(count) else int(count),
            "log10_parameter_count": float(np.log10(count)) if np.isfinite(count) else None,
            "model_type": meta["model_type"],
            "is_slide_model": int(meta["model_type"] == "slide_encoder"),
            "modality": meta["modality"],
            "primary_source_url": meta["primary_source_url"],
        })

    def tests(rows, features):
        return {
            metric: {
                feature: spearman_test(
                    np.asarray([float(row[feature]) if row[feature] is not None else np.nan for row in rows]),
                    np.asarray([float(row[metric]) for row in rows]),
                )
                for feature in features
            }
            for metric in ("medape", "medae")
        }

    benchmark_features = ("rank2_r2", "rank1_r2", "median_score", "score_std")
    model_features = (
        "log10_parameter_count", "is_slide_model", "median_score", "rank2_r2", "rank1_r2"
    )
    return {
        "benchmark": {
            "rows": evaluation_rows,
            "tests": tests(evaluation_rows, benchmark_features),
            "hypotheses": {
                "H1": "rank2_r2", "H1_pathology_rank": "rank1_r2",
                "H2": "median_score", "H3": "score_std",
            },
        },
        "model": {
            "rows": model_rows,
            "tests": tests(model_rows, model_features),
            "hypotheses": {
                "H1": "log10_parameter_count", "H2": "is_slide_model",
                "H3": "median_score", "H4": "rank2_r2",
                "H4_pathology_rank": "rank1_r2",
            },
        },
    }


def _flatten_records(
    hypothesis: str,
    outputs: list[dict[str, object]],
    *,
    side: str,
    models: list[str],
    evaluations: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records = []
    raw = []
    for output in outputs:
        target_id = str(output.get("target_id") or "")
        seed = int(output["seed"])
        for record in output.get("records", []):
            clean = {key: value for key, value in record.items() if not key.startswith("raw_")}
            clean.update({"hypothesis": hypothesis, "target_id": target_id, "seed": seed})
            records.append(clean)
            setting = str(record["setting"])
            for condition, key in (("baseline", "raw_base"), ("treatment", "raw_treat")):
                for i, j, actual, predicted in record[key]:
                    raw.append({
                        "side": side, "hypothesis": hypothesis, "condition": condition,
                        "setting": setting, "seed": seed,
                        "model_id": models[int(i)], "evaluation_id": evaluations[int(j)],
                        "actual": float(actual), "predicted": float(predicted),
                    })
    return records, raw


def _summary_for_records(hypothesis: str, records: list[dict[str, object]], side: str):
    if hypothesis == "benchmark_h5":
        by_setting = {
            str(threshold): grouped_wilcoxon(
                [row for row in records if float(row["setting"]) == threshold],
                group_key="target_id", drop_zeros_for_test=True,
            )
            for threshold in NEIGHBOR_THRESHOLDS
        }
        return {"by_setting": by_setting, "headline_setting": "0.85", "tests": by_setting["0.85"]}
    if hypothesis in {"benchmark_h4", "benchmark_h6", "model_h6"}:
        by_setting = {
            str(rate): grouped_wilcoxon(
                [row for row in records if float(row["setting"]) == rate],
                group_key="target_id",
            )
            for rate in DROP_RATES
        }
        return {"by_setting": by_setting, "headline_setting": "0.75", "tests": by_setting["0.75"]}
    return {"tests": grouped_wilcoxon(records, group_key="target_id", drop_zeros_for_test=True)}


def _h8_summary(raw: list[dict[str, object]]) -> dict[str, object]:
    baseline = pooled_model_metrics(raw, condition="baseline")
    by_condition = {}
    for condition in ("hide_25pct", "hide_75pct"):
        treatment = pooled_model_metrics(raw, condition=condition)
        by_condition[condition] = {
            "tests": paired_model_wilcoxon(baseline, treatment),
            "n_baseline_models": len(baseline),
            "n_treatment_models": len(treatment),
        }
    return {"by_condition": by_condition}


def _h9_summary(outputs: list[dict[str, object]]) -> dict[str, object]:
    units = []
    for output in outputs:
        _, condition, k_part, seed_part = str(output["unit_id"]).split(":")
        units.append({
            "condition": condition, "k": int(k_part[1:]), "seed": int(seed_part[4:]),
            **output["metrics"],
        })
    by_k = {}
    for k in (1, 3, 5, 8, 10, 15):
        by_k[str(k)] = {}
        for metric in ("medape", "medae"):
            a = sorted((row for row in units if row["condition"] == "A_oldest" and row["k"] == k), key=lambda row: row["seed"])
            b = sorted((row for row in units if row["condition"] == "B_middle" and row["k"] == k), key=lambda row: row["seed"])
            a_values = [float(row[metric]) for row in a]
            b_values = [float(row[metric]) for row in b]
            delta = np.asarray(a_values) - np.asarray(b_values)
            test = wilcoxon_signed_rank(delta, min_n=3)
            by_k[str(k)][metric] = {
                **test,
                "median_A": float(np.median(a_values)),
                "median_B": float(np.median(b_values)),
                "values_A": a_values, "values_B": b_values,
            }
    return {"units": units, "comparison_A_vs_B": by_k}


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=ROOT / "data" / "scores.csv")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data" / "tasks.csv")
    parser.add_argument("--model-metadata", type=Path, default=ROOT / "data" / "model_metadata.csv")
    parser.add_argument("--release-dates", type=Path, default=ROOT / "data" / "model_release_dates.csv")
    parser.add_argument("--error-analysis", type=Path, default=ROOT / "experiments" / "error_analysis_rank1.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "experiments" / "prediction_error_factor_shards")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "prediction_error_factors_rank1.json")
    parser.add_argument("--records-csv", type=Path, default=ROOT / "outputs" / "prediction_error_factor_records_rank1.csv")
    parser.add_argument("--raw-csv", type=Path, default=ROOT / "outputs" / "prediction_error_factor_predictions_rank1.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments" / "prediction_error_factor_manifest.json")
    parser.add_argument("--hypotheses", default=",".join(ALL_HYPOTHESES))
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--limit-units", type=int, default=None)
    args = parser.parse_args()
    validate_benchpress_pin()
    selected = tuple(part.strip() for part in args.hypotheses.split(",") if part.strip())
    unknown = sorted(set(selected) - set(ALL_HYPOTHESES))
    if unknown:
        raise ValueError(f"unknown hypotheses: {unknown}")
    if args.workers < 1 or args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid workers or shard configuration")

    matrix, models, evaluations = make_matrix(load_scores(args.scores))
    matrix, models, evaluations = filter_matrix(matrix, models, evaluations)
    matrix_sha = _matrix_identity(matrix, models, evaluations)
    metadata = load_model_metadata(args.model_metadata)
    if set(models) - set(metadata):
        raise ValueError(f"model metadata missing: {sorted(set(models)-set(metadata))}")
    families = _task_families(args.tasks)
    correlational = _correlational_payload(args.error_analysis, metadata)
    results = {}
    all_records: list[dict[str, object]] = []
    all_raw: list[dict[str, object]] = []

    for hypothesis in selected:
        if hypothesis in BENCHMARK_HYPOTHESES:
            units = _benchmark_units(hypothesis, matrix, evaluations, families)
            outputs = _run_cached(
                hypothesis, units, _benchmark_worker,
                cache_dir=args.cache_dir, matrix_sha=matrix_sha, workers=args.workers,
                num_shards=args.num_shards, shard_id=args.shard_id,
                merge_only=args.merge_only, limit_units=args.limit_units,
            )
            records, raw = _flatten_records(
                hypothesis, outputs, side="benchmark", models=models, evaluations=evaluations
            )
            results[hypothesis] = {
                "side": "benchmark", "n_units": len(outputs), "n_records": len(records),
                **_summary_for_records(hypothesis, records, "benchmark"),
            }
            all_records.extend(records); all_raw.extend(raw)
        elif hypothesis in {"model_h5", "model_h6", "model_h7"}:
            units = _model_units(hypothesis, matrix, models, metadata)
            outputs = _run_cached(
                hypothesis, units, _model_worker,
                cache_dir=args.cache_dir, matrix_sha=matrix_sha, workers=args.workers,
                num_shards=args.num_shards, shard_id=args.shard_id,
                merge_only=args.merge_only, limit_units=args.limit_units,
            )
            records, raw = _flatten_records(
                hypothesis, outputs, side="model", models=models, evaluations=evaluations
            )
            if hypothesis in {"model_h5", "model_h7"}:
                baseline = pooled_model_metrics(raw, condition="baseline")
                treatment = pooled_model_metrics(raw, condition="treatment")
                summary = {"tests": paired_model_wilcoxon(baseline, treatment),
                           "n_baseline_models": len(baseline), "n_treatment_models": len(treatment)}
            else:
                summary = _summary_for_records(hypothesis, records, "model")
            results[hypothesis] = {
                "side": "model", "n_units": len(outputs), "n_records": len(records), **summary,
            }
            all_records.extend(records); all_raw.extend(raw)
        elif hypothesis == "model_h8":
            units = _model_units(hypothesis, matrix, models, metadata)
            outputs = _run_cached(
                hypothesis, units, _model_h8_worker,
                cache_dir=args.cache_dir, matrix_sha=matrix_sha, workers=args.workers,
                num_shards=args.num_shards, shard_id=args.shard_id,
                merge_only=args.merge_only, limit_units=args.limit_units,
            )
            raw = []
            for output in outputs:
                seed = int(output["seed"])
                for row in output["raw"]:
                    raw.append({
                        "side": "model", "hypothesis": hypothesis,
                        "condition": row["condition"], "setting": row["fraction_hidden"],
                        "seed": seed, "model_id": models[int(row["model_index"])],
                        "evaluation_id": evaluations[int(row["evaluation_index"])],
                        "actual": row["actual"], "predicted": row["predicted"],
                    })
            results[hypothesis] = {"side": "model", "n_units": len(outputs), **_h8_summary(raw)}
            all_raw.extend(raw)
        elif hypothesis == "model_h9":
            units, temporal_metadata = _model_h9_units(matrix, models, args.release_dates)
            outputs = _run_cached(
                hypothesis, units, _model_h9_worker,
                cache_dir=args.cache_dir, matrix_sha=matrix_sha, workers=args.workers,
                num_shards=args.num_shards, shard_id=args.shard_id,
                merge_only=args.merge_only, limit_units=args.limit_units,
            )
            raw = []
            for output in outputs:
                _, condition, k_part, seed_part = str(output["unit_id"]).split(":")
                for row in output["raw"]:
                    raw.append({
                        "side": "model", "hypothesis": hypothesis,
                        "condition": condition, "setting": int(k_part[1:]),
                        "seed": int(seed_part[4:]),
                        "model_id": models[int(row["model_index"])],
                        "evaluation_id": evaluations[int(row["evaluation_index"])],
                        "actual": row["actual"], "predicted": row["predicted"],
                    })
            results[hypothesis] = {
                "side": "model", "n_units": len(outputs),
                **temporal_metadata, **_h9_summary(outputs),
            }
            all_raw.extend(raw)

    if args.num_shards > 1 and not args.merge_only:
        print(
            f"shard {args.shard_id}/{args.num_shards} complete; "
            "run once with --merge-only after every shard finishes",
            flush=True,
        )
        return

    _write_csv(args.records_csv, all_records)
    _write_csv(
        args.raw_csv, all_raw,
        ["side", "hypothesis", "condition", "setting", "seed", "model_id",
         "evaluation_id", "actual", "predicted"],
    )
    parameter_n = sum(row["parameter_count"] is not None for row in correlational["model"]["rows"])
    model_error_ids = {row["model_id"] for row in correlational["model"]["rows"]}
    payload = {
        "schema_version": 1,
        "description": "BenchPress Section 6 prediction-error factors ported to pathology.",
        "matrix": {
            "shape": list(matrix.shape), "n_observed": int(np.isfinite(matrix).sum()),
            "sha256": matrix_sha,
        },
        "protocol": {
            "predictor": "logit bias ALS rank=1 regularization=0.1",
            "upstream_rank": 2,
            "pathology_rank": 1,
            "benchmark_correlational_source": _display_path(args.error_analysis),
            "paired_metrics": "baseline and treatment are scored only on their common finite predictions",
            "inference": "two-sided Wilcoxon on one median seed-level delta per benchmark/model, except upstream pooled model H5/H7/H8 semantics",
            "resume_cache": _display_path(args.cache_dir),
            "upstream_reference": {
                "repository": BENCHPRESS_REPOSITORY,
                "pinned_commit": BENCHPRESS_PINNED_COMMIT,
                "benchmark_analysis": "experiments/sec6_trust/prediction_error_analysis/benchmark_analysis",
                "model_analysis": "experiments/sec6_trust/prediction_error_analysis/model_analysis",
                "statistics": "benchpress/stats.py",
            },
        },
        "metadata_denominators": {
            "matrix_models": len(models),
            "model_error_rows": len(correlational["model"]["rows"]),
            "parameter_count_available_for_model_error_rows": parameter_n,
            "provider_available": sum(bool(metadata[m]["provider"]) for m in models),
            "family_available": sum(bool(metadata[m]["family"]) for m in models),
            "model_type_available": sum(bool(metadata[m]["model_type"]) for m in models),
            "provider_available_for_model_error_rows": sum(
                bool(metadata[m]["provider"]) for m in model_error_ids
            ),
            "family_available_for_model_error_rows": sum(
                bool(metadata[m]["family"]) for m in model_error_ids
            ),
            "model_type_available_for_model_error_rows": sum(
                bool(metadata[m]["model_type"]) for m in model_error_ids
            ),
        },
        "correlational": correlational,
        "interventions": results,
        "artifacts": {
            "model_metadata": _display_path(args.model_metadata),
            "records_csv": _display_path(args.records_csv),
            "raw_predictions_csv": _display_path(args.raw_csv),
        },
        "pathology_adaptations": [
            "Rank-2 low-rank fit is retained for direct H1/H4 parity; rank-1 fit is reported alongside it because pathology selected rank 1.",
            "Model H2 uses slide-encoder status as the pathology-specific binary model-type analogue of BenchPress reasoning-model status.",
            "Task family is the benchmark-category field for H7.",
            "Model parameter counts are nominal encoder counts only; unavailable or non-comparable slide-system totals remain missing and are excluded with denominators reported.",
            "Model H7 retains upstream same-provider semantics; canonical family is preserved in metadata but is not substituted into the headline intervention.",
            "Model H5 and H7 use one identical target-only hide-half training matrix for each baseline/treatment pair; this repairs the upstream scripts' mismatch between an all-model baseline mask and target-only treatment mask.",
            "Models with unavailable provider metadata are excluded from H7 rather than treated as having zero same-provider peers.",
        ],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    cache_files = list(args.cache_dir.glob("*/*.json"))
    active_cache_units = sum(int(block.get("n_units", 0)) for block in results.values())
    manifest = {
        "schema_version": 1,
        "matrix_sha256": matrix_sha,
        "regeneration_command": "PYTHONPATH=src python3 experiments/run_prediction_error_factors.py --workers 8",
        "resume_command": "PYTHONPATH=src python3 experiments/run_prediction_error_factors.py --workers 8",
        "merge_only_command": "PYTHONPATH=src python3 experiments/run_prediction_error_factors.py --merge-only",
        "artifacts": {
            "merged_summary": {
                "path": _display_path(args.output), "tracked": True,
                "bytes": args.output.stat().st_size, "sha256": _file_sha256(args.output),
            },
            "merged_records": {
                "path": _display_path(args.records_csv), "tracked": True,
                "rows": len(all_records), "bytes": args.records_csv.stat().st_size,
                "sha256": _file_sha256(args.records_csv),
            },
            "raw_predictions": {
                "path": _display_path(args.raw_csv), "tracked": False,
                "ignore_rule": "outputs/prediction_error_factor_predictions_rank1.csv",
                "rows": len(all_raw), "bytes": args.raw_csv.stat().st_size,
                "sha256": _file_sha256(args.raw_csv),
            },
            "unit_cache": {
                "path": _display_path(args.cache_dir), "tracked": False,
                "ignore_rule": "experiments/prediction_error_factor_shards/",
                "files": len(cache_files),
                "active_units": active_cache_units,
                "inactive_preserved_files": max(0, len(cache_files) - active_cache_units),
                "bytes": sum(path.stat().st_size for path in cache_files),
                "contract": "One atomic JSON shard per deterministic hypothesis/target/seed unit; reruns validate matrix identity and skip matching shards.",
            },
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(all_records), "raw": len(all_raw)}, indent=2))


if __name__ == "__main__":
    main()
