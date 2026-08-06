#!/usr/bin/env python3
"""Deterministic small-k exhaustive probe search with sharded raw artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.probe_compression import (  # noqa: E402
    merge_shards,
    objective_value,
    predict_all_known,
    score_predictions,
    sharded_combinations,
)


OBJECTIVES = ("medae", "medape", "pairwise_margin_error", "top_fraction_error")


def _eval(job: tuple[np.ndarray, tuple[int, ...], int]) -> dict[str, float | int | None]:
    matrix, probes, rank = job
    metrics = score_predictions(predict_all_known(matrix, probes, rank=rank))
    return {
        key: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for key, value in metrics.items()
    }


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _run_space(
    executor: ProcessPoolExecutor,
    matrix: np.ndarray,
    evaluations: list[str],
    candidates: list[int],
    k: int,
    *,
    rank: int,
    num_shards: int,
    label: str,
    shard_dir: Path,
) -> dict[str, Any]:
    config = {
        "label": label,
        "rank": rank,
        "candidate_indices": candidates,
        "k": k,
        "num_shards": num_shards,
        "num_waves": 1,
        "assignment": "ordinal % (num_shards*num_waves) == wave_index + num_waves*shard_index",
    }
    config_sha = _config_hash(config)
    shard_dir.mkdir(parents=True, exist_ok=True)
    identity_shards: list[list[tuple[int, tuple[int, ...]]]] = []
    all_rows: list[dict[str, Any]] = []
    for shard_index in range(num_shards):
        identities = list(
            sharded_combinations(
                candidates, k, shard_index=shard_index, num_shards=num_shards
            )
        )
        identity_shards.append(identities)
        metrics = list(
            executor.map(
                _eval,
                [(matrix, probes, rank) for _, probes in identities],
            )
        )
        rows = [
            {
                "ordinal": ordinal,
                "probe_indices": list(probes),
                "probe_ids": [evaluations[index] for index in probes],
                "metrics": result,
            }
            for (ordinal, probes), result in zip(identities, metrics)
        ]
        shard_payload = {
            "schema_version": 1,
            "config_sha256": config_sha,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "n_rows": len(rows),
            "rows": rows,
        }
        path = shard_dir / f"{label}_k{k}_shard{shard_index:03d}-of-{num_shards:03d}.json"
        path.write_text(json.dumps(shard_payload, indent=2) + "\n", encoding="utf-8")
        all_rows.extend(rows)
    expected = math.comb(len(candidates), k)
    merged_identities = merge_shards(identity_shards, expected)
    by_ordinal = {int(row["ordinal"]): row for row in all_rows}
    if len(by_ordinal) != expected:
        raise ValueError("duplicate or missing prediction rows across shards")
    merged_rows = [by_ordinal[ordinal] for ordinal, _ in merged_identities]
    best: dict[str, Any] = {}
    for objective in OBJECTIVES:
        def loss(row: dict[str, Any]) -> float:
            metrics = row["metrics"]
            return objective_value(metrics, objective)
        selected = min(merged_rows, key=lambda row: (loss(row), int(row["ordinal"])))
        best[objective] = {**selected, "objective_value": loss(selected)}
    return {
        "configuration": config,
        "config_sha256": config_sha,
        "expected_combinations": expected,
        "merged_count": len(merged_rows),
        "merge_validation": "no duplicate ordinals; exact contiguous ordinal coverage",
        "best_by_objective": best,
        "rows": merged_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "data/scores.csv")
    parser.add_argument("--allowlist", type=Path, default=ROOT / "data/low_friction_allowlist_v1.json")
    parser.add_argument("--compression", type=Path, default=ROOT / "experiments/probe_compression_rank1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/probe_exhaustive_rank1.json")
    parser.add_argument("--shard-dir", type=Path, default=ROOT / "experiments/probe_exhaustive_shards")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--pruned-keep", type=int, default=12)
    parser.add_argument("--workers", type=int, default=max(1, min(28, (os.cpu_count() or 2) - 1)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores)
    matrix, _, evaluations = filter_matrix(*make_matrix(scores))
    allowlist = json.loads(args.allowlist.read_text(encoding="utf-8"))
    allow_indices = [evaluations.index(value) for value in allowlist["evaluation_ids"]]
    compression = json.loads(args.compression.read_text(encoding="utf-8"))
    pruned_ids = compression["pruning"]["evaluation_ids"][: args.pruned_keep]
    pruned_indices = [evaluations.index(value) for value in pruned_ids]
    spaces: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for k in range(1, len(allow_indices) + 1):
            spaces[f"pre_error_allowlist_k{k}"] = _run_space(
                executor, matrix, evaluations, allow_indices, k, rank=args.rank,
                num_shards=args.num_shards, label="pre_error_allowlist", shard_dir=args.shard_dir,
            )
        spaces["error_informed_pruned_k2"] = _run_space(
            executor, matrix, evaluations, pruned_indices, 2, rank=args.rank,
            num_shards=args.num_shards, label="error_informed_pruned", shard_dir=args.shard_dir,
        )
    payload = {
        "schema_version": 1,
        "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
        "compression_sha256": hashlib.sha256(args.compression.read_bytes()).hexdigest(),
        "semantics": "Exact all-known BenchPress masking for every listed combination. Allowlist was defined before errors; pruned space is separately labelled error-informed.",
        "spaces": spaces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
