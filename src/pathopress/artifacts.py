"""Immutable experiment-matrix and fold artifacts shared across PathoPress runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_within_model_folds(
    matrix: np.ndarray, *, n_seeds: int = 10, n_folds: int = 3, base_seed: int = 42
) -> list[dict[str, object]]:
    """Build BenchPress's persisted within-model fold assignments."""

    records: list[dict[str, object]] = []
    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        rng = np.random.RandomState(seed)
        assignments = []
        for row in range(matrix.shape[0]):
            observed = list(np.flatnonzero(np.isfinite(matrix[row])))
            rng.shuffle(observed)
            assignments.append(observed)
        for fold in range(n_folds):
            heldout = []
            for row, observed in enumerate(assignments):
                if not observed:
                    continue
                fold_size = max(1, len(observed) // n_folds)
                start = fold * fold_size
                end = start + fold_size if fold < n_folds - 1 else len(observed)
                heldout.extend([[int(row), int(column)] for column in observed[start:end]])
            records.append({"seed": seed, "fold": fold, "test_cells": heldout})
    return records


def write_fold_artifact(
    path: str | Path,
    matrix: np.ndarray,
    models: list[str],
    evaluations: list[str],
    *,
    n_seeds: int = 10,
    n_folds: int = 3,
    base_seed: int = 42,
) -> None:
    payload = {
        "schema_version": 1,
        "protocol": "benchpress_within_model_folds",
        "configuration": {
            "n_seeds": n_seeds,
            "n_folds": n_folds,
            "base_seed": base_seed,
            "matrix_shape": list(matrix.shape),
            "models": models,
            "evaluations": evaluations,
        },
        "folds": build_within_model_folds(
            matrix, n_seeds=n_seeds, n_folds=n_folds, base_seed=base_seed
        ),
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_fold_artifact(
    path: str | Path,
    matrix: np.ndarray,
    models: list[str],
    evaluations: list[str],
) -> list[tuple[int, int, np.ndarray, list[tuple[int, int]]]]:
    """Validate and materialize persisted test cells as training matrices."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    configuration = payload.get("configuration", {})
    expected = {
        "matrix_shape": list(matrix.shape),
        "models": models,
        "evaluations": evaluations,
    }
    actual = {key: configuration.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"fold artifact matrix identity mismatch: expected {expected}, found {actual}")
    output = []
    for record in payload["folds"]:
        cells = [(int(row), int(column)) for row, column in record["test_cells"]]
        if len(cells) != len(set(cells)):
            raise ValueError("fold artifact contains duplicate test cells")
        train = np.asarray(matrix, dtype=float).copy()
        for row, column in cells:
            if not np.isfinite(train[row, column]):
                raise ValueError(f"fold artifact holds out an unobserved cell: {(row, column)}")
            train[row, column] = np.nan
        output.append((int(record["seed"]), int(record["fold"]), train, cells))
    return output
