"""BenchPress hard-rule temporal deployment for pathology score matrices."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pathopress.completion import complete
from pathopress.metrics import MEDAPE_EPSILON, median_absolute_percentage_error


PROTOCOL_VERSION = "pathology_temporal_deployment_hard_rule_v1"


@dataclass(frozen=True)
class ReleaseMetadata:
    model_id: str
    release_date: date | None
    verification_status: str
    date_basis: str
    is_proxy: bool
    primary_source_url: str
    source_title: str
    audit_notes: str


def load_release_metadata(path: str | Path) -> dict[str, ReleaseMetadata]:
    """Load the citation ledger, preserving explicitly unverified blank dates."""

    result: dict[str, ReleaseMetadata] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model_id = row["model_id"].strip()
            if model_id in result:
                raise ValueError(f"duplicate model release metadata: {model_id}")
            raw_date = row["release_date"].strip()
            parsed_date = date.fromisoformat(raw_date) if raw_date else None
            result[model_id] = ReleaseMetadata(
                model_id=model_id,
                release_date=parsed_date,
                verification_status=row["verification_status"].strip(),
                date_basis=row["date_basis"].strip(),
                is_proxy=row["is_proxy"].strip().lower() == "true",
                primary_source_url=row["primary_source_url"].strip(),
                source_title=row["source_title"].strip(),
                audit_notes=row["audit_notes"].strip(),
            )
    return result


def validate_metadata_coverage(
    models: Sequence[str], metadata: dict[str, ReleaseMetadata]
) -> None:
    missing = sorted(set(models) - set(metadata))
    extra = sorted(set(metadata) - set(models))
    if missing or extra:
        raise ValueError(f"release metadata mismatch: missing={missing}, extra={extra}")


def select_targets(
    matrix: np.ndarray,
    models: Sequence[str],
    metadata: dict[str, ReleaseMetadata],
    *,
    start_date: date,
    end_date: date,
    observed_score_count_gt: int,
) -> list[str]:
    """Select targets using only release metadata and pre-existing coverage."""

    selected: list[str] = []
    for index, model_id in enumerate(models):
        record = metadata.get(model_id)
        if record is None or record.release_date is None:
            continue
        if record.verification_status != "verified":
            continue
        observed = int(np.isfinite(matrix[index]).sum())
        if (
            start_date <= record.release_date <= end_date
            and observed > observed_score_count_gt
        ):
            selected.append(model_id)
    return sorted(
        selected,
        key=lambda model_id: (metadata[model_id].release_date, model_id),
    )


def training_models(
    models: Sequence[str],
    metadata: dict[str, ReleaseMetadata],
    target_model_id: str,
) -> list[str]:
    """Return verified models released strictly before the target cutoff."""

    target = metadata[target_model_id]
    if target.release_date is None:
        raise ValueError(f"target {target_model_id} has no release date")
    return sorted(
        [
            model_id
            for model_id in models
            if model_id != target_model_id
            and (record := metadata.get(model_id)) is not None
            and record.verification_status == "verified"
            and record.release_date is not None
            and record.release_date < target.release_date
        ],
        key=lambda model_id: (metadata[model_id].release_date, model_id),
    )


def deterministic_rng(base_seed: int, target_model_id: str, seed: int) -> np.random.RandomState:
    material = f"{int(base_seed)}:model_{target_model_id}:{int(seed)}".encode()
    rng_seed = int(hashlib.sha256(material).hexdigest()[:8], 16)
    return np.random.RandomState(rng_seed)


def supported_completion(
    training: np.ndarray,
    *,
    rank: int,
    regularization: float = 0.1,
) -> np.ndarray:
    """Complete supported columns and leave pre-cutoff-unsupported columns NaN."""

    values = np.asarray(training, dtype=float)
    predicted = np.full_like(values, np.nan)
    supported_columns = np.isfinite(values).any(axis=0)
    if not supported_columns.any():
        return predicted
    submatrix = values[:, supported_columns]
    active_rows = np.isfinite(submatrix).any(axis=1)
    if not active_rows.any():
        return predicted
    completed = complete(
        submatrix[active_rows],
        rank=rank,
        regularization=regularization,
        allow_empty_rows=False,
    )
    predicted[np.ix_(active_rows, supported_columns)] = completed
    return predicted


def aggregate_metric(values: Iterable[float | None]) -> dict[str, object]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=float,
    )
    if not finite.size:
        return {"median": None, "iqr": None, "mean": None, "std": None, "values": []}
    return {
        "median": float(np.median(finite)),
        "iqr": float(np.percentile(finite, 75) - np.percentile(finite, 25)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "values": finite.tolist(),
    }


def run_unit(
    matrix: np.ndarray,
    models: Sequence[str],
    evaluations: Sequence[str],
    metadata: dict[str, ReleaseMetadata],
    *,
    target_model_id: str,
    k: int,
    seed: int,
    base_seed: int = 42,
    rank: int = 1,
    regularization: float = 0.1,
) -> dict[str, object]:
    """Run one target/k/seed shard with upstream raw-row semantics."""

    model_index = {model_id: i for i, model_id in enumerate(models)}
    target_i = model_index[target_model_id]
    observed_js = np.flatnonzero(np.isfinite(matrix[target_i])).astype(int)
    if len(observed_js) < k:
        raise ValueError(f"{target_model_id} has fewer than k={k} observed scores")
    train_ids = training_models(models, metadata, target_model_id)
    if not train_ids:
        raise ValueError(f"{target_model_id} has no verified earlier training models")

    shuffled = observed_js.copy()
    deterministic_rng(base_seed, target_model_id, seed).shuffle(shuffled)
    revealed_js = sorted(int(j) for j in shuffled[:k])
    revealed_set = set(revealed_js)

    train = np.full_like(matrix, np.nan, dtype=float)
    for model_id in train_ids:
        train[model_index[model_id]] = matrix[model_index[model_id]]
    train[target_i, revealed_js] = matrix[target_i, revealed_js]
    predicted_matrix = supported_completion(
        train, rank=rank, regularization=regularization
    )

    raw: list[dict[str, object]] = []
    metric_actual: list[float] = []
    metric_predicted: list[float] = []
    for evaluation_j in observed_js:
        evaluation_j = int(evaluation_j)
        actual = float(matrix[target_i, evaluation_j])
        is_revealed = evaluation_j in revealed_set
        predicted = actual if is_revealed else float(predicted_matrix[target_i, evaluation_j])
        finite = bool(np.isfinite(predicted))
        is_metric_cell = is_revealed or finite
        if is_metric_cell:
            metric_actual.append(actual)
            metric_predicted.append(predicted)
        raw.append(
            {
                "target_model_id": target_model_id,
                "cutoff_date": metadata[target_model_id].release_date.isoformat(),
                "k": int(k),
                "seed": int(seed),
                "evaluation_id": evaluations[evaluation_j],
                "actual": actual,
                "pred": predicted if finite else None,
                "is_revealed": is_revealed,
                "is_metric_cell": is_metric_cell,
                "prediction_source": (
                    "revealed" if is_revealed else "pathopress" if finite else "not_predictable"
                ),
            }
        )

    actual_array = np.asarray(metric_actual)
    predicted_array = np.asarray(metric_predicted)
    absolute_error = np.abs(predicted_array - actual_array)
    not_predictable = sum(row["prediction_source"] == "not_predictable" for row in raw)
    return {
        "config": {
            "protocol_version": PROTOCOL_VERSION,
            "target_model_id": target_model_id,
            "cutoff_date": metadata[target_model_id].release_date.isoformat(),
            "target_date_is_proxy": metadata[target_model_id].is_proxy,
            "k": int(k),
            "seed": int(seed),
            "base_seed": int(base_seed),
            "rank": int(rank),
            "regularization": float(regularization),
            "train_rule": "verified models with release_date strictly before cutoff_date",
            "probe_rule": "randomly reveal k of the target model's observed scores",
            "metric_rule": "revealed exact cells plus hidden cells with finite predictions",
            "medape_epsilon": MEDAPE_EPSILON,
            "train_model_ids": train_ids,
            "revealed_evaluation_ids": [evaluations[j] for j in revealed_js],
            "n_eval_cells": len(raw),
            "n_metric_cells": len(metric_actual),
            "n_revealed_cells": int(k),
            "n_hidden_cells": len(raw) - int(k),
            "n_not_predictable_cells": int(not_predictable),
        },
        "metrics": {
            "n": len(metric_actual),
            "medae": float(np.median(absolute_error)),
            "medape": median_absolute_percentage_error(actual_array, predicted_array),
        },
        "raw_predictions": raw,
    }
