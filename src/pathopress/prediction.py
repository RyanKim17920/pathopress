"""User-facing prediction and deploy-time confidence contracts.

The point predictor is PathoPress's selected pathology adaptation of BenchPress:
logit + per-evaluation z-score + bias ALS, rank 1 and lambda 0.1.  Confidence
intervals are loaded from a separate, hash-bound artifact and deliberately do
not apply to a genuinely new model row.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .completion import complete
from .matrix import Score, filter_matrix, load_scores, make_matrix


DEFAULT_RANK = 1
DEFAULT_REGULARIZATION = 0.1
DEFAULT_CONFIDENCE_LEVEL = 0.90


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class PredictionDataset:
    matrix: np.ndarray
    models: list[str]
    evaluations: list[str]
    scores: list[Score]
    score_by_cell: dict[tuple[str, str], Score]

    @property
    def model_index(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.models)}

    @property
    def evaluation_index(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.evaluations)}


def load_prediction_dataset(
    scores_path: str | Path,
    *,
    min_scores_per_model: int = 3,
    min_models_per_evaluation: int = 5,
) -> PredictionDataset:
    scores = load_scores(scores_path)
    full_matrix, full_models, full_evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(
        full_matrix,
        full_models,
        full_evaluations,
        min_scores_per_model=min_scores_per_model,
        min_models_per_evaluation=min_models_per_evaluation,
    )
    model_set, evaluation_set = set(models), set(evaluations)
    selected = [
        score
        for score in scores
        if score.model_id in model_set and score.evaluation_id in evaluation_set
    ]
    return PredictionDataset(
        matrix=matrix,
        models=models,
        evaluations=evaluations,
        scores=selected,
        score_by_cell={(score.model_id, score.evaluation_id): score for score in selected},
    )


def parse_known_scores(values: Iterable[str]) -> dict[str, float]:
    """Parse repeated or comma-separated ``evaluation=value`` arguments."""
    output: dict[str, float] = {}
    for raw_group in values:
        for raw_pair in raw_group.split(","):
            pair = raw_pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"malformed known score {pair!r}; expected evaluation=value")
            evaluation, raw_value = pair.split("=", 1)
            evaluation = evaluation.strip()
            if not evaluation:
                raise ValueError("known score evaluation ID cannot be empty")
            if evaluation in output:
                raise ValueError(f"duplicate known score for evaluation {evaluation!r}")
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"score for {evaluation!r} is not numeric") from exc
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise ValueError(f"normalized score for {evaluation!r} must lie in [0, 100]")
            output[evaluation] = value
    if not output:
        raise ValueError("at least one known score is required")
    return output


def complete_dataset(
    dataset: PredictionDataset,
    *,
    rank: int = DEFAULT_RANK,
    regularization: float = DEFAULT_REGULARIZATION,
) -> np.ndarray:
    return complete(dataset.matrix, rank=rank, regularization=regularization)


def predict_new_model(
    dataset: PredictionDataset,
    known_scores: Mapping[str, float],
    *,
    rank: int = DEFAULT_RANK,
    regularization: float = DEFAULT_REGULARIZATION,
) -> np.ndarray:
    unknown = sorted(set(known_scores) - set(dataset.evaluations))
    if unknown:
        raise ValueError(
            "known scores reference evaluations outside the supported matrix: "
            + ", ".join(unknown)
        )
    new_row = np.full((1, len(dataset.evaluations)), np.nan, dtype=float)
    index = dataset.evaluation_index
    for evaluation, value in known_scores.items():
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 100.0:
            raise ValueError(f"normalized score for {evaluation!r} must lie in [0, 100]")
        new_row[0, index[evaluation]] = float(value)
    augmented = np.vstack([dataset.matrix, new_row])
    return complete(
        augmented,
        rank=rank,
        regularization=regularization,
    )[-1]


def _finite_sample_quantile(values: np.ndarray, level: float) -> float:
    """Higher empirical quantile at ceil((n+1)*level), capped at n."""
    values = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if not values.size:
        raise ValueError("confidence calibration contains no finite residuals")
    position = min(values.size, int(math.ceil((values.size + 1) * level))) - 1
    return float(values[position])


def build_deployment_confidence_artifact(
    cells_path: str | Path,
    scores_path: str | Path,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    rank: int = DEFAULT_RANK,
    regularization: float = DEFAULT_REGULARIZATION,
) -> dict[str, object]:
    """Build a compact interval artifact from cross-fitted held-out residuals.

    The repeated predictions for each observed cell are first collapsed to one
    median absolute residual. This prevents the ten seeds from masquerading as
    independent calibration examples.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    grouped: dict[tuple[str, str, str], list[float]] = {}
    with Path(cells_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["model_id"], row["evaluation_id"], row["suite_id"])
            grouped.setdefault(key, []).append(float(row["absolute_error"]))
    collapsed = [
        (suite, float(np.median(errors)))
        for (_, _, suite), errors in sorted(grouped.items())
    ]
    if not collapsed:
        raise ValueError("confidence cell artifact is empty")
    global_errors = np.asarray([error for _, error in collapsed], dtype=float)
    suites: dict[str, dict[str, float | int]] = {}
    for suite in sorted({suite for suite, _ in collapsed}):
        values = np.asarray([error for name, error in collapsed if name == suite])
        suites[suite] = {
            "n_unique_cells": int(values.size),
            "absolute_error_quantile": _finite_sample_quantile(values, confidence_level),
        }
    return {
        "schema_version": 1,
        "artifact_type": "pathopress_absolute_residual_conformal_v1",
        "description": "Symmetric held-out-cell intervals calibrated from one median cross-fitted residual per observed model-evaluation cell.",
        "scores": {"sha256": sha256_file(scores_path)},
        "calibration_cells": {
            "sha256": sha256_file(cells_path),
            "n_unique_cells": len(collapsed),
        },
        "predictor": {
            "method": "logit_bias_als",
            "rank": rank,
            "regularization": regularization,
        },
        "confidence_level": confidence_level,
        "global": {
            "n_unique_cells": int(global_errors.size),
            "absolute_error_quantile": _finite_sample_quantile(
                global_errors, confidence_level
            ),
        },
        "by_suite": suites,
        "applicability": {
            "existing_supported_models": True,
            "new_model_rows": False,
            "reason": "The calibration population contains held-out cells from supported rows, not genuinely unseen models.",
        },
    }


def load_confidence_artifact(
    path: str | Path,
    scores_path: str | Path,
    *,
    rank: int = DEFAULT_RANK,
    regularization: float = DEFAULT_REGULARIZATION,
) -> dict[str, object]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("artifact_type") != "pathopress_absolute_residual_conformal_v1":
        raise ValueError("unsupported deploy-time confidence artifact")
    expected_hash = artifact.get("scores", {}).get("sha256")
    actual_hash = sha256_file(scores_path)
    if expected_hash != actual_hash:
        raise ValueError(
            f"confidence artifact score hash mismatch: expected {expected_hash}, found {actual_hash}"
        )
    predictor = artifact.get("predictor", {})
    expected = {
        "method": "logit_bias_als",
        "rank": rank,
        "regularization": regularization,
    }
    if predictor != expected:
        raise ValueError(
            f"confidence artifact predictor mismatch: expected {expected}, found {predictor}"
        )
    return artifact


def calibrated_interval(
    prediction: float,
    suite_id: str,
    artifact: Mapping[str, object],
) -> tuple[float, float, str]:
    suite = artifact.get("by_suite", {}).get(suite_id)
    source = f"suite:{suite_id}"
    if not isinstance(suite, Mapping):
        suite = artifact["global"]
        source = "global"
    radius = float(suite["absolute_error_quantile"])
    return max(0.0, prediction - radius), min(100.0, prediction + radius), source
