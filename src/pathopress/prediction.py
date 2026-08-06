"""User-facing prediction and deploy-time confidence contracts.

The point predictor is PathoPress's selected pathology adaptation of BenchPress:
logit + per-evaluation z-score + bias ALS, rank 1 and lambda 0.1. Existing-row
and genuinely unseen-row intervals use separate hash-bound calibration
artifacts; see :mod:`pathopress.new_model_confidence` for the latter.
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
from .confidence import predict_serialized_trust
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
    confidence_calibration_path: str | Path | None = None,
) -> dict[str, object]:
    """Build a compact interval artifact from cross-fitted held-out residuals.

    The repeated predictions for each observed cell are first collapsed to one
    median absolute residual. This prevents the ten seeds from masquerading as
    independent calibration examples.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    grouped: dict[tuple[str, str, str], list[float]] = {}
    risk_grouped: dict[tuple[str, str], list[float]] = {}
    with Path(cells_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["model_id"], row["evaluation_id"], row["suite_id"])
            grouped.setdefault(key, []).append(float(row["absolute_error"]))
            raw_risk = row.get("combined_risk_model_risk")
            if raw_risk not in {None, ""}:
                risk = float(raw_risk)
                if math.isfinite(risk):
                    risk_grouped.setdefault((row["model_id"], row["evaluation_id"]), []).append(risk)
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
    artifact: dict[str, object] = {
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
    if confidence_calibration_path is None:
        return artifact

    calibration_path = Path(confidence_calibration_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    expected_cells_hash = calibration.get("input", {}).get("cells_sha256")
    actual_cells_hash = sha256_file(cells_path)
    if expected_cells_hash != actual_cells_hash:
        raise ValueError(
            "confidence calibration/cell hash mismatch: "
            f"expected {expected_cells_hash}, found {actual_cells_hash}"
        )
    expected_scores_hash = calibration.get("input", {}).get("scores_sha256")
    actual_scores_hash = sha256_file(scores_path)
    if expected_scores_hash != actual_scores_hash:
        raise ValueError(
            "confidence calibration/score hash mismatch: "
            f"expected {expected_scores_hash}, found {actual_scores_hash}"
        )
    combined = calibration.get("confidence_methods", {}).get("combined_risk_model", {})
    trust = calibration.get("trust_calibration", {}).get("combined_risk_model", {})
    calibrator = trust.get("full_heldout_calibrator_for_deployment")
    scale = combined.get("conformal_90_scale_median")
    if not isinstance(calibrator, Mapping) or not isinstance(scale, (int, float)):
        raise ValueError("confidence calibration lacks the hybrid deployment calibrator")
    if not risk_grouped:
        raise ValueError("confidence cells lack hybrid risk values")
    collapsed_risk = {
        key: float(np.median(values)) for key, values in risk_grouped.items()
    }
    global_risk = float(np.median(list(collapsed_risk.values())))
    by_model: dict[str, float] = {}
    by_evaluation: dict[str, float] = {}
    for model in sorted({key[0] for key in collapsed_risk}):
        by_model[model] = float(np.median([
            value for (candidate, _), value in collapsed_risk.items() if candidate == model
        ]))
    for evaluation in sorted({key[1] for key in collapsed_risk}):
        by_evaluation[evaluation] = float(np.median([
            value for (_, candidate), value in collapsed_risk.items() if candidate == evaluation
        ]))
    artifact.update({
        "schema_version": 2,
        "artifact_type": "pathopress_hybrid_confidence_v2",
        "description": "Hybrid-risk conformal intervals and trust probabilities calibrated from cross-fitted held-out pathology cells.",
        "calibration": {
            "file": calibration_path.name,
            "sha256": sha256_file(calibration_path),
            "upstream_commit": calibration.get("upstream", {}).get("commit"),
        },
        "hybrid_risk": {
            "conformal_scale": float(scale),
            "global_median": global_risk,
            "by_model_median": by_model,
            "by_evaluation_median": by_evaluation,
            "cell_estimator": "0.5 * model median cross-fitted hybrid risk + 0.5 * evaluation median cross-fitted hybrid risk",
            "abstain_without_both_model_and_evaluation_support": True,
        },
        "trust_probability": {
            "supported": True,
            "event": "abs(predicted_normalized_score - actual_normalized_score) <= 10",
            "threshold_normalized_points": float(calibrator["threshold_normalized_points"]),
            "threshold_justification": calibration.get("configuration", {}).get("trust_threshold_justification"),
            "calibrator": dict(calibrator),
            "evaluation_protocol": "cross-fitted probabilities in the source cell artifact; full held-out monotone mapping for outcome-free deployment",
        },
    })
    artifact["applicability"] = {
        "existing_supported_models": True,
        "new_model_rows": False,
        "reason": "Hybrid risk requires a model-specific held-out risk history; genuinely unseen model rows use the separate new-model confidence artifact and abstain from this trust probability.",
    }
    return artifact


def load_confidence_artifact(
    path: str | Path,
    scores_path: str | Path,
    *,
    rank: int = DEFAULT_RANK,
    regularization: float = DEFAULT_REGULARIZATION,
) -> dict[str, object]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("artifact_type") not in {
        "pathopress_absolute_residual_conformal_v1",
        "pathopress_hybrid_confidence_v2",
    }:
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
    *,
    model_id: str | None = None,
    evaluation_id: str | None = None,
) -> tuple[float, float, str]:
    hybrid = artifact.get("hybrid_risk")
    if isinstance(hybrid, Mapping) and model_id is not None and evaluation_id is not None:
        model_risk = hybrid.get("by_model_median", {}).get(model_id)
        evaluation_risk = hybrid.get("by_evaluation_median", {}).get(evaluation_id)
        if isinstance(model_risk, (int, float)) and isinstance(evaluation_risk, (int, float)):
            risk = 0.5 * float(model_risk) + 0.5 * float(evaluation_risk)
            radius = float(hybrid["conformal_scale"]) * risk
            return (
                max(0.0, prediction - radius), min(100.0, prediction + radius),
                "hybrid:model+evaluation",
            )
    suite = artifact.get("by_suite", {}).get(suite_id)
    source = f"suite:{suite_id}"
    if not isinstance(suite, Mapping):
        suite = artifact["global"]
        source = "global"
    radius = float(suite["absolute_error_quantile"])
    return max(0.0, prediction - radius), min(100.0, prediction + radius), source


def calibrated_trust_probability(
    model_id: str,
    evaluation_id: str,
    artifact: Mapping[str, object],
) -> dict[str, object]:
    """Return outcome-free trust for a supported existing-model cell or abstain."""

    base = {
        "trust_event": "abs_error_le_10_normalized_points",
        "trust_threshold_normalized_points": 10.0,
    }
    trust = artifact.get("trust_probability")
    hybrid = artifact.get("hybrid_risk")
    if not isinstance(trust, Mapping) or not trust.get("supported") or not isinstance(hybrid, Mapping):
        return {
            **base,
            "trust_probability": None,
            "trust_probability_status": "abstained_missing_calibrator",
            "trust_abstention_reason": "hybrid trust calibrator unavailable",
        }
    model_risk = hybrid.get("by_model_median", {}).get(model_id)
    evaluation_risk = hybrid.get("by_evaluation_median", {}).get(evaluation_id)
    if not isinstance(model_risk, (int, float)) or not isinstance(evaluation_risk, (int, float)):
        return {
            **base,
            "trust_probability": None,
            "trust_probability_status": "abstained_unsupported_cell",
            "trust_abstention_reason": "both model and evaluation held-out risk support are required",
        }
    risk = 0.5 * float(model_risk) + 0.5 * float(evaluation_risk)
    probability = float(predict_serialized_trust(risk, trust["calibrator"]))
    return {
        **base,
        "trust_probability": probability,
        "trust_probability_status": "calibrated_existing_model",
        "trust_risk": risk,
        "trust_calibration_scope": "hybrid:model+evaluation",
    }
