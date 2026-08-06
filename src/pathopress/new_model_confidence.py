"""Leakage-free confidence calibration for genuinely unseen model rows.

The calibration population consists only of sparse-row predictions where the
target model's hidden values were unavailable to the point predictor.  Risk
lookups and conformal scales are evaluated leave-one-model-out, so a target
row never calibrates its own hidden cells.  Repeated cells and probe seeds are
collapsed within model before estimating risk, preventing dense models from
dominating the deployment artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SUPPORTED_PROBE_COUNTS = (1, 3, 5, 10)
ARTIFACT_TYPE = "pathopress_new_model_group_conformal_v1"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_probe_count(n_known: int) -> int:
    """Return the conservative calibrated probe-count bucket."""
    if n_known < 1:
        raise ValueError("new-model confidence requires at least one known score")
    return max(k for k in SUPPORTED_PROBE_COUNTS if k <= n_known)


def _finite_sample_quantile(values: Sequence[float], level: float) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    array = array[np.isfinite(array)]
    if not array.size:
        raise ValueError("cannot calibrate a quantile without finite residuals")
    position = min(array.size, int(math.ceil((array.size + 1) * level))) - 1
    return float(array[position])


def _model_balanced_median(groups: Mapping[str, Sequence[float]]) -> float | None:
    medians = [
        float(np.median(np.asarray(values, dtype=float)))
        for values in groups.values()
        if len(values)
    ]
    return float(np.median(medians)) if medians else None


def _model_balanced_quantile(
    groups: Mapping[str, Sequence[float]], level: float
) -> float:
    """Weighted quantile where every target model contributes total weight one."""
    values: list[float] = []
    weights: list[float] = []
    for group in groups.values():
        finite = [float(value) for value in group if math.isfinite(float(value))]
        if not finite:
            continue
        values.extend(finite)
        weights.extend([1.0 / len(finite)] * len(finite))
    if not values:
        raise ValueError("cannot calibrate a model-balanced quantile without residuals")
    order = np.argsort(np.asarray(values))
    ordered_values = np.asarray(values)[order]
    ordered_weights = np.asarray(weights)[order]
    cumulative = np.cumsum(ordered_weights) / np.sum(ordered_weights)
    return float(ordered_values[min(int(np.searchsorted(cumulative, level, side="left")), len(values) - 1)])


def _risk_coverage(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    valid = [
        row for row in rows
        if row.get("confidence_status") == "calibrated"
        and math.isfinite(float(row["crossfit_risk"]))
    ]
    valid.sort(key=lambda row: (float(row["crossfit_risk"]), str(row["target_model_id"])))
    output = []
    for fraction in (1.0, 0.8, 0.6, 0.4, 0.2):
        count = max(1, int(math.ceil(fraction * len(valid)))) if valid else 0
        selected = valid[:count]
        errors = np.asarray([float(row["absolute_error"]) for row in selected])
        output.append({
            "kept_fraction": fraction,
            "n": count,
            "medae": float(np.median(errors)) if count else None,
            "mae": float(np.mean(errors)) if count else None,
        })
    return output


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    calibrated = [row for row in rows if row.get("confidence_status") == "calibrated"]
    covered = [
        float(row["crossfit_lower_90"]) <= float(row["actual"]) <= float(row["crossfit_upper_90"])
        for row in calibrated
    ]
    widths = [
        float(row["crossfit_upper_90"]) - float(row["crossfit_lower_90"])
        for row in calibrated
    ]
    return {
        "n_predictions": len(rows),
        "n_calibrated": len(calibrated),
        "n_abstained": len(rows) - len(calibrated),
        "prediction_coverage": len(calibrated) / len(rows) if rows else None,
        "interval_coverage": float(np.mean(covered)) if covered else None,
        "median_interval_width": float(np.median(widths)) if widths else None,
        "median_absolute_error": (
            float(np.median([float(row["absolute_error"]) for row in calibrated]))
            if calibrated else None
        ),
        "risk_coverage_curve": _risk_coverage(rows),
    }


def build_new_model_confidence_artifact(
    records: Iterable[Mapping[str, object]],
    scores_path: str | Path,
    *,
    confidence_level: float = 0.90,
    min_evaluation_models: int = 5,
    min_context_models: int = 5,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build deployment lookups and leave-model-out audited predictions.

    Each input record must describe a hidden prediction from a sparse new-row
    simulation.  The function returns the compact deployable artifact and a
    copy of every record enriched with cross-fitted risk and interval fields.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    rows = [dict(row) for row in records]
    required = {
        "target_model_id", "evaluation_id", "suite_id", "k", "actual",
        "predicted", "absolute_error", "same_suite_probe_count", "source",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"new-model confidence row {index} missing {missing}")
        row["k"] = int(row["k"])
        if row["k"] not in SUPPORTED_PROBE_COUNTS:
            raise ValueError(f"unsupported probe count {row['k']}")
        row["same_suite_probe"] = bool(int(row["same_suite_probe_count"]) > 0)
        for field in ("actual", "predicted", "absolute_error"):
            row[field] = float(row[field])

    # Index residuals as key -> model -> values. All lookup estimates are
    # medians of per-model medians, rather than medians of duplicated cells.
    evaluation_index: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    suite_same_index: dict[tuple[int, str, bool], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    suite_index: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    global_index: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        model, k, evaluation, suite = (
            str(row["target_model_id"]), int(row["k"]),
            str(row["evaluation_id"]), str(row["suite_id"]),
        )
        error = float(row["absolute_error"])
        evaluation_index[(k, evaluation)][model].append(error)
        suite_same_index[(k, suite, bool(row["same_suite_probe"]))][model].append(error)
        suite_index[(k, suite)][model].append(error)
        global_index[k][model].append(error)

    def without(
        groups: Mapping[str, Sequence[float]],
        excluded: str | set[str] | None,
    ):
        excluded_set = (
            set() if excluded is None else {excluded} if isinstance(excluded, str) else excluded
        )
        return {key: value for key, value in groups.items() if key not in excluded_set}

    def risk_for(
        row: Mapping[str, object], exclude: str | set[str] | None
    ) -> tuple[float | None, str, int]:
        k, evaluation, suite = int(row["k"]), str(row["evaluation_id"]), str(row["suite_id"])
        evaluation_groups = without(evaluation_index[(k, evaluation)], exclude)
        n_evaluation_models = len(evaluation_groups)
        if n_evaluation_models < min_evaluation_models:
            return None, "unsupported_evaluation", n_evaluation_models
        evaluation_risk = _model_balanced_median(evaluation_groups)
        candidates = (
            ("evaluation+suite_same_probe", suite_same_index[(k, suite, bool(row["same_suite_probe"]))]),
            ("evaluation+suite", suite_index[(k, suite)]),
            ("evaluation+global_k", global_index[k]),
        )
        for scope, raw_groups in candidates:
            groups = without(raw_groups, exclude)
            if len(groups) >= min_context_models:
                context_risk = _model_balanced_median(groups)
                assert evaluation_risk is not None and context_risk is not None
                return max(1e-6, 0.5 * evaluation_risk + 0.5 * context_risk), scope, n_evaluation_models
        return None, "unsupported_context", n_evaluation_models

    # Cross-fitted risk: every target model is excluded from all lookup levels.
    for row in rows:
        model = str(row["target_model_id"])
        risk, scope, n_models = risk_for(row, model)
        row["crossfit_risk"] = risk
        row["calibration_scope"] = scope
        row["calibration_evaluation_models"] = n_models
        row["calibration_excluded_target_model"] = model
        row["confidence_status"] = "calibrated" if risk is not None else "abstained"

    # Model-group-balanced conformal scaling. Every target model has total
    # weight one regardless of its score coverage or number of probe contexts;
    # leave-one-model-out scales audit deployment on a genuinely unseen row.
    ratio_groups: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    target_models = sorted({str(row["target_model_id"]) for row in rows})
    nested_scales: dict[tuple[str, int], float] = {}
    risk_cache: dict[tuple[str, str, int, str, str, bool], float | None] = {}
    for excluded_target in target_models:
        nested_ratio_groups: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for calibration_row in rows:
            calibration_model = str(calibration_row["target_model_id"])
            if calibration_model == excluded_target:
                continue
            cache_key = (
                excluded_target,
                calibration_model,
                int(calibration_row["k"]),
                str(calibration_row["evaluation_id"]),
                str(calibration_row["suite_id"]),
                bool(calibration_row["same_suite_probe"]),
            )
            if cache_key not in risk_cache:
                risk_cache[cache_key] = risk_for(
                    calibration_row, {excluded_target, calibration_model}
                )[0]
            nested_risk = risk_cache[cache_key]
            if nested_risk is not None:
                nested_ratio_groups[int(calibration_row["k"])][calibration_model].append(
                    float(calibration_row["absolute_error"]) / nested_risk
                )
        for k, groups in nested_ratio_groups.items():
            if len(groups) < min_context_models:
                continue
            corrected_level = min(
                1.0,
                math.ceil((len(groups) + 1) * confidence_level) / len(groups),
            )
            nested_scales[(excluded_target, k)] = _model_balanced_quantile(
                groups, corrected_level
            )

    for row in rows:
        if row["crossfit_risk"] is not None:
            ratio_groups[int(row["k"])][str(row["target_model_id"])].append(
                float(row["absolute_error"]) / float(row["crossfit_risk"])
            )
    for row in rows:
        if row["crossfit_risk"] is None:
            row.update({"crossfit_conformal_scale": None, "crossfit_lower_90": None, "crossfit_upper_90": None})
            continue
        model, k = str(row["target_model_id"]), int(row["k"])
        scale = nested_scales.get((model, k))
        if scale is None:
            row.update({"confidence_status": "abstained", "calibration_scope": "unsupported_conformal_groups", "crossfit_conformal_scale": None, "crossfit_lower_90": None, "crossfit_upper_90": None})
            continue
        radius = scale * float(row["crossfit_risk"])
        row["crossfit_conformal_scale"] = scale
        row["crossfit_lower_90"] = max(0.0, float(row["predicted"]) - radius)
        row["crossfit_upper_90"] = min(100.0, float(row["predicted"]) + radius)

    # Final deployment lookups use every held-out target group. A genuinely new
    # model is absent from all of them by construction.
    by_evaluation: dict[str, dict[str, object]] = {}
    for (k, evaluation), groups in sorted(evaluation_index.items()):
        suite = next(str(row["suite_id"]) for row in rows if int(row["k"]) == k and str(row["evaluation_id"]) == evaluation)
        item = by_evaluation.setdefault(evaluation, {"suite_id": suite, "by_k": {}})
        item["by_k"][str(k)] = {
            "n_models": len(groups),
            "n_predictions": sum(len(value) for value in groups.values()),
            "risk_median": _model_balanced_median(groups),
            "supported": len(groups) >= min_evaluation_models,
        }

    def serialize_context(index: Mapping[object, Mapping[str, Sequence[float]]]):
        output = {}
        for key, groups in sorted(index.items(), key=lambda item: str(item[0])):
            label = "|".join(str(value).lower() if isinstance(value, bool) else str(value) for value in (key if isinstance(key, tuple) else (key,)))
            output[label] = {
                "n_models": len(groups),
                "n_predictions": sum(len(values) for values in groups.values()),
                "risk_median": _model_balanced_median(groups),
                "supported": len(groups) >= min_context_models,
            }
        return output

    scales = {}
    for k in SUPPORTED_PROBE_COUNTS:
        groups = {name: values for name, values in ratio_groups[k].items() if values}
        corrected_level = min(
            1.0,
            math.ceil((len(groups) + 1) * confidence_level) / len(groups),
        ) if groups else confidence_level
        scales[str(k)] = {
            "n_models": len(groups),
            "scale": _model_balanced_quantile(groups, corrected_level) if groups else None,
            "group_corrected_quantile_level": corrected_level,
        }

    metrics = {
        "overall": _summary(rows),
        "by_k": {str(k): _summary([row for row in rows if int(row["k"]) == k]) for k in SUPPORTED_PROBE_COUNTS},
        "by_source": {source: _summary([row for row in rows if row["source"] == source]) for source in sorted({str(row["source"]) for row in rows})},
        "by_suite": {suite: _summary([row for row in rows if row["suite_id"] == suite]) for suite in sorted({str(row["suite_id"]) for row in rows})},
    }
    artifact = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "description": "Group-balanced 90% conformal intervals for a genuinely unseen pathology model row given sparse known scores.",
        "scores": {"sha256": _sha256(scores_path)},
        "predictor": {"method": "logit_bias_als", "rank": 1, "regularization": 0.1},
        "confidence_level": confidence_level,
        "supported_probe_counts": list(SUPPORTED_PROBE_COUNTS),
        "probe_bucket_rule": "largest supported k not exceeding the number of known scores",
        "minimum_support": {"evaluation_models": min_evaluation_models, "context_models": min_context_models},
        "calibration_population": {
            "n_predictions": len(rows),
            "n_target_models": len({str(row["target_model_id"]) for row in rows}),
            "sources": {source: sum(row["source"] == source for row in rows) for source in sorted({str(row["source"]) for row in rows})},
            "unit": "target-model-balanced residual medians",
            "leakage_control": "risk and scale metrics exclude the target model; deploy lookups use only held-out model/probe and temporal residuals",
        },
        "crossfit_group_audit": {
            model: {
                "excluded_target_model": model,
                "target_absent": True,
                "training_model_ids": sorted(
                    {str(row["target_model_id"]) for row in rows} - {model}
                ),
            }
            for model in sorted({str(row["target_model_id"]) for row in rows})
        },
        "by_evaluation": by_evaluation,
        "context_risk": {
            "suite_same_probe": serialize_context(suite_same_index),
            "suite": serialize_context(suite_index),
            "global_k": serialize_context(global_index),
        },
        "conformal_scale_by_k": scales,
        "crossfit_metrics": metrics,
        "applicability": {
            "new_model_rows": True,
            "known_score_counts": list(SUPPORTED_PROBE_COUNTS),
            "unsupported_columns": "abstain when fewer than the declared number of distinct held-out target models calibrate the evaluation",
            "clinical_guarantee": False,
        },
        "limitations": [
            "Coverage is retrospective and marginal over the available pathology-model population, not a clinical guarantee.",
            "Temporal residuals cover only the pinned release landmarks and do not eliminate future domain shift.",
            "Intervals are symmetric in normalized-score space and clipped to [0, 100].",
            "Probe-count bucketing is conservative but does not model every possible probe-set selection policy.",
        ],
    }
    return artifact, rows


def load_new_model_confidence_artifact(
    path: str | Path,
    scores_path: str | Path,
    *,
    rank: int = 1,
    regularization: float = 0.1,
) -> dict[str, object]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("unsupported new-model confidence artifact")
    expected_hash = artifact.get("scores", {}).get("sha256")
    actual_hash = _sha256(scores_path)
    if expected_hash != actual_hash:
        raise ValueError(f"new-model confidence score hash mismatch: expected {expected_hash}, found {actual_hash}")
    expected = {"method": "logit_bias_als", "rank": rank, "regularization": regularization}
    if artifact.get("predictor") != expected:
        raise ValueError(f"new-model confidence predictor mismatch: expected {expected}, found {artifact.get('predictor')}")
    return artifact


def calibrated_new_model_interval(
    prediction: float,
    evaluation_id: str,
    suite_id: str,
    known_evaluation_suites: Sequence[str],
    artifact: Mapping[str, object],
) -> dict[str, object]:
    """Return a deploy-time interval or an explicit abstention record."""
    n_known = len(known_evaluation_suites)
    k = calibration_probe_count(n_known)
    evaluation = artifact.get("by_evaluation", {}).get(evaluation_id)
    base = {"requested_known_scores": n_known, "calibration_k": k}
    if not isinstance(evaluation, Mapping) or evaluation.get("suite_id") != suite_id:
        return {**base, "confidence_status": "abstained_unsupported_column", "abstention_reason": "evaluation_missing_from_calibration"}
    entry = evaluation.get("by_k", {}).get(str(k))
    if not isinstance(entry, Mapping) or not entry.get("supported"):
        return {**base, "confidence_status": "abstained_unsupported_column", "abstention_reason": "insufficient_distinct_calibration_models"}
    same_suite = any(value == suite_id for value in known_evaluation_suites)
    contexts = artifact["context_risk"]
    keys = (
        ("evaluation+suite_same_probe", "suite_same_probe", f"{k}|{suite_id}|{str(same_suite).lower()}"),
        ("evaluation+suite", "suite", f"{k}|{suite_id}"),
        ("evaluation+global_k", "global_k", str(k)),
    )
    context = None
    scope = None
    for candidate_scope, section, key in keys:
        candidate = contexts[section].get(key)
        if isinstance(candidate, Mapping) and candidate.get("supported"):
            context, scope = candidate, candidate_scope
            break
    if context is None:
        return {**base, "confidence_status": "abstained_unsupported_context", "abstention_reason": "suite_and_global_context_unavailable"}
    scale_entry = artifact["conformal_scale_by_k"].get(str(k))
    if not isinstance(scale_entry, Mapping) or scale_entry.get("scale") is None:
        return {**base, "confidence_status": "abstained_unsupported_context", "abstention_reason": "conformal_scale_unavailable"}
    risk = max(1e-6, 0.5 * float(entry["risk_median"]) + 0.5 * float(context["risk_median"]))
    scale = float(scale_entry["scale"])
    radius = risk * scale
    return {
        **base,
        "confidence_status": "calibrated_new_model",
        "confidence_method": artifact["artifact_type"],
        "confidence_level": artifact["confidence_level"],
        "risk_score": risk,
        "conformal_scale": scale,
        "lower_90": max(0.0, float(prediction) - radius),
        "upper_90": min(100.0, float(prediction) + radius),
        "calibration_scope": scope,
        "calibration_evaluation_models": int(entry["n_models"]),
        "calibration_evaluation_predictions": int(entry["n_predictions"]),
        "calibration_context_models": int(context["n_models"]),
        "calibration_context_predictions": int(context["n_predictions"]),
    }
