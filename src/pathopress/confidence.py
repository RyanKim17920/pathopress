"""BenchPress-style confidence calibration for held-out matrix cells.

The implementation ports the experiment primitives from Microsoft's
``benchpress.methods.confidence`` while accepting PathoPress arrays directly.
Risk scores predict ``log1p(abs(point prediction error))`` and are cross-fitted
by outer point-prediction fold. Larger scores mean less confidence.
The audited upstream revision and MIT notice are in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Iterable, Mapping

import numpy as np

from .metrics import median_absolute_percentage_error


DEFAULT_RISK_MODEL_GRID = (
    ("ridge", ()),
    ("mlp", (16,)),
    ("mlp", (32,)),
    ("mlp", (64, 32)),
)

DEFAULT_TRUST_THRESHOLD = 10.0
DEFAULT_TRUST_BINS = 20
# Pinned BenchPress interval-width denominator guard.  This is intentionally
# distinct from metrics.MEDAPE_EPSILON (1e-6): upstream confidence diagnostics
# use 1e-8 for relative interval width while MedAPE excludes through 1e-6.
RELATIVE_WIDTH_DENOMINATOR_EPSILON = 1e-8


def relative_width_denominator_mask(actual: np.ndarray) -> np.ndarray:
    """Return the exact upstream-supported denominators for relative width."""

    values = np.asarray(actual, dtype=float)
    return np.isfinite(values) & (np.abs(values) > RELATIVE_WIDTH_DENOMINATOR_EPSILON)


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not np.any(valid):
        return {"n": 0, "mae": float("nan"), "medae": float("nan"), "medape": float("nan")}
    error = np.abs(predicted[valid] - actual[valid])
    return {
        "n": int(valid.sum()),
        "mae": float(np.mean(error)),
        "medae": float(np.median(error)),
        "medape": median_absolute_percentage_error(actual[valid], predicted[valid]),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + end - 1) + 1.0
        cursor = end
    return ranks


def spearman_uncertainty_error(
    actual: np.ndarray, predicted: np.ndarray, uncertainty: np.ndarray
) -> float:
    error = np.abs(np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float))
    uncertainty = np.asarray(uncertainty, dtype=float)
    valid = np.isfinite(error) & np.isfinite(uncertainty)
    if valid.sum() < 3:
        return float("nan")
    x = _rankdata(uncertainty[valid])
    y = _rankdata(error[valid])
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def coverage_width(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(valid):
        return {
            "coverage": float("nan"),
            "median_width": float("nan"),
            "median_relative_width": float("nan"),
            "n": 0,
        }
    widths = upper[valid] - lower[valid]
    covered = (actual[valid] >= lower[valid]) & (actual[valid] <= upper[valid])
    relative = relative_width_denominator_mask(actual[valid])
    return {
        "coverage": float(np.mean(covered)),
        "median_width": float(np.median(widths)),
        "median_relative_width": float(
            np.median(widths[relative] / np.abs(actual[valid][relative])) * 100.0
        )
        if np.any(relative)
        else float("nan"),
        "n": int(valid.sum()),
    }


def conformal_interval(
    actual: np.ndarray,
    predicted: np.ndarray,
    uncertainty: np.ndarray,
    fold_id: np.ndarray,
    *,
    confidence_level: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BenchPress leave-fold-out conformal scaling for a risk score."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    fold_id = np.asarray(fold_id, dtype=int)
    if not (actual.shape == predicted.shape == uncertainty.shape == fold_id.shape):
        raise ValueError("actual, predicted, uncertainty, and fold_id must have equal shape")
    scale = np.full_like(predicted, np.nan, dtype=float)
    for fold in np.unique(fold_id):
        calibration = fold_id != fold
        valid = (
            calibration
            & np.isfinite(actual)
            & np.isfinite(predicted)
            & np.isfinite(uncertainty)
            & (uncertainty > 1e-8)
        )
        if valid.sum() < 5:
            continue
        ratio = np.abs(predicted[valid] - actual[valid]) / uncertainty[valid]
        scale[fold_id == fold] = float(np.quantile(ratio, confidence_level))
    return predicted - scale * uncertainty, predicted + scale * uncertainty, scale


def risk_coverage_curve(
    actual: np.ndarray,
    predicted: np.ndarray,
    uncertainty: np.ndarray,
    fractions: Iterable[float] = (1.0, 0.8, 0.6, 0.4, 0.2),
) -> list[dict[str, float | int]]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(uncertainty)
    order = np.argsort(uncertainty[valid], kind="mergesort")
    actual = actual[valid][order]
    predicted = predicted[valid][order]
    rows = []
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("kept fractions must lie in (0, 1]")
        keep = max(1, int(math.ceil(fraction * len(actual))))
        rows.append({"kept_fraction": float(fraction), **error_metrics(actual[:keep], predicted[:keep])})
    return rows


def uncertainty_tercile_errors(
    actual: np.ndarray, predicted: np.ndarray, uncertainty: np.ndarray
) -> list[dict[str, float | int | str]]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(uncertainty)
    if valid.sum() < 3:
        return []
    order = np.argsort(uncertainty[valid], kind="mergesort")
    actual = actual[valid]
    predicted = predicted[valid]
    labels = ("low_uncertainty", "medium_uncertainty", "high_uncertainty")
    return [
        {"bin": label, **error_metrics(actual[index], predicted[index])}
        for label, index in zip(labels, np.array_split(order, 3))
    ]


def mad_uncertainty(stack: np.ndarray) -> np.ndarray:
    stack = np.asarray(stack, dtype=float)
    median = np.nanmedian(stack, axis=0)
    return 1.4826 * np.nanmedian(np.abs(stack - median[None, :]), axis=0)


def stack_features(stack: np.ndarray, target_prediction: np.ndarray) -> dict[str, np.ndarray]:
    stack = np.asarray(stack, dtype=float)
    target_prediction = np.asarray(target_prediction, dtype=float)
    if stack.ndim != 2 or stack.shape[1] != target_prediction.size:
        raise ValueError("stack must have shape (methods, cells)")
    center = np.nanmedian(stack, axis=0)
    return {
        "std": np.nanstd(stack, axis=0),
        "mad": mad_uncertainty(stack),
        "delta_to_median": np.abs(target_prediction - center),
        "p90_p10_span": np.nanpercentile(stack, 90, axis=0)
        - np.nanpercentile(stack, 10, axis=0),
    }


def _safe_stat(matrix: np.ndarray, function, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        values = function(matrix, axis=axis)
    nan_count = int(np.sum(np.isnan(values)))
    inf_count = int(np.sum(np.isinf(values)))
    if nan_count or inf_count:
        parts = []
        if nan_count:
            parts.append(f"{nan_count} NaN")
        if inf_count:
            parts.append(f"{inf_count} Inf")
        warnings.warn(
            f"{', '.join(parts)} slice(s) replaced with 0.0 in _safe_stat",
            RuntimeWarning,
            stacklevel=2,
        )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def best_axis_correlations(matrix: np.ndarray, *, axis: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    if axis == 0:
        values = values.T
    elif axis != 1:
        raise ValueError("axis must be 0 (columns) or 1 (rows)")
    best_correlation = np.zeros(values.shape[0], dtype=float)
    best_overlap = np.zeros(values.shape[0], dtype=float)
    for left in range(values.shape[0]):
        for right in range(left + 1, values.shape[0]):
            valid = np.isfinite(values[left]) & np.isfinite(values[right])
            overlap = int(valid.sum())
            if overlap < 3:
                continue
            x = values[left, valid]
            y = values[right, valid]
            if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
                continue
            correlation = abs(float(np.corrcoef(x, y)[0, 1]))
            if not np.isfinite(correlation):
                continue
            if correlation > best_correlation[left]:
                best_correlation[left], best_overlap[left] = correlation, overlap
            if correlation > best_correlation[right]:
                best_correlation[right], best_overlap[right] = correlation, overlap
    return best_correlation, best_overlap


def structural_support_features_for_cells(
    training_matrix: np.ndarray, cells: list[tuple[int, int]]
) -> dict[str, np.ndarray]:
    training_matrix = np.asarray(training_matrix, dtype=float)
    rows = np.asarray([int(row) for row, _ in cells], dtype=int)
    columns = np.asarray([int(column) for _, column in cells], dtype=int)
    row_correlation, row_overlap = best_axis_correlations(training_matrix, axis=1)
    column_correlation, _ = best_axis_correlations(training_matrix, axis=0)
    row_all_nan = np.isfinite(training_matrix).sum(axis=1) == 0
    col_all_nan = np.isfinite(training_matrix).sum(axis=0) == 0
    return {
        "row_obs_count": np.isfinite(training_matrix).sum(axis=1).astype(float)[rows],
        "col_obs_count": np.isfinite(training_matrix).sum(axis=0).astype(float)[columns],
        "row_median_score": _safe_stat(training_matrix, np.nanmedian, axis=1)[rows],
        "col_median_score": _safe_stat(training_matrix, np.nanmedian, axis=0)[columns],
        "col_score_dispersion": _safe_stat(training_matrix, np.nanstd, axis=0)[columns],
        "row_best_peer_abs_corr": row_correlation[rows],
        "row_best_peer_overlap": row_overlap[rows],
        "col_best_neighbor_abs_corr": column_correlation[columns],
        "row_is_all_nan": row_all_nan[rows].astype(float),
        "col_is_all_nan": col_all_nan[columns].astype(float),
    }


def confidence_feature_sets(
    hp_features: dict[str, np.ndarray],
    strong_features: dict[str, np.ndarray],
    structural_features: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    disagreement = {
        **{f"hp_{key}": value for key, value in hp_features.items()},
        **{f"strong_{key}": value for key, value in strong_features.items()},
    }
    return {
        "disagreement": disagreement,
        "structural_support": structural_features,
        "combined_risk_model": {
            **{f"structural_{key}": value for key, value in structural_features.items()},
            **disagreement,
        },
    }


def feature_matrix(features: dict[str, np.ndarray], *, boolean_keys: Iterable[str] = ()) -> tuple[np.ndarray, list[str]]:
    names = sorted(features)
    boolean_keys = set(boolean_keys) | {name for name in names if name.endswith("_is_all_nan")}
    columns = []
    for name in names:
        col = np.asarray(features[name], dtype=float)
        if name not in boolean_keys:
            col = np.log1p(np.maximum(col, 0.0))
        columns.append(col)
    matrix = np.column_stack(columns)
    return matrix, names


def _risk_model(config: tuple[str, tuple[int, ...]], seed: int):
    try:
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise ImportError("confidence calibration requires `pip install -e '.[confidence]'`") from exc
    kind, hidden = config
    if kind == "ridge":
        return Ridge(alpha=1e-3)
    if kind == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=3e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=seed,
        )
    raise ValueError(f"unknown risk model: {kind}")


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    config: tuple[str, tuple[int, ...]],
    seed: int,
) -> np.ndarray:
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise ImportError("confidence calibration requires `pip install -e '.[confidence]'`") from exc
    scaler = StandardScaler()
    train = scaler.fit_transform(x_train)
    model = _risk_model(config, seed)
    model.fit(train, y_train)
    return model.predict(scaler.transform(x_test))


def _config_metadata(config: tuple[str, tuple[int, ...]]) -> dict[str, object]:
    return {"model": config[0], "hidden_layers": list(config[1])}


def select_risk_model_config(
    design: np.ndarray,
    target: np.ndarray,
    fold_id: np.ndarray,
    train_mask: np.ndarray,
    *,
    model_grid: tuple[tuple[str, tuple[int, ...]], ...] = DEFAULT_RISK_MODEL_GRID,
    seed: int = 42,
) -> tuple[str, tuple[int, ...]]:
    inner_train = train_mask & ((fold_id % 5) != 0)
    inner_validation = train_mask & ((fold_id % 5) == 0)
    if inner_validation.sum() < 50 or inner_train.sum() < design.shape[1] + 50:
        inner_train = train_mask & ((fold_id % 3) != 0)
        inner_validation = train_mask & ((fold_id % 3) == 0)
    if inner_validation.sum() < 50 or inner_train.sum() < design.shape[1] + 50:
        return model_grid[0]
    candidates = []
    for index, config in enumerate(model_grid):
        prediction = _fit_predict(
            design[inner_train], target[inner_train], design[inner_validation], config, seed + index
        )
        candidates.append((float(error_metrics(target[inner_validation], prediction)["medae"]), index, config))
    return min(candidates)[2]


def crossfit_error_risk(
    actual: np.ndarray,
    predicted: np.ndarray,
    fold_id: np.ndarray,
    features: dict[str, np.ndarray],
    *,
    seed: int = 42,
    label: str = "risk",
    verbose: bool = False,
) -> tuple[np.ndarray, list[str], dict[str, dict[str, object]]]:
    """Port of BenchPress's leave-fold risk calibrator."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    fold_id = np.asarray(fold_id, dtype=int)
    design, names = feature_matrix(features)
    target = np.log1p(np.abs(predicted - actual))
    output = np.full(len(target), np.nan, dtype=float)
    valid = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
    selected = {}
    for fold in np.unique(fold_id):
        if verbose:
            print(f"[{label}] risk fold {int(fold)} start", flush=True)
        train = (fold_id != fold) & valid
        test = (fold_id == fold) & np.all(np.isfinite(design), axis=1)
        if train.sum() < design.shape[1] + 50 or not np.any(test):
            continue
        config = select_risk_model_config(design, target, fold_id, train, seed=seed)
        selected[str(int(fold))] = _config_metadata(config)
        inner5_train = train & ((fold_id % 5) != 0)
        inner5_val = train & ((fold_id % 5) == 0)
        inner3_train = train & ((fold_id % 3) != 0)
        inner3_val = train & ((fold_id % 3) == 0)
        if inner5_val.sum() < 50 or inner5_train.sum() < design.shape[1] + 50:
            if inner3_val.sum() < 50 or inner3_train.sum() < design.shape[1] + 50:
                selected[str(int(fold))]["fallback"] = True
                warnings.warn(
                    f"risk fold {int(fold)}: both inner splits too small, "
                    "falling back to default Ridge model",
                    RuntimeWarning,
                    stacklevel=2,
                )
        output[test] = np.expm1(
            _fit_predict(design[train], target[train], design[test], config, seed + 1000 + int(fold))
        )
        if verbose:
            print(f"[{label}] risk fold {int(fold)} done config={config}", flush=True)
    return np.maximum(output, 0.0), names, selected


def summarize_confidence_method(
    actual: np.ndarray,
    predicted: np.ndarray,
    fold_id: np.ndarray,
    uncertainty: np.ndarray,
) -> dict[str, object]:
    lower, upper, scale = conformal_interval(actual, predicted, uncertainty, fold_id)
    normal_width = 1.6448536269514722 * np.asarray(uncertainty, dtype=float)
    conformal_total_cells = int(len(scale))
    conformal_skipped_cells = int(np.sum(np.isnan(scale)))
    if conformal_skipped_cells:
        warnings.warn(
            f"{conformal_skipped_cells}/{conformal_total_cells} conformal cells "
            "skipped (insufficient calibration samples per fold)",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "spearman_uncertainty_abs_error": spearman_uncertainty_error(
            actual, predicted, uncertainty
        ),
        "risk_coverage_curve": risk_coverage_curve(actual, predicted, uncertainty),
        "uncertainty_terciles": uncertainty_tercile_errors(actual, predicted, uncertainty),
        "normal_90_interval": coverage_width(
            actual, predicted - normal_width, predicted + normal_width
        ),
        "conformal_90_interval": coverage_width(actual, lower, upper),
        "conformal_90_scale_median": float(np.nanmedian(scale)),
        "conformal_skipped_cells": conformal_skipped_cells,
        "conformal_total_cells": conformal_total_cells,
    }


def _pava_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators fit for nondecreasing values.

    This is the same deterministic primitive used by BenchPress's public trust
    probability export.  Keeping it local avoids an optional isotonic-regression
    dependency and makes the serialized mapping directly reproducible.
    """

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.shape != weights.shape:
        raise ValueError("values and weights must be equal-length vectors")
    block_values: list[float] = []
    block_weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        if not np.isfinite(value) or not np.isfinite(weight) or weight <= 0:
            raise ValueError("PAVA inputs must be finite with positive weights")
        block_values.append(float(value))
        block_weights.append(float(weight))
        starts.append(index)
        ends.append(index + 1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / merged_weight
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [merged_weight]
            starts[-2:] = [starts[-2]]
            ends[-2:] = [ends[-1]]
    output = np.empty_like(values)
    for value, start, end in zip(block_values, starts, ends, strict=True):
        output[start:end] = value
    return output


def fit_trust_calibrator(
    uncertainty: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    threshold: float = DEFAULT_TRUST_THRESHOLD,
    n_bins: int = DEFAULT_TRUST_BINS,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, object]]:
    """Fit BenchPress's decreasing binned-isotonic trust mapping.

    Trust is the probability that absolute prediction error is at most
    ``threshold`` normalized score points.  The input predictions must already
    be out-of-fold when this function is used for evaluation.
    """

    uncertainty = np.asarray(uncertainty, dtype=float)
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if not (uncertainty.shape == actual.shape == predicted.shape):
        raise ValueError("uncertainty, actual, and predicted must have equal shape")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("trust threshold must be finite and positive")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    finite = np.isfinite(uncertainty) & np.isfinite(actual) & np.isfinite(predicted)
    risk = uncertainty[finite]
    trusted = (np.abs(predicted[finite] - actual[finite]) <= threshold).astype(float)
    if not risk.size:
        raise ValueError("no finite held-out risk values available for trust calibration")
    order = np.argsort(risk, kind="mergesort")
    risk = risk[order]
    trusted = trusted[order]
    bins = np.array_split(np.arange(risk.size), min(int(n_bins), risk.size))
    centers = np.asarray([np.median(risk[index]) for index in bins], dtype=float)
    empirical = np.asarray([np.mean(trusted[index]) for index in bins], dtype=float)
    weights = np.asarray([len(index) for index in bins], dtype=float)
    # Higher risk must not imply higher trust.
    calibrated = -_pava_increasing(-empirical, weights)
    calibrated = np.clip(calibrated, 0.0, 1.0)

    def predict(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        output = np.full(values.shape, np.nan, dtype=float)
        valid = np.isfinite(values)
        output[valid] = np.interp(
            values[valid], centers, calibrated,
            left=calibrated[0], right=calibrated[-1],
        )
        return output

    metadata: dict[str, object] = {
        "threshold_normalized_points": float(threshold),
        "n_calibration_cells": int(risk.size),
        "n_bins": int(len(bins)),
        "bin_risk_median": centers.tolist(),
        "bin_empirical_trust_probability": empirical.tolist(),
        "bin_calibrated_trust_probability": calibrated.tolist(),
        "bin_weight": weights.astype(int).tolist(),
        "monotonicity": "nonincreasing_probability_with_increasing_risk",
    }
    return predict, metadata


def predict_serialized_trust(
    uncertainty: np.ndarray | float,
    calibrator: Mapping[str, object],
) -> np.ndarray:
    """Apply a JSON-serializable trust calibrator without target outcomes."""

    values = np.asarray(uncertainty, dtype=float)
    centers = np.asarray(calibrator.get("bin_risk_median", []), dtype=float)
    probabilities = np.asarray(
        calibrator.get("bin_calibrated_trust_probability", []), dtype=float
    )
    if not centers.size or centers.shape != probabilities.shape:
        raise ValueError("invalid serialized trust calibrator")
    output = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    output[valid] = np.interp(
        values[valid], centers, probabilities,
        left=probabilities[0], right=probabilities[-1],
    )
    return output


def trust_probability_summary(
    actual: np.ndarray,
    predicted: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float = DEFAULT_TRUST_THRESHOLD,
    n_bins: int = 10,
) -> dict[str, object]:
    """Evaluate cross-fitted trust probabilities and emit a reliability curve."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    probability = np.asarray(probability, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(probability)
    if not np.any(valid):
        raise ValueError("no finite trust probabilities")
    outcome = (np.abs(predicted[valid] - actual[valid]) <= threshold).astype(float)
    probability = np.clip(probability[valid], 0.0, 1.0)
    order = np.argsort(probability, kind="mergesort")
    groups = np.array_split(order, min(int(n_bins), len(order)))
    reliability = [
        {
            "n": int(len(index)),
            "mean_predicted_probability": float(np.mean(probability[index])),
            "empirical_probability": float(np.mean(outcome[index])),
        }
        for index in groups
    ]
    weights = np.asarray([row["n"] for row in reliability], dtype=float)
    calibration_gap = np.asarray(
        [abs(row["mean_predicted_probability"] - row["empirical_probability"])
         for row in reliability],
        dtype=float,
    )
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return {
        "threshold_normalized_points": float(threshold),
        "n": int(len(outcome)),
        "event_prevalence": float(np.mean(outcome)),
        "mean_predicted_probability": float(np.mean(probability)),
        "brier_score": float(np.mean((probability - outcome) ** 2)),
        "log_loss": float(-np.mean(outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped))),
        "expected_calibration_error": float(np.average(calibration_gap, weights=weights)),
        "reliability_curve": reliability,
    }


def crossfit_trust_probability(
    uncertainty: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    fold_id: np.ndarray,
    *,
    group_id: np.ndarray | None = None,
    threshold: float = DEFAULT_TRUST_THRESHOLD,
    n_bins: int = DEFAULT_TRUST_BINS,
) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    """Calibrate trust while leaving each target point-prediction fold out."""

    uncertainty = np.asarray(uncertainty, dtype=float)
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    fold_id = np.asarray(fold_id, dtype=int)
    if not (uncertainty.shape == actual.shape == predicted.shape == fold_id.shape):
        raise ValueError("cross-fit trust inputs must have equal shape")
    groups = None if group_id is None else np.asarray(group_id)
    if groups is not None and groups.shape != fold_id.shape:
        raise ValueError("group_id must match the cross-fit trust input shape")
    output = np.full_like(uncertainty, np.nan, dtype=float)
    metadata: dict[str, dict[str, object]] = {}
    for fold in np.unique(fold_id):
        test = fold_id == fold
        training = fold_id != fold
        purged = 0
        if groups is not None:
            repeated_target = np.isin(groups, np.unique(groups[test]))
            purged = int(np.sum(training & repeated_target))
            training &= ~repeated_target
        predictor, fold_metadata = fit_trust_calibrator(
            uncertainty[training], actual[training], predicted[training],
            threshold=threshold, n_bins=n_bins,
        )
        output[test] = predictor(uncertainty[test])
        metadata[str(int(fold))] = {
            **fold_metadata,
            "held_out_fold": int(fold),
            "n_output_cells": int(test.sum()),
            "n_purged_repeated_target_instances": purged,
        }
    return output, metadata
