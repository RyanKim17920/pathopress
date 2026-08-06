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
from typing import Iterable

import numpy as np


DEFAULT_RISK_MODEL_GRID = (
    ("ridge", ()),
    ("mlp", (16,)),
    ("mlp", (32,)),
    ("mlp", (64, 32)),
)


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not np.any(valid):
        return {"n": 0, "mae": float("nan"), "medae": float("nan"), "medape": float("nan")}
    error = np.abs(predicted[valid] - actual[valid])
    denominator = np.abs(actual[valid])
    relative = denominator > 1e-8
    return {
        "n": int(valid.sum()),
        "mae": float(np.mean(error)),
        "medae": float(np.median(error)),
        "medape": float(np.median(error[relative] / denominator[relative]) * 100.0)
        if np.any(relative)
        else float("nan"),
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
    relative = np.abs(actual[valid]) > 1e-8
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
    return {
        "row_obs_count": np.isfinite(training_matrix).sum(axis=1).astype(float)[rows],
        "col_obs_count": np.isfinite(training_matrix).sum(axis=0).astype(float)[columns],
        "row_median_score": _safe_stat(training_matrix, np.nanmedian, axis=1)[rows],
        "col_median_score": _safe_stat(training_matrix, np.nanmedian, axis=0)[columns],
        "col_score_dispersion": _safe_stat(training_matrix, np.nanstd, axis=0)[columns],
        "row_best_peer_abs_corr": row_correlation[rows],
        "row_best_peer_overlap": row_overlap[rows],
        "col_best_neighbor_abs_corr": column_correlation[columns],
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


def feature_matrix(features: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    names = sorted(features)
    matrix = np.column_stack(
        [np.log1p(np.maximum(np.asarray(features[name], dtype=float), 0.0)) for name in names]
    )
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
    }
