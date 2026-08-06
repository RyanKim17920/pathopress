"""Low-rank and benchmark-correlation analyses ported from BenchPress."""

from __future__ import annotations

from typing import Any

import numpy as np

from pathopress.metrics import median_absolute_percentage_error


def find_largest_complete_submatrix(
    matrix: np.ndarray, *, min_evaluations: int = 5
) -> tuple[list[int], list[int]]:
    observed = np.isfinite(matrix)
    order = np.argsort(-observed.sum(axis=0), kind="stable")
    best_rows: list[int] = []
    best_columns: list[int] = []
    for count in range(min_evaluations, matrix.shape[1] + 1):
        columns = order[:count]
        rows = np.flatnonzero(observed[:, columns].all(axis=1))
        if len(rows) * count > len(best_rows) * len(best_columns):
            best_rows = rows.astype(int).tolist()
            best_columns = columns.astype(int).tolist()
    return best_rows, best_columns


def complete_submatrix_for_count(
    matrix: np.ndarray, count: int
) -> tuple[list[int], list[int]]:
    observed = np.isfinite(matrix)
    order = np.argsort(-observed.sum(axis=0), kind="stable")
    columns = order[:count]
    rows = np.flatnonzero(observed[:, columns].all(axis=1))
    return rows.astype(int).tolist(), columns.astype(int).tolist()


def singular_summary(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("complete submatrix needs at least two rows and one column")
    centered = matrix - matrix.mean(axis=0)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    squared = singular**2
    total = float(squared.sum())
    stable_rank = float(total / squared[0]) if squared[0] > 0 else 0.0
    cumulative = np.cumsum(squared) / total if total > 0 else np.zeros_like(squared)
    return {
        "singular_values": singular.tolist(),
        "stable_rank": stable_rank,
        "cumulative_variance_explained": cumulative.tolist(),
        "var_rank1": float(cumulative[0]) if len(cumulative) else 0.0,
        "var_rank2": float(cumulative[min(1, len(cumulative) - 1)]) if len(cumulative) else 0.0,
    }


def logit_z_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = np.asarray(matrix, dtype=float).copy()
    observed = np.isfinite(transformed)
    probability = np.clip(transformed[observed], 0.5, 99.5) / 100.0
    transformed[observed] = np.log(probability / (1.0 - probability))
    means = np.nanmean(transformed, axis=0)
    stds = np.nanstd(transformed, axis=0)
    stds[~np.isfinite(stds) | (stds < 1e-8)] = 1.0
    return (transformed - means) / stds, means, stds


def _inverse_logit(values: np.ndarray) -> np.ndarray:
    return 100.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def pairwise_correlations(
    matrix: np.ndarray, *, min_shared: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalized, means, stds = logit_z_matrix(matrix)
    n_columns = matrix.shape[1]
    correlations = np.eye(n_columns)
    counts = np.zeros((n_columns, n_columns), dtype=int)
    for left in range(n_columns):
        counts[left, left] = int(np.isfinite(matrix[:, left]).sum())
        for right in range(left + 1, n_columns):
            shared = np.isfinite(matrix[:, left]) & np.isfinite(matrix[:, right])
            counts[left, right] = counts[right, left] = int(shared.sum())
            if shared.sum() < min_shared:
                correlations[left, right] = correlations[right, left] = np.nan
                continue
            x, y = normalized[shared, left], normalized[shared, right]
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                correlations[left, right] = correlations[right, left] = np.nan
                continue
            correlation = float(np.corrcoef(x, y)[0, 1])
            correlations[left, right] = correlations[right, left] = correlation
    return correlations, counts, means, stds


def best_neighbor_ols(
    matrix: np.ndarray, evaluation_ids: list[str], *, min_shared: int = 5
) -> dict[str, dict[str, Any]]:
    normalized, means, stds = logit_z_matrix(matrix)
    correlations, counts, _, _ = pairwise_correlations(matrix, min_shared=min_shared)
    output: dict[str, dict[str, Any]] = {}
    for target in range(matrix.shape[1]):
        candidates = [
            predictor for predictor in range(matrix.shape[1])
            if predictor != target and np.isfinite(correlations[target, predictor])
        ]
        if not candidates:
            continue
        predictor = max(candidates, key=lambda index: correlations[target, index] ** 2)
        shared = np.isfinite(matrix[:, target]) & np.isfinite(matrix[:, predictor])
        x = normalized[shared, predictor]
        y = normalized[shared, target]
        variance_x = float(np.sum((x - x.mean()) ** 2))
        slope = float(np.sum((x - x.mean()) * (y - y.mean())) / variance_x)
        intercept = float(y.mean() - slope * x.mean())
        prediction_z = intercept + slope * x
        prediction_logit = prediction_z * stds[target] + means[target]
        prediction = _inverse_logit(prediction_logit)
        actual = matrix[shared, target]
        absolute = np.abs(prediction - actual)
        correlation = float(correlations[target, predictor])
        output[evaluation_ids[target]] = {
            "best_neighbor": evaluation_ids[predictor],
            "max_r": round(correlation, 6),
            "max_abs_r": round(abs(correlation), 6),
            "max_r2": round(correlation**2, 6),
            "medape": round(median_absolute_percentage_error(actual, prediction), 6),
            "medae": round(float(np.median(absolute)), 6),
            "n_shared": int(counts[target, predictor]),
            "raw_predictions": {
                "model_indices": np.flatnonzero(shared).astype(int).tolist(),
                "actuals": actual.tolist(),
                "predictions": prediction.tolist(),
            },
        }
    return output


def classical_mds_from_correlations(correlations: np.ndarray) -> np.ndarray:
    correlations = np.asarray(correlations, dtype=float)
    off_diagonal = ~np.eye(len(correlations), dtype=bool)
    finite = correlations[np.isfinite(correlations) & off_diagonal]
    fill = float(np.median(np.abs(finite))) if len(finite) else 0.0
    absolute = np.where(np.isfinite(correlations), np.abs(correlations), fill)
    distance = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - absolute)))
    np.fill_diagonal(distance, 0.0)
    centering = np.eye(len(distance)) - np.ones_like(distance) / len(distance)
    gram = -0.5 * centering @ (distance**2) @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order[:2]], 0.0)
    return vectors[:, order[:2]] * np.sqrt(values)
