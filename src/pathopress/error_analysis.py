"""Predictability-factor primitives adapted from BenchPress Section 6.1."""

from __future__ import annotations

import numpy as np


def low_rank_r2(matrix: np.ndarray, *, rank: int, axis: int) -> np.ndarray:
    """Best rank-k R2 after column z-scoring and zero-imputing missing cells."""

    values = np.asarray(matrix, dtype=float)
    means = np.nanmean(values, axis=0)
    stds = np.nanstd(values, axis=0)
    stds[~np.isfinite(stds) | (stds < 1e-12)] = 1.0
    normalized = (values - means) / stds
    imputed = np.where(np.isfinite(normalized), normalized, 0.0)
    left, singular, right = np.linalg.svd(imputed, full_matrices=False)
    keep = min(rank, len(singular))
    fitted = left[:, :keep] @ np.diag(singular[:keep]) @ right[:keep, :]
    total = np.sum(imputed**2, axis=axis)
    residual = np.sum((imputed - fitted) ** 2, axis=axis)
    return 1.0 - residual / np.where(total > 1e-12, total, 1.0)


def pairwise_abs_correlation(
    matrix: np.ndarray, *, axis: int, min_shared: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute Pearson correlations and pairwise shared counts."""

    values = np.asarray(matrix, dtype=float)
    if axis == 0:
        values = values.T
    elif axis != 1:
        raise ValueError("axis must be 0 (evaluations) or 1 (models)")
    size = values.shape[0]
    correlation = np.full((size, size), np.nan, dtype=float)
    shared = np.zeros((size, size), dtype=int)
    for left in range(size):
        for right in range(left + 1, size):
            valid = np.isfinite(values[left]) & np.isfinite(values[right])
            count = int(valid.sum())
            shared[left, right] = shared[right, left] = count
            if count < min_shared:
                continue
            x = values[left, valid]
            y = values[right, valid]
            if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
                continue
            value = abs(float(np.corrcoef(x, y)[0, 1]))
            if np.isfinite(value):
                correlation[left, right] = correlation[right, left] = value
    return correlation, shared


def best_neighbor_rows(
    correlation: np.ndarray, shared: np.ndarray
) -> list[dict[str, float | int | None]]:
    """Summarize the strongest finite neighbor for each row."""

    rows = []
    for index in range(correlation.shape[0]):
        values = correlation[index].copy()
        values[index] = np.nan
        if np.any(np.isfinite(values)):
            neighbor = int(np.nanargmax(values))
            rows.append(
                {
                    "best_neighbor_index": neighbor,
                    "best_neighbor_abs_r": float(values[neighbor]),
                    "best_neighbor_shared": int(shared[index, neighbor]),
                }
            )
        else:
            rows.append(
                {
                    "best_neighbor_index": None,
                    "best_neighbor_abs_r": float("nan"),
                    "best_neighbor_shared": 0,
                }
            )
    return rows


def spearman_test(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Spearman rho and two-sided p-value with finite-pair filtering."""

    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": int(valid.sum())}
    rho, p_value = stats.spearmanr(x[valid], y[valid])
    return {"rho": float(rho), "p": float(p_value), "n": int(valid.sum())}
