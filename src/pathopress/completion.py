"""A small BenchPress-style logit + bias-ALS implementation.

The formulation follows microsoft/benchpress's MIT-licensed default method,
adapted to accept an arbitrary pathology score matrix rather than global
package state. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _logit_percent(values: np.ndarray, eps: float = 0.5) -> np.ndarray:
    probabilities = np.clip(values, eps, 100.0 - eps) / 100.0
    return np.log(probabilities / (1.0 - probabilities))


def _inverse_logit(values: np.ndarray) -> np.ndarray:
    return 100.0 / (1.0 + np.exp(-values))


def _prepare(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = matrix.copy().astype(float)
    observed = np.isfinite(transformed)
    transformed[observed] = _logit_percent(transformed[observed])
    means = np.nanmean(transformed, axis=0)
    stds = np.nanstd(transformed, axis=0)
    stds[~np.isfinite(stds) | (stds < 1e-12)] = 1.0
    transformed = (transformed - means) / stds
    return transformed, means, stds


def _prepare_identity(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = matrix.copy().astype(float)
    means = np.nanmean(transformed, axis=0)
    stds = np.nanstd(transformed, axis=0)
    stds[~np.isfinite(stds) | (stds < 1e-8)] = 1.0
    transformed = (transformed - means) / stds
    return transformed, means, stds


def _bias_als(
    matrix: np.ndarray,
    *,
    rank: int = 2,
    regularization: float = 0.1,
    iterations: int = 40,
    ensembles: int = 10,
    seed: int = 42,
) -> np.ndarray:
    observed = np.isfinite(matrix)
    n_models, n_evaluations = matrix.shape
    row_observed = [np.flatnonzero(observed[i]) for i in range(n_models)]
    col_observed = [np.flatnonzero(observed[:, j]) for j in range(n_evaluations)]
    observed_cells = np.argwhere(observed)
    observed_values = matrix[observed]
    global_mean = float(np.mean(observed_values))
    ridge = np.eye(rank + 1) * regularization

    def run_one(run_seed: int) -> np.ndarray:
        rng = np.random.RandomState(run_seed)
        mean = global_mean
        row_bias = np.zeros(n_models)
        col_bias = np.zeros(n_evaluations)
        row_factors = rng.normal(0.0, 0.01, size=(n_models, rank))
        col_factors = rng.normal(0.0, 0.01, size=(n_evaluations, rank))
        for _ in range(iterations):
            for i, js in enumerate(row_observed):
                if not js.size:
                    continue
                target = matrix[i, js] - mean - col_bias[js]
                design = np.column_stack([np.ones(js.size), col_factors[js]])
                solution = np.linalg.solve(design.T @ design + ridge, design.T @ target)
                row_bias[i], row_factors[i] = solution[0], solution[1:]
            for j, rows in enumerate(col_observed):
                if not rows.size:
                    continue
                target = matrix[rows, j] - mean - row_bias[rows]
                design = np.column_stack([np.ones(rows.size), row_factors[rows]])
                solution = np.linalg.solve(design.T @ design + ridge, design.T @ target)
                col_bias[j], col_factors[j] = solution[0], solution[1:]
            interactions = np.einsum(
                "ij,ij->i",
                row_factors[observed_cells[:, 0]],
                col_factors[observed_cells[:, 1]],
            )
            mean = float(
                np.mean(
                    observed_values
                    - row_bias[observed_cells[:, 0]]
                    - col_bias[observed_cells[:, 1]]
                    - interactions
                )
            )
        return mean + row_bias[:, None] + col_bias[None, :] + row_factors @ col_factors.T

    predictions = sum(run_one(seed + offset) for offset in range(ensembles)) / ensembles
    predictions[observed] = matrix[observed]
    return predictions


def complete(
    matrix: np.ndarray,
    *,
    rank: int = 2,
    regularization: float = 0.1,
) -> np.ndarray:
    if matrix.ndim != 2 or not np.isfinite(matrix).any():
        raise ValueError("matrix must be a non-empty 2D array with observed scores")
    if rank < 0:
        raise ValueError("rank must be non-negative")
    if np.any(np.sum(np.isfinite(matrix), axis=1) == 0):
        raise ValueError("every model row must have at least one observed score")
    if np.any(np.sum(np.isfinite(matrix), axis=0) == 0):
        raise ValueError("every evaluation column must have at least one observed score")
    transformed, means, stds = _prepare(matrix)
    completed_z = _bias_als(
        transformed,
        rank=rank,
        regularization=regularization,
    )
    completed_logit = completed_z * stds + means
    predictions = _inverse_logit(completed_logit)
    predictions[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
    return np.clip(predictions, 0.0, 100.0)


def complete_soft_impute(
    matrix: np.ndarray,
    *,
    rank: int,
    transform: str = "logit",
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """BenchPress's iterative truncated-SVD rank-sweep method.

    This is the method used for BenchPress's raw/logit rank U-curve, distinct
    from the bias-ALS method used by its final default predictor.
    """
    if matrix.ndim != 2 or not np.isfinite(matrix).any():
        raise ValueError("matrix must be a non-empty 2D array with observed scores")
    if rank < 1:
        raise ValueError("rank must be at least 1")
    if transform not in {"identity", "logit"}:
        raise ValueError("transform must be 'identity' or 'logit'")
    if np.any(np.sum(np.isfinite(matrix), axis=0) == 0):
        raise ValueError("every evaluation column must have at least one observed score")

    observed = np.isfinite(matrix)
    if transform == "logit":
        working, means, stds = _prepare(matrix)
    else:
        working, means, stds = _prepare_identity(matrix)
    column_mean = np.nanmean(working, axis=0)
    imputed = working.copy()
    missing = ~observed
    imputed[missing] = np.broadcast_to(column_mean, imputed.shape)[missing]

    for _ in range(max_iterations):
        previous = imputed.copy()
        left, singular, right = np.linalg.svd(imputed, full_matrices=False)
        keep = min(rank, len(singular))
        approximation = (
            left[:, :keep] @ np.diag(singular[:keep]) @ right[:keep, :]
        )
        imputed = np.where(observed, working, approximation)
        difference = float(np.sqrt(np.mean((imputed - previous) ** 2)))
        relative = difference / (float(np.sqrt(np.mean(previous**2))) + 1e-12)
        if relative < tolerance:
            break

    restored = imputed * stds + means
    if transform == "logit":
        restored = _inverse_logit(restored)
        restored = np.clip(restored, 0.0, 100.0)
    restored[observed] = matrix[observed]
    return restored


@dataclass(frozen=True)
class ValidationResult:
    n_predictions: int
    median_absolute_error: float
    mean_absolute_error: float


def validate(
    matrix: np.ndarray,
    *,
    holdout_fraction: float = 0.2,
    repeats: int = 10,
    seed: int = 42,
    rank: int = 2,
) -> ValidationResult:
    """Random cell holdout smoke test; not a publication-grade split."""
    rng = np.random.RandomState(seed)
    observed_cells = np.argwhere(np.isfinite(matrix))
    errors: list[float] = []
    for _ in range(repeats):
        shuffled = observed_cells[rng.permutation(len(observed_cells))]
        count = max(1, int(len(shuffled) * holdout_fraction))
        train = matrix.copy()
        held_out: list[tuple[int, int]] = []
        for i, j in shuffled:
            if np.sum(np.isfinite(train[i])) <= 2 or np.sum(np.isfinite(train[:, j])) <= 3:
                continue
            train[i, j] = np.nan
            held_out.append((int(i), int(j)))
            if len(held_out) >= count:
                break
        predicted = complete(train, rank=rank)
        errors.extend(abs(float(predicted[i, j] - matrix[i, j])) for i, j in held_out)
    if not errors:
        raise ValueError("matrix is too sparse for holdout validation")
    return ValidationResult(
        n_predictions=len(errors),
        median_absolute_error=float(np.median(errors)),
        mean_absolute_error=float(np.mean(errors)),
    )
