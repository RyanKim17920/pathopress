"""Shared BenchPress-compatible score-error metrics."""

from __future__ import annotations

import numpy as np


# Pinned to Microsoft's BenchPress denominator guard. Values at or below this
# absolute magnitude are excluded rather than producing unstable percentages.
MEDAPE_EPSILON = 1e-6


def absolute_percentage_errors(
    actual: np.ndarray | list[float],
    predicted: np.ndarray | list[float],
    *,
    epsilon: float = MEDAPE_EPSILON,
) -> np.ndarray:
    """Return finite absolute percentage errors under the pinned denominator.

    Non-finite actual/predicted pairs and targets with ``abs(actual) <= epsilon``
    are excluded. The result is expressed in percentage points.
    """

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted must have equal shape")
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be a finite non-negative number")
    valid = (
        np.isfinite(actual_array)
        & np.isfinite(predicted_array)
        & (np.abs(actual_array) > epsilon)
    )
    return (
        100.0
        * np.abs(predicted_array[valid] - actual_array[valid])
        / np.abs(actual_array[valid])
    )


def median_absolute_percentage_error(
    actual: np.ndarray | list[float],
    predicted: np.ndarray | list[float],
    *,
    epsilon: float = MEDAPE_EPSILON,
) -> float:
    """Return MedAPE, or NaN when no pair has a supported denominator."""

    errors = absolute_percentage_errors(actual, predicted, epsilon=epsilon)
    return float(np.median(errors)) if errors.size else float("nan")
