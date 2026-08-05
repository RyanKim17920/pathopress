"""BenchPress-style leaderboard-ranking preservation metrics.

These metrics compare an observed score matrix with a completed matrix.  They
assume that larger scores are better and that the completed matrix already
contains the true values for cells that were not held out.

Pathology endpoints do not share a universal clinically meaningful score gap,
so pairwise margins are explicit.  Callers may pass one scalar margin or one
margin per evaluation column; the default is zero (all non-tied true pairs).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PairwiseColumnResult:
    """Margin-aware ranking result for one evaluation column."""

    column: int
    margin: float
    n_models: int
    n_heldout_models: int
    n_pairs: int
    n_correct: int
    n_predicted_ties: int
    accuracy: float


@dataclass(frozen=True)
class PairwiseRankingResult:
    """Per-column results and both benchmark-median and pooled summaries."""

    columns: tuple[PairwiseColumnResult, ...]
    n_eligible_columns: int
    n_pairs: int
    n_correct: int
    n_predicted_ties: int
    median_accuracy: float
    pooled_accuracy: float


@dataclass(frozen=True)
class TopFractionColumnResult:
    """Top-fraction set recovery for one evaluation column."""

    column: int
    top_fraction: float
    n_models: int
    n_heldout_models: int
    k: int
    overlap: int
    recovery: float


@dataclass(frozen=True)
class TopFractionRecoveryResult:
    """Per-column top-set results and benchmark-median/pooled summaries."""

    top_fraction: float
    columns: tuple[TopFractionColumnResult, ...]
    n_eligible_columns: int
    total_k: int
    total_overlap: int
    median_recovery: float
    pooled_recovery: float


def _validate_inputs(
    actual: np.ndarray,
    predicted: np.ndarray,
    heldout: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    heldout_array = np.asarray(heldout, dtype=bool)
    if actual_array.ndim != 2:
        raise ValueError("actual, predicted, and heldout must be 2D matrices")
    if predicted_array.shape != actual_array.shape or heldout_array.shape != actual_array.shape:
        raise ValueError(
            "actual, predicted, and heldout must have identical shapes; "
            f"got {actual_array.shape}, {predicted_array.shape}, and {heldout_array.shape}"
        )
    return actual_array, predicted_array, heldout_array


def _column_margins(
    margin: float | Sequence[float] | np.ndarray,
    n_columns: int,
) -> np.ndarray:
    if np.isscalar(margin):
        margins = np.full(n_columns, float(margin), dtype=float)
    else:
        margins = np.asarray(margin, dtype=float)
        if margins.ndim != 1 or len(margins) != n_columns:
            raise ValueError(
                "per-column margin must be a one-dimensional sequence with "
                f"length {n_columns}"
            )
    if not np.isfinite(margins).all() or np.any(margins < 0.0):
        raise ValueError("margins must be finite and non-negative")
    return margins


def pairwise_ranking_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
    heldout: np.ndarray,
    *,
    margin: float | Sequence[float] | np.ndarray = 0.0,
) -> PairwiseRankingResult:
    """Measure whether completed leaderboards preserve same-column ordering.

    A pair is eligible when both true and predicted values are finite, at least
    one member was held out, the true scores are not tied, and their absolute
    true-score gap is at least that column's margin.  Predicted ties count as
    errors.  ``median_accuracy`` gives every eligible evaluation column equal
    weight, matching BenchPress's paper summary; ``pooled_accuracy`` weights by
    the number of eligible pairs and is included to make the denominator clear.
    """

    actual_array, predicted_array, heldout_array = _validate_inputs(
        actual, predicted, heldout
    )
    margins = _column_margins(margin, actual_array.shape[1])
    column_results: list[PairwiseColumnResult] = []

    for column in range(actual_array.shape[1]):
        valid = np.isfinite(actual_array[:, column]) & np.isfinite(
            predicted_array[:, column]
        )
        true_scores = actual_array[valid, column]
        predicted_scores = predicted_array[valid, column]
        column_heldout = heldout_array[valid, column]
        n_models = int(len(true_scores))
        n_heldout = int(np.sum(column_heldout))

        if n_models < 2 or n_heldout == 0:
            n_pairs = n_correct = n_predicted_ties = 0
            accuracy = float("nan")
        else:
            true_difference = true_scores[:, None] - true_scores[None, :]
            predicted_difference = (
                predicted_scores[:, None] - predicted_scores[None, :]
            )
            upper_triangle = np.triu(
                np.ones((n_models, n_models), dtype=bool), k=1
            )
            has_holdout = column_heldout[:, None] | column_heldout[None, :]
            comparable = (
                upper_triangle
                & has_holdout
                & (true_difference != 0.0)
                & (np.abs(true_difference) >= margins[column])
            )
            true_sign = np.sign(true_difference[comparable])
            predicted_sign = np.sign(predicted_difference[comparable])
            n_pairs = int(np.sum(comparable))
            n_correct = int(np.sum(true_sign == predicted_sign))
            n_predicted_ties = int(np.sum(predicted_sign == 0.0))
            accuracy = n_correct / n_pairs if n_pairs else float("nan")

        column_results.append(
            PairwiseColumnResult(
                column=column,
                margin=float(margins[column]),
                n_models=n_models,
                n_heldout_models=n_heldout,
                n_pairs=n_pairs,
                n_correct=n_correct,
                n_predicted_ties=n_predicted_ties,
                accuracy=float(accuracy),
            )
        )

    eligible = [result for result in column_results if result.n_pairs > 0]
    n_pairs = sum(result.n_pairs for result in eligible)
    n_correct = sum(result.n_correct for result in eligible)
    return PairwiseRankingResult(
        columns=tuple(column_results),
        n_eligible_columns=len(eligible),
        n_pairs=n_pairs,
        n_correct=n_correct,
        n_predicted_ties=sum(result.n_predicted_ties for result in eligible),
        median_accuracy=(
            float(np.median([result.accuracy for result in eligible]))
            if eligible
            else float("nan")
        ),
        pooled_accuracy=n_correct / n_pairs if n_pairs else float("nan"),
    )


def top_fraction_recovery(
    actual: np.ndarray,
    predicted: np.ndarray,
    heldout: np.ndarray,
    *,
    top_fraction: float,
) -> TopFractionRecoveryResult:
    """Measure overlap between true and completed top-model sets by column.

    Each eligible column needs at least two finite model scores and at least one
    held-out cell.  Its top-set size is ``ceil(top_fraction * n_models)``.  As
    in BenchPress, seen and held-out models both participate in the two full
    leaderboard top sets.
    """

    actual_array, predicted_array, heldout_array = _validate_inputs(
        actual, predicted, heldout
    )
    if not np.isfinite(top_fraction) or not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be finite and in (0, 1]")

    column_results: list[TopFractionColumnResult] = []
    for column in range(actual_array.shape[1]):
        valid = np.isfinite(actual_array[:, column]) & np.isfinite(
            predicted_array[:, column]
        )
        true_scores = actual_array[valid, column]
        predicted_scores = predicted_array[valid, column]
        column_heldout = heldout_array[valid, column]
        n_models = int(len(true_scores))
        n_heldout = int(np.sum(column_heldout))
        if n_models < 2 or n_heldout == 0:
            k = overlap = 0
            recovery = float("nan")
        else:
            k = max(1, min(n_models, int(math.ceil(top_fraction * n_models))))
            true_order = np.argsort(-true_scores, kind="stable")
            predicted_order = np.argsort(-predicted_scores, kind="stable")
            overlap = len(set(true_order[:k]) & set(predicted_order[:k]))
            recovery = overlap / k
        column_results.append(
            TopFractionColumnResult(
                column=column,
                top_fraction=float(top_fraction),
                n_models=n_models,
                n_heldout_models=n_heldout,
                k=k,
                overlap=overlap,
                recovery=float(recovery),
            )
        )

    eligible = [result for result in column_results if result.k > 0]
    total_k = sum(result.k for result in eligible)
    total_overlap = sum(result.overlap for result in eligible)
    return TopFractionRecoveryResult(
        top_fraction=float(top_fraction),
        columns=tuple(column_results),
        n_eligible_columns=len(eligible),
        total_k=total_k,
        total_overlap=total_overlap,
        median_recovery=(
            float(np.median([result.recovery for result in eligible]))
            if eligible
            else float("nan")
        ),
        pooled_recovery=total_overlap / total_k if total_k else float("nan"),
    )
