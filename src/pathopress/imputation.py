"""Export a completed pathology benchmark score matrix."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .completion import complete


IMPUTATION_FIELDS = (
    "model_id",
    "evaluation_id",
    "suite_id",
    "metric",
    "status",
    "normalized_score",
    "native_score_estimate",
    "model_observations",
    "evaluation_observations",
    "method",
    "rank",
    "comparison_rank",
    "comparison_normalized_score",
    "rank_sensitivity_absolute_difference",
)


def to_native_score(normalized_score: float, metric: str) -> float:
    """Map the common 0--100 score back to the source metric's scale."""
    if metric == "pearson_r":
        return normalized_score / 50.0 - 1.0
    if metric == "robustness_index":
        return normalized_score / 100.0
    return normalized_score


def build_imputation_rows(
    matrix: np.ndarray,
    models: list[str],
    evaluations: list[str],
    evaluation_metadata: dict[str, tuple[str, str]],
    *,
    rank: int = 2,
    comparison_rank: int | None = None,
) -> list[dict[str, str]]:
    """Complete a supported matrix and return one explicit row per cell."""
    if matrix.shape != (len(models), len(evaluations)):
        raise ValueError("matrix dimensions do not match model/evaluation labels")
    missing_metadata = [name for name in evaluations if name not in evaluation_metadata]
    if missing_metadata:
        raise ValueError(f"missing evaluation metadata: {missing_metadata[0]}")

    if comparison_rank is None:
        comparison_rank = 1 if rank != 1 else 2
    if comparison_rank < 1 or comparison_rank == rank:
        raise ValueError("comparison rank must be positive and differ from rank")
    completed = complete(matrix, rank=rank)
    comparison = complete(matrix, rank=comparison_rank)
    row_support = np.sum(np.isfinite(matrix), axis=1)
    col_support = np.sum(np.isfinite(matrix), axis=0)
    rows: list[dict[str, str]] = []
    for i, model in enumerate(models):
        for j, evaluation in enumerate(evaluations):
            observed = np.isfinite(matrix[i, j])
            score = float(completed[i, j])
            comparison_score = float(comparison[i, j])
            suite, metric = evaluation_metadata[evaluation]
            rows.append(
                {
                    "model_id": model,
                    "evaluation_id": evaluation,
                    "suite_id": suite,
                    "metric": metric,
                    "status": "observed" if observed else "imputed",
                    "normalized_score": f"{score:.6f}",
                    "native_score_estimate": f"{to_native_score(score, metric):.6f}",
                    "model_observations": str(int(row_support[i])),
                    "evaluation_observations": str(int(col_support[j])),
                    "method": "logit_zscore_bias_als",
                    "rank": str(rank),
                    "comparison_rank": str(comparison_rank),
                    "comparison_normalized_score": (
                        "" if observed else f"{comparison_score:.6f}"
                    ),
                    "rank_sensitivity_absolute_difference": (
                        "" if observed else f"{abs(score - comparison_score):.6f}"
                    ),
                }
            )
    return rows


def write_imputations(path: str | Path, rows: list[dict[str, str]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=IMPUTATION_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
