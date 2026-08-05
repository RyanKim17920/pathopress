"""Load citation-backed scores into a model-by-evaluation matrix."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Score:
    model_id: str
    evaluation_id: str
    value: float
    normalized_score: float
    suite_id: str
    metric: str
    reference_url: str
    audit_status: str


PROTOTYPE_EVIDENCE_STATUSES = {"verified", "parsed_primary_source"}


def load_scores(path: str | Path, *, verified_only: bool = True) -> list[Score]:
    """Load scores, excluding external/unreviewed candidates by default.

    ``parsed_primary_source`` is accepted for the research prototype even
    though it is not equivalent to dual human verification. The status is
    retained on every returned row so downstream releases can impose a
    stricter policy without rewriting the evidence file.
    """
    rows: list[Score] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if verified_only and row["audit_status"] not in PROTOTYPE_EVIDENCE_STATUSES:
                continue
            rows.append(
                Score(
                    model_id=row["model_id"],
                    evaluation_id=row["evaluation_id"],
                    value=float(row["value"]),
                    normalized_score=float(row["normalized_score"]),
                    suite_id=row["suite_id"],
                    metric=row["metric"],
                    reference_url=row["reference_url"],
                    audit_status=row["audit_status"],
                )
            )
    return rows


def make_matrix(scores: list[Score]) -> tuple[np.ndarray, list[str], list[str]]:
    models = sorted({score.model_id for score in scores})
    evaluations = sorted({score.evaluation_id for score in scores})
    model_idx = {name: i for i, name in enumerate(models)}
    evaluation_idx = {name: j for j, name in enumerate(evaluations)}
    matrix = np.full((len(models), len(evaluations)), np.nan, dtype=float)
    seen: set[tuple[str, str]] = set()

    for score in scores:
        key = (score.model_id, score.evaluation_id)
        if key in seen:
            raise ValueError(f"duplicate score cell: {score.model_id}/{score.evaluation_id}")
        seen.add(key)
        if not np.isfinite(score.normalized_score):
            raise ValueError(f"non-finite normalized score: {score.model_id}/{score.evaluation_id}")
        i = model_idx[score.model_id]
        j = evaluation_idx[score.evaluation_id]
        matrix[i, j] = score.normalized_score
    return matrix, models, evaluations


def filter_matrix(
    matrix: np.ndarray,
    models: list[str],
    evaluations: list[str],
    *,
    min_scores_per_model: int = 3,
    min_models_per_evaluation: int = 5,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Iteratively remove rows/columns without enough support."""
    keep_rows = np.ones(matrix.shape[0], dtype=bool)
    keep_cols = np.ones(matrix.shape[1], dtype=bool)
    changed = True
    while changed:
        changed = False
        sub = matrix[np.ix_(keep_rows, keep_cols)]
        row_ok = np.sum(np.isfinite(sub), axis=1) >= min_scores_per_model
        col_ok = np.sum(np.isfinite(sub), axis=0) >= min_models_per_evaluation
        row_positions = np.flatnonzero(keep_rows)
        col_positions = np.flatnonzero(keep_cols)
        if not np.all(row_ok):
            keep_rows[row_positions[~row_ok]] = False
            changed = True
        if not np.all(col_ok):
            keep_cols[col_positions[~col_ok]] = False
            changed = True
    return (
        matrix[np.ix_(keep_rows, keep_cols)],
        [m for m, keep in zip(models, keep_rows) if keep],
        [e for e, keep in zip(evaluations, keep_cols) if keep],
    )
