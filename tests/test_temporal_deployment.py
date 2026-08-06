from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from pathopress.matrix import filter_matrix, load_scores, make_matrix
from pathopress.temporal import (
    ReleaseMetadata,
    load_release_metadata,
    run_unit,
    select_targets,
    training_models,
    validate_metadata_coverage,
)


ROOT = Path(__file__).resolve().parents[1]


def release(model_id: str, value: str) -> ReleaseMetadata:
    return ReleaseMetadata(
        model_id=model_id,
        release_date=date.fromisoformat(value),
        verification_status="verified",
        date_basis="paper_first_public",
        is_proxy=False,
        primary_source_url="https://example.test/primary",
        source_title="Primary source",
        audit_notes="test fixture",
    )


def test_release_metadata_exactly_covers_retained_matrix() -> None:
    matrix, models, evaluations = make_matrix(load_scores(ROOT / "data" / "scores.csv"))
    _, models, _ = filter_matrix(matrix, models, evaluations)
    metadata = load_release_metadata(ROOT / "data" / "model_release_dates.csv")
    validate_metadata_coverage(models, metadata)
    assert len(models) == len(metadata) == 59
    assert all(row.primary_source_url.startswith("https://") for row in metadata.values())
    assert all(row.release_date is not None for row in metadata.values())


def test_hard_rule_selection_uses_date_and_coverage() -> None:
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, np.nan], [5.0, 6.0, 7.0]])
    models = ["eligible", "too_sparse", "too_late"]
    metadata = {
        "eligible": release("eligible", "2025-02-01"),
        "too_sparse": release("too_sparse", "2025-02-01"),
        "too_late": release("too_late", "2026-02-01"),
    }
    assert select_targets(
        matrix,
        models,
        metadata,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        observed_score_count_gt=1,
    ) == ["eligible"]


def test_training_models_are_strictly_earlier() -> None:
    models = ["old", "same_day", "target", "new"]
    metadata = {
        "old": release("old", "2024-01-01"),
        "same_day": release("same_day", "2025-01-01"),
        "target": release("target", "2025-01-01"),
        "new": release("new", "2025-01-02"),
    }
    assert training_models(models, metadata, "target") == ["old"]


def test_raw_rows_preserve_revealed_hidden_and_not_predictable() -> None:
    # Column c exists only on the target. It must remain not-predictable when hidden.
    matrix = np.array([[20.0, 30.0, np.nan], [40.0, 50.0, 60.0]])
    models = ["old", "target"]
    evaluations = ["a", "b", "c"]
    metadata = {
        "old": release("old", "2024-01-01"),
        "target": release("target", "2025-01-01"),
    }
    unit = run_unit(
        matrix,
        models,
        evaluations,
        metadata,
        target_model_id="target",
        k=1,
        seed=0,
        rank=1,
    )
    assert len(unit["raw_predictions"]) == 3
    revealed = [row for row in unit["raw_predictions"] if row["is_revealed"]]
    assert len(revealed) == 1
    assert revealed[0]["pred"] == revealed[0]["actual"]
    sources = {row["evaluation_id"]: row["prediction_source"] for row in unit["raw_predictions"]}
    # If c happened to be revealed, both supported hidden columns remain predictable;
    # otherwise c is explicitly recorded as not-predictable.
    if sources["c"] != "revealed":
        assert sources["c"] == "not_predictable"
    assert unit["config"]["n_eval_cells"] == 3
    assert unit["metrics"]["n"] == unit["config"]["n_metric_cells"]


def test_revealed_sampling_is_reproducible() -> None:
    matrix = np.array([[20.0, 30.0, 40.0], [45.0, 55.0, 65.0]])
    models = ["old", "target"]
    evaluations = ["a", "b", "c"]
    metadata = {
        "old": release("old", "2024-01-01"),
        "target": release("target", "2025-01-01"),
    }
    kwargs = dict(
        matrix=matrix,
        models=models,
        evaluations=evaluations,
        metadata=metadata,
        target_model_id="target",
        k=1,
        seed=7,
        rank=1,
    )
    first = run_unit(**kwargs)
    second = run_unit(**kwargs)
    assert first["config"]["revealed_evaluation_ids"] == second["config"]["revealed_evaluation_ids"]
    assert first["raw_predictions"] == second["raw_predictions"]
