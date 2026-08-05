"""BenchPress-style global probe evaluation and selection.

A probe is an evaluation column measured on a target model before completing
the rest of that model's scorecard.  Other model rows remain visible.  This is
the transductive, all-known-cell construction protocol used by BenchPress's
hero figure; model-row holdout is a separate validation protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np

from pathopress.completion import complete


Predictor = Callable[[np.ndarray], np.ndarray]
ProbeObjective = Literal["parity_medae", "hidden_medae", "model_average_mae"]


@dataclass(frozen=True)
class ErrorSummary:
    """Absolute-error summary for one explicitly defined denominator."""

    n: int
    median_absolute_error: float
    mean_absolute_error: float


@dataclass(frozen=True)
class ModelAveragePrediction:
    """Prediction of one model's mean over its observed target cells."""

    model_index: int
    n_target_cells: int
    actual_average: float
    predicted_average: float
    absolute_error: float


@dataclass(frozen=True)
class ProbeEvaluation:
    """Three complementary views of one global probe set.

    ``parity`` matches the BenchPress all-known-cell denominator: measured
    probe cells are included as exact, zero-error predictions. ``hidden_only``
    excludes those already measured cells. ``model_average`` measures error in
    each model's mean score over its own observed target-cell universe.
    """

    probe_indices: tuple[int, ...]
    n_target_cells: int
    n_revealed_cells: int
    n_hidden_cells: int
    parity: ErrorSummary
    hidden_only: ErrorSummary
    model_average: ErrorSummary
    model_average_predictions: tuple[ModelAveragePrediction, ...]


@dataclass(frozen=True)
class CandidateScore:
    probe_index: int
    objective_value: float


@dataclass(frozen=True)
class GreedyProbeStep:
    step: int
    added_probe_index: int
    probe_indices: tuple[int, ...]
    objective: ProbeObjective
    objective_value: float
    evaluation: ProbeEvaluation
    candidate_scores: tuple[CandidateScore, ...]


@dataclass(frozen=True)
class RandomProbeEvaluation:
    repeat: int
    k: int
    probe_indices: tuple[int, ...]
    evaluation: ProbeEvaluation


@dataclass(frozen=True)
class HeldoutProbeEvaluation:
    """Isolated-new-model probe evaluation against fixed held-out rows.

    ``primary`` is ``scorecard.hidden_only`` unless ``include_probe_targets``
    was requested, in which case it is the BenchPress-compatible
    ``scorecard.parity`` denominator.
    """

    probe_indices: tuple[int, ...]
    target_model_indices: tuple[int, ...]
    context_model_indices: tuple[int, ...]
    include_probe_targets: bool
    primary: ErrorSummary
    scorecard: ProbeEvaluation


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).any():
        raise ValueError("matrix must be a non-empty 2D array with observed scores")
    if np.any(np.sum(np.isfinite(values), axis=0) == 0):
        raise ValueError("every evaluation column must have at least one observed score")
    return values


def _normalize_indices(
    indices: Sequence[int], n_evaluations: int, *, label: str
) -> tuple[int, ...]:
    normalized = tuple(int(index) for index in indices)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must not contain duplicates")
    if any(index < 0 or index >= n_evaluations for index in normalized):
        raise ValueError(f"{label} contains an out-of-range evaluation index")
    return normalized


def _error_summary(errors: Sequence[float]) -> ErrorSummary:
    values = np.asarray(errors, dtype=float)
    if not values.size:
        return ErrorSummary(0, float("nan"), float("nan"))
    return ErrorSummary(
        n=int(values.size),
        median_absolute_error=float(np.median(values)),
        mean_absolute_error=float(np.mean(values)),
    )


def _summarize_predictions(
    matrix: np.ndarray,
    predictions: np.ndarray,
    probe_indices: tuple[int, ...],
    *,
    target_observed: np.ndarray | None = None,
) -> ProbeEvaluation:
    observed = np.isfinite(matrix)
    targets = (
        observed
        if target_observed is None
        else np.asarray(target_observed, dtype=bool)
    )
    if targets.shape != matrix.shape:
        raise ValueError("target_observed must have the matrix shape")
    targets = targets & observed
    probe_mask = np.zeros(matrix.shape[1], dtype=bool)
    probe_mask[list(probe_indices)] = True
    revealed = targets & probe_mask[None, :]
    hidden = targets & ~probe_mask[None, :]

    if predictions.shape != matrix.shape:
        raise ValueError("predictor must return an array with the matrix shape")
    if not np.isfinite(predictions[targets]).all():
        raise ValueError("predictor returned a non-finite target-cell prediction")

    absolute = np.abs(predictions - matrix)
    model_averages: list[ModelAveragePrediction] = []
    for model_index in range(matrix.shape[0]):
        row_targets = targets[model_index]
        if not row_targets.any():
            continue
        actual_average = float(np.mean(matrix[model_index, row_targets]))
        predicted_average = float(np.mean(predictions[model_index, row_targets]))
        model_averages.append(
            ModelAveragePrediction(
                model_index=model_index,
                n_target_cells=int(row_targets.sum()),
                actual_average=actual_average,
                predicted_average=predicted_average,
                absolute_error=abs(predicted_average - actual_average),
            )
        )

    return ProbeEvaluation(
        probe_indices=probe_indices,
        n_target_cells=int(targets.sum()),
        n_revealed_cells=int(revealed.sum()),
        n_hidden_cells=int(hidden.sum()),
        parity=_error_summary(absolute[targets]),
        hidden_only=_error_summary(absolute[hidden]),
        model_average=_error_summary([row.absolute_error for row in model_averages]),
        model_average_predictions=tuple(model_averages),
    )


def evaluate_global_probes(
    matrix: np.ndarray,
    probe_indices: Sequence[int],
    *,
    rank: int = 1,
    regularization: float = 0.1,
    predictor: Predictor | None = None,
) -> ProbeEvaluation:
    """Evaluate one global probe set on the fixed observed-cell universe.

    Each model is treated as the target in turn. Only that row's observed probe
    cells remain visible; its other observed scores are hidden. All other rows
    retain their original observations. A target row with none of the selected
    probes is allowed and is completed from column/model context alone.
    """

    values = _validate_matrix(matrix)
    probes = _normalize_indices(
        probe_indices, values.shape[1], label="probe_indices"
    )
    observed = np.isfinite(values)
    probe_mask = np.zeros(values.shape[1], dtype=bool)
    probe_mask[list(probes)] = True
    predictions = np.full_like(values, np.nan, dtype=float)

    if predictor is None:

        def predict(train: np.ndarray) -> np.ndarray:
            return complete(
                train,
                rank=rank,
                regularization=regularization,
                allow_empty_rows=True,
            )
    else:
        predict = predictor

    for model_index in range(values.shape[0]):
        target_mask = observed[model_index]
        if not target_mask.any():
            continue
        revealed_targets = target_mask & probe_mask
        hidden_targets = target_mask & ~probe_mask
        predictions[model_index, revealed_targets] = values[
            model_index, revealed_targets
        ]
        if not hidden_targets.any():
            continue

        train = values.copy()
        train[model_index, ~probe_mask] = np.nan
        completed = np.asarray(predict(train), dtype=float)
        if completed.shape != values.shape:
            raise ValueError("predictor must return an array with the matrix shape")
        predictions[model_index, hidden_targets] = completed[
            model_index, hidden_targets
        ]

    return _summarize_predictions(values, predictions, probes)


def evaluate_column_median_baseline(matrix: np.ndarray) -> ProbeEvaluation:
    """BenchPress hero k=0 baseline: broadcast each full column median."""

    values = _validate_matrix(matrix)
    medians = np.nanmedian(values, axis=0)
    predictions = np.broadcast_to(medians, values.shape).copy()
    return _summarize_predictions(values, predictions, ())


def evaluate_heldout_model_probes(
    matrix: np.ndarray,
    probe_indices: Sequence[int],
    target_model_indices: Sequence[int],
    context_model_indices: Sequence[int],
    *,
    rank: int = 1,
    regularization: float = 0.1,
    include_probe_targets: bool = False,
    predictor: Predictor | None = None,
) -> HeldoutProbeEvaluation:
    """Evaluate probes on model rows excluded from the completion context.

    Each target model is evaluated in isolation. The completion matrix contains
    all observations from ``context_model_indices`` and only that target
    model's available probe scores. Other target rows and all unlisted rows are
    fully missing. This matches BenchPress's held-out-model probe protocol.
    """

    values = _validate_matrix(matrix)
    probes = _normalize_indices(
        probe_indices, values.shape[1], label="probe_indices"
    )
    targets = _normalize_indices(
        target_model_indices, values.shape[0], label="target_model_indices"
    )
    context = _normalize_indices(
        context_model_indices, values.shape[0], label="context_model_indices"
    )
    if not targets:
        raise ValueError("target_model_indices must not be empty")
    overlap = sorted(set(targets) & set(context))
    if overlap:
        raise ValueError(
            "target_model_indices and context_model_indices must be disjoint"
        )

    observed = np.isfinite(values)
    target_observed = np.zeros_like(observed, dtype=bool)
    target_observed[list(targets)] = observed[list(targets)]
    probe_mask = np.zeros(values.shape[1], dtype=bool)
    probe_mask[list(probes)] = True
    predictions = np.full_like(values, np.nan, dtype=float)

    if predictor is None:

        def predict(train: np.ndarray) -> np.ndarray:
            return complete(
                train,
                rank=rank,
                regularization=regularization,
                allow_empty_rows=True,
            )
    else:
        predict = predictor

    for model_index in targets:
        row_targets = observed[model_index]
        if not row_targets.any():
            continue
        revealed_targets = row_targets & probe_mask
        hidden_targets = row_targets & ~probe_mask
        predictions[model_index, revealed_targets] = values[
            model_index, revealed_targets
        ]
        if not hidden_targets.any():
            continue

        train = np.full_like(values, np.nan, dtype=float)
        if context:
            train[list(context)] = values[list(context)]
        train[model_index, revealed_targets] = values[
            model_index, revealed_targets
        ]
        completed = np.asarray(predict(train), dtype=float)
        if completed.shape != values.shape:
            raise ValueError("predictor must return an array with the matrix shape")
        predictions[model_index, hidden_targets] = completed[
            model_index, hidden_targets
        ]

    scorecard = _summarize_predictions(
        values,
        predictions,
        probes,
        target_observed=target_observed,
    )
    return HeldoutProbeEvaluation(
        probe_indices=probes,
        target_model_indices=targets,
        context_model_indices=context,
        include_probe_targets=include_probe_targets,
        primary=scorecard.parity if include_probe_targets else scorecard.hidden_only,
        scorecard=scorecard,
    )


def _objective_value(evaluation: ProbeEvaluation, objective: ProbeObjective) -> float:
    if objective == "parity_medae":
        return evaluation.parity.median_absolute_error
    if objective == "hidden_medae":
        return evaluation.hidden_only.median_absolute_error
    if objective == "model_average_mae":
        return evaluation.model_average.mean_absolute_error
    raise ValueError(f"unknown probe objective: {objective!r}")


def greedy_probe_selection(
    matrix: np.ndarray,
    *,
    max_probes: int,
    candidate_indices: Sequence[int] | None = None,
    objective: ProbeObjective = "parity_medae",
    rank: int = 1,
    regularization: float = 0.1,
    predictor: Predictor | None = None,
) -> tuple[GreedyProbeStep, ...]:
    """Build a deterministic forward-greedy global probe prefix."""

    values = _validate_matrix(matrix)
    if max_probes < 0:
        raise ValueError("max_probes must be non-negative")
    if objective not in {"parity_medae", "hidden_medae", "model_average_mae"}:
        raise ValueError(f"unknown probe objective: {objective!r}")
    candidates = _normalize_indices(
        range(values.shape[1]) if candidate_indices is None else candidate_indices,
        values.shape[1],
        label="candidate_indices",
    )
    if max_probes > len(candidates):
        raise ValueError("max_probes cannot exceed the candidate count")

    selected: list[int] = []
    remaining = list(candidates)
    trajectory: list[GreedyProbeStep] = []
    for step in range(1, max_probes + 1):
        evaluated: list[tuple[int, float, ProbeEvaluation]] = []
        for candidate in remaining:
            result = evaluate_global_probes(
                values,
                [*selected, candidate],
                rank=rank,
                regularization=regularization,
                predictor=predictor,
            )
            score = _objective_value(result, objective)
            if not np.isfinite(score):
                score = float("inf")
            evaluated.append((candidate, score, result))

        # Candidate order is the stable tie-break, matching BenchPress's loop.
        best_candidate, best_score, best_result = min(
            evaluated, key=lambda item: (item[1], remaining.index(item[0]))
        )
        candidate_scores = tuple(
            CandidateScore(probe_index=candidate, objective_value=score)
            for candidate, score, _ in evaluated
        )
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        trajectory.append(
            GreedyProbeStep(
                step=step,
                added_probe_index=best_candidate,
                probe_indices=tuple(selected),
                objective=objective,
                objective_value=best_score,
                evaluation=best_result,
                candidate_scores=candidate_scores,
            )
        )
    return tuple(trajectory)


def random_global_probe_prefixes(
    n_evaluations: int,
    *,
    max_probes: int,
    repeats: int = 10,
    seed: int = 42,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Generate BenchPress-compatible nested random global prefixes."""

    if n_evaluations < 1:
        raise ValueError("n_evaluations must be positive")
    if max_probes < 0 or max_probes > n_evaluations:
        raise ValueError("max_probes must be between zero and n_evaluations")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    all_repeats: list[tuple[tuple[int, ...], ...]] = []
    for repeat in range(repeats):
        rng = np.random.RandomState((int(seed) + repeat) * 100000)
        permutation = np.arange(n_evaluations)
        rng.shuffle(permutation)
        all_repeats.append(
            tuple(
                tuple(int(index) for index in permutation[:k])
                for k in range(1, max_probes + 1)
            )
        )
    return tuple(all_repeats)


def evaluate_random_global_prefixes(
    matrix: np.ndarray,
    *,
    max_probes: int,
    repeats: int = 10,
    seed: int = 42,
    rank: int = 1,
    regularization: float = 0.1,
    predictor: Predictor | None = None,
) -> tuple[RandomProbeEvaluation, ...]:
    """Evaluate nested random global prefixes for every repeat and budget."""

    values = _validate_matrix(matrix)
    prefixes = random_global_probe_prefixes(
        values.shape[1], max_probes=max_probes, repeats=repeats, seed=seed
    )
    results: list[RandomProbeEvaluation] = []
    for repeat, repeat_prefixes in enumerate(prefixes):
        for k, probes in enumerate(repeat_prefixes, start=1):
            results.append(
                RandomProbeEvaluation(
                    repeat=repeat,
                    k=k,
                    probe_indices=probes,
                    evaluation=evaluate_global_probes(
                        values,
                        probes,
                        rank=rank,
                        regularization=regularization,
                        predictor=predictor,
                    ),
                )
            )
    return tuple(results)
