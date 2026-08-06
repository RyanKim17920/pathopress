"""BenchPress-compatible probe prediction plus pathology compression metrics.

The predictor preserves the upstream all-known and isolated-heldout masking
semantics.  Metric computation is kept separate so one expensive completion
can support MedAE, MedAPE, pairwise-margin, and top-fraction objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from pathopress.completion import complete
from pathopress.ranking import pairwise_ranking_accuracy, top_fraction_recovery


@dataclass(frozen=True)
class ProbePredictions:
    probe_indices: tuple[int, ...]
    actual: np.ndarray
    predicted: np.ndarray
    target_mask: np.ndarray
    revealed_mask: np.ndarray
    heldout_mask: np.ndarray


def _indices(values: Sequence[int], limit: int, label: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicates")
    if any(value < 0 or value >= limit for value in result):
        raise ValueError(f"{label} contains an out-of-range index")
    return result


def predict_all_known(
    matrix: np.ndarray,
    probe_indices: Sequence[int],
    *,
    rank: int = 1,
    regularization: float = 0.1,
) -> ProbePredictions:
    """Predict every observed row in turn with all other rows visible."""

    actual = np.asarray(matrix, dtype=float)
    if actual.ndim != 2:
        raise ValueError("matrix must be 2D")
    probes = _indices(probe_indices, actual.shape[1], "probe_indices")
    observed = np.isfinite(actual)
    probe_columns = np.zeros(actual.shape[1], dtype=bool)
    probe_columns[list(probes)] = True
    revealed = observed & probe_columns[None, :]
    heldout = observed & ~probe_columns[None, :]
    predicted = np.full_like(actual, np.nan)
    predicted[revealed] = actual[revealed]
    for row in range(actual.shape[0]):
        hidden = heldout[row]
        if not hidden.any():
            continue
        train = actual.copy()
        train[row, ~probe_columns] = np.nan
        completed = complete(
            train,
            rank=rank,
            regularization=regularization,
            allow_empty_rows=True,
        )
        predicted[row, hidden] = completed[row, hidden]
    return ProbePredictions(probes, actual, predicted, observed, revealed, heldout)


def predict_heldout_models(
    matrix: np.ndarray,
    probe_indices: Sequence[int],
    target_model_indices: Sequence[int],
    context_model_indices: Sequence[int],
    *,
    rank: int = 1,
    regularization: float = 0.1,
) -> ProbePredictions:
    """Predict isolated target rows using fixed context rows and target probes."""

    actual = np.asarray(matrix, dtype=float)
    if actual.ndim != 2:
        raise ValueError("matrix must be 2D")
    probes = _indices(probe_indices, actual.shape[1], "probe_indices")
    targets = _indices(target_model_indices, actual.shape[0], "target_model_indices")
    context = _indices(context_model_indices, actual.shape[0], "context_model_indices")
    if set(targets) & set(context):
        raise ValueError("target and context model indices must be disjoint")
    observed = np.isfinite(actual)
    target_mask = np.zeros_like(observed)
    target_mask[list(targets)] = observed[list(targets)]
    probe_columns = np.zeros(actual.shape[1], dtype=bool)
    probe_columns[list(probes)] = True
    revealed = target_mask & probe_columns[None, :]
    heldout = target_mask & ~probe_columns[None, :]
    predicted = np.full_like(actual, np.nan)
    predicted[revealed] = actual[revealed]
    for row in targets:
        hidden = heldout[row]
        if not hidden.any():
            continue
        train = np.full_like(actual, np.nan)
        train[list(context)] = actual[list(context)]
        train[row, revealed[row]] = actual[row, revealed[row]]
        completed = complete(
            train,
            rank=rank,
            regularization=regularization,
            allow_empty_rows=True,
        )
        predicted[row, hidden] = completed[row, hidden]
    return ProbePredictions(probes, actual, predicted, target_mask, revealed, heldout)


def score_predictions(
    result: ProbePredictions,
    *,
    pairwise_margin: float = 2.0,
    top_fraction: float = 0.2,
) -> dict[str, float | int]:
    """Compute all probe objectives from one prediction artifact.

    MedAE and MedAPE use BenchPress's parity denominator, including measured
    probe cells as exact predictions.  MedAPE excludes zero-valued targets,
    matching the conventional percentage-error denominator guard.
    """

    target = result.target_mask
    if not np.isfinite(result.predicted[target]).all():
        raise ValueError("predictions do not cover every target cell")
    errors = np.abs(result.predicted - result.actual)
    parity_errors = errors[target]
    hidden_errors = errors[result.heldout_mask]
    nonzero = target & (np.abs(result.actual) > 1e-12)
    percentages = 100.0 * errors[nonzero] / np.abs(result.actual[nonzero])
    hidden_nonzero = result.heldout_mask & (np.abs(result.actual) > 1e-12)
    hidden_percentages = (
        100.0 * errors[hidden_nonzero] / np.abs(result.actual[hidden_nonzero])
    )
    pairwise = pairwise_ranking_accuracy(
        result.actual, result.predicted, result.heldout_mask, margin=pairwise_margin
    )
    top = top_fraction_recovery(
        result.actual,
        result.predicted,
        result.heldout_mask,
        top_fraction=top_fraction,
    )
    return {
        "n_target": int(target.sum()),
        "n_revealed": int(result.revealed_mask.sum()),
        "n_hidden": int(result.heldout_mask.sum()),
        "medae": float(np.median(parity_errors)) if parity_errors.size else float("nan"),
        "mae": float(np.mean(parity_errors)) if parity_errors.size else float("nan"),
        "medape": float(np.median(percentages)) if percentages.size else float("nan"),
        "hidden_medae": float(np.median(hidden_errors)) if hidden_errors.size else float("nan"),
        "hidden_medape": float(np.median(hidden_percentages)) if hidden_percentages.size else float("nan"),
        "pairwise_margin": float(pairwise_margin),
        "pairwise_n_pairs": pairwise.n_pairs,
        "pairwise_median_accuracy": pairwise.median_accuracy,
        "pairwise_pooled_accuracy": pairwise.pooled_accuracy,
        "top_fraction": float(top_fraction),
        "top_total_k": top.total_k,
        "top_median_recovery": top.median_recovery,
        "top_pooled_recovery": top.pooled_recovery,
    }


def objective_value(metrics: dict[str, float | int], objective: str) -> float:
    """Convert every supported objective to a deterministic minimization loss."""

    if objective in {"medae", "medape", "hidden_medae", "hidden_medape"}:
        return float(metrics[objective])
    if objective == "pairwise_margin_error":
        return 1.0 - float(metrics["pairwise_median_accuracy"])
    if objective == "top_fraction_error":
        return 1.0 - float(metrics["top_median_recovery"])
    raise ValueError(f"unknown objective: {objective}")


def candidate_prefixes(
    candidate_indices: Sequence[int], *, max_probes: int, repeats: int, seed: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Nested BenchPress random prefixes restricted to an explicit candidate set."""

    candidates = tuple(int(value) for value in candidate_indices)
    if max_probes < 0 or max_probes > len(candidates) or repeats < 1:
        raise ValueError("invalid prefix dimensions")
    output = []
    for repeat in range(repeats):
        rng = np.random.RandomState((int(seed) + repeat) * 100000)
        order = np.asarray(candidates, dtype=int)
        rng.shuffle(order)
        output.append(
            tuple(tuple(int(value) for value in order[:k]) for k in range(1, max_probes + 1))
        )
    return tuple(output)


def sharded_combinations(
    candidate_indices: Sequence[int],
    k: int,
    *,
    shard_index: int,
    num_shards: int,
    wave_index: int = 0,
    num_waves: int = 1,
) -> Iterable[tuple[int, tuple[int, ...]]]:
    """Yield deterministic upstream-style residue-partitioned combinations."""

    if num_shards < 1 or num_waves < 1:
        raise ValueError("num_shards and num_waves must be positive")
    if not 0 <= shard_index < num_shards or not 0 <= wave_index < num_waves:
        raise ValueError("shard_index or wave_index out of range")
    modulus = num_shards * num_waves
    residue = wave_index + num_waves * shard_index
    for ordinal, combination in enumerate(combinations(candidate_indices, k)):
        if ordinal % modulus == residue:
            yield ordinal, tuple(int(value) for value in combination)


def merge_shards(
    shards: Sequence[Sequence[tuple[int, tuple[int, ...]]]], expected_count: int
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Validate and merge sharded combination identities without prediction data."""

    merged = sorted((item for shard in shards for item in shard), key=lambda item: item[0])
    ordinals = [item[0] for item in merged]
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("duplicate exhaustive combination ordinal")
    if ordinals != list(range(expected_count)):
        raise ValueError("exhaustive shards are incomplete")
    return tuple(merged)
