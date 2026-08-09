"""BenchPress-compatible probe prediction plus pathology compression metrics.

The predictor preserves the upstream all-known and isolated-heldout masking
semantics.  Metric computation is kept separate so one expensive completion
can support MedAE, MedAPE, pairwise-margin, and top-fraction objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np

from pathopress.completion import complete
from pathopress.metrics import absolute_percentage_errors
from pathopress.ranking import pairwise_ranking_accuracy, top_fraction_recovery


# This default is attached to score-reconstruction outputs only as an ancillary
# ordering diagnostic.  Dedicated ranking-aware probe selection supplies its
# own explicit margin (5 normalized-score points in the public experiment).
SCORE_RECONSTRUCTION_PAIRWISE_DIAGNOSTIC_MARGIN = 2.0

# Consistency constant turning a median absolute deviation into a robust
# standard-deviation estimate.  Mirrors ``pathopress.probes.MAD_TO_SD_SCALE``;
# the observed columns hold a median of only ~7 models, where a sample SD is
# high-variance and outlier-dominated.
MAD_TO_SD_SCALE = 1.4826


@dataclass(frozen=True)
class ProbePredictions:
    """Predictions plus the dispersion estimates their metrics normalize by.

    ``column_sd`` is the legacy per-column sample SD.  In the all-known track
    it is computed over the full matrix, which includes each target cell's own
    value even though that value is hidden while the cell is being predicted;
    in the held-out track it is computed over context rows only.  The two are
    therefore **not** comparable, and the ``medae_normalized*`` keys derived
    from ``column_sd`` are retained only so previously published artifacts stay
    reproducible.

    ``column_dispersion_by_cell`` (FIX 2) is the leakage-free replacement and
    has the shape of the matrix.  For every cell it holds the robust dispersion
    (``MAD * 1.4826``) of that cell's column computed from exactly the rows a
    predictor legitimately sees when that cell is its target -- the target row's
    own observation is never included on either track.  The ``*_loo`` metric
    keys are derived from it and are comparable across tracks.
    """

    probe_indices: tuple[int, ...]
    actual: np.ndarray
    predicted: np.ndarray
    target_mask: np.ndarray
    revealed_mask: np.ndarray
    heldout_mask: np.ndarray
    column_sd: np.ndarray | None = None
    column_dispersion_by_cell: np.ndarray | None = None


# Leave-target-out dispersion depends only on the score matrix, which greedy
# selection holds fixed across thousands of candidate probe sets.  A tiny
# content-keyed cache keeps that from being recomputed every call.
_DISPERSION_CACHE: dict[tuple[tuple[int, ...], bytes], np.ndarray] = {}
_DISPERSION_CACHE_MAX = 8


def _robust_dispersion(values: np.ndarray) -> float:
    """``MAD * 1.4826`` over finite values; NaN when fewer than two remain."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    median = float(np.median(finite))
    return MAD_TO_SD_SCALE * float(np.median(np.abs(finite - median)))


def _leave_target_out_dispersion(actual: np.ndarray) -> np.ndarray:
    """All-known dispersion: per cell, its column excluding that cell's row.

    In the all-known protocol every other row stays fully visible while one
    target row is masked, so the legitimately-available dispersion estimate for
    cell ``(r, c)`` is column ``c`` over all observed rows except ``r``.

    Depends only on ``actual``, which greedy selection holds fixed across every
    candidate probe set, so the result is memoized on the matrix contents.
    """

    key = (actual.shape, actual.tobytes())
    cached = _DISPERSION_CACHE.get(key)
    if cached is not None:
        return cached

    dispersion = np.full(actual.shape, np.nan, dtype=float)
    for col in range(actual.shape[1]):
        column = actual[:, col]
        observed = np.isfinite(column)
        # Rows with no observation in this column drop nothing when removed.
        missing_value = _robust_dispersion(column)
        dispersion[~observed, col] = missing_value
        for row in np.flatnonzero(observed):
            dispersion[row, col] = _robust_dispersion(np.delete(column, row))

    if len(_DISPERSION_CACHE) >= _DISPERSION_CACHE_MAX:
        _DISPERSION_CACHE.clear()
    _DISPERSION_CACHE[key] = dispersion
    return dispersion


def _context_dispersion(actual: np.ndarray, context: Sequence[int]) -> np.ndarray:
    """Held-out dispersion: per cell, its column over context rows only.

    Target rows are entirely unseen on this track, so excluding all of them --
    not merely the one being predicted -- is what the protocol makes available.
    The result is broadcast to matrix shape so both tracks expose the same
    per-cell interface.
    """

    dispersion = np.full(actual.shape, np.nan, dtype=float)
    if not context:
        return dispersion
    for col in range(actual.shape[1]):
        dispersion[:, col] = _robust_dispersion(actual[list(context), col])
    return dispersion


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
    """Predict every observed row in turn with all other rows visible.

    Emits both dispersion estimates: the legacy full-matrix ``column_sd`` and
    the leakage-free ``column_dispersion_by_cell``, which drops each target
    cell's own row from its column before estimating dispersion.
    """

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
    # Legacy per-column SD over the full observed matrix.  This DOES include
    # each target cell's own value, which is hidden while that cell is being
    # predicted, so it is retained for artifact compatibility only.
    column_sd = np.asarray([
        float(np.std(actual[np.isfinite(actual[:, c]), c], ddof=0))
        if np.isfinite(actual[:, c]).sum() > 1 else 0.0
        for c in range(actual.shape[1])
    ])
    return ProbePredictions(
        probes,
        actual,
        predicted,
        observed,
        revealed,
        heldout,
        column_sd,
        _leave_target_out_dispersion(actual),
    )


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

    # Column SD from context rows only (leakage-safe: target rows are held-out
    # models, so we must NOT include their observations in the column
    # dispersion estimate.  Context rows are the training-set models that are
    # genuinely available during probe selection and prediction.)
    col_sd = np.full(actual.shape[1], np.nan)
    for col in range(actual.shape[1]):
        if context:
            col_vals = actual[list(context), col]
            col_finite = col_vals[np.isfinite(col_vals)]
        else:
            col_finite = np.array([])
        if col_finite.size > 1:
            col_sd[col] = float(np.std(col_finite, ddof=0))
        elif col_finite.size == 1:
            col_sd[col] = 0.0

    return ProbePredictions(
        probes,
        actual,
        predicted,
        target_mask,
        revealed,
        heldout,
        col_sd,
        _context_dispersion(actual, context),
    )


def score_predictions(
    result: ProbePredictions,
    *,
    pairwise_margin: float = SCORE_RECONSTRUCTION_PAIRWISE_DIAGNOSTIC_MARGIN,
    top_fraction: float = 0.2,
    ranking_scope: str = "at_least_one_hidden",
) -> dict[str, float | int | str]:
    """Compute all probe objectives from one prediction artifact.

    MedAE and MedAPE use BenchPress's parity denominator, including measured
    probe cells as exact predictions. MedAPE excludes targets with absolute
    value at most ``1e-6``, matching the pinned upstream denominator guard and
    avoiding unstable percentage errors near zero.

    ``pairwise_margin`` is a generic computation argument retained for API and
    artifact compatibility.  Its default value of 2 is only an ancillary
    diagnostic on score-reconstruction curves.  Ranking-aware selection must
    pass its objective margin explicitly; the public ranking contract uses 5.

    Two families of dispersion-normalized keys are emitted.  ``medae_normalized``
    and ``medae_normalized_pooled`` divide by the legacy per-column sample SD;
    on the all-known track that SD is computed over the full matrix and so
    includes the very values being predicted, while on the held-out track it is
    computed over context rows only.  Those two keys are therefore NOT
    comparable across tracks and exist for artifact compatibility.
    ``medae_normalized_loo`` and ``medae_normalized_pooled_loo`` divide by
    ``column_dispersion_by_cell``, which excludes each target cell's own row on
    both tracks, and are the comparable pair.
    """

    target = result.target_mask
    if not np.isfinite(result.predicted[target]).all():
        raise ValueError("predictions do not cover every target cell")
    errors = np.abs(result.predicted - result.actual)
    parity_errors = errors[target]
    hidden_errors = errors[result.heldout_mask]
    percentages = absolute_percentage_errors(
        result.actual[target], result.predicted[target]
    )
    hidden_percentages = absolute_percentage_errors(
        result.actual[result.heldout_mask], result.predicted[result.heldout_mask]
    )
    if ranking_scope == "at_least_one_hidden":
        ranking_actual = result.actual
        ranking_predicted = result.predicted
        ranking_heldout = result.heldout_mask
    elif ranking_scope == "all_target":
        # BenchPress ranking-aware all-known and with-probe-zero validation
        # flatten every reported target and mark every one as evaluated.  This
        # deliberately retains probe/probe pairs in the denominator.
        ranking_actual = np.where(target, result.actual, np.nan)
        ranking_predicted = np.where(target, result.predicted, np.nan)
        ranking_heldout = target
    elif ranking_scope == "hidden_only":
        # BenchPress non-probe holdout validation omits probe targets from the
        # prediction list entirely, so pairs are formed only among hidden
        # target cells (not hidden/probe mixtures).
        ranking_actual = np.where(result.heldout_mask, result.actual, np.nan)
        ranking_predicted = np.where(result.heldout_mask, result.predicted, np.nan)
        ranking_heldout = result.heldout_mask
    else:
        raise ValueError(
            "ranking_scope must be one of 'at_least_one_hidden', "
            "'all_target', or 'hidden_only'"
        )
    pairwise = pairwise_ranking_accuracy(
        ranking_actual, ranking_predicted, ranking_heldout, margin=pairwise_margin
    )
    top = top_fraction_recovery(
        ranking_actual,
        ranking_predicted,
        ranking_heldout,
        top_fraction=top_fraction,
    )
    # Dispersion-normalized metrics (FIX 2).  Uses column_sd from the
    # ProbePredictions artifact.  column_sd is computed from the observed
    # context matrix only — for all_known that's the full matrix, for
    # heldout models that's the context rows only.  Both keys are omitted
    # entirely when no finite value is available to prevent NaN from leaking
    # into JSON-serialized artifacts.
    medae_normalized = None
    medae_normalized_pooled = None
    if result.column_sd is not None:
        # Per-cell normalized error: |pred - actual| / column_sd for that column.
        norm_errors = np.full_like(errors, np.nan)
        for col in range(errors.shape[1]):
            sd = result.column_sd[col]
            if np.isfinite(sd) and sd > 0:
                norm_errors[:, col] = errors[:, col] / sd
        # Per-cell pooled: median of all finite normalized errors over target cells.
        flat_norm = norm_errors[target]
        finite_flat = flat_norm[np.isfinite(flat_norm)]
        if finite_flat.size:
            medae_normalized = float(np.median(finite_flat))
        # Per-column pooled: median over columns of (per-column medae / column_sd).
        col_medae_norms: list[float] = []
        for col in range(errors.shape[1]):
            col_target = target[:, col]
            if col_target.any():
                col_medae = float(np.median(errors[col_target, col]))
                sd = result.column_sd[col]
                if np.isfinite(sd) and sd > 0:
                    col_medae_norms.append(col_medae / sd)
        if col_medae_norms:
            medae_normalized_pooled = float(np.median(col_medae_norms))

    # Leakage-free dispersion normalization (FIX 2).  ``column_dispersion_by_cell``
    # never includes a target cell's own row, on either track, so the ``*_loo``
    # keys ARE comparable between all-known and held-out runs — unlike the
    # ``medae_normalized*`` keys above, whose all-known denominator is computed
    # over the full matrix including the very cells being predicted.
    medae_normalized_loo = None
    medae_normalized_pooled_loo = None
    if result.column_dispersion_by_cell is not None:
        dispersion = np.asarray(result.column_dispersion_by_cell, dtype=float)
        if dispersion.shape != errors.shape:
            raise ValueError("column_dispersion_by_cell must have the matrix shape")
        usable = target & np.isfinite(dispersion) & (dispersion > 0)
        if usable.any():
            medae_normalized_loo = float(
                np.median(errors[usable] / dispersion[usable])
            )
            col_loo_norms: list[float] = []
            for col in range(errors.shape[1]):
                col_usable = usable[:, col]
                if col_usable.any():
                    col_loo_norms.append(
                        float(
                            np.median(
                                errors[col_usable, col] / dispersion[col_usable, col]
                            )
                        )
                    )
            if col_loo_norms:
                medae_normalized_pooled_loo = float(np.median(col_loo_norms))

    return_dict: dict[str, float | int | str] = {
        "n_target": int(target.sum()),
        "n_revealed": int(result.revealed_mask.sum()),
        "n_hidden": int(result.heldout_mask.sum()),
        "medae": float(np.median(parity_errors)) if parity_errors.size else float("nan"),
        "mae": float(np.mean(parity_errors)) if parity_errors.size else float("nan"),
        "medape": float(np.median(percentages)) if percentages.size else float("nan"),
        "hidden_medae": float(np.median(hidden_errors)) if hidden_errors.size else float("nan"),
        "hidden_medape": float(np.median(hidden_percentages)) if hidden_percentages.size else float("nan"),
        "ranking_scope": ranking_scope,
        "pairwise_margin": float(pairwise_margin),
        "pairwise_n_pairs": pairwise.n_pairs,
        "pairwise_median_accuracy": pairwise.median_accuracy,
        "pairwise_pooled_accuracy": pairwise.pooled_accuracy,
        "top_fraction": float(top_fraction),
        "top_total_k": top.total_k,
        "top_median_recovery": top.median_recovery,
        "top_pooled_recovery": top.pooled_recovery,
    }
    if medae_normalized is not None and np.isfinite(medae_normalized):
        return_dict["medae_normalized"] = medae_normalized
    if medae_normalized_pooled is not None and np.isfinite(medae_normalized_pooled):
        return_dict["medae_normalized_pooled"] = medae_normalized_pooled
    if medae_normalized_loo is not None and np.isfinite(medae_normalized_loo):
        return_dict["medae_normalized_loo"] = medae_normalized_loo
    if medae_normalized_pooled_loo is not None and np.isfinite(
        medae_normalized_pooled_loo
    ):
        return_dict["medae_normalized_pooled_loo"] = medae_normalized_pooled_loo
    return return_dict


def objective_value(metrics: dict[str, float | int | str | None], objective: str) -> float:
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


def rank_prune_trajectory(
    trajectory: Sequence[dict[str, Any]],
    candidate_ids: Sequence[str],
    *,
    keep_count: int,
    max_steps: int | None = None,
    score_key: str = "score",
) -> dict[str, Any]:
    """Apply BenchPress's exact aggregate-normalized-rank pruning rule.

    Candidates are ranked independently inside every greedy context by
    ``(score, candidate_id)``.  Their zero-based rank is normalized by
    ``max(1, n_candidates - 1)`` and averaged only over contexts where the
    candidate still appears.  A candidate selected by greedy is therefore
    absent from later contexts, matching the upstream trajectory contract.
    """

    ids = [str(value) for value in candidate_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("candidate_ids must be non-empty and unique")
    if not 0 < keep_count < len(ids):
        raise ValueError("keep_count must be between 1 and len(candidate_ids)-1")
    steps = list(trajectory)
    if not steps:
        raise ValueError("trajectory must contain at least one greedy step")
    if max_steps is not None:
        if max_steps <= 0 or len(steps) < max_steps:
            raise ValueError("max_steps must be positive and available in trajectory")
        steps = steps[:max_steps]

    known = set(ids)
    by_candidate: dict[str, dict[str, Any]] = {
        candidate_id: {
            "candidate_id": candidate_id,
            "selected_by_greedy_step": None,
            "rank_records": [],
        }
        for candidate_id in ids
    }
    ranked_steps: list[dict[str, Any]] = []
    for position, source_step in enumerate(steps, start=1):
        step_number = int(source_step.get("step", source_step.get("k", position)))
        added = source_step.get(
            "added_evaluation_id", source_step.get("added_benchmark")
        )
        if added in by_candidate:
            by_candidate[str(added)]["selected_by_greedy_step"] = step_number
        source_records = source_step.get("candidate_results", [])
        if isinstance(source_records, dict):
            records = [
                {"candidate_id": candidate_id, **record}
                for candidate_id, record in source_records.items()
            ]
        else:
            records = list(source_records)
        ranked_input: list[tuple[str, float]] = []
        for record in records:
            candidate_id = str(
                record.get(
                    "candidate_id",
                    record.get("evaluation_id", record.get("benchmark_id", "")),
                )
            )
            if candidate_id not in known:
                raise ValueError(f"unknown candidate in source trajectory: {candidate_id}")
            raw_score = record.get(score_key)
            score = float("inf") if raw_score is None else float(raw_score)
            ranked_input.append((candidate_id, score))
        if len(ranked_input) != len({candidate_id for candidate_id, _ in ranked_input}):
            raise ValueError(f"duplicate candidate in greedy step {step_number}")
        ranked_input.sort(key=lambda item: (item[1], item[0]))
        denominator = max(1, len(ranked_input) - 1)
        ranks = []
        for rank, (candidate_id, score) in enumerate(ranked_input, start=1):
            record = {
                "candidate_id": candidate_id,
                "score": score if np.isfinite(score) else None,
                "rank": rank,
                "n_candidates": len(ranked_input),
                "rank_percentile": float((rank - 1) / denominator),
            }
            ranks.append(record)
            by_candidate[candidate_id]["rank_records"].append(
                {
                    "source_greedy_step": step_number,
                    "rank": rank,
                    "n_candidates": len(ranked_input),
                    "rank_percentile": record["rank_percentile"],
                    "score": record["score"],
                }
            )
        ranked_steps.append(
            {
                "source_greedy_step": step_number,
                "added_candidate_id": added,
                "n_candidates": len(ranks),
                "ranks": ranks,
            }
        )

    for record in by_candidate.values():
        rank_records = record["rank_records"]
        record["n_ranked_steps"] = len(rank_records)
        record["avg_rank_percentile"] = (
            float(np.mean([item["rank_percentile"] for item in rank_records]))
            if rank_records
            else 1.0
        )
        record["avg_rank"] = (
            float(np.mean([item["rank"] for item in rank_records]))
            if rank_records
            else None
        )
    ranked_ids = sorted(
        ids,
        key=lambda candidate_id: (
            by_candidate[candidate_id]["avg_rank_percentile"],
            by_candidate[candidate_id]["avg_rank"]
            if by_candidate[candidate_id]["avg_rank"] is not None
            else float("inf"),
            candidate_id,
        ),
    )
    kept_ids = ranked_ids[:keep_count]
    kept_set = set(kept_ids)
    for aggregate_rank, candidate_id in enumerate(ranked_ids, start=1):
        by_candidate[candidate_id]["aggregate_rank"] = aggregate_rank
        by_candidate[candidate_id]["decision"] = (
            "keep" if candidate_id in kept_set else "remove"
        )
    return {
        "candidate_ids": ids,
        "ranked_steps": ranked_steps,
        "by_candidate": by_candidate,
        "kept_ids": kept_ids,
        "removed_ids": ranked_ids[keep_count:],
        "keep_count": keep_count,
        "source_steps_used": len(steps),
    }


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
