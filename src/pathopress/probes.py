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


# Consistency constant converting a median absolute deviation into a robust
# estimate of the standard deviation under normality.  Used instead of the raw
# sample SD because the observed columns carry a median of only ~7 models, and
# a 7-point SD is both noisy and outlier-dominated.
MAD_TO_SD_SCALE = 1.4826

# A column whose robust dispersion is below this many normalized-score points
# cannot support a skill estimate: any model error, however small in absolute
# terms, divides by a denominator that is itself inside the reporting noise of
# the published scores, so the resulting ratio is dominated by rounding rather
# than by model quality.  The published matrix has a per-column baseline MedAE
# first quartile of 1.10 points, so this threshold excludes only the tightest
# tail of columns rather than a meaningful share of the panel.  Columns below
# it are reported with an explicit exclusion flag, never with a silent NaN.
#
# CONFOUND -- read before quoting any fraction that uses this threshold.  The
# exclusion is NOT neutral with respect to the statistic it filters.  A column
# with low robust dispersion also has a small leave-one-out baseline MedAE,
# which is the denominator of the skill score, so such a column is close to
# structurally guaranteed to score negative skill: the completion model would
# have to beat an already-tiny error to come out positive.  Dropping these
# columns therefore removes near-certain negatives and pushes the reported
# positive fraction UP.  The measured size of that shift on the published
# matrix, from `scripts/replay_lofo_matched_cells.py`, is 86/174 = 49.4% with
# the exclusion versus 94/187 = 50.3% without it -- 8 of the 12 excluded
# columns are in fact positive.  Any headline built on this threshold must be
# reported next to the all-column fraction, never on its own.
SKILL_NOISE_FLOOR_DISPERSION = 0.5


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
    # Per-evaluation-column MedAE over that column's hidden target cells.
    # NaN for columns with no hidden target cell (probe columns, or columns
    # with no observation).  This is the numerator of the per-column skill
    # score, and is denominated identically to the per-column leave-one-out
    # baseline it is compared against.
    per_column_hidden_medae: tuple[float, ...] = ()


@dataclass(frozen=True)
class CandidateScore:
    probe_index: int
    objective_value: float


def family_blocked_model_split(
    model_ids: Sequence[str],
    *,
    model_metadata_path: str | None = None,
    seed: int = 42,
    train_fraction: float = 0.7,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, object]]:
    """Family-blocked model split (FIX 3).

    All models sharing a ``family`` in ``model_metadata.csv`` land entirely on
    one side of the split.  Models with a blank family are treated as singleton
    groups (not lumped together).  Uses sklearn ``GroupShuffleSplit`` for
    deterministic, reproducible group-level partitioning.

    Returns ``(train_indices, validation_indices, split_info)``.  ``split_info``
    records the families on each side for auditability.
    """

    try:
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError:
        raise ImportError(
            "family_blocked split mode requires scikit-learn; "
            "install with 'pip install scikit-learn'"
        ) from None

    families_by_model: dict[str, str] = {}
    if model_metadata_path is not None:
        from pathopress.model_metadata import load_model_metadata
        metadata = load_model_metadata(model_metadata_path)
        for mid in model_ids:
            row = metadata.get(mid)
            if row is not None:
                family = row.get("family", "").strip()
            else:
                family = ""
            families_by_model[mid] = family

    # Build group assignments: each unique family is its own group, blank
    # families are each treated as a singleton group keyed by model_id.
    model_to_group: dict[str, str] = {}
    for idx, mid in enumerate(model_ids):
        family = families_by_model.get(mid, "").strip()
        if family:
            model_to_group[mid] = family
        else:
            model_to_group[mid] = f"__singleton__{mid}"

    groups = np.array([model_to_group[mid] for mid in model_ids])
    indices = np.arange(len(model_ids))

    gss = GroupShuffleSplit(n_splits=1, test_size=1.0 - train_fraction, random_state=seed)
    for train_idx, val_idx in gss.split(indices, groups=groups):
        train_indices = tuple(sorted(int(i) for i in train_idx))
        validation_indices = tuple(sorted(int(i) for i in val_idx))

    train_families = sorted({model_to_group[model_ids[i]] for i in train_indices})
    val_families = sorted({model_to_group[model_ids[i]] for i in validation_indices})

    split_info: dict[str, object] = {
        "split_mode": "family_blocked",
        "seed": seed,
        "train_fraction": train_fraction,
        "train_model_ids": [model_ids[i] for i in train_indices],
        "validation_model_ids": [model_ids[i] for i in validation_indices],
        "train_families": train_families,
        "validation_families": val_families,
        "n_train_families": len(train_families),
        "n_validation_families": len(val_families),
    }

    return train_indices, validation_indices, split_info


def random_model_split(
    model_ids: Sequence[str],
    *,
    seed: int = 42,
    train_fraction: float = 0.7,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, object]]:
    """Random model split (legacy, for backward compatibility).

    Returns ``(train_indices, validation_indices, split_info)``.
    """

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(model_ids))
    n_train = min(max(1, round(train_fraction * len(model_ids))), len(model_ids) - 1)
    train_indices = tuple(sorted(int(i) for i in order[:n_train]))
    validation_indices = tuple(sorted(int(i) for i in order[n_train:]))

    split_info: dict[str, object] = {
        "split_mode": "random",
        "seed": seed,
        "train_fraction": train_fraction,
        "train_model_ids": [model_ids[i] for i in train_indices],
        "validation_model_ids": [model_ids[i] for i in validation_indices],
    }

    return train_indices, validation_indices, split_info


@dataclass(frozen=True)
class FamilyFold:
    """One leave-one-family-out cross-validation fold."""

    fold: int
    family: str
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


def model_family_groups(
    model_ids: Sequence[str],
    *,
    model_metadata_path: str | None = None,
) -> dict[str, str]:
    """Map each model id to its blocking group.

    Models sharing a non-blank ``family`` in ``model_metadata.csv`` share a
    group.  Blank families become singleton groups keyed by model id so that
    unrelated single-model entries are never lumped together.
    """

    families_by_model: dict[str, str] = {}
    if model_metadata_path is not None:
        from pathopress.model_metadata import load_model_metadata
        metadata = load_model_metadata(model_metadata_path)
        for mid in model_ids:
            row = metadata.get(mid)
            families_by_model[mid] = (
                row.get("family", "").strip() if row is not None else ""
            )

    model_to_group: dict[str, str] = {}
    for mid in model_ids:
        family = families_by_model.get(mid, "").strip()
        model_to_group[mid] = family if family else f"__singleton__{mid}"
    return model_to_group


def leave_one_family_out_folds(
    model_ids: Sequence[str],
    *,
    model_metadata_path: str | None = None,
) -> tuple[tuple[FamilyFold, ...], dict[str, object]]:
    """Leave-one-family-out cross-validation over model families (FIX 3).

    A single 70/30 ``GroupShuffleSplit`` holdout leaves very few independent
    validation models (11 of 59 at seed 42, all of them singleton families),
    and the count swings with the seed.  Leave-one-family-out removes the seed
    lottery entirely: every family is held out exactly once, so every model is
    validated exactly once and no family is ever on both sides of a fold.

    Returns ``(folds, info)``.  ``info`` records per-fold and aggregate
    validation sizes for the artifact.
    """

    model_to_group = model_family_groups(
        model_ids, model_metadata_path=model_metadata_path
    )
    groups = [model_to_group[mid] for mid in model_ids]
    ordered_families = sorted(set(groups))

    folds: list[FamilyFold] = []
    for fold_index, family in enumerate(ordered_families):
        validation = tuple(
            index for index, group in enumerate(groups) if group == family
        )
        train = tuple(
            index for index, group in enumerate(groups) if group != family
        )
        if not train:
            raise ValueError("leave-one-family-out requires at least two families")
        folds.append(
            FamilyFold(
                fold=fold_index,
                family=family,
                train_indices=train,
                validation_indices=validation,
            )
        )

    validation_sizes = [len(fold.validation_indices) for fold in folds]
    info: dict[str, object] = {
        "split_mode": "leave_one_family_out",
        "n_folds": len(folds),
        "n_models": len(model_ids),
        "per_fold": [
            {
                "fold": fold.fold,
                "family": fold.family,
                "n_train_models": len(fold.train_indices),
                "n_validation_models": len(fold.validation_indices),
                "validation_model_ids": [model_ids[i] for i in fold.validation_indices],
            }
            for fold in folds
        ],
        "aggregate_validation_models": int(sum(validation_sizes)),
        "min_fold_validation_models": int(min(validation_sizes)) if folds else 0,
        "max_fold_validation_models": int(max(validation_sizes)) if folds else 0,
        "median_fold_validation_models": (
            float(np.median(validation_sizes)) if folds else float("nan")
        ),
    }
    return tuple(folds), info


@dataclass(frozen=True)
class ColumnSkill:
    """Leave-one-out error ratio and bounded skill for one evaluation column.

    ``medae_ratio`` -- ``MedAE_model / MedAE_LOO_baseline`` -- is the primary
    field.  It has no pole at the origin and is symmetric on a log scale, so it
    is the quantity to compare across columns.  ``skill_score_raw`` is the
    unbounded ``1 - medae_ratio`` form retained verbatim for auditability, and
    ``skill_score`` is that same value clipped to ``[-1, 1]`` for reporting.
    All three are ``None`` when the column is excluded.
    """

    column: int
    model_medae: float
    loo_baseline_medae: float
    robust_dispersion: float
    excluded_below_noise_floor: bool
    medae_ratio: float | None
    skill_score_raw: float | None
    skill_score: float | None
    # Why the column carries no skill value: ``"noise_floor"`` (dispersion
    # below the floor), ``"degenerate_baseline"`` (no leave-one-out baseline
    # is defined), ``"no_model_error"`` (the column has no scored target
    # cell, e.g. it is itself a probe), or ``None`` when scored.
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class SkillFractionSummary:
    """Headline skill statistic: share of columns the model actually beats.

    A mean or median over ``skill_score_raw`` is not reportable because that
    quantity is unbounded below (a fixed 2-point error is skill -21 on a tight
    column and +0.9 on a wide one), so a single pathological column dominates
    the aggregate.  The fraction of columns with positive skill is bounded in
    ``[0, 1]`` by construction and is what this summary reports.
    """

    n_columns_total: int
    n_columns_excluded: int
    n_columns_scored: int
    n_columns_positive: int
    fraction_positive: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int
    seed: int


def compute_column_median_baseline_medae(matrix: np.ndarray, column: int) -> float:
    """In-sample per-column column-median-prediction MedAE (legacy).

    For a single column ``c``, predict every observed cell in that column with
    the column's own median -- computed from the **whole** column, including
    the cell being predicted -- and return the median absolute error.

    This is an oracle baseline: with a median of only ~7 observations per
    column, a one-parameter constant fitted in sample on those same 7 points is
    an unfairly strong opponent that the completion model is never granted.
    It is kept only so previously published artifacts stay reproducible; the
    skill score now uses :func:`compute_column_loo_baseline_medae` instead.
    """

    values = _validate_matrix(matrix)
    col_median = float(np.nanmedian(values[:, column]))
    col_observed = np.isfinite(values[:, column])
    col_errors = np.abs(values[col_observed, column] - col_median)
    if col_errors.size == 0:
        return float("nan")
    return float(np.median(col_errors))


def compute_column_loo_baseline_medae(matrix: np.ndarray, column: int) -> float:
    """Leave-one-out per-column column-median-prediction MedAE (FIX 1).

    Each observed cell in the column is predicted by the median of the column
    computed **without** that cell, exactly mirroring the information the
    completion model is given when the same cell is its target.  Returns the
    median of those absolute errors, or NaN when the column has fewer than two
    observations (no leave-one-out prediction is defined).
    """

    values = _validate_matrix(matrix)
    finite = values[np.isfinite(values[:, column]), column]
    if finite.size < 2:
        return float("nan")
    errors = [
        abs(float(finite[i]) - float(np.median(np.delete(finite, i))))
        for i in range(finite.size)
    ]
    return float(np.median(errors))


def compute_column_robust_dispersion(matrix: np.ndarray, column: int) -> float:
    """Robust column dispersion: MAD scaled to a standard-deviation estimate.

    ``MAD * 1.4826`` is used in place of the sample SD because the observed
    columns hold a median of only ~7 models, where an SD is both high-variance
    and dominated by a single outlying model.  This is the quantity compared
    against :data:`SKILL_NOISE_FLOOR_DISPERSION`.
    """

    values = _validate_matrix(matrix)
    finite = values[np.isfinite(values[:, column]), column]
    if finite.size < 2:
        return float("nan")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return MAD_TO_SD_SCALE * mad


def compute_column_skill(
    matrix: np.ndarray,
    column: int,
    model_medae: float,
    *,
    noise_floor: float = SKILL_NOISE_FLOOR_DISPERSION,
) -> ColumnSkill:
    """Bounded leave-one-out skill score for one column (FIX 1).

    ``skill = 1 - MedAE_model / MedAE_LOO_baseline``, clipped to ``[-1, 1]``.
    Columns whose robust dispersion is below ``noise_floor``, or whose
    leave-one-out baseline is non-finite or zero, are flagged excluded and
    carry no skill value at all rather than an unbounded or NaN number.
    """

    loo_baseline = compute_column_loo_baseline_medae(matrix, column)
    dispersion = compute_column_robust_dispersion(matrix, column)
    reason: str | None = None
    if not np.isfinite(model_medae):
        reason = "no_model_error"
    elif not np.isfinite(loo_baseline) or loo_baseline <= 0.0:
        reason = "degenerate_baseline"
    elif not np.isfinite(dispersion) or dispersion < noise_floor:
        reason = "noise_floor"
    if reason is not None:
        return ColumnSkill(
            column=int(column),
            model_medae=float(model_medae),
            loo_baseline_medae=float(loo_baseline),
            robust_dispersion=float(dispersion),
            excluded_below_noise_floor=True,
            medae_ratio=None,
            skill_score_raw=None,
            skill_score=None,
            exclusion_reason=reason,
        )
    ratio = float(model_medae) / float(loo_baseline)
    raw = 1.0 - ratio
    return ColumnSkill(
        column=int(column),
        model_medae=float(model_medae),
        loo_baseline_medae=float(loo_baseline),
        robust_dispersion=float(dispersion),
        excluded_below_noise_floor=False,
        medae_ratio=ratio,
        skill_score_raw=float(raw),
        skill_score=float(np.clip(raw, -1.0, 1.0)),
    )


def summarize_skill_positive_fraction(
    column_skills: Sequence[ColumnSkill],
    *,
    n_bootstrap: int = 10000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> SkillFractionSummary:
    """Fraction of scored columns with positive skill, with a bootstrap CI.

    This is the headline skill statistic.  The bootstrap resamples columns
    (the independent units) with replacement and reports the percentile
    interval at ``ci_level``.  Excluded columns never enter the denominator.
    """

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be strictly between 0 and 1")

    scored = [
        skill for skill in column_skills if not skill.excluded_below_noise_floor
    ]
    n_total = len(column_skills)
    n_scored = len(scored)
    n_excluded = n_total - n_scored
    if not scored:
        return SkillFractionSummary(
            n_columns_total=n_total,
            n_columns_excluded=n_excluded,
            n_columns_scored=0,
            n_columns_positive=0,
            fraction_positive=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            ci_level=float(ci_level),
            n_bootstrap=int(n_bootstrap),
            seed=int(seed),
        )

    positives = np.asarray(
        [1.0 if float(skill.skill_score) > 0.0 else 0.0 for skill in scored],
        dtype=float,
    )
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, n_scored, size=(int(n_bootstrap), n_scored))
    resampled = positives[draws].mean(axis=1)
    tail = (1.0 - ci_level) / 2.0
    lower, upper = np.percentile(resampled, [100.0 * tail, 100.0 * (1.0 - tail)])
    return SkillFractionSummary(
        n_columns_total=n_total,
        n_columns_excluded=n_excluded,
        n_columns_scored=n_scored,
        n_columns_positive=int(positives.sum()),
        fraction_positive=float(positives.mean()),
        ci_lower=float(lower),
        ci_upper=float(upper),
        ci_level=float(ci_level),
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
    )


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

    per_column_hidden_medae = tuple(
        float(np.median(absolute[hidden[:, col], col]))
        if hidden[:, col].any()
        else float("nan")
        for col in range(matrix.shape[1])
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
        per_column_hidden_medae=per_column_hidden_medae,
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
