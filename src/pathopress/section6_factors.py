"""Primitives for BenchPress Section 6 prediction-error factor experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

from pathopress.completion import complete
from pathopress.predictability import prediction_error


def supported_complete(
    training: np.ndarray, *, rank: int = 1, regularization: float = 0.1
) -> np.ndarray:
    """Complete only rows and columns retaining training evidence."""

    values = np.asarray(training, dtype=float)
    predicted = np.full_like(values, np.nan)
    rows = np.flatnonzero(np.any(np.isfinite(values), axis=1))
    columns = np.flatnonzero(np.any(np.isfinite(values), axis=0))
    if not rows.size or not columns.size:
        return predicted
    completed = complete(
        values[np.ix_(rows, columns)],
        rank=rank,
        regularization=regularization,
    )
    predicted[np.ix_(rows, columns)] = completed
    return predicted


def holdout_half_per_benchmark(
    matrix: np.ndarray,
    benchmark_index: int,
    rng: np.random.RandomState,
    *,
    min_test: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Match BenchPress's target-column hide-half construction."""

    observed = np.flatnonzero(np.isfinite(matrix[:, benchmark_index]))
    permutation = rng.permutation(len(observed))
    n_test = max(int(min_test), len(observed) // 2)
    return observed[permutation[:n_test]], observed[permutation[n_test:]]


def paired_error_record(
    actual: Sequence[float],
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    min_predictions: int = 2,
) -> dict[str, float | int] | None:
    """Compute paired MedAE/MedAPE on the common finite prediction set."""

    actual_array = np.asarray(actual, dtype=float)
    base_array = np.asarray(baseline, dtype=float)
    treat_array = np.asarray(treatment, dtype=float)
    valid = np.isfinite(actual_array) & np.isfinite(base_array) & np.isfinite(treat_array)
    if int(valid.sum()) < min_predictions:
        return None
    base = prediction_error(actual_array[valid], base_array[valid])
    treat = prediction_error(actual_array[valid], treat_array[valid])
    result: dict[str, float | int] = {"n_test": int(valid.sum())}
    for metric in ("medape", "medae"):
        result[f"base_{metric}"] = float(base[metric])
        result[f"treat_{metric}"] = float(treat[metric])
        result[f"delta_{metric}"] = float(treat[metric]) - float(base[metric])
    return result


def wilcoxon_signed_rank(
    deltas: Iterable[float],
    *,
    min_n: int = 5,
    drop_zeros_for_test: bool = False,
) -> dict[str, float | int]:
    """BenchPress-compatible two-sided signed-rank summary."""

    from scipy import stats

    finite = np.asarray([float(value) for value in deltas], dtype=float)
    finite = finite[np.isfinite(finite)]
    tested = finite[finite != 0] if drop_zeros_for_test else finite
    median = float(np.median(finite)) if finite.size else float("nan")
    if tested.size < min_n or not np.any(tested != 0):
        p_value = 1.0 if finite.size and not np.any(finite != 0) else float("nan")
    else:
        _, p_value = stats.wilcoxon(tested, alternative="two-sided")
    return {
        "median_delta": median,
        "p_value": float(p_value),
        "n": int(finite.size),
        "n_positive": int(np.sum(finite > 0)),
        "n_negative": int(np.sum(finite < 0)),
        "n_zero": int(np.sum(finite == 0)),
    }


def grouped_wilcoxon(
    records: Sequence[dict[str, object]],
    *,
    group_key: str,
    drop_zeros_for_test: bool = False,
) -> dict[str, dict[str, float | int]]:
    """Test one median seed-level treatment delta per benchmark/model."""

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record[group_key])].append(record)
    result = {}
    for metric in ("medape", "medae"):
        per_group = []
        for rows in grouped.values():
            values = np.asarray(
                [float(row[f"delta_{metric}"]) for row in rows], dtype=float
            )
            values = values[np.isfinite(values)]
            if values.size:
                per_group.append(float(np.median(values)))
        result[metric] = wilcoxon_signed_rank(
            per_group,
            drop_zeros_for_test=drop_zeros_for_test,
        )
    return result


def pooled_model_metrics(
    raw_rows: Sequence[dict[str, object]],
    *,
    condition: str,
) -> dict[str, dict[str, float | int]]:
    """Pool seeded predictions by model, as BenchPress model H5–H8 do."""

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        if row["condition"] == condition:
            grouped[str(row["model_id"])].append(row)
    result = {}
    for model_id, rows in grouped.items():
        metrics = prediction_error(
            (float(row["actual"]) for row in rows),
            (float(row["predicted"]) for row in rows),
        )
        if int(metrics["n"]) >= 3:
            result[model_id] = metrics
    return result


def paired_model_wilcoxon(
    baseline: dict[str, dict[str, float | int]],
    treatment: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    common = sorted(set(baseline) & set(treatment))
    return {
        metric: wilcoxon_signed_rank(
            float(treatment[model][metric]) - float(baseline[model][metric])
            for model in common
        )
        for metric in ("medape", "medae")
    }
