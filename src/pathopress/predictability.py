"""BenchPress-style per-evaluation and per-model predictability summaries."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from pathopress.metrics import median_absolute_percentage_error


def prediction_error(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float | int]:
    """Return BenchPress's pooled MedAPE/MedAE score-error metrics."""

    a = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    valid = np.isfinite(a) & np.isfinite(p)
    a = a[valid]
    p = p[valid]
    if not a.size:
        return {"n": 0, "medape": float("nan"), "medae": float("nan")}
    absolute = np.abs(p - a)
    return {
        "n": int(absolute.size),
        "medape": median_absolute_percentage_error(a, p),
        "medae": float(np.median(absolute)),
    }


def holdout_half_per_model(
    matrix: np.ndarray,
    rng: np.random.RandomState,
    *,
    min_observed: int = 4,
) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Match BenchPress's hide-half-per-model split construction exactly."""

    train = np.asarray(matrix, dtype=float).copy()
    heldout = {i: [] for i in range(train.shape[0])}
    observed = np.isfinite(train)
    for i in range(train.shape[0]):
        columns = np.where(observed[i])[0]
        if len(columns) < min_observed:
            continue
        rng.shuffle(columns)
        for j in columns[: len(columns) // 2]:
            train[i, j] = np.nan
            heldout[i].append(int(j))
    return train, heldout


def aggregate_raw_predictions(
    raw_predictions: list[dict[str, float | int | str]],
    *,
    group_key: str,
    group_ids: list[str],
) -> list[dict[str, float | int | str | list[float]]]:
    """Aggregate raw seeded predictions by model or evaluation identifier."""

    grouped: dict[str, list[dict[str, float | int | str]]] = {
        group_id: [] for group_id in group_ids
    }
    for row in raw_predictions:
        grouped[str(row[group_key])].append(row)

    results = []
    for group_id in group_ids:
        rows = grouped[group_id]
        if not rows:
            continue
        pooled = prediction_error(
            (float(row["actual"]) for row in rows),
            (float(row["predicted"]) for row in rows),
        )
        if int(pooled["n"]) < 3 or not np.isfinite(float(pooled["medae"])):
            continue
        by_seed: dict[int, list[dict[str, float | int | str]]] = {}
        for row in rows:
            by_seed.setdefault(int(row["seed"]), []).append(row)
        seed_medae = []
        seed_medape = []
        for seed in sorted(by_seed):
            seed_rows = by_seed[seed]
            metric = prediction_error(
                (float(row["actual"]) for row in seed_rows),
                (float(row["predicted"]) for row in seed_rows),
            )
            if np.isfinite(float(metric["medae"])):
                seed_medae.append(float(metric["medae"]))
            if np.isfinite(float(metric["medape"])):
                seed_medape.append(float(metric["medape"]))
        results.append(
            {
                group_key: group_id,
                "n_test_cells": int(pooled["n"]),
                "n_seeds": len(by_seed),
                "medape": float(pooled["medape"]),
                "medae": float(pooled["medae"]),
                "seed_medapes": seed_medape,
                "seed_medaes": seed_medae,
            }
        )
    return results
