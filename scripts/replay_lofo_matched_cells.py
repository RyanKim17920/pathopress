#!/usr/bin/env python3
"""Offline matched-cell replay of the leave-one-family-out probe experiment.

WHY THIS SCRIPT EXISTS
----------------------
The published leave-one-family-out (LOFO) artifacts report a held-out MedAE for
three arms -- the ``k=0`` no-probe baseline, the greedy probe prefix, and the
random probe prefix -- but each arm was scored on a *different* set of cells.
A probe cell is revealed (and therefore removed from the hidden denominator) by
whichever arm selected it, so the greedy arm and the random arm never answered
the same question.  Worse, the arms differ in *which* columns they hide, and
column difficulty in this matrix spans more than two orders of magnitude, so a
naive greedy-vs-random comparison is confounded by cell composition.

The published per-cell CSV cannot settle this either: at
``experiments/run_probe_compression.py:314`` the random arm refused to emit
held-out per-cell rows (``ValueError: random raw rows currently support
all-known only``).  That restriction is lifted for future runs, but the already
published 379 MB CSV predates the fix, so the per-cell predictions have to be
re-derived here.

WHAT IT DOES
------------
1. Rebuilds the 59 x 187 score matrix from ``data/scores.csv`` exactly as the
   experiments do (``load_scores`` -> ``make_matrix`` -> ``filter_matrix``).
2. Reads the *recorded probe sets* (never re-selects them) from
   ``experiments/probe_selection_results_rank1.json`` for the k=0, greedy and
   random arms, and from ``experiments/probe_compression_rank1.json`` for the
   pre-error low-friction allowlist arm.
3. Re-derives every held-out per-cell prediction with
   ``pathopress.completion.complete(rank=1)`` under the published held-out
   protocol: the target model's row is isolated, only its revealed probe cells
   are visible, and the fold's training rows supply the context.
4. Applies the MATCHED-CELL protocol: within each fold, every cell that *any*
   compared arm reveals is dropped, and all arms are then scored on the
   identical remaining cells.
5. Emits both random-arm aggregation conventions explicitly, paired per-fold
   statistics (win counts + Wilcoxon signed-rank), a bootstrap-over-folds CI on
   the reduction, and a per-column skill table scoped to matched cells.

Nothing here re-runs selection, so it is cheap (~3 minutes) and cannot change
which probes were chosen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.completion import complete  # noqa: E402
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.probes import (  # noqa: E402
    SKILL_NOISE_FLOOR_DISPERSION,
    ColumnSkill,
    compute_column_loo_baseline_medae,
    compute_column_robust_dispersion,
    compute_column_skill,
    summarize_skill_positive_fraction,
)

SCHEMA_VERSION = 1

# The allowlist arm's greedy prefixes live in the compression artifact under
# this candidate mode.  ``medae`` is the published selection objective.
ALLOWLIST_MODE = "pre_error_low_friction_allowlist"
ANY_MODE = "any_candidate"
SELECTION_OBJECTIVE = "medae"


# --------------------------------------------------------------------------
# Worker side: re-derive held-out predictions for one (fold, probe set) pair.
# --------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker(matrix: np.ndarray) -> None:
    _WORKER["matrix"] = matrix
    _WORKER["observed"] = np.isfinite(matrix)


def _predict_heldout(
    matrix: np.ndarray,
    observed: np.ndarray,
    targets: Sequence[int],
    context: Sequence[int],
    probes: Sequence[int],
    rank: int,
) -> dict[tuple[int, int], float]:
    """Predictions for every observed, non-probe cell of the target rows.

    Mirrors ``pathopress.probe_compression.predict_heldout_models``: the target
    row keeps only its observed probe cells, the context rows keep everything,
    and one rank-``rank`` completion is run per target row.  Revealed probe
    cells are not returned -- the matched-cell protocol drops them anyway, and
    returning them would invite scoring a revealed cell as a zero-error
    "prediction".
    """

    probe_set = set(int(p) for p in probes)
    context_rows = list(context)
    out: dict[tuple[int, int], float] = {}
    for target in targets:
        target_columns = np.flatnonzero(observed[target])
        known = [int(j) for j in target_columns if int(j) in probe_set]
        hidden = [int(j) for j in target_columns if int(j) not in probe_set]
        if not hidden:
            continue
        train = np.full_like(matrix, np.nan)
        train[context_rows, :] = matrix[context_rows, :]
        if known:
            train[target, known] = matrix[target, known]
        completed = complete(train, rank=rank, allow_empty_rows=True)
        for j in hidden:
            out[(int(target), j)] = float(completed[target, j])
    return out


def _job(payload: tuple[Any, ...]) -> tuple[tuple[Any, ...], dict[tuple[int, int], float]]:
    key, targets, context, probes, rank = payload
    predictions = _predict_heldout(
        _WORKER["matrix"], _WORKER["observed"], targets, context, probes, rank
    )
    return key, predictions


# --------------------------------------------------------------------------
# Artifact readers
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_selection_folds(
    path: Path, model_index: dict[str, int], n_models: int, max_k: int
) -> list[dict[str, Any]]:
    """Per-fold recorded probe sets for the k=0, greedy and random arms."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    block = payload.get("heldout_leave_one_family_out")
    if not block:
        raise ValueError(f"{path} has no heldout_leave_one_family_out block")
    folds: list[dict[str, Any]] = []
    for record in block["folds"]:
        validation = tuple(model_index[m] for m in record["validation_models"])
        context = tuple(i for i in range(n_models) if i not in set(validation))
        greedy = {
            int(step["step"]): tuple(int(i) for i in step["probe_indices"])
            for step in record["validation"]
            if int(step["step"]) <= max_k
        }
        random_sets: dict[tuple[int, int], tuple[int, ...]] = {}
        for row in record["random"]:
            k = int(row["k"])
            if k > max_k:
                continue
            random_sets[(int(row["repeat"]), k)] = tuple(
                int(i) for i in row["probe_indices"]
            )
        folds.append(
            {
                "fold": int(record["fold"]),
                "family": record["family"],
                "validation_indices": validation,
                "context_indices": context,
                "greedy": greedy,
                "random": random_sets,
            }
        )
    folds.sort(key=lambda item: item["fold"])
    return folds


def read_allowlist_prefixes(
    path: Path, max_k: int
) -> dict[int, dict[int, tuple[int, ...]]]:
    """Per-fold greedy prefixes for the pre-error low-friction allowlist arm.

    The compression artifact is ~100 MB, so this is only read when the
    allowlist arm is requested.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    lofo = payload["curves"][ALLOWLIST_MODE]["lofo"]
    prefixes: dict[int, dict[int, tuple[int, ...]]] = {}
    for fold_key, block in lofo.items():
        steps = block[ALLOWLIST_MODE][f"heldout_greedy_{SELECTION_OBJECTIVE}"]
        prefixes[int(fold_key)] = {
            int(step["k"]): tuple(int(i) for i in step["probe_indices"])
            for step in steps
            if int(step["k"]) <= max_k
        }
    return prefixes


# --------------------------------------------------------------------------
# Matched-cell bookkeeping
# --------------------------------------------------------------------------


def _revealed_cells(
    observed: np.ndarray, targets: Sequence[int], probes: Iterable[int]
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for target in targets:
        for j in probes:
            if observed[target, int(j)]:
                cells.add((int(target), int(j)))
    return cells


def _medae(errors: Sequence[float]) -> float:
    values = np.asarray(errors, dtype=float)
    if not values.size:
        return float("nan")
    return float(np.median(values))


def _median_finite(values: Sequence[float]) -> float:
    """Median over the finite entries only.

    A fold can end up with zero matched cells -- every one of its observed
    target cells was revealed by some arm -- and such a fold carries no
    information about any arm.  It is dropped from the aggregate rather than
    propagating a NaN through the median.
    """

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan")
    return float(np.median(array))


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank over paired per-fold MedAEs."""

    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    differences = left - right
    non_zero = int(np.count_nonzero(differences))
    if non_zero == 0:
        return {"n_pairs": int(left.size), "n_nonzero_pairs": 0, "statistic": None,
                "p_value": None}
    from scipy.stats import wilcoxon as _scipy_wilcoxon

    result = _scipy_wilcoxon(left, right, zero_method="wilcox", alternative="two-sided")
    return {
        "n_pairs": int(left.size),
        "n_nonzero_pairs": non_zero,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _bootstrap_reduction(
    baseline_by_fold: Sequence[float],
    arm_by_fold: Sequence[float],
    *,
    n_bootstrap: int,
    seed: int,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Percentile CI on ``1 - median(arm) / median(baseline)``, folds resampled.

    Folds are the independent unit: a fold is one held-out model family, and
    families never appear on both sides of a fold.  Resampling cells instead
    would treat the many cells of one family as independent evidence.
    """

    base = np.asarray(baseline_by_fold, dtype=float)
    arm = np.asarray(arm_by_fold, dtype=float)
    keep = np.isfinite(base) & np.isfinite(arm)
    base, arm = base[keep], arm[keep]
    n = base.size
    if n == 0:
        return {"point_estimate": None, "ci_lower": None, "ci_upper": None,
                "ci_level": ci_level, "n_bootstrap": n_bootstrap, "seed": seed,
                "n_folds": 0}
    point = 1.0 - float(np.median(arm)) / float(np.median(base))
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, n, size=(n_bootstrap, n))
    resampled = 1.0 - np.median(arm[draws], axis=1) / np.median(base[draws], axis=1)
    tail = (1.0 - ci_level) / 2.0
    lower, upper = np.percentile(resampled, [100.0 * tail, 100.0 * (1.0 - tail)])
    return {
        "point_estimate": float(point),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_level": float(ci_level),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "n_folds": int(n),
    }


def _bootstrap_fraction(
    flags: Sequence[bool], *, n_bootstrap: int, seed: int, ci_level: float = 0.95
) -> dict[str, Any]:
    values = np.asarray([1.0 if flag else 0.0 for flag in flags], dtype=float)
    n = values.size
    if n == 0:
        return {"n": 0, "n_positive": 0, "fraction": None, "ci_lower": None,
                "ci_upper": None, "ci_level": ci_level, "n_bootstrap": n_bootstrap,
                "seed": seed}
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, n, size=(n_bootstrap, n))
    resampled = values[draws].mean(axis=1)
    tail = (1.0 - ci_level) / 2.0
    lower, upper = np.percentile(resampled, [100.0 * tail, 100.0 * (1.0 - tail)])
    return {
        "n": int(n),
        "n_positive": int(values.sum()),
        "fraction": float(values.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_level": float(ci_level),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


# --------------------------------------------------------------------------
# Main replay
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=ROOT / "data/scores.csv")
    parser.add_argument(
        "--selection", type=Path,
        default=ROOT / "experiments/probe_selection_results_rank1.json",
    )
    parser.add_argument(
        "--compression", type=Path,
        default=ROOT / "experiments/probe_compression_rank1.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/lofo_matched_cells_rank1.json",
    )
    parser.add_argument(
        "--informativeness-output", type=Path,
        default=ROOT / "outputs/probe_informativeness_rank1.csv",
        help="per-column CSV rewritten only when --write-informativeness is set",
    )
    parser.add_argument("--write-informativeness", action="store_true")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--headline-k", type=int, default=4,
                        help="k used for the headline arm comparison and the "
                             "per-column skill numerator")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-allowlist-arm", dest="allowlist", action="store_true", default=True,
        help="also replay the pre-error low-friction allowlist greedy arm "
             "(reads the ~100 MB compression artifact)",
    )
    parser.add_argument("--skip-allowlist-arm", dest="allowlist", action="store_false")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    scores = load_scores(args.scores)
    matrix, models, evaluations = filter_matrix(*make_matrix(scores))
    observed = np.isfinite(matrix)
    model_index = {model: i for i, model in enumerate(models)}
    metadata: dict[str, tuple[str, str]] = {
        score.evaluation_id: (score.suite_id, score.metric) for score in scores
    }

    folds = read_selection_folds(
        args.selection, model_index, len(models), args.max_k
    )
    allowlist_prefixes: dict[int, dict[int, tuple[int, ...]]] = {}
    if args.allowlist:
        allowlist_prefixes = read_allowlist_prefixes(args.compression, args.max_k)

    # ---- build the job list and, in the same pass, the matched-cell sets ----
    #
    # The matched set is defined PER DEPTH k, not once for all depths.  At
    # depth k the compared arms are the k=0 baseline, the greedy k-prefix and
    # the ten random k-prefixes; those are the arms whose revealed cells have
    # to come out.  Excluding the union over every depth up to max_k instead
    # would shrink the k=1 denominator using probes that only the k=5 arms ever
    # measure, which is a different (and strictly weaker) experiment.
    jobs: list[tuple[Any, ...]] = []
    matched_by_k: dict[int, dict[int, set[tuple[int, int]]]] = {}
    strict_by_k: dict[int, dict[int, set[tuple[int, int]]]] = {}
    target_by_fold: dict[int, set[tuple[int, int]]] = {}
    excluded_by_k: dict[int, dict[int, set[tuple[int, int]]]] = {}
    n_random_repeats = 0
    depths = sorted({k for fold in folds for k in fold["greedy"]})

    for fold in folds:
        fold_id = fold["fold"]
        targets = fold["validation_indices"]
        context = fold["context_indices"]
        all_target_cells = {
            (int(i), int(j))
            for i in targets
            for j in np.flatnonzero(observed[i])
        }
        target_by_fold[fold_id] = all_target_cells
        repeats = sorted({repeat for repeat, _ in fold["random"]})
        n_random_repeats = max(n_random_repeats, len(repeats))
        allow = allowlist_prefixes.get(fold_id, {})

        for k in depths:
            revealed: set[tuple[int, int]] = set()
            if k in fold["greedy"]:
                revealed |= _revealed_cells(observed, targets, fold["greedy"][k])
            for repeat in repeats:
                probes = fold["random"].get((repeat, k))
                if probes is not None:
                    revealed |= _revealed_cells(observed, targets, probes)
            revealed &= all_target_cells
            excluded_by_k.setdefault(k, {})[fold_id] = revealed
            matched_by_k.setdefault(k, {})[fold_id] = all_target_cells - revealed
            # The allowlist arm reveals cells the headline arms do not, so it
            # gets its own strictly smaller matched set.  Folding it into the
            # headline set would silently change the headline denominator.
            allow_revealed = (
                _revealed_cells(observed, targets, allow[k]) if k in allow else set()
            )
            strict_by_k.setdefault(k, {})[fold_id] = (
                all_target_cells - revealed - allow_revealed
            )

        jobs.append((("k0", fold_id, 0, -1), targets, context, (), args.rank))
        for k, probes in sorted(fold["greedy"].items()):
            jobs.append((("greedy", fold_id, k, -1), targets, context, probes, args.rank))
        for (repeat, k), probes in sorted(fold["random"].items()):
            jobs.append((("random", fold_id, k, repeat), targets, context, probes,
                         args.rank))
        for k, probes in sorted(allow.items()):
            jobs.append((("allowlist_greedy", fold_id, k, -1), targets, context, probes,
                         args.rank))

    print(f"replaying {len(jobs)} (fold, probe-set) evaluations across "
          f"{len(folds)} folds", flush=True)

    predictions: dict[tuple[Any, ...], dict[tuple[int, int], float]] = {}
    if args.workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(matrix,)
        ) as executor:
            for key, result in executor.map(_job, jobs, chunksize=4):
                predictions[key] = result
    else:
        _init_worker(matrix)
        for job in jobs:
            key, result = _job(job)
            predictions[key] = result

    # ---- score every arm on the identical matched cells ---------------------
    fold_ids = [fold["fold"] for fold in folds]

    def fold_errors(
        key: tuple[Any, ...], cell_sets: dict[int, set[tuple[int, int]]]
    ) -> list[float]:
        fold_id = key[1]
        preds = predictions[key]
        return [
            abs(preds[cell] - float(matrix[cell[0], cell[1]]))
            for cell in cell_sets[fold_id]
            if cell in preds
        ]

    def score_depth(
        k: int,
        cell_sets: dict[int, set[tuple[int, int]]],
        *,
        include_allowlist: bool,
    ) -> dict[str, Any]:
        """Score every arm at depth ``k`` on one shared per-fold cell set."""

        k0_by_fold = {
            f: _medae(fold_errors(("k0", f, 0, -1), cell_sets)) for f in fold_ids
        }
        k0_values = [k0_by_fold[f] for f in fold_ids]

        def deterministic(name: str) -> list[float]:
            series = []
            for f in fold_ids:
                key = (name, f, k, -1)
                series.append(
                    _medae(fold_errors(key, cell_sets)) if key in predictions
                    else float("nan")
                )
            return series

        greedy = deterministic("greedy")
        allowlist = (
            deterministic("allowlist_greedy")
            if include_allowlist and allowlist_prefixes
            else None
        )

        random_by_fold: dict[int, dict[int, float]] = {}
        for fold in folds:
            f = fold["fold"]
            for repeat in sorted({r for r, kk in fold["random"] if kk == k}):
                random_by_fold.setdefault(f, {})[repeat] = _medae(
                    fold_errors(("random", f, k, repeat), cell_sets)
                )
        random_fold_medians = [
            _median_finite(list(random_by_fold.get(f, {}).values())) for f in fold_ids
        ]
        random_flat = [
            value
            for f in fold_ids
            for value in random_by_fold.get(f, {}).values()
        ]

        def against_k0(series: Sequence[float]) -> dict[str, Any]:
            comparable = int(sum(
                1 for value, base in zip(series, k0_values)
                if np.isfinite(value) and np.isfinite(base)
            ))
            wins = sum(
                1 for value, base in zip(series, k0_values)
                if np.isfinite(value) and np.isfinite(base) and value < base
            )
            return {
                "folds_improved": wins,
                "n_folds": comparable,
                "wilcoxon_signed_rank": _wilcoxon(k0_values, series),
                "bootstrap_reduction_over_folds": _bootstrap_reduction(
                    k0_values, series, n_bootstrap=args.bootstrap, seed=args.seed
                ),
            }

        def series_row(series: Sequence[float]) -> dict[str, Any]:
            return {
                "medae_median_of_fold_medians": _finite(_median_finite(series)),
                "n_folds": int(sum(1 for value in series if np.isfinite(value))),
                "per_fold_medae": {
                    str(f): _finite(value) for f, value in zip(fold_ids, series)
                },
            }

        random_wins = sum(
            1 for a, b in zip(random_fold_medians, greedy)
            if np.isfinite(a) and np.isfinite(b) and b < a
        )
        block = {
            "k": k,
            "n_matched_cells": sum(len(cells) for cells in cell_sets.values()),
            "n_excluded_revealed_cells": sum(
                len(target_by_fold[f]) - len(cell_sets[f]) for f in fold_ids
            ),
            "arms": {
                "k0": series_row(k0_values),
                "greedy": {**series_row(greedy), "vs_k0": against_k0(greedy)},
                "random": {
                    # Convention A: every (fold, repeat) MedAE is one
                    # observation.  Repeats within a fold are not independent,
                    # so this convention up-weights folds whose repeats
                    # disagree.
                    "medae_median_over_fold_repeat_medaes": _finite(
                        _median_finite(random_flat)
                    ),
                    "n_fold_repeat_pairs": int(
                        sum(1 for value in random_flat if np.isfinite(value))
                    ),
                    # Convention B: collapse the repeats inside each fold
                    # first, so each fold contributes exactly one number --
                    # directly comparable with the k=0 and greedy arms, which
                    # have no repeat dimension.
                    "medae_median_of_fold_medians": _finite(
                        _median_finite(random_fold_medians)
                    ),
                    "n_folds": int(
                        sum(1 for value in random_fold_medians if np.isfinite(value))
                    ),
                    "per_fold_median_over_repeats": {
                        str(f): _finite(value)
                        for f, value in zip(fold_ids, random_fold_medians)
                    },
                    "vs_k0": against_k0(random_fold_medians),
                },
            },
            "greedy_vs_random": {
                "random_aggregation": "median_of_fold_medians",
                "folds_greedy_better": random_wins,
                "n_folds": int(sum(
                    1 for a, b in zip(random_fold_medians, greedy)
                    if np.isfinite(a) and np.isfinite(b)
                )),
                "wilcoxon_signed_rank": _wilcoxon(random_fold_medians, greedy),
                "bootstrap_reduction_over_folds": _bootstrap_reduction(
                    random_fold_medians, greedy,
                    n_bootstrap=args.bootstrap, seed=args.seed,
                ),
            },
        }
        if allowlist is not None:
            block["arms"]["allowlist_greedy"] = {
                **series_row(allowlist),
                "vs_k0": against_k0(allowlist),
            }
        return block

    by_depth = [
        score_depth(k, matched_by_k[k], include_allowlist=False) for k in depths
    ]
    allowlist_by_depth = (
        [score_depth(k, strict_by_k[k], include_allowlist=True) for k in depths]
        if allowlist_prefixes
        else None
    )

    # ---- per-column skill on matched cells ---------------------------------
    headline_k = args.headline_k
    column_table = build_column_table(
        matrix=matrix,
        evaluations=evaluations,
        folds=folds,
        matched_by_fold=matched_by_k[headline_k],
        predictions=predictions,
        headline_k=headline_k,
        metadata=metadata,
    )
    skill = summarize_column_skill(
        matrix=matrix,
        column_table=column_table,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    n_target = sum(len(cells) for cells in target_by_fold.values())
    headline_matched = matched_by_k[headline_k]
    headline_excluded = excluded_by_k[headline_k]

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "lofo_matched_cell_replay_v1",
        "description": (
            "Held-out per-cell predictions re-derived from the recorded LOFO "
            "probe sets, then scored on the identical per-fold cell set for "
            "every arm.  Cells revealed by any compared arm are excluded."
        ),
        "matrix": {
            "n_models": len(models),
            "n_evaluations": len(evaluations),
            "n_observed": int(observed.sum()),
        },
        "configuration": {
            "rank": args.rank,
            "max_k": args.max_k,
            "headline_k": headline_k,
            "n_folds": len(folds),
            "n_random_repeats": n_random_repeats,
            "allowlist_arm_included": bool(allowlist_prefixes),
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "noise_floor_dispersion": SKILL_NOISE_FLOOR_DISPERSION,
        },
        "matched_cells": {
            "n_target_cells": n_target,
            "exclusion_rule": (
                "at each depth k, the union over the compared arms of the "
                "cells each reveals: the greedy k-prefix and every random "
                "repeat's k-prefix"
            ),
            "headline_k": headline_k,
            "n_excluded_revealed_cells": sum(
                len(cells) for cells in headline_excluded.values()
            ),
            "n_matched_cells": sum(len(cells) for cells in headline_matched.values()),
            "by_k": [
                {
                    "k": k,
                    "n_excluded_revealed_cells": sum(
                        len(cells) for cells in excluded_by_k[k].values()
                    ),
                    "n_matched_cells": sum(
                        len(cells) for cells in matched_by_k[k].values()
                    ),
                }
                for k in depths
            ],
            "per_fold_at_headline_k": [
                {
                    "fold": fold["fold"],
                    "family": fold["family"],
                    "n_validation_models": len(fold["validation_indices"]),
                    "n_target_cells": len(target_by_fold[fold["fold"]]),
                    "n_excluded_revealed_cells": len(headline_excluded[fold["fold"]]),
                    "n_matched_cells": len(headline_matched[fold["fold"]]),
                }
                for fold in folds
            ],
        },
        "by_k": by_depth,
        # The allowlist arm reveals cells the headline arms do not, so it gets
        # its own strictly smaller matched set and its own copy of every arm.
        # Never mix numbers across the two blocks.
        "allowlist_arm_by_k": allowlist_by_depth,
        "per_column_skill": skill,
        "columns": column_table,
        "caveats": [
            "Arms are compared on matched cells only; the published per-arm "
            "MedAEs in probe_compression_rank1.json use each arm's own hidden "
            "denominator and are therefore not directly comparable.",
            "The random arm has two defensible aggregations and both are "
            "reported: median over all (fold, repeat) MedAEs, and median of "
            "per-fold medians over repeats.  They differ, so any single number "
            "quoted from this artifact must name its convention.",
            "Folds, not cells, are the bootstrap unit: one fold is one held-out "
            "model family.",
        ],
        "provenance": {
            "scores_sha256": _sha256(args.scores),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "selection_artifact": str(args.selection.relative_to(ROOT)),
            "selection_sha256": _sha256(args.selection),
            **(
                {
                    "compression_artifact": str(args.compression.relative_to(ROOT)),
                    "compression_sha256": _sha256(args.compression),
                }
                if allowlist_prefixes
                else {}
            ),
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    return payload


def build_column_table(
    *,
    matrix: np.ndarray,
    evaluations: Sequence[str],
    folds: Sequence[dict[str, Any]],
    matched_by_fold: dict[int, set[tuple[int, int]]],
    predictions: dict[tuple[Any, ...], dict[tuple[int, int], float]],
    headline_k: int,
    metadata: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    """Per-column matched-cell MedAE for the k=0 and greedy-k arms.

    Errors are pooled across folds *within* a column.  This is the correctly
    scoped skill numerator: it is denominated over exactly the cells of one
    evaluation column, matching the per-column leave-one-out baseline it is
    divided by.  The historical numerator was the matrix-wide parity MedAE,
    which spans only 1.75-2.65 across the panel while the per-column baseline
    spans 0.15-32.1, so the resulting "skill" was a monotone re-encoding of
    column dispersion rather than a statement about the completion model.
    """

    n_columns = matrix.shape[1]
    k0_errors: list[list[float]] = [[] for _ in range(n_columns)]
    greedy_errors: list[list[float]] = [[] for _ in range(n_columns)]
    for fold in folds:
        fold_id = fold["fold"]
        matched = matched_by_fold[fold_id]
        k0_pred = predictions[("k0", fold_id, 0, -1)]
        greedy_pred = predictions.get(("greedy", fold_id, headline_k, -1), {})
        for cell in matched:
            actual = float(matrix[cell[0], cell[1]])
            if cell in k0_pred:
                k0_errors[cell[1]].append(abs(k0_pred[cell] - actual))
            if cell in greedy_pred:
                greedy_errors[cell[1]].append(abs(greedy_pred[cell] - actual))

    table: list[dict[str, Any]] = []
    for column in range(n_columns):
        suite, metric = metadata.get(evaluations[column], ("", ""))
        table.append({
            "evaluation_index": column,
            "evaluation_id": evaluations[column],
            "suite_id": suite,
            "metric": metric,
            "n_matched_cells": len(greedy_errors[column]),
            "matched_k0_medae": _finite(_medae(k0_errors[column])),
            "matched_greedy_medae": _finite(_medae(greedy_errors[column])),
            "column_loo_baseline_medae": _finite(
                compute_column_loo_baseline_medae(matrix, column)
            ),
            "column_robust_dispersion": _finite(
                compute_column_robust_dispersion(matrix, column)
            ),
        })
    return table


def _column_skill(matrix: np.ndarray, row: dict[str, Any]) -> ColumnSkill:
    medae = row["matched_greedy_medae"]
    return compute_column_skill(
        matrix,
        int(row["evaluation_index"]),
        float(medae) if medae is not None else float("nan"),
    )


def summarize_column_skill(
    *,
    matrix: np.ndarray,
    column_table: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Per-column utility of the greedy probe set, plus its exclusion bias.

    The headline question is per column: on the matched cells of that column,
    does the greedy k-probe scorecard predict it better than the k=0 no-probe
    scorecard?  That is a like-for-like comparison -- same column, same cells,
    same completion model, only the probe budget differs.

    Two fractions are reported side by side on purpose.  The headline drops
    columns that ``compute_column_skill`` cannot score, which is dominated by
    the robust-dispersion noise floor.  That exclusion is *not* neutral: a
    low-dispersion column also has a tiny leave-one-out baseline and is
    therefore close to structurally guaranteed to score negative skill.
    Reporting only the headline would hide a bias with a known sign, so the
    all-column fraction is emitted in band.
    """

    skills = [_column_skill(matrix, row) for row in column_table]
    for row, skill in zip(column_table, skills):
        # Corrected, correctly scoped skill: per-column matched-cell numerator
        # over the same column's leave-one-out baseline.
        row["skill_score"] = _finite(skill.skill_score)
        row["skill_score_raw"] = _finite(skill.skill_score_raw)
        row["medae_ratio_to_loo_baseline"] = _finite(skill.medae_ratio)
        row["skill_excluded_below_noise_floor"] = bool(skill.excluded_below_noise_floor)
        row["skill_exclusion_reason"] = skill.exclusion_reason
        raw = skill.skill_score_raw
        row["skill_vs_loo_baseline_positive"] = bool(raw is not None and raw > 0.0)
        # Headline flag: greedy beats this column's own k=0 arm on the same
        # matched cells.  Ties count as non-positive -- a tie means the probe
        # budget bought nothing for that column.
        greedy = row["matched_greedy_medae"]
        k0 = row["matched_k0_medae"]
        row["beats_k0"] = bool(
            greedy is not None and k0 is not None and float(greedy) < float(k0)
        )

    scored = [
        row for row in column_table if not row["skill_excluded_below_noise_floor"]
    ]
    headline = _bootstrap_fraction(
        [row["beats_k0"] for row in scored], n_bootstrap=n_bootstrap, seed=seed
    )
    all_columns = _bootstrap_fraction(
        [row["beats_k0"] for row in column_table], n_bootstrap=n_bootstrap, seed=seed
    )

    # Secondary framing, kept because it is what the ``skill_score`` column in
    # the informativeness CSV encodes: greedy against the column-median
    # leave-one-out baseline rather than against the k=0 completion.
    skill_summary = summarize_skill_positive_fraction(
        skills, n_bootstrap=n_bootstrap, seed=seed
    )

    excluded = [
        {
            "evaluation_id": row["evaluation_id"],
            "column_loo_baseline_medae": row["column_loo_baseline_medae"],
            "column_robust_dispersion": row["column_robust_dispersion"],
            "n_matched_cells": row["n_matched_cells"],
            "matched_k0_medae": row["matched_k0_medae"],
            "matched_greedy_medae": row["matched_greedy_medae"],
            "skill_exclusion_reason": row["skill_exclusion_reason"],
            "would_have_been_positive": bool(row["beats_k0"]),
        }
        for row in column_table
        if row["skill_excluded_below_noise_floor"]
    ]

    return {
        "headline_definition": (
            "fraction of scored evaluation columns whose matched-cell greedy-k "
            "MedAE is strictly below that same column's matched-cell k=0 MedAE"
        ),
        "scored_column_rule": (
            "a column is scored unless compute_column_skill excludes it: no "
            "matched cell at all (no_model_error), a non-finite or zero "
            "leave-one-out baseline (degenerate_baseline), or robust dispersion "
            "below SKILL_NOISE_FLOOR_DISPERSION (noise_floor)"
        ),
        "skill_score_definition": (
            "skill_score = 1 - matched-cell greedy-k MedAE for the column / "
            "that column's leave-one-out column-median baseline MedAE, clipped "
            "to [-1, 1]"
        ),
        "noise_floor_dispersion": SKILL_NOISE_FLOOR_DISPERSION,
        "headline_scored_columns": {
            "n_columns_total": len(column_table),
            "n_columns_excluded": len(column_table) - len(scored),
            "n_columns_scored": headline["n"],
            "n_columns_positive": headline["n_positive"],
            "fraction_positive": headline["fraction"],
            "bootstrap_ci_lower": headline["ci_lower"],
            "bootstrap_ci_upper": headline["ci_upper"],
            "bootstrap_ci_level": headline["ci_level"],
            "n_bootstrap": headline["n_bootstrap"],
            "seed": headline["seed"],
        },
        "n_columns_excluded_by_reason": {
            reason: sum(1 for skill in skills if skill.exclusion_reason == reason)
            for reason in ("no_model_error", "degenerate_baseline", "noise_floor")
        },
        # Task-3 bias check.  The noise-floor exclusion is confounded with
        # structurally-guaranteed-negative status, so the headline fraction is
        # not neutral with respect to it.  This variant keeps every column.
        "all_columns_including_noise_floor": all_columns,
        "n_excluded_positive": sum(
            1 for row in excluded if row["would_have_been_positive"]
        ),
        "excluded_columns": excluded,
        "secondary_skill_vs_loo_baseline": {
            "n_columns_scored": skill_summary.n_columns_scored,
            "n_columns_positive": skill_summary.n_columns_positive,
            "fraction_positive": _finite(skill_summary.fraction_positive),
            "bootstrap_ci_lower": _finite(skill_summary.ci_lower),
            "bootstrap_ci_upper": _finite(skill_summary.ci_upper),
            # Computed unconditionally, so noise-floor columns are judged on
            # the same rule rather than counted negative by suppression.
            "n_columns_positive_all_columns": sum(
                1
                for row in column_table
                if row["matched_greedy_medae"] is not None
                and row["column_loo_baseline_medae"]
                and float(row["matched_greedy_medae"])
                < float(row["column_loo_baseline_medae"])
            ),
        },
        "by_suite": _suite_breakdown(column_table),
    }


def _suite_breakdown(column_table: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    breakdown: dict[str, dict[str, int]] = {}
    for row in column_table:
        suite = str(row["suite_id"])
        bucket = breakdown.setdefault(
            suite, {"n_scored": 0, "n_positive": 0, "n_excluded": 0, "n_total": 0}
        )
        bucket["n_total"] += 1
        if row["skill_excluded_below_noise_floor"]:
            bucket["n_excluded"] += 1
            continue
        bucket["n_scored"] += 1
        if row["beats_k0"]:
            bucket["n_positive"] += 1
    return dict(sorted(breakdown.items()))


# --------------------------------------------------------------------------
# Informativeness CSV rewrite (Task 2)
# --------------------------------------------------------------------------

INFORMATIVENESS_FIELDS = [
    "informativeness_rank",
    "evaluation_id",
    "suite_id",
    "metric",
    "models_with_score",
    "model_coverage",
    "parity_medae",
    "hidden_only_medae",
    "model_average_mae",
    "improvement_over_column_median",
    "column_baseline_medae",
    "skill_in_sample_matrixwide_numerator_DEPRECATED",
    "column_loo_baseline_medae",
    "column_robust_dispersion",
    # Correctly scoped skill inputs: LOFO, matched-cell, per-column numerator.
    "matched_cell_n",
    "matched_cell_k0_medae",
    "matched_cell_greedy_medae",
    "matched_cell_greedy_beats_k0",
    "skill_numerator_medae",
    "skill_numerator_scope",
    "medae_ratio_to_loo_baseline",
    "skill_score",
    "skill_score_raw",
    "skill_excluded_below_noise_floor",
    "skill_exclusion_reason",
    "parity_medae_normalized",
    "parity_medae_normalized_sd_legacy",
]


# Columns renamed in place.  ``skill_score_in_sample`` carried BOTH an
# in-sample oracle denominator and the matrix-wide numerator scope bug, so it
# was renamed to say so rather than left sitting next to the corrected
# ``skill_score`` where a consumer could grab the wrong one.
LEGACY_RENAMES = {
    "skill_score_in_sample": "skill_in_sample_matrixwide_numerator_DEPRECATED",
}


def rewrite_informativeness(
    csv_path: Path, column_table: list[dict[str, Any]], headline_k: int
) -> dict[str, Any]:
    """Replace the mis-scoped skill columns in the published CSV.

    The three fields ``medae_ratio_to_loo_baseline``, ``skill_score`` and
    ``skill_score_raw`` previously divided a matrix-wide numerator by a
    column-scoped denominator.  They are recomputed here from the matched-cell,
    per-column numerator.  The old values are NOT kept under a near-identical
    name -- a consumer would inevitably pick the wrong one -- they are dropped,
    and the matched-cell inputs are written alongside so the ratio can be
    audited by hand.

    ``parity_medae``, ``hidden_only_medae``, ``model_average_mae`` and
    ``improvement_over_column_median`` are untouched: they come from the
    transductive single-probe track and were never mis-scoped.
    """

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        unknown = (
            set(reader.fieldnames or ())
            - set(INFORMATIVENESS_FIELDS)
            - set(LEGACY_RENAMES)
        )
        if unknown:
            raise ValueError(
                f"{csv_path} has columns this writer does not know how to "
                f"carry forward and would silently drop: {sorted(unknown)}"
            )
        existing = list(reader)
    by_id = {row["evaluation_id"]: row for row in column_table}
    rows = []
    for source in existing:
        column = by_id[source["evaluation_id"]]
        for legacy, renamed in LEGACY_RENAMES.items():
            if legacy in source and renamed not in source:
                source[renamed] = source.pop(legacy)
        row = {key: source.get(key, "") for key in INFORMATIVENESS_FIELDS}
        row["column_loo_baseline_medae"] = column["column_loo_baseline_medae"]
        row["column_robust_dispersion"] = column["column_robust_dispersion"]
        row["matched_cell_n"] = column["n_matched_cells"]
        row["matched_cell_k0_medae"] = column["matched_k0_medae"]
        row["matched_cell_greedy_medae"] = column["matched_greedy_medae"]
        row["matched_cell_greedy_beats_k0"] = column["beats_k0"]
        row["skill_numerator_medae"] = column["matched_greedy_medae"]
        row["skill_numerator_scope"] = (
            f"lofo_matched_cells_greedy_k{headline_k}_per_column"
        )
        row["medae_ratio_to_loo_baseline"] = column["medae_ratio_to_loo_baseline"]
        row["skill_score"] = column["skill_score"]
        row["skill_score_raw"] = column["skill_score_raw"]
        row["skill_excluded_below_noise_floor"] = column[
            "skill_excluded_below_noise_floor"
        ]
        row["skill_exclusion_reason"] = column["skill_exclusion_reason"] or ""
        rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=INFORMATIVENESS_FIELDS, lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: ("" if value is None else value) for key, value in row.items()
            })
    return {"path": str(csv_path), "n_rows": len(rows), "headline_k": headline_k}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    written = None
    if args.write_informativeness:
        written = rewrite_informativeness(
            args.informativeness_output, payload["columns"], args.headline_k
        )

    matched = payload["matched_cells"]
    skill = payload["per_column_skill"]
    headline = skill["headline_scored_columns"]
    depth = next(
        block for block in payload["by_k"] if block["k"] == args.headline_k
    )
    print(json.dumps({
        "output": str(args.output),
        "informativeness": written,
        "matched_cells_at_headline_k": {
            "n_target": matched["n_target_cells"],
            "n_excluded": matched["n_excluded_revealed_cells"],
            "n_matched": matched["n_matched_cells"],
        },
        "headline_k": args.headline_k,
        "k0_medae": depth["arms"]["k0"]["medae_median_of_fold_medians"],
        "greedy_medae": depth["arms"]["greedy"]["medae_median_of_fold_medians"],
        "greedy_vs_k0": {
            "folds_improved": depth["arms"]["greedy"]["vs_k0"]["folds_improved"],
            "n_folds": depth["arms"]["greedy"]["vs_k0"]["n_folds"],
            "wilcoxon_p": depth["arms"]["greedy"]["vs_k0"][
                "wilcoxon_signed_rank"]["p_value"],
            "reduction": depth["arms"]["greedy"]["vs_k0"][
                "bootstrap_reduction_over_folds"],
        },
        "random_medae_median_of_fold_medians": depth["arms"]["random"][
            "medae_median_of_fold_medians"],
        "random_medae_median_over_fold_repeat_medaes": depth["arms"]["random"][
            "medae_median_over_fold_repeat_medaes"],
        "medae_by_k": {
            str(block["k"]): {
                "k0": block["arms"]["k0"]["medae_median_of_fold_medians"],
                "greedy": block["arms"]["greedy"]["medae_median_of_fold_medians"],
                "random_fold_medians": block["arms"]["random"][
                    "medae_median_of_fold_medians"],
                "random_fold_repeat": block["arms"]["random"][
                    "medae_median_over_fold_repeat_medaes"],
            }
            for block in payload["by_k"]
        },
        "skill_headline": {
            "positive": headline["n_columns_positive"],
            "scored": headline["n_columns_scored"],
            "fraction": headline["fraction_positive"],
            "ci": [headline["bootstrap_ci_lower"], headline["bootstrap_ci_upper"]],
        },
        "skill_all_columns": skill["all_columns_including_noise_floor"],
        "runtime_seconds": payload["runtime_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
