#!/usr/bin/env python3
"""Reproduce BenchPress's hero probe policy on the pathology score matrix.

The primary curve intentionally matches the optimistic all-known protocol.
We additionally report hidden-only error, literal row-average error, and an
isolated 70/30 held-out-model validation curve.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathopress.completion import complete  # noqa: E402
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.probes import (  # noqa: E402
    SKILL_NOISE_FLOOR_DISPERSION,
    ProbeEvaluation,
    compute_column_loo_baseline_medae,
    compute_column_median_baseline_medae,
    compute_column_robust_dispersion,
    compute_column_skill,
    evaluate_column_median_baseline,
    evaluate_global_probes,
    family_blocked_model_split,
    leave_one_family_out_folds,
    random_global_probe_prefixes,
    random_model_split,
    summarize_skill_positive_fraction,
)


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _summary(result: ProbeEvaluation) -> dict[str, object]:
    return {
        "n_target_cells": result.n_target_cells,
        "n_revealed_cells": result.n_revealed_cells,
        "n_hidden_cells": result.n_hidden_cells,
        "parity": {
            "n": result.parity.n,
            "medae": _finite(result.parity.median_absolute_error),
            "mae": _finite(result.parity.mean_absolute_error),
        },
        "hidden_only": {
            "n": result.hidden_only.n,
            "medae": _finite(result.hidden_only.median_absolute_error),
            "mae": _finite(result.hidden_only.mean_absolute_error),
        },
        "model_average": {
            "n": result.model_average.n,
            "medae": _finite(result.model_average.median_absolute_error),
            "mae": _finite(result.model_average.mean_absolute_error),
        },
    }


def _skill_summary(
    matrix: np.ndarray,
    result: ProbeEvaluation,
    evaluations: list[str],
    *,
    seed: int,
) -> dict[str, object]:
    """Headline skill statistic for one probe set (FIX 1).

    For every evaluation column, the model's MedAE over that column's hidden
    target cells is compared against the column's leave-one-out column-median
    baseline MedAE.  The headline number is the FRACTION of scored columns with
    positive skill plus a percentile bootstrap CI over columns -- never a mean
    or median of the raw ratio, which is unbounded below and so is dominated by
    whichever column happens to be tightest.
    """

    column_skills = [
        compute_column_skill(matrix, column, medae)
        for column, medae in enumerate(result.per_column_hidden_medae)
    ]
    summary = summarize_skill_positive_fraction(column_skills, seed=seed)
    return {
        "probe_indices": list(result.probe_indices),
        "probe_ids": [evaluations[index] for index in result.probe_indices],
        "noise_floor_dispersion": SKILL_NOISE_FLOOR_DISPERSION,
        "n_columns_total": summary.n_columns_total,
        "n_columns_excluded": summary.n_columns_excluded,
        "n_columns_excluded_by_reason": {
            reason: sum(
                1 for skill in column_skills if skill.exclusion_reason == reason
            )
            for reason in ("no_model_error", "degenerate_baseline", "noise_floor")
        },
        "n_columns_scored": summary.n_columns_scored,
        "n_columns_positive": summary.n_columns_positive,
        "fraction_columns_positive_skill": _finite(summary.fraction_positive),
        "bootstrap_ci_lower": _finite(summary.ci_lower),
        "bootstrap_ci_upper": _finite(summary.ci_upper),
        "bootstrap_ci_level": summary.ci_level,
        "n_bootstrap": summary.n_bootstrap,
        "bootstrap_seed": summary.seed,
        "per_column": [
            {
                "evaluation_index": skill.column,
                "evaluation_id": evaluations[skill.column],
                "model_hidden_medae": _finite(skill.model_medae),
                "loo_baseline_medae": _finite(skill.loo_baseline_medae),
                "robust_dispersion": _finite(skill.robust_dispersion),
                "excluded_below_noise_floor": skill.excluded_below_noise_floor,
                "exclusion_reason": skill.exclusion_reason,
                "medae_ratio_to_loo_baseline": skill.medae_ratio,
                "skill_score_raw": skill.skill_score_raw,
                "skill_score": skill.skill_score,
            }
            for skill in column_skills
        ],
    }


def _suite_coverage_summary(
    matrix: np.ndarray,
    models: list[str],
    evaluations: list[str],
    suite_by_evaluation: dict[str, str],
) -> dict[str, object]:
    """Summarize suite support from the filtered matrix actually being analyzed."""

    if matrix.shape != (len(models), len(evaluations)):
        raise ValueError("matrix shape must match model and evaluation labels")
    missing = [evaluation for evaluation in evaluations if evaluation not in suite_by_evaluation]
    if missing:
        raise ValueError(f"missing suite metadata for evaluations: {missing}")

    represented_suites = sorted({suite_by_evaluation[value] for value in evaluations})
    columns_by_suite = {
        suite: np.asarray(
            [suite_by_evaluation[evaluation] == suite for evaluation in evaluations],
            dtype=bool,
        )
        for suite in represented_suites
    }
    fully_represented_models = [
        model
        for row, model in enumerate(models)
        if represented_suites
        and all(np.isfinite(matrix[row, columns]).any() for columns in columns_by_suite.values())
    ]
    return {
        "represented_suites": represented_suites,
        "n_represented_suites": len(represented_suites),
        "models_with_all_represented_suites": fully_represented_models,
        "n_models_with_all_represented_suites": len(fully_represented_models),
    }


def _eval_task(matrix: np.ndarray, probes: tuple[int, ...], rank: int) -> ProbeEvaluation:
    return evaluate_global_probes(matrix, probes, rank=rank)


def _parallel_evaluate(
    executor: ProcessPoolExecutor,
    matrix: np.ndarray,
    probe_sets: list[tuple[int, ...]],
    rank: int,
) -> list[ProbeEvaluation]:
    futures = [executor.submit(_eval_task, matrix, probes, rank) for probes in probe_sets]
    return [future.result() for future in futures]


def _greedy(
    matrix: np.ndarray,
    *,
    rank: int,
    max_probes: int,
    executor: ProcessPoolExecutor,
    evaluations: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected: list[int] = []
    remaining = list(range(matrix.shape[1]))
    trajectory: list[dict[str, object]] = []
    first_step: list[dict[str, object]] = []
    for step in range(1, max_probes + 1):
        probe_sets = [tuple([*selected, candidate]) for candidate in remaining]
        results = _parallel_evaluate(executor, matrix, probe_sets, rank)
        # KNOWN SELECTION-OBJECTIVE BIAS -- deliberately NOT changed here.
        #
        # ``parity`` is BenchPress's all-known-cell denominator: a revealed
        # probe cell enters the error pool as a literal 0.0
        # (``probes.py`` sets ``predicted[revealed] = actual[revealed]``, and
        # ``_summarize_predictions`` scores ``absolute[targets]`` over revealed
        # AND hidden cells alike).  Greedy therefore gets credit both for
        # probes that make the REST of the scorecard predictable and for
        # probes that simply reveal cells the model would otherwise have
        # predicted badly.  The second effect is not probe informativeness --
        # it is buying the answer.
        #
        # Size of the gap on the published LOFO run at k=4: parity MedAE
        # 1.5026 (revealed cells scored as zero error, n_target 2122) versus
        # hidden-only MedAE 1.6055 (hidden cells only, n_hidden 2037), i.e.
        # the parity objective reads ~6.4% more optimistic than the held-out
        # quantity anyone actually cares about.
        #
        # Switching to ``hidden_only`` would change WHICH probes are selected
        # and so invalidates every published probe set; that is an 8.7-hour
        # rerun and a separate decision.  Until then the objective stays as
        # published and the bias is disclosed rather than silently fixed.
        scores = [result.parity.median_absolute_error for result in results]
        best_position = min(range(len(remaining)), key=lambda pos: (scores[pos], pos))
        if step == 1:
            first_step = [
                {
                    "evaluation_index": candidate,
                    "evaluation_id": evaluations[candidate],
                    **_summary(result),
                }
                for candidate, result in zip(remaining, results)
            ]
        candidate_table = [
            {
                "evaluation_index": candidate,
                "evaluation_id": evaluations[candidate],
                "parity_medae": _finite(result.parity.median_absolute_error),
            }
            for candidate, result in zip(remaining, results)
        ]
        selected.append(remaining.pop(best_position))
        best = results[best_position]
        record = {
            "step": step,
            "added_evaluation_index": selected[-1],
            "added_evaluation_id": evaluations[selected[-1]],
            "probe_indices": selected.copy(),
            "probe_ids": [evaluations[index] for index in selected],
            **_summary(best),
            "candidate_results": candidate_table,
        }
        trajectory.append(record)
        print(
            f"all-known step {step}: {record['added_evaluation_id']} "
            f"MedAE={best.parity.median_absolute_error:.4f}",
            flush=True,
        )
    return trajectory, first_step


def _heldout_evaluate(
    matrix: np.ndarray,
    probes: tuple[int, ...],
    train_indices: tuple[int, ...],
    validation_indices: tuple[int, ...],
    rank: int,
) -> dict[str, object]:
    observed = np.isfinite(matrix)
    probe_set = set(probes)
    hidden_errors: list[float] = []
    parity_errors: list[float] = []
    average_errors: list[float] = []
    revealed = 0
    for target in validation_indices:
        target_columns = np.flatnonzero(observed[target])
        hidden = [int(j) for j in target_columns if int(j) not in probe_set]
        known = [int(j) for j in target_columns if int(j) in probe_set]
        revealed += len(known)
        parity_errors.extend([0.0] * len(known))
        predicted_by_column = {j: float(matrix[target, j]) for j in known}
        if hidden:
            train = np.full_like(matrix, np.nan)
            train[list(train_indices), :] = matrix[list(train_indices), :]
            if known:
                train[target, known] = matrix[target, known]
            completed = complete(train, rank=rank, allow_empty_rows=True)
            for column in hidden:
                error = abs(float(completed[target, column] - matrix[target, column]))
                hidden_errors.append(error)
                parity_errors.append(error)
                predicted_by_column[column] = float(completed[target, column])
        if predicted_by_column:
            actual_average = float(np.mean(matrix[target, target_columns]))
            predicted_average = float(
                np.mean([predicted_by_column[int(j)] for j in target_columns])
            )
            average_errors.append(abs(predicted_average - actual_average))

    def stats(values: list[float]) -> dict[str, object]:
        array = np.asarray(values, dtype=float)
        return {
            "n": int(array.size),
            "medae": _finite(float(np.median(array))) if array.size else None,
            "mae": _finite(float(np.mean(array))) if array.size else None,
        }

    return {
        "probe_indices": list(probes),
        "n_revealed_cells": revealed,
        "parity": stats(parity_errors),
        "hidden_only": stats(hidden_errors),
        "model_average": stats(average_errors),
    }


def _heldout_task(args: tuple[object, ...]) -> dict[str, object]:
    return _heldout_evaluate(*args)  # type: ignore[arg-type]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "data" / "scores.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "probe_selection_results_rank1.json",
    )
    parser.add_argument(
        "--informativeness-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "probe_informativeness_rank1.csv",
    )
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--max-probes", type=int, default=10)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=("random", "family_blocked", "leave_one_family_out"),
        default="leave_one_family_out",
        help=(
            "Held-out reporting protocol. 'leave_one_family_out' (default) runs "
            "one fold per family so every model is validated exactly once and "
            "the split-seed lottery disappears; it still emits the 70/30 "
            "family-blocked curve alongside for comparison. 'family_blocked' "
            "and 'random' reproduce the earlier single-draw arms exactly."
        ),
    )
    parser.add_argument(
        "--lofo-max-probes",
        type=int,
        default=5,
        help=(
            "Probe-prefix depth re-selected independently inside each "
            "leave-one-family-out fold; selection cost is linear in this value "
            "times the fold count"
        ),
    )
    parser.add_argument(
        "--heldout-random-repeats",
        type=int,
        default=10,
        help="Random-probe repeats for the held-out random control curve",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(28, (os.cpu_count() or 2) - 1)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores)
    matrix, models, evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(matrix, models, evaluations)
    metadata: dict[str, tuple[str, str]] = {}
    for score in scores:
        if score.evaluation_id in evaluations:
            metadata[score.evaluation_id] = (score.suite_id, score.metric)
    suite_coverage = _suite_coverage_summary(
        matrix,
        models,
        evaluations,
        {evaluation: suite for evaluation, (suite, _) in metadata.items()},
    )

    baseline = evaluate_column_median_baseline(matrix)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        trajectory, one_probe = _greedy(
            matrix,
            rank=args.rank,
            max_probes=args.max_probes,
            executor=executor,
            evaluations=evaluations,
        )
        random_prefixes = random_global_probe_prefixes(
            len(evaluations),
            max_probes=args.max_probes,
            repeats=args.random_repeats,
            seed=args.seed,
        )
        random_sets = [probes for repeat in random_prefixes for probes in repeat]
        random_results = _parallel_evaluate(executor, matrix, random_sets, args.rank)

        # The 70/30 draw is always emitted: under the default protocol it is the
        # reproducible comparison arm, and it is what `heldout_model` has always
        # meant for downstream consumers.
        if args.split_mode in {"family_blocked", "leave_one_family_out"}:
            model_metadata_path = PROJECT_ROOT / "data" / "model_metadata.csv"
            train_indices, validation_indices, split_info = family_blocked_model_split(
                models, model_metadata_path=model_metadata_path, seed=args.seed
            )
        else:
            train_indices, validation_indices, split_info = random_model_split(
                models, seed=args.seed
            )
        train_matrix = matrix[list(train_indices), :]
        heldout_train_trajectory, _ = _greedy(
            train_matrix,
            rank=args.rank,
            max_probes=args.max_probes,
            executor=executor,
            evaluations=evaluations,
        )
        heldout_args = [
            (
                matrix,
                tuple(int(i) for i in step["probe_indices"]),
                train_indices,
                validation_indices,
                args.rank,
            )
            for step in heldout_train_trajectory
        ]
        heldout_results = list(executor.map(_heldout_task, heldout_args))

        # FIX 4: the greedy-vs-random comparison previously existed only in the
        # all-known (in-sample) track.  Both controls are now run against the
        # same held-out rows as the greedy curve above, so the comparison is
        # made out-of-sample.
        heldout_k0 = _heldout_evaluate(
            matrix, (), train_indices, validation_indices, args.rank
        )
        heldout_random_prefixes = random_global_probe_prefixes(
            len(evaluations),
            max_probes=args.max_probes,
            repeats=args.heldout_random_repeats,
            seed=args.seed,
        )
        heldout_random_args = [
            (matrix, probes, train_indices, validation_indices, args.rank)
            for repeat_prefixes in heldout_random_prefixes
            for probes in repeat_prefixes
        ]
        heldout_random_results = list(
            executor.map(_heldout_task, heldout_random_args)
        )

        lofo_folds: tuple[object, ...] = ()
        lofo_info: dict[str, object] = {}
        lofo_fold_payloads: list[dict[str, object]] = []
        if args.split_mode == "leave_one_family_out":
            lofo_folds, lofo_info = leave_one_family_out_folds(
                models, model_metadata_path=PROJECT_ROOT / "data" / "model_metadata.csv"
            )
            lofo_depth = min(args.lofo_max_probes, len(evaluations))
            for fold in lofo_folds:
                fold_train_matrix = matrix[list(fold.train_indices), :]
                fold_trajectory, _ = _greedy(
                    fold_train_matrix,
                    rank=args.rank,
                    max_probes=lofo_depth,
                    executor=executor,
                    evaluations=evaluations,
                )
                fold_args = [
                    (
                        matrix,
                        tuple(int(i) for i in step["probe_indices"]),
                        fold.train_indices,
                        fold.validation_indices,
                        args.rank,
                    )
                    for step in fold_trajectory
                ]
                fold_results = list(executor.map(_heldout_task, fold_args))
                fold_k0 = _heldout_evaluate(
                    matrix, (), fold.train_indices, fold.validation_indices, args.rank
                )
                fold_random_prefixes = random_global_probe_prefixes(
                    len(evaluations),
                    max_probes=lofo_depth,
                    repeats=args.heldout_random_repeats,
                    seed=args.seed + fold.fold,
                )
                fold_random_args = [
                    (matrix, probes, fold.train_indices, fold.validation_indices, args.rank)
                    for repeat_prefixes in fold_random_prefixes
                    for probes in repeat_prefixes
                ]
                fold_random_results = list(
                    executor.map(_heldout_task, fold_random_args)
                )
                fold_random_rows: list[dict[str, object]] = []
                cursor = 0
                for repeat, prefixes in enumerate(fold_random_prefixes):
                    for k, probes in enumerate(prefixes, start=1):
                        fold_random_rows.append(
                            {
                                "repeat": repeat,
                                "k": k,
                                "probe_ids": [evaluations[index] for index in probes],
                                **fold_random_results[cursor],
                            }
                        )
                        cursor += 1
                lofo_fold_payloads.append(
                    {
                        "fold": fold.fold,
                        "family": fold.family,
                        "n_train_models": len(fold.train_indices),
                        "n_validation_models": len(fold.validation_indices),
                        "validation_models": [
                            models[i] for i in fold.validation_indices
                        ],
                        "train_selected_trajectory": fold_trajectory,
                        "k0_baseline": fold_k0,
                        "validation": [
                            {
                                "step": step["step"],
                                "added_evaluation_id": step["added_evaluation_id"],
                                "probe_ids": step["probe_ids"],
                                **result,
                            }
                            for step, result in zip(fold_trajectory, fold_results)
                        ],
                        "random": fold_random_rows,
                    }
                )

    random_rows: list[dict[str, object]] = []
    cursor = 0
    for repeat, prefixes in enumerate(random_prefixes):
        for k, probes in enumerate(prefixes, start=1):
            random_rows.append(
                {
                    "repeat": repeat,
                    "k": k,
                    "probe_indices": list(probes),
                    "probe_ids": [evaluations[index] for index in probes],
                    **_summary(random_results[cursor]),
                }
            )
            cursor += 1

    # Re-evaluate the final greedy prefix so the per-column hidden MedAE vector
    # is available for the headline skill statistic.
    greedy_final_probes = (
        tuple(int(i) for i in trajectory[-1]["probe_indices"]) if trajectory else ()
    )
    greedy_final_evaluation = evaluate_global_probes(
        matrix, greedy_final_probes, rank=args.rank
    )

    heldout_random_rows: list[dict[str, object]] = []
    cursor = 0
    for repeat, prefixes in enumerate(heldout_random_prefixes):
        for k, probes in enumerate(prefixes, start=1):
            heldout_random_rows.append(
                {
                    "repeat": repeat,
                    "k": k,
                    "probe_ids": [evaluations[index] for index in probes],
                    **heldout_random_results[cursor],
                }
            )
            cursor += 1

    baseline_medae = baseline.parity.median_absolute_error
    # Pre-compute per-column baselines and dispersions for FIX 1/FIX 2.
    column_baseline_medae_cache = np.full(matrix.shape[1], np.nan)
    column_loo_baseline_cache = np.full(matrix.shape[1], np.nan)
    column_robust_dispersion_cache = np.full(matrix.shape[1], np.nan)
    column_sd_cache = np.full(matrix.shape[1], np.nan)
    for col in range(matrix.shape[1]):
        column_baseline_medae_cache[col] = compute_column_median_baseline_medae(
            matrix, col
        )
        column_loo_baseline_cache[col] = compute_column_loo_baseline_medae(matrix, col)
        column_robust_dispersion_cache[col] = compute_column_robust_dispersion(
            matrix, col
        )
        col_observed = matrix[:, col]
        col_finite = col_observed[np.isfinite(col_observed)]
        if col_finite.size > 1:
            column_sd_cache[col] = float(np.std(col_finite, ddof=0))
        elif col_finite.size == 1:
            column_sd_cache[col] = 0.0

    informativeness = []
    for row in one_probe:
        index = int(row["evaluation_index"])
        evaluation_id = evaluations[index]
        # MATRIX-WIDE number: the MedAE over EVERY finite cell of the 59x187
        # matrix when this column is the only probe (probes.py
        # ``parity=_error_summary(absolute[targets])``, where ``targets`` is the
        # whole observed matrix).  It is the correct denominator for the
        # informativeness ranking below, and it must NEVER be used as a
        # per-column skill numerator -- see the FIX note further down.
        parity_medae = float(row["parity"]["medae"])  # type: ignore[index]
        suite, metric = metadata[evaluation_id]
        col_baseline = column_baseline_medae_cache[index]
        col_loo_baseline = column_loo_baseline_cache[index]
        col_dispersion = column_robust_dispersion_cache[index]
        col_sd = column_sd_cache[index]
        # DEPRECATED legacy field, kept only so previously published artifacts
        # stay diffable.  It carries BOTH defects at once: an in-sample oracle
        # denominator (a one-parameter constant fitted on the same ~7
        # observations it is scored against) AND the FIX 5 scope mismatch (a
        # matrix-wide numerator over a column-scoped denominator).  The name
        # spells that out so nobody reaches for it as a skill score.
        if np.isfinite(col_baseline) and col_baseline > 0:
            skill_in_sample_matrixwide_numerator_deprecated = 1.0 - (
                parity_medae / col_baseline
            )
        else:
            skill_in_sample_matrixwide_numerator_deprecated = None
        # FIX 1: leave-one-out denominator, bounded to [-1, 1] for reporting,
        # raw value retained separately, and an explicit exclusion flag for
        # columns whose dispersion sits below the noise floor.
        #
        # FIX 5 (scope bug).  This used to pass ``parity_medae``, a MATRIX-WIDE
        # error, into a COLUMN-SCOPED denominator
        # (compute_column_loo_baseline_medae).  The numerator then varied only
        # between 1.750 and 2.646 across the whole panel while the denominator
        # varied between 0.150 and 32.1, so ``skill_score`` was very nearly a
        # monotone re-encoding of column dispersion -- it said almost nothing
        # about how well the completion model predicted the column.
        #
        # The correctly scoped numerator is the k=0 baseline's per-column MedAE
        # over that column's own hidden target cells.  It is denominated over
        # exactly the cells the leave-one-out baseline is denominated over, and
        # (unlike the one-probe evaluation ``row``) the column is genuinely
        # hidden there rather than revealed as a probe.  ``_skill_summary``
        # already consumes this same field correctly.
        skill_numerator_medae = float(baseline.per_column_hidden_medae[index])
        skill = compute_column_skill(matrix, index, skill_numerator_medae)
        # parity_medae_normalized now divides by MAD * 1.4826 rather than the
        # sample SD: with a median of ~7 models per column an SD is both
        # high-variance and destroyed by one outlier, while the MAD shares the
        # L1 geometry of the MedAE numerator.  NOTE: this stays algebraically
        # close to the skill score, because the column-median MedAE IS
        # essentially the column MAD.  The two are reported together
        # deliberately; they are not independent evidence.
        if np.isfinite(col_dispersion) and col_dispersion > 0:
            parity_medae_normalized = parity_medae / col_dispersion
        else:
            parity_medae_normalized = None
        # Previous SD-scaled value, retained so published artifacts stay diffable.
        if np.isfinite(col_sd) and col_sd > 0:
            parity_medae_normalized_sd_legacy = parity_medae / col_sd
        else:
            parity_medae_normalized_sd_legacy = None
        informativeness.append(
            {
                "evaluation_index": index,
                "evaluation_id": evaluation_id,
                "suite_id": suite,
                "metric": metric,
                "models_with_score": int(np.sum(np.isfinite(matrix[:, index]))),
                "model_coverage": float(np.mean(np.isfinite(matrix[:, index]))),
                "parity_medae": parity_medae,
                "hidden_only_medae": row["hidden_only"]["medae"],  # type: ignore[index]
                "model_average_mae": row["model_average"]["mae"],  # type: ignore[index]
                # Global pooled baseline (column-median over ALL columns).  Kept for
                # backward compatibility so old artifacts remain comparable.
                "improvement_over_column_median": baseline_medae - parity_medae,
                # Per-column in-sample column-median baseline MedAE (legacy).
                "column_baseline_medae": col_baseline if np.isfinite(col_baseline) else None,
                "skill_in_sample_matrixwide_numerator_DEPRECATED": (
                    skill_in_sample_matrixwide_numerator_deprecated
                ),
                # Per-column leave-one-out baseline MedAE and robust dispersion.
                "column_loo_baseline_medae": _finite(col_loo_baseline),
                "column_robust_dispersion": _finite(col_dispersion),
                # The exact numerator behind skill_score, and the scope it was
                # measured over.  Both are written so no consumer has to guess
                # which error a skill number was built from; a matrix-wide
                # numerator here would be the FIX 5 bug returning.
                "skill_numerator_medae": _finite(skill_numerator_medae),
                "skill_numerator_scope": "transductive_all_known_k0_per_column",
                # Primary field: MedAE_model / MedAE_LOO_baseline.  No pole at
                # the origin and symmetric on a log scale.
                "medae_ratio_to_loo_baseline": skill.medae_ratio,
                # Bounded skill score in [-1, 1]; raw (unbounded) value kept
                # alongside; None when the column is excluded.
                "skill_score": skill.skill_score,
                "skill_score_raw": skill.skill_score_raw,
                "skill_excluded_below_noise_floor": skill.excluded_below_noise_floor,
                "skill_exclusion_reason": skill.exclusion_reason,
                # Dispersion-normalized error: MedAE / (MAD * 1.4826).
                "parity_medae_normalized": parity_medae_normalized,
                "parity_medae_normalized_sd_legacy": parity_medae_normalized_sd_legacy,
            }
        )
    informativeness.sort(key=lambda row: (float(row["parity_medae"]), int(row["evaluation_index"])))
    for position, row in enumerate(informativeness, start=1):
        row["informativeness_rank"] = position

    args.informativeness_output.parent.mkdir(parents=True, exist_ok=True)
    # Shared CSV schema.  ``scripts/replay_lofo_matched_cells.py`` rewrites the
    # matched-cell and skill columns of this same file from the held-out LOFO
    # replay; the two writers must agree on the header, so INFORMATIVENESS_FIELDS
    # there is kept identical to this list.  The matched_cell_* columns cannot be
    # produced from the transductive track and are written empty here -- re-run
    # `python scripts/replay_lofo_matched_cells.py --write-informativeness`
    # after this experiment to fill them.
    fields = [
        "informativeness_rank", "evaluation_id", "suite_id", "metric",
        "models_with_score", "model_coverage", "parity_medae",
        "hidden_only_medae", "model_average_mae",
        "improvement_over_column_median",
        "column_baseline_medae",
        "skill_in_sample_matrixwide_numerator_DEPRECATED",
        "column_loo_baseline_medae", "column_robust_dispersion",
        "matched_cell_n", "matched_cell_k0_medae", "matched_cell_greedy_medae",
        "matched_cell_greedy_beats_k0",
        "skill_numerator_medae", "skill_numerator_scope",
        "medae_ratio_to_loo_baseline",
        "skill_score", "skill_score_raw",
        "skill_excluded_below_noise_floor", "skill_exclusion_reason",
        "parity_medae_normalized", "parity_medae_normalized_sd_legacy",
    ]
    with args.informativeness_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(informativeness)

    payload = {
        "schema_version": 1,
        "protocol": "benchpress_all_known_probe_cells_zero_error_plus_strict_diagnostics_v1",
        "matrix": {
            "n_models": len(models),
            "n_evaluations": len(evaluations),
            "n_observed": int(np.sum(np.isfinite(matrix))),
            "density": float(np.mean(np.isfinite(matrix))),
            "suite_coverage": suite_coverage,
        },
        "configuration": {
            "rank": args.rank,
            "max_probes": args.max_probes,
            "random_repeats": args.random_repeats,
            "seed": args.seed,
            "workers": args.workers,
            "greedy_objective": "pooled all-known-cell MedAE including exact probe cells",
            "random_seed_formula": "RandomState((seed + repeat) * 100000)",
        },
        "baseline": _summary(baseline),
        "all_known_greedy": trajectory,
        "random_global_prefixes": random_rows,
        "heldout_model": {
            "train_fraction": 0.7,
            "train_models": [models[i] for i in train_indices],
            "validation_models": [models[i] for i in validation_indices],
            "split_mode": split_info.get("split_mode", args.split_mode),
            "train_selected_trajectory": heldout_train_trajectory,
            "validation": [
                {
                    "step": step["step"],
                    "added_evaluation_id": step["added_evaluation_id"],
                    "probe_ids": step["probe_ids"],
                    **result,
                }
                for step, result in zip(heldout_train_trajectory, heldout_results)
            ],
            # FIX 4: out-of-sample controls for the greedy curve above.
            "k0_baseline": heldout_k0,
            "random": heldout_random_rows,
            "random_repeats": args.heldout_random_repeats,
        },
        "heldout_leave_one_family_out": (
            {
                **lofo_info,
                "max_probes": min(args.lofo_max_probes, len(evaluations)),
                "random_repeats": args.heldout_random_repeats,
                "folds": lofo_fold_payloads,
            }
            if args.split_mode == "leave_one_family_out"
            else None
        ),
        "skill": {
            "noise_floor_dispersion": SKILL_NOISE_FLOOR_DISPERSION,
            "k0_baseline": _skill_summary(matrix, baseline, evaluations, seed=args.seed),
            "greedy_final": _skill_summary(
                matrix, greedy_final_evaluation, evaluations, seed=args.seed
            ),
        },
        "informativeness": informativeness,
        "provenance": {
            "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "benchpress_commit_audited": "0a684b63ee0e4a401cb907a3827a82ea997d74c4",
        },
        "caveats": [
            "The hero-parity curve is transductive and includes revealed probes as zero-error targets.",
            "The held-out-model primary curve excludes revealed probes and isolates each validation row.",
            (
                "The headline skill statistic is the fraction of evaluation columns with "
                "positive leave-one-out skill plus a percentile bootstrap CI over columns. "
                "Means and medians of the raw skill ratio are not reported because that "
                "ratio is unbounded below and is dominated by the tightest column."
            ),
            (
                "skill_score and parity_medae_normalized are not independent evidence: "
                "the column-median MedAE is essentially the column MAD, so the two differ "
                "mainly by the MAD-to-SD scale factor."
            ),
            (
                "A single 70/30 family-blocked holdout leaves few independent validation "
                "models and its size swings with the seed; the leave-one-family-out block "
                "validates every model exactly once and is the split to prefer."
            ),
            (
                "The current matrix is suite-blocked across "
                f"{suite_coverage['n_represented_suites']} represented suites; "
                f"{suite_coverage['n_models_with_all_represented_suites']} of {len(models)} "
                "supported models have at least one observed score in every represented suite."
            ),
            "Normalized F1, Pearson r, and robustness index are not a validated common clinical utility scale.",
            "No low-cost curve is claimed because pathology acquisition/runtime costs have not been audited.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "informativeness": str(args.informativeness_output)}, indent=2))


if __name__ == "__main__":
    main()
