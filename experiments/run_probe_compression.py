#!/usr/bin/env python3
"""Run BenchPress-style probe-compression curves and ranking objectives."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.probe_compression import (  # noqa: E402
    ProbePredictions,
    SCORE_RECONSTRUCTION_PAIRWISE_DIAGNOSTIC_MARGIN,
    candidate_prefixes,
    objective_value,
    predict_all_known,
    predict_heldout_models,
    rank_prune_trajectory,
    score_predictions,
)


def _finite(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _metric_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: _finite(value) for key, value in metrics.items()}


def _eval_all_known(
    job: tuple[np.ndarray, tuple[int, ...], int, float, str]
) -> dict[str, Any]:
    matrix, probes, rank, pairwise_margin, ranking_scope = job
    return score_predictions(
        predict_all_known(matrix, probes, rank=rank),
        pairwise_margin=pairwise_margin,
        ranking_scope=ranking_scope,
    )


def _eval_heldout(
    job: tuple[
        np.ndarray, tuple[int, ...], tuple[int, ...], tuple[int, ...], int, float, str
    ]
) -> dict[str, Any]:
    matrix, probes, targets, context, rank, pairwise_margin, ranking_scope = job
    return score_predictions(
        predict_heldout_models(matrix, probes, targets, context, rank=rank),
        pairwise_margin=pairwise_margin,
        ranking_scope=ranking_scope,
    )


def _parallel_all_known(
    executor: ProcessPoolExecutor,
    matrix: np.ndarray,
    probe_sets: list[tuple[int, ...]],
    rank: int,
    *,
    pairwise_margin: float = 2.0,
    ranking_scope: str = "at_least_one_hidden",
) -> list[dict[str, Any]]:
    return list(executor.map(
        _eval_all_known,
        [
            (matrix, probes, rank, pairwise_margin, ranking_scope)
            for probes in probe_sets
        ],
    ))


def _parallel_heldout(
    executor: ProcessPoolExecutor,
    matrix: np.ndarray,
    probe_sets: list[tuple[int, ...]],
    targets: tuple[int, ...],
    context: tuple[int, ...],
    rank: int,
    *,
    pairwise_margin: float = 2.0,
    ranking_scope: str = "at_least_one_hidden",
) -> list[dict[str, Any]]:
    return list(
        executor.map(
            _eval_heldout,
            [
                (
                    matrix, probes, targets, context, rank,
                    pairwise_margin, ranking_scope,
                )
                for probes in probe_sets
            ],
        )
    )


def _greedy(
    executor: ProcessPoolExecutor,
    matrix: np.ndarray,
    candidates: list[int],
    *,
    objective: str,
    max_k: int,
    rank: int,
    evaluations: list[str],
    label: str,
    pairwise_margin: float = 2.0,
    ranking_scope: str = "at_least_one_hidden",
) -> list[dict[str, Any]]:
    selected: list[int] = []
    remaining = candidates.copy()
    rows = []
    for k in range(1, min(max_k, len(candidates)) + 1):
        probe_sets = [tuple([*selected, candidate]) for candidate in remaining]
        results = _parallel_all_known(
            executor,
            matrix,
            probe_sets,
            rank,
            pairwise_margin=pairwise_margin,
            ranking_scope=ranking_scope,
        )
        losses = [objective_value(result, objective) for result in results]
        best_pos = min(range(len(remaining)), key=lambda pos: (losses[pos], pos))
        candidate_rows = [
            {
                "evaluation_index": candidate,
                "evaluation_id": evaluations[candidate],
                "objective_value": _finite(loss),
            }
            for candidate, loss in zip(remaining, losses)
        ]
        selected.append(remaining.pop(best_pos))
        rows.append({
            "k": k,
            "added_evaluation_index": selected[-1],
            "added_evaluation_id": evaluations[selected[-1]],
            "probe_indices": selected.copy(),
            "probe_ids": [evaluations[index] for index in selected],
            "selection_objective": objective,
            "selection_metrics": _metric_dict(results[best_pos]),
            "candidate_results": candidate_rows,
        })
        print(f"{label} {objective} k={k}: {evaluations[selected[-1]]} loss={losses[best_pos]:.6f}", flush=True)
    return rows


def _random_curves(
    executor: ProcessPoolExecutor,
    matrix: np.ndarray,
    candidates: list[int],
    *,
    max_k: int,
    repeats: int,
    seed: int,
    rank: int,
    evaluations: list[str],
    heldout: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    pairwise_margin: float = 2.0,
    ranking_scope: str = "at_least_one_hidden",
) -> list[dict[str, Any]]:
    prefixes = candidate_prefixes(
        candidates,
        max_probes=min(max_k, len(candidates)),
        repeats=repeats,
        seed=seed,
    )
    sets = [probes for repeat in prefixes for probes in repeat]
    if heldout is None:
        results = _parallel_all_known(
            executor, matrix, sets, rank,
            pairwise_margin=pairwise_margin,
            ranking_scope=ranking_scope,
        )
    else:
        targets, context = heldout
        results = _parallel_heldout(
            executor, matrix, sets, targets, context, rank,
            pairwise_margin=pairwise_margin,
            ranking_scope=ranking_scope,
        )
    rows = []
    cursor = 0
    for repeat, repeat_prefixes in enumerate(prefixes):
        for k, probes in enumerate(repeat_prefixes, 1):
            rows.append({
                "repeat": repeat,
                "k": k,
                "probe_indices": list(probes),
                "probe_ids": [evaluations[index] for index in probes],
                "metrics": _metric_dict(results[cursor]),
            })
            cursor += 1
    return rows


def _pruned_candidates(path: Path, evaluations: list[str], keep: int) -> tuple[list[int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectory = payload.get("all_known_greedy")
    if trajectory is None:
        trajectory = payload["all_known"]["greedy_trajectory"]
    summary = rank_prune_trajectory(
        trajectory,
        evaluations,
        keep_count=keep,
        score_key="parity_medae",
    )
    kept = summary["kept_ids"]
    return [evaluations.index(value) for value in kept], {
        "eval_protocol": "all_known_greedy_rank_pruning_v1",
        "method": "aggregate_normalized_candidate_rank_over_all_available_all_known_medae_greedy_steps",
        "source": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "keep_count": keep,
        "source_steps_used": summary["source_steps_used"],
        "rank_direction": "lower score is better within each greedy context",
        "missing_after_selection": "Candidates selected by greedy are ranked only until their selected step; later contexts omit already-selected candidates.",
        "evaluation_ids": kept,
        "removed_evaluation_ids": summary["removed_ids"],
        "ranked_steps": summary["ranked_steps"],
        "by_candidate": summary["by_candidate"],
        "error_informed": True,
        "distinction": "This computational pruning is separate from the pre-error feasibility allowlist.",
    }


def _raw_rows(
    prediction: ProbePredictions,
    models: list[str],
    evaluations: list[str],
    common: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for i, j in np.argwhere(prediction.target_mask):
        rows.append({
            **common,
            "model_id": models[int(i)],
            "evaluation_id": evaluations[int(j)],
            "actual_normalized_score": float(prediction.actual[i, j]),
            "predicted_normalized_score": float(prediction.predicted[i, j]),
            "is_revealed_probe_cell": bool(prediction.revealed_mask[i, j]),
            "is_hidden_prediction": bool(prediction.heldout_mask[i, j]),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "data/scores.csv")
    parser.add_argument("--allowlist", type=Path, default=ROOT / "data/low_friction_allowlist_v2_top25.json")
    parser.add_argument("--previous-probes", type=Path, default=ROOT / "experiments/probe_selection_results_rank1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/probe_compression_rank1.json")
    parser.add_argument("--raw-output", type=Path, default=ROOT / "outputs/probe_compression_selected_raw_rank1.csv")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--max-any-k", type=int, default=10)
    parser.add_argument("--max-random-k", type=int, default=10)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--pruned-keep", type=int, default=30)
    parser.add_argument("--ranking-margin", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, min(28, (os.cpu_count() or 2) - 1)))
    parser.add_argument(
        "--reuse-any-score-curves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse hash-matched unrestricted score curves/raw rows from --output; ranking curves are always regenerated",
    )
    parser.add_argument(
        "--ranking-random-only",
        action="store_true",
        help="Enrich an existing current-schema output with exact margin-5 random ranking curves",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores)
    matrix, models, evaluations = filter_matrix(*make_matrix(scores))
    scores_sha256 = hashlib.sha256(args.scores.read_bytes()).hexdigest()
    reusable_any = None
    reusable_any_raw: list[dict[str, Any]] = []
    if args.reuse_any_score_curves and args.output.exists() and args.raw_output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        previous_any = previous.get("curves", {}).get("any_candidate")
        previous_config = previous.get("configuration", {})
        if (
            previous_config.get("scores_sha256") == scores_sha256
            and previous_config.get("matrix_shape") == list(matrix.shape)
            and int(previous_config.get("prediction_rank", -1)) == args.rank
            and isinstance(previous_any, dict)
            and previous_any.get("candidate_ids") == evaluations
            and all(
                len(previous_any.get(key, [])) >= args.max_any_k
                for key in (
                    "all_known_greedy_medae", "heldout_greedy_medae",
                    "all_known_greedy_medape", "heldout_greedy_medape",
                )
            )
        ):
            reusable_any = copy.deepcopy(previous_any)
            with args.raw_output.open(newline="", encoding="utf-8") as handle:
                reusable_any_raw = [
                    row for row in csv.DictReader(handle)
                    if row["candidate_mode"] == "any_candidate"
                ]
    allow_payload = json.loads(args.allowlist.read_text(encoding="utf-8"))
    allow_indices = [evaluations.index(value) for value in allow_payload["evaluation_ids"]]
    any_indices = list(range(len(evaluations)))
    pruned_indices, pruning = _pruned_candidates(args.previous_probes, evaluations, args.pruned_keep)

    if args.ranking_random_only:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        config = existing.get("configuration", {})
        if (
            config.get("scores_sha256") != scores_sha256
            or config.get("allowlist_sha256") != hashlib.sha256(args.allowlist.read_bytes()).hexdigest()
            or float(config.get("ranking_margin", -1)) != args.ranking_margin
        ):
            raise ValueError("existing compression artifact does not match current score/allowlist/margin configuration")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for candidate_mode, candidates in (
                ("any_candidate", any_indices),
                ("pre_error_low_friction_allowlist", allow_indices),
            ):
                existing["ranking_aware"][candidate_mode]["all_known_random"] = _random_curves(
                    executor,
                    matrix,
                    candidates,
                    max_k=min(args.max_random_k, len(candidates)),
                    repeats=args.random_repeats,
                    seed=args.seed,
                    rank=args.rank,
                    evaluations=evaluations,
                    pairwise_margin=args.ranking_margin,
                    ranking_scope="all_target",
                )
        existing["configuration"]["medape_epsilon"] = 1e-6
        existing["configuration"]["medape_semantics"] = "median(100*absolute_error/abs(actual)); targets with abs(actual)<=1e-6 excluded"
        existing["configuration"]["score_curve_pairwise_diagnostic_margin"] = SCORE_RECONSTRUCTION_PAIRWISE_DIAGNOSTIC_MARGIN
        existing["configuration"]["score_curve_pairwise_diagnostic_semantics"] = "The pairwise_margin=2 fields nested under score-reconstruction curves are ancillary diagnostics only, never the ranking selection objective. Dedicated ranking_aware curves use ranking_margin=5."
        # Keep the public ranking schema limited to the two upstream-comparable
        # candidate universes.  Pruning diagnostics live in ``pruning`` above.
        existing["ranking_aware"].pop("error_informed_pruned_diagnostic", None)
        args.output.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        print(f"enriched {args.output} with exact margin-{args.ranking_margin:g} random ranking curves")
        return

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(models))
    n_train = min(max(1, round(0.7 * len(models))), len(models) - 1)
    train_indices = tuple(sorted(int(value) for value in order[:n_train]))
    validation_indices = tuple(sorted(int(value) for value in order[n_train:]))
    train_matrix = matrix[list(train_indices)]
    train_models = [models[index] for index in train_indices]

    payload: dict[str, Any] = {
        "schema_version": 1,
        "configuration": {
            "prediction_rank": args.rank,
            "matrix_shape": list(matrix.shape),
            "n_observed": int(np.isfinite(matrix).sum()),
            "scores_sha256": scores_sha256,
            "allowlist_sha256": hashlib.sha256(args.allowlist.read_bytes()).hexdigest(),
            "all_known_semantics": "Each target row retains selected probe cells; all other rows remain visible; revealed cells enter parity metrics at zero error.",
            "heldout_semantics": "Probes selected on training rows only; each validation row is then completed in isolation from fixed training context plus its selected probes.",
            "medape_semantics": "median(100*absolute_error/abs(actual)); zero actual targets excluded",
            "medape_epsilon": 1e-6,
            "ranking_semantics": "BenchPress ranking objective: median per-evaluation pairwise accuracy at true normalized-score gap >=5. All-known includes all target cells (including probe/probe pairs); holdout non-probe contains hidden cells only; with-probe-zero includes every validation target.",
            "ranking_margin": args.ranking_margin,
            "score_curve_pairwise_diagnostic_margin": SCORE_RECONSTRUCTION_PAIRWISE_DIAGNOSTIC_MARGIN,
            "score_curve_pairwise_diagnostic_semantics": "The pairwise_margin=2 fields nested under score-reconstruction curves are ancillary diagnostics only, never the ranking selection objective. Dedicated ranking_aware curves use ranking_margin=5.",
            "candidate_tie_break": "stable candidate order",
            "unrestricted_curve_limit": args.max_any_k,
            "limit_reason": "full candidate rescoring is exact at each reported k; the reported curve is deliberately bounded because each set masks and completes every model row",
        },
        "allowlist": allow_payload,
        "pruning": pruning,
        "split": {
            "seed": args.seed,
            "train_model_indices": list(train_indices),
            "train_model_ids": train_models,
            "validation_model_indices": list(validation_indices),
            "validation_model_ids": [models[index] for index in validation_indices],
        },
        "curves": {},
        "ranking_aware": {},
    }
    raw: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for candidate_mode, candidates, max_k in (
            ("any_candidate", any_indices, args.max_any_k),
            ("pre_error_low_friction_allowlist", allow_indices, args.max_any_k),
        ):
            if candidate_mode == "any_candidate" and reusable_any is not None:
                payload["curves"][candidate_mode] = reusable_any
                raw.extend(reusable_any_raw)
                print(
                    "reused hash-matched unrestricted score curves and raw rows; "
                    "ranking trajectories will be regenerated",
                    flush=True,
                )
                continue
            mode_result: dict[str, Any] = {"candidate_indices": candidates, "candidate_ids": [evaluations[i] for i in candidates]}
            for objective in ("medae", "medape"):
                if candidate_mode == "any_candidate" and objective == "medae":
                    prior = json.loads(args.previous_probes.read_text(encoding="utf-8"))
                    prior_all = prior["all_known_greedy"][:max_k]
                    prior_train = prior["heldout_model"]["train_selected_trajectory"][:max_k]
                    all_sets = [tuple(row["probe_indices"]) for row in prior_all]
                    train_sets = [tuple(row["probe_indices"]) for row in prior_train]
                    all_metrics = _parallel_all_known(executor, matrix, all_sets, args.rank)
                    train_metrics = _parallel_all_known(executor, train_matrix, train_sets, args.rank)
                    all_greedy = [
                        {
                            "k": int(row["step"]),
                            "added_evaluation_index": row["added_evaluation_index"],
                            "added_evaluation_id": row["added_evaluation_id"],
                            "probe_indices": row["probe_indices"],
                            "probe_ids": row["probe_ids"],
                            "selection_objective": objective,
                            "selection_metrics": _metric_dict(metrics),
                            "candidate_results": row["candidate_results"],
                            "reused_exact_search": str(args.previous_probes.relative_to(ROOT)),
                        }
                        for row, metrics in zip(prior_all, all_metrics)
                    ]
                    train_greedy = [
                        {
                            "k": int(row["step"]),
                            "added_evaluation_index": row["added_evaluation_index"],
                            "added_evaluation_id": row["added_evaluation_id"],
                            "probe_indices": row["probe_indices"],
                            "probe_ids": row["probe_ids"],
                            "selection_objective": objective,
                            "selection_metrics": _metric_dict(metrics),
                            "candidate_results": row["candidate_results"],
                            "reused_exact_search": str(args.previous_probes.relative_to(ROOT)),
                        }
                        for row, metrics in zip(prior_train, train_metrics)
                    ]
                    print("reused exact unrestricted MedAE trajectories through k=10", flush=True)
                else:
                    all_greedy = _greedy(
                        executor, matrix, candidates, objective=objective, max_k=max_k,
                        rank=args.rank, evaluations=evaluations, label=f"all-known/{candidate_mode}",
                    )
                    train_greedy = _greedy(
                        executor, train_matrix, candidates, objective=objective, max_k=max_k,
                        rank=args.rank, evaluations=evaluations, label=f"heldout-train/{candidate_mode}",
                    )
                validation_sets = [tuple(row["probe_indices"]) for row in train_greedy]
                validation_results = _parallel_heldout(
                    executor, matrix, validation_sets, validation_indices, train_indices, args.rank
                )
                heldout_greedy = []
                for selection, validation in zip(train_greedy, validation_results):
                    heldout_greedy.append({
                        **selection,
                        "training_selection_metrics": selection.pop("selection_metrics"),
                        "validation_metrics": _metric_dict(validation),
                    })
                mode_result[f"all_known_greedy_{objective}"] = all_greedy
                mode_result[f"heldout_greedy_{objective}"] = heldout_greedy
                for protocol, steps in (("all_known", all_greedy), ("heldout", heldout_greedy)):
                    for step in steps:
                        probes = tuple(step["probe_indices"])
                        prediction = (
                            predict_all_known(matrix, probes, rank=args.rank)
                            if protocol == "all_known"
                            else predict_heldout_models(matrix, probes, validation_indices, train_indices, rank=args.rank)
                        )
                        raw.extend(_raw_rows(prediction, models, evaluations, {
                            "protocol": protocol, "candidate_mode": candidate_mode,
                            "method": "greedy", "selection_objective": objective, "k": step["k"],
                        }))
            mode_result["all_known_random"] = _random_curves(
                executor, matrix, candidates, max_k=min(args.max_random_k, len(candidates)),
                repeats=args.random_repeats, seed=args.seed, rank=args.rank, evaluations=evaluations,
            )
            mode_result["heldout_random"] = _random_curves(
                executor, matrix, candidates, max_k=min(args.max_random_k, len(candidates)),
                repeats=args.random_repeats, seed=args.seed, rank=args.rank, evaluations=evaluations,
                heldout=(validation_indices, train_indices),
            )
            payload["curves"][candidate_mode] = mode_result

        for candidate_mode, candidates in (
            ("any_candidate", any_indices),
            ("pre_error_low_friction_allowlist", allow_indices),
        ):
            all_known = _greedy(
                executor,
                matrix,
                candidates,
                objective="pairwise_margin_error",
                max_k=min(args.max_any_k, len(candidates)),
                rank=args.rank,
                evaluations=evaluations,
                label=f"ranking-all-known/{candidate_mode}",
                pairwise_margin=args.ranking_margin,
                ranking_scope="all_target",
            )
            train_selected = _greedy(
                executor,
                train_matrix,
                candidates,
                objective="pairwise_margin_error",
                max_k=min(args.max_any_k, len(candidates)),
                rank=args.rank,
                evaluations=evaluations,
                label=f"ranking-heldout-train/{candidate_mode}",
                pairwise_margin=args.ranking_margin,
                ranking_scope="all_target",
            )
            validation_sets = [tuple(row["probe_indices"]) for row in train_selected]
            validation_non_probe = _parallel_heldout(
                executor,
                matrix,
                validation_sets,
                validation_indices,
                train_indices,
                args.rank,
                pairwise_margin=args.ranking_margin,
                ranking_scope="hidden_only",
            )
            validation_with_probe_zero = _parallel_heldout(
                executor,
                matrix,
                validation_sets,
                validation_indices,
                train_indices,
                args.rank,
                pairwise_margin=args.ranking_margin,
                ranking_scope="all_target",
            )
            heldout = []
            for selected, non_probe, with_probe in zip(
                train_selected, validation_non_probe, validation_with_probe_zero
            ):
                selection_without_metrics = {
                    key: value
                    for key, value in selected.items()
                    if key != "selection_metrics"
                }
                heldout.append({
                    **selection_without_metrics,
                    "training_selection_metrics": selected["selection_metrics"],
                    "validation_non_probe": _metric_dict(non_probe),
                    "validation_with_probe_zero": _metric_dict(with_probe),
                })
            payload["ranking_aware"][candidate_mode] = {
                "eval_protocol_all_known": "all_known_probe_cells_zero_error_v1",
                "eval_protocol_holdout": "model_split_ranking_probe_validation_v1",
                "objective": f"margin{args.ranking_margin:g}_pairwise_ranking_accuracy",
                "margin": args.ranking_margin,
                "all_known_greedy": all_known,
                "heldout_greedy": heldout,
                "all_known_random": _random_curves(
                    executor,
                    matrix,
                    candidates,
                    max_k=min(args.max_random_k, len(candidates)),
                    repeats=args.random_repeats,
                    seed=args.seed,
                    rank=args.rank,
                    evaluations=evaluations,
                    pairwise_margin=args.ranking_margin,
                    ranking_scope="all_target",
                ),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    if raw:
        with args.raw_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(raw[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(raw)
    print(f"wrote {args.output} and {args.raw_output}")


if __name__ == "__main__":
    main()
