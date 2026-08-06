#!/usr/bin/env python3
"""Run BenchPress-style probe-compression curves and ranking objectives."""

from __future__ import annotations

import argparse
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
    candidate_prefixes,
    objective_value,
    predict_all_known,
    predict_heldout_models,
    score_predictions,
)


def _finite(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _metric_dict(metrics: dict[str, float | int]) -> dict[str, float | int | None]:
    return {key: _finite(value) for key, value in metrics.items()}


def _eval_all_known(job: tuple[np.ndarray, tuple[int, ...], int]) -> dict[str, float | int]:
    matrix, probes, rank = job
    return score_predictions(predict_all_known(matrix, probes, rank=rank))


def _eval_heldout(
    job: tuple[np.ndarray, tuple[int, ...], tuple[int, ...], tuple[int, ...], int]
) -> dict[str, float | int]:
    matrix, probes, targets, context, rank = job
    return score_predictions(
        predict_heldout_models(matrix, probes, targets, context, rank=rank)
    )


def _parallel_all_known(
    executor: ProcessPoolExecutor,
    matrix: np.ndarray,
    probe_sets: list[tuple[int, ...]],
    rank: int,
) -> list[dict[str, float | int]]:
    return list(executor.map(_eval_all_known, [(matrix, probes, rank) for probes in probe_sets]))


def _parallel_heldout(
    executor: ProcessPoolExecutor,
    matrix: np.ndarray,
    probe_sets: list[tuple[int, ...]],
    targets: tuple[int, ...],
    context: tuple[int, ...],
    rank: int,
) -> list[dict[str, float | int]]:
    return list(
        executor.map(
            _eval_heldout,
            [(matrix, probes, targets, context, rank) for probes in probe_sets],
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
) -> list[dict[str, Any]]:
    selected: list[int] = []
    remaining = candidates.copy()
    rows = []
    for k in range(1, min(max_k, len(candidates)) + 1):
        probe_sets = [tuple([*selected, candidate]) for candidate in remaining]
        results = _parallel_all_known(executor, matrix, probe_sets, rank)
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
) -> list[dict[str, Any]]:
    prefixes = candidate_prefixes(
        candidates,
        max_probes=min(max_k, len(candidates)),
        repeats=repeats,
        seed=seed,
    )
    sets = [probes for repeat in prefixes for probes in repeat]
    if heldout is None:
        results = _parallel_all_known(executor, matrix, sets, rank)
    else:
        targets, context = heldout
        results = _parallel_heldout(executor, matrix, sets, targets, context, rank)
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
    aggregate = {evaluation: 0.0 for evaluation in evaluations}
    appearances = {evaluation: 0 for evaluation in evaluations}
    for step in trajectory[:3]:
        ordered = sorted(
            step["candidate_results"],
            key=lambda row: (float(row["parity_medae"]), int(row["evaluation_index"])),
        )
        denominator = max(1, len(ordered) - 1)
        for rank_position, row in enumerate(ordered):
            evaluation = row["evaluation_id"]
            aggregate[evaluation] += rank_position / denominator
            appearances[evaluation] += 1
    ranked = sorted(
        evaluations,
        key=lambda evaluation: (
            aggregate[evaluation] / appearances[evaluation]
            if appearances[evaluation]
            else float("inf"),
            evaluations.index(evaluation),
        ),
    )
    kept = ranked[:keep]
    return [evaluations.index(value) for value in kept], {
        "method": "aggregate_normalized_candidate_rank_over_first_three_all_known_medae_greedy_steps",
        "source": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "keep_count": keep,
        "evaluation_ids": kept,
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
    parser.add_argument("--allowlist", type=Path, default=ROOT / "data/low_friction_allowlist_v1.json")
    parser.add_argument("--previous-probes", type=Path, default=ROOT / "experiments/probe_selection_results_rank1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/probe_compression_rank1.json")
    parser.add_argument("--raw-output", type=Path, default=ROOT / "outputs/probe_compression_selected_raw_rank1.csv")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--max-any-k", type=int, default=10)
    parser.add_argument("--max-random-k", type=int, default=10)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--pruned-keep", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, min(28, (os.cpu_count() or 2) - 1)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores)
    matrix, models, evaluations = filter_matrix(*make_matrix(scores))
    allow_payload = json.loads(args.allowlist.read_text(encoding="utf-8"))
    allow_indices = [evaluations.index(value) for value in allow_payload["evaluation_ids"]]
    any_indices = list(range(len(evaluations)))
    pruned_indices, pruning = _pruned_candidates(args.previous_probes, evaluations, args.pruned_keep)

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
            "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
            "allowlist_sha256": hashlib.sha256(args.allowlist.read_bytes()).hexdigest(),
            "all_known_semantics": "Each target row retains selected probe cells; all other rows remain visible; revealed cells enter parity metrics at zero error.",
            "heldout_semantics": "Probes selected on training rows only; each validation row is then completed in isolation from fixed training context plus its selected probes.",
            "medape_semantics": "median(100*absolute_error/abs(actual)); zero actual targets excluded",
            "ranking_semantics": "median across eligible evaluation columns; pairwise true margin=2 normalized points and top fraction=0.20",
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
            ("pre_error_low_friction_allowlist", allow_indices, len(allow_indices)),
        ):
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
            ("error_informed_pruned", pruned_indices),
            ("pre_error_low_friction_allowlist", allow_indices),
        ):
            payload["ranking_aware"][candidate_mode] = {}
            for objective in ("pairwise_margin_error", "top_fraction_error"):
                payload["ranking_aware"][candidate_mode][objective] = _greedy(
                    executor, matrix, candidates, objective=objective,
                    max_k=min(args.max_any_k, len(candidates)), rank=args.rank,
                    evaluations=evaluations, label=f"ranking/{candidate_mode}",
                )

        # One exact unpruned ranking budget.  Both objectives reuse the same
        # expensive prediction set; subsequent budgets are the separately
        # labelled error-informed pruned search above.
        first_sets = [(candidate,) for candidate in any_indices]
        first_metrics = _parallel_all_known(executor, matrix, first_sets, args.rank)
        payload["ranking_aware"]["any_candidate_exact_k1"] = {}
        for objective in ("pairwise_margin_error", "top_fraction_error"):
            losses = [objective_value(metrics, objective) for metrics in first_metrics]
            best = min(range(len(any_indices)), key=lambda pos: (losses[pos], pos))
            candidate = any_indices[best]
            payload["ranking_aware"]["any_candidate_exact_k1"][objective] = [{
                "k": 1,
                "added_evaluation_index": candidate,
                "added_evaluation_id": evaluations[candidate],
                "probe_indices": [candidate],
                "probe_ids": [evaluations[candidate]],
                "selection_objective": objective,
                "selection_metrics": _metric_dict(first_metrics[best]),
                "candidate_results": [
                    {"evaluation_index": index, "evaluation_id": evaluations[index], "objective_value": _finite(loss)}
                    for index, loss in zip(any_indices, losses)
                ],
            }]

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
