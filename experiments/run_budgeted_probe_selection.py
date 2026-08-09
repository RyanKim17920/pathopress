#!/usr/bin/env python3
"""Select pathology benchmark probes under source-backed measured budgets."""

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


# A process pool must not multiply BLAS threads.  Keep these assignments before
# importing NumPy or PathoPress matrix-completion modules.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.budgeted_probes import (  # noqa: E402
    exact_search,
    greedy_search,
    load_json,
    load_receipts,
    random_feasible_prefixes,
    screen_candidates,
    set_burden,
    validate_budget,
)
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.probe_compression import (  # noqa: E402
    objective_value,
    predict_all_known,
    predict_heldout_models,
    score_predictions,
)
from pathopress.probes import (  # noqa: E402
    family_blocked_model_split,
    random_model_split,
)


def _sha256(path: Path) -> str:
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(path.glob("*.json")):
            digest.update(child.name.encode("utf-8"))
            digest.update(child.read_bytes())
        return digest.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _finite(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_finite(value) for value in payload]
    if isinstance(payload, (float, np.floating)) and not np.isfinite(payload):
        return None
    return payload


def _all_known_job(
    job: tuple[np.ndarray, tuple[int, ...], int, str, float]
) -> tuple[float, dict[str, Any]]:
    matrix, probes, rank, objective, margin = job
    metrics = score_predictions(
        predict_all_known(matrix, probes, rank=rank),
        pairwise_margin=margin,
        ranking_scope="all_target" if objective == "pairwise_margin_error" else "at_least_one_hidden",
    )
    return objective_value(metrics, objective), _finite(metrics)


def _heldout_job(
    job: tuple[np.ndarray, tuple[int, ...], tuple[int, ...], tuple[int, ...], int, str, float]
) -> tuple[float, dict[str, Any]]:
    matrix, probes, targets, context, rank, objective, margin = job
    metrics = score_predictions(
        predict_heldout_models(matrix, probes, targets, context, rank=rank),
        pairwise_margin=margin,
        ranking_scope="hidden_only" if objective == "pairwise_margin_error" else "at_least_one_hidden",
    )
    return objective_value(metrics, objective), _finite(metrics)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "data/scores.csv")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/tasks.csv")
    parser.add_argument(
        "--burden", type=Path,
        default=ROOT / "data/evaluation_burden_measurements_v1.example.json",
    )
    parser.add_argument(
        "--budget", type=Path, default=ROOT / "data/probe_budget_v1.example.json"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/budgeted_probe_selection_rank1.json",
    )
    parser.add_argument(
        "--objective", choices=("medae", "pairwise_margin_error"), default="medae"
    )
    parser.add_argument("--search", choices=("greedy", "exact"), default="greedy")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--ranking-margin", type=float, default=5.0)
    parser.add_argument("--max-probes", type=int, default=10)
    parser.add_argument("--max-subsets", type=int, default=200_000)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=("random", "family_blocked"),
        default="family_blocked",
        help="Model split strategy for held-out validation (default: family_blocked)",
    )
    parser.add_argument("--missing-policy", choices=("error", "exclude"), default="exclude")
    parser.add_argument(
        "--one-per-task-identity", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1))
    )
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    if args.rank < 1 or args.max_probes < 1 or args.max_subsets < 1:
        parser.error("rank, max-probes, and max-subsets must be positive")
    if args.random_repeats < 1:
        parser.error("--random-repeats must be positive")
    return args


def _task_identities(path: Path, evaluation_ids: list[str]) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["evaluation_id"]: row for row in csv.DictReader(handle)}
    missing = sorted(set(evaluation_ids) - set(rows))
    if missing:
        raise ValueError(f"retained evaluations lack task metadata: {missing}")
    return {evaluation_id: rows[evaluation_id]["task_identity_id"] for evaluation_id in evaluation_ids}


def run(args: argparse.Namespace) -> dict[str, Any]:
    ledger = load_receipts(args.burden)
    budget = validate_budget(load_json(args.budget))
    matrix, models, evaluations = filter_matrix(*make_matrix(load_scores(args.scores)))
    identity_by_id = _task_identities(args.tasks, evaluations)
    eligible, exclusions = screen_candidates(
        evaluations, ledger, budget, missing_policy=args.missing_policy,
    )
    common: dict[str, Any] = {
        "schema_version": "pathopress-budgeted-probe-selection-v1",
        "status": "complete" if eligible else "insufficient_cost_coverage",
        "inputs": {
            "scores": _display(args.scores),
            "scores_sha256": _sha256(args.scores),
            "tasks": _display(args.tasks),
            "tasks_sha256": _sha256(args.tasks),
            "burden": _display(args.burden),
            "burden_sha256": _sha256(args.burden),
            "budget": _display(args.budget),
            "budget_sha256": _sha256(args.budget),
            "runner_sha256": _sha256(Path(__file__)),
            "budgeted_probes_sha256": _sha256(ROOT / "src/pathopress/budgeted_probes.py"),
        },
        "configuration": {
            "scenario": budget.get("scenario"),
            "objective": args.objective,
            "prediction_rank": args.rank,
            "ranking_margin": args.ranking_margin,
            "search": args.search,
            "max_probes": args.max_probes,
            "max_subsets": args.max_subsets,
            "missing_policy": args.missing_policy,
            "one_per_task_identity": args.one_per_task_identity,
            "seed": args.seed,
            "workers": args.workers,
            "worker_cap": 4,
            "blas_threads_per_worker": 1,
            "tie_break": "objective within 1e-12; normalized budget pressure; canonical cost vector; lexicographic evaluation IDs",
        },
        "budget": budget,
        "matrix": {
            "shape": list(matrix.shape),
            "n_observed": int(np.isfinite(matrix).sum()),
            "n_retained_evaluations": len(evaluations),
        },
        "coverage": {
            "n_retained_candidates": len(evaluations),
            "n_canonical_receipts": len(ledger["measurements"]),
            "n_eligible_candidates": len(eligible),
            "eligible_evaluation_ids": eligible,
            "n_excluded_candidates": len(exclusions),
            "exclusions": exclusions,
        },
    }
    if not eligible:
        common["reason"] = (
            "No retained evaluation has complete accepted burden evidence under all active "
            "numeric, capacity, access, and license constraints. No selection or chart is valid."
        )
        return common

    if args.split_mode == "family_blocked":
        model_metadata_path = ROOT / "data" / "model_metadata.csv"
        train_indices, validation_indices, split_info = family_blocked_model_split(
            models, model_metadata_path=model_metadata_path, seed=args.seed
        )
    else:
        train_indices, validation_indices, split_info = random_model_split(
            models, seed=args.seed
        )
    index_by_id = {evaluation_id: index for index, evaluation_id in enumerate(evaluations)}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        def evaluate_all_known(sets: list[tuple[str, ...]]) -> list[tuple[float, dict[str, Any]]]:
            return list(executor.map(
                _all_known_job,
                [
                    (
                        matrix, tuple(index_by_id[value] for value in selected),
                        args.rank, args.objective, args.ranking_margin,
                    )
                    for selected in sets
                ],
            ))

        def evaluate_heldout(sets: list[tuple[str, ...]]) -> list[tuple[float, dict[str, Any]]]:
            return list(executor.map(
                _heldout_job,
                [
                    (
                        matrix, tuple(index_by_id[value] for value in selected),
                        validation_indices, train_indices, args.rank, args.objective,
                        args.ranking_margin,
                    )
                    for selected in sets
                ],
            ))

        baseline_all_known = evaluate_all_known([()])[0]
        baseline_heldout = evaluate_heldout([()])[0]
        if args.search == "greedy":
            selection = greedy_search(
                eligible, identity_by_id, ledger, budget, evaluate_all_known,
                max_probes=args.max_probes,
                one_per_identity=args.one_per_task_identity,
            )
            selected_sets = [tuple(row["evaluation_ids"]) for row in selection["trajectory"]]
        else:
            selection = exact_search(
                eligible, identity_by_id, ledger, budget, evaluate_all_known,
                max_probes=args.max_probes, max_subsets=args.max_subsets,
                one_per_identity=args.one_per_task_identity,
            )
            selected_sets = [] if selection["optimum"] is None else [
                tuple(selection["optimum"]["evaluation_ids"])
            ]
        validation_results = evaluate_heldout(selected_sets) if selected_sets else []
        random_prefixes = random_feasible_prefixes(
            eligible, identity_by_id, ledger, budget,
            max_probes=args.max_probes, repeats=args.random_repeats, seed=args.seed,
            one_per_identity=args.one_per_task_identity,
        )
        random_rows = []
        for repeat, prefixes in enumerate(random_prefixes):
            all_results = evaluate_all_known(prefixes) if prefixes else []
            heldout_results = evaluate_heldout(prefixes) if prefixes else []
            for selected, all_result, heldout_result in zip(prefixes, all_results, heldout_results):
                random_rows.append({
                    "repeat": repeat,
                    "k": len(selected),
                    "evaluation_ids": list(selected),
                    "burden": set_burden(selected, ledger, budget),
                    "all_known": {"objective_loss": all_result[0], "metrics": all_result[1]},
                    "heldout": {"objective_loss": heldout_result[0], "metrics": heldout_result[1]},
                })

    validation = [
        {
            "evaluation_ids": list(selected),
            "objective_loss": result[0],
            "metrics": result[1],
        }
        for selected, result in zip(selected_sets, validation_results)
    ]
    common.update({
        "split": {
            "seed": args.seed,
            "split_mode": split_info.get("split_mode", args.split_mode),
            "train_model_ids": [models[index] for index in train_indices],
            "validation_model_ids": [models[index] for index in validation_indices],
        },
        "selection": selection,
        "zero_probe_baseline": {
            "all_known": {
                "objective_loss": baseline_all_known[0],
                "metrics": baseline_all_known[1],
            },
            "heldout": {
                "objective_loss": baseline_heldout[0],
                "metrics": baseline_heldout[1],
            },
        },
        "heldout_validation": validation,
        "random_feasible_baseline": random_rows,
    })
    return _finite(common)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    _write_json_atomic(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "eligible_candidates": payload["coverage"]["n_eligible_candidates"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
