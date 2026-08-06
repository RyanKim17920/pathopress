#!/usr/bin/env python3
"""Run BenchPress-style probe-compression curves and ranking objectives."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import io
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


def _temporary_sibling(path: Path) -> Path:
    """Return the deterministic sibling used for an atomic artifact replace."""

    return path.with_name(f".{path.name}.tmp")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _checkpoint_identity(args: argparse.Namespace, scores_sha256: str) -> dict[str, Any]:
    return {
        "scores_sha256": scores_sha256,
        "allowlist_sha256": hashlib.sha256(args.allowlist.read_bytes()).hexdigest(),
        "previous_probes_sha256": hashlib.sha256(args.previous_probes.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prediction_rank": args.rank,
        "max_any_k": args.max_any_k,
        "max_random_k": args.max_random_k,
        "max_heldout_random_k": args.max_heldout_random_k,
        "max_ranking_random_k": args.max_ranking_random_k,
        "random_repeats": args.random_repeats,
        "pruned_keep": args.pruned_keep,
        "ranking_margin": args.ranking_margin,
        "seed": args.seed,
    }


def _load_phase_checkpoint(
    path: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"identity": identity, "curve_greedy": {}, "ranking": {}, "raw": []}
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("identity") != identity
        or checkpoint.get("identity_sha256") != _sha256_json(identity)
    ):
        return {"identity": identity, "curve_greedy": {}, "ranking": {}, "raw": []}
    if not all(key in checkpoint for key in ("curve_greedy", "ranking", "raw")):
        return {"identity": identity, "curve_greedy": {}, "ranking": {}, "raw": []}
    return checkpoint


def _save_phase_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["schema_version"] = 1
    checkpoint["identity_sha256"] = _sha256_json(checkpoint["identity"])
    _write_json_atomic(path, checkpoint)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            if rows:
                writer = csv.DictWriter(
                    handle, fieldnames=list(rows[0]), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _eval_all_known_with_predictions(
    job: tuple[np.ndarray, tuple[int, ...], int, float, str]
) -> tuple[dict[str, Any], ProbePredictions]:
    matrix, probes, rank, pairwise_margin, ranking_scope = job
    prediction = predict_all_known(matrix, probes, rank=rank)
    return (
        score_predictions(
            prediction,
            pairwise_margin=pairwise_margin,
            ranking_scope=ranking_scope,
        ),
        prediction,
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
    raw_writer: csv.DictWriter | None = None,
    models: list[str] | None = None,
    candidate_mode: str | None = None,
) -> list[dict[str, Any]]:
    prefixes = candidate_prefixes(
        candidates,
        max_probes=min(max_k, len(candidates)),
        repeats=repeats,
        seed=seed,
    )
    sets = [probes for repeat in prefixes for probes in repeat]
    if heldout is None:
        jobs = [
            (matrix, probes, rank, pairwise_margin, ranking_scope)
            for probes in sets
        ]
        if raw_writer is None:
            results = list(executor.map(_eval_all_known, jobs))
            predictions: list[ProbePredictions | None] = [None] * len(results)
        else:
            if models is None or candidate_mode is None:
                raise ValueError(
                    "models and candidate_mode are required when writing random raw rows"
                )
            detailed = list(executor.map(_eval_all_known_with_predictions, jobs))
            results = [result for result, _ in detailed]
            predictions = [prediction for _, prediction in detailed]
    else:
        if raw_writer is not None:
            raise ValueError("random raw rows currently support all-known only")
        targets, context = heldout
        results = _parallel_heldout(
            executor, matrix, sets, targets, context, rank,
            pairwise_margin=pairwise_margin,
            ranking_scope=ranking_scope,
        )
        predictions = [None] * len(results)
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
            prediction = predictions[cursor]
            if raw_writer is not None and prediction is not None:
                raw_writer.writerows(
                    _raw_rows(
                        prediction,
                        models or [],
                        evaluations,
                        {
                            "protocol": "all_known",
                            "candidate_mode": candidate_mode,
                            "method": "random_prefix",
                            "selection_objective": "none_random_baseline",
                            "repeat": repeat,
                            "k": k,
                        },
                    )
                )
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
    parser.add_argument(
        "--random-raw-output",
        type=Path,
        default=ROOT / "outputs/probe_compression_random_all_known_raw_rank1.csv.gz",
        help="Gzip CSV of every all-known random-prefix target prediction",
    )
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--max-any-k", type=int, default=10)
    parser.add_argument("--max-random-k", type=int, default=30)
    parser.add_argument("--max-heldout-random-k", type=int, default=10)
    parser.add_argument("--max-ranking-random-k", type=int, default=10)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--pruned-keep", type=int, default=30)
    parser.add_argument("--ranking-margin", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, min(28, (os.cpu_count() or 2) - 1)))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "experiments/probe_compression_checkpoints",
        help="Hash-addressed durable checkpoints for completed greedy/ranking phases",
    )
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
    checkpoint_identity = _checkpoint_identity(args, scores_sha256)
    checkpoint_path = args.checkpoint_dir / f"{_sha256_json(checkpoint_identity)}.json"
    phase_checkpoint = _load_phase_checkpoint(checkpoint_path, checkpoint_identity)
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
        _write_json_atomic(args.output, existing)
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
            "all_known_random_curve_limit": args.max_random_k,
            "heldout_random_curve_limit": args.max_heldout_random_k,
            "ranking_random_curve_limit": args.max_ranking_random_k,
            "random_raw_output": str(args.random_raw_output.relative_to(ROOT)),
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
    raw: list[dict[str, Any]] = copy.deepcopy(phase_checkpoint["raw"])
    random_raw_fields = [
        "protocol", "candidate_mode", "method", "selection_objective",
        "repeat", "k", "model_id", "evaluation_id",
        "actual_normalized_score", "predicted_normalized_score",
        "is_revealed_probe_cell", "is_hidden_prediction",
    ]
    args.random_raw_output.parent.mkdir(parents=True, exist_ok=True)
    random_raw_temporary = _temporary_sibling(args.random_raw_output)
    try:
        with random_raw_temporary.open("wb") as random_raw_binary, gzip.GzipFile(
            filename="", fileobj=random_raw_binary, mode="wb", mtime=0
        ) as random_raw_gzip, io.TextIOWrapper(
            random_raw_gzip, newline="", encoding="utf-8"
        ) as random_raw_handle, ProcessPoolExecutor(max_workers=args.workers) as executor:
            random_raw_writer = csv.DictWriter(
                random_raw_handle, fieldnames=random_raw_fields, lineterminator="\n"
            )
            random_raw_writer.writeheader()
            for candidate_mode, candidates, max_k in (
                ("any_candidate", any_indices, args.max_any_k),
                ("pre_error_low_friction_allowlist", allow_indices, args.max_any_k),
            ):
                if candidate_mode == "any_candidate" and reusable_any is not None:
                    mode_result = copy.deepcopy(reusable_any)
                    raw.extend(reusable_any_raw)
                    print(
                        "reused hash-matched unrestricted greedy curves and selected raw rows; "
                        "random and ranking trajectories will be regenerated",
                        flush=True,
                    )
                    mode_result["all_known_random"] = _random_curves(
                        executor, matrix, candidates,
                        max_k=min(args.max_random_k, len(candidates)),
                        repeats=args.random_repeats, seed=args.seed, rank=args.rank,
                        evaluations=evaluations, raw_writer=random_raw_writer,
                        models=models, candidate_mode=candidate_mode,
                    )
                    mode_result["heldout_random"] = _random_curves(
                        executor, matrix, candidates,
                        max_k=min(args.max_heldout_random_k, len(candidates)),
                        repeats=args.random_repeats, seed=args.seed, rank=args.rank,
                        evaluations=evaluations,
                        heldout=(validation_indices, train_indices),
                    )
                    payload["curves"][candidate_mode] = mode_result
                    continue
                mode_result: dict[str, Any] = {
                    "candidate_indices": candidates,
                    "candidate_ids": [evaluations[i] for i in candidates],
                }
                for objective in ("medae", "medape"):
                    cached_mode = phase_checkpoint["curve_greedy"].get(
                        candidate_mode, {}
                    )
                    all_key = f"all_known_greedy_{objective}"
                    heldout_key = f"heldout_greedy_{objective}"
                    if all_key in cached_mode and heldout_key in cached_mode:
                        mode_result[all_key] = copy.deepcopy(cached_mode[all_key])
                        mode_result[heldout_key] = copy.deepcopy(
                            cached_mode[heldout_key]
                        )
                        print(
                            f"resumed current-hash {candidate_mode} {objective} greedy phase",
                            flush=True,
                        )
                        continue
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
                    phase_checkpoint["curve_greedy"].setdefault(
                        candidate_mode, {}
                    )[all_key] = copy.deepcopy(all_greedy)
                    phase_checkpoint["curve_greedy"][candidate_mode][heldout_key] = copy.deepcopy(
                        heldout_greedy
                    )
                    phase_checkpoint["raw"] = copy.deepcopy(raw)
                    _save_phase_checkpoint(checkpoint_path, phase_checkpoint)
                mode_result["all_known_random"] = _random_curves(
                    executor, matrix, candidates, max_k=min(args.max_random_k, len(candidates)),
                    repeats=args.random_repeats, seed=args.seed, rank=args.rank, evaluations=evaluations,
                    raw_writer=random_raw_writer, models=models, candidate_mode=candidate_mode,
                )
                mode_result["heldout_random"] = _random_curves(
                    executor, matrix, candidates, max_k=min(args.max_heldout_random_k, len(candidates)),
                    repeats=args.random_repeats, seed=args.seed, rank=args.rank, evaluations=evaluations,
                    heldout=(validation_indices, train_indices),
                )
                payload["curves"][candidate_mode] = mode_result

            for candidate_mode, candidates in (
                ("any_candidate", any_indices),
                ("pre_error_low_friction_allowlist", allow_indices),
            ):
                cached_ranking = phase_checkpoint["ranking"].get(candidate_mode)
                if cached_ranking is not None:
                    payload["ranking_aware"][candidate_mode] = copy.deepcopy(
                        cached_ranking
                    )
                    print(
                        f"resumed current-hash {candidate_mode} ranking phase",
                        flush=True,
                    )
                    continue
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
                        max_k=min(args.max_ranking_random_k, len(candidates)),
                        repeats=args.random_repeats,
                        seed=args.seed,
                        rank=args.rank,
                        evaluations=evaluations,
                        pairwise_margin=args.ranking_margin,
                        ranking_scope="all_target",
                    ),
                }
                phase_checkpoint["ranking"][candidate_mode] = copy.deepcopy(
                    payload["ranking_aware"][candidate_mode]
                )
                phase_checkpoint["raw"] = copy.deepcopy(raw)
                _save_phase_checkpoint(checkpoint_path, phase_checkpoint)
        os.replace(random_raw_temporary, args.random_raw_output)
    finally:
        random_raw_temporary.unlink(missing_ok=True)

    _write_json_atomic(args.output, payload)
    _write_csv_atomic(args.raw_output, raw)
    print(f"wrote {args.output}, {args.raw_output}, and {args.random_raw_output}")


if __name__ == "__main__":
    main()
