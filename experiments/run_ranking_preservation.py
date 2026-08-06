#!/usr/bin/env python3
"""BenchPress Section 5.2 ranking-preservation experiment for PathoPress."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.ranking import pairwise_ranking_accuracy, top_fraction_recovery  # noqa: E402


MARGINS = (0.0, 1.0, 2.0, 5.0)
TOP_FRACTIONS = (0.10, 0.20, 0.30)


def _round(value: float) -> float:
    return round(float(value), 6)


def summarize_pairwise(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pool folds within evaluations, then median across evaluations."""
    output: dict[str, Any] = {}
    evaluation_rows: list[dict[str, Any]] = []
    for margin in MARGINS:
        subset = [row for row in rows if row["margin"] == margin and row["n_pairs"] > 0]
        grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
            lambda: {"n_groups": 0, "n_pairs": 0, "n_correct": 0, "n_predicted_ties": 0}
        )
        for row in subset:
            bucket = grouped[(row["evaluation_id"], row["suite_id"], row["metric"])]
            bucket["n_groups"] += 1
            for key in ("n_pairs", "n_correct", "n_predicted_ties"):
                bucket[key] += int(row[key])
        current: list[dict[str, Any]] = []
        for (evaluation_id, suite, metric), bucket in sorted(grouped.items()):
            result = {
                "evaluation_id": evaluation_id,
                "suite_id": suite,
                "metric": metric,
                "margin": margin,
                **bucket,
                "accuracy": bucket["n_correct"] / bucket["n_pairs"],
            }
            current.append(result)
            evaluation_rows.append(result)
        accuracies = [row["accuracy"] for row in current]
        n_pairs = sum(row["n_pairs"] for row in current)
        n_correct = sum(row["n_correct"] for row in current)
        suite_summary = {}
        for suite in sorted({row["suite_id"] for row in current}):
            values = [row["accuracy"] for row in current if row["suite_id"] == suite]
            suite_summary[suite] = {"n_evaluations": len(values), "median_accuracy": _round(np.median(values))}
        output[str(margin)] = {
            "n_groups": len(subset),
            "n_evaluations": len(current),
            "n_pairs": n_pairs,
            "n_correct": n_correct,
            "n_predicted_ties": sum(row["n_predicted_ties"] for row in current),
            "median_accuracy": _round(np.median(accuracies)) if accuracies else None,
            "pooled_accuracy": _round(n_correct / n_pairs) if n_pairs else None,
            "by_suite": suite_summary,
        }
    return output, evaluation_rows


def summarize_top(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pool shortlist overlap within evaluations, then median across evaluations."""
    output: dict[str, Any] = {}
    evaluation_rows: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        subset = [row for row in rows if row["top_fraction"] == fraction and row["k"] > 0]
        grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
            lambda: {"n_groups": 0, "total_k": 0, "overlap": 0}
        )
        for row in subset:
            bucket = grouped[(row["evaluation_id"], row["suite_id"], row["metric"])]
            bucket["n_groups"] += 1
            bucket["total_k"] += int(row["k"])
            bucket["overlap"] += int(row["overlap"])
        current: list[dict[str, Any]] = []
        for (evaluation_id, suite, metric), bucket in sorted(grouped.items()):
            result = {
                "evaluation_id": evaluation_id,
                "suite_id": suite,
                "metric": metric,
                "top_fraction": fraction,
                **bucket,
                "recovery": bucket["overlap"] / bucket["total_k"],
            }
            current.append(result)
            evaluation_rows.append(result)
        recoveries = [row["recovery"] for row in current]
        total_k = sum(row["total_k"] for row in current)
        overlap = sum(row["overlap"] for row in current)
        suite_summary = {}
        for suite in sorted({row["suite_id"] for row in current}):
            values = [row["recovery"] for row in current if row["suite_id"] == suite]
            suite_summary[suite] = {"n_evaluations": len(values), "median_recovery": _round(np.median(values))}
        output[str(fraction)] = {
            "n_groups": len(subset),
            "n_evaluations": len(current),
            "total_k": total_k,
            "overlap": overlap,
            "median_recovery": _round(np.median(recoveries)) if recoveries else None,
            "pooled_recovery": _round(overlap / total_k) if total_k else None,
            "by_suite": suite_summary,
        }
    return output, evaluation_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "data/scores.csv")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "experiments/benchpress_style_predictions_rank1.csv")
    parser.add_argument("--validation-results", type=Path, default=PROJECT_ROOT / "experiments/benchpress_style_results.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments/ranking_preservation_rank1.json")
    parser.add_argument("--pairwise-csv", type=Path, default=PROJECT_ROOT / "outputs/ranking_preservation_pairwise_rank1.csv")
    parser.add_argument("--top-csv", type=Path, default=PROJECT_ROOT / "outputs/ranking_preservation_top_fraction_rank1.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores)
    matrix, models, evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(matrix, models, evaluations)
    model_index = {name: index for index, name in enumerate(models)}
    evaluation_index = {name: index for index, name in enumerate(evaluations)}
    metadata: dict[str, tuple[str, str]] = {}
    for score in scores:
        if score.evaluation_id in evaluation_index:
            previous = metadata.setdefault(score.evaluation_id, (score.suite_id, score.metric))
            if previous != (score.suite_id, score.metric):
                raise ValueError(f"inconsistent evaluation metadata: {score.evaluation_id}")

    with args.predictions.open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    groups: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[int, str, str]] = set()
    for row in predictions:
        seed, fold = int(row["seed"]), int(row["fold"])
        model, evaluation = row["model_id"], row["evaluation_id"]
        if model not in model_index or evaluation not in evaluation_index:
            raise ValueError(f"prediction cache is inconsistent with filtered matrix: {model}/{evaluation}")
        key = (seed, model, evaluation)
        if key in seen:
            raise ValueError(f"duplicate seed/cell prediction: {key}")
        seen.add(key)
        i, j = model_index[model], evaluation_index[evaluation]
        actual = float(row["actual_normalized_score"])
        if not np.isclose(actual, matrix[i, j], atol=5e-7):
            raise ValueError(f"actual-score drift for {key}: {actual} != {matrix[i, j]}")
        groups[(seed, fold, j)].append(row)

    observed = int(np.isfinite(matrix).sum())
    seeds = sorted({int(row["seed"]) for row in predictions})
    if len(predictions) != observed * len(seeds):
        raise ValueError(
            f"prediction cache must cover every observed cell once per seed: {len(predictions)} != {observed}*{len(seeds)}"
        )

    pairwise_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    for (seed, fold, j), held_rows in sorted(groups.items()):
        observed_models = np.flatnonzero(np.isfinite(matrix[:, j]))
        actual = matrix[observed_models, j].reshape(-1, 1)
        completed = actual.copy()
        heldout = np.zeros_like(actual, dtype=bool)
        positions = {int(model): pos for pos, model in enumerate(observed_models)}
        for row in held_rows:
            pos = positions[model_index[row["model_id"]]]
            completed[pos, 0] = float(row["predicted_normalized_score"])
            heldout[pos, 0] = True
        evaluation = evaluations[j]
        suite, metric = metadata[evaluation]
        common = {
            "seed": seed,
            "fold": fold,
            "evaluation_id": evaluation,
            "suite_id": suite,
            "metric": metric,
            "n_models": int(len(observed_models)),
            "n_heldout_models": int(heldout.sum()),
            "n_seen_models": int(len(observed_models) - heldout.sum()),
        }
        for margin in MARGINS:
            result = pairwise_ranking_accuracy(actual, completed, heldout, margin=margin).columns[0]
            pairwise_rows.append(
                {**common, "margin": margin, "n_pairs": result.n_pairs, "n_correct": result.n_correct,
                 "n_predicted_ties": result.n_predicted_ties,
                 "accuracy": _round(result.accuracy) if result.n_pairs else None}
            )
        for fraction in TOP_FRACTIONS:
            result = top_fraction_recovery(actual, completed, heldout, top_fraction=fraction).columns[0]
            top_rows.append(
                {**common, "top_fraction": fraction, "k": result.k, "overlap": result.overlap,
                 "recovery": _round(result.recovery) if result.k else None}
            )

    pairwise_summary, pairwise_evaluations = summarize_pairwise(pairwise_rows)
    top_summary, top_evaluations = summarize_top(top_rows)
    _write_csv(args.pairwise_csv, pairwise_evaluations)
    _write_csv(args.top_csv, top_evaluations)
    validation = json.loads(args.validation_results.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "metadata": {
            "experiment": "BenchPress Section 5.2 ranking preservation",
            "prediction_rank": validation["configuration"]["prediction_rank"],
            "margins": list(MARGINS),
            "top_fractions": list(TOP_FRACTIONS),
            "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
            "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
            "upstream_semantics": "True seen cells plus out-of-fold predictions for held cells; discard seen/seen pairs; pool counts across folds within evaluation; report median across evaluations.",
            "pathology_adaptation": "No numerical adaptation: upstream 0/1/2/5 margins are retained because normalized scores are 0-100. They are sensitivity thresholds, not universal clinically meaningful differences.",
        },
        "matrix": {"n_models": len(models), "n_evaluations": len(evaluations), "n_observed": observed, "n_seeds": len(seeds)},
        "summary": {"pairwise_by_margin": pairwise_summary, "top_by_fraction": top_summary},
        "pairwise_rows": pairwise_rows,
        "top_rows": top_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
