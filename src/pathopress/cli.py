"""PathoPress command-line audit and prediction surface."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .completion import validate
from .imputation import build_imputation_rows, write_imputations
from .matrix import filter_matrix, load_scores, make_matrix
from .prediction import (
    DEFAULT_RANK,
    DEFAULT_REGULARIZATION,
    PredictionDataset,
    calibrated_interval,
    complete_dataset,
    load_confidence_artifact,
    load_prediction_dataset,
    parse_known_scores,
    predict_new_model,
)


PRODUCT_COMMANDS = {
    "list-models",
    "list-evaluations",
    "predict",
    "complete-model",
    "add-model",
}


def _matrix(args: argparse.Namespace) -> tuple[np.ndarray, list[str], list[str]]:
    scores = load_scores(args.scores)
    matrix, models, evaluations = make_matrix(scores)
    return filter_matrix(
        matrix,
        models,
        evaluations,
        min_scores_per_model=args.min_scores_per_model,
        min_models_per_evaluation=args.min_models_per_evaluation,
    )


def _dataset(args: argparse.Namespace) -> PredictionDataset:
    return load_prediction_dataset(
        args.scores,
        min_scores_per_model=args.min_scores_per_model,
        min_models_per_evaluation=args.min_models_per_evaluation,
    )


def _write_output(text: str, output: Path | None) -> None:
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip("\n") + "\n", encoding="utf-8")


def _format_rows(rows: Iterable[dict[str, object]], fmt: str) -> str:
    materialized = list(rows)
    if fmt == "json":
        return json.dumps(materialized, indent=2, sort_keys=True)
    if not materialized:
        return ""
    columns: list[str] = []
    for row in materialized:
        for key in row:
            if key not in columns:
                columns.append(key)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(materialized)
    return output.getvalue().rstrip("\n")


def _metadata(dataset: PredictionDataset, evaluation: str) -> tuple[str, str]:
    for score in dataset.scores:
        if score.evaluation_id == evaluation:
            return score.suite_id, score.metric
    raise ValueError(f"evaluation metadata unavailable for {evaluation!r}")


def _prediction_row(
    *,
    model: str,
    evaluation: str,
    suite: str,
    metric: str,
    prediction: float,
    observed: float | None,
    status: str,
    confidence: dict[str, object] | None = None,
    confidence_status: str = "not_requested",
) -> dict[str, object]:
    row: dict[str, object] = {
        "model_id": model,
        "evaluation_id": evaluation,
        "suite_id": suite,
        "metric": metric,
        "normalized_score": round(float(prediction), 6),
        "observed_normalized_score": (
            round(float(observed), 6) if observed is not None else None
        ),
        "status": status,
        "point_method": "logit_bias_als_rank1_lambda0.1",
        "confidence_status": confidence_status,
    }
    if confidence is not None:
        lower, upper, calibration_scope = calibrated_interval(
            prediction, suite, confidence
        )
        row.update(
            {
                "confidence_method": confidence["artifact_type"],
                "confidence_level": confidence["confidence_level"],
                "lower_90": round(lower, 6),
                "upper_90": round(upper, 6),
                "calibration_scope": calibration_scope,
            }
        )
    return row


def _product_rows(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> list[dict[str, object]]:
    dataset = _dataset(args)
    if args.command == "list-models":
        return [
            {
                "model_id": model,
                "observed_scores": int(np.isfinite(dataset.matrix[index]).sum()),
                "supported_evaluations": len(dataset.evaluations),
            }
            for index, model in enumerate(dataset.models)
        ]
    if args.command == "list-evaluations":
        rows = []
        for index, evaluation in enumerate(dataset.evaluations):
            suite, metric = _metadata(dataset, evaluation)
            rows.append(
                {
                    "evaluation_id": evaluation,
                    "suite_id": suite,
                    "metric": metric,
                    "observed_models": int(np.isfinite(dataset.matrix[:, index]).sum()),
                    "supported_models": len(dataset.models),
                }
            )
        return rows

    confidence = None
    if args.confidence and args.command != "add-model":
        try:
            confidence = load_confidence_artifact(
                args.confidence_artifact,
                args.scores,
                rank=args.rank,
                regularization=args.regularization,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))

    if args.command in {"predict", "complete-model"}:
        if not args.model:
            parser.error(f"{args.command} requires --model")
        if args.model not in dataset.model_index:
            parser.error(
                f"unknown model {args.model!r}; use list-models for supported IDs"
            )
        if args.command == "predict" and not args.evaluation:
            parser.error("predict requires --evaluation")
        if args.evaluation and args.evaluation not in dataset.evaluation_index:
            parser.error(
                f"unknown evaluation {args.evaluation!r}; use list-evaluations for supported IDs"
            )
        row_index = dataset.model_index[args.model]
        target_evaluations = (
            [args.evaluation] if args.command == "predict" else dataset.evaluations
        )
        needs_prediction = any(
            not np.isfinite(dataset.matrix[row_index, dataset.evaluation_index[name]])
            for name in target_evaluations
        )
        completed = complete_dataset(
            dataset, rank=args.rank, regularization=args.regularization
        ) if needs_prediction else dataset.matrix
        rows = []
        for evaluation in target_evaluations:
            column = dataset.evaluation_index[evaluation]
            observed_value = dataset.matrix[row_index, column]
            is_observed = bool(np.isfinite(observed_value))
            if args.command == "complete-model" and is_observed and not args.include_observed:
                continue
            suite, metric = _metadata(dataset, evaluation)
            rows.append(
                _prediction_row(
                    model=args.model,
                    evaluation=evaluation,
                    suite=suite,
                    metric=metric,
                    prediction=float(completed[row_index, column]),
                    observed=float(observed_value) if is_observed else None,
                    status="observed" if is_observed else "predicted",
                    confidence=confidence if not is_observed else None,
                    confidence_status=(
                        "calibrated_existing_model"
                        if confidence is not None and not is_observed
                        else "not_applicable_observed"
                        if is_observed and args.confidence
                        else "not_requested"
                    ),
                )
            )
        return rows

    if args.command == "add-model":
        if not args.model:
            parser.error("add-model requires --model")
        if args.model in dataset.model_index:
            parser.error(
                f"model {args.model!r} already exists; use complete-model instead"
            )
        try:
            known = parse_known_scores(args.known_score or [])
            predicted = predict_new_model(
                dataset,
                known,
                rank=args.rank,
                regularization=args.regularization,
            )
        except ValueError as exc:
            parser.error(str(exc))
        rows = []
        for column, evaluation in enumerate(dataset.evaluations):
            is_known = evaluation in known
            if is_known and not args.include_observed:
                continue
            suite, metric = _metadata(dataset, evaluation)
            rows.append(
                _prediction_row(
                    model=args.model,
                    evaluation=evaluation,
                    suite=suite,
                    metric=metric,
                    prediction=float(predicted[column]),
                    observed=known.get(evaluation),
                    status="provided" if is_known else "predicted",
                    confidence=None,
                    confidence_status=(
                        "not_applicable_new_model"
                        if args.confidence and not is_known
                        else "not_applicable_provided"
                        if args.confidence
                        else "not_requested"
                    ),
                )
            )
        return rows
    raise AssertionError(args.command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and complete a citation-backed pathology score matrix"
    )
    parser.add_argument(
        "command",
        choices=(
            "audit",
            "validate",
            "impute",
            "list-models",
            "list-evaluations",
            "predict",
            "complete-model",
            "add-model",
        ),
    )
    parser.add_argument("--scores", type=Path, default=Path("data/scores.csv"))
    parser.add_argument("--min-scores-per-model", type=int, default=3)
    parser.add_argument("--min-models-per-evaluation", type=int, default=5)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--regularization", type=float, default=DEFAULT_REGULARIZATION)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    parser.add_argument("--model", help="Canonical model ID")
    parser.add_argument("--evaluation", help="Canonical evaluation/protocol ID")
    parser.add_argument(
        "--known-score",
        action="append",
        help="Known normalized score as evaluation=value; repeat or comma-separate",
    )
    parser.add_argument(
        "--include-observed",
        action="store_true",
        help="Include observed/provided cells in model-wide output",
    )
    parser.add_argument("--confidence", action="store_true")
    parser.add_argument(
        "--confidence-artifact",
        type=Path,
        default=Path("experiments/deployment_confidence_rank1.json"),
    )
    args = parser.parse_args()
    if args.rank < 0:
        parser.error("rank must be non-negative")
    if args.regularization < 0:
        parser.error("regularization must be non-negative")

    if args.command in PRODUCT_COMMANDS:
        rows = _product_rows(args, parser)
        _write_output(_format_rows(rows, args.format), args.output)
        return

    matrix, models, evaluations = _matrix(args)
    observed = int(np.sum(np.isfinite(matrix)))
    total = int(matrix.size)
    print(f"matrix={len(models)} models x {len(evaluations)} evaluations")
    density = observed / total if total else 0.0
    print(f"observed={observed}/{total} ({density:.1%})")
    if args.command == "validate":
        if total == 0:
            parser.error("no matrix cells remain after support filtering")
        result = validate(matrix, rank=args.rank)
        print(f"held_out_predictions={result.n_predictions}")
        print(f"median_absolute_error={result.median_absolute_error:.3f} points")
        print(f"mean_absolute_error={result.mean_absolute_error:.3f} points")
    elif args.command == "impute":
        if total == 0:
            parser.error("no matrix cells remain after support filtering")
        score_rows = load_scores(args.scores)
        metadata: dict[str, tuple[str, str]] = {}
        for score in score_rows:
            previous = metadata.setdefault(
                score.evaluation_id, (score.suite_id, score.metric)
            )
            if previous != (score.suite_id, score.metric):
                raise ValueError(
                    f"inconsistent evaluation metadata: {score.evaluation_id}"
                )
        output = args.output or Path("outputs/imputations_rank1.csv")
        rows = build_imputation_rows(
            matrix, models, evaluations, metadata, rank=args.rank
        )
        write_imputations(output, rows)
        n_imputed = sum(row["status"] == "imputed" for row in rows)
        print(f"output={output}")
        print(f"imputed={n_imputed}")


if __name__ == "__main__":
    main()
