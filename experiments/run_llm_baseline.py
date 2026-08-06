#!/usr/bin/env python3
"""Prepare, mock-test, and merge provider-neutral LLM baseline artifacts.

This runner has no provider client and cannot make external model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.artifacts import load_fold_artifact, sha256_file  # noqa: E402
from pathopress.llm_baseline import (  # noqa: E402
    CONDITIONS,
    build_request,
    deterministic_mock_response,
    evaluate_cached_responses,
    make_config,
    validate_config,
    validate_request,
    validate_response,
)
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.publication import read_csv  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _load_context(args: argparse.Namespace):
    score_objects = load_scores(args.scores)
    matrix, models, evaluations = filter_matrix(*make_matrix(score_objects))
    folds = load_fold_artifact(args.folds, matrix, models, evaluations)
    tasks = {
        row["evaluation_id"]: row
        for row in read_csv(args.tasks)
        if row["evaluation_id"] in set(evaluations)
    }
    return matrix, models, evaluations, folds, tasks


def _prepare(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matrix, models, evaluations, folds, tasks = _load_context(args)
    fold_ids = sorted(set(args.fold_ids))
    if any(value < 0 or value >= len(folds) for value in fold_ids):
        raise ValueError("fold id out of range")
    config = make_config(
        scores_sha256=sha256_file(args.scores),
        folds_sha256=sha256_file(args.folds),
        models=models,
        evaluations=evaluations,
        fold_ids=fold_ids,
        cell_limit=args.cell_limit,
    )
    requests = []
    for fold_id in fold_ids:
        seed, fold, train, test_cells = folds[fold_id]
        cells = test_cells if args.cell_limit is None else test_cells[: args.cell_limit]
        for condition in CONDITIONS:
            for batch_index, start in enumerate(range(0, len(cells), args.batch_size)):
                batch = cells[start : start + args.batch_size]
                requests.append(
                    build_request(
                        config=config,
                        condition=condition,
                        fold_id=fold_id,
                        seed=seed,
                        fold=fold,
                        batch_index=batch_index,
                        train=train,
                        cells=batch,
                        models=models,
                        evaluations=evaluations,
                        task_metadata=tasks,
                    )
                )
    prior_config = json.loads(args.config.read_text()) if args.config.exists() else None
    if prior_config is not None and prior_config.get("config_sha256") != config["config_sha256"] and not args.force:
        raise ValueError("existing config differs; use --force or a new output directory")
    prior_requests = _read_jsonl(args.requests)
    if prior_requests and [row["request_sha256"] for row in prior_requests] != [row["request_sha256"] for row in requests] and not args.force:
        raise ValueError("existing requests differ; use --force or a new output directory")
    _write_json(args.config, config)
    _write_jsonl(args.requests, requests)
    _write_json(args.request_schema, {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PathoPress provider-neutral chat request JSONL record",
        "type": "object",
        "required": ["schema_version", "request_id", "request_sha256", "config_sha256", "condition", "fold_id", "targets", "messages", "response_contract"],
        "properties": {
            "schema_version": {"const": 1}, "request_id": {"type": "string"},
            "request_sha256": {"type": "string", "minLength": 64, "maxLength": 64},
            "config_sha256": {"type": "string", "minLength": 64, "maxLength": 64},
            "condition": {"enum": list(CONDITIONS)}, "fold_id": {"type": "integer"},
            "targets": {"type": "array"}, "messages": {"type": "array"},
            "response_contract": {"type": "object"}
        }
    })
    _write_json(args.response_schema, {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PathoPress provider-neutral chat response JSONL record",
        "type": "object",
        "required": ["schema_version", "request_id", "request_sha256", "config_sha256", "backend_kind", "provider", "model", "status", "headline_eligible", "response_text", "response_sha256"],
        "properties": {
            "schema_version": {"const": 1}, "request_id": {"type": "string"},
            "backend_kind": {"enum": ["deterministic_mock", "openai_compatible", "anthropic_compatible", "local_model"]},
            "provider": {"type": "string"}, "model": {"type": "string"},
            "headline_eligible": {"type": "boolean"}, "response_text": {"type": "string"},
            "response_sha256": {"type": "string", "minLength": 64, "maxLength": 64}
        }
    })
    real_status = {
        "schema_version": 1,
        "config_sha256": config["config_sha256"],
        "status": "unrun",
        "headline_eligible": False,
        "reason": "No real provider response shards are present. No external API calls are implemented by this runner.",
        "conditions": {condition: "unrun" for condition in CONDITIONS},
        "expected_response_contract": "one JSONL response per request_id; validate with merge mode",
    }
    _write_json(args.real_status, real_status)
    manifest = {
        "schema_version": 1,
        "kind": "llm_baseline_dry_run_manifest",
        "config_sha256": config["config_sha256"],
        "status": "prepared_unrun",
        "headline_eligible": False,
        "n_requests": len(requests),
        "n_target_cells": sum(len(row["targets"]) for row in requests),
        "condition_request_counts": {condition: sum(row["condition"] == condition for row in requests) for condition in CONDITIONS},
        "inputs": {
            str(args.scores.relative_to(ROOT)): sha256_file(args.scores),
            str(args.folds.relative_to(ROOT)): sha256_file(args.folds),
            str(args.tasks.relative_to(ROOT)): sha256_file(args.tasks),
        },
        "artifacts": {
            str(args.config.relative_to(ROOT)): sha256_file(args.config),
            str(args.requests.relative_to(ROOT)): sha256_file(args.requests),
            str(args.real_status.relative_to(ROOT)): sha256_file(args.real_status),
            str(args.request_schema.relative_to(ROOT)): sha256_file(args.request_schema),
            str(args.response_schema.relative_to(ROOT)): sha256_file(args.response_schema),
        },
    }
    _write_json(args.manifest, manifest)
    print(f"prepared {len(requests)} requests; real-provider status=unrun")
    return config, requests


def _mock(args: argparse.Namespace) -> list[dict[str, Any]]:
    matrix, models, evaluations, folds, _ = _load_context(args)
    config = json.loads(args.config.read_text())
    validate_config(config)
    requests = _read_jsonl(args.requests)
    existing = {row["request_id"]: row for row in _read_jsonl(args.mock_responses)}
    responses = []
    for request in requests:
        validate_request(request, config)
        prior = existing.get(request["request_id"])
        if prior is not None:
            validate_response(prior, request, config)
            responses.append(prior)
            continue
        _, _, train, _ = folds[int(request["fold_id"])]
        responses.append(deterministic_mock_response(request, train, config))
    _write_jsonl(args.mock_responses, responses)
    print(f"mock cache complete: {len(responses)} responses (headline_eligible=false)")
    return responses


def _merge(args: argparse.Namespace, response_path: Path) -> dict[str, Any]:
    matrix, _, _, _, _ = _load_context(args)
    config = json.loads(args.config.read_text())
    requests = _read_jsonl(args.requests)
    responses = _read_jsonl(response_path)
    result = evaluate_cached_responses(requests, responses, matrix, config)
    result["response_path"] = str(response_path.resolve().relative_to(ROOT.resolve()))
    result["response_sha256"] = sha256_file(response_path) if response_path.exists() else None
    output = args.mock_metrics if result["result_status"] == "mock_contract_validation_only" else args.real_metrics
    _write_json(output, result)
    print(f"merged {len(result['raw_predictions'])} predictions; status={result['result_status']}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "mock", "merge-mock", "merge-real", "all-mock"))
    parser.add_argument("--scores", type=Path, default=ROOT / "data/scores.csv")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/tasks.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "experiments/folds_s10_f3_bs42.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/llm_baseline")
    parser.add_argument("--fold-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--cell-limit", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--real-responses", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.config = args.output_dir / "config.json"
    args.requests = args.output_dir / "requests.jsonl"
    args.mock_responses = args.output_dir / "mock_responses.jsonl"
    args.mock_metrics = args.output_dir / "mock_metrics.json"
    args.real_metrics = args.output_dir / "real_metrics.json"
    args.real_status = args.output_dir / "real_run_status.json"
    args.manifest = args.output_dir / "dry_run_manifest.json"
    args.request_schema = args.output_dir / "request.schema.json"
    args.response_schema = args.output_dir / "response.schema.json"
    return args


def main() -> None:
    args = parse_args()
    if args.mode in {"prepare", "all-mock"}:
        _prepare(args)
    if args.mode in {"mock", "all-mock"}:
        _mock(args)
    if args.mode in {"merge-mock", "all-mock"}:
        _merge(args, args.mock_responses)
    if args.mode == "merge-real":
        if args.real_responses is None:
            raise ValueError("--real-responses is required for merge-real")
        _merge(args, args.real_responses)


if __name__ == "__main__":
    main()
