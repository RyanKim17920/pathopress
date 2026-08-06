#!/usr/bin/env python3
"""Prepare, mock-test, and merge provider-neutral LLM baseline artifacts.

This runner has no provider client and cannot make external model calls.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.artifacts import load_fold_artifact, sha256_file  # noqa: E402
from pathopress.llm_baseline import (  # noqa: E402
    CONDITIONS,
    build_request,
    deterministic_mock_response,
    evaluate_cached_responses,
    make_config,
    seal_external_response,
    summarize_real_execution,
    validate_config,
    validate_request,
    validate_response,
)
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.publication import read_csv  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for shard in sorted(path.glob("requests-*.jsonl.gz")):
            rows.extend(_read_jsonl(shard))
        return rows
    if path.suffix == ".gz":
        text = gzip.decompress(path.read_bytes()).decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_request_pack(path: Path, rows: list[dict[str, Any]], shard_size: int = 100) -> list[Path]:
    path.mkdir(parents=True, exist_ok=True)
    expected_names = []
    shards = []
    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        shard = path / f"requests-{shard_index:04d}.jsonl.gz"
        expected_names.append(shard.name)
        payload = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows[start : start + shard_size]
        ).encode("utf-8")
        with shard.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                compressed.write(payload)
        shards.append(shard)
    for stale in list(path.glob("requests-*.jsonl")) + list(path.glob("requests-*.jsonl.gz")):
        if stale.name not in expected_names:
            stale.unlink()
    return shards


def _request_pack_sha256(shards: list[Path]) -> str:
    digest = hashlib.sha256()
    for shard in shards:
        digest.update(shard.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(shard)))
    return digest.hexdigest()


def _request_pack_uncompressed_sha256(shards: list[Path]) -> str:
    digest = hashlib.sha256()
    for shard in shards:
        digest.update(gzip.decompress(shard.read_bytes()))
    return digest.hexdigest()


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


def _condition_batches(
    condition: str, cells: list[tuple[int, int]], config: dict[str, Any]
) -> list[list[tuple[int, int]]]:
    """Apply the pinned upstream batching unit for each prompt family."""

    if condition.startswith("zero_shot"):
        by_model: dict[int, list[tuple[int, int]]] = {}
        for cell in cells:
            by_model.setdefault(int(cell[0]), []).append(cell)
        model_ids = sorted(by_model)
        size = int(config["batch_sizes"]["zero_shot_models"])
        return [
            [cell for model_id in model_ids[start : start + size] for cell in by_model[model_id]]
            for start in range(0, len(model_ids), size)
        ]
    key = f"{condition}_cells"
    size = int(config["batch_sizes"][key])
    return [cells[start : start + size] for start in range(0, len(cells), size)]


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
            for batch_index, batch in enumerate(_condition_batches(condition, cells, config)):
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
    request_shards = _write_request_pack(args.requests, requests)
    request_pack_sha256 = _request_pack_sha256(request_shards)
    request_pack_uncompressed_sha256 = _request_pack_uncompressed_sha256(request_shards)
    _write_json(args.request_index, {
        "schema_version": 2,
        "kind": "provider_neutral_request_pack_index",
        "request_directory": args.requests.name,
        "request_count": len(requests),
        "shard_count": len(request_shards),
        "pack_sha256": request_pack_sha256,
        "canonical_uncompressed_sha256": request_pack_uncompressed_sha256,
        "shards": [
            {"path": shard.name, "sha256": sha256_file(shard), "bytes": shard.stat().st_size}
            for shard in request_shards
        ],
    })
    _write_json(args.request_schema, {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PathoPress provider-neutral chat request JSONL record",
        "type": "object",
        "required": ["schema_version", "request_id", "request_sha256", "config_sha256", "condition", "fold_id", "targets", "messages", "response_contract"],
        "properties": {
            "schema_version": {"const": 2}, "request_id": {"type": "string"},
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
            "schema_version": {"const": 2}, "request_id": {"type": "string"},
            "backend_kind": {"enum": ["deterministic_mock", "openai_compatible", "anthropic_compatible", "local_model"]},
            "provider": {"type": "string"}, "model": {"type": "string"},
            "headline_eligible": {"type": "boolean"}, "response_text": {"type": "string"},
            "response_sha256": {"type": "string", "minLength": 64, "maxLength": 64},
            "execution_metadata_hashes": {"type": "object"}
        }
    })
    _write_json(args.raw_response_schema, {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PathoPress raw provider-neutral response JSONL record",
        "type": "object",
        "additionalProperties": False,
        "required": ["request_id", "backend_kind", "provider", "model", "response_text"],
        "properties": {
            "request_id": {"type": "string"},
            "backend_kind": {"enum": ["openai_compatible", "anthropic_compatible", "local_model"]},
            "provider": {"type": "string", "minLength": 1},
            "model": {"type": "string", "minLength": 1},
            "response_text": {"type": "string"},
            "usage": {"type": "object"},
            "cost": {"type": "object"},
            "execution_metadata": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "model_version": {"type": "string", "minLength": 1},
                    "settings": {"type": "object"},
                    "receipt": {}
                }
            }
        }
    })
    request_counts = {
        condition: sum(row["condition"] == condition for row in requests)
        for condition in CONDITIONS
    }
    target_counts = {
        condition: sum(len(row["targets"]) for row in requests if row["condition"] == condition)
        for condition in CONDITIONS
    }
    real_status = {
        "schema_version": 2,
        "config_sha256": config["config_sha256"],
        "status": "unrun",
        "headline_eligible": False,
        "reason": "The complete offline request pack is prepared, but no genuine provider responses have been supplied.",
        "conditions": {condition: "unrun" for condition in CONDITIONS},
        "request_count": len(requests),
        "expected_response_count": len(requests),
        "received_response_count": 0,
        "target_prediction_count": sum(target_counts.values()),
        "request_counts_by_condition": request_counts,
        "target_counts_by_condition": target_counts,
        "cost": {"status": "unknown_until_external_execution", "currency": None, "amount": None},
        "token_usage": {"status": "unknown_until_external_execution", "input_tokens": None, "output_tokens": None},
        "credential_handling": "not_read_or_used_by_this_runner",
        "sole_external_action": "Execute every record in the provider-neutral requests/ JSONL shards with one chosen genuine LLM and supply one raw JSONL response per request_id.",
        "expected_response_contract": "exactly one complete response per request_id; import-real validates and hash-seals it",
    }
    _write_json(args.real_status, real_status)
    manifest = {
        "schema_version": 2,
        "kind": "llm_baseline_complete_request_manifest",
        "config_sha256": config["config_sha256"],
        "status": "prepared_unrun",
        "headline_eligible": False,
        "n_requests": len(requests),
        "expected_response_count": len(requests),
        "n_target_predictions": sum(target_counts.values()),
        "condition_request_counts": request_counts,
        "condition_target_counts": target_counts,
        "fold_count": len(fold_ids),
        "fold_ids": fold_ids,
        "complete_fold_protocol": len(fold_ids) == len(folds) and args.cell_limit is None,
        "batching": config["batch_sizes"],
        "external_execution": {"status": "unrun", "cost": "unknown", "responses_received": 0},
        "request_pack": {
            "directory": str(args.requests.relative_to(ROOT)),
            "index": str(args.request_index.relative_to(ROOT)),
            "shard_count": len(request_shards),
            "sha256": request_pack_sha256,
            "canonical_uncompressed_sha256": request_pack_uncompressed_sha256,
        },
        "inputs": {
            str(args.scores.relative_to(ROOT)): sha256_file(args.scores),
            str(args.folds.relative_to(ROOT)): sha256_file(args.folds),
            str(args.tasks.relative_to(ROOT)): sha256_file(args.tasks),
        },
        "artifacts": {
            str(args.config.relative_to(ROOT)): sha256_file(args.config),
            str(args.request_index.relative_to(ROOT)): sha256_file(args.request_index),
            str(args.real_status.relative_to(ROOT)): sha256_file(args.real_status),
            str(args.request_schema.relative_to(ROOT)): sha256_file(args.request_schema),
            str(args.response_schema.relative_to(ROOT)): sha256_file(args.response_schema),
            str(args.raw_response_schema.relative_to(ROOT)): sha256_file(args.raw_response_schema),
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
        if prior is not None and not args.force:
            validate_response(prior, request, config)
            responses.append(prior)
            continue
        _, _, train, _ = folds[int(request["fold_id"])]
        responses.append(deterministic_mock_response(request, train, config))
    _write_jsonl(args.mock_responses, responses)
    print(f"mock cache complete: {len(responses)} responses (headline_eligible=false)")
    return responses


def _merge(
    args: argparse.Namespace, response_path: Path, *, require_real: bool = False
) -> dict[str, Any]:
    matrix, _, _, _, _ = _load_context(args)
    config = json.loads(args.config.read_text())
    requests = _read_jsonl(args.requests)
    responses = _read_jsonl(response_path)
    result = evaluate_cached_responses(
        requests,
        responses,
        matrix,
        config,
        require_complete=require_real,
        require_real=require_real,
    )
    baseline = json.loads(args.baseline_results.read_text(encoding="utf-8"))
    if (
        baseline["input"]["scores_sha256"] != config["scores_sha256"]
        or baseline["input"]["folds_sha256"] != config["folds_sha256"]
    ):
        raise ValueError("rank-1 comparator does not match the LLM request matrix/folds")
    rank1 = baseline["by_rank"]["1"]
    result["rank1_comparator"] = {
        "method": "PathoPress rank-1 Bias ALS",
        "medae": float(rank1["fold_medae"]["median"]),
        "medape": float(np.median([row["medape"] for row in rank1["folds"]])),
        "n_folds": len(rank1["folds"]),
        "source": str(args.baseline_results.relative_to(ROOT)),
    }
    result["response_path"] = str(response_path.resolve().relative_to(ROOT.resolve()))
    result["response_sha256"] = sha256_file(response_path) if response_path.exists() else None
    output = args.mock_metrics if result["result_status"] == "mock_contract_validation_only" else args.real_metrics
    _write_json(output, result)
    print(f"merged {len(result['raw_predictions'])} predictions; status={result['result_status']}")
    return result


def _import_real(args: argparse.Namespace) -> dict[str, Any]:
    if args.raw_responses is None:
        raise ValueError("--raw-responses is required for import-real")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    requests = _read_jsonl(args.requests)
    request_by_id = {row["request_id"]: row for row in requests}
    raw_rows = _read_jsonl(args.raw_responses)
    raw_ids = [row.get("request_id") for row in raw_rows]
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("duplicate request_id in raw response pack")
    missing = sorted(set(request_by_id) - set(raw_ids))
    extra = sorted(set(raw_ids) - set(request_by_id))
    if missing or extra:
        raise ValueError(
            f"raw response pack must exactly cover requests; missing={len(missing)}, extra={len(extra)}"
        )
    raw_by_id = {row["request_id"]: row for row in raw_rows}
    sealed = [
        seal_external_response(raw_by_id[request["request_id"]], request, config)
        for request in requests
    ]
    _write_jsonl(args.sealed_real_responses, sealed)
    result = _merge(args, args.sealed_real_responses, require_real=True)
    providers = sorted({(row["provider"], row["model"]) for row in sealed})
    execution = summarize_real_execution(sealed)
    input_values = [row.get("usage", {}).get("input_tokens") for row in sealed]
    output_values = [row.get("usage", {}).get("output_tokens") for row in sealed]
    tokens_reported = all(isinstance(value, int) and value >= 0 for value in input_values + output_values)
    costs = [row.get("cost", {}) for row in sealed]
    currencies = {row.get("currency") for row in costs if row.get("amount") is not None}
    amounts = [row.get("amount") for row in costs]
    cost_reported = (
        len(currencies) == 1
        and all(isinstance(value, (int, float)) and value >= 0 for value in amounts)
    )
    status = {
        "schema_version": 2,
        "config_sha256": config["config_sha256"],
        "status": "complete_validated_real",
        "headline_eligible": True,
        "reason": "Every prepared request has one strict, hash-sealed genuine response.",
        "conditions": {condition: "complete_validated_real" for condition in CONDITIONS},
        "request_count": len(requests),
        "expected_response_count": len(requests),
        "received_response_count": len(sealed),
        "target_prediction_count": len(result["raw_predictions"]),
        "providers_and_models": [
            {"provider": provider, "model": model} for provider, model in providers
        ],
        "real_execution": execution,
        "token_usage": {
            "status": "reported" if tokens_reported else "not_fully_reported",
            "input_tokens": sum(input_values) if tokens_reported else None,
            "output_tokens": sum(output_values) if tokens_reported else None,
        },
        "cost": {
            "status": "reported" if cost_reported else "unknown_not_fully_reported",
            "currency": next(iter(currencies)) if cost_reported else None,
            "amount": float(sum(amounts)) if cost_reported else None,
        },
        "sealed_response_path": str(args.sealed_real_responses.relative_to(ROOT)),
        "real_metrics_path": str(args.real_metrics.relative_to(ROOT)),
        "credential_handling": "not_read_or_used_by_this_runner",
    }
    _write_json(args.real_status, status)
    return result


def _materialize(args: argparse.Namespace) -> None:
    if args.materialized_output is None:
        raise ValueError("--materialized-output is required for materialize")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    requests = _read_jsonl(args.requests)
    for request in requests:
        validate_request(request, config)
    args.materialized_output.mkdir(parents=True, exist_ok=True)
    shard_size = 100
    for shard_index, start in enumerate(range(0, len(requests), shard_size)):
        _write_jsonl(
            args.materialized_output / f"requests-{shard_index:04d}.jsonl",
            requests[start : start + shard_size],
        )
    print(f"validated and materialized {len(requests)} requests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "materialize", "mock", "merge-mock", "merge-real", "import-real", "all-mock"))
    parser.add_argument("--scores", type=Path, default=ROOT / "data/scores.csv")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/tasks.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "experiments/folds_s10_f3_bs42.json")
    parser.add_argument("--baseline-results", type=Path, default=ROOT / "experiments/benchpress_style_results.json")
    parser.add_argument("--scope", choices=("full", "smoke"), default="full")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fold-ids", nargs="+", type=int)
    parser.add_argument("--cell-limit", type=int)
    parser.add_argument("--real-responses", type=Path)
    parser.add_argument("--raw-responses", type=Path)
    parser.add_argument("--materialized-output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        suffix = "llm_baseline" if args.scope == "full" else "llm_baseline_smoke"
        args.output_dir = ROOT / "experiments" / suffix
    if args.fold_ids is None:
        args.fold_ids = list(range(30)) if args.scope == "full" else [0]
    if args.scope == "smoke" and args.cell_limit is None:
        args.cell_limit = 8
    args.config = args.output_dir / "config.json"
    args.requests = args.output_dir / "requests"
    args.request_index = args.output_dir / "requests.jsonl"
    args.mock_responses = args.output_dir / "mock_responses.jsonl"
    args.mock_metrics = args.output_dir / "mock_metrics.json"
    args.real_metrics = args.output_dir / "real_metrics.json"
    args.sealed_real_responses = args.output_dir / "real_responses.sealed.jsonl"
    args.real_status = args.output_dir / "real_run_status.json"
    args.manifest = args.output_dir / "dry_run_manifest.json"
    args.request_schema = args.output_dir / "request.schema.json"
    args.response_schema = args.output_dir / "response.schema.json"
    args.raw_response_schema = args.output_dir / "raw_response.schema.json"
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
        _merge(args, args.real_responses, require_real=True)
    if args.mode == "import-real":
        _import_real(args)
    if args.mode == "materialize":
        _materialize(args)


if __name__ == "__main__":
    main()
