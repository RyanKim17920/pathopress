"""Provider-neutral BenchPress-style LLM score-completion scaffolding.

This module builds prompts and validates request/response caches.  It performs
no network calls.  The deterministic mock backend exists only to exercise the
artifact contract and is never headline-eligible.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = 2
CONDITIONS = (
    "zero_shot_named",
    "zero_shot_blind",
    "five_shot_named",
    "five_shot_blind",
)
REAL_BACKEND_KINDS = {"openai_compatible", "anthropic_compatible", "local_model"}
EXECUTION_METADATA_KEYS = {"model_version", "settings", "receipt"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def make_config(
    *,
    scores_sha256: str,
    folds_sha256: str,
    models: Sequence[str],
    evaluations: Sequence[str],
    fold_ids: Sequence[int],
    cell_limit: int | None,
    n_shots: int = 5,
    min_shared: int = 5,
    max_target_known: int = 12,
    max_peer_shared: int = 4,
    zero_shot_model_batch_size: int = 10,
    five_shot_named_cell_batch_size: int = 64,
    five_shot_blind_cell_batch_size: int = 16,
) -> dict[str, Any]:
    config = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "pathopress_llm_baseline_s10_f3_bs42",
        "upstream": {
            "repository": "https://github.com/microsoft/benchpress",
            "commit": "0a684b63ee0e4a401cb907a3827a82ea997d74c4",
            "matrix_source": "experiments/sec4_building_benchpress/llm_completer/shared.py",
            "five_shot_source": "experiments/sec4_building_benchpress/llm_completer/five_shot_predictor/run.py",
        },
        "upstream_semantics": {
            "zero_shot": "full sparse score matrix; named/informed versus anonymized/blind; target models batched",
            "five_shot": "five highest-Pearson peers with target score observed and at least five shared visible scores",
        },
        "pathology_adaptation": {
            "score_scale": "normalized 0-100 for every retained evaluation",
            "selected_completion_rank": 1,
            "reasoning_flag": "omitted because pathology encoders do not have the upstream LLM reasoning-mode attribute",
        },
        "scores_sha256": scores_sha256,
        "folds_sha256": folds_sha256,
        "matrix_shape": [len(models), len(evaluations)],
        "models": list(models),
        "evaluations": list(evaluations),
        "fold_ids": [int(value) for value in fold_ids],
        "cell_limit": cell_limit,
        "n_shots": int(n_shots),
        "min_shared": int(min_shared),
        "max_target_known": int(max_target_known),
        "max_peer_shared": int(max_peer_shared),
        "batch_sizes": {
            "zero_shot_models": int(zero_shot_model_batch_size),
            "five_shot_named_cells": int(five_shot_named_cell_batch_size),
            "five_shot_blind_cells": int(five_shot_blind_cell_batch_size),
        },
        "conditions": list(CONDITIONS),
    }
    config["config_sha256"] = object_sha256(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    supplied = config.get("config_sha256")
    unsigned = {key: value for key, value in config.items() if key != "config_sha256"}
    expected = object_sha256(unsigned)
    if supplied != expected:
        raise ValueError("LLM baseline configuration hash mismatch")
    if tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError("LLM baseline conditions do not match the pinned condition set")
    if config.get("n_shots") != 5 or config.get("min_shared") != 5:
        raise ValueError("five-shot peer semantics drifted from the pinned protocol")
    if config.get("max_target_known") != 12 or config.get("max_peer_shared") != 4:
        raise ValueError("five-shot prompt caps drifted from the pinned upstream protocol")
    if config.get("batch_sizes") != {
        "zero_shot_models": 10,
        "five_shot_named_cells": 64,
        "five_shot_blind_cells": 16,
    }:
        raise ValueError("LLM baseline batch sizes drifted from the pinned upstream protocol")


def format_matrix_csv(
    matrix: np.ndarray,
    models: Sequence[str],
    evaluations: Sequence[str],
    *,
    blind: bool,
) -> str:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (len(models), len(evaluations)):
        raise ValueError("matrix identifiers do not match matrix shape")
    columns = [f"B{j}" for j in range(len(evaluations))] if blind else list(evaluations)
    lines = ["model," + ",".join(columns)]
    for i, model in enumerate(models):
        label = f"M{i}" if blind else model
        cells = ["?" if not np.isfinite(value) else f"{value:.3f}" for value in values[i]]
        lines.append(label + "," + ",".join(cells))
    return "\n".join(lines)


def _target_records(
    cells: Sequence[tuple[int, int]], models: Sequence[str], evaluations: Sequence[str]
) -> list[dict[str, Any]]:
    return [
        {
            "query_id": f"q{offset}",
            "model_index": int(i),
            "evaluation_index": int(j),
            "model_id": models[i],
            "evaluation_id": evaluations[j],
        }
        for offset, (i, j) in enumerate(cells)
    ]


def build_matrix_messages(
    train: np.ndarray,
    cells: Sequence[tuple[int, int]],
    models: Sequence[str],
    evaluations: Sequence[str],
    task_metadata: dict[str, dict[str, str]],
    *,
    blind: bool,
) -> list[dict[str, str]]:
    matrix_csv = format_matrix_csv(train, models, evaluations, blind=blind)
    if blind:
        definitions = "Model and evaluation identities and metadata are anonymized."
    else:
        definitions = "Evaluation definitions:\n" + "\n".join(
            f"- {evaluation}: family={task_metadata[evaluation].get('task_family', 'unknown')}; "
            f"sample_unit={task_metadata[evaluation].get('sample_unit', 'unknown')}; "
            f"metric={task_metadata[evaluation].get('metric', 'unknown')}"
            for evaluation in evaluations
        )
    system = (
        "You predict missing normalized pathology foundation-model evaluation scores. "
        "Scores use a common 0-100 orientation where larger is better. The retained "
        "matrix has strong low-dimensional structure; pathology validation selected a "
        "rank-1 completion model. Return JSON only, mapping each query_id to one number.\n\n"
        + definitions
        + "\n\nObserved matrix ('?' means hidden):\n"
        + matrix_csv
    )
    requests = []
    for offset, (i, j) in enumerate(cells):
        model = f"M{i}" if blind else models[i]
        evaluation = f"B{j}" if blind else evaluations[j]
        requests.append(f"q{offset}: predict {model} on {evaluation}")
    user = "Predict these hidden cells. Return only JSON like {\"q0\": 72.5}.\n" + "\n".join(requests)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _shared_columns(train: np.ndarray, target: int, peer: int, excluded: int) -> np.ndarray:
    mask = np.isfinite(train[target]) & np.isfinite(train[peer])
    mask[excluded] = False
    return np.flatnonzero(mask)


def select_peer_examples(
    train: np.ndarray,
    target: int,
    evaluation: int,
    *,
    n_shots: int = 5,
    min_shared: int = 5,
) -> list[dict[str, Any]]:
    """Select nearest peers using visible training cells only."""

    candidates = []
    for peer in range(train.shape[0]):
        if peer == target or not np.isfinite(train[peer, evaluation]):
            continue
        shared = _shared_columns(train, target, peer, evaluation)
        if len(shared) < min_shared:
            continue
        x, y = train[target, shared], train[peer, shared]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        correlation = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(correlation):
            candidates.append(
                {"peer_index": int(peer), "correlation": correlation, "shared_indices": [int(value) for value in shared]}
            )
    return sorted(candidates, key=lambda row: (-row["correlation"], row["peer_index"]))[:n_shots]


def build_five_shot_queries(
    train: np.ndarray,
    cells: Sequence[tuple[int, int]],
    *,
    n_shots: int = 5,
    min_shared: int = 5,
    max_target_known: int = 12,
    max_peer_shared: int = 4,
) -> list[dict[str, Any]]:
    queries = []
    for offset, (i, j) in enumerate(cells):
        target_known = [int(value) for value in np.flatnonzero(np.isfinite(train[i])) if int(value) != j][
            :max_target_known
        ]
        peers = select_peer_examples(train, i, j, n_shots=n_shots, min_shared=min_shared)
        for peer in peers:
            peer["shared_indices"] = peer["shared_indices"][:max_peer_shared]
        queries.append(
            {
                "query_id": f"q{offset}",
                "model_index": int(i),
                "evaluation_index": int(j),
                "target_known_indices": target_known,
                "examples": peers,
            }
        )
    return queries


def _score_list(
    train: np.ndarray,
    row: int,
    indices: Sequence[int],
    evaluations: Sequence[str],
    labels: dict[int, str] | None,
) -> str:
    return ", ".join(
        f"{labels[j] if labels is not None else evaluations[j]}={train[row, j]:.3f}" for j in indices
    ) or "none"


def render_five_shot_messages(
    train: np.ndarray,
    queries: Sequence[dict[str, Any]],
    models: Sequence[str],
    evaluations: Sequence[str],
    task_metadata: dict[str, dict[str, str]],
    *,
    blind: bool,
) -> list[dict[str, str]]:
    batch_labels = None
    if blind:
        used_indices: list[int] = []
        for query in queries:
            used_indices.extend([query["evaluation_index"], *query["target_known_indices"]])
            for example in query["examples"]:
                used_indices.extend(example["shared_indices"])
        unique_indices = list(dict.fromkeys(used_indices))
        batch_labels = {
            value: f"Benchmark {chr(ord('A') + offset) if offset < 26 else offset + 1}"
            for offset, value in enumerate(unique_indices)
        }
    lines = [
        "You are estimating pathology benchmark results before running expensive evaluations.",
        "Each query gives compact known scores for a target model and five nearest peer-model examples.",
        "Make a quick numerical estimate from the nearest peers; do not explain or show calculations.",
        "Return ONLY valid JSON mapping each query_id to a 0-100 numeric score, e.g. {\"q0\": 72.5}.",
    ]
    for query in queries:
        i, j = query["model_index"], query["evaluation_index"]
        labels = batch_labels
        target_model = f"Target model {query['query_id']}" if blind else models[i]
        target_evaluation = labels[j] if labels is not None else evaluations[j]
        if not blind:
            metadata = task_metadata[evaluations[j]]
            target_evaluation += (
                f" [family={metadata.get('task_family', 'unknown')}; "
                f"unit={metadata.get('sample_unit', 'unknown')}; metric={metadata.get('metric', 'unknown')}]"
            )
        lines.extend(
            [
                "",
                f"Query {query['query_id']}",
                f"Target model: {target_model}",
                f"Known target scores: {_score_list(train, i, query['target_known_indices'], evaluations, labels)}",
                f"Predict: {target_evaluation}",
                "Nearest peer examples:",
            ]
        )
        if not query["examples"]:
            lines.append("- No eligible peers.")
        for number, example in enumerate(query["examples"], 1):
            peer = example["peer_index"]
            peer_name = f"Peer model {query['query_id']}-{number}" if blind else models[peer]
            lines.append(
                f"- {peer_name}; shared={_score_list(train, peer, example['shared_indices'], evaluations, labels)}; "
                f"{target_evaluation}={train[peer, j]:.3f}"
            )
    return [{"role": "user", "content": "\n".join(lines)}]


def build_request(
    *,
    config: dict[str, Any],
    condition: str,
    fold_id: int,
    seed: int,
    fold: int,
    batch_index: int,
    train: np.ndarray,
    cells: Sequence[tuple[int, int]],
    models: Sequence[str],
    evaluations: Sequence[str],
    task_metadata: dict[str, dict[str, str]],
) -> dict[str, Any]:
    validate_config(config)
    if condition not in CONDITIONS:
        raise ValueError(f"unknown LLM baseline condition: {condition}")
    blind = condition.endswith("blind")
    if condition.startswith("zero_shot"):
        messages = build_matrix_messages(train, cells, models, evaluations, task_metadata, blind=blind)
        query_meta: list[dict[str, Any]] = []
    else:
        query_meta = build_five_shot_queries(
            train,
            cells,
            n_shots=config["n_shots"],
            min_shared=config["min_shared"],
            max_target_known=config["max_target_known"],
            max_peer_shared=config["max_peer_shared"],
        )
        messages = render_five_shot_messages(train, query_meta, models, evaluations, task_metadata, blind=blind)
    identity = {
        "config_sha256": config["config_sha256"],
        "condition": condition,
        "fold_id": int(fold_id),
        "batch_index": int(batch_index),
        "targets": [[int(i), int(j)] for i, j in cells],
    }
    request = {
        "schema_version": SCHEMA_VERSION,
        "request_id": object_sha256(identity)[:24],
        **identity,
        "seed": int(seed),
        "fold": int(fold),
        "targets": _target_records(cells, models, evaluations),
        "query_meta": query_meta,
        "messages": messages,
        "response_contract": {"type": "json_object", "values": "normalized_score_0_100"},
        "execution_status": "prepared_unrun",
    }
    request["request_sha256"] = object_sha256(request)
    return request


def validate_request(request: dict[str, Any], config: dict[str, Any]) -> None:
    validate_config(config)
    supplied = request.get("request_sha256")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if supplied != object_sha256(unsigned):
        raise ValueError("request hash mismatch")
    if request.get("config_sha256") != config["config_sha256"]:
        raise ValueError("request configuration mismatch")
    if request.get("condition") not in CONDITIONS:
        raise ValueError("request has an unknown condition")


def parse_prediction_payload(
    text: str, query_ids: set[str], *, strict: bool = False
) -> dict[str, float]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as error:
        if strict:
            raise ValueError("response_text is not valid JSON") from error
        return {}
    if isinstance(data, dict) and isinstance(data.get("predictions"), dict):
        data = data["predictions"]
    if not isinstance(data, dict):
        if strict:
            raise ValueError("response payload must be a JSON object")
        return {}
    if strict and set(data) != query_ids:
        missing = sorted(query_ids - set(data))
        extra = sorted(set(data) - query_ids)
        raise ValueError(f"response query set mismatch; missing={missing}; extra={extra}")
    parsed = {}
    for key, value in data.items():
        if key not in query_ids:
            if strict:
                raise ValueError(f"unexpected query id: {key}")
            continue
        if isinstance(value, dict):
            value = value.get("score", value.get("prediction"))
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            if strict:
                raise ValueError(f"non-numeric prediction for {key}") from error
            continue
        if not np.isfinite(numeric):
            if strict:
                raise ValueError(f"non-finite prediction for {key}")
            continue
        if strict and not 0.0 <= numeric <= 100.0:
            raise ValueError(f"prediction outside normalized 0-100 range for {key}")
        parsed[key] = float(numeric if strict else np.clip(numeric, 0.0, 100.0))
    return parsed


def seal_external_response(
    raw: dict[str, Any], request: dict[str, Any], config: dict[str, Any],
    execution_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly validate and hash one provider-neutral external response row."""

    validate_request(request, config)
    required = {"request_id", "backend_kind", "provider", "model", "response_text"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"external response is missing required fields: {missing}")
    extra = sorted(set(raw) - required - {"usage", "cost", "execution_metadata"})
    if extra:
        raise ValueError(f"external response has unsupported fields: {extra}")
    if raw["request_id"] != request["request_id"]:
        raise ValueError("external response request_id mismatch")
    if raw["backend_kind"] not in REAL_BACKEND_KINDS:
        raise ValueError("external response must declare a supported real backend kind")
    if not isinstance(raw["provider"], str) or not raw["provider"].strip():
        raise ValueError("external response provider must be non-empty")
    if not isinstance(raw["model"], str) or not raw["model"].strip():
        raise ValueError("external response model must be non-empty")
    execution_metadata = raw.get("execution_metadata", {})
    if not isinstance(execution_metadata, dict):
        raise ValueError("external response execution_metadata must be an object")
    metadata_extra = sorted(set(execution_metadata) - EXECUTION_METADATA_KEYS)
    if metadata_extra:
        raise ValueError(f"unsupported execution_metadata fields: {metadata_extra}")
    version = execution_metadata.get("model_version")
    if version is not None and (not isinstance(version, str) or not version.strip()):
        raise ValueError("execution_metadata model_version must be a non-empty string")
    settings = execution_metadata.get("settings")
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("execution_metadata settings must be an object")
    if "receipt" in execution_metadata and execution_metadata["receipt"] is None:
        raise ValueError("execution_metadata receipt must not be null when supplied")
    metadata_hashes = {
        key + "_sha256": object_sha256(value.strip() if key == "model_version" else value)
        for key, value in execution_metadata.items()
    }
    execution_lineage_sha256 = None
    if execution_contract is not None:
        unsigned_contract = dict(execution_contract)
        execution_lineage_sha256 = unsigned_contract.pop("execution_contract_sha256", None)
        if execution_lineage_sha256 != object_sha256(unsigned_contract):
            raise ValueError("approved execution contract self-hash mismatch")
        expected_model = execution_contract.get("response_model")
        if raw["provider"].strip() != execution_contract.get("provider") or raw["model"].strip() != expected_model:
            raise ValueError("raw response identity does not match approved execution contract")
        if metadata_hashes.get("settings_sha256") != execution_contract.get("required_settings_sha256"):
            raise ValueError("raw response settings do not match approved execution contract")
        version = execution_metadata.get("model_version")
        expected_version = execution_contract.get("required_model_version")
        if expected_version is not None and version != expected_version:
            raise ValueError("raw response model_version does not match approved snapshot")
        if expected_version is None:
            match = re.fullmatch(r"gpt-5\.5-(\d{4})-(\d{2})-(\d{2})", str(version))
            if match is None:
                raise ValueError("mutable alias requires a dated resolved gpt-5.5 model_version")
            try:
                date(*(int(value) for value in match.groups()))
            except ValueError as error:
                raise ValueError("mutable alias resolved model_version has an invalid date") from error
        receipt = execution_metadata.get("receipt")
        if not isinstance(receipt, dict):
            raise ValueError("raw response receipt must be an object")
        lineage = {
            "approval_manifest_sha256": execution_contract.get("approval_manifest_sha256"),
            "execution_preflight_sha256": execution_contract.get("preflight_sha256"),
            "execution_contract_sha256": execution_contract.get("execution_contract_sha256"),
            "transport_profile_sha256": execution_contract.get("transport_profile_sha256"),
            "capacity_profile_sha256": execution_contract.get("capacity_profile_sha256"),
            "settings_sha256": execution_contract.get("required_settings_sha256"),
            "model_sha256": object_sha256(expected_model),
        }
        if execution_contract.get("transport_kind") == "openai_batch":
            lineage["adapter_manifest_sha256"] = execution_contract.get("adapter_manifest_sha256")
        for key, expected in lineage.items():
            if not isinstance(expected, str) or receipt.get(key) != expected:
                raise ValueError(f"raw response receipt lineage mismatch: {key}")
    query_ids = {target["query_id"] for target in request["targets"]}
    parsed = parse_prediction_payload(raw["response_text"], query_ids, strict=True)
    usage = raw.get("usage", {"input_tokens": None, "output_tokens": None, "status": "not_reported"})
    if not isinstance(usage, dict):
        raise ValueError("external response usage must be an object")
    cost = raw.get("cost", {"status": "not_reported", "currency": None, "amount": None})
    if not isinstance(cost, dict):
        raise ValueError("external response cost must be an object")
    if cost.get("amount") is not None and (
        not isinstance(cost.get("amount"), (int, float))
        or isinstance(cost.get("amount"), bool)
        or not math.isfinite(float(cost["amount"]))
        or cost["amount"] < 0
    ):
        raise ValueError("external response cost amount must be nonnegative when reported")
    response = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "config_sha256": config["config_sha256"],
        "backend_kind": raw["backend_kind"],
        "provider": raw["provider"].strip(),
        "model": raw["model"].strip(),
        "status": "complete_validated_real_unapproved",
        "headline_eligible": False,
        "response_text": raw["response_text"],
        "parsed_predictions": parsed,
        "usage": usage,
        "cost": cost,
        "execution_metadata_hashes": metadata_hashes,
        "execution_lineage_sha256": execution_lineage_sha256,
    }
    response["response_sha256"] = object_sha256(response)
    return response


def summarize_real_execution(
    responses: Sequence[dict[str, Any]],
    *,
    execution_contract: dict[str, Any] | None = None,
    require_complete_evidence: bool = False,
) -> dict[str, Any]:
    """Require one fixed identity and optionally enforce an approved contract."""

    if not responses:
        raise ValueError("real response pack is empty")
    identities = {
        (row.get("backend_kind"), row.get("provider"), row.get("model")) for row in responses
    }
    if len(identities) != 1:
        raise ValueError(
            "real response pack must use exactly one backend_kind/provider/model identity"
        )
    backend_kind, provider, model = next(iter(identities))
    identity = {"backend_kind": backend_kind, "provider": provider, "model": model}
    metadata = [row.get("execution_metadata_hashes", {}) for row in responses]
    allowed = {"model_version_sha256", "settings_sha256", "receipt_sha256"}
    for record in metadata:
        if not isinstance(record, dict) or set(record) - allowed:
            raise ValueError("sealed response has invalid execution metadata hashes")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in record.values()
        ):
            raise ValueError("sealed response has malformed execution metadata hash")
    fixed_evidence: dict[str, Any] = {}
    for key in ("model_version_sha256", "settings_sha256"):
        values = {record[key] for record in metadata if key in record}
        if len(values) > 1:
            raise ValueError(f"real response pack has inconsistent {key}")
        fixed_evidence[key] = next(iter(values)) if values else None
        fixed_evidence[key.removesuffix("_sha256") + "_reported_responses"] = sum(
            key in record for record in metadata
        )
    receipt_records = sorted([
        {"request_id": row["request_id"], "receipt_sha256": record["receipt_sha256"]}
        for row, record in zip(responses, metadata)
        if "receipt_sha256" in record
    ], key=lambda row: row["request_id"])
    if require_complete_evidence:
        if execution_contract is None:
            raise ValueError("headline execution requires an approved execution contract")
        unsigned_contract = dict(execution_contract)
        supplied_contract_hash = unsigned_contract.pop("execution_contract_sha256", None)
        if supplied_contract_hash != object_sha256(unsigned_contract):
            raise ValueError("approved execution contract self-hash mismatch")
        expected_count = execution_contract.get("expected_response_count")
        if expected_count != len(responses):
            raise ValueError(
                f"execution contract expects {expected_count} responses, received {len(responses)}"
            )
        expected_identity = {
            "provider": execution_contract.get("provider"),
            "model": execution_contract.get("response_model"),
        }
        if provider != expected_identity["provider"] or model != expected_identity["model"]:
            raise ValueError("real response provider/model does not match the approved model profile")
        required_settings = execution_contract.get("required_settings_sha256")
        if not isinstance(required_settings, str) or len(required_settings) != 64:
            raise ValueError("approved execution contract has no valid required_settings_sha256")
        required_version = execution_contract.get("required_model_version")
        required_version_hash = None if required_version is None else object_sha256(required_version)
        missing_settings = [
            response["request_id"] for response, record in zip(responses, metadata)
            if "settings_sha256" not in record
        ]
        missing_versions = [
            response["request_id"] for response, record in zip(responses, metadata)
            if "model_version_sha256" not in record
        ]
        missing_receipts = [
            response["request_id"] for response, record in zip(responses, metadata)
            if "receipt_sha256" not in record
        ]
        if missing_settings:
            raise ValueError(f"settings evidence missing for {len(missing_settings)} responses")
        if missing_versions:
            raise ValueError(f"model-version evidence missing for {len(missing_versions)} responses")
        if missing_receipts:
            raise ValueError(f"provider receipt evidence missing for {len(missing_receipts)} responses")
        if fixed_evidence["settings_sha256"] != required_settings:
            raise ValueError("response settings are consistent but do not match the approved settings hash")
        if required_version_hash is not None and fixed_evidence["model_version_sha256"] != required_version_hash:
            raise ValueError("response model-version evidence does not match the approved snapshot")
        if (
            execution_contract.get("require_resolved_version_distinct_from_alias")
            and fixed_evidence["model_version_sha256"] == object_sha256(execution_contract.get("model_alias"))
        ):
            raise ValueError("mutable alias was repeated as model_version; resolved provider model-version evidence is required")
    return {
        "fixed_identity": identity,
        "fixed_identity_sha256": object_sha256(identity),
        "model_sha256": object_sha256(model),
        **fixed_evidence,
        "receipt_reported_responses": len(receipt_records),
        "receipt_pack_sha256": object_sha256(receipt_records) if receipt_records else None,
        "execution_contract_sha256": (
            object_sha256(execution_contract) if execution_contract is not None else None
        ),
        "complete_execution_evidence": bool(require_complete_evidence),
    }


def authorize_external_responses(
    responses: Sequence[dict[str, Any]], execution_contract: dict[str, Any]
) -> list[dict[str, Any]]:
    """Promote a complete evidence-matched pack to headline-eligible responses."""

    for response in responses:
        supplied = response.get("response_sha256")
        unsigned = {key: value for key, value in response.items() if key != "response_sha256"}
        if supplied != object_sha256(unsigned):
            raise ValueError("cannot authorize a response with a mismatched seal hash")
        if (
            response.get("backend_kind") not in REAL_BACKEND_KINDS
            or response.get("status") != "complete_validated_real_unapproved"
            or response.get("headline_eligible") is not False
        ):
            raise ValueError("response is not a validated unapproved real response")
        if response.get("execution_lineage_sha256") != execution_contract.get("execution_contract_sha256"):
            raise ValueError("response lacks validated receipt lineage for this execution contract")
    summarize_real_execution(
        responses,
        execution_contract=execution_contract,
        require_complete_evidence=True,
    )
    approval_sha256 = execution_contract.get("approval_manifest_sha256")
    preflight_sha256 = execution_contract.get("preflight_sha256")
    if not all(isinstance(value, str) and len(value) == 64 for value in (approval_sha256, preflight_sha256)):
        raise ValueError("approved execution contract is missing approval/preflight hashes")
    authorized = []
    for response in responses:
        row = dict(response)
        row["status"] = "complete_validated_real"
        row["headline_eligible"] = True
        row["execution_approval_sha256"] = approval_sha256
        row["execution_preflight_sha256"] = preflight_sha256
        row.pop("response_sha256", None)
        row["response_sha256"] = object_sha256(row)
        authorized.append(row)
    return authorized


def deterministic_mock_response(
    request: dict[str, Any], train: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    """Return a deterministic contract test response, never a real LLM result."""

    validate_request(request, config)
    predictions = {}
    meta_by_id = {row["query_id"]: row for row in request.get("query_meta", [])}
    for target in request["targets"]:
        qid, j = target["query_id"], target["evaluation_index"]
        meta = meta_by_id.get(qid)
        if meta and meta.get("examples"):
            peer_values = [train[example["peer_index"], j] for example in meta["examples"]]
            prediction = float(np.mean(peer_values))
        else:
            prediction = float(np.nanmedian(train[:, j]))
        predictions[qid] = round(float(np.clip(prediction, 0.0, 100.0)), 6)
    content = canonical_json(predictions)
    response = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "config_sha256": config["config_sha256"],
        "backend_kind": "deterministic_mock",
        "provider": "none",
        "model": "deterministic-column-or-peer-mean",
        "status": "complete_mock_only",
        "headline_eligible": False,
        "response_text": content,
        "parsed_predictions": predictions,
        "usage": {"input_tokens": None, "output_tokens": None, "status": "not_applicable"},
    }
    response["response_sha256"] = object_sha256(response)
    return response


def validate_response(
    response: dict[str, Any], request: dict[str, Any], config: dict[str, Any], *,
    require_real: bool = False,
    execution_contract: dict[str, Any] | None = None,
) -> None:
    validate_request(request, config)
    supplied = response.get("response_sha256")
    unsigned = {key: value for key, value in response.items() if key != "response_sha256"}
    if supplied != object_sha256(unsigned):
        raise ValueError("response hash mismatch")
    for key in ("request_id", "request_sha256", "config_sha256"):
        expected = request["request_id"] if key == "request_id" else request[key]
        if response.get(key) != expected:
            raise ValueError(f"response {key} mismatch")
    if response.get("backend_kind") == "deterministic_mock" and response.get("headline_eligible") is not False:
        raise ValueError("mock responses must never be headline eligible")
    if require_real:
        if response.get("backend_kind") not in REAL_BACKEND_KINDS:
            raise ValueError("real merge received a non-real backend kind")
        if response.get("headline_eligible") is not True or response.get("status") != "complete_validated_real":
            raise ValueError("real response was not sealed as a complete validated response")
        if execution_contract is None:
            raise ValueError("headline response validation requires an approved execution contract")
        if response.get("execution_approval_sha256") != execution_contract.get("approval_manifest_sha256"):
            raise ValueError("response execution approval hash mismatch")
        if response.get("execution_preflight_sha256") != execution_contract.get("preflight_sha256"):
            raise ValueError("response execution preflight hash mismatch")
        expected = {target["query_id"] for target in request["targets"]}
        parsed = parse_prediction_payload(response.get("response_text", ""), expected, strict=True)
        if response.get("parsed_predictions") != parsed:
            raise ValueError("sealed parsed_predictions do not match response_text")


def evaluate_cached_responses(
    requests: Sequence[dict[str, Any]],
    responses: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    config: dict[str, Any],
    *,
    require_complete: bool = False,
    require_real: bool = False,
    execution_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_request = {row["request_id"]: row for row in responses}
    if len(by_request) != len(responses):
        raise ValueError("duplicate cached response request_id")
    request_ids = {row["request_id"] for row in requests}
    unexpected = sorted(set(by_request) - request_ids)
    missing = sorted(request_ids - set(by_request))
    if unexpected:
        raise ValueError(f"responses contain unknown request_ids: {unexpected[:5]}")
    if require_complete and missing:
        raise ValueError(f"response pack is incomplete: {len(missing)} request_ids missing")
    execution = (
        summarize_real_execution(
            responses,
            execution_contract=execution_contract,
            require_complete_evidence=True,
        )
        if require_real else None
    )
    raw = []
    backend_kinds = set()
    for request in requests:
        response = by_request.get(request["request_id"])
        if response is None:
            continue
        validate_response(
            response, request, config,
            require_real=require_real,
            execution_contract=execution_contract,
        )
        backend_kinds.add(response["backend_kind"])
        parsed = parse_prediction_payload(
            response["response_text"],
            {target["query_id"] for target in request["targets"]},
            strict=require_complete,
        )
        for target in request["targets"]:
            qid, i, j = target["query_id"], target["model_index"], target["evaluation_index"]
            if qid not in parsed:
                continue
            raw.append(
                {
                    "request_id": request["request_id"],
                    "condition": request["condition"],
                    "fold_id": request["fold_id"],
                    "seed": request["seed"],
                    "fold": request["fold"],
                    "model_id": target["model_id"],
                    "evaluation_id": target["evaluation_id"],
                    "actual": float(matrix[i, j]),
                    "predicted": float(parsed[qid]),
                    "backend_kind": response["backend_kind"],
                    "provider": response.get("provider"),
                    "provider_model": response.get("model"),
                    "headline_eligible": bool(response["headline_eligible"]),
                }
            )
    summaries = []
    fold_metrics = []
    for condition in CONDITIONS:
        rows = [row for row in raw if row["condition"] == condition]
        errors = np.asarray([abs(row["predicted"] - row["actual"]) for row in rows], dtype=float)
        ape = np.asarray(
            [100 * abs(row["predicted"] - row["actual"]) / abs(row["actual"]) for row in rows if abs(row["actual"]) > 1e-12],
            dtype=float,
        )
        requested = sum(len(request["targets"]) for request in requests if request["condition"] == condition)
        condition_folds = []
        for fold_id in sorted({row["fold_id"] for row in rows}):
            fold_rows = [row for row in rows if row["fold_id"] == fold_id]
            fold_errors = np.asarray(
                [abs(row["predicted"] - row["actual"]) for row in fold_rows], dtype=float
            )
            fold_ape = np.asarray(
                [
                    100 * abs(row["predicted"] - row["actual"]) / abs(row["actual"])
                    for row in fold_rows if abs(row["actual"]) > 1e-12
                ],
                dtype=float,
            )
            record = {
                "condition": condition,
                "fold_id": int(fold_id),
                "seed": int(fold_rows[0]["seed"]),
                "fold": int(fold_rows[0]["fold"]),
                "n": len(fold_rows),
                "medae": float(np.median(fold_errors)) if len(fold_errors) else None,
                "medape": float(np.median(fold_ape)) if len(fold_ape) else None,
            }
            condition_folds.append(record)
            fold_metrics.append(record)
        medae_folds = [row["medae"] for row in condition_folds if row["medae"] is not None]
        medape_folds = [row["medape"] for row in condition_folds if row["medape"] is not None]
        summaries.append(
            {
                "condition": condition,
                "n": len(rows),
                "n_requested": requested,
                "coverage": len(rows) / requested if requested else 0.0,
                "n_folds": len(condition_folds),
                "medae": float(np.median(medae_folds)) if medae_folds else None,
                "medape": float(np.median(medape_folds)) if medape_folds else None,
                "pooled_medae": float(np.median(errors)) if len(errors) else None,
                "pooled_medape": float(np.median(ape)) if len(ape) else None,
            }
        )
    complete = len(responses) == len(requests) and all(
        row["coverage"] == 1.0 for row in summaries if row["n_requested"] > 0
    )
    headline = complete and bool(backend_kinds) and backend_kinds.issubset(REAL_BACKEND_KINDS) and all(
        row["headline_eligible"] for row in raw
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config["config_sha256"],
        "backend_kinds": sorted(backend_kinds),
        "request_count": len(requests),
        "response_count": len(responses),
        "missing_response_count": len(missing),
        "complete": complete,
        "headline_eligible": headline,
        "result_status": "real_cached_results" if headline else ("mock_contract_validation_only" if raw else "unrun"),
        "summary": summaries,
        "fold_metrics": fold_metrics,
        "raw_predictions": raw,
    }
    if execution is not None:
        result["real_execution"] = execution
    return result
