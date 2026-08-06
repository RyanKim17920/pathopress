"""Offline token, capacity, and cost preflight for provider-neutral LLM packs.

The functions in this module never contact a provider and never infer a model,
tokenizer, price, or capacity limit.  Model-agnostic token scenarios are
deliberately labelled estimates; an optional explicitly selected ``tiktoken``
encoding adds a content-token count without pretending to know a provider's
chat template.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .llm_baseline import CONDITIONS, canonical_json, object_sha256, validate_config, validate_request


PREFLIGHT_SCHEMA_VERSION = 1
PINNED_INVOCATION_SETTINGS = {
    "schema_version": 1,
    "profile_kind": "literal_upstream_contract",
    "temperature": 0.0,
    "max_output_tokens_per_request": 16384,
    "provider_parameter_name": "max_tokens",
    "response_format": "json_object_requested_in_prompt",
    "endpoint_api": "chat_completions",
    "source_repository_commit": "0a684b63ee0e4a401cb907a3827a82ea997d74c4",
}
ESTIMATOR_SPEC = {
    "name": "pathopress_model_agnostic_chat_v1",
    "input_serialization": "canonical JSON of the request messages array",
    "best": "ceil(UTF-8 bytes / 4) + 3 tokens per message + 3 reply-priming tokens",
    "base": "ceil(UTF-8 bytes / 3) + 6 tokens per message + 6 reply-priming tokens",
    "worst": "UTF-8 bytes + 32 tokens per message + 64 reply-priming tokens",
    "output_best": "compact JSON with every query value rendered as 0",
    "output_base": "compact JSON with every query value rendered as 72.5",
    "output_worst": "configured max_output_tokens_per_request ceiling",
    "scope_note": (
        "The worst heuristic is conservative for common byte-backed tokenizers but is not a "
        "formal upper bound for every possible proprietary tokenizer or chat template."
    ),
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_file_sha256(path: Path) -> str:
    """Hash parsed JSON, so whitespace-only changes do not change a profile binding."""

    return object_sha256(json.loads(path.read_text(encoding="utf-8")))


def _require_nonnegative_int(value: Any, name: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < (1 if positive else 0):
        relation = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {relation} integer")
    return value


def _optional_positive_int(mapping: dict[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    return _require_nonnegative_int(value, key, positive=True)


def validate_execution_settings(settings: dict[str, Any]) -> None:
    if settings.get("schema_version") != 1:
        raise ValueError("execution settings schema_version must be 1")
    _require_nonnegative_int(
        settings.get("max_output_tokens_per_request"),
        "max_output_tokens_per_request",
        positive=True,
    )
    if "temperature" in settings and (
        not isinstance(settings["temperature"], (int, float))
        or isinstance(settings["temperature"], bool)
        or not math.isfinite(float(settings["temperature"]))
    ):
        raise ValueError("temperature must be a finite numeric value when supplied")


def validate_transport_profile(profile: dict[str, Any]) -> None:
    """Validate transport independently from the inner model invocation."""

    if profile.get("schema_version") != 1:
        raise ValueError("transport profile schema_version must be 1")
    kind = profile.get("transport_kind")
    if kind not in {"online_chat_completions", "openai_batch"}:
        raise ValueError("unknown transport_kind")
    if profile.get("pricing_rate_key") not in {"online", "batch"}:
        raise ValueError("transport pricing_rate_key must be online or batch")
    limits = profile.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("transport profile limits must be an object")
    allowed_limits = {"max_requests_per_batch_file", "max_batch_file_bytes"}
    if set(limits) - allowed_limits:
        raise ValueError("transport limits may contain only file request-count and byte limits")
    for key in allowed_limits:
        _optional_positive_int(limits, key)
    if kind == "online_chat_completions":
        if profile.get("protocol_compatibility") != "literal_upstream_online_transport":
            raise ValueError("online transport must retain the literal upstream label")
        if profile.get("pricing_rate_key") != "online":
            raise ValueError("online transport must select online pricing")
    else:
        if profile.get("protocol_compatibility") != "batch_transport_adaptation_not_upstream_exact":
            raise ValueError("Batch transport must be explicitly labelled as an adaptation")
        if profile.get("pricing_rate_key") != "batch":
            raise ValueError("Batch transport must select batch pricing")
        if profile.get("completion_window") != "24h":
            raise ValueError("OpenAI Batch completion_window must be 24h")
        _require_nonnegative_int(
            profile.get("expected_input_file_count"),
            "expected_input_file_count",
            positive=True,
        )


def validate_model_profile(profile: dict[str, Any]) -> None:
    if profile.get("schema_version") != 1:
        raise ValueError("model profile schema_version must be 1")
    if profile.get("profile_type") not in {
        "user_supplied", "official_checked_in", "upstream_contract", "controlled_adaptation"
    }:
        raise ValueError(
            "model profile_type must be user_supplied, official_checked_in, "
            "upstream_contract, or controlled_adaptation"
        )
    for key in ("provider", "model_alias"):
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            raise ValueError(f"model profile {key} must be a non-empty string")
    if profile.get("model_snapshot") is not None and (
        not isinstance(profile["model_snapshot"], str) or not profile["model_snapshot"].strip()
    ):
        raise ValueError("model_snapshot must be null or a non-empty string")
    endpoint = profile.get("endpoint")
    if not isinstance(endpoint, dict) or not isinstance(endpoint.get("api"), str) or not endpoint["api"].strip():
        raise ValueError("model profile endpoint.api must be a non-empty string")
    compatibility = profile.get("protocol_compatibility")
    if compatibility not in {
        "literal_upstream_contract_mutable_alias",
        "controlled_snapshot_substitution_not_upstream_exact",
        "adapted_current_api_not_upstream_exact",
    }:
        raise ValueError("model profile has an unknown protocol_compatibility label")
    limits = profile.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("model profile limits must be an object")
    for key in (
        "context_window_tokens",
        "max_output_tokens",
        "max_input_tokens_per_request",
    ):
        _optional_positive_int(limits, key)
    if profile["profile_type"] in {"official_checked_in", "upstream_contract", "controlled_adaptation"}:
        source = profile.get("source")
        if not isinstance(source, dict):
            raise ValueError("official model profile requires a source object")
        for key in ("url", "title", "retrieved_at", "effective_date"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                raise ValueError(f"official model profile source requires {key}")


def validate_capacity_profile(profile: dict[str, Any], model_profile: dict[str, Any] | None) -> None:
    if profile.get("schema_version") != 1:
        raise ValueError("capacity profile schema_version must be 1")
    if model_profile is None:
        raise ValueError("capacity profile requires an explicit model profile")
    for key in ("provider", "model_alias", "model_snapshot"):
        if profile.get(key) != model_profile.get(key):
            raise ValueError(f"capacity profile {key} must exactly match the model profile")
    if not isinstance(profile.get("account_scope_label"), str) or not profile["account_scope_label"].strip():
        raise ValueError("capacity profile requires a nonsecret account_scope_label")
    evidence = profile.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("capacity profile requires an evidence object")
    for key in ("source", "retrieved_at"):
        if not isinstance(evidence.get(key), str) or not evidence[key].strip():
            raise ValueError(f"capacity profile evidence requires {key}")
    if evidence.get("active_credential_scope_attested") is not True:
        raise ValueError("capacity profile must attest that the active API credential/project matches its scope")
    limits = profile.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("capacity profile limits must be an object")
    for key in (
        "context_window_tokens", "max_output_tokens",
        "max_input_tokens_per_request", "max_queued_input_tokens",
    ):
        _optional_positive_int(limits, key)


def validate_profile_settings_contract(profile: dict[str, Any], settings: dict[str, Any]) -> None:
    """Prevent a modified run from being labelled as the literal upstream contract."""

    validate_model_profile(profile)
    validate_execution_settings(settings)
    if settings.get("endpoint_api") != profile["endpoint"]["api"]:
        raise ValueError("execution settings endpoint_api does not match the model profile")
    if profile["profile_type"] == "upstream_contract":
        required_profile = {
            "provider": "OpenAI",
            "model_alias": "gpt-5.5",
            "model_snapshot": None,
            "protocol_compatibility": "literal_upstream_contract_mutable_alias",
        }
        for key, expected in required_profile.items():
            if profile.get(key) != expected:
                raise ValueError(f"upstream_contract requires {key}={expected!r}")
        if profile["endpoint"] != {"api": "chat_completions", "base_url": None}:
            raise ValueError("upstream_contract requires the unmodified Chat Completions endpoint contract")
        if settings != PINNED_INVOCATION_SETTINGS:
            raise ValueError("upstream_contract invocation settings must exactly equal the pinned settings object")
    if profile["profile_type"] == "controlled_adaptation" and profile.get("model_snapshot") is None:
        raise ValueError("controlled snapshot adaptation requires an explicit immutable model_snapshot")
    if profile["profile_type"] == "controlled_adaptation" and settings != PINNED_INVOCATION_SETTINGS:
        raise ValueError("controlled snapshot adaptation must preserve the exact pinned invocation settings")


def validate_pricing_profile(profile: dict[str, Any], model_profile: dict[str, Any]) -> None:
    if profile.get("schema_version") != 1:
        raise ValueError("pricing profile schema_version must be 1")
    if profile.get("profile_type") not in {"user_supplied", "official_checked_in"}:
        raise ValueError("pricing profile_type must be user_supplied or official_checked_in")
    for key in ("provider", "model_alias", "model_snapshot"):
        if profile.get(key) != model_profile.get(key):
            raise ValueError(f"pricing profile {key} must exactly match the model profile")
    currency = profile.get("currency")
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("pricing profile currency must be a non-empty string")
    rates = profile.get("rates")
    if not isinstance(rates, dict) or not rates:
        raise ValueError("pricing profile rates must be a non-empty object")
    for mode, values in rates.items():
        if mode not in {"online", "batch"} or not isinstance(values, dict):
            raise ValueError("pricing rates may contain only online and batch objects")
        for key in ("input_per_million_tokens", "output_per_million_tokens"):
            value = values.get(key)
            if (
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value)) or value < 0
            ):
                raise ValueError(f"pricing {mode}.{key} must be nonnegative")
    if profile["profile_type"] == "official_checked_in":
        source = profile.get("source")
        if not isinstance(source, dict):
            raise ValueError("official pricing profile requires a source object")
        for key in ("url", "title", "retrieved_at", "effective_date"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                raise ValueError(f"official pricing profile source requires {key}")


def load_request_pack(request_dir: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read, validate, and inventory deterministic gzip request shards."""

    validate_config(config)
    requests: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for shard_index, path in enumerate(sorted(request_dir.glob("requests-*.jsonl.gz"))):
        compressed = path.read_bytes()
        uncompressed = gzip.decompress(compressed)
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(uncompressed.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid request JSON at {path}:{line_number}") from error
            validate_request(row, config)
            request_id = row["request_id"]
            if request_id in seen:
                raise ValueError(f"duplicate request_id in request pack: {request_id}")
            seen.add(request_id)
            rows.append(row)
        requests.extend(rows)
        shards.append(
            {
                "shard_index": shard_index,
                "path": path.name,
                "request_count": len(rows),
                "compressed_bytes": len(compressed),
                "uncompressed_bytes": len(uncompressed),
                "sha256": _sha256_bytes(compressed),
                "canonical_uncompressed_sha256": _sha256_bytes(uncompressed),
                "request_ids": [row["request_id"] for row in rows],
            }
        )
    if not shards:
        raise ValueError(f"no request shards found in {request_dir}")
    return requests, shards


def verify_pack_index(
    index: dict[str, Any],
    requests: Sequence[dict[str, Any]],
    shards: Sequence[dict[str, Any]],
    request_dir: Path,
) -> None:
    if index.get("request_count") != len(requests) or index.get("shard_count") != len(shards):
        raise ValueError("request pack index count mismatch")
    index_shards = index.get("shards")
    if not isinstance(index_shards, list) or len(index_shards) != len(shards):
        raise ValueError("request pack index shard inventory mismatch")
    for expected, observed in zip(index_shards, shards):
        for key in ("path", "sha256"):
            if expected.get(key) != observed.get(key):
                raise ValueError(f"request pack index {key} mismatch for {observed['path']}")
        if expected.get("bytes") != observed.get("compressed_bytes"):
            raise ValueError(f"request pack index byte count mismatch for {observed['path']}")
    compressed_digest = hashlib.sha256()
    uncompressed_digest = hashlib.sha256()
    for row in shards:
        compressed_digest.update(row["path"].encode("utf-8"))
        compressed_digest.update(bytes.fromhex(row["sha256"]))
        # The upstream pack digest concatenates decompressed shard payloads.
        uncompressed_digest.update(gzip.decompress((request_dir / row["path"]).read_bytes()))
    if compressed_digest.hexdigest() != index.get("pack_sha256"):
        raise ValueError("request pack compressed hash mismatch")
    if uncompressed_digest.hexdigest() != index.get("canonical_uncompressed_sha256"):
        raise ValueError("request pack canonical-uncompressed hash mismatch")


def _heuristic_tokens(byte_count: int, message_count: int) -> dict[str, int]:
    return {
        "best": math.ceil(byte_count / 4) + 3 * message_count + 3,
        "base": math.ceil(byte_count / 3) + 6 * message_count + 6,
        "worst": byte_count + 32 * message_count + 64,
    }


def _load_tiktoken(encoding_name: str | None):
    if encoding_name is None:
        return None
    try:
        import tiktoken  # type: ignore
    except ImportError as error:
        raise ValueError("tiktoken encoding was requested but tiktoken is not installed") from error
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as error:  # pragma: no cover - backend-specific exception type
        raise ValueError(f"unknown tiktoken encoding: {encoding_name}") from error


def _tiktoken_content_count(messages: Sequence[dict[str, str]], encoder: Any) -> int:
    return sum(len(encoder.encode(str(message["role"]))) + len(encoder.encode(str(message["content"]))) for message in messages)


def _output_payload(target_count: int, value: str) -> str:
    return "{" + ",".join(f'"q{index}":{value}' for index in range(target_count)) + "}"


def estimate_requests(
    requests: Sequence[dict[str, Any]],
    settings: dict[str, Any],
    *,
    tiktoken_encoding: str | None = None,
) -> list[dict[str, Any]]:
    validate_execution_settings(settings)
    encoder = _load_tiktoken(tiktoken_encoding)
    max_output = int(settings["max_output_tokens_per_request"])
    estimates = []
    for request in requests:
        messages = request["messages"]
        wire = canonical_json(messages).encode("utf-8")
        input_scenarios = _heuristic_tokens(len(wire), len(messages))
        target_count = len(request["targets"])
        best_output = _output_payload(target_count, "0").encode("utf-8")
        base_output = _output_payload(target_count, "72.5").encode("utf-8")
        best_tokens = _heuristic_tokens(len(best_output), 0)["best"]
        base_tokens = _heuristic_tokens(len(base_output), 0)["base"]
        tiktoken_counts = None
        if encoder is not None:
            tiktoken_counts = {
                "encoding": tiktoken_encoding,
                "scope": "message roles and contents only; provider chat-template overhead excluded",
                "input_content_tokens": _tiktoken_content_count(messages, encoder),
                "output_compact_zero_tokens": len(encoder.encode(best_output.decode("utf-8"))),
                "output_compact_base_tokens": len(encoder.encode(base_output.decode("utf-8"))),
            }
        estimates.append(
            {
                "request_id": request["request_id"],
                "request_sha256": request["request_sha256"],
                "condition": request["condition"],
                "fold_id": request["fold_id"],
                "batch_index": request["batch_index"],
                "target_count": target_count,
                "message_count": len(messages),
                "message_json_utf8_bytes": len(wire),
                "input_tokens": input_scenarios,
                "output_tokens": {
                    "best": best_tokens,
                    "base": base_tokens,
                    "worst": max_output,
                    "configured_max": max_output,
                },
                "configured_context_tokens": {
                    scenario: input_scenarios[scenario] + (max_output if scenario == "worst" else (best_tokens if scenario == "best" else base_tokens))
                    for scenario in ("best", "base", "worst")
                },
                "tiktoken": tiktoken_counts,
            }
        )
    return estimates


def _sum_scenarios(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    materialized = list(rows)
    return {
        scenario: sum(int(row[field][scenario]) for row in materialized)
        for scenario in ("best", "base", "worst")
    }


def _group_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = _sum_scenarios(rows, "input_tokens")
    output_tokens = _sum_scenarios(rows, "output_tokens")
    return {
        "request_count": len(rows),
        "target_count": sum(row["target_count"] for row in rows),
        "message_json_utf8_bytes": sum(row["message_json_utf8_bytes"] for row in rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": {key: input_tokens[key] + output_tokens[key] for key in input_tokens},
        "max_request_input_tokens": {
            key: max((row["input_tokens"][key] for row in rows), default=0)
            for key in input_tokens
        },
        "max_request_context_tokens": {
            key: max((row["configured_context_tokens"][key] for row in rows), default=0)
            for key in input_tokens
        },
    }


def _limit_check(value: int, limit: int | None, name: str) -> dict[str, Any]:
    return {
        "metric": name,
        "observed": value,
        "limit": limit,
        "status": "unknown_limit_not_supplied" if limit is None else ("pass" if value <= limit else "fail"),
        "headroom": None if limit is None else limit - value,
    }


def _batch_transport_payload(
    rows: Sequence[dict[str, Any]],
    settings: dict[str, Any],
    model_profile: dict[str, Any],
) -> bytes:
    """Materialize the exact OpenAI Batch JSONL wire plan without an API call."""

    model = model_profile.get("model_snapshot") or model_profile["model_alias"]
    parameter_name = settings["provider_parameter_name"]
    body_setting = settings["max_output_tokens_per_request"]
    lines = []
    for request in rows:
        line = {
            "custom_id": request["request_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": request["messages"],
                "temperature": settings["temperature"],
                parameter_name: body_setting,
            },
        }
        lines.append(canonical_json(line) + "\n")
    return "".join(lines).encode("utf-8")


def _cost_scenarios(
    totals: dict[str, Any],
    pricing: dict[str, Any] | None,
    transport_profile: dict[str, Any],
) -> dict[str, Any]:
    if pricing is None:
        return {
            "status": "unavailable_no_explicit_pricing_profile",
            "currency": None,
            "scenarios": None,
        }
    mode = transport_profile["pricing_rate_key"]
    if mode not in pricing["rates"]:
        return {
            "status": f"unavailable_no_{mode}_rates_in_explicit_profile",
            "currency": pricing["currency"],
            "scenarios": None,
        }
    rates = pricing["rates"][mode]
    scenarios = {}
    for scenario in ("best", "base", "worst"):
        input_cost = totals["input_tokens"][scenario] * rates["input_per_million_tokens"] / 1_000_000
        output_cost = totals["output_tokens"][scenario] * rates["output_per_million_tokens"] / 1_000_000
        scenarios[scenario] = {
            "input": input_cost,
            "output": output_cost,
            "total": input_cost + output_cost,
        }
    return {
        "status": "estimated_from_explicit_profile",
        "currency": pricing["currency"],
        "pricing_rate_key": mode,
        "rates": rates,
        "scenarios": scenarios,
    }


def build_preflight(
    *,
    requests: Sequence[dict[str, Any]],
    shards: Sequence[dict[str, Any]],
    pack_index: dict[str, Any],
    config: dict[str, Any],
    settings: dict[str, Any],
    transport_profile: dict[str, Any],
    model_profile: dict[str, Any] | None,
    capacity_profile: dict[str, Any] | None,
    pricing_profile: dict[str, Any] | None,
    budget: dict[str, Any] | None,
    adapter_manifest: dict[str, Any] | None = None,
    tiktoken_encoding: str | None = None,
    acknowledge_mutable_alias: bool = False,
    acknowledge_cost_estimate_uncertainty: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Build a hash-bound preflight, detailed request ledger, and approval gate."""

    validate_config(config)
    validate_execution_settings(settings)
    validate_transport_profile(transport_profile)
    if model_profile is not None:
        validate_profile_settings_contract(model_profile, settings)
    if capacity_profile is not None:
        validate_capacity_profile(capacity_profile, model_profile)
    if pricing_profile is not None:
        if model_profile is None:
            raise ValueError("pricing profile cannot be used without an explicit model profile")
        validate_pricing_profile(pricing_profile, model_profile)
    if budget is not None:
        if not isinstance(budget.get("currency"), str) or not budget["currency"]:
            raise ValueError("budget currency must be a non-empty string")
        maximum = budget.get("max_amount")
        if (
            not isinstance(maximum, (int, float)) or isinstance(maximum, bool)
            or not math.isfinite(float(maximum)) or maximum < 0
        ):
            raise ValueError("budget max_amount must be nonnegative")
    estimates = estimate_requests(requests, settings, tiktoken_encoding=tiktoken_encoding)
    estimate_by_id = {row["request_id"]: row for row in estimates}
    request_by_id = {row["request_id"]: row for row in requests}
    by_condition = {
        condition: _group_summary([row for row in estimates if row["condition"] == condition])
        for condition in CONDITIONS
    }
    totals = _group_summary(estimates)
    model_limits = (
        capacity_profile["limits"]
        if capacity_profile is not None
        else (model_profile.get("limits", {}) if model_profile is not None else {})
    )
    transport_limits = transport_profile["limits"]
    max_output = int(settings["max_output_tokens_per_request"])
    request_checks = []
    for row in estimates:
        checks = [
            _limit_check(row["input_tokens"]["worst"], _optional_positive_int(model_limits, "max_input_tokens_per_request"), "worst_estimated_input_tokens"),
            _limit_check(row["configured_context_tokens"]["worst"], _optional_positive_int(model_limits, "context_window_tokens"), "worst_estimated_input_plus_configured_output"),
        ]
        request_checks.append(
            {
                "request_id": row["request_id"],
                "checks": checks,
                "status": "fail" if any(check["status"] == "fail" for check in checks) else ("unknown" if any(check["status"].startswith("unknown") for check in checks) else "pass"),
            }
        )
    shard_summaries = []
    for shard in shards:
        rows = [estimate_by_id[value] for value in shard["request_ids"]]
        source_requests = [request_by_id[value] for value in shard["request_ids"]]
        summary = _group_summary(rows)
        batch_payload = (
            _batch_transport_payload(source_requests, settings, model_profile)
            if transport_profile["transport_kind"] == "openai_batch" and model_profile is not None
            else None
        )
        batch_bytes = None if batch_payload is None else len(batch_payload)
        byte_check = (
            {
                "metric": "openai_batch_jsonl_bytes",
                "observed": None,
                "limit": _optional_positive_int(transport_limits, "max_batch_file_bytes"),
                "status": "unknown_model_profile_not_supplied",
                "headroom": None,
            }
            if transport_profile["transport_kind"] == "openai_batch" and batch_bytes is None
            else _limit_check(
                shard["uncompressed_bytes"] if batch_bytes is None else batch_bytes,
                _optional_positive_int(transport_limits, "max_batch_file_bytes"),
                "canonical_source_request_bytes" if batch_bytes is None else "openai_batch_jsonl_bytes",
            )
        )
        checks = [
            _limit_check(shard["request_count"], _optional_positive_int(transport_limits, "max_requests_per_batch_file"), "requests_per_batch_file"),
            byte_check,
        ]
        shard_summaries.append(
            {
                **{key: shard[key] for key in ("shard_index", "path", "sha256", "canonical_uncompressed_sha256", "compressed_bytes", "uncompressed_bytes")},
                **summary,
                "openai_batch_jsonl_bytes": batch_bytes,
                "openai_batch_jsonl_sha256": None if batch_payload is None else _sha256_bytes(batch_payload),
                "limit_checks": checks,
                "status": "fail" if any(check["status"] == "fail" for check in checks) else ("unknown" if any(check["status"].startswith("unknown") for check in checks) else "pass"),
            }
        )
    queue_check = _limit_check(
        totals["input_tokens"]["worst"],
        _optional_positive_int(model_limits, "max_queued_input_tokens"),
        "whole_pack_worst_estimated_queued_input_tokens",
    )
    batch_queue = {
        "transport_kind": transport_profile["transport_kind"],
        "completion_window": transport_profile.get("completion_window"),
        "expected_input_file_count": transport_profile.get("expected_input_file_count"),
        "status": (
            "not_selected_upstream_contract_uses_online_chat_completions"
            if transport_profile["transport_kind"] == "online_chat_completions"
            else queue_check["status"]
        ),
        "whole_pack_request_count": len(estimates),
        "whole_pack_worst_estimated_input_tokens": totals["input_tokens"]["worst"],
        "existing_transport_shard_count": len(shards),
        "max_requests_in_existing_shard": max(row["request_count"] for row in shards),
        "max_source_uncompressed_bytes_in_existing_shard": max(row["uncompressed_bytes"] for row in shards),
        "max_openai_batch_jsonl_bytes": max(
            (row["openai_batch_jsonl_bytes"] or 0 for row in shard_summaries),
            default=0,
        ) or None,
        "explicit_queue_limit_check": queue_check,
        "note": (
            "The literal upstream contract uses online Chat Completions. Batch feasibility is "
            "only decision-bearing when an explicit batch execution profile and its limits are selected."
        ),
    }
    output_limit_check = _limit_check(max_output, _optional_positive_int(model_limits, "max_output_tokens"), "configured_max_output_tokens_per_request")
    expected_file_count = transport_profile.get("expected_input_file_count")
    expected_file_count_check = {
        "metric": "planned_input_file_count_matches_transport_profile",
        "observed": len(shards),
        "limit": expected_file_count,
        "status": (
            "not_applicable_online_transport"
            if transport_profile["transport_kind"] == "online_chat_completions"
            else ("pass" if len(shards) == expected_file_count else "fail")
        ),
        "headroom": 0 if expected_file_count == len(shards) else None,
    }
    tiktoken_total = None
    if tiktoken_encoding is not None:
        tiktoken_total = {
            "encoding": tiktoken_encoding,
            "scope": "content only; not used as an exact billed-token claim",
            "input_content_tokens": sum(row["tiktoken"]["input_content_tokens"] for row in estimates),
            "output_compact_zero_tokens": sum(row["tiktoken"]["output_compact_zero_tokens"] for row in estimates),
            "output_compact_base_tokens": sum(row["tiktoken"]["output_compact_base_tokens"] for row in estimates),
        }
    model_identity = None if model_profile is None else {
        "provider": model_profile.get("provider"),
        "model_alias": model_profile.get("model_alias"),
        "model_snapshot": model_profile.get("model_snapshot"),
        "snapshot_status": "unresolved_alias" if model_profile.get("model_snapshot") is None else "explicit_snapshot",
        "protocol_compatibility": model_profile.get("protocol_compatibility"),
    }
    endpoint_identity = None if model_profile is None else model_profile.get("endpoint")
    transport_plan = [
        {
            key: row[key]
            for key in (
                "shard_index", "path", "sha256", "canonical_uncompressed_sha256",
                "compressed_bytes", "uncompressed_bytes", "request_count",
                "openai_batch_jsonl_bytes", "openai_batch_jsonl_sha256",
            )
        }
        for row in shard_summaries
    ]
    adapter_transport_plan = [
        {
            "shard_index": row["shard_index"],
            "input_sha256": row["openai_batch_jsonl_sha256"],
            "bytes": row["openai_batch_jsonl_bytes"],
            "request_count": row["request_count"],
        }
        for row in shard_summaries
    ]
    adapter_manifest_sha256 = None
    if adapter_manifest is not None:
        if transport_profile["transport_kind"] != "openai_batch":
            raise ValueError("adapter manifest is valid only for OpenAI Batch transport")
        unsigned_adapter = dict(adapter_manifest)
        adapter_manifest_sha256 = unsigned_adapter.pop("manifest_sha256", None)
        if adapter_manifest_sha256 != object_sha256(unsigned_adapter):
            raise ValueError("adapter manifest self-hash mismatch")
        pack = adapter_manifest.get("pack", {})
        if (
            pack.get("config_sha256") != config["config_sha256"]
            or pack.get("pack_sha256") != pack_index["pack_sha256"]
            or pack.get("canonical_uncompressed_sha256") != pack_index["canonical_uncompressed_sha256"]
            or pack.get("request_count") != len(requests)
            or pack.get("shard_count") != len(shards)
        ):
            raise ValueError("adapter manifest pack binding mismatch")
        if adapter_manifest.get("execution_settings_sha256") != object_sha256(settings):
            raise ValueError("adapter manifest invocation settings mismatch")
        if adapter_manifest.get("transport_profile_sha256") != object_sha256(transport_profile):
            raise ValueError("adapter manifest transport profile mismatch")
        if adapter_manifest.get("materialized_transport_plan_sha256") != object_sha256(adapter_transport_plan):
            raise ValueError("adapter materialized Batch JSONL plan mismatch")
    binding = {
        "request_pack_sha256": pack_index["pack_sha256"],
        "request_pack_canonical_uncompressed_sha256": pack_index["canonical_uncompressed_sha256"],
        "config_sha256": config["config_sha256"],
        "estimator_spec_sha256": object_sha256(ESTIMATOR_SPEC),
        "model_identity_sha256": None if model_identity is None else object_sha256(model_identity),
        "endpoint_identity_sha256": None if endpoint_identity is None else object_sha256(endpoint_identity),
        "model_profile_sha256": None if model_profile is None else object_sha256(model_profile),
        "capacity_profile_sha256": None if capacity_profile is None else object_sha256(capacity_profile),
        "execution_settings_sha256": object_sha256(settings),
        "transport_profile_sha256": object_sha256(transport_profile),
        "transport_plan_sha256": object_sha256(transport_plan),
        "adapter_manifest_sha256": adapter_manifest_sha256,
        "adapter_materialized_transport_plan_sha256": object_sha256(adapter_transport_plan),
        "pricing_profile_sha256": None if pricing_profile is None else object_sha256(pricing_profile),
        "tokenizer_binding_sha256": object_sha256({"backend": "tiktoken", "encoding": tiktoken_encoding}) if tiktoken_encoding else object_sha256({"backend": "model_agnostic"}),
        "request_estimates_canonical_sha256": object_sha256(estimates),
    }
    binding["estimate_binding_sha256"] = object_sha256(binding)
    cost = _cost_scenarios(totals, pricing_profile, transport_profile)
    batch_selected = transport_profile["transport_kind"] == "openai_batch"
    capacity_failures = (
        sum(row["status"] == "fail" for row in request_checks)
        + int(output_limit_check["status"] == "fail")
        + (sum(row["status"] == "fail" for row in shard_summaries) if batch_selected else 0)
        + (int(queue_check["status"] == "fail") if batch_selected else 0)
        + (int(expected_file_count_check["status"] == "fail") if batch_selected else 0)
    )
    unknown_capacity = (
        sum(row["status"] == "unknown" for row in request_checks)
        + int(output_limit_check["status"].startswith("unknown"))
        + (sum(row["status"] == "unknown" for row in shard_summaries) if batch_selected else 0)
        + (int(queue_check["status"].startswith("unknown")) if batch_selected else 0)
    )
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "kind": "pathopress_llm_execution_preflight",
        "offline_only": True,
        "network_calls": 0,
        "request_pack": {
            "request_count": len(requests),
            "target_prediction_count": sum(len(row["targets"]) for row in requests),
            "shard_count": len(shards),
            "compressed_bytes": sum(row["compressed_bytes"] for row in shards),
            "canonical_uncompressed_bytes": sum(row["uncompressed_bytes"] for row in shards),
            "pack_sha256": pack_index["pack_sha256"],
            "canonical_uncompressed_sha256": pack_index["canonical_uncompressed_sha256"],
        },
        "estimator": {**ESTIMATOR_SPEC, "spec_sha256": object_sha256(ESTIMATOR_SPEC)},
        "model_selection": {
            "status": "unselected" if model_profile is None else "explicit_profile_supplied",
            "identity": model_identity,
            "profile": model_profile,
        },
        "execution_settings": settings,
        "transport_profile": transport_profile,
        "capacity_profile": capacity_profile,
        "transport_plan": transport_plan,
        "tokenizer_cross_check": tiktoken_total,
        "totals": totals,
        "by_condition": by_condition,
        "by_shard": shard_summaries,
        "capacity": {
            "status": "fail" if capacity_failures else ("unknown_missing_explicit_limits" if unknown_capacity else "pass"),
            "failure_count": capacity_failures,
            "unknown_check_count": unknown_capacity,
            "request_check_status_counts": {
                status: sum(row["status"] == status for row in request_checks)
                for status in ("pass", "fail", "unknown")
            },
            "failed_request_ids": [row["request_id"] for row in request_checks if row["status"] == "fail"],
            "queue_check": queue_check,
            "batch_queue_feasibility": batch_queue,
            "expected_file_count_check": expected_file_count_check,
            "configured_output_limit_check": output_limit_check,
            "transport_scope_note": "For OpenAI Batch, byte/hash checks use the exact canonical Batch JSONL wire plan and are compared with the offline adapter materialization manifest.",
        },
        "cost": cost,
        "binding": binding,
    }
    unsigned_preflight = dict(preflight)
    preflight["preflight_sha256"] = object_sha256(unsigned_preflight)
    approval_reasons = []
    if model_profile is None:
        approval_reasons.append("explicit model profile not supplied")
    elif model_profile.get("model_snapshot") is None and not acknowledge_mutable_alias:
        approval_reasons.append(
            "mutable model alias requires an explicit preflight acknowledgement"
        )
    if capacity_profile is None:
        approval_reasons.append("explicit model/account capacity profile not supplied")
    if capacity_failures:
        approval_reasons.append(f"{capacity_failures} explicit capacity checks failed")
    if unknown_capacity:
        approval_reasons.append(f"{unknown_capacity} capacity checks lack explicit model/provider limits")
    if cost["status"] != "estimated_from_explicit_profile":
        approval_reasons.append(
            f"selected {transport_profile['pricing_rate_key']} transport cost is not ready: {cost['status']}"
        )
    if batch_selected and adapter_manifest_sha256 is None:
        approval_reasons.append("materialized OpenAI Batch adapter manifest not supplied or hash-bound")
    if budget is None:
        approval_reasons.append("explicit human-authorized currency/max_amount budget not supplied")
    elif cost["status"] == "estimated_from_explicit_profile" and (
        budget["currency"] != cost["currency"]
        or cost["scenarios"]["worst"]["total"] > budget["max_amount"]
    ):
        approval_reasons.append("worst-case selected-transport cost exceeds or mismatches the explicit budget")
    approval = {
        "schema_version": 1,
        "kind": "pathopress_llm_human_execution_approval_manifest",
        "approval_status": "not_ready" if approval_reasons else "awaiting_human_approval",
        "execution_authorized": False,
        "network_calls_performed": 0,
        "human_must_set_execution_authorized_true_after_review": True,
        "mutable_alias_acknowledged": bool(acknowledge_mutable_alias),
        "blocking_reasons": approval_reasons,
        "review_checklist": [
            "Confirm the request-pack and config hashes match the intended experiment.",
            "Confirm the fixed provider/model/model-version and tokenizer/chat-template assumptions.",
            "Confirm worst-case per-request context, output, batch-file, and queue checks pass.",
            "Confirm explicit dated prices and the worst-case spend are acceptable.",
            "Acknowledge that the planning ceiling is not provider-enforced and actual billed cost may differ or exceed it.",
            "Confirm credentials and raw provider receipts remain outside version control.",
        ],
        "execution_evidence_completeness": {
            "expected_response_count": len(requests),
            "single_fixed_provider_model_identity_required": True,
            "model_alias": None if model_profile is None else model_profile.get("model_alias"),
            "model_snapshot_expected": None if model_profile is None else model_profile.get("model_snapshot"),
            "model_version_evidence_required_responses": len(requests),
            "settings_evidence_required_responses": len(requests),
            "required_settings_sha256": object_sha256(settings),
            "required_transport_profile_sha256": object_sha256(transport_profile),
            "provider_receipt_required_responses": len(requests),
            "note": (
                "The execution importer must preserve one model-version, identical settings, and a "
                "provider receipt for every response before this self-hashed record can be human-authorized."
            ),
        },
        "approved_identity": model_identity,
        "execution_settings": settings,
        "transport_profile": transport_profile,
        "capacity_profile": capacity_profile,
        "cost_summary": cost,
        "authorized_planning_cost_ceiling": budget,
        "cost_estimate_uncertainty_acknowledged": bool(acknowledge_cost_estimate_uncertainty),
        "planning_ceiling_note": "This is an authorization decision over estimated scenarios, not a provider-enforced hard billing cap; actual billed cost may differ or exceed it.",
        "capacity_status": preflight["capacity"]["status"],
        "binding": binding,
        "preflight_sha256": preflight["preflight_sha256"],
    }
    approval["approval_manifest_sha256"] = object_sha256(approval)
    return preflight, estimates, approval


def write_request_estimates(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    """Write deterministic gzip JSONL details and return its SHA-256."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(payload)
    return _sha256_bytes(path.read_bytes())


def build_approved_execution_contract(
    preflight: dict[str, Any], approval: dict[str, Any]
) -> dict[str, Any]:
    """Validate a human-approved manifest and derive the headline gate contract."""

    unsigned_preflight = dict(preflight)
    supplied_preflight_hash = unsigned_preflight.pop("preflight_sha256", None)
    if supplied_preflight_hash != object_sha256(unsigned_preflight):
        raise ValueError("execution preflight self-hash mismatch")
    unsigned_approval = dict(approval)
    supplied_approval_hash = unsigned_approval.pop("approval_manifest_sha256", None)
    if supplied_approval_hash != object_sha256(unsigned_approval):
        raise ValueError("execution approval manifest self-hash mismatch")
    if approval.get("preflight_sha256") != supplied_preflight_hash:
        raise ValueError("execution approval does not bind the selected preflight")
    if approval.get("approval_status") != "approved" or approval.get("execution_authorized") is not True:
        raise ValueError("execution approval manifest has not been approved by a human")
    if approval.get("blocking_reasons"):
        raise ValueError("execution approval manifest still contains blocking reasons")
    if approval.get("binding") != preflight.get("binding"):
        raise ValueError("execution approval binding does not exactly match the preflight")
    if approval.get("execution_settings") != preflight.get("execution_settings"):
        raise ValueError("execution approval invocation settings do not exactly match the preflight")
    if approval.get("transport_profile") != preflight.get("transport_profile"):
        raise ValueError("execution approval transport profile does not exactly match the preflight")
    if approval.get("capacity_profile") != preflight.get("capacity_profile"):
        raise ValueError("execution approval capacity profile does not exactly match the preflight")
    if approval.get("cost_summary") != preflight.get("cost"):
        raise ValueError("execution approval cost summary does not exactly match the preflight")
    if preflight.get("capacity", {}).get("status") != "pass":
        raise ValueError("execution preflight capacity status is not pass")
    if preflight.get("cost", {}).get("status") != "estimated_from_explicit_profile":
        raise ValueError("execution preflight has no approved explicit transport-rate cost")
    if preflight.get("binding", {}).get("pricing_profile_sha256") is None:
        raise ValueError("execution preflight has no pricing-profile hash")
    if preflight.get("binding", {}).get("capacity_profile_sha256") is None:
        raise ValueError("execution preflight has no explicit capacity-profile hash")
    budget = approval.get("authorized_planning_cost_ceiling")
    cost = preflight["cost"]
    if not isinstance(budget, dict):
        raise ValueError("execution approval has no explicit authorized budget")
    if approval.get("cost_estimate_uncertainty_acknowledged") is not True:
        raise ValueError("estimated-cost uncertainty was not explicitly acknowledged")
    if budget.get("currency") != cost.get("currency"):
        raise ValueError("execution approval budget currency does not match estimated cost")
    maximum = budget.get("max_amount")
    if (
        not isinstance(maximum, (int, float)) or isinstance(maximum, bool)
        or not math.isfinite(float(maximum)) or maximum < 0
    ):
        raise ValueError("execution approval budget max_amount is invalid")
    if cost["scenarios"]["worst"]["total"] > maximum:
        raise ValueError("worst-case estimated cost exceeds the human-authorized budget")
    identity = preflight.get("model_selection", {}).get("identity")
    if not isinstance(identity, dict):
        raise ValueError("execution preflight has no explicit model identity")
    evidence = approval.get("execution_evidence_completeness")
    if not isinstance(evidence, dict):
        raise ValueError("execution approval has no evidence-completeness contract")
    settings_hash = preflight.get("binding", {}).get("execution_settings_sha256")
    if evidence.get("required_settings_sha256") != settings_hash:
        raise ValueError("execution approval settings hash does not match the preflight")
    transport = preflight.get("transport_profile")
    if not isinstance(transport, dict):
        raise ValueError("execution preflight has no transport profile")
    validate_transport_profile(transport)
    transport_hash = preflight.get("binding", {}).get("transport_profile_sha256")
    if evidence.get("required_transport_profile_sha256") != transport_hash:
        raise ValueError("execution approval transport hash does not match the preflight")
    adapter_manifest_hash = preflight.get("binding", {}).get("adapter_manifest_sha256")
    if transport.get("transport_kind") == "openai_batch" and not adapter_manifest_hash:
        raise ValueError("approved Batch execution has no bound adapter materialization manifest")
    expected_count = preflight.get("request_pack", {}).get("request_count")
    if evidence.get("expected_response_count") != expected_count:
        raise ValueError("execution approval response count does not match the preflight")
    for key in (
        "model_version_evidence_required_responses",
        "settings_evidence_required_responses",
        "provider_receipt_required_responses",
    ):
        if evidence.get(key) != expected_count:
            raise ValueError(f"execution approval {key} is incomplete")
    alias = identity.get("model_alias")
    snapshot = identity.get("model_snapshot")
    if snapshot is None and approval.get("mutable_alias_acknowledged") is not True:
        raise ValueError("mutable model alias was not explicitly acknowledged")
    contract = {
        "schema_version": 1,
        "provider": identity.get("provider"),
        "model_alias": alias,
        "model_snapshot": snapshot,
        "response_model": snapshot or alias,
        "required_model_version": snapshot,
        "require_resolved_version_distinct_from_alias": snapshot is None,
        "required_settings_sha256": settings_hash,
        "transport_kind": transport.get("transport_kind"),
        "completion_window": transport.get("completion_window"),
        "expected_input_file_count": transport.get("expected_input_file_count"),
        "transport_profile_sha256": transport_hash,
        "transport_plan_sha256": preflight.get("binding", {}).get("transport_plan_sha256"),
        "adapter_manifest_sha256": adapter_manifest_hash,
        "pricing_profile_sha256": preflight.get("binding", {}).get("pricing_profile_sha256"),
        "capacity_profile_sha256": preflight.get("binding", {}).get("capacity_profile_sha256"),
        "authorized_planning_cost_ceiling": budget,
        "cost_estimate_uncertainty_acknowledged": True,
        "planning_ceiling_note": "Not provider-enforced; actual billed cost may differ or exceed the estimated planning ceiling.",
        "transport_protocol_compatibility": transport.get("protocol_compatibility"),
        "expected_response_count": expected_count,
        "config_sha256": preflight.get("binding", {}).get("config_sha256"),
        "request_pack_sha256": preflight.get("binding", {}).get("request_pack_sha256"),
        "endpoint_identity_sha256": preflight.get("binding", {}).get("endpoint_identity_sha256"),
        "approval_manifest_sha256": supplied_approval_hash,
        "preflight_sha256": supplied_preflight_hash,
    }
    contract["execution_contract_sha256"] = object_sha256(contract)
    return contract


def approve_execution_manifest(
    preflight: dict[str, Any],
    approval: dict[str, Any],
    *,
    human_review_complete: bool,
    acknowledge_estimated_cost_uncertainty: bool,
) -> dict[str, Any]:
    """Perform the offline human-approval state transition; never calls a provider."""

    if not human_review_complete:
        raise ValueError("approval requires explicit confirmation that human review is complete")
    if not acknowledge_estimated_cost_uncertainty:
        raise ValueError("approval requires acknowledging estimated-cost uncertainty")
    unsigned = dict(approval)
    supplied = unsigned.pop("approval_manifest_sha256", None)
    if supplied != object_sha256(unsigned):
        raise ValueError("execution approval manifest self-hash mismatch")
    if approval.get("approval_status") != "awaiting_human_approval":
        raise ValueError("approval manifest is not technically ready for human approval")
    if approval.get("blocking_reasons"):
        raise ValueError("approval manifest still has technical blocking reasons")
    approved = json.loads(json.dumps(approval))
    approved.pop("approval_manifest_sha256", None)
    approved["approval_status"] = "approved"
    approved["execution_authorized"] = True
    approved["human_review_complete"] = True
    approved["cost_estimate_uncertainty_acknowledged"] = True
    approved["approval_manifest_sha256"] = object_sha256(approved)
    build_approved_execution_contract(preflight, approved)
    return approved
