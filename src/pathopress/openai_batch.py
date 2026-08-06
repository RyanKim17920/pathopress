"""Fail-closed OpenAI Batch adapter for the frozen PathoPress LLM pack.

Local preparation is the default.  Network operations are exposed separately
and require explicit gates in the CLI; this module never discovers or stores
credentials itself.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from .llm_baseline import canonical_json, object_sha256, validate_config, validate_request


EXPECTED_CONFIG_SHA256 = "ac7a7cb84dfe961fb50996ea6ae641b1829423aa8a02881b707688537d5e3e4d"
EXPECTED_PACK_SHA256 = "186bac8c081d63e08f1d713dd6c5a4083c6bd2888bff592b1c16f7b5fed236c5"
EXPECTED_UNCOMPRESSED_SHA256 = "e431cfc0fc0a14610fd26c0818ae1cadb6849c2c339db2996d01a7988488c574"
EXPECTED_REQUEST_COUNT = 1990
EXPECTED_SHARD_COUNT = 20
EXPECTED_REQUEST_IDS_SHA256 = "298225fa860dbdf97fd74a9d9edfcb0e04463f3502484eb1696cbd78d7018318"
EXPECTED_MESSAGES_SHA256 = "668a7f677508c51e418ee3f37479ac2efe1388495c88bc6e815fd1c627596d51"
BUILTIN_SNAPSHOT = "gpt-5.5-2026-04-23"
ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"
API_ROOT = "https://api.openai.com/v1"
OFFICIAL_DOCS = {
    "batch": "https://platform.openai.com/docs/api-reference/batch",
    "files": "https://platform.openai.com/docs/api-reference/files",
    "chat": "https://platform.openai.com/docs/api-reference/chat/create",
}

_SNAPSHOT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*-(\d{4})-(\d{2})-(\d{2})$")
_REMOTE_ID = re.compile(r"^(?:file|batch)[-_][A-Za-z0-9_-]{3,128}$")
EXECUTION_SETTINGS = {
    "schema_version": 1,
    "profile_kind": "literal_upstream_contract",
    "temperature": 0.0,
    "max_output_tokens_per_request": 16384,
    "provider_parameter_name": "max_tokens",
    "response_format": "json_object_requested_in_prompt",
    "endpoint_api": "chat_completions",
    "source_repository_commit": "0a684b63ee0e4a401cb907a3827a82ea997d74c4",
}
BATCH_TRANSPORT_PROFILE = {
    "schema_version": 1,
    "transport_kind": "openai_batch",
    "protocol_compatibility": "batch_transport_adaptation_not_upstream_exact",
    "pricing_rate_key": "batch",
    "completion_window": "24h",
    "expected_input_file_count": 20,
    "limits": {
        "max_requests_per_batch_file": 50000,
        "max_batch_file_bytes": 200000000,
    },
    "source": {
        "url": "https://platform.openai.com/docs/api-reference/batch",
        "title": "OpenAI Batch API reference",
        "retrieved_at": "2026-08-06",
        "limits_note": "Request-count and file-byte limits are transport limits; enqueued-token capacity is model-specific and intentionally unset.",
    },
}


@dataclass(frozen=True)
class Profile:
    name: str
    model: str
    mutable_alias: bool
    settings: dict[str, Any]
    adaptations: tuple[str, ...]


def profile(name: str, model: str | None) -> Profile:
    if name == "upstream_exact":
        if model not in (None, "gpt-5.5"):
            raise ValueError("upstream_exact fixes the literal upstream model alias gpt-5.5")
        return Profile(
            name=name,
            model="gpt-5.5",
            mutable_alias=True,
            settings={"temperature": 0.0, "max_tokens": 16384},
            adaptations=(),
        )
    if name not in {"chat_snapshot", "chat_custom_snapshot"}:
        raise ValueError(f"unknown OpenAI Batch profile: {name}")
    if not model:
        raise ValueError("chat_snapshot requires --model with an immutable dated snapshot")
    validate_snapshot_model(model)
    if not model.startswith("gpt-5.5-"):
        raise ValueError("controlled snapshot substitution must remain in the gpt-5.5 family")
    if name == "chat_snapshot" and model != BUILTIN_SNAPSHOT:
        raise ValueError(f"chat_snapshot is pinned to {BUILTIN_SNAPSHOT}; use chat_custom_snapshot explicitly")
    return Profile(
        name=name,
        model=model,
        mutable_alias=False,
        settings={"temperature": 0.0, "max_tokens": 16384},
        adaptations=("mutable gpt-5.5 alias replaced by an explicit immutable snapshot",),
    )


def validate_snapshot_model(model: str) -> None:
    match = _SNAPSHOT.fullmatch(model)
    if not match:
        raise ValueError(
            "model must be an explicit immutable snapshot ending YYYY-MM-DD; aliases are rejected"
        )
    try:
        date(*(int(part) for part in match.groups()))
    except ValueError as error:
        raise ValueError("model snapshot has an invalid calendar date") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_lines(payload: bytes, source: str) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL in {source}:{number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row in {source}:{number}")
        rows.append(row)
    return rows


def _atomic_write(path: Path, payload: bytes, *, private: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if private:
        try:
            path.parent.chmod(0o700)
        except OSError:
            pass
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if private:
            os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def load_frozen_pack(pack_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads((pack_dir / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    if config["config_sha256"] != EXPECTED_CONFIG_SHA256:
        raise ValueError("LLM configuration is not the frozen 1,990-request release")
    index = json.loads((pack_dir / "requests.jsonl").read_text(encoding="utf-8"))
    if index.get("pack_sha256") != EXPECTED_PACK_SHA256:
        raise ValueError("request index pack hash does not match the frozen release")
    if index.get("canonical_uncompressed_sha256") != EXPECTED_UNCOMPRESSED_SHA256:
        raise ValueError("request index canonical hash does not match the frozen release")
    if index.get("request_count") != EXPECTED_REQUEST_COUNT or index.get("shard_count") != EXPECTED_SHARD_COUNT:
        raise ValueError("request index count does not match the frozen release")

    pack_digest = hashlib.sha256()
    uncompressed_digest = hashlib.sha256()
    shards: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for item in index.get("shards", []):
        name = item.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("unsafe request shard path")
        path = pack_dir / "requests" / name
        compressed = path.read_bytes()
        if len(compressed) != item.get("bytes") or sha256_bytes(compressed) != item.get("sha256"):
            raise ValueError(f"request shard integrity mismatch: {name}")
        payload = gzip.decompress(compressed)
        rows = _json_lines(payload, name)
        for row in rows:
            validate_request(row, config)
        shards.append({"name": name, "sha256": item["sha256"], "payload": payload, "requests": rows})
        requests.extend(rows)
        pack_digest.update(name.encode())
        pack_digest.update(bytes.fromhex(item["sha256"]))
        uncompressed_digest.update(payload)
    if len(shards) != EXPECTED_SHARD_COUNT:
        raise ValueError("request shard set is incomplete")
    if pack_digest.hexdigest() != EXPECTED_PACK_SHA256:
        raise ValueError("computed request pack hash mismatch")
    if uncompressed_digest.hexdigest() != EXPECTED_UNCOMPRESSED_SHA256:
        raise ValueError("computed uncompressed request hash mismatch")
    ids = [row["request_id"] for row in requests]
    if len(requests) != EXPECTED_REQUEST_COUNT or len(ids) != len(set(ids)):
        raise ValueError("frozen request IDs are incomplete or duplicated")
    return config, shards


def batch_line(request: dict[str, Any], selected: Profile) -> dict[str, Any]:
    body = {"model": selected.model, "messages": request["messages"], **selected.settings}
    return {"custom_id": request["request_id"], "method": "POST", "url": ENDPOINT, "body": body}


def materialize(
    pack_dir: Path, output_dir: Path, selected: Profile
) -> dict[str, Any]:
    config, source_shards = load_frozen_pack(pack_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts = []
    all_ids: list[str] = []
    for number, source in enumerate(source_shards):
        lines = [batch_line(request, selected) for request in source["requests"]]
        for line, request in zip(lines, source["requests"]):
            if line["body"]["messages"] != request["messages"]:
                raise AssertionError("prompt mutation detected")
            all_ids.append(line["custom_id"])
        payload = "".join(canonical_json(line) + "\n" for line in lines).encode("utf-8")
        path = output_dir / f"batch-{number:04d}.jsonl"
        _atomic_write(path, payload)
        receipts.append(
            {
                "shard_index": number,
                "source_shard": source["name"],
                "source_shard_sha256": source["sha256"],
                "input_path": path.name,
                "input_sha256": sha256_bytes(payload),
                "bytes": len(payload),
                "request_count": len(lines),
                "first_custom_id": lines[0]["custom_id"],
                "last_custom_id": lines[-1]["custom_id"],
            }
        )
    stale = set(output_dir.glob("batch-*.jsonl")) - {output_dir / row["input_path"] for row in receipts}
    if stale:
        raise ValueError("unexpected stale batch input files; use a clean output directory")
    settings_receipt = {
        "profile": selected.name,
        "endpoint": ENDPOINT,
        "completion_window": COMPLETION_WINDOW,
        "model_sha256": object_sha256(selected.model),
        "api_body_settings_sha256": object_sha256(selected.settings),
        "execution_settings_sha256": object_sha256(EXECUTION_SETTINGS),
    }
    transport_plan = [
        {key: row[key] for key in ("shard_index", "input_sha256", "bytes", "request_count")}
        for row in receipts
    ]
    manifest = {
        "schema_version": 1,
        "kind": "pathopress_openai_batch_preflight",
        "status": "materialized_no_api_call",
        "api_calls_made": 0,
        "eligible_after_all_gates": True,
        "additional_acknowledgement_required": (
            "mutable_alias" if selected.mutable_alias else (
                "custom_snapshot" if selected.name == "chat_custom_snapshot" else None
            )
        ),
        "profile": selected.name,
        "exact_upstream_request_body_contract": selected.name == "upstream_exact",
        "adaptations": list(selected.adaptations),
        "transport_adaptations": [
            "asynchronous 24-hour OpenAI Batch transport replaces upstream online Chat Completions"
        ],
        "pack": {
            "config_sha256": config["config_sha256"],
            "pack_sha256": EXPECTED_PACK_SHA256,
            "canonical_uncompressed_sha256": EXPECTED_UNCOMPRESSED_SHA256,
            "request_count": len(all_ids),
            "request_ids_sha256": object_sha256(all_ids),
            "shard_count": len(receipts),
        },
        "settings_receipt": settings_receipt,
        "settings_receipt_sha256": object_sha256(settings_receipt),
        "execution_settings_sha256": object_sha256(EXECUTION_SETTINGS),
        "transport_profile": BATCH_TRANSPORT_PROFILE,
        "transport_profile_sha256": object_sha256(BATCH_TRANSPORT_PROFILE),
        "materialized_transport_plan_sha256": object_sha256(transport_plan),
        "shards": receipts,
        "official_docs": OFFICIAL_DOCS,
        "submission_gates": [
            "--submit", "--authorize-paid-run", "OPENAI_API_KEY",
            "--acknowledge-estimated-cost-uncertainty",
        ],
        "credential_persisted": False,
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    atomic_json(output_dir / "preflight.json", manifest)
    return manifest


def load_manifest(path: Path, selected: Profile) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    supplied = value.get("manifest_sha256")
    if supplied != object_sha256({key: item for key, item in value.items() if key != "manifest_sha256"}):
        raise ValueError("preflight manifest hash mismatch")
    pack = value.get("pack", {})
    if (
        pack.get("config_sha256") != EXPECTED_CONFIG_SHA256
        or pack.get("pack_sha256") != EXPECTED_PACK_SHA256
        or pack.get("canonical_uncompressed_sha256") != EXPECTED_UNCOMPRESSED_SHA256
        or pack.get("request_count") != EXPECTED_REQUEST_COUNT
        or pack.get("shard_count") != EXPECTED_SHARD_COUNT
    ):
        raise ValueError("preflight manifest is not bound to the frozen pack")
    expected = {
        "profile": selected.name,
        "endpoint": ENDPOINT,
        "completion_window": COMPLETION_WINDOW,
        "model_sha256": object_sha256(selected.model),
        "api_body_settings_sha256": object_sha256(selected.settings),
        "execution_settings_sha256": object_sha256(EXECUTION_SETTINGS),
    }
    if value.get("settings_receipt") != expected or value.get("settings_receipt_sha256") != object_sha256(expected):
        raise ValueError("preflight model/settings identity mismatch")
    if (
        value.get("transport_profile") != BATCH_TRANSPORT_PROFILE
        or value.get("transport_profile_sha256") != object_sha256(BATCH_TRANSPORT_PROFILE)
    ):
        raise ValueError("preflight transport profile is not the checked OpenAI Batch contract")
    receipts = value.get("shards")
    if not isinstance(receipts, list) or len(receipts) != EXPECTED_SHARD_COUNT:
        raise ValueError("preflight must contain all 20 shard receipts")
    if [row.get("shard_index") for row in receipts] != list(range(EXPECTED_SHARD_COUNT)):
        raise ValueError("preflight shard receipts must be sequential and unique")
    if sum(row.get("request_count", -EXPECTED_REQUEST_COUNT) for row in receipts) != EXPECTED_REQUEST_COUNT:
        raise ValueError("preflight shard request counts do not sum to 1,990")
    all_ids: list[str] = []
    all_messages: list[list[dict[str, str]]] = []
    for shard in receipts:
        input_path = path.parent / shard["input_path"]
        if input_path.name != f"batch-{shard['shard_index']:04d}.jsonl":
            raise ValueError("preflight batch filename does not match its shard index")
        if sha256_file(input_path) != shard["input_sha256"] or input_path.stat().st_size != shard["bytes"]:
            raise ValueError(f"materialized batch input drifted: {input_path.name}")
        rows = _json_lines(input_path.read_bytes(), input_path.name)
        if len(rows) != shard["request_count"]:
            raise ValueError("materialized batch line count differs from receipt")
        for row in rows:
            if set(row) != {"custom_id", "method", "url", "body"}:
                raise ValueError("batch input has unexpected fields")
            if row["method"] != "POST" or row["url"] != ENDPOINT:
                raise ValueError("batch input endpoint contract drifted")
            body = row["body"]
            if body.get("model") != selected.model or {key: body.get(key) for key in selected.settings} != selected.settings:
                raise ValueError("batch input model/settings drifted")
            if set(body) != {"model", "messages", *selected.settings}:
                raise ValueError("batch request body has unexpected fields")
            all_ids.append(row["custom_id"])
            all_messages.append(body["messages"])
    if len(all_ids) != EXPECTED_REQUEST_COUNT or len(set(all_ids)) != EXPECTED_REQUEST_COUNT:
        raise ValueError("materialized custom IDs are incomplete or duplicated")
    if object_sha256(all_ids) != EXPECTED_REQUEST_IDS_SHA256 or value["pack"].get("request_ids_sha256") != EXPECTED_REQUEST_IDS_SHA256:
        raise ValueError("materialized custom ID sequence drifted from frozen pack")
    if object_sha256(all_messages) != EXPECTED_MESSAGES_SHA256:
        raise ValueError("materialized prompts drifted from the frozen pack")
    transport_plan = [
        {key: row[key] for key in ("shard_index", "input_sha256", "bytes", "request_count")}
        for row in receipts
    ]
    if value.get("materialized_transport_plan_sha256") != object_sha256(transport_plan):
        raise ValueError("materialized transport plan hash mismatch")
    return value


def paid_gate(
    selected: Profile, submit: bool, authorize: bool, key: str | None, *,
    acknowledge_mutable_alias: bool = False, acknowledge_custom_snapshot: bool = False,
    acknowledge_estimated_cost_uncertainty: bool = False,
) -> str:
    if selected.mutable_alias and not acknowledge_mutable_alias:
        raise PermissionError("literal upstream run additionally requires --acknowledge-mutable-alias")
    if selected.name == "chat_custom_snapshot" and not acknowledge_custom_snapshot:
        raise PermissionError("custom snapshot run additionally requires --acknowledge-custom-snapshot")
    if not acknowledge_estimated_cost_uncertainty:
        raise PermissionError(
            "paid run requires --acknowledge-estimated-cost-uncertainty; the planning ceiling is not a provider billing cap"
        )
    if not submit or not authorize or not key:
        raise PermissionError("paid write requires --submit, --authorize-paid-run, and OPENAI_API_KEY")
    if "\n" in key or "\r" in key:
        raise ValueError("OPENAI_API_KEY contains forbidden newline characters")
    return key


def approved_contract(
    preflight_path: Path, approval_path: Path, selected: Profile
) -> dict[str, Any]:
    """Validate the independent capacity/cost/human gate before any API write."""

    from .llm_preflight import build_approved_execution_contract

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    contract = build_approved_execution_contract(preflight, approval)
    if preflight.get("cost", {}).get("status") != "estimated_from_explicit_profile":
        raise ValueError("execution preflight cost is not bound to an explicit pricing profile")
    if preflight.get("cost", {}).get("pricing_rate_key") != "batch":
        raise ValueError("execution preflight did not select Batch pricing")
    required_transport = {
        "transport_kind": "openai_batch",
        "completion_window": COMPLETION_WINDOW,
        "expected_input_file_count": EXPECTED_SHARD_COUNT,
        "transport_profile_sha256": object_sha256(BATCH_TRANSPORT_PROFILE),
    }
    for key, expected in required_transport.items():
        if contract.get(key) != expected:
            raise ValueError(f"approved execution contract {key} does not match OpenAI Batch")
    capacity = preflight.get("capacity", {})
    for key in ("expected_file_count_check",):
        if capacity.get(key, {}).get("status") != "pass":
            raise ValueError(f"approved Batch capacity {key} is not pass")
    if "batch_file_count_check" in capacity and capacity["batch_file_count_check"].get("status") != "pass":
        raise ValueError("approved Batch file-count capacity check is not pass")
    if capacity.get("batch_queue_feasibility", {}).get("status") != "pass":
        raise ValueError("approved Batch queue feasibility is not pass")
    adapter_hash = preflight.get("binding", {}).get("adapter_manifest_sha256")
    if not isinstance(adapter_hash, str) or len(adapter_hash) != 64:
        raise ValueError("execution preflight does not bind the materialized adapter manifest")
    contract["adapter_manifest_sha256"] = adapter_hash
    contract["execution_contract_sha256"] = object_sha256(
        {key: value for key, value in contract.items() if key != "execution_contract_sha256"}
    )
    required = {
        "provider": "OpenAI",
        "response_model": selected.model,
        "required_settings_sha256": object_sha256(EXECUTION_SETTINGS),
        "expected_response_count": EXPECTED_REQUEST_COUNT,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "request_pack_sha256": EXPECTED_PACK_SHA256,
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise ValueError(f"approved execution contract {key} does not match selected run")
    if selected.mutable_alias:
        if contract.get("model_alias") != "gpt-5.5" or contract.get("model_snapshot") is not None:
            raise ValueError("literal profile requires an approved mutable-alias contract")
    elif contract.get("model_snapshot") != selected.model:
        raise ValueError("snapshot profile requires approval for the exact selected snapshot")
    return contract


def online_read_gate(online: bool, key: str | None) -> str:
    if not online or not key:
        raise PermissionError("online read requires --online and OPENAI_API_KEY")
    if "\n" in key or "\r" in key:
        raise ValueError("OPENAI_API_KEY contains forbidden newline characters")
    return key


class OpenAIHTTP:
    """Small fixed-origin client; write requests are deliberately not retried."""

    def __init__(self, key: str):
        self._key = key

    def _request(self, method: str, path: str, *, body: bytes | None = None, content_type: str | None = None, retries: int = 0) -> tuple[bytes, str | None]:
        if not path.startswith("/") or "//" in path:
            raise ValueError("unsafe API path")
        headers = {"Authorization": f"Bearer {self._key}", "User-Agent": "pathopress-openai-batch/1"}
        if content_type:
            headers["Content-Type"] = content_type
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(API_ROOT + path, data=body, headers=headers, method=method),
                    timeout=60,
                ) as response:
                    return response.read(), response.headers.get("x-request-id")
            except urllib.error.HTTPError as error:
                request_id = error.headers.get("x-request-id") if error.headers else None
                if attempt < retries and error.code in {408, 409, 429, 500, 502, 503, 504}:
                    time.sleep(min(2**attempt, 8))
                    continue
                try:
                    payload = json.loads(error.read())
                    code = payload.get("error", {}).get("code") or payload.get("error", {}).get("type")
                except Exception:
                    code = None
                raise RuntimeError(f"OpenAI API failed: status={error.code} request_id={request_id} code={code}") from None
        raise AssertionError("unreachable")

    def upload(self, path: Path) -> tuple[dict[str, Any], str | None]:
        boundary = "pathopress-" + uuid.uuid4().hex
        payload = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            "Content-Type: application/jsonl\r\n\r\n"
        ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        raw, request_id = self._request("POST", "/files", body=payload, content_type=f"multipart/form-data; boundary={boundary}")
        return json.loads(raw), request_id

    def create_batch(self, file_id: str, metadata: dict[str, str]) -> tuple[dict[str, Any], str | None]:
        validate_remote_id(file_id, "file")
        body = json.dumps({"input_file_id": file_id, "endpoint": ENDPOINT, "completion_window": COMPLETION_WINDOW, "metadata": metadata}).encode()
        raw, request_id = self._request("POST", "/batches", body=body, content_type="application/json")
        return json.loads(raw), request_id

    def batch(self, batch_id: str) -> tuple[dict[str, Any], str | None]:
        validate_remote_id(batch_id, "batch")
        raw, request_id = self._request("GET", f"/batches/{batch_id}", retries=4)
        return json.loads(raw), request_id

    def cancel(self, batch_id: str) -> tuple[dict[str, Any], str | None]:
        validate_remote_id(batch_id, "batch")
        raw, request_id = self._request("POST", f"/batches/{batch_id}/cancel", body=b"", content_type="application/json")
        return json.loads(raw), request_id

    def file_content(self, file_id: str) -> tuple[bytes, str | None]:
        validate_remote_id(file_id, "file")
        return self._request("GET", f"/files/{file_id}/content", retries=4)


def validate_remote_id(value: str, kind: str) -> None:
    if not isinstance(value, str) or not _REMOTE_ID.fullmatch(value) or not value.startswith(kind):
        raise ValueError(f"invalid remote {kind} ID")


def _new_state(manifest: dict[str, Any], execution_contract: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "pathopress_openai_batch_runtime_state",
        "pack": manifest["pack"],
        "settings_receipt_sha256": manifest["settings_receipt_sha256"],
        "execution_contract": None if execution_contract is None else {
            key: execution_contract[key]
            for key in ("execution_contract_sha256", "approval_manifest_sha256", "preflight_sha256", "adapter_manifest_sha256", "transport_profile_sha256", "capacity_profile_sha256")
        },
        "status": "submitting",
        "shards": [
            {
                "shard_index": row["shard_index"],
                "input_sha256": row["input_sha256"],
                "request_count": row["request_count"],
                "input_file_id": None,
                "batch_id": None,
                "batch_status": "not_submitted",
                "submission_pending": False,
                "output_file_id": None,
                "error_file_id": None,
                "request_counts": None,
                "usage": None,
                "api_request_ids": [],
            }
            for row in manifest["shards"]
        ],
        "credential_persisted": False,
    }


def _state(path: Path, manifest: dict[str, Any], execution_contract: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return _new_state(manifest, execution_contract)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("pack") != manifest["pack"] or state.get("settings_receipt_sha256") != manifest["settings_receipt_sha256"]:
        raise ValueError("runtime state belongs to a different pack/model/settings identity")
    if len(state.get("shards", [])) != EXPECTED_SHARD_COUNT:
        raise ValueError("runtime state shard set is incomplete")
    if execution_contract is not None:
        expected_contract = {
            key: execution_contract[key]
            for key in ("execution_contract_sha256", "approval_manifest_sha256", "preflight_sha256", "adapter_manifest_sha256", "transport_profile_sha256", "capacity_profile_sha256")
        }
        if state.get("execution_contract") != expected_contract:
            raise ValueError("runtime state belongs to a different approved execution contract")
    return state


def submit(
    manifest_path: Path, state_path: Path, selected: Profile, api: OpenAIHTTP,
    execution_contract: dict[str, Any],
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, selected)
    if manifest.get("manifest_sha256") != execution_contract["adapter_manifest_sha256"]:
        raise ValueError("approved execution contract does not bind this materialized adapter manifest")
    limits = BATCH_TRANSPORT_PROFILE["limits"]
    for receipt in manifest["shards"]:
        if receipt["bytes"] > limits["max_batch_file_bytes"] or receipt["request_count"] > limits["max_requests_per_batch_file"]:
            raise ValueError("materialized Batch shard exceeds approved transport limits")
    state = _state(state_path, manifest, execution_contract)
    for receipt, shard in zip(manifest["shards"], state["shards"]):
        if shard["batch_id"]:
            continue
        input_path = manifest_path.parent / receipt["input_path"]
        if not shard["input_file_id"]:
            uploaded, request_id = api.upload(input_path)
            if uploaded.get("purpose") != "batch" or not uploaded.get("id"):
                raise RuntimeError("OpenAI file upload returned an invalid file object")
            validate_remote_id(uploaded["id"], "file")
            shard["input_file_id"] = uploaded["id"]
            if request_id:
                shard["api_request_ids"].append(request_id)
            atomic_json(state_path, state)
        if shard.get("submission_pending"):
            raise RuntimeError(
                f"shard {shard['shard_index']} has an ambiguous prior create; reconcile it and use record-batch"
            )
        shard["submission_pending"] = True
        shard["batch_status"] = "submission_pending"
        atomic_json(state_path, state)
        created, request_id = api.create_batch(
            shard["input_file_id"],
            {"pathopress_pack": EXPECTED_UNCOMPRESSED_SHA256[:32], "pathopress_shard": f"{shard['shard_index']:04d}"},
        )
        if (
            created.get("endpoint") != ENDPOINT
            or created.get("completion_window") != COMPLETION_WINDOW
            or created.get("input_file_id") != shard["input_file_id"]
            or (created.get("metadata") is not None and created.get("metadata") != {"pathopress_pack": EXPECTED_UNCOMPRESSED_SHA256[:32], "pathopress_shard": f"{shard['shard_index']:04d}"})
            or not created.get("id")
        ):
            raise RuntimeError("OpenAI batch creation returned an invalid batch object")
        validate_remote_id(created["id"], "batch")
        shard["batch_id"] = created["id"]
        shard["submission_pending"] = False
        shard["batch_status"] = created.get("status")
        if request_id:
            shard["api_request_ids"].append(request_id)
        atomic_json(state_path, state)
    state["status"] = "submitted"
    atomic_json(state_path, state)
    return state


def record_batch(
    manifest_path: Path, state_path: Path, selected: Profile, shard_index: int, batch_id: str
) -> dict[str, Any]:
    """Record a dashboard-reconciled ID after an ambiguous create response."""

    validate_remote_id(batch_id, "batch")
    manifest = load_manifest(manifest_path, selected)
    state = _state(state_path, manifest)
    if shard_index < 0 or shard_index >= len(state["shards"]):
        raise ValueError("shard index out of range")
    shard = state["shards"][shard_index]
    if shard.get("batch_id") or not shard.get("submission_pending"):
        raise ValueError("record-batch is allowed only for an unresolved submission_pending shard")
    if any(row.get("batch_id") == batch_id for row in state["shards"]):
        raise ValueError("batch ID is already assigned to another shard")
    shard["batch_id"] = batch_id
    shard["submission_pending"] = False
    shard["batch_status"] = "reconciled_unpolled"
    atomic_json(state_path, state)
    return state


def refresh(manifest_path: Path, state_path: Path, selected: Profile, api: OpenAIHTTP) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, selected)
    state = _state(state_path, manifest)
    terminal = {"completed", "failed", "expired", "cancelled"}
    for shard in state["shards"]:
        if not shard["batch_id"]:
            continue
        value, request_id = api.batch(shard["batch_id"])
        expected_metadata = {"pathopress_pack": EXPECTED_UNCOMPRESSED_SHA256[:32], "pathopress_shard": f"{shard['shard_index']:04d}"}
        if (
            value.get("id") != shard["batch_id"]
            or value.get("input_file_id") != shard["input_file_id"]
            or value.get("endpoint") != ENDPOINT
            or value.get("completion_window") != COMPLETION_WINDOW
            or (value.get("metadata") is not None and value.get("metadata") != expected_metadata)
        ):
            raise ValueError("retrieved batch identity does not match submitted shard")
        shard.update(
            batch_status=value.get("status"),
            output_file_id=value.get("output_file_id"),
            error_file_id=value.get("error_file_id"),
            request_counts=value.get("request_counts"),
            usage=value.get("usage"),
        )
        if request_id:
            shard["api_request_ids"].append(request_id)
    statuses = {row["batch_status"] for row in state["shards"]}
    state["status"] = "terminal" if statuses and statuses <= terminal else "in_progress"
    atomic_json(state_path, state)
    return state


def cancel_all(manifest_path: Path, state_path: Path, selected: Profile, api: OpenAIHTTP) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, selected)
    state = _state(state_path, manifest)
    for shard in state["shards"]:
        if not shard["batch_id"] or shard["batch_status"] in {"completed", "failed", "expired", "cancelled"}:
            continue
        value, request_id = api.cancel(shard["batch_id"])
        shard["batch_status"] = value.get("status")
        if request_id:
            shard["api_request_ids"].append(request_id)
        atomic_json(state_path, state)
    state["status"] = "cancellation_requested"
    atomic_json(state_path, state)
    return state


def _chat_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Chat completion must contain exactly one choice")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Chat completion has no text content")
    return content


def terminal_ids(
    payload: bytes, expected_ids: set[str], source: str, *, require_success: bool
) -> set[str]:
    ids: set[str] = set()
    for row in _json_lines(payload, source):
        custom_id = row.get("custom_id")
        if custom_id not in expected_ids or custom_id in ids:
            raise ValueError(f"{source} contains a cross-shard, unknown, or duplicate custom_id")
        response = row.get("response")
        success = row.get("error") is None and isinstance(response, dict) and response.get("status_code") == 200
        if success != require_success:
            raise ValueError(f"{source} contains a terminal row in the wrong output/error file")
        ids.add(custom_id)
    return ids


def convert_outputs(
    raw_paths: Iterable[Path], selected: Profile, expected_ids: set[str], settings_hash: str,
    batch_by_id: dict[str, str], execution_lineage: dict[str, str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    resolved_models: set[str] = set()
    for path in raw_paths:
        for value in _json_lines(path.read_bytes(), path.name):
            custom_id = value.get("custom_id")
            if custom_id not in expected_ids or custom_id in seen:
                raise ValueError("output custom_id is unknown or duplicated")
            seen.add(custom_id)
            response = value.get("response")
            if value.get("error") is not None or not isinstance(response, dict) or response.get("status_code") != 200:
                continue
            body = response.get("body", {})
            resolved_model = body.get("model")
            if not isinstance(resolved_model, str) or not resolved_model:
                raise ValueError("response has no resolved model identity")
            if not selected.mutable_alias and resolved_model != selected.model:
                raise ValueError("response model differs from the immutable submitted snapshot")
            if selected.mutable_alias:
                try:
                    validate_snapshot_model(resolved_model)
                except ValueError as error:
                    raise ValueError("mutable alias response does not provide a dated resolved model version") from error
                if not resolved_model.startswith("gpt-5.5-"):
                    raise ValueError("mutable alias resolved to a model outside the gpt-5.5 family")
            resolved_models.add(resolved_model)
            request_id = response.get("request_id")
            response_id = body.get("id")
            if not request_id or not response_id or not batch_by_id.get(custom_id):
                raise ValueError("response receipt evidence is incomplete")
            usage = body.get("usage")
            if not isinstance(usage, dict):
                raise ValueError("response usage evidence is missing")
            token_values = [usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")]
            if not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in token_values):
                raise ValueError("response token usage is incomplete")
            if token_values[2] != token_values[0] + token_values[1]:
                raise ValueError("response total token usage is internally inconsistent")
            rows.append(
                {
                    "request_id": custom_id,
                    "backend_kind": "openai_compatible",
                    "provider": "OpenAI",
                    "model": selected.model,
                    "response_text": _chat_text(body),
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    },
                    "execution_metadata": {
                        "model_version": resolved_model,
                        "settings": EXECUTION_SETTINGS,
                        "receipt": {
                            "batch_id": batch_by_id[custom_id],
                            "openai_request_id": request_id,
                            "response_id": response_id,
                            "custom_id": custom_id,
                            "adapter": "pathopress.openai_batch/v1",
                            "settings_sha256": settings_hash,
                            "model_sha256": object_sha256(selected.model),
                            "approval_manifest_sha256": None if execution_lineage is None else execution_lineage.get("approval_manifest_sha256"),
                            "execution_preflight_sha256": None if execution_lineage is None else execution_lineage.get("preflight_sha256"),
                            "execution_contract_sha256": None if execution_lineage is None else execution_lineage.get("execution_contract_sha256"),
                            "adapter_manifest_sha256": None if execution_lineage is None else execution_lineage.get("adapter_manifest_sha256"),
                            "transport_profile_sha256": None if execution_lineage is None else execution_lineage.get("transport_profile_sha256"),
                            "capacity_profile_sha256": None if execution_lineage is None else execution_lineage.get("capacity_profile_sha256"),
                        },
                    },
                }
            )
    if len(resolved_models) > 1:
        raise ValueError("responses contain mixed resolved model versions")
    return rows


def fetch_outputs(manifest_path: Path, state_path: Path, selected: Profile, api: OpenAIHTTP) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, selected)
    state = refresh(manifest_path, state_path, selected, api)
    expected_ids: set[str] = set()
    batch_by_id: dict[str, str] = {}
    downloaded_outputs: list[Path] = []
    allowed_raw_paths: set[Path] = set()
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}
    for receipt, shard in zip(manifest["shards"], state["shards"]):
        if shard["batch_status"] not in terminal_statuses:
            raise RuntimeError(f"batch shard {shard['shard_index']} is not terminal; poll status before fetch")
        shard_ids: set[str] = set()
        for line in _json_lines((manifest_path.parent / receipt["input_path"]).read_bytes(), receipt["input_path"]):
            expected_ids.add(line["custom_id"])
            shard_ids.add(line["custom_id"])
            batch_by_id[line["custom_id"]] = shard["batch_id"]
        terminal_by_kind: dict[str, set[str]] = {"output": set(), "error": set()}
        for kind in ("output", "error"):
            file_id = shard[f"{kind}_file_id"]
            if not file_id:
                continue
            path = manifest_path.parent / f"raw_{kind}-{shard['shard_index']:04d}.jsonl"
            allowed_raw_paths.add(path)
            payload, request_id = api.file_content(file_id)
            terminal_by_kind[kind] = terminal_ids(
                payload, shard_ids, path.name, require_success=kind == "output"
            )
            _atomic_write(path, payload)
            if kind == "output":
                downloaded_outputs.append(path)
            shard[f"raw_{kind}_sha256"] = sha256_bytes(payload)
            if request_id:
                shard["api_request_ids"].append(request_id)
        if terminal_by_kind["output"] & terminal_by_kind["error"]:
            raise ValueError("provider output and error files duplicate a custom_id")
        if terminal_by_kind["output"] | terminal_by_kind["error"] != shard_ids:
            raise ValueError(f"terminal output/error union is incomplete for shard {shard['shard_index']}")
        counts = shard.get("request_counts")
        if not isinstance(counts, dict) or (
            counts.get("total") != len(shard_ids)
            or counts.get("completed") != len(terminal_by_kind["output"])
            or counts.get("failed") != len(terminal_by_kind["error"])
        ):
            raise ValueError("provider request_counts disagree with terminal output/error files")
    existing_raw = set(manifest_path.parent.glob("raw_output-*.jsonl")) | set(
        manifest_path.parent.glob("raw_error-*.jsonl")
    )
    stale = existing_raw - allowed_raw_paths
    if stale:
        raise ValueError(f"stale output/error files are present: {[path.name for path in sorted(stale)]}")
    converted = convert_outputs(
        downloaded_outputs, selected, expected_ids, manifest["execution_settings_sha256"],
        batch_by_id,
        state.get("execution_contract"),
    )
    output = manifest_path.parent / "raw_provider_responses.jsonl"
    _atomic_write(output, "".join(canonical_json(row) + "\n" for row in converted).encode())
    state["successful_response_count"] = len(converted)
    state["raw_provider_responses_sha256"] = sha256_file(output)
    state["status"] = "outputs_fetched_complete" if len(converted) == EXPECTED_REQUEST_COUNT else "outputs_fetched_incomplete"
    atomic_json(state_path, state)
    return state
