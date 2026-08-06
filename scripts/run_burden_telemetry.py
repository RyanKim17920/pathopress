#!/usr/bin/env python3
"""Run one benchmark phase and write a provenance-bound burden receipt.

The runner intentionally records only resources that the local operating
system can observe directly.  Accelerator time, VRAM, data transfer, labor,
tissue, and dollars remain explicit ``not_measured`` values until a dedicated
instrument supplies them; unknown values are never converted to zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "data/evaluation_burden_measurements.schema.json"

ALLOWED_STATUSES = {
    "measured",
    "source_reported",
    "configured_ceiling",
    "not_applicable",
    "not_measured",
    "not_reported",
    "inaccessible",
}
PHASES = {
    "shared_artifact_setup",
    "per_model_feature_extraction",
    "per_protocol_head_fit",
    "per_protocol_evaluation",
}
CACHE_SCOPES = {"cold", "warm", "mixed", "not_applicable"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def measured(value: float, unit: str, method: str) -> dict[str, Any]:
    return {
        "status": "measured",
        "value": value,
        "unit": unit,
        "measurement_method": method,
    }


def unknown(unit: str, reason: str) -> dict[str, Any]:
    return {
        "status": "not_measured",
        "value": None,
        "unit": unit,
        "reason": reason,
    }


def categorical_unknown(reason: str) -> dict[str, Any]:
    return {"status": "not_measured", "value": None, "reason": reason}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-revision")
    parser.add_argument("--evaluation-id")
    parser.add_argument("--artifact-group-id", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--cache-scope", choices=sorted(CACHE_SCOPES), required=True)
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument("--run-config", type=Path)
    config.add_argument("--run-config-hash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing receipt.")
    parser.add_argument(
        "--redact-command",
        action="store_true",
        help="Store only the command hash, not its argument vector.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.phase != "shared_artifact_setup":
        if not args.model_revision or not args.evaluation_id:
            parser.error("model revision and evaluation id are required for per-model/protocol phases")
    elif args.model_revision is not None:
        parser.error("model revision must be omitted for shared artifact setup")
    if args.run_config_hash is not None:
        value = args.run_config_hash.lower()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            parser.error("--run-config-hash must be a 64-character lowercase SHA-256 digest")
    return args


def config_identity(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.run_config is not None:
        raw = args.run_config.read_bytes()
        return sha256_bytes(raw), str(args.run_config)
    return args.run_config_hash, None


def _rss_gib(usage: resource.struct_rusage) -> float:
    # Linux reports ru_maxrss in KiB; macOS reports bytes.
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return float(usage.ru_maxrss) / divisor


def build_receipt(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    run_config_hash, run_config_path = config_identity(args)
    command_bytes = json.dumps(args.command, ensure_ascii=False).encode("utf-8")
    command_hash = sha256_bytes(command_bytes)
    key_payload = {
        "model_revision": args.model_revision,
        "evaluation_id": args.evaluation_id,
        "run_config_hash": run_config_hash,
        "hardware_id": args.hardware_id,
        "cache_scope": args.cache_scope,
        "artifact_group_id": args.artifact_group_id,
        "phase": args.phase,
    }
    measurement_id = sha256_bytes(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    started_at = datetime.now(timezone.utc).isoformat()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_start = time.perf_counter()
    completed = subprocess.run(args.command, check=False)
    wall_seconds = time.perf_counter() - wall_start
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    completed_at = datetime.now(timezone.utc).isoformat()

    user_seconds = max(0.0, child_after.ru_utime - child_before.ru_utime)
    system_seconds = max(0.0, child_after.ru_stime - child_before.ru_stime)
    resources = {
        "wall_time": measured(wall_seconds, "seconds", "time.perf_counter around child process"),
        "cpu_user_time": measured(user_seconds, "seconds", "getrusage(RUSAGE_CHILDREN)"),
        "cpu_system_time": measured(system_seconds, "seconds", "getrusage(RUSAGE_CHILDREN)"),
        "peak_ram": measured(_rss_gib(child_after), "GiB", "getrusage(RUSAGE_CHILDREN).ru_maxrss"),
        "accelerator_time": unknown("accelerator-seconds", "No accelerator utilization sampler was attached."),
        "peak_vram": unknown("GiB", "No accelerator-memory sampler was attached."),
        "download_volume": unknown("GiB", "Network transfer was not instrumented."),
        "storage_volume": unknown("GiB", "Filesystem staging volume was not instrumented."),
        "access_lead_time": unknown("hours", "Data-access lead time was not instrumented."),
        "access_admin_labor": unknown("hours", "Administrative access labor was not instrumented."),
        "annotation_labor": unknown("hours", "Annotation labor was not instrumented."),
        "pathologist_labor": unknown("hours", "Pathologist labor was not instrumented."),
        "new_tissue_cases": unknown("cases", "Tissue acquisition was not instrumented."),
        "new_slides": unknown("slides", "Slide acquisition was not instrumented."),
        "direct_cost": unknown("USD", "No billing record was attached."),
    }
    constraints = {
        "access_class": categorical_unknown("Dataset access class was not attached to this run."),
        "dataset_license": categorical_unknown("Dataset license evidence was not attached to this run."),
        "commercial_use_allowed": categorical_unknown(
            "Commercial-use permission was not attached to this run."
        ),
        "redistribution_allowed": categorical_unknown(
            "Redistribution permission was not attached to this run."
        ),
        "new_tissue_required": categorical_unknown(
            "The run did not record whether new tissue acquisition was required."
        ),
    }
    measurement = {
        "measurement_id": measurement_id,
        **key_payload,
        "execution_status": "completed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "started_at": started_at,
        "completed_at": completed_at,
        "command_sha256": command_hash,
        "command": None if args.redact_command else args.command,
        "run_config_path": run_config_path,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "resources": resources,
        "constraints": constraints,
        "scope_note": (
            "Shared artifact setup is charged once per artifact group."
            if args.phase == "shared_artifact_setup"
            else "Per-model/protocol phase; do not treat shared artifact setup as a repeated cost."
        ),
    }
    return {
        "schema_version": 1,
        "schema_path": str(SCHEMA_PATH.relative_to(ROOT)),
        "measurement": measurement,
    }, completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite existing receipt: {args.output}")
    receipt, exit_code = build_receipt(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
