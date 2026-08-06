#!/usr/bin/env python3
"""Validate exact merged summaries against full-record integrity-derived ordering."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant rejected: {value}")


def load_json(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"input must be a non-symlink regular file: {path}")
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def validate_runner_contract(config: dict[str, Any], runner_version: str) -> None:
    if runner_version == "v2" and (
        config.get("schema_version") != 2
        or config.get("config_schema") != "pathopress.probe_exhaustive.run.v2"
    ):
        raise RuntimeError("expected a schema-v2 run")
    if runner_version == "legacy" and config.get("schema_version", 1) != 1:
        raise RuntimeError("expected an archived legacy-v1 run")


def validate_run(
    run_dir: Path,
    integrity_path: Path,
    integrity: dict[str, Any],
    runner_version: str | None = None,
) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    merged_path = run_dir / "merged_summary.json.gz"
    config = load_json(config_path)
    merged = load_json(merged_path)
    if runner_version is not None:
        try:
            validate_runner_contract(config, runner_version)
        except RuntimeError as error:
            raise RuntimeError(f"{error}: {run_dir}") from error
    config_hash = sha256(config_path)
    matches = [
        row
        for row in integrity.get("runs", [])
        if row.get("config_sha256") == config_hash
    ]
    if len(matches) != 1:
        raise RuntimeError(f"integrity run match count is {len(matches)}: {run_dir}")
    certified = matches[0]
    total = int(config["total_combinations"])
    expected_top = certified.get("expected_top")
    top = merged.get("top")
    if (
        merged.get("config") != config
        or merged.get("complete") is not True
        or int(merged.get("n_records", -1)) != total
        or merged.get("missing_chunks") != []
        or merged.get("invalid_chunks") != []
    ):
        raise RuntimeError(f"merged completeness/config mismatch: {run_dir}")
    provenance = merged.get("integrity_manifest")
    if not isinstance(provenance, dict) or (
        provenance.get("path") != display(integrity_path)
        or provenance.get("sha256") != sha256(integrity_path)
        or provenance.get("config_sha256") != config_hash
        or provenance.get("chunk_digest_aggregate_sha256")
        != certified.get("chunk_digest_aggregate_sha256")
    ):
        raise RuntimeError(f"merged integrity provenance mismatch: {run_dir}")
    if not isinstance(expected_top, list) or top != expected_top:
        raise RuntimeError(f"merged top differs from all-record ordering: {run_dir}")
    if len(top) != int(certified.get("expected_top_count", -1)):
        raise RuntimeError(f"merged top length mismatch: {run_dir}")
    ordering = [
        (float(row["score"]), int(row["combo_index"])) for row in top
    ]
    if ordering != sorted(ordering):
        raise RuntimeError(f"merged top ordering is not deterministic: {run_dir}")
    if merged.get("best") != top[0]:
        raise RuntimeError(f"merged best is not global rank one: {run_dir}")
    return {
        "run_dir": display(run_dir),
        "config_sha256": config_hash,
        "merged_summary": display(merged_path),
        "merged_summary_sha256": sha256(merged_path),
        "total_combinations": total,
        "top_rows_validated": len(top),
        "best": top[0],
        "ordering_rule": "ascending (score, combo_index)",
        "all_record_order_source": (
            "expected_top derived by deep validation from every raw combination record"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--runner-version", choices=("v2", "legacy"), default="v2",
        help=(
            "validate schema-v2 production merges by default; use legacy only "
            "for archived v1 runs"
        ),
    )
    parser.add_argument(
        "--integrity-manifest",
        type=Path,
        default=ROOT / "experiments/probe_exhaustive_integrity_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/probe_exhaustive_merged_validation.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    integrity_path = args.integrity_manifest.resolve()
    integrity = load_json(integrity_path)
    if integrity.get("status") != "passed":
        raise RuntimeError("integrity manifest did not pass")
    expected_runner = (
        "experiments/run_probe_exhaustive_v2.py"
        if args.runner_version == "v2"
        else "experiments/run_probe_exhaustive.py"
    )
    if integrity.get("inputs", {}).get("runner_path") != expected_runner:
        raise RuntimeError("integrity manifest runner does not match --runner-version")
    runs = [
        validate_run(
            run_dir.resolve(), integrity_path, integrity, args.runner_version
        )
        for run_dir in args.run_dirs
    ]
    payload = {
        "schema_version": 1,
        "status": "passed",
        "runner_version": args.runner_version,
        "integrity_manifest": display(integrity_path),
        "integrity_manifest_sha256": sha256(integrity_path),
        "runs": runs,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
