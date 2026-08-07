#!/usr/bin/env python3
"""Build the compact, hash-bound summary of the completed exact probe searches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cheap-run", type=Path, required=True)
    parser.add_argument("--pruned-run", type=Path, required=True)
    parser.add_argument(
        "--scalar-validation",
        type=Path,
        default=ROOT / "experiments/probe_exhaustive_scalar_top_validation.json",
    )
    parser.add_argument(
        "--integrity-manifest",
        type=Path,
        default=ROOT / "experiments/probe_exhaustive_integrity_manifest.json",
    )
    parser.add_argument(
        "--merged-validation",
        type=Path,
        default=ROOT / "experiments/probe_exhaustive_merged_validation.json",
    )
    parser.add_argument(
        "--fast-equivalence",
        type=Path,
        default=ROOT / "experiments/probe_exhaustive_fast_equivalence_v2.json",
    )
    parser.add_argument(
        "--compression",
        type=Path,
        default=ROOT / "experiments/probe_compression_rank1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/probe_exhaustive_rank1.json",
    )
    parser.add_argument("--retain-top", type=int, default=100)
    return parser.parse_args()


def summarize_space(
    run_dir: Path,
    scalar_by_config: dict[str, dict[str, Any]],
    integrity_by_config: dict[str, dict[str, Any]],
    merged_by_config: dict[str, dict[str, Any]],
    retain_top: int,
    *,
    label: str,
    candidate_semantics: str,
) -> tuple[dict[str, Any], str]:
    config_path = run_dir / "config.json"
    merged_path = run_dir / "merged_summary.json.gz"
    config = load_json(config_path)
    if (
        config.get("schema_version") != 2
        or config.get("config_schema") != "pathopress.probe_exhaustive.run.v2"
    ):
        raise RuntimeError(f"exact summary requires a schema-v2 run: {run_dir}")
    merged = load_json(merged_path)
    config_hash = sha256(config_path)
    if merged.get("config") != config:
        raise RuntimeError(f"merged/config mismatch: {run_dir}")
    total = int(config["total_combinations"])
    if (
        not merged.get("complete")
        or int(merged.get("n_records", -1)) != total
        or merged.get("missing_chunks")
        or merged.get("invalid_chunks")
    ):
        raise RuntimeError(f"exact search is not strictly complete: {run_dir}")
    top = list(merged.get("top", []))
    if len(top) < retain_top:
        raise RuntimeError(
            f"merged summary retains {len(top)} rows, need {retain_top}: {run_dir}"
        )
    scalar = scalar_by_config.get(config_hash)
    if scalar is None or not scalar.get("winner_certified"):
        raise RuntimeError(f"missing scalar winner certification: {run_dir}")
    if int(scalar.get("total_combinations", -1)) != total:
        raise RuntimeError(f"scalar/config total mismatch: {run_dir}")
    integrity = integrity_by_config.get(config_hash)
    if (
        integrity is None
        or int(integrity.get("validated_records", -1)) != total
        or int(integrity.get("validated_chunks", -1))
        != int(integrity.get("expected_chunks", -2))
    ):
        raise RuntimeError(f"missing full-record integrity validation: {run_dir}")
    merged_validation = merged_by_config.get(config_hash)
    if (
        merged_validation is None
        or merged_validation.get("merged_summary_sha256") != sha256(merged_path)
        or int(merged_validation.get("total_combinations", -1)) != total
    ):
        raise RuntimeError(f"missing merged-order validation: {run_dir}")
    best = merged.get("best")
    scalar_best = scalar.get("best", {})
    if (
        best is None
        or int(best["combo_index"]) != int(scalar_best.get("combo_index", -1))
        or list(best["probe_set"]) != list(scalar_best.get("probe_set", []))
    ):
        raise RuntimeError(f"accelerated/scalar winner mismatch: {run_dir}")
    return (
        {
            "label": label,
            "candidate_semantics": candidate_semantics,
            "globality": (
                f"exhaustive within the declared {len(config['candidate_ids'])}-task "
                "candidate universe; not globally exhaustive over all retained evaluations"
            ),
            "candidate_count": len(config["candidate_ids"]),
            "candidate_ids": config["candidate_ids"],
            "k": int(config["k"]),
            "objective": config["metric"],
            "combination_count": total,
            "complete": True,
            "config": display(config_path),
            "config_sha256": config_hash,
            "candidate_allowlist": config["candidate_allowlist_path"],
            "candidate_allowlist_sha256": config["candidate_allowlist_sha256"],
            "merged_summary": display(merged_path),
            "merged_summary_sha256": sha256(merged_path),
            "merge_validation": (
                "strict expected-chunk validation; exact unique combo_index coverage "
                f"0..{total - 1}; no missing, invalid, or duplicate records"
            ),
            "best": best,
            "top": top[:retain_top],
            "scalar_certification": {
                key: value
                for key, value in scalar.items()
                if key not in {"comparisons", "run_dir"}
            },
            "full_record_integrity": {
                key: value
                for key, value in integrity.items()
                if key not in {"chunks", "run_dir", "config"}
            },
            "merged_order_validation": {
                key: value
                for key, value in merged_validation.items()
                if key not in {"run_dir", "merged_summary", "best"}
            },
        },
        str(config["scores_sha256"]),
    )


def main() -> int:
    args = parse_args()
    if args.retain_top < 2:
        raise ValueError("--retain-top must be at least 2")
    scalar_payload = load_json(args.scalar_validation)
    if scalar_payload.get("status") != "passed":
        raise RuntimeError("scalar validation did not pass")
    scalar_by_config = {
        str(row["config_sha256"]): row for row in scalar_payload.get("runs", [])
    }
    integrity_payload = load_json(args.integrity_manifest)
    if integrity_payload.get("status") != "passed":
        raise RuntimeError("full-record integrity validation did not pass")
    integrity_by_config = {
        str(row["config_sha256"]): row
        for row in integrity_payload.get("runs", [])
    }
    merged_payload = load_json(args.merged_validation)
    if merged_payload.get("status") != "passed":
        raise RuntimeError("merged-order validation did not pass")
    merged_by_config = {
        str(row["config_sha256"]): row for row in merged_payload.get("runs", [])
    }
    cheap, cheap_scores_hash = summarize_space(
        args.cheap_run.resolve(),
        scalar_by_config,
        integrity_by_config,
        merged_by_config,
        args.retain_top,
        label="pre_error_pipeline_proxy_25_choose_5",
        candidate_semantics=(
            "pre-error low-friction input/label pipeline proxy; this is not measured "
            "evaluation cost or cheapness"
        ),
    )
    pruned, pruned_scores_hash = summarize_space(
        args.pruned_run.resolve(),
        scalar_by_config,
        integrity_by_config,
        merged_by_config,
        args.retain_top,
        label="error_informed_pruned_30_choose_5",
        candidate_semantics=(
            "error-informed aggregate rank over all ten source MedAE greedy steps"
        ),
    )
    if cheap_scores_hash != pruned_scores_hash:
        raise RuntimeError("the two exact searches use different score matrices")
    if str(scalar_payload.get("scores_sha256")) != cheap_scores_hash:
        raise RuntimeError("scalar validation uses a different score matrix")
    compression = load_json(args.compression)
    compression_config = compression.get("configuration", {})
    if compression_config.get("scores_sha256") != cheap_scores_hash:
        raise RuntimeError("compression artifact uses a different score matrix")
    payload = {
        "schema_version": 2,
        "status": "executed_complete_scalar_certified",
        "scores_sha256": cheap_scores_hash,
        "matrix_shape": [
            int(load_json(args.cheap_run / "config.json")["n_models"]),
            int(load_json(args.cheap_run / "config.json")["n_evaluations"]),
        ],
        "n_observed": int(
            load_json(args.cheap_run / "config.json")["n_observed"]
        ),
        "scientific_protocol": "all_known_probe_bruteforce_v1",
        "semantics": (
            "Exact all-known BenchPress masking at k=5: each observed probe cell is "
            "revealed and scores with zero error; every other observed target cell is "
            "predicted by the declared rank-1 pathology completion model."
        ),
        "upstream_reference_commit": load_json(args.cheap_run / "config.json")[
            "upstream_reference_commit"
        ],
        "pathology_adaptation": (
            "The upstream search/masking/partition/merge contracts are preserved; "
            "pathology evaluation IDs and the independently selected rank-1 predictor "
            "replace BenchPress's source-domain matrix and rank-2 predictor."
        ),
        "provenance": {
            "compression": display(args.compression),
            "compression_sha256": sha256(args.compression),
            "fast_backend_equivalence": display(args.fast_equivalence),
            "fast_backend_equivalence_sha256": sha256(args.fast_equivalence),
            "scalar_validation": display(args.scalar_validation),
            "scalar_validation_sha256": sha256(args.scalar_validation),
            "integrity_manifest": display(args.integrity_manifest),
            "integrity_manifest_sha256": sha256(args.integrity_manifest),
            "merged_validation": display(args.merged_validation),
            "merged_validation_sha256": sha256(args.merged_validation),
        },
        "spaces": {
            "pre_error_proxy_25_choose_5": cheap,
            "error_informed_pruned_30_choose_5": pruned,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
