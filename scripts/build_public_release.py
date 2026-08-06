#!/usr/bin/env python3
"""Build local public tables, deploy-time confidence, and static website JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.prediction import build_deployment_confidence_artifact  # noqa: E402
from pathopress.public_data import build_public_export, build_website_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "data" / "scores.csv")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data" / "tasks.csv")
    parser.add_argument("--suites", type=Path, default=ROOT / "data" / "suites.csv")
    parser.add_argument(
        "--provenance", type=Path, default=ROOT / "data" / "provenance.json"
    )
    parser.add_argument(
        "--model-metadata", type=Path, default=ROOT / "data" / "model_metadata.csv"
    )
    parser.add_argument(
        "--confidence-cells",
        type=Path,
        default=ROOT / "experiments" / "confidence_cells_rank1.csv",
    )
    parser.add_argument(
        "--confidence-output",
        type=Path,
        default=ROOT / "experiments" / "deployment_confidence_rank1.json",
    )
    parser.add_argument(
        "--confidence-calibration",
        type=Path,
        default=ROOT / "experiments" / "confidence_calibration_rank1.json",
        help="Full cross-fitted hybrid confidence/trust artifact",
    )
    parser.add_argument(
        "--new-model-confidence",
        type=Path,
        default=ROOT / "experiments" / "new_model_confidence_rank1.json",
    )
    parser.add_argument(
        "--probe-compression",
        type=Path,
        default=ROOT / "experiments" / "probe_compression_rank1.json",
    )
    parser.add_argument(
        "--export-dir", type=Path, default=ROOT / "exports" / "pathopress_public"
    )
    parser.add_argument(
        "--website-data", type=Path, default=ROOT / "website" / "data.json"
    )
    parser.add_argument(
        "--core-only", action="store_true",
        help=(
            "Build registry/matrix exports without probe or confidence artifacts; "
            "the static site exposes those optional features as pending."
        ),
    )
    parser.add_argument(
        "--allowlist-legacy25", type=Path,
        default=ROOT / "data" / "low_friction_allowlist_v2_legacy25.json",
    )
    parser.add_argument(
        "--allowlist-top25", type=Path,
        default=ROOT / "data" / "low_friction_allowlist_v2_top25.json",
    )
    parser.add_argument(
        "--allowlist-pipeline-all", type=Path,
        default=ROOT / "data" / "low_friction_pipeline_eligible_v2_all.json",
    )
    args = parser.parse_args()

    if not args.core_only:
        confidence = build_deployment_confidence_artifact(
            args.confidence_cells, args.scores,
            confidence_calibration_path=args.confidence_calibration,
        )
        args.confidence_output.parent.mkdir(parents=True, exist_ok=True)
        args.confidence_output.write_text(
            json.dumps(confidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    manifest = build_public_export(
        scores_path=args.scores,
        tasks_path=args.tasks,
        suites_path=args.suites,
        provenance_path=args.provenance,
        model_metadata_path=args.model_metadata,
        out_dir=args.export_dir,
    )
    website = build_website_data(
        scores_path=args.scores,
        tasks_path=args.tasks,
        model_metadata_path=args.model_metadata,
        output_path=args.website_data,
        probe_compression_path=None if args.core_only else args.probe_compression,
        confidence_artifact_path=None if args.core_only else args.confidence_output,
        new_model_confidence_artifact_path=(
            None if args.core_only else args.new_model_confidence
        ),
        feasibility_allowlist_paths={
            "legacy25": args.allowlist_legacy25,
            "top25": args.allowlist_top25,
            "pipeline_eligible_all": args.allowlist_pipeline_all,
        },
    )
    print(
        "confidence=omitted (core-only)"
        if args.core_only else f"confidence={args.confidence_output}"
    )
    print(f"export={args.export_dir}")
    print(
        "paper_matrix="
        f"{manifest['paper_filter']['models']}x{manifest['paper_filter']['evaluations']} "
        f"observed={manifest['paper_filter']['observations']}"
    )
    print(
        f"website={args.website_data} models={website['meta']['models']} "
        f"evaluations={website['meta']['evaluations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
