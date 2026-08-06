#!/usr/bin/env python3
"""Create or verify hash-bound artifact freshness manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.maintenance import (  # noqa: E402
    build_freshness_manifest,
    build_result_graph_manifest,
    check_freshness_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "check"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/artifact_freshness_manifest.json")
    parser.add_argument("--experiment-set", type=Path, default=ROOT / "experiments/experiment_set.json")
    parser.add_argument("--inputs", nargs="*", type=Path, default=[])
    parser.add_argument("--artifacts", nargs="*", type=Path, default=[])
    parser.add_argument("--kind", default="pathopress_completed_result_graph")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "build":
        if args.inputs or args.artifacts:
            if not args.inputs or not args.artifacts:
                raise ValueError("legacy build requires both --inputs and --artifacts")
            payload = build_freshness_manifest(
                ROOT, inputs=args.inputs, artifacts=args.artifacts, kind=args.kind
            )
        else:
            experiment_set = json.loads(args.experiment_set.read_text(encoding="utf-8"))
            payload = build_result_graph_manifest(
                ROOT,
                experiment_set_path=args.experiment_set,
                experiment_set=experiment_set,
                kind=args.kind,
            )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.manifest}")
        return
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = check_freshness_manifest(ROOT, payload)
    if failures:
        print(json.dumps({"status": "stale", "failures": failures}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"status": "fresh", "manifest": str(args.manifest)}, indent=2))


if __name__ == "__main__":
    main()
