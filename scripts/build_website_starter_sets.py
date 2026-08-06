#!/usr/bin/env python3
"""Build the hash-bound starter-set payload consumed by the static site."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.starter_sets import (  # noqa: E402
    build_pending_starter_sets,
    build_starter_sets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-compression", type=Path,
        default=ROOT / "experiments/probe_compression_rank1.json",
    )
    parser.add_argument(
        "--feasibility-allowlist", type=Path,
        default=ROOT / "data/low_friction_allowlist_v2_top25.json",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "website/starter_sets.json"
    )
    parser.add_argument(
        "--pending", action="store_true",
        help="Write a current hash-bound placeholder without consuming probe results.",
    )
    parser.add_argument("--scores", type=Path, default=ROOT / "data/scores.csv")
    parser.add_argument(
        "--allowlist-legacy25", type=Path,
        default=ROOT / "data/low_friction_allowlist_v2_legacy25.json",
    )
    parser.add_argument(
        "--allowlist-pipeline-all", type=Path,
        default=ROOT / "data/low_friction_pipeline_eligible_v2_all.json",
    )
    args = parser.parse_args()
    if args.pending:
        payload = build_pending_starter_sets(
            args.scores,
            {
                "legacy25": args.allowlist_legacy25,
                "top25": args.feasibility_allowlist,
                "pipeline_eligible_all": args.allowlist_pipeline_all,
            },
        )
    else:
        payload = build_starter_sets(
            args.probe_compression, args.feasibility_allowlist, count=args.count
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
