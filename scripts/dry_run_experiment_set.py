#!/usr/bin/env python3
"""Validate the repository experiment set without executing any experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.maintenance import validate_experiment_set  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="experiment_set", type=Path, default=ROOT / "experiments/experiment_set.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/experiment_set_dry_run.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.experiment_set.read_text(encoding="utf-8"))
    results = validate_experiment_set(ROOT, payload)
    output = {
        "schema_version": 2,
        "status": "ready" if all(row["status"] == "ready" for row in results) else "blocked",
        "execution": "none_dry_run_only",
        "summary": {
            "components": len(results),
            "declared_inputs": sum(row["declared_inputs"] for row in results),
            "declared_dependencies": sum(row["declared_dependencies"] for row in results),
            "declared_artifacts": sum(row["declared_artifacts"] for row in results),
            "external_call_components": sum(bool(row["external_calls"]) for row in results),
        },
        "experiments": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if output["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
