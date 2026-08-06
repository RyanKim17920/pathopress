#!/usr/bin/env python3
"""Validate and merge cached provider-neutral LLM response shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.llm_baseline import evaluate_cached_responses, validate_config  # noqa: E402
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "data/scores.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/llm_baseline/config.json")
    parser.add_argument("--requests", type=Path, default=ROOT / "experiments/llm_baseline/requests.jsonl")
    parser.add_argument("--responses", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/llm_baseline/merged_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix, _, _ = filter_matrix(*make_matrix(load_scores(args.scores)))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    requests = _read_jsonl(args.requests)
    responses = [row for path in args.responses for row in _read_jsonl(path)]
    ids = [row["request_id"] for row in responses]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate request_id across response shards")
    result = evaluate_cached_responses(requests, responses, matrix, config)
    result["response_shards"] = [str(path.resolve().relative_to(ROOT.resolve())) for path in args.responses]
    result["publication_policy"] = (
        "Eligible for headline comparison only when all used responses declare a validated real backend kind. "
        "Mock metrics remain contract validation only."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"status={result['result_status']}; headline_eligible={result['headline_eligible']}")


if __name__ == "__main__":
    main()
