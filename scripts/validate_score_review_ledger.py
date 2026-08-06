#!/usr/bin/env python3
"""Validate the committed score-review ledger without network access."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.score_review import read_csv, sha256_path, validate_ledger  # noqa: E402


def main() -> None:
    data = ROOT / "data"
    ledger_path = data / "score_review_ledger.csv"
    summary_path = data / "score_review_summary.json"
    result = validate_ledger(
        read_csv(data / "scores.csv"),
        read_csv(data / "tasks.csv"),
        read_csv(data / "deduplication.csv"),
        read_csv(ledger_path),
    )
    expected = json.loads(summary_path.read_text(encoding="utf-8"))
    if expected["ledger_sha256"] != sha256_path(ledger_path):
        raise ValueError("ledger hash does not match summary")
    for key, value in result.items():
        if expected.get(key) != value:
            raise ValueError(f"summary mismatch: {key}")
    for relative, digest in expected["source_files"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_path(source) != digest:
            raise ValueError(f"pinned evidence mismatch: {relative}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
