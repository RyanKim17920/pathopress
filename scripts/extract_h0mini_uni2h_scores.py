#!/usr/bin/env python3
"""Validate the committed H0-mini/UNI2-h official-source score snapshot.

The evidence CSV records exact raw-source revisions, hashes, table locators,
leaf/aggregate/private dispositions, and values. This command is intentionally
network-free; use it after independently acquiring the three pinned sources.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evidence.h0mini_uni2h_scores import load_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path,
        default=Path(__file__).resolve().parents[1] / "source_data/h0mini_uni2h_official_scores_2025.csv",
    )
    args = parser.parse_args()
    rows = load_evidence(args.snapshot)
    print(json.dumps({
        "rows": len(rows),
        "sources": dict(sorted(Counter(row["source_id"] for row in rows).items())),
        "dispositions": dict(sorted(Counter(row["disposition"] for row in rows).items())),
        "public_cells_by_suite": dict(sorted(Counter(
            row["suite_id"] for row in rows if row["disposition"].startswith("accepted_public")
        ).items())),
        "public_cells_by_model": dict(sorted(Counter(
            row["model_id"] for row in rows if row["disposition"].startswith("accepted_public")
        ).items())),
    }, indent=2))


if __name__ == "__main__":
    main()
