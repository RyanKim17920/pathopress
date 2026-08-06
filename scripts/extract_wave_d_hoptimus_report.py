#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evidence.wave_d_hoptimus_report import load_evidence


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "source_data/wave_d_hoptimus1_official_report_2025.csv"
    rows = load_evidence(path)
    print(json.dumps({
        "rows": len(rows),
        "dispositions": dict(Counter(row["disposition"] for row in rows)),
        "models": dict(Counter(row["model_id"] for row in rows)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
