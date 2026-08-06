#!/usr/bin/env python3
"""Validate the pinned Virchow primary-paper evidence snapshot."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from evidence.wave_d_virchow_paper import build_protocols, build_scores, load_evidence


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "source_data/wave_d_virchow_paper_2309.07778.csv"


def main() -> None:
    rows = load_evidence(SNAPSHOT)
    protocols = build_protocols(SNAPSHOT)
    scores, _ = build_scores(SNAPSHOT)
    print({"dispositions": dict(Counter(row["disposition"] for row in rows)), "protocols": len(protocols), "scores": len(scores)})


if __name__ == "__main__":
    main()
