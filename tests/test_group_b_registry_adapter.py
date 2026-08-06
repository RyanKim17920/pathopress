from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from scripts.evidence.group_b_official_scores import build_protocols, build_scores, selected

ROOT = Path(__file__).resolve().parents[1]
PATHS = [
    ROOT / "source_data/genbio_pathfm_official_2026.csv",
    ROOT / "source_data/midnight_miccai2025_official_scores.csv",
    ROOT / "source_data/openmidnight_technical_report_2025.csv",
]


def test_group_b_selected_inventory() -> None:
    rows = selected(PATHS)
    assert len(rows) == 57
    assert Counter(row["model_id"] for row in rows) == {
        "genbio-pathfm": 21, "midnight-92k": 12, "midnight": 12, "openmidnight": 12,
    }


def test_group_b_protocols_and_scores() -> None:
    with (ROOT / "data/tasks.csv").open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle))
    protocols = build_protocols(PATHS, tasks)
    scores, _ = build_scores(PATHS)
    assert len(protocols) == 45
    assert len(scores) == 57
    assert sum(row["audit_status"].endswith("analysis_ineligible") for row in scores) == 6
    assert all(row["normalized_score"] == "" for row in scores if row["audit_status"].endswith("analysis_ineligible"))
    by_id = {(row["model_id"], row["evaluation_id"]): row for row in scores}
    assert by_id[("genbio-pathfm", "thunder.genbio2026.bach.knn")]["normalized_score"] == "81.8"
    assert by_id[("midnight", "eva.miccai2025.clsmean_224.bach.validation")]["normalized_score"] == "90.7"
