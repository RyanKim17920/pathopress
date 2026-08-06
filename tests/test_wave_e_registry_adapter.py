from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from scripts.evidence.wave_e_official_scores import build_protocols, build_scores, selected

ROOT = Path(__file__).resolve().parents[1]
PATHS = [
    ROOT / "source_data/conch_official_scores_2024.csv",
    ROOT / "source_data/conch15_titan_official_scores_2025.csv",
    ROOT / "source_data/phikon_family_official_scores_2023_2024.csv",
    ROOT / "source_data/ctranspath_official_evidence_2022_2024.csv",
]
GROUP_C = ROOT / "source_data/virchow2g_gigapath_titan_official_scores_2024_2025.csv"


def test_wave_e_selected_inventory() -> None:
    rows = selected(PATHS)
    assert Counter(row["model_id"] for row in rows) == {
        "conch": 77, "conch-1.5": 297, "phikon": 34, "phikon-v2": 9,
    }
    assert not any(row["model_id"] == "ctranspath" for row in rows)


def test_wave_e_protocols_scores_and_titan_identity_links() -> None:
    with (ROOT / "data/tasks.csv").open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle))
    # Current checked-in tasks predate Group C, so seed its protocol rows for exact-link testing.
    from scripts.evidence.group_c_official_scores import build_protocols as group_c_protocols
    tasks.extend(group_c_protocols(GROUP_C))
    protocols = build_protocols(PATHS, tasks, GROUP_C)
    scores, _ = build_scores(PATHS)
    assert len(protocols) == len(scores) == 417
    t22 = next(row for row in protocols if row["evaluation_id"] == "conch15.titan2025.t22.0.logistic_regression.balanced_accuracy")
    assert t22["task_identity_id"] == "task.tcga-ut-8k.tcga-ut-8k-tumor-subtype"
    by_id = {row["evaluation_id"]: row for row in scores}
    assert by_id["conch.natmed2024.t1.0.zero-shot_classification.balanced_accuracy"]["normalized_score"] == "91.3"
    assert by_id["phikon-v2.phikonv2.t2.metastasis.camelyon16.auroc"]["normalized_score"] == "99.7"
