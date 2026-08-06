from __future__ import annotations
import csv
from collections import Counter
from pathlib import Path
from scripts.evidence.wave_d_virchow2_paper import build_protocols, build_scores, load_evidence

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "source_data/wave_d_virchow2_paper_2408.00738.csv"

def test_virchow2_paper_complete_dispositions() -> None:
    rows = load_evidence(SNAPSHOT)
    assert Counter(row["disposition"] for row in rows) == {
        "accepted_public_leaf": 144, "quarantined_internal_cohort": 24,
        "excluded_derived_aggregate": 24,
    }

def test_virchow2_recipes_and_hest_protocols_stay_separate() -> None:
    scores, _ = build_scores(SNAPSHOT)
    assert len(scores) == 144
    assert Counter(row["model_id"] for row in scores) == {
        "virchow-2": 36, "virchow": 36, "h-optimus-0": 36, "uni": 36,
    }
    by_key = {(row["model_id"], row["evaluation_id"]): row for row in scores}
    assert by_key[("virchow-2", "virchow2paper2024.cls_mean.mhist.weighted_f1")]["value"] == "0.859"
    assert by_key[("virchow-2", "virchow2paper2024.cls_only.mhist.weighted_f1")]["value"] == "0.86"
    assert by_key[("h-optimus-0", "hest.virchow2paper2024.cls_mean.skcm.random_forest")]["value"] == "0.65"

def test_virchow2_protocol_inventory() -> None:
    with (ROOT / "data/tasks.csv").open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle))
    protocols = build_protocols(SNAPSHOT, tasks)
    assert len(protocols) == 36
    assert len({row["protocol_id"] for row in protocols}) == 36
