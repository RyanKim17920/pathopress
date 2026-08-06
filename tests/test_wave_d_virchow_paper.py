from __future__ import annotations

from collections import Counter
from pathlib import Path

from scripts.evidence.wave_d_virchow_paper import build_protocols, build_scores, load_evidence


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "source_data/wave_d_virchow_paper_2309.07778.csv"


def test_virchow_paper_complete_dispositions() -> None:
    rows = load_evidence(SNAPSHOT)
    assert Counter(row["disposition"] for row in rows) == {
        "accepted_public_leaf": 15,
        "quarantined_internal_cohort": 6,
    }


def test_virchow_public_protocol_and_score_inventory() -> None:
    protocols = build_protocols(SNAPSHOT)
    scores, aliases = build_scores(SNAPSHOT)
    assert len(protocols) == len(scores) == 15
    assert len({row["protocol_id"] for row in protocols}) == 15
    assert aliases[0]["model_id"] == "virchow"
    by_id = {row["evaluation_id"]: row for row in scores}
    assert by_id["virchowpaper2023.crc.linear_probe.accuracy"]["normalized_score"] == "97.3"
    assert by_id["virchowpaper2023.mhist.linear_probe.weighted_f1"]["value"] == "0.835"


def test_virchow_metrics_remain_separate_protocols() -> None:
    protocols = build_protocols(SNAPSHOT)
    assert Counter(row["metric"] for row in protocols) == {
        "accuracy": 5,
        "balanced_accuracy": 5,
        "weighted_f1": 5,
    }
    assert len({row["task_identity_id"] for row in protocols}) == 5
