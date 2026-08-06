from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from scripts.evidence.wave_d_uni_paper import build_protocols, build_scores, load_evidence


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "source_data/wave_d_uni_paper_2308.15474.csv"


def test_uni_paper_exhaustive_row_and_disposition_audit() -> None:
    rows = load_evidence(SNAPSHOT)
    assert Counter(row["disposition"] for row in rows) == {
        "accepted_public_leaf": 227, "quarantined_finetuned": 24,
        "quarantined_internal_cohort": 23, "excluded_derived_aggregate": 3,
    }
    assert len({row["source_locator"].split("|", 1)[0] for row in rows}) == 83


def test_uni_public_metric_inventory_and_values() -> None:
    scores, aliases = build_scores(SNAPSHOT)
    assert len(scores) == 227
    assert Counter(row["metric"] for row in scores) == {
        "balanced_accuracy": 43, "weighted_f1": 43, "auroc": 35,
        "one_nn_balanced_accuracy": 19, "one_nn_weighted_f1": 19,
        "twenty_nn_balanced_accuracy": 19, "twenty_nn_weighted_f1": 19,
        "retrieval_accuracy_at_1": 7, "retrieval_accuracy_at_3": 7,
        "retrieval_accuracy_at_5": 7, "retrieval_majority_vote_accuracy_at_5": 7,
        "weighted_kappa": 2,
    }
    by_id = {row["evaluation_id"]: row for row in scores}
    assert by_id["unipaper2023.patch_crc100k_lin.default.balanced_accuracy"]["value"] == "0.874"
    assert by_id["unipaper2023.slide_panda.default.weighted_kappa"]["normalized_score"] == "97.3"
    assert aliases[0]["model_id"] == "uni"


def test_uni_protocol_inventory_keeps_settings_separate() -> None:
    with (ROOT / "data/tasks.csv").open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle))
    protocols = build_protocols(SNAPSHOT, tasks)
    assert len(protocols) == 227
    assert len({row["protocol_id"] for row in protocols}) == 227
    bach = [row for row in protocols if row["evaluation_id"].startswith("unipaper2023.patch_bach_lin.")]
    assert len(bach) == 12
    assert len({row["protocol"].split("row setting: ", 1)[1] for row in bach}) == 4
