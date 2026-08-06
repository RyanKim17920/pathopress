from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from scripts.evidence.h0mini_uni2h_scores import (
    build_protocols,
    build_scores,
    load_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "source_data/h0mini_uni2h_official_scores_2025.csv"


def _tasks() -> list[dict[str, object]]:
    with (ROOT / "data/tasks.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_evidence_accounts_for_every_public_private_and_aggregate_cell() -> None:
    rows = load_evidence(SNAPSHOT)
    assert len(rows) == 86
    assert Counter(row["source_id"] for row in rows) == {
        "h0mini_arxiv_v3": 68,
        "plism_repo": 10,
        "uni_official_repo": 8,
    }
    assert Counter(row["disposition"] for row in rows) == {
        "accepted_public_leaf": 54,
        "accepted_public_metric_unspecified": 6,
        "excluded_derived_aggregate": 7,
        "quarantined_private_cohort": 19,
    }


def test_public_cells_are_versioned_and_private_or_derived_cells_do_not_enter() -> None:
    scores, _ = build_scores(SNAPSHOT)
    assert len(scores) == 60
    assert Counter(row["suite_id"] for row in scores) == {
        "eva": 16, "hest": 18, "plism": 20, "uni2_benchmark": 6,
    }
    assert Counter(row["model_id"] for row in scores) == {"uni2-h": 33, "h0-mini": 27}
    assert len({(row["model_id"], row["evaluation_id"]) for row in scores}) == 60
    assert not any("aggregate" in row["evaluation_id"] for row in scores)
    assert not any(row["suite_id"] == "breastbm_private" for row in scores)
    uni_repo = [row for row in scores if row["suite_id"] == "uni2_benchmark"]
    assert len(uni_repo) == 6
    assert all(row["normalized_score"] == "" for row in uni_repo)
    assert all(row["audit_status"] == "parsed_primary_source_analysis_ineligible" for row in uni_repo)


def test_exact_values_and_protocol_variants_are_preserved() -> None:
    scores, _ = build_scores(SNAPSHOT)
    by_key = {(row["model_id"], row["evaluation_id"]): row for row in scores}
    assert by_key[("h0-mini", "eva.h0mini2025.bach")]["value"] == "0.774"
    assert by_key[("uni2-h", "hest.h0mini2025.skcm.gene_expression")]["value"] == "0.6829"
    assert by_key[("h0-mini", "plism.h0mini2025.fixed_staining_cross_scanner.top10_accuracy")]["value"] == "0.86"
    assert by_key[("h0-mini", "plism.repo2025.fixed_staining_cross_scanner.top10_accuracy")]["value"] == "0.864"
    assert by_key[("uni2-h", "uni2repo2026.ebrains.reported_performance")]["value"] == "0.711"
    protocols = build_protocols(SNAPSHOT, _tasks())
    assert len(protocols) == 33
    by_id = {row["evaluation_id"]: row for row in protocols}
    assert (
        by_id["plism.h0mini2025.fixed_staining_cross_scanner.top10_accuracy"]["task_identity_id"]
        == by_id["plism.repo2025.fixed_staining_cross_scanner.top10_accuracy"]["task_identity_id"]
    )
    assert (
        by_id["plism.h0mini2025.fixed_staining_cross_scanner.top10_accuracy"]["protocol_id"]
        != by_id["plism.repo2025.fixed_staining_cross_scanner.top10_accuracy"]["protocol_id"]
    )
