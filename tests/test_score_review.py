import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from pathopress.score_review import read_csv, sha256_path, validate_ledger


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _inputs():
    return (
        read_csv(DATA / "scores.csv"),
        read_csv(DATA / "tasks.csv"),
        read_csv(DATA / "deduplication.csv"),
        read_csv(DATA / "score_review_ledger.csv"),
    )


def test_review_ledger_has_exact_score_coverage_and_preserves_boundaries():
    scores, tasks, dedup, ledger = _inputs()
    summary = validate_ledger(scores, tasks, dedup, ledger)
    audit_counts = Counter(row["audit_status"] for row in scores)
    assert summary["score_rows"] == len(scores) == len(ledger)
    assert summary["retained_primary_rows"] == audit_counts["parsed_primary_source"]
    assert summary["analysis_ineligible_rows"] == audit_counts[
        "parsed_primary_source_analysis_ineligible"
    ]
    assert summary["reported_external_rows"] == audit_counts["reported_external"]
    assert summary["locator_reachable_rows"] == len(scores)
    assert {row["reviewer_type"] for row in ledger} == {"automated_agent_review"}
    assert not any("human" in row["reviewer_type"] for row in ledger)


def test_review_summary_and_all_pinned_evidence_hashes_are_self_contained():
    summary = json.loads((DATA / "score_review_summary.json").read_text())
    ledger = DATA / "score_review_ledger.csv"
    assert summary["ledger_sha256"] == sha256_path(ledger)
    assert set(summary["source_files"]) == {row["evidence_path"] for row in _inputs()[3]}
    for relative, digest in summary["source_files"].items():
        source = ROOT / relative
        assert source.is_file()
        assert sha256_path(source) == digest


def test_external_rows_never_receive_a_promoted_outcome():
    scores, _, _, ledger = _inputs()
    by_key = {(row["model_id"], row["evaluation_id"]): row for row in ledger}
    external = [row for row in scores if row["audit_status"] == "reported_external"]
    assert len(external) == 9
    for score in external:
        review = by_key[(score["model_id"], score["evaluation_id"])]
        assert review["review_outcome"] == "reported_external"
        assert review["promotion_eligible"] == "false"
        assert review["canonical_setting_decision"] == "retain_reported_external_only"


def test_validator_rejects_a_locator_claim_without_reachability():
    scores, tasks, dedup, ledger = _inputs()
    ledger[0] = dict(ledger[0], source_locator_reachable="false")
    with pytest.raises(ValueError, match="locator was not reached"):
        validate_ledger(scores, tasks, dedup, ledger)
