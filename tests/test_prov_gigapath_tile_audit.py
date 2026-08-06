from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "source_data/prov_gigapath_tile_evidence_audit_2024_2026.csv"


def rows() -> list[dict[str, str]]:
    with AUDIT.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_inventory_is_exhaustive_and_has_no_claimed_eligible_leaf() -> None:
    evidence = rows()
    assert len(evidence) == 11
    assert len({row["evidence_id"] for row in evidence}) == 11
    assert {row["eligible_public_leaf"] for row in evidence} == {"no"}
    assert Counter(row["source"] for row in evidence) == {
        "Nature supplement": 3,
        "Nature main article": 3,
        "Official GitHub repository": 3,
        "Official Hugging Face model card": 1,
        "Official Hugging Face dataset": 1,
    }


def test_public_numeric_evidence_is_not_misattributed_to_tile_encoder() -> None:
    evidence = {row["evidence_id"]: row for row in rows()}
    table = evidence["nature_supp_table2_tcga"]
    assert table["dataset_access"] == "public_tcga_luad"
    assert table["exact_numeric_result"] == "five_exact_auroc_means_with_standard_errors"
    assert table["downstream_component"] == "task-specifically fine-tuned LongNet slide encoder plus ABMIL/classifier"
    assert table["exclusion_reason"] == "task_specific_slide_encoder_finetuning"
    fig4 = evidence["nature_supp_fig4"]
    assert fig4["exact_numeric_result"] == "graphical_bars_without_exact_labels"
    assert fig4["exclusion_reason"] == "not_tile_encoder_only_and_no_exact_numeric_table"


def test_runnable_pcam_materials_do_not_claim_an_official_score() -> None:
    evidence = {row["evidence_id"]: row for row in rows()}
    for evidence_id in ("repo_pcam_example", "repo_linear_probe", "hf_model_card_pcam", "hf_embeddings_pcam"):
        assert evidence[evidence_id]["eligible_public_leaf"] == "no"
        assert evidence[evidence_id]["exact_numeric_result"] in {"none_published", "none_checked_in"}
    assert evidence["hf_embeddings_pcam"]["pinned_revision"] == "hf:0c5eebef6041f990eb04e1f6f9d835958b0b941f"
    assert evidence["hf_model_card_pcam"]["pinned_revision"] == "hf:0dd9f5561e6b98d27c1d7a919c252d75c2eb66cb"


def test_primary_sources_and_repo_are_pinned() -> None:
    revisions = {row["pinned_revision"] for row in rows()}
    assert "sha256:52eda6b291c49c85274f034462e3e0d168d8bca505d845bed648fabae4192b76" in revisions
    assert "sha256:b8279191eb635a60edd51621eab931f04d8d75b7af291794d8b91b58f73b866b" in revisions
    assert "git:92d8d20430f9645675db01c5f55e95ad58538526" in revisions
