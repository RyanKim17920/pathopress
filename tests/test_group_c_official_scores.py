from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from scripts.evidence.group_c_official_scores import build_protocols, build_scores


ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "source_data/virchow2g_gigapath_titan_official_scores_2024_2025.csv"
QUARANTINE = ROOT / "source_data/virchow2g_gigapath_titan_official_quarantine_2024_2025.csv"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_public_cell_inventory_and_component_boundaries() -> None:
    rows = _read(SCORES)
    assert len(rows) == 737
    assert Counter(row["source_id"] for row in rows) == {
        "virchow2g2024": 108,
        "titan2025": 629,
    }
    assert Counter(row["model_id"] for row in rows) == {
        "virchow-2g": 36,
        "virchow-2": 36,
        "virchow": 36,
        "titan-v-slide": 296,
        "titan-slide": 333,
    }
    assert len({(row["model_id"], row["evaluation_id"]) for row in rows}) == 737
    assert {row["suite_id"] for row in rows} == {"virchow2g_paper", "titan_paper"}


def test_virchow_embedding_and_hest_protocols_are_not_collapsed() -> None:
    rows = [row for row in _read(SCORES) if row["source_id"] == "virchow2g2024"]
    assert Counter(row["table"] for row in rows) == {"Table 2": 48, "Table 3": 60}
    assert not any(row["dataset_id"] == "panmsk" for row in rows)
    by_key = {(row["model_alias"], row["task_name"]): row for row in rows}
    assert by_key[("Virchow2G CLS+Mean", "MIDOG")]["value"] == "0.836"
    assert by_key[("Virchow2G CLS-Only", "MIDOG")]["value"] == "0.805"
    assert by_key[("Virchow2G CLS+Mean", "HEST IDC")]["value"] == "0.559"
    assert by_key[("Virchow2G CLS-Only", "HEST IDC")]["value"] == "0.547"
    assert by_key[("Virchow2 CLS+Mean", "MIDOG")]["value"] == "0.804"
    assert by_key[("Virchow CLS-Only", "MIDOG")]["value"] == "0.760"
    assert by_key[("Virchow2 CLS-Only", "HEST IDC")]["value"] == "0.563"
    assert by_key[("Virchow CLS+Mean", "HEST IDC")]["value"] == "0.545"
    assert all("random_forest" in row["protocol"] for row in rows if row["table"] == "Table 3")
    assert all("ridge" not in row["protocol"] and "pca" not in row["protocol"] for row in rows)


def test_titan_full_few_zero_shot_and_retrieval_protocols_are_distinct() -> None:
    rows = [row for row in _read(SCORES) if row["source_id"] == "titan2025"]
    assert Counter(row["endpoint"] for row in rows) == {
        "slide_level_classification": 567,
        "slide_retrieval": 40,
        "survival_prediction": 12,
        "cross_modal_retrieval": 10,
    }
    assert not any(row["dataset_id"] == "ot108" for row in rows)
    assert not any("finetun" in row["protocol"] for row in rows)
    by_locator = {row["source_locator"]: row for row in rows}
    assert by_locator[
        "paper=TITAN|supplementary_table=22|model=TITAN|evaluator=logistic_regression|metric=balanced_accuracy"
    ]["value"] == "0.832"
    assert by_locator[
        "paper=TITAN|supplementary_table=97|model=TITAN|protocol=zero-shot|metric=balanced_accuracy"
    ]["value"] == "0.761"
    assert by_locator[
        "paper=TITAN|supplementary_table=123|model=TITAN|metric=top1_accuracy"
    ]["value"] == "0.750"
    assert by_locator[
        "paper=TITAN|supplementary_table=43|cohort=EBRAINS|model=TITAN|evaluator=logistic_regression|metric=auroc"
    ]["value"] == "0.960"
    assert by_locator[
        "paper=TITAN|supplementary_table=64|cohort=BRCA|model=TITAN|metric=concordance_index"
    ]["value"] == "0.757"
    assert by_locator[
        "paper=TITAN|supplementary_table=119|model=TITAN|metric=top1_accuracy"
    ]["value"] == "0.555"
    assert by_locator[
        "paper=TITAN|supplementary_table=126|model=TITAN|metric=recall_at_1"
    ]["value"] == "0.784"


def test_quarantine_accounts_for_private_finetuned_and_aggregate_cells() -> None:
    rows = _read(QUARANTINE)
    assert len(rows) == 346
    assert Counter(row["source_id"] for row in rows) == {
        "virchow2g2024": 36,
        "provgigapath2024": 26,
        "titan2025": 284,
    }
    assert Counter(row["quarantine_reason"] for row in rows) == {
        "internal_private_cohort": 278,
        "aggregate_not_leaf_task": 20,
        "internal_private_cohort_and_finetuning": 21,
        "task_specific_finetuning": 23,
        "aggregate_not_leaf_task_and_mixes_internal_cohort": 4,
    }
    giga = [row for row in rows if row["source_id"] == "provgigapath2024"]
    assert len(giga) == 26
    assert sum(row["dataset_id"] == "tcga-luad" for row in giga) == 5
    by_task = {row["task_name"]: row for row in giga}
    assert by_task["LUAD EGFR"]["value"] == "0.543"
    assert by_task["LUAD EGFR (TCGA)"]["value"] == "0.766"


def test_primary_source_hashes_and_revisions_are_pinned() -> None:
    scores = _read(SCORES)
    quarantine = _read(QUARANTINE)
    assert {row["source_sha256"] for row in scores if row["source_id"] == "virchow2g2024"} == {
        "41054dcfa720f5da2c933cb2a711c9d4618689a990513236b256652865418125"
    }
    assert {row["source_sha256"] for row in scores if row["source_id"] == "titan2025"} == {
        "26321e4018bec7b80f2fe7ea7cc497139c83b44fb60df5128417623ad1f71a70"
    }
    assert {row["source_sha256"] for row in quarantine if row["source_id"] == "provgigapath2024"} == {
        "b8279191eb635a60edd51621eab931f04d8d75b7af291794d8b91b58f73b866b"
    }
    titan_revisions = {row["model_revision"] for row in scores if row["source_id"] == "titan2025"}
    assert titan_revisions == {"github:mahmoodlab/TITAN@9e34c66ff66445c6c590da0dbf7acc103d39a40b"}
    virchow_revisions = {row["model_revision"] for row in scores if row["source_id"] == "virchow2g2024"}
    assert virchow_revisions == {"paper_only_unreleased_checkpoint", "paper_reported_checkpoint"}


def test_registry_adapter_preserves_every_cell_and_metric_scale() -> None:
    protocols = build_protocols(SCORES)
    scores, aliases = build_scores(SCORES)
    assert len(protocols) == 369
    assert len(scores) == 737
    assert len(aliases) == 8
    by_locator = {row["source_locator"]: row for row in scores}
    assert by_locator["paper=Virchow2|table=3|model=Virchow2G|embedding=CLS+Mean|task=IDC"]["normalized_score"] == "77.95"
    assert by_locator[
        "paper=TITAN|supplementary_table=22|model=TITAN|evaluator=logistic_regression|metric=balanced_accuracy"
    ]["normalized_score"] == "83.2"
    assert by_locator[
        "paper=TITAN|supplementary_table=64|cohort=BRCA|model=TITAN|metric=concordance_index"
    ]["normalized_score"] == "75.7"
