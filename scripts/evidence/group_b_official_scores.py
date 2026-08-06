"""Registry adapter for the pinned Group B official-source snapshots."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


EXPECTED = {
    "genbio_pathfm_official_2026.csv": Counter({"canonical_candidate": 15, "canonical_candidate_analysis_ineligible": 6, "duplicate_alternate_evidence": 12, "fine_tuned_excluded": 10, "aggregate_excluded": 6}),
    "midnight_miccai2025_official_scores.csv": Counter({"canonical_candidate": 24, "fine_tuned_excluded": 12, "aggregate_excluded": 6}),
    "openmidnight_technical_report_2025.csv": Counter({"canonical_candidate": 12, "aggregate_excluded": 2, "narrative_conflict_excluded": 2}),
}

THUNDER_BASE = {
    "BACH": "thunder.bach.linear_probing", "BRACS": "thunder.bracs.linear_probing",
    "BreakHis": "thunder.break_his.linear_probing", "CCRCC": "thunder.ccrcc.linear_probing",
    "CRC-100K": "thunder.crc.linear_probing", "ESCA": "thunder.esca.linear_probing",
    "MHIST": "thunder.mhist.linear_probing", "PCAM": "thunder.patch_camelyon.linear_probing",
    "TCGA-CRC": "thunder.tcga_crc_msi.linear_probing", "TCGA-TILS": "thunder.tcga_tils.linear_probing",
    "TCGA-Unif": "thunder.tcga_uniform.linear_probing", "TCGA-Uniform": "thunder.tcga_uniform.linear_probing",
    "WILDS": "thunder.wilds.linear_probing",
}


def load_snapshot(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = EXPECTED[path.name]
    if Counter(row["inclusion_status"] for row in rows) != expected:
        raise ValueError(f"Group B disposition drift: {path.name}")
    for row in rows:
        row["_snapshot_name"] = path.name
    return rows


def selected(paths: list[Path]) -> list[dict[str, str]]:
    rows = [row for path in paths for row in load_snapshot(path)]
    chosen = [row for row in rows if row["inclusion_status"].startswith("canonical_candidate")]
    if len(chosen) != 57 or len({(row["model_id"], row["evaluation_id"]) for row in chosen}) != 57:
        raise ValueError("Group B selected-cell audit failed")
    return chosen


def build_protocols(paths: list[Path], tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    by_eval = {str(row["evaluation_id"]): row for row in tasks}
    output: dict[str, dict[str, object]] = {}
    for evidence in selected(paths):
        evaluation_id = evidence["evaluation_id"]
        if evaluation_id in output:
            continue
        if evidence["suite_id"] == "eva":
            base = dict(by_eval[evidence["dedup_key"]])
        elif evidence["suite_id"] == "thunder":
            base = dict(by_eval[THUNDER_BASE[evidence["task_label"]]])
        else:
            dataset = evaluation_id.split(".")[2]
            dataset = "tcga_4x4" if dataset == "tcga" else dataset
            if ".apd_" in evaluation_id:
                generic = dict(by_eval[f"pathorob.{dataset}.average_performance_drop"])
                base = generic
                base["target"] = evaluation_id.rsplit(".", 1)[-1]
                base["endpoint"] = base["target"]
            else:
                base = {
                    "task_identity_id": f"task.pathorob.{dataset}.balanced-accuracy",
                    "dataset_artifact_id": f"artifact.pathorob.{dataset}", "dataset_id": dataset,
                    "task_name": f"{dataset} balanced accuracy", "task_family": "classification_robustness",
                    "target": "balanced_accuracy", "sample_unit": "patch", "task_type": "robustness_analysis",
                    "num_samples": "not_reported", "endpoint": "balanced_accuracy",
                }
        metric = evidence["metric"]
        if evidence["value_unit"] == "percent" and metric == "balanced_accuracy":
            metric = "balanced_accuracy_percent"
        base.update({
            "suite_id": evidence["suite_id"], "evaluation_id": evaluation_id,
            "protocol_id": evaluation_id, "metric": metric, "direction": "higher",
            "protocol": f"{evidence['protocol_variant']}; embedding={evidence['embedding_recipe']}",
            "reference_url": evidence["reference_url"],
            "audit_status": "parsed_primary_source_analysis_ineligible" if evidence["inclusion_status"].endswith("analysis_ineligible") else "parsed_primary_source",
            "audit_notes": evidence["inclusion_reason"],
        })
        output[evaluation_id] = base
    if len(output) != 45:
        raise ValueError(f"expected 45 Group B protocols, found {len(output)}")
    return list(output.values())


def build_scores(paths: list[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scores = []
    aliases: dict[tuple[str, str], dict[str, object]] = {}
    for row in selected(paths):
        value = float(row["value"])
        ineligible = row["inclusion_status"].endswith("analysis_ineligible")
        metric = row["metric"]
        if row["value_unit"] == "percent" and metric == "balanced_accuracy":
            metric = "balanced_accuracy_percent"
        if ineligible:
            normalized = ""
        elif row["value_unit"] == "fraction":
            normalized = f"{value * 100:.6g}"
        else:
            normalized = f"{value:.6g}"
        scores.append({
            "model_id": row["model_id"], "reported_model_alias": row["model_alias"],
            "model_revision": row["model_revision"], "evaluation_id": row["evaluation_id"],
            "value": f"{value:.6g}", "normalized_score": normalized, "suite_id": row["suite_id"],
            "metric": metric, "reference_url": row["reference_url"], "source_locator": row["source_locator"],
            "extraction_date": "2026-08-06", "review_status": "machine_parsed_single_source",
            "uncertainty": "not_reported",
            "lineage": f"group-b:{row['_snapshot_name']}@sha256:{row['source_sha256']} -> {row['source_revision']} -> {row['source_locator']}",
            "audit_status": "parsed_primary_source_analysis_ineligible" if ineligible else "parsed_primary_source",
        })
        aliases[(row["suite_id"], row["model_alias"])] = {
            "alias": row["model_alias"], "model_id": row["model_id"], "suite_id": row["suite_id"],
            "reference_url": row["reference_url"], "audit_notes": "Exact component alias from the selected official Group B source row.",
        }
    return scores, list(aliases.values())
