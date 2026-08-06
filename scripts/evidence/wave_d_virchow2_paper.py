"""Pinned Virchow2 paper table adapter with embedding recipes kept distinct."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


EXPECTED = Counter({"accepted_public_leaf": 144, "quarantined_internal_cohort": 24, "excluded_derived_aggregate": 24})


def load_evidence(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 192 or Counter(row["disposition"] for row in rows) != EXPECTED:
        raise ValueError("Virchow2 paper evidence audit failed")
    if {row["source_sha256"] for row in rows} != {"1ee7317aa683b5c948f0c9b6e3fc588fd3a95c7f0a3db39d71511268db1f3fda"}:
        raise ValueError("Virchow2 paper archive hash drift")
    return rows


def build_protocols(path: Path, tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    by_id = {str(row["evaluation_id"]): row for row in tasks}
    public = [row for row in load_evidence(path) if row["disposition"] == "accepted_public_leaf"]
    unique = {row["evaluation_id"]: row for row in public}
    output = []
    for evaluation_id, evidence in sorted(unique.items()):
        if evaluation_id.startswith("hest.") and evidence["dataset_id"] != "hcc":
            base = dict(by_id[f"hest.{evidence['dataset_id']}.gene_expression"])
        else:
            base = {
                "task_identity_id": f"task.virchow2paper.{evidence['dataset_id']}.{evidence['task_family']}",
                "dataset_artifact_id": f"artifact.virchow2paper.{evidence['dataset_id']}",
                "suite_id": "virchow2_paper", "dataset_id": evidence["dataset_id"],
                "task_name": evidence["dataset_id"], "task_family": evidence["task_family"],
                "target": evidence["task_family"],
                "sample_unit": "spatial_transcriptomics_spot" if evidence["metric"] == "pearson_r" else "image",
                "task_type": "regression" if evidence["metric"] == "pearson_r" else "classification",
                "num_samples": "not_reported",
            }
        base.update({
            "evaluation_id": evaluation_id, "protocol_id": evaluation_id,
            "endpoint": "random_forest_gene_expression" if evidence["metric"] == "pearson_r" else "linear_probe",
            "metric": evidence["metric"], "direction": "higher", "protocol": evidence["protocol"],
            "reference_url": evidence["reference_url"], "audit_status": "parsed_primary_source",
            "audit_notes": evidence["disposition_reason"],
        })
        output.append(base)
    if len(output) != 36:
        raise ValueError(f"expected 36 Virchow2 protocols, found {len(output)}")
    return output


def build_scores(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    public = [row for row in load_evidence(path) if row["disposition"] == "accepted_public_leaf"]
    scores, aliases = [], {}
    for row in public:
        value = float(row["value"])
        normalized = (value + 1) * 50 if row["metric"] == "pearson_r" else value * 100
        suite = "hest" if row["evaluation_id"].startswith("hest.") else "virchow2_paper"
        scores.append({
            "model_id": row["model_id"], "reported_model_alias": row["model_alias"],
            "model_revision": "not_reported", "evaluation_id": row["evaluation_id"],
            "value": f"{value:.6g}", "normalized_score": f"{normalized:.6g}",
            "suite_id": suite, "metric": row["metric"], "reference_url": row["reference_url"],
            "source_locator": row["source_locator"], "extraction_date": "2026-08-06",
            "review_status": "machine_parsed_single_source", "uncertainty": "not_reported",
            "lineage": f"arxiv:2408.00738 source@sha256:{row['source_sha256']} -> wave_d_virchow2_paper_2408.00738.csv -> build_registry.py -> scores.csv",
            "audit_status": "parsed_primary_source",
        })
        aliases[(suite, row["model_alias"])] = {
            "alias": row["model_alias"], "model_id": row["model_id"], "suite_id": suite,
            "reference_url": row["reference_url"], "audit_notes": "Exact model and embedding-recipe row in the Virchow2 paper.",
        }
    return scores, list(aliases.values())
