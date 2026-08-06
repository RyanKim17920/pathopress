"""Pinned exhaustive UNI primary-paper leaf-cell adapter."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


EXPECTED = Counter({
    "accepted_public_leaf": 227,
    "quarantined_finetuned": 24,
    "quarantined_internal_cohort": 23,
    "excluded_derived_aggregate": 3,
})
SOURCE_SHA256 = "26da25ced22b205570f480c3a562af31ed74bdaf679455f30cbcaa5dfdad4e60"

# Exact artifact/target matches already represented in the canonical registry.
EXACT_IDENTITIES = {
    ("crc100k", "crc_tissue_9class"): "task.crc100k.nine_class_classification",
    ("tcga_hel_ccrcc", "ccrcc_tissue_3class"): "task.thunder.ccrcc.ccrcc-linear-probing",
    ("bach", "brca_subtyping_4class"): "task.bach.four_class_classification",
    ("unitopatho", "crc_polyp_6class"): "task.eva.unitopatho.unitopatho-classification",
    ("tcga_msi", "crc_msi_2class"): "task.thunder.tcga_crc_msi.tcga-crc-msi-linear-probing",
    ("tcga_uniform", "pan_cancer_tissue_32class"): "task.thunder.tcga_uniform.tcga-uniform-linear-probing",
    ("tcga_tils", "til_detection_2class"): "task.thunder.tcga_tils.tcga-tils-linear-probing",
    ("camelyon16", "breast_metastasis_2class"): "task.eva.camelyon16.camelyon16-classification",
    ("bracs", "brca_coarse_3class"): "task.pathobench.bracs.slidelevel-coarse",
    ("bracs", "brca_fine_7class"): "task.pathobench.bracs.slidelevel-fine",
    ("panda", "isup_grading_6class"): "task.panda.isup_grade",
}


def load_evidence(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 277 or Counter(row["disposition"] for row in rows) != EXPECTED:
        raise ValueError("UNI paper evidence audit failed")
    if {row["source_sha256"] for row in rows} != {SOURCE_SHA256}:
        raise ValueError("UNI paper archive hash drift")
    if len({row["evaluation_id"] for row in rows}) != len(rows):
        raise ValueError("UNI paper evaluation ids are not unique")
    return rows


def build_protocols(path: Path, tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    by_identity = {str(row["task_identity_id"]): row for row in tasks}
    public = [row for row in load_evidence(path) if row["disposition"] == "accepted_public_leaf"]
    output: list[dict[str, object]] = []
    for evidence in sorted(public, key=lambda row: row["evaluation_id"]):
        exact = EXACT_IDENTITIES.get((evidence["dataset_id"], evidence["task_family"]))
        if exact and exact in by_identity:
            base = dict(by_identity[exact])
        else:
            base = {
                "task_identity_id": exact or f"task.unipaper.{evidence['dataset_id']}.{evidence['task_family']}",
                "dataset_artifact_id": f"artifact.unipaper.{evidence['dataset_id']}",
                "dataset_id": evidence["dataset_id"],
                "task_name": evidence["task_family"],
                "task_family": "classification",
                "target": evidence["task_family"],
                "sample_unit": "slide" if any(token in evidence["evaluation_id"] for token in ("slide_", "proto_")) else "image",
                "task_type": "classification",
                "num_samples": "not_reported",
            }
        endpoint = evidence["protocol"].split(" on ", 1)[0]
        base.update({
            "suite_id": "uni_paper",
            "evaluation_id": evidence["evaluation_id"],
            "protocol_id": evidence["evaluation_id"],
            "endpoint": endpoint,
            "metric": evidence["metric"],
            "direction": "higher",
            "protocol": evidence["protocol"],
            "reference_url": evidence["reference_url"],
            "audit_status": "parsed_primary_source",
            "audit_notes": evidence["disposition_reason"],
        })
        output.append(base)
    if len(output) != 227 or len({row["protocol_id"] for row in output}) != 227:
        raise ValueError(f"expected 227 UNI paper protocols, found {len(output)}")
    return output


def build_scores(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    public = [row for row in load_evidence(path) if row["disposition"] == "accepted_public_leaf"]
    scores = []
    for row in public:
        value = float(row["value"])
        normalized = (value + 1) * 50 if row["metric"] == "weighted_kappa" else value * 100
        scores.append({
            "model_id": "uni", "reported_model_alias": "UNI", "model_revision": "not_reported",
            "evaluation_id": row["evaluation_id"], "value": f"{value:.6g}",
            "normalized_score": f"{normalized:.6g}", "suite_id": "uni_paper",
            "metric": row["metric"], "reference_url": row["reference_url"],
            "source_locator": row["source_locator"], "extraction_date": "2026-08-06",
            "review_status": "machine_parsed_single_source", "uncertainty": row["uncertainty"],
            "lineage": f"arxiv:2308.15474 source@sha256:{row['source_sha256']} -> wave_d_uni_paper_2308.15474.csv -> build_registry.py -> scores.csv",
            "audit_status": "parsed_primary_source",
        })
    aliases = [{
        "alias": "UNI", "model_id": "uni", "suite_id": "uni_paper",
        "reference_url": "https://arxiv.org/abs/2308.15474",
        "audit_notes": "Exact UNI row in the primary-paper supplementary task tables.",
    }]
    return scores, aliases
