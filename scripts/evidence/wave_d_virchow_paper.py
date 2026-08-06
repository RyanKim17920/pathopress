"""Pinned Virchow primary-paper table adapter.

Only public frozen-feature tile-classification leaves are materialized.  The
internal PanMSK and non-downloadable clinical biomarker cohorts remain in the
snapshot as quarantined evidence.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


EXPECTED = Counter({"accepted_public_leaf": 15, "quarantined_internal_cohort": 6})
SOURCE_SHA256 = "b2c31918a05d22534f38e35c5fc2cf3704439f7eb30af637dfd1e698373ba6b0"


def load_evidence(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 21 or Counter(row["disposition"] for row in rows) != EXPECTED:
        raise ValueError("Virchow paper evidence audit failed")
    if {row["source_sha256"] for row in rows} != {SOURCE_SHA256}:
        raise ValueError("Virchow paper archive hash drift")
    return rows


def build_protocols(path: Path) -> list[dict[str, object]]:
    public = [row for row in load_evidence(path) if row["disposition"] == "accepted_public_leaf"]
    output: list[dict[str, object]] = []
    for evidence in sorted(public, key=lambda row: row["evaluation_id"]):
        output.append({
            "task_identity_id": f"task.virchow2paper.{evidence['dataset_id']}.tile_classification",
            "dataset_artifact_id": f"artifact.virchow2paper.{evidence['dataset_id']}",
            "suite_id": "virchow_paper",
            "dataset_id": evidence["dataset_id"],
            "task_name": evidence["dataset_id"],
            "task_family": "tile_classification",
            "target": "tile_classification",
            "sample_unit": "image",
            "task_type": "classification",
            "num_samples": "not_reported",
            "evaluation_id": evidence["evaluation_id"],
            "protocol_id": evidence["evaluation_id"],
            "endpoint": "linear_probe",
            "metric": evidence["metric"],
            "direction": "higher",
            "protocol": evidence["protocol"],
            "reference_url": evidence["reference_url"],
            "audit_status": "parsed_primary_source",
            "audit_notes": evidence["disposition_reason"],
        })
    if len(output) != 15 or len({row["protocol_id"] for row in output}) != 15:
        raise ValueError(f"expected 15 Virchow protocols, found {len(output)}")
    return output


def build_scores(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    public = [row for row in load_evidence(path) if row["disposition"] == "accepted_public_leaf"]
    scores = []
    for row in public:
        value = float(row["value"])
        scores.append({
            "model_id": row["model_id"],
            "reported_model_alias": row["model_alias"],
            "model_revision": "not_reported",
            "evaluation_id": row["evaluation_id"],
            "value": f"{value:.6g}",
            "normalized_score": f"{value * 100:.6g}",
            "suite_id": "virchow_paper",
            "metric": row["metric"],
            "reference_url": row["reference_url"],
            "source_locator": row["source_locator"],
            "extraction_date": "2026-08-06",
            "review_status": "machine_parsed_single_source",
            "uncertainty": row["uncertainty"],
            "lineage": f"arxiv:2309.07778 source@sha256:{row['source_sha256']} -> wave_d_virchow_paper_2309.07778.csv -> build_registry.py -> scores.csv",
            "audit_status": "parsed_primary_source",
        })
    aliases = [{
        "alias": "Virchow",
        "model_id": "virchow",
        "suite_id": "virchow_paper",
        "reference_url": "https://arxiv.org/abs/2309.07778",
        "audit_notes": "Exact Virchow primary-paper model row under the frozen CLS+Mean linear-probe protocol.",
    }]
    return scores, aliases
