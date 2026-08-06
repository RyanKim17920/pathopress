"""Registry adapter for the pinned first-party H-optimus-1 benchmark report."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


EXPECTED = Counter({
    "accepted_public_leaf": 95,
    "quarantined_nonpublic_or_access_ambiguous": 60,
    "excluded_derived_aggregate": 15,
})

TILE_TASK_BASES = {
    "cam17_wilds": "thunder.wilds.linear_probing",
    "crc_no_norm": "thunder.crc.linear_probing",
    "mhist": "thunder.mhist.linear_probing",
    "tcga_uniform": "thunder.tcga_uniform.linear_probing",
}


def load_evidence(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 170 or Counter(row["disposition"] for row in rows) != EXPECTED:
        raise ValueError("Wave D H-optimus report evidence audit failed")
    if {row["source_sha256"] for row in rows} != {
        "a46ad75f2194ac2700eedfde6cef0edde5b56ff59438fa724cf53e4adc997bf3"
    }:
        raise ValueError("Wave D H-optimus report source hash drift")
    public = [row for row in rows if row["disposition"] == "accepted_public_leaf"]
    if len({(row["model_id"], row["evaluation_id"]) for row in public}) != 95:
        raise ValueError("duplicate Wave D public score cell")
    return rows


def build_protocols(path: Path, tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    by_id = {str(row["evaluation_id"]): row for row in tasks}
    public = [row for row in load_evidence(path) if row["disposition"] == "accepted_public_leaf"]
    unique = {row["evaluation_id"]: row for row in public}
    output: list[dict[str, object]] = []
    for evaluation_id, evidence in sorted(unique.items()):
        if evaluation_id.startswith("hest.hoptimus1report2025."):
            organ = evaluation_id.split(".")[2]
            base = dict(by_id[f"hest.{organ}.gene_expression"])
        elif evidence["dataset_id"] in TILE_TASK_BASES:
            base = dict(by_id[TILE_TASK_BASES[evidence["dataset_id"]]])
        else:
            base = {
                "task_identity_id": f"task.hoptimus1report.{evaluation_id.removeprefix('hoptimus1report2025.')}",
                "dataset_artifact_id": f"artifact.hoptimus1report.{evidence['dataset_id']}",
                "suite_id": "hoptimus1_report",
                "dataset_id": evidence["dataset_id"],
                "task_name": evaluation_id.removeprefix("hoptimus1report2025."),
                "task_family": evidence["task_family"],
                "target": evidence["task_family"],
                "sample_unit": "slide" if evidence["task_family"].startswith("slide") else "image",
                "task_type": "regression" if evidence["metric"] == "pearson_r" else "classification",
                "num_samples": "not_reported",
            }
        base.update({
            "suite_id": "hest" if evaluation_id.startswith("hest.") else "hoptimus1_report",
            "evaluation_id": evaluation_id,
            "protocol_id": evaluation_id,
            "endpoint": "hoptimus1_report_2025_frozen_features",
            "metric": evidence["metric"],
            "direction": "higher",
            "protocol": evidence["protocol"],
            "reference_url": evidence["reference_url"],
            "audit_status": "parsed_primary_source",
            "audit_notes": evidence["disposition_reason"],
        })
        output.append(base)
    if len(output) != 19:
        raise ValueError(f"expected 19 report protocols, found {len(output)}")
    return output


def build_scores(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    public = [row for row in load_evidence(path) if row["disposition"] == "accepted_public_leaf"]
    scores, aliases = [], {}
    for row in public:
        value = float(row["value"])
        normalized = (value + 1.0) * 50.0 if row["metric"] == "pearson_r" else value * 100.0
        scores.append({
            "model_id": row["model_id"], "reported_model_alias": row["model_alias"],
            "model_revision": "not_reported", "evaluation_id": row["evaluation_id"],
            "value": f"{value:.6g}", "normalized_score": f"{normalized:.6g}",
            "suite_id": "hest" if row["evaluation_id"].startswith("hest.") else "hoptimus1_report",
            "metric": row["metric"], "reference_url": row["reference_url"],
            "source_locator": row["source_locator"], "extraction_date": "2026-08-06",
            "review_status": "machine_parsed_single_source", "uncertainty": row["uncertainty"],
            "lineage": f"official Bioptimus report@sha256:{row['source_sha256']} -> wave_d_hoptimus1_official_report_2025.csv -> build_registry.py -> scores.csv",
            "audit_status": "parsed_primary_source",
        })
        suite = "hest" if row["evaluation_id"].startswith("hest.") else "hoptimus1_report"
        aliases[(suite, row["model_alias"])] = {
            "alias": row["model_alias"], "model_id": row["model_id"], "suite_id": suite,
            "reference_url": row["reference_url"],
            "audit_notes": "Exact model row in the first-party H-optimus-1 benchmark report.",
        }
    return scores, list(aliases.values())
