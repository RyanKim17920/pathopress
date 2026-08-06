"""Pinned H0-mini/UNI2-h score evidence and registry adapters.

The committed CSV is the audit boundary: public leaf cells, derived aggregates,
and private/internal cells are all retained there with explicit dispositions.
Only public leaf cells are materialized as registry scores.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


FIELDS = {
    "source_id", "source_revision", "source_sha256", "source_locator",
    "model_alias", "model_id", "suite_id", "evaluation_id",
    "base_evaluation_id", "dataset_id", "task_name", "task_family",
    "target", "sample_unit", "task_type", "endpoint", "metric",
    "direction", "value", "uncertainty", "protocol", "reference_url",
    "disposition", "disposition_reason",
}

EXPECTED_SOURCE_REVISIONS = {
    "h0mini_arxiv_v3": (
        "arxiv:2501.16239v3",
        "222798059c15b554528d61f8caa04de8fcc2d5cc23997607dc25d851282a6f08",
    ),
    "plism_repo": (
        "git:5ec9511893af993f6faa099f093d1924b291aed2",
        "d1715234a41f8da728ad669560bbcfc5253680db3403b400673d6a40a3955a64",
    ),
    "uni_official_repo": (
        "git:42715efc11722a496e0a67f3369505a8f277206c",
        "4ac024c83dbcdc39987a81f4983474b0e6c6f15352226677809a6fb492f9cdb8",
    ),
}

EXPECTED_DISPOSITIONS = Counter({
    "accepted_public_leaf": 54,
    "accepted_public_metric_unspecified": 6,
    "excluded_derived_aggregate": 7,
    "quarantined_private_cohort": 19,
})


def load_evidence(snapshot: Path) -> list[dict[str, str]]:
    with snapshot.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != FIELDS:
        raise ValueError("unexpected H0-mini/UNI2-h evidence schema")
    if len(rows) != 86 or Counter(row["disposition"] for row in rows) != EXPECTED_DISPOSITIONS:
        raise ValueError("H0-mini/UNI2-h evidence disposition audit failed")
    for source_id, (revision, digest) in EXPECTED_SOURCE_REVISIONS.items():
        source_rows = [row for row in rows if row["source_id"] == source_id]
        if not source_rows:
            raise ValueError(f"missing evidence source: {source_id}")
        if {row["source_revision"] for row in source_rows} != {revision}:
            raise ValueError(f"source revision drift: {source_id}")
        if {row["source_sha256"] for row in source_rows} != {digest}:
            raise ValueError(f"source hash drift: {source_id}")
    public = [row for row in rows if row["disposition"].startswith("accepted_public")]
    if len(public) != 60 or len({(row["model_id"], row["evaluation_id"]) for row in public}) != 60:
        raise ValueError("public H0-mini/UNI2-h score-cell audit failed")
    return rows


def public_rows(snapshot: Path) -> list[dict[str, str]]:
    return [
        row for row in load_evidence(snapshot)
        if row["disposition"] in {"accepted_public_leaf", "accepted_public_metric_unspecified"}
    ]


def build_protocols(snapshot: Path, tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    """Materialize 33 source-versioned protocols without merging score versions."""
    existing = {str(row["evaluation_id"]): row for row in tasks}
    unique: dict[str, dict[str, str]] = {}
    for evidence in public_rows(snapshot):
        previous = unique.setdefault(evidence["evaluation_id"], evidence)
        invariant = (
            "base_evaluation_id", "suite_id", "dataset_id", "task_name", "target",
            "sample_unit", "task_type", "endpoint", "metric", "direction", "protocol",
        )
        if any(previous[key] != evidence[key] for key in invariant):
            raise ValueError(f"inconsistent protocol metadata: {evidence['evaluation_id']}")

    output: list[dict[str, object]] = []
    for evaluation_id, evidence in sorted(unique.items()):
        base_id = evidence["base_evaluation_id"]
        if base_id:
            if base_id not in existing:
                raise ValueError(f"missing base protocol: {base_id}")
            row = dict(existing[base_id])
        else:
            row = {
                "task_identity_id": f"task.{evidence['suite_id']}.{evidence['dataset_id']}.{evidence['metric']}",
                "dataset_artifact_id": (
                    "artifact.plism.processed_public_v1"
                    if evidence["suite_id"] == "plism"
                    else f"artifact.{evidence['suite_id']}.{evidence['dataset_id']}"
                ),
                "suite_id": evidence["suite_id"],
                "dataset_id": evidence["dataset_id"],
                "task_name": evidence["task_name"],
                "task_family": evidence["task_family"],
                "target": evidence["target"],
                "sample_unit": evidence["sample_unit"],
                "task_type": evidence["task_type"],
                "num_samples": "not_reported",
            }
        row.update({
            "evaluation_id": evaluation_id,
            "protocol_id": evaluation_id,
            "suite_id": evidence["suite_id"],
            "dataset_id": evidence["dataset_id"],
            "task_name": evidence["task_name"],
            "task_family": evidence["task_family"],
            "target": evidence["target"],
            "sample_unit": evidence["sample_unit"],
            "task_type": evidence["task_type"],
            "endpoint": evidence["endpoint"],
            "metric": evidence["metric"],
            "direction": evidence["direction"],
            "protocol": evidence["protocol"],
            "reference_url": evidence["reference_url"],
            "audit_status": (
                "parsed_primary_source_analysis_ineligible"
                if evidence["disposition"] == "accepted_public_metric_unspecified"
                else "parsed_primary_source"
            ),
            "audit_notes": evidence["disposition_reason"],
        })
        output.append(row)

    # The refreshed repository top-10 values are protocol variants of the same
    # three biological robustness tasks reported in the paper, not new tasks.
    for context in (
        "fixed_staining_cross_scanner", "fixed_scanner_cross_staining",
        "cross_staining_cross_scanner",
    ):
        identity = f"task.plism.{context}.top10_accuracy"
        for evaluation_id in (
            f"plism.h0mini2025.{context}.top10_accuracy",
            f"plism.repo2025.{context}.top10_accuracy",
        ):
            next(row for row in output if row["evaluation_id"] == evaluation_id)["task_identity_id"] = identity
    if len(output) != 33:
        raise ValueError(f"expected 33 versioned protocols, found {len(output)}")
    return output


def _normalized(row: dict[str, str], value: float) -> float | None:
    if row["disposition"] == "accepted_public_metric_unspecified":
        return None
    if row["metric"] == "pearson_r":
        return (value + 1.0) * 50.0
    if row["metric"] in {
        "balanced_accuracy", "dice", "cosine_similarity", "top10_accuracy",
    }:
        return value * 100.0
    raise ValueError(f"no normalization contract for metric: {row['metric']}")


def build_scores(snapshot: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scores: list[dict[str, object]] = []
    aliases: dict[tuple[str, str], dict[str, object]] = {}
    for row in public_rows(snapshot):
        value = float(row["value"])
        normalized = _normalized(row, value)
        audit = (
            "parsed_primary_source_analysis_ineligible"
            if normalized is None else "parsed_primary_source"
        )
        source_file = (
            "source archive" if row["source_id"] == "h0mini_arxiv_v3" else "README.md"
        )
        scores.append({
            "model_id": row["model_id"],
            "reported_model_alias": row["model_alias"],
            "model_revision": (
                "hf:d517a8dd47902dd7c308b3c36f63bce47e7b9a43"
                if row["model_id"] == "uni2-h"
                else "hf:5b5cc0505d19ae558270045eb0df8c34df4d9609"
            ),
            "evaluation_id": row["evaluation_id"],
            "value": f"{value:.6g}",
            "normalized_score": "" if normalized is None else f"{normalized:.6g}",
            "suite_id": row["suite_id"],
            "metric": row["metric"],
            "reference_url": row["reference_url"],
            "source_locator": row["source_locator"],
            "extraction_date": "2026-08-06",
            "review_status": "machine_parsed_single_source",
            "uncertainty": row["uncertainty"],
            "lineage": (
                f"{row['source_revision']} {source_file}@sha256:{row['source_sha256']} "
                f"-> source_data/h0mini_uni2h_official_scores_2025.csv -> build_registry.py -> scores.csv"
            ),
            "audit_status": audit,
        })
        aliases[(row["suite_id"], row["model_alias"])] = {
            "alias": row["model_alias"],
            "model_id": row["model_id"],
            "suite_id": row["suite_id"],
            "reference_url": row["reference_url"],
            "audit_notes": "Exact model row in a pinned official paper or repository report.",
        }
    if len(scores) != 60:
        raise ValueError("expected 60 public score cells")
    return scores, list(aliases.values())
