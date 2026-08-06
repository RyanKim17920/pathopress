"""Registry adapters for the pinned group-C primary-paper score extract."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


EXPECTED_FIELDS = {
    "source_id", "suite_id", "table", "dataset_id", "task_name", "task_family",
    "target", "sample_unit", "task_type", "num_samples", "endpoint", "metric",
    "protocol", "model_alias", "model_id", "model_revision", "value", "uncertainty",
    "evaluation_id", "protocol_id", "task_identity_id", "dataset_artifact_id",
    "reference_url", "source_locator", "source_sha256", "review_status", "audit_status",
    "audit_notes",
}
EXPECTED_HASHES = {
    "virchow2g2024": "41054dcfa720f5da2c933cb2a711c9d4618689a990513236b256652865418125",
    "titan2025": "26321e4018bec7b80f2fe7ea7cc497139c83b44fb60df5128417623ad1f71a70",
}


def load_group_c_official_scores(snapshot: Path) -> list[dict[str, str]]:
    with snapshot.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != EXPECTED_FIELDS:
        raise ValueError("unexpected group-C score schema")
    if len(rows) != 737 or Counter(row["source_id"] for row in rows) != {
        "virchow2g2024": 108, "titan2025": 629,
    }:
        raise ValueError("group-C score inventory drift")
    for source_id, digest in EXPECTED_HASHES.items():
        if {row["source_sha256"] for row in rows if row["source_id"] == source_id} != {digest}:
            raise ValueError(f"group-C source hash drift: {source_id}")
    if len({(row["model_id"], row["evaluation_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate group-C model/evaluation cell")
    return rows


def build_protocols(snapshot: Path) -> list[dict[str, object]]:
    unique: dict[str, dict[str, str]] = {}
    for row in load_group_c_official_scores(snapshot):
        previous = unique.setdefault(row["evaluation_id"], row)
        invariant = (
            "suite_id", "dataset_id", "task_name", "task_family", "target", "sample_unit",
            "task_type", "num_samples", "endpoint", "metric", "protocol", "task_identity_id",
            "dataset_artifact_id",
        )
        if any(previous[key] != row[key] for key in invariant):
            raise ValueError(f"inconsistent group-C protocol metadata: {row['evaluation_id']}")
    output = []
    for evaluation_id, row in sorted(unique.items()):
        output.append({
            "evaluation_id": evaluation_id,
            "protocol_id": row["protocol_id"],
            "task_identity_id": row["task_identity_id"],
            "dataset_artifact_id": row["dataset_artifact_id"],
            "suite_id": row["suite_id"],
            "dataset_id": row["dataset_id"],
            "task_name": row["task_name"],
            "task_family": row["task_family"],
            "target": row["target"],
            "sample_unit": row["sample_unit"],
            "task_type": row["task_type"],
            "num_samples": row["num_samples"],
            "endpoint": row["endpoint"],
            "metric": row["metric"],
            "direction": "higher",
            "protocol": row["protocol"],
            "reference_url": row["reference_url"],
            "audit_status": row["audit_status"],
            "audit_notes": row["audit_notes"],
        })
    if len(output) != 369:
        raise ValueError(f"expected 369 group-C protocols, found {len(output)}")
    return output


def _normalized(metric: str, value: float) -> float:
    if metric == "pearson_r":
        return (value + 1.0) * 50.0
    if metric in {
        "weighted_f1", "balanced_accuracy", "auroc", "top1_accuracy", "top3_accuracy",
        "top5_accuracy", "majority_vote_at_3_accuracy", "majority_vote_at_5_accuracy",
        "quadratic_weighted_kappa", "concordance_index", "recall_at_1", "recall_at_3",
        "recall_at_5", "recall_at_10", "mean_recall",
    }:
        return value * 100.0
    raise ValueError(f"no group-C normalization contract for {metric}")


def build_scores(snapshot: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scores = []
    aliases: dict[tuple[str, str], dict[str, object]] = {}
    for row in load_group_c_official_scores(snapshot):
        value = float(row["value"])
        scores.append({
            "model_id": row["model_id"],
            "reported_model_alias": row["model_alias"],
            "model_revision": row["model_revision"],
            "evaluation_id": row["evaluation_id"],
            "value": f"{value:.6g}",
            "normalized_score": f"{_normalized(row['metric'], value):.6g}",
            "suite_id": row["suite_id"],
            "metric": row["metric"],
            "reference_url": row["reference_url"],
            "source_locator": row["source_locator"],
            "extraction_date": "2026-08-06",
            "review_status": row["review_status"],
            "uncertainty": row["uncertainty"],
            "lineage": (
                f"official PDF@sha256:{row['source_sha256']} -> "
                "scripts/extract_group_c_official_scores.py -> "
                "source_data/virchow2g_gigapath_titan_official_scores_2024_2025.csv -> "
                "build_registry.py -> scores.csv"
            ),
            "audit_status": row["audit_status"],
        })
        aliases[(row["suite_id"], row["model_alias"])] = {
            "alias": row["model_alias"], "model_id": row["model_id"],
            "suite_id": row["suite_id"], "reference_url": row["reference_url"],
            "audit_notes": "Exact component and protocol label from the official primary paper.",
        }
    return scores, list(aliases.values())
