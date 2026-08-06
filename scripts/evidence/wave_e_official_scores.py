"""Registry adapter for selected public Wave E primary-paper leaves."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED = {
    "conch_official_scores_2024.csv": Counter({"canonical_candidate": 77, "private_internal_excluded": 16, "fine_tuned_excluded": 11}),
    "conch15_titan_official_scores_2025.csv": Counter({"canonical_candidate": 297, "private_internal_excluded": 124, "fine_tuned_excluded": 12}),
    "phikon_family_official_scores_2023_2024.csv": Counter({"canonical_candidate": 43, "private_internal_excluded": 4}),
    "ctranspath_official_evidence_2022_2024.csv": Counter({"secondary_only_excluded": 50}),
}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_snapshot(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if Counter(row["inclusion_status"] for row in rows) != EXPECTED[path.name]:
        raise ValueError(f"Wave E disposition drift: {path.name}")
    for row in rows:
        row["_snapshot_name"] = path.name
    return rows


def selected(paths: list[Path]) -> list[dict[str, str]]:
    rows = [row for path in paths for row in load_snapshot(path)]
    chosen = [row for row in rows if row["inclusion_status"] == "canonical_candidate"]
    if len(chosen) != 417 or len({(row["model_id"], row["evaluation_id"]) for row in chosen}) != 417:
        raise ValueError("Wave E selected-cell audit failed")
    return chosen


def _group_c_identities(group_c_snapshot: Path) -> dict[tuple[str, int], str]:
    by_table: dict[str, list[str]] = defaultdict(list)
    with group_c_snapshot.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["suite_id"] != "titan_paper":
                continue
            identity = row["task_identity_id"]
            if identity not in by_table[row["table"]]:
                by_table[row["table"]].append(identity)
    return {(table, index): identity for table, identities in by_table.items() for index, identity in enumerate(identities)}


def _identity(row: dict[str, str], group_c: dict[tuple[str, int], str]) -> str:
    if row["model_id"] == "conch-1.5":
        match = re.search(r"\.t\d+\.(\d+)\.", row["evaluation_id"])
        index = int(match.group(1)) if match else 0
        return group_c[(row["source_table"], index)]
    if row["model_id"] == "conch":
        return "task.wavee.conch." + slug(row["dedup_key"].rsplit(".", 1)[0])
    return "task.wavee.phikon." + slug(row["dedup_key"].rsplit(".", 1)[0])


def build_protocols(paths: list[Path], tasks: list[dict[str, object]], group_c_snapshot: Path) -> list[dict[str, object]]:
    by_identity = {str(row["task_identity_id"]): row for row in tasks}
    group_c = _group_c_identities(group_c_snapshot)
    output = []
    for row in sorted(selected(paths), key=lambda item: item["evaluation_id"]):
        identity = _identity(row, group_c)
        if identity in by_identity:
            base = dict(by_identity[identity])
        else:
            sample_unit = "slide" if row["level"] == "slide" else "image"
            metric = row["metric"]
            task_type = "survival" if metric == "c_index" else "segmentation" if metric in {"dice", "precision", "recall"} else "classification"
            base = {
                "task_identity_id": identity, "dataset_artifact_id": "artifact." + identity.removeprefix("task."),
                "dataset_id": slug(row["task_label"])[:96], "task_name": row["task_label"],
                "task_family": task_type, "target": row["task_label"], "sample_unit": sample_unit,
                "task_type": task_type, "num_samples": "not_reported",
            }
        base.update({
            "suite_id": row["suite_id"], "evaluation_id": row["evaluation_id"],
            "protocol_id": row["evaluation_id"], "endpoint": row["downstream_protocol"],
            "metric": row["metric"], "direction": "higher",
            "protocol": f"{row['downstream_protocol']}; embedding={row['embedding_recipe']}; magnification={row['magnification']}",
            "reference_url": row["reference_url"], "audit_status": "parsed_primary_source",
            "audit_notes": row["inclusion_reason"],
        })
        output.append(base)
    if len(output) != 417 or len({row["protocol_id"] for row in output}) != 417:
        raise ValueError(f"expected 417 Wave E protocols, found {len(output)}")
    return output


def build_scores(paths: list[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scores = []
    aliases: dict[tuple[str, str], dict[str, object]] = {}
    for row in selected(paths):
        value = float(row["value"])
        normalized = value * 100
        scores.append({
            "model_id": row["model_id"], "reported_model_alias": row["model_alias"],
            "model_revision": row["model_revision"], "evaluation_id": row["evaluation_id"],
            "value": f"{value:.6g}", "normalized_score": f"{normalized:.6g}",
            "suite_id": row["suite_id"], "metric": row["metric"],
            "reference_url": row["reference_url"], "source_locator": row["source_locator"],
            "extraction_date": "2026-08-06", "review_status": "machine_parsed_single_source",
            "uncertainty": row["uncertainty"],
            "lineage": f"wave-e:{row['_snapshot_name']}@sha256:{row['source_sha256']} -> {row['source_revision']} -> {row['source_locator']}",
            "audit_status": "parsed_primary_source",
        })
        aliases[(row["suite_id"], row["model_alias"])] = {
            "alias": row["model_alias"], "model_id": row["model_id"], "suite_id": row["suite_id"],
            "reference_url": row["reference_url"], "audit_notes": "Exact versioned component alias from the selected Wave E primary-paper row.",
        }
    return scores, list(aliases.values())
