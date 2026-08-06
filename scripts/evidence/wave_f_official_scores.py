"""Registry adapter for public Hibou, MUSK, and GPFM primary-paper leaves."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


EXPECTED = {
    "hibou_official_scores_2024.csv": Counter({
        "canonical_candidate": 18,
        "fine_tuned_excluded": 20,
    }),
    "musk_official_scores_2025.csv": Counter({
        "canonical_candidate": 68,
        "aggregate_excluded": 54,
        "fine_tuned_excluded": 9,
        "private_internal_excluded": 4,
    }),
    "gpfm_official_scores_2025.csv": Counter({
        "canonical_candidate": 99,
        "private_internal_excluded": 75,
        "fine_tuned_excluded": 43,
        "aggregate_excluded": 7,
    }),
}

EXACT_CLASSIFICATION_IDENTITIES = {
    "bach": "task.bach.four_class_classification",
    "bracs-3": "task.pathobench.bracs.slidelevel-coarse",
    "bracs-6": "task.bracs.seven_class_classification",
    "brcas7": "task.bracs.seven_class_classification",
    "breakhis": "task.breakhis.selected_four_class_classification",
    "camelyon": "task.eva.camelyon16.camelyon16-classification",
    "crc-100k": "task.crc100k.nine_class_classification",
    "crc100": "task.crc100k.nine_class_classification",
    "mhist": "task.mhist.hp_vs_ssa_classification",
    "nct-crc": "task.crc100k.nine_class_classification",
    "panda": "task.panda.isup_grade",
    "patchcamelyon": "task.patchcamelyon.binary_metastasis_classification",
    "pcam": "task.patchcamelyon.binary_metastasis_classification",
    "unitopatho": "task.eva.unitopatho.unitopatho-classification",
}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_snapshot(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if Counter(row["inclusion_status"] for row in rows) != EXPECTED[path.name]:
        raise ValueError(f"Wave F disposition drift: {path.name}")
    for row in rows:
        row["_snapshot_name"] = path.name
    return rows


def selected(paths: list[Path]) -> list[dict[str, str]]:
    rows = [row for path in paths for row in load_snapshot(path)]
    chosen = [row for row in rows if row["inclusion_status"] == "canonical_candidate"]
    keys = {(row["model_id"], row["evaluation_id"]) for row in chosen}
    if len(chosen) != 185 or len(keys) != 185:
        raise ValueError("Wave F selected-cell audit failed")
    return chosen


def _family(row: dict[str, str]) -> str:
    protocol = row["downstream_protocol"].lower()
    if "retrieval" in protocol:
        if "text-to-image" in protocol:
            return "text-to-image-retrieval"
        if "image-to-text" in protocol:
            return "image-to-text-retrieval"
        return "image-retrieval"
    if "survival" in protocol or "prognosis" in protocol:
        return "survival"
    if "biomarker" in protocol:
        return "biomarker"
    return "classification"


def _identity(row: dict[str, str]) -> str:
    family = _family(row)
    task = slug(row["task_label"])
    if family == "classification" and task in EXACT_CLASSIFICATION_IDENTITIES:
        return EXACT_CLASSIFICATION_IDENTITIES[task]
    return f"task.wavef.{family}.{task}"


def _metric(row: dict[str, str]) -> str:
    metric = row["metric"]
    return f"{metric}_percent" if row["value_unit"] == "percent" else metric


def build_protocols(
    paths: list[Path], tasks: list[dict[str, object]]
) -> list[dict[str, object]]:
    by_identity = {str(row["task_identity_id"]): row for row in tasks}
    output: list[dict[str, object]] = []
    for row in sorted(selected(paths), key=lambda item: item["evaluation_id"]):
        identity = _identity(row)
        if identity in by_identity:
            base = dict(by_identity[identity])
        else:
            family = _family(row)
            task = slug(row["task_label"])
            sample_unit = "slide" if row["level"] == "slide" else "image"
            task_type = "survival" if family == "survival" else family
            base = {
                "task_identity_id": identity,
                "dataset_artifact_id": f"artifact.wavef.{task}",
                "dataset_id": task,
                "task_name": row["task_label"],
                "task_family": task_type,
                "target": (
                    "survival risk"
                    if family == "survival"
                    else f"{row['task_label']} {family.replace('-', ' ')} target"
                ),
                "sample_unit": sample_unit,
                "task_type": task_type,
                "num_samples": "not_reported",
            }
        base.update({
            "suite_id": row["suite_id"],
            "evaluation_id": row["evaluation_id"],
            "protocol_id": row["evaluation_id"],
            "endpoint": row["downstream_protocol"],
            "metric": _metric(row),
            "direction": "higher",
            "protocol": (
                f"{row['downstream_protocol']}; embedding={row['embedding_recipe']}; "
                f"magnification={row['magnification']}"
            ),
            "reference_url": row["reference_url"],
            "audit_status": "parsed_primary_source",
            "audit_notes": row["inclusion_reason"],
        })
        output.append(base)
    if len(output) != 185 or len({row["protocol_id"] for row in output}) != 185:
        raise ValueError(f"expected 185 Wave F protocols, found {len(output)}")
    return output


def build_scores(
    paths: list[Path],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scores: list[dict[str, object]] = []
    aliases: dict[tuple[str, str], dict[str, object]] = {}
    for row in selected(paths):
        value = float(row["value"])
        normalized = value * 100 if row["value_unit"] == "fraction" else value
        scores.append({
            "model_id": row["model_id"],
            "reported_model_alias": row["model_alias"],
            "model_revision": row["model_revision"],
            "evaluation_id": row["evaluation_id"],
            "value": f"{value:.6g}",
            "normalized_score": f"{normalized:.6g}",
            "suite_id": row["suite_id"],
            "metric": _metric(row),
            "reference_url": row["reference_url"],
            "source_locator": row["source_locator"],
            "extraction_date": "2026-08-06",
            "review_status": "machine_parsed_single_source",
            "uncertainty": row["uncertainty"] or "not_reported",
            "lineage": (
                f"wave-f:{row['_snapshot_name']}@sha256:{row['source_sha256']} -> "
                f"{row['source_revision']} -> {row['source_locator']}"
            ),
            "audit_status": "parsed_primary_source",
        })
        aliases[(row["suite_id"], row["model_alias"])] = {
            "alias": row["model_alias"],
            "model_id": row["model_id"],
            "suite_id": row["suite_id"],
            "reference_url": row["reference_url"],
            "audit_notes": (
                "Exact model/checkpoint alias from a selected Wave F primary-paper row."
            ),
        }
    return scores, list(aliases.values())
