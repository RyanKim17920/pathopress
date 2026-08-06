#!/usr/bin/env python3
"""Build the versioned, pre-error pathology probe-feasibility registry.

The registry deliberately does not call any task "cheap".  It separates
source-reported protocol facts from declared feasibility proxies, and leaves
runtime, compute, annotation-hours, and dollar cost incomplete unless a
primary source reports them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402


ALLOWLIST_ID = "pathology_low_friction_proxy_v1"
MAX_SAMPLES = 10_000


def _label_proxy(task_type: str) -> str:
    return {
        "classification": "existing_class_label",
        "regression": "existing_continuous_outcome",
        "survival": "existing_time_to_event_outcome",
        "segmentation": "dense_segmentation_mask",
        "robustness": "multicenter_robustness_labels",
    }.get(task_type, "metadata_incomplete")


def _processing_proxy(sample_unit: str) -> str:
    return {
        "image": "image_direct",
        "patch": "patch_direct",
        "slide": "whole_slide_processing",
        "case": "case_level_slide_processing",
        "ST spot": "spatial_transcriptomics_alignment",
    }.get(sample_unit, "metadata_incomplete")


def main() -> None:
    tasks_path = ROOT / "data/tasks.csv"
    scores_path = ROOT / "data/scores.csv"
    output_path = ROOT / "data/evaluation_feasibility.csv"
    allowlist_path = ROOT / "data/low_friction_allowlist_v1.json"

    scores = load_scores(scores_path)
    matrix, models, evaluations = make_matrix(scores)
    _, _, evaluations = filter_matrix(matrix, models, evaluations)
    retained = set(evaluations)
    with tasks_path.open(newline="", encoding="utf-8") as handle:
        task_rows = {row["evaluation_id"]: row for row in csv.DictReader(handle)}
    missing = sorted(retained - task_rows.keys())
    if missing:
        raise ValueError(f"retained evaluations missing task metadata: {missing}")

    fieldnames = [
        "evaluation_id", "suite_id", "dataset_id", "sample_count",
        "sample_count_status", "sample_unit", "task_type", "label_burden_proxy",
        "label_burden_basis", "input_processing_proxy", "proxy_basis",
        "source_compute", "source_runtime", "compute_runtime_status",
        "annotation_hours", "annotation_hours_status", "dollar_cost",
        "dollar_cost_status", "reference_url", "audit_status",
        "allowlist_id", "allowlisted", "allowlist_decision_reason",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected: list[str] = []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for evaluation_id in evaluations:
            row = task_rows[evaluation_id]
            raw_n = row["num_samples"].strip()
            try:
                sample_count = int(raw_n)
                sample_status = "source_reported"
            except ValueError:
                sample_count = None
                sample_status = "not_reported"
            allowed = (
                row["sample_unit"] in {"image", "patch"}
                and row["task_type"] == "classification"
                and sample_count is not None
                and sample_count <= MAX_SAMPLES
            )
            if allowed:
                selected.append(evaluation_id)
                reason = "image_or_patch_classification_with_source_reported_n_le_10000"
            elif row["sample_unit"] not in {"image", "patch"}:
                reason = "requires_non_image_patch_sample_unit"
            elif row["task_type"] != "classification":
                reason = "not_classification"
            elif sample_count is None:
                reason = "sample_count_metadata_incomplete"
            else:
                reason = "source_reported_n_gt_10000"
            writer.writerow({
                "evaluation_id": evaluation_id,
                "suite_id": row["suite_id"],
                "dataset_id": row["dataset_id"],
                "sample_count": "" if sample_count is None else sample_count,
                "sample_count_status": sample_status,
                "sample_unit": row["sample_unit"],
                "task_type": row["task_type"],
                "label_burden_proxy": _label_proxy(row["task_type"]),
                "label_burden_basis": "declared_protocol_proxy_not_measured_labor",
                "input_processing_proxy": _processing_proxy(row["sample_unit"]),
                "proxy_basis": "sample_unit_and_task_type_only",
                "source_compute": "",
                "source_runtime": "",
                "compute_runtime_status": "metadata_incomplete",
                "annotation_hours": "",
                "annotation_hours_status": "metadata_incomplete",
                "dollar_cost": "",
                "dollar_cost_status": "not_reported_no_numeric_cost_invented",
                "reference_url": row["reference_url"],
                "audit_status": row["audit_status"],
                "allowlist_id": ALLOWLIST_ID,
                "allowlisted": str(allowed).lower(),
                "allowlist_decision_reason": reason,
            })

    allowlist = {
        "schema_version": 1,
        "allowlist_id": ALLOWLIST_ID,
        "derivation_timing": "defined_from_protocol_metadata_before_prediction_error_analysis",
        "semantics": "conservative low-friction feasibility proxy; not a monetary or wall-clock cost claim",
        "rule": {
            "sample_unit_in": ["image", "patch"],
            "task_type_equals": "classification",
            "sample_count_status_equals": "source_reported",
            "sample_count_max_inclusive": MAX_SAMPLES,
        },
        "excluded_cost_fields": ["dollar_cost", "annotation_hours", "compute_runtime"],
        "excluded_cost_reason": "metadata_incomplete; no numeric values imputed",
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        "n_retained_evaluations": len(evaluations),
        "n_allowlisted": len(selected),
        "evaluation_ids": selected,
    }
    allowlist_path.write_text(json.dumps(allowlist, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_retained": len(evaluations), "n_allowlisted": len(selected), "ids": selected}, indent=2))


if __name__ == "__main__":
    main()
