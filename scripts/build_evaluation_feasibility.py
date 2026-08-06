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
ALLOWLIST_V2_ID = "pathology_low_friction_pipeline_proxy_top25_v2"
MAX_SAMPLES = 10_000
LEGACY_V2_EVALUATIONS = (
    "eva.leaderboard.bach.validation",
    "eva.leaderboard.bracs.validation",
    "eva.leaderboard.breakhis.validation",
    "eva.leaderboard.crc.validation",
    "eva.leaderboard.gleason_arvaniti.validation",
    "eva.leaderboard.mhist.test",
    "eva.leaderboard.patch_camelyon.test",
    "eva.leaderboard.patch_camelyon.validation",
    "eva.leaderboard.patch_camelyon_10shot.test",
    "thunder.bach.linear_probing",
    "thunder.bracs.linear_probing",
    "thunder.break_his.linear_probing",
    "thunder.ccrcc.linear_probing",
    "thunder.crc.linear_probing",
    "thunder.esca.linear_probing",
    "thunder.mhist.linear_probing",
    "thunder.patch_camelyon.linear_probing",
    "thunder.spider_breast.linear_probing",
    "thunder.spider_colorectal.linear_probing",
    "thunder.spider_skin.linear_probing",
    "thunder.spider_thorax.linear_probing",
    "thunder.tcga_crc_msi.linear_probing",
    "thunder.tcga_tils.linear_probing",
    "thunder.tcga_uniform.linear_probing",
    "thunder.wilds.linear_probing",
)


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


def _pre_error_burden_key(row: dict[str, str]) -> tuple[int, int, int, str]:
    """Order protocols using only declared pre-error acquisition metadata."""

    try:
        sample_count = int(row["num_samples"].strip())
        missing_count = 0
    except ValueError:
        sample_count = 2**63 - 1
        missing_count = 1
    sample_unit_order = 0 if row["sample_unit"] == "image" else 1
    return missing_count, sample_count, sample_unit_order, row["evaluation_id"]


def main() -> None:
    tasks_path = ROOT / "data/tasks.csv"
    scores_path = ROOT / "data/scores.csv"
    output_path = ROOT / "data/evaluation_feasibility.csv"
    allowlist_path = ROOT / "data/low_friction_allowlist_v1.json"
    allowlist_v2_path = ROOT / "data/low_friction_allowlist_v2_top25.json"
    legacy_v2_path = ROOT / "data/low_friction_allowlist_v2_legacy25.json"
    eligible_v2_path = ROOT / "data/low_friction_pipeline_eligible_v2_all.json"

    scores = load_scores(scores_path)
    matrix, models, evaluations = make_matrix(scores)
    _, _, evaluations = filter_matrix(matrix, models, evaluations)
    retained = set(evaluations)
    with tasks_path.open(newline="", encoding="utf-8") as handle:
        task_rows = {row["evaluation_id"]: row for row in csv.DictReader(handle)}
    missing = sorted(retained - task_rows.keys())
    if missing:
        raise ValueError(f"retained evaluations missing task metadata: {missing}")

    pipeline_eligible_v2 = [
        evaluation_id
        for evaluation_id in evaluations
        if task_rows[evaluation_id]["sample_unit"] in {"image", "patch"}
        and task_rows[evaluation_id]["task_type"] == "classification"
    ]
    candidates_by_identity: dict[str, list[str]] = {}
    for evaluation_id in pipeline_eligible_v2:
        identity = task_rows[evaluation_id]["task_identity_id"]
        candidates_by_identity.setdefault(identity, []).append(evaluation_id)
    identity_representatives = sorted(
        (
            min(candidates, key=lambda value: _pre_error_burden_key(task_rows[value]))
            for candidates in candidates_by_identity.values()
        ),
        key=lambda value: _pre_error_burden_key(task_rows[value]),
    )
    remaining_variants = sorted(
        set(pipeline_eligible_v2) - set(identity_representatives),
        key=lambda value: _pre_error_burden_key(task_rows[value]),
    )
    refreshed_selection_order = [
        *identity_representatives,
        *remaining_variants[: 25 - len(identity_representatives)],
    ]
    refreshed_v2 = set(refreshed_selection_order)
    legacy_v2 = set(LEGACY_V2_EVALUATIONS)

    fieldnames = [
        "evaluation_id", "suite_id", "dataset_id", "sample_count",
        "sample_count_status", "sample_unit", "task_type", "label_burden_proxy",
        "label_burden_basis", "input_processing_proxy", "proxy_basis",
        "source_compute", "source_runtime", "compute_runtime_status",
        "annotation_hours", "annotation_hours_status", "dollar_cost",
        "dollar_cost_status", "reference_url", "audit_status",
        "allowlist_id", "allowlisted", "allowlist_decision_reason",
        "allowlist_v2_id", "pipeline_eligible_v2", "legacy_allowlisted_v2",
        "allowlisted_v2", "allowlist_v2_decision_reason",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected: list[str] = []
    selected_v2: list[str] = []
    representative_set = set(identity_representatives)
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
            pipeline_eligible = evaluation_id in pipeline_eligible_v2
            allowed_v2 = evaluation_id in refreshed_v2
            if allowed_v2:
                selected_v2.append(evaluation_id)
                reason_v2 = (
                    "refreshed_top25_identity_representative"
                    if evaluation_id in representative_set
                    else "refreshed_top25_variant_fill_by_pre_error_burden_order"
                )
            elif pipeline_eligible:
                reason_v2 = "pipeline_eligible_but_below_refreshed_top25_pre_error_order"
            elif row["sample_unit"] not in {"image", "patch"}:
                reason_v2 = "requires_non_image_patch_sample_unit"
            else:
                reason_v2 = "not_classification"
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
                "allowlist_v2_id": ALLOWLIST_V2_ID,
                "pipeline_eligible_v2": str(pipeline_eligible).lower(),
                "legacy_allowlisted_v2": str(evaluation_id in legacy_v2).lower(),
                "allowlisted_v2": str(allowed_v2).lower(),
                "allowlist_v2_decision_reason": reason_v2,
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
    source_binding = {
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        "n_retained_evaluations": len(evaluations),
    }
    all_eligible_v2 = {
        "schema_version": 2,
        "allowlist_id": "pathology_low_friction_pipeline_eligible_v2",
        "derivation_timing": "defined_from_protocol_metadata_before_prediction_error_analysis",
        "semantics": "all retained direct image/patch classification protocols; eligibility only, without a claim of measured monetary or runtime cost",
        "rule": {
            "sample_unit_in": ["image", "patch"],
            "task_type_equals": "classification",
        },
        "cost_fields": {
            "dollar_cost": "not_reported_no_numeric_cost_invented",
            "annotation_hours": "metadata_incomplete",
            "compute_runtime": "metadata_incomplete",
        },
        **source_binding,
        "n_task_identities": len(candidates_by_identity),
        "n_eligible": len(pipeline_eligible_v2),
        "evaluation_ids": sorted(
            pipeline_eligible_v2,
            key=lambda value: _pre_error_burden_key(task_rows[value]),
        ),
    }
    legacy_allowlist_v2 = {
        "schema_version": 2,
        "allowlist_id": "pathology_low_friction_pipeline_proxy_legacy25_v2",
        "derivation_timing": "frozen before the Wave D/F registry expansion",
        "semantics": "longitudinal comparison set only; superseded for current-data searches by the refreshed top-25 set",
        "selection": "explicit legacy evaluation identities, retained without retroactive reselection",
        "cost_fields": all_eligible_v2["cost_fields"],
        **source_binding,
        "n_allowlisted": len(LEGACY_V2_EVALUATIONS),
        "evaluation_ids": list(LEGACY_V2_EVALUATIONS),
    }
    allowlist_v2 = {
        "schema_version": 2,
        "allowlist_id": ALLOWLIST_V2_ID,
        "derivation_timing": "defined_from_protocol_metadata_before_prediction_error_analysis",
        "semantics": "refreshed low-friction input/label pipeline proxy aligned to the upstream 25-candidate budget; not a monetary, compute, annotation, or wall-clock cost claim",
        "rule": {
            "candidate_eligibility": {
                "sample_unit_in": ["image", "patch"],
                "task_type_equals": "classification",
            },
            "selection": "select one protocol per deduplicated task_identity_id first, then fill remaining slots with protocol variants",
            "pre_error_tie_break": [
                "source-reported sample count before missing sample count",
                "lower source-reported sample count",
                "image before patch sample unit",
                "lexicographic evaluation_id",
            ],
            "prediction_error_used": False,
        },
        "upstream_contract_alignment": {
            "benchpress_commit": "0a684b63ee0e4a401cb907a3827a82ea997d74c4",
            "upstream_user_cheap_current_matrix_candidates": 25,
            "pathology_pipeline_eligible_candidates": len(pipeline_eligible_v2),
            "pathology_eligible_task_identities": len(candidates_by_identity),
            "pathology_candidates": len(refreshed_selection_order),
            "candidate_count_match": len(refreshed_selection_order) == 25,
            "semantic_adaptation": "BenchPress's user-provided cheap list is replaced by a deterministic pathology protocol proxy; candidate count and search budgets match, cost semantics do not.",
        },
        "cost_fields": all_eligible_v2["cost_fields"],
        "cost_field_note": "No missing monetary, labor, compute, or runtime value is imputed. Source-reported sample count is used only as a deterministic pre-error burden proxy and is not called cost.",
        "large_dataset_warning": "Some selected evaluations have hundreds of thousands of samples; low-friction describes pipeline shape and existing labels only.",
        **source_binding,
        "n_allowlisted": len(refreshed_selection_order),
        "n_identity_representatives": len(identity_representatives),
        "n_variant_fill": 25 - len(identity_representatives),
        "evaluation_ids": refreshed_selection_order,
    }
    missing_legacy = sorted(legacy_v2 - set(evaluations))
    ineligible_legacy = sorted(legacy_v2 - set(pipeline_eligible_v2))
    if (
        len(refreshed_selection_order) != 25
        or len(refreshed_v2) != 25
        or missing_legacy
        or ineligible_legacy
    ):
        raise ValueError(
            "v2 feasibility artifacts violate their declared contracts: "
            f"refreshed={len(refreshed_v2)} missing_legacy={missing_legacy} "
            f"ineligible_legacy={ineligible_legacy}"
        )
    eligible_v2_path.write_text(
        json.dumps(all_eligible_v2, indent=2) + "\n", encoding="utf-8"
    )
    legacy_v2_path.write_text(
        json.dumps(legacy_allowlist_v2, indent=2) + "\n", encoding="utf-8"
    )
    allowlist_v2_path.write_text(
        json.dumps(allowlist_v2, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "n_retained": len(evaluations),
        "n_allowlisted_v1": len(selected),
        "n_pipeline_eligible_v2": len(pipeline_eligible_v2),
        "n_pipeline_eligible_task_identities_v2": len(candidates_by_identity),
        "n_legacy_allowlisted_v2": len(LEGACY_V2_EVALUATIONS),
        "n_refreshed_allowlisted_v2": len(refreshed_selection_order),
        "refreshed_v2_ids": refreshed_selection_order,
    }, indent=2))


if __name__ == "__main__":
    main()
