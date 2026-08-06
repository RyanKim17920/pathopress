"""Deterministic, evidence-bounded review ledger for registry score rows.

This module never assigns human review.  It validates a materialized ledger
against the registry contracts and the immutable hashes captured when pinned
upstream sources were reparsed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping


REVIEWER_TYPE = "automated_agent_review"
REVIEWED_AT = "2026-08-06T00:00:00Z"
PRIMARY_AUDIT_STATUSES = {
    "parsed_primary_source",
    "parsed_primary_source_analysis_ineligible",
}
LEDGER_FIELDS = (
    "review_id",
    "model_id",
    "evaluation_id",
    "suite_id",
    "reviewer_type",
    "reviewed_at",
    "evidence_kind",
    "evidence_path",
    "evidence_revision",
    "evidence_sha256",
    "source_locator",
    "source_locator_reachable",
    "checks_passed",
    "duplicate_group_ids",
    "canonical_setting_decision",
    "audit_status_preserved",
    "prior_review_status",
    "review_outcome",
    "promotion_eligible",
    "decision_reason",
)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: Mapping[str, str]) -> tuple[str, str]:
    return str(row["model_id"]), str(row["evaluation_id"])


def duplicate_memberships(rows: Iterable[Mapping[str, str]]) -> dict[str, tuple[str, ...]]:
    memberships: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        memberships[str(row["member_evaluation_id"])].append(str(row["group_id"]))
    return {key: tuple(sorted(value)) for key, value in memberships.items()}


def validate_ledger(
    scores: list[Mapping[str, str]],
    tasks: list[Mapping[str, str]],
    dedup: list[Mapping[str, str]],
    ledger: list[Mapping[str, str]],
) -> dict[str, object]:
    """Validate one-to-one coverage and all non-network review invariants."""
    errors: list[str] = []
    task_by_id = {str(row["evaluation_id"]): row for row in tasks}
    score_by_key = {row_key(row): row for row in scores}
    ledger_by_key = {row_key(row): row for row in ledger}
    memberships = duplicate_memberships(dedup)
    if len(score_by_key) != len(scores):
        errors.append("scores contain duplicate model/evaluation cells")
    if len(ledger_by_key) != len(ledger):
        errors.append("ledger contains duplicate model/evaluation cells")
    if set(score_by_key) != set(ledger_by_key):
        errors.append("ledger keys do not exactly cover scores")
    if (tuple(ledger[0].keys()) if ledger else ()) != LEDGER_FIELDS:
        errors.append("unexpected ledger schema")

    groups: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in dedup:
        groups[str(row["group_id"])].append(row)
    for group_id, members in groups.items():
        member_ids = [str(row["member_evaluation_id"]) for row in members]
        canonicals = {str(row["canonical_evaluation_id"]) for row in members}
        match_types = {str(row["match_type"]) for row in members}
        decisions = {str(row["decision"]) for row in members}
        if len(member_ids) != len(set(member_ids)) or len(canonicals) != 1:
            errors.append(f"{group_id}: malformed duplicate group")
        if not set(member_ids) <= set(task_by_id):
            errors.append(f"{group_id}: references missing task")
        if match_types == {"exact"}:
            if decisions != {"link_only"}:
                errors.append(f"{group_id}: exact group must be link_only")
            identities = {str(task_by_id[item]["task_identity_id"]) for item in member_ids if item in task_by_id}
            declared = {str(row["task_identity_id"]) for row in members}
            if len(identities) != 1 or identities != declared:
                errors.append(f"{group_id}: exact identity mismatch")
        elif match_types == {"semantic"}:
            if decisions != {"keep_separate"}:
                errors.append(f"{group_id}: semantic group must stay separate")
        else:
            errors.append(f"{group_id}: inconsistent match types")

    for key, score in score_by_key.items():
        review = ledger_by_key.get(key)
        if review is None:
            continue
        task = task_by_id.get(str(score["evaluation_id"]))
        expected_groups = ";".join(memberships.get(str(score["evaluation_id"]), ()))
        if task is None:
            errors.append(f"missing task contract for {key}")
            continue
        expected_review_id = "review." + hashlib.sha256(
            (key[0] + "\0" + key[1]).encode("utf-8")
        ).hexdigest()[:20]
        exact = {
            "review_id": expected_review_id,
            "suite_id": score["suite_id"],
            "reviewer_type": REVIEWER_TYPE,
            "reviewed_at": REVIEWED_AT,
            "source_locator": score["source_locator"],
            "duplicate_group_ids": expected_groups,
            "audit_status_preserved": score["audit_status"],
            "prior_review_status": score["review_status"],
        }
        for field, expected in exact.items():
            if review[field] != expected:
                errors.append(f"{key}: {field} mismatch")
        if review["source_locator_reachable"] != "true":
            errors.append(f"{key}: locator was not reached")
        if len(review["evidence_sha256"]) != 64:
            errors.append(f"{key}: invalid evidence hash")
        required_checks = {
            "source_value",
            "source_locator",
            "metric_direction_scale",
            "model_alias",
            "protocol_setting",
            "split_version",
        }
        checks = set(review["checks_passed"].split(";"))
        if not required_checks <= checks:
            errors.append(f"{key}: incomplete checks")
        if task["metric"] != score["metric"]:
            errors.append(f"{key}: score/task metric mismatch")
        raw = float(score["value"])
        normalized = score["normalized_score"]
        metric = score["metric"]
        if metric in {"macro_ovr_auc", "bacc", "cindex", "balanced_accuracy", "dice", "robustness_index"}:
            expected_normalized = raw * 100.0
        elif metric in {"weighted_kappa", "pearson_r", "clustering_score"}:
            expected_normalized = (raw + 1.0) * 50.0
        elif metric == "f1":
            expected_normalized = raw
        elif metric == "average_performance_drop_percent":
            expected_normalized = None
        else:
            expected_normalized = float(normalized) if normalized else None
        if expected_normalized is None:
            if normalized:
                errors.append(f"{key}: ineligible metric has normalization")
        elif not normalized or abs(float(normalized) - expected_normalized) > 5e-4:
            errors.append(f"{key}: metric direction/scale mismatch")
        if score["audit_status"] == "reported_external":
            if review["review_outcome"] != "reported_external" or review["promotion_eligible"] != "false":
                errors.append(f"{key}: external row was promoted")
        else:
            expected = "source_locator_crosschecked" if expected_groups else "source_locator_validated"
            if review["review_outcome"] != expected or review["promotion_eligible"] != "true":
                errors.append(f"{key}: incorrect primary-source outcome")
        if score["audit_status"] == "parsed_primary_source_analysis_ineligible":
            if review["canonical_setting_decision"] != "retain_analysis_ineligible_apd":
                errors.append(f"{key}: APD eligibility changed")
        elif score["audit_status"] == "reported_external":
            if review["canonical_setting_decision"] != "retain_reported_external_only":
                errors.append(f"{key}: external eligibility changed")
        elif review["canonical_setting_decision"] != "retain_protocol_specific_cell":
            errors.append(f"{key}: canonical decision mismatch")

    counts = Counter(row["review_outcome"] for row in ledger)
    audit_counts = Counter(row["audit_status_preserved"] for row in ledger)
    if audit_counts != Counter(row["audit_status"] for row in scores):
        errors.append("audit-status counts changed")
    if errors:
        raise ValueError("score review ledger invalid:\n- " + "\n- ".join(errors[:50]))
    return {
        "schema_version": 1,
        "reviewer_type": REVIEWER_TYPE,
        "reviewed_at": REVIEWED_AT,
        "score_rows": len(scores),
        "retained_primary_rows": audit_counts["parsed_primary_source"],
        "analysis_ineligible_apd_rows": audit_counts["parsed_primary_source_analysis_ineligible"],
        "reported_external_rows": audit_counts["reported_external"],
        "locator_reachable_rows": sum(row["source_locator_reachable"] == "true" for row in ledger),
        "review_outcomes": dict(sorted(counts.items())),
        "duplicate_group_crosschecked_rows": sum(bool(row["duplicate_group_ids"]) for row in ledger),
    }


def write_summary(path: Path, summary: Mapping[str, object], ledger_path: Path) -> None:
    payload = dict(summary)
    payload["ledger_path"] = ledger_path.as_posix()
    payload["ledger_sha256"] = sha256_path(ledger_path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
