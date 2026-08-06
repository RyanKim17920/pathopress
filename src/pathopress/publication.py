"""Deterministic publication-layer summaries for PathoPress figures/tables."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_hero_target(raw_rows: Iterable[dict[str, str]]) -> str:
    """Choose the best-covered target deterministically from all-known k=10."""

    counts: Counter[str] = Counter()
    for row in raw_rows:
        if (
            row["protocol"] == "all_known"
            and row["candidate_mode"] == "any_candidate"
            and row["selection_objective"] == "medae"
            and int(row["k"]) == 10
        ):
            counts[row["model_id"]] += 1
    if not counts:
        raise ValueError("raw probe artifact has no all-known unrestricted MedAE k=10 rows")
    return min(counts, key=lambda model: (-counts[model], model))


def hero_target_cells(
    raw_rows: Iterable[dict[str, str]], model_id: str, ks: tuple[int, ...] = (1, 3, 10)
) -> dict[int, list[dict[str, str]]]:
    """Return aligned unrestricted MedAE cell predictions for a concrete target."""

    grouped: dict[int, list[dict[str, str]]] = {}
    for k in ks:
        current = [
            row
            for row in raw_rows
            if row["protocol"] == "all_known"
            and row["candidate_mode"] == "any_candidate"
            and row["selection_objective"] == "medae"
            and row["model_id"] == model_id
            and int(row["k"]) == k
        ]
        grouped[k] = sorted(current, key=lambda row: row["evaluation_id"])
    identities = [{row["evaluation_id"] for row in grouped[k]} for k in ks]
    if not identities or any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("target prediction rows are not aligned across k")
    return grouped


def quarter(value: str) -> str:
    parsed = date.fromisoformat(value)
    return f"{parsed.year}-Q{(parsed.month - 1) // 3 + 1}"


def metadata_panel_counts(
    scores: list[dict[str, str]],
    tasks: list[dict[str, str]],
    releases: list[dict[str, str]],
    retained_evaluations: set[str],
    retained_models: set[str],
) -> dict[str, Any]:
    """Build the exact categorical counts shown in the metadata overview."""

    retained_tasks = [row for row in tasks if row["evaluation_id"] in retained_evaluations]
    retained_scores = [
        row
        for row in scores
        if row["evaluation_id"] in retained_evaluations
        and row["model_id"] in retained_models
        and row.get("audit_status", "") in {"verified", "parsed_primary_source"}
    ]
    task_by_evaluation = {row["evaluation_id"]: row for row in retained_tasks}
    release_by_model = {
        row["model_id"]: row
        for row in releases
        if row["model_id"] in retained_models and row["verification_status"] == "verified"
    }
    release_quarters = Counter(quarter(row["release_date"]) for row in release_by_model.values())
    observed_quarters: Counter[str] = Counter()
    for row in retained_scores:
        release = release_by_model.get(row["model_id"])
        if release:
            observed_quarters[quarter(release["release_date"])] += 1
    task_family = Counter(row["task_family"] for row in retained_tasks)
    observed_family = Counter(
        task_by_evaluation[row["evaluation_id"]]["task_family"] for row in retained_scores
    )
    suites = Counter(row["suite_id"] for row in retained_tasks)
    audit = Counter(row["audit_status"] for row in retained_scores)
    source_domains: Counter[str] = Counter()
    for row in retained_tasks:
        url = row["reference_url"].split("//", 1)[-1]
        source_domains[url.split("/", 1)[0]] += 1
    return {
        "release_quarters": dict(sorted(release_quarters.items())),
        "observed_score_quarters": dict(sorted(observed_quarters.items())),
        "task_family": dict(sorted(task_family.items(), key=lambda item: (-item[1], item[0]))),
        "observed_family": dict(sorted(observed_family.items(), key=lambda item: (-item[1], item[0]))),
        "suite_tasks": dict(sorted(suites.items(), key=lambda item: (-item[1], item[0]))),
        "score_audit_status": dict(sorted(audit.items(), key=lambda item: (-item[1], item[0]))),
        "task_source_domains": dict(sorted(source_domains.items(), key=lambda item: (-item[1], item[0]))),
        "n_models": len(retained_models),
        "n_evaluations": len(retained_evaluations),
        "n_observed": len(retained_scores),
        "n_release_dates": len(release_by_model),
    }


def top_with_other(counts: dict[str, int], keep: int) -> tuple[list[str], list[int]]:
    items = list(counts.items())
    head = items[:keep]
    tail = items[keep:]
    labels = [key for key, _ in head]
    values = [value for _, value in head]
    if tail:
        labels.append("other")
        values.append(sum(value for _, value in tail))
    return labels, values


def group_scores_by_model(scores: Iterable[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"n_scores": 0, "suites": set(), "evaluations": set()})
    for row in scores:
        bucket = grouped[row["model_id"]]
        bucket["n_scores"] += 1
        bucket["suites"].add(row["suite_id"])
        bucket["evaluations"].add(row["evaluation_id"])
    return grouped
