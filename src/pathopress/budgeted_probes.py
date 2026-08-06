"""Canonical burden-receipt accounting and deterministic probe-set search."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence


RECEIPT_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_PATH = "data/evaluation_burden_measurements.schema.json"
BUDGET_SCHEMA_VERSION = "pathopress-probe-budget-v1"
EVIDENCE_STATUSES = {
    "measured", "source_reported", "configured_ceiling", "not_applicable",
    "not_measured", "not_reported", "inaccessible",
}
ACCEPTABLE_NUMERIC_STATUSES = {"measured", "source_reported", "configured_ceiling"}
UNKNOWN_STATUSES = {"not_measured", "not_reported", "inaccessible"}
PHASES = {
    "shared_artifact_setup", "per_model_feature_extraction",
    "per_protocol_head_fit", "per_protocol_evaluation",
}
LOSS_TOLERANCE = 1e-12


class BurdenValidationError(ValueError):
    """Raised when canonical receipts or a budget violate their contract."""


class MissingBurdenError(BurdenValidationError):
    """Raised when fail-closed selection encounters incomplete burden data."""


class SearchSpaceTooLargeError(ValueError):
    """Raised before an exact search whose upper bound exceeds its cap."""


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BurdenValidationError(f"{path} must contain a JSON object")
    return payload


def _validate_fact(fact: Any, *, name: str, numeric: bool = True) -> dict[str, Any]:
    if not isinstance(fact, dict):
        raise BurdenValidationError(f"{name} must be an object")
    status = fact.get("status")
    value = fact.get("value")
    if status not in EVIDENCE_STATUSES:
        raise BurdenValidationError(f"{name}.status is not canonical: {status!r}")
    if numeric and (not isinstance(fact.get("unit"), str) or not fact["unit"]):
        raise BurdenValidationError(f"{name}.unit is required")
    if status in {"not_applicable", *UNKNOWN_STATUSES}:
        if value is not None:
            raise BurdenValidationError(f"{name} status {status} requires value=null")
        if not isinstance(fact.get("reason"), str) or not fact["reason"]:
            raise BurdenValidationError(f"{name} status {status} requires a reason")
    elif numeric:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BurdenValidationError(f"{name}.value must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise BurdenValidationError(f"{name}.value must be finite and nonnegative")
    elif status not in {"not_applicable", *UNKNOWN_STATUSES} and not isinstance(value, (str, bool)):
        raise BurdenValidationError(f"{name}.value must be a string or boolean")
    if status == "measured" and not fact.get("measurement_method"):
        raise BurdenValidationError(f"{name} measured fact requires measurement_method")
    if status in {"source_reported", "configured_ceiling"}:
        if not fact.get("source_url") or not fact.get("source_locator"):
            raise BurdenValidationError(f"{name} {status} fact requires source_url and source_locator")
    return fact


def validate_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the optimizer-relevant canonical measurement contract."""

    if payload.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise BurdenValidationError("canonical receipt schema_version must be 1")
    if payload.get("schema_path") != RECEIPT_SCHEMA_PATH:
        raise BurdenValidationError(f"canonical receipt schema_path must be {RECEIPT_SCHEMA_PATH!r}")
    measurement = payload.get("measurement")
    if not isinstance(measurement, dict):
        raise BurdenValidationError("measurement object is required")
    required = {
        "measurement_id", "model_revision", "evaluation_id", "run_config_hash",
        "hardware_id", "cache_scope", "artifact_group_id", "phase",
        "execution_status", "resources", "constraints",
    }
    missing = sorted(required - set(measurement))
    if missing:
        raise BurdenValidationError(f"measurement missing keys: {missing}")
    if measurement["phase"] not in PHASES:
        raise BurdenValidationError(f"unknown phase: {measurement['phase']!r}")
    if measurement["phase"] == "shared_artifact_setup":
        if measurement["model_revision"] is not None:
            raise BurdenValidationError("shared setup model_revision must be null")
    elif not measurement["model_revision"] or not measurement["evaluation_id"]:
        raise BurdenValidationError("non-shared phase requires model_revision and evaluation_id")
    resources = measurement["resources"]
    if not isinstance(resources, dict):
        raise BurdenValidationError("measurement.resources must be an object")
    for resource, fact in resources.items():
        _validate_fact(fact, name=f"measurement.resources.{resource}")
    constraints = measurement["constraints"]
    if not isinstance(constraints, dict):
        raise BurdenValidationError("measurement.constraints must be an object")
    for constraint, fact in constraints.items():
        _validate_fact(fact, name=f"measurement.constraints.{constraint}", numeric=False)
    return payload


def load_receipts(path: str | Path) -> dict[str, Any]:
    """Load one canonical receipt or every JSON receipt in a directory."""

    source = Path(path)
    paths = sorted(source.glob("*.json")) if source.is_dir() else [source]
    if not paths:
        raise BurdenValidationError(f"no JSON receipts found under {source}")
    measurements = []
    seen: set[str] = set()
    for receipt_path in paths:
        receipt = validate_receipt(load_json(receipt_path))
        measurement = receipt["measurement"]
        if measurement["measurement_id"] in seen:
            raise BurdenValidationError(f"duplicate measurement_id: {measurement['measurement_id']}")
        seen.add(measurement["measurement_id"])
        measurements.append(measurement)
    return {"receipt_paths": [str(value) for value in paths], "measurements": measurements}


def validate_budget(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != BUDGET_SCHEMA_VERSION:
        raise BurdenValidationError(f"unsupported budget schema: {payload.get('schema_version')!r}")
    accepted = payload.get("accepted_evidence_statuses", ["measured"])
    if not isinstance(accepted, list) or not accepted:
        raise BurdenValidationError("accepted_evidence_statuses must be a non-empty list")
    if not set(accepted).issubset(ACCEPTABLE_NUMERIC_STATUSES):
        raise BurdenValidationError(
            "accepted statuses must be measured, source_reported, or configured_ceiling"
        )
    required_phases = payload.get("required_phases", [])
    if not isinstance(required_phases, list) or not set(required_phases).issubset(PHASES):
        raise BurdenValidationError("required_phases contains an unknown phase")
    for field in ("additive_limits", "capacity_limits"):
        limits = payload.get(field, {})
        if not isinstance(limits, dict):
            raise BurdenValidationError(f"{field} must be an object")
        for resource, limit in limits.items():
            if not isinstance(limit, dict):
                raise BurdenValidationError(f"{field}.{resource} must be an object")
            value = limit.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BurdenValidationError(f"{field}.{resource}.value must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0 or not limit.get("unit"):
                raise BurdenValidationError(f"{field}.{resource} needs nonnegative value and unit")
    measurement_filter = payload.get("measurement_filter", {})
    if not isinstance(measurement_filter, dict):
        raise BurdenValidationError("measurement_filter must be an object")
    allowed_constraints = payload.get("allowed_constraints", {})
    if not isinstance(allowed_constraints, dict):
        raise BurdenValidationError("allowed_constraints must be an object")
    for field, values in allowed_constraints.items():
        if not isinstance(values, list) or any(not isinstance(value, (str, bool)) for value in values):
            raise BurdenValidationError(f"allowed_constraints.{field} must be a list of strings or booleans")
    return payload


def _matches_filter(measurement: dict[str, Any], budget: dict[str, Any]) -> bool:
    for key, value in budget.get("measurement_filter", {}).items():
        if measurement["phase"] == "shared_artifact_setup" and key in {"model_revision", "evaluation_id"}:
            continue
        if measurement.get(key) != value:
            return False
    return True


def _fact_value(
    fact: dict[str, Any] | None, *, expected_unit: str, accepted: set[str], label: str
) -> tuple[float | None, str | None]:
    if fact is None:
        return None, f"{label}:not_reported"
    status = fact["status"]
    if status == "not_applicable":
        # The canonical fact remains null; zero is used only in the aggregate
        # arithmetic because the source explicitly says the resource does not apply.
        return 0.0, None
    if status in UNKNOWN_STATUSES:
        return None, f"{label}:{status}"
    if status not in accepted:
        return None, f"{label}:status_{status}_not_accepted"
    if fact["unit"] != expected_unit:
        return None, f"{label}:unit_{fact['unit']!r}_expected_{expected_unit!r}"
    return float(fact["value"]), None


def _policy_value(
    fact: dict[str, Any] | None, *, accepted: set[str], label: str
) -> tuple[str | None, str | None]:
    if fact is None:
        return None, f"{label}:not_reported"
    status = fact.get("status")
    if status in UNKNOWN_STATUSES or status == "not_applicable":
        return None, f"{label}:{status}"
    if status not in accepted:
        return None, f"{label}:status_{status}_not_accepted"
    return fact.get("value"), None


def _selected_measurements(
    selected_ids: Sequence[str], ledger: dict[str, Any], budget: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    selected = set(selected_ids)
    all_measurements = [row for row in ledger["measurements"] if _matches_filter(row, budget)]
    nonshared = [
        row for row in all_measurements
        if row["phase"] != "shared_artifact_setup" and row.get("evaluation_id") in selected
    ]
    reasons: list[str] = []
    required = set(budget.get("required_phases", []))
    groups: set[str] = set()
    for evaluation_id in sorted(selected):
        rows = [row for row in nonshared if row["evaluation_id"] == evaluation_id]
        if not rows:
            reasons.append(f"{evaluation_id}:not_measured")
            continue
        groups.update(row["artifact_group_id"] for row in rows)
        for phase in sorted(required - {"shared_artifact_setup"}):
            matching = [row for row in rows if row["phase"] == phase and row["execution_status"] == "completed"]
            if len(matching) != 1:
                reasons.append(f"{evaluation_id}.{phase}:expected_one_completed_receipt_found_{len(matching)}")
    shared: list[dict[str, Any]] = []
    if "shared_artifact_setup" in required:
        for group in sorted(groups):
            matching = [
                row for row in all_measurements
                if row["phase"] == "shared_artifact_setup"
                and row["artifact_group_id"] == group
                and row["execution_status"] == "completed"
            ]
            if len(matching) != 1:
                reasons.append(f"{group}.shared_artifact_setup:expected_one_completed_receipt_found_{len(matching)}")
            else:
                shared.extend(matching)
    usable_nonshared = [
        row for row in nonshared
        if row["execution_status"] == "completed"
        and (not required or row["phase"] in required)
    ]
    # Feature extraction is purchased once per model revision and artifact
    # group, even if several selected protocols reference it. Conflicting
    # receipts for the same accounting key fail closed.
    deduplicated_nonshared: list[dict[str, Any]] = []
    feature_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in usable_nonshared:
        if row["phase"] == "per_model_feature_extraction":
            feature_groups.setdefault(
                (str(row["model_revision"]), str(row["artifact_group_id"])), []
            ).append(row)
        else:
            deduplicated_nonshared.append(row)
    for key, rows in sorted(feature_groups.items()):
        canonical = sorted(rows, key=lambda row: row["measurement_id"])[0]
        if any(
            row["resources"] != canonical["resources"]
            or row["constraints"] != canonical["constraints"]
            for row in rows[1:]
        ):
            reasons.append(
                f"{key[0]}.{key[1]}.per_model_feature_extraction:conflicting_receipts"
            )
        deduplicated_nonshared.append(canonical)
    return [*deduplicated_nonshared, *shared], sorted(groups), reasons


def set_burden(
    selected_ids: Sequence[str],
    ledger: dict[str, Any],
    budget: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate canonical phase receipts, charging shared setup once per group."""

    measurements, charged_groups, reasons = _selected_measurements(selected_ids, ledger, budget)
    accepted = set(budget["accepted_evidence_statuses"])
    additive: dict[str, float] = {}
    capacity: dict[str, float] = {}
    for resource, limit in budget.get("additive_limits", {}).items():
        total = 0.0
        for measurement in measurements:
            value, reason = _fact_value(
                measurement["resources"].get(resource), expected_unit=limit["unit"],
                accepted=accepted, label=f"{measurement['measurement_id']}.{resource}",
            )
            if reason:
                reasons.append(reason)
            else:
                total += float(value)
        additive[resource] = total
        if total > float(limit["value"]) + LOSS_TOLERANCE:
            reasons.append(f"additive_limit_exceeded:{resource}")
    for resource, limit in budget.get("capacity_limits", {}).items():
        maximum = 0.0
        for measurement in measurements:
            value, reason = _fact_value(
                measurement["resources"].get(resource), expected_unit=limit["unit"],
                accepted=accepted, label=f"{measurement['measurement_id']}.{resource}",
            )
            if reason:
                reasons.append(reason)
            else:
                maximum = max(maximum, float(value))
        capacity[resource] = maximum
        if maximum > float(limit["value"]) + LOSS_TOLERANCE:
            reasons.append(f"capacity_limit_exceeded:{resource}")
    for measurement in measurements:
        for field, allowed_values in budget.get("allowed_constraints", {}).items():
            allowed = set(allowed_values)
            value, reason = _policy_value(
                measurement["constraints"].get(field), accepted=accepted,
                label=f"{measurement['measurement_id']}.constraints.{field}",
            )
            if reason:
                reasons.append(reason)
            elif value not in allowed:
                reasons.append(
                    f"{measurement['measurement_id']}.constraints.{field}:value_{value!r}_not_allowed"
                )
    return {
        "feasible": not reasons,
        "reasons": sorted(set(reasons)),
        "additive": additive,
        "capacity": capacity,
        "charged_artifact_group_ids": charged_groups,
        "charged_measurement_ids": sorted(row["measurement_id"] for row in measurements),
    }


def screen_candidates(
    candidate_ids: Sequence[str], ledger: dict[str, Any], budget: dict[str, Any],
    *, missing_policy: str,
) -> tuple[list[str], dict[str, list[str]]]:
    if missing_policy not in {"error", "exclude"}:
        raise ValueError("missing_policy must be 'error' or 'exclude'")
    eligible: list[str] = []
    excluded: dict[str, list[str]] = {}
    for evaluation_id in sorted(set(candidate_ids)):
        result = set_burden([evaluation_id], ledger, budget)
        if result["reasons"]:
            missing_reasons = [
                reason for reason in result["reasons"]
                if any(marker in reason for marker in (
                    ":not_measured", ":not_reported", ":inaccessible",
                    "not_accepted", "expected_", "conflicting_receipts",
                ))
            ]
            if missing_policy == "error" and missing_reasons:
                raise MissingBurdenError(f"{evaluation_id}: {'; '.join(missing_reasons)}")
            excluded[evaluation_id] = result["reasons"]
        else:
            eligible.append(evaluation_id)
    return eligible, excluded


def _pressure(result: dict[str, Any], budget: dict[str, Any]) -> float:
    pressure = 0.0
    for field, values in (("additive_limits", result["additive"]), ("capacity_limits", result["capacity"])):
        for resource, value in values.items():
            limit = float(budget[field][resource]["value"])
            pressure += value / limit if limit > 0 else (0.0 if value == 0 else math.inf)
    return pressure


def _choice_key(loss: float, selected: tuple[str, ...], burden: dict[str, Any], budget: dict[str, Any]) -> tuple[Any, ...]:
    costs = tuple(burden["additive"][key] for key in sorted(burden["additive"]))
    capacities = tuple(burden["capacity"][key] for key in sorted(burden["capacity"]))
    return (round(float(loss) / LOSS_TOLERANCE), _pressure(burden, budget), costs, capacities, selected)


EvaluateSets = Callable[[list[tuple[str, ...]]], list[tuple[float, dict[str, Any]]]]


def greedy_search(
    candidate_ids: Sequence[str], identity_by_id: dict[str, str], ledger: dict[str, Any],
    budget: dict[str, Any], evaluate_sets: EvaluateSets, *, max_probes: int,
    one_per_identity: bool = True,
) -> dict[str, Any]:
    selected: tuple[str, ...] = ()
    trajectory: list[dict[str, Any]] = []
    candidates = tuple(sorted(set(candidate_ids)))
    for step in range(1, max_probes + 1):
        old_selected = selected
        used_identities = {identity_by_id[value] for value in selected}
        proposals: list[tuple[str, ...]] = []
        burdens: list[dict[str, Any]] = []
        for candidate in candidates:
            if candidate in selected or (one_per_identity and identity_by_id[candidate] in used_identities):
                continue
            proposal = tuple(sorted((*selected, candidate)))
            burden = set_burden(proposal, ledger, budget)
            if burden["feasible"]:
                proposals.append(proposal)
                burdens.append(burden)
        if not proposals:
            break
        evaluated = evaluate_sets(proposals)
        if len(evaluated) != len(proposals):
            raise ValueError("evaluate_sets returned the wrong number of results")
        choices = [
            (_choice_key(loss, proposal, burden, budget), proposal, burden, loss, metrics)
            for proposal, burden, (loss, metrics) in zip(proposals, burdens, evaluated)
        ]
        _, selected, chosen_burden, loss, metrics = min(choices, key=lambda row: row[0])
        prior_groups = set(trajectory[-1]["burden"]["charged_artifact_group_ids"]) if trajectory else set()
        trajectory.append({
            "k": step,
            "added_evaluation_id": next(value for value in selected if value not in old_selected),
            "evaluation_ids": list(selected), "objective_loss": float(loss), "metrics": metrics,
            "burden": chosen_burden,
            "newly_charged_artifact_group_ids": sorted(
                set(chosen_burden["charged_artifact_group_ids"]) - prior_groups
            ),
        })
    return {"search": "greedy", "globally_exact": False, "trajectory": trajectory}


def exact_search_preflight(n_candidates: int, max_probes: int, max_subsets: int) -> int:
    upper_bound = sum(math.comb(n_candidates, size) for size in range(1, min(n_candidates, max_probes) + 1))
    if upper_bound > max_subsets:
        raise SearchSpaceTooLargeError(
            f"exact-search upper bound {upper_bound} exceeds --max-subsets={max_subsets}"
        )
    return upper_bound


def exact_search(
    candidate_ids: Sequence[str], identity_by_id: dict[str, str], ledger: dict[str, Any],
    budget: dict[str, Any], evaluate_sets: EvaluateSets, *, max_probes: int,
    max_subsets: int, one_per_identity: bool = True, batch_size: int = 256,
) -> dict[str, Any]:
    candidates = tuple(sorted(set(candidate_ids)))
    upper_bound = exact_search_preflight(len(candidates), max_probes, max_subsets)
    best: tuple[Any, ...] | None = None
    evaluated_count = feasible_count = 0
    batch_sets: list[tuple[str, ...]] = []
    batch_burdens: list[dict[str, Any]] = []

    def consume() -> None:
        nonlocal best, evaluated_count
        if not batch_sets:
            return
        results = evaluate_sets(batch_sets)
        if len(results) != len(batch_sets):
            raise ValueError("evaluate_sets returned the wrong number of results")
        evaluated_count += len(batch_sets)
        for selected, burden, (loss, metrics) in zip(batch_sets, batch_burdens, results):
            row = (_choice_key(loss, selected, burden, budget), selected, burden, float(loss), metrics)
            if best is None or row[0] < best[0]:
                best = row
        batch_sets.clear(); batch_burdens.clear()

    for size in range(1, min(len(candidates), max_probes) + 1):
        for selected in itertools.combinations(candidates, size):
            if one_per_identity and len({identity_by_id[value] for value in selected}) != len(selected):
                continue
            burden = set_burden(selected, ledger, budget)
            if not burden["feasible"]:
                continue
            feasible_count += 1; batch_sets.append(selected); batch_burdens.append(burden)
            if len(batch_sets) >= batch_size:
                consume()
    consume()
    optimum = None if best is None else {
        "evaluation_ids": list(best[1]), "objective_loss": best[3],
        "metrics": best[4], "burden": best[2],
    }
    return {
        "search": "exact", "globally_exact": True, "subset_upper_bound": upper_bound,
        "n_feasible_subsets": feasible_count, "n_evaluated_subsets": evaluated_count,
        "optimum": optimum,
    }


def random_feasible_prefixes(
    candidate_ids: Sequence[str], identity_by_id: dict[str, str], ledger: dict[str, Any],
    budget: dict[str, Any], *, max_probes: int, repeats: int, seed: int,
    one_per_identity: bool = True,
) -> list[list[tuple[str, ...]]]:
    import random
    candidates = sorted(set(candidate_ids)); output: list[list[tuple[str, ...]]] = []
    for repeat in range(repeats):
        order = candidates.copy(); random.Random((seed + repeat) * 100000).shuffle(order)
        selected: tuple[str, ...] = (); prefixes: list[tuple[str, ...]] = []
        for candidate in order:
            if len(selected) >= max_probes:
                break
            if one_per_identity and identity_by_id[candidate] in {identity_by_id[value] for value in selected}:
                continue
            proposal = tuple(sorted((*selected, candidate)))
            if set_burden(proposal, ledger, budget)["feasible"]:
                selected = proposal; prefixes.append(selected)
        output.append(prefixes)
    return output
