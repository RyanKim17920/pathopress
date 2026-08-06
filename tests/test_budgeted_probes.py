from __future__ import annotations

import unittest

from pathopress.budgeted_probes import (
    BurdenValidationError,
    MissingBurdenError,
    SearchSpaceTooLargeError,
    exact_search,
    greedy_search,
    random_feasible_prefixes,
    screen_candidates,
    set_burden,
    validate_budget,
    validate_receipt,
)


def fact(value, unit="USD", status="measured"):
    result = {"status": status, "value": value, "unit": unit}
    if status == "measured":
        result["measurement_method"] = "unit test"
    elif status in {"source_reported", "configured_ceiling"}:
        result.update(source_url="https://example.test", source_locator="test")
    elif status in {"not_applicable", "not_measured", "not_reported", "inaccessible"}:
        result["reason"] = "unit-test reason"
    return result


def category(value, status="measured"):
    result = {"status": status, "value": value}
    if status == "measured":
        result["measurement_method"] = "unit test"
    elif status in {"source_reported", "configured_ceiling"}:
        result.update(source_url="https://example.test", source_locator="test")
    else:
        result["reason"] = "unit-test reason"
    return result


def measurement(evaluation_id, cost, *, group=None, phase="per_protocol_evaluation", peak=2,
                cost_status="measured", access="open", suffix=""):
    shared = phase == "shared_artifact_setup"
    constraints = {
        "access_class": category(access),
        "dataset_license": category("research"),
        "commercial_use_allowed": category(True),
        "redistribution_allowed": category(False),
        "new_tissue_required": category(False),
    }
    return {
        "measurement_id": f"m-{phase}-{evaluation_id}-{suffix}",
        "model_revision": None if shared else "model@revision",
        "evaluation_id": None if shared else evaluation_id,
        "run_config_hash": "a" * 64,
        "hardware_id": "test",
        "cache_scope": "warm",
        "artifact_group_id": group or f"dataset.{evaluation_id}",
        "phase": phase,
        "execution_status": "completed",
        "resources": {
            "direct_cost": fact(cost, status=cost_status),
            "peak_vram": fact(peak, "GiB"),
        },
        "constraints": constraints,
    }


def receipt(row):
    return validate_receipt({
        "schema_version": 1,
        "schema_path": "data/evaluation_burden_measurements.schema.json",
        "measurement": row,
    })


def ledger(rows):
    return {"receipt_paths": [], "measurements": list(rows)}


def budget(limit=10, peak=10, *, accepted=("measured",), phases=("per_protocol_evaluation",)):
    return validate_budget({
        "schema_version": "pathopress-probe-budget-v1",
        "scenario": "test",
        "accepted_evidence_statuses": list(accepted),
        "required_phases": list(phases),
        "measurement_filter": {"model_revision": "model@revision"} if "shared_artifact_setup" not in phases else {},
        "additive_limits": {"direct_cost": {"value": limit, "unit": "USD"}},
        "capacity_limits": {"peak_vram": {"value": peak, "unit": "GiB"}},
        "allowed_constraints": {"access_class": ["open"], "commercial_use_allowed": [True]},
    })


def evaluator(losses):
    def evaluate(sets):
        return [(float(losses.get(selected, 10 - len(selected))), {"selected": list(selected)}) for selected in sets]
    return evaluate


class BudgetedProbeTests(unittest.TestCase):
    def test_canonical_status_contract_and_null_not_applicable(self):
        row = measurement("a", None, cost_status="not_applicable")
        receipt(row)
        result = set_burden(["a"], ledger([row]), budget())
        self.assertTrue(result["feasible"])
        self.assertEqual(result["additive"]["direct_cost"], 0)
        self.assertIsNone(row["resources"]["direct_cost"]["value"])

        bad = measurement("a", 0, cost_status="not_applicable")
        with self.assertRaises(BurdenValidationError):
            receipt(bad)
        bad = measurement("a", 1, cost_status="estimated")
        with self.assertRaises(BurdenValidationError):
            receipt(bad)

    def test_source_reported_and_configured_ceiling_are_opt_in(self):
        for status in ("source_reported", "configured_ceiling"):
            row = measurement("a", 3, cost_status=status)
            receipt(row)
            self.assertFalse(set_burden(["a"], ledger([row]), budget())["feasible"])
            accepted = budget(accepted=("measured", status))
            self.assertTrue(set_burden(["a"], ledger([row]), accepted)["feasible"])

    def test_shared_setup_charged_once_and_peak_not_summed(self):
        rows = [
            measurement("a", 1, group="dataset", peak=3),
            measurement("b", 2, group="dataset", peak=4),
            measurement("setup", 5, group="dataset", phase="shared_artifact_setup", peak=8),
        ]
        result = set_burden(
            ["a", "b"], ledger(rows),
            budget(limit=20, peak=8, phases=("shared_artifact_setup", "per_protocol_evaluation")),
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["additive"]["direct_cost"], 8)
        self.assertEqual(result["capacity"]["peak_vram"], 8)
        self.assertEqual(result["charged_artifact_group_ids"], ["dataset"])
        self.assertEqual(len(result["charged_measurement_ids"]), 3)

    def test_feature_extraction_charged_once_per_model_and_artifact(self):
        rows = [
            measurement("a", 4, group="dataset", phase="per_model_feature_extraction"),
            measurement("b", 4, group="dataset", phase="per_model_feature_extraction"),
        ]
        result = set_burden(
            ["a", "b"], ledger(rows),
            budget(limit=10, phases=("per_model_feature_extraction",)),
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["additive"]["direct_cost"], 4)
        self.assertEqual(len(result["charged_measurement_ids"]), 1)

    def test_missing_error_or_exclusion_and_unknown_never_zero(self):
        row = measurement("a", None, cost_status="not_measured")
        receipt(row)
        data = ledger([row])
        eligible, excluded = screen_candidates(["a", "absent"], data, budget(), missing_policy="exclude")
        self.assertEqual(eligible, [])
        self.assertTrue(any("not_measured" in reason for reason in excluded["a"]))
        self.assertTrue(any("not_measured" in reason for reason in excluded["absent"]))
        with self.assertRaises(MissingBurdenError):
            screen_candidates(["a"], data, budget(), missing_policy="error")

    def test_capacity_budget_and_canonical_constraints_are_hard(self):
        row = measurement("a", 11, peak=20, access="application")
        result = set_burden(["a"], ledger([row]), budget())
        self.assertFalse(result["feasible"])
        self.assertIn("additive_limit_exceeded:direct_cost", result["reasons"])
        self.assertIn("capacity_limit_exceeded:peak_vram", result["reasons"])
        self.assertTrue(any("not_allowed" in reason for reason in result["reasons"]))

    def test_required_phase_and_filter_are_fail_closed(self):
        row = measurement("a", 1)
        strict = budget(phases=("per_protocol_head_fit", "per_protocol_evaluation"))
        result = set_burden(["a"], ledger([row]), strict)
        self.assertFalse(result["feasible"])
        self.assertTrue(any("per_protocol_head_fit" in reason for reason in result["reasons"]))

    def test_greedy_deterministic_budget_feasible_and_identity_unique(self):
        rows = [measurement("a", 2), measurement("b", 1), measurement("c", 2)]
        identities = {"a": "same", "b": "same", "c": "other"}
        result = greedy_search(
            ["c", "b", "a"], identities, ledger(rows), budget(limit=4),
            evaluator({("a",): 2, ("b",): 2, ("c",): 3, ("b", "c"): 1}),
            max_probes=3,
        )
        self.assertEqual(result["trajectory"][0]["evaluation_ids"], ["b"])
        self.assertEqual(result["trajectory"][1]["evaluation_ids"], ["b", "c"])
        self.assertEqual(len(result["trajectory"]), 2)

    def test_exact_finds_result_greedy_misses(self):
        rows = [measurement("a", 6), measurement("b", 5), measurement("c", 5)]
        identities = {value: value for value in "abc"}
        losses = {("a",): 1, ("b",): 2, ("c",): 2, ("b", "c"): 0}
        greedy = greedy_search(
            list("abc"), identities, ledger(rows), budget(), evaluator(losses), max_probes=2
        )
        exact = exact_search(
            list("abc"), identities, ledger(rows), budget(), evaluator(losses),
            max_probes=2, max_subsets=10,
        )
        self.assertEqual(greedy["trajectory"][-1]["evaluation_ids"], ["a"])
        self.assertEqual(exact["optimum"]["evaluation_ids"], ["b", "c"])
        self.assertTrue(exact["globally_exact"])

    def test_exact_preflight_cap(self):
        rows = [measurement(str(index), 1) for index in range(8)]
        identities = {str(index): str(index) for index in range(8)}
        with self.assertRaises(SearchSpaceTooLargeError):
            exact_search(
                list(identities), identities, ledger(rows), budget(limit=100), evaluator({}),
                max_probes=4, max_subsets=10,
            )

    def test_random_prefixes_reproducible_feasible_and_identity_unique(self):
        rows = [measurement("a", 4), measurement("b", 4), measurement("c", 4)]
        identities = {"a": "same", "b": "same", "c": "other"}
        first = random_feasible_prefixes(
            list(identities), identities, ledger(rows), budget(limit=8),
            max_probes=3, repeats=3, seed=42,
        )
        second = random_feasible_prefixes(
            list(reversed(identities)), identities, ledger(rows), budget(limit=8),
            max_probes=3, repeats=3, seed=42,
        )
        self.assertEqual(first, second)
        for prefixes in first:
            for selected in prefixes:
                self.assertTrue(set_burden(selected, ledger(rows), budget(limit=8))["feasible"])
                self.assertEqual(len({identities[value] for value in selected}), len(selected))


if __name__ == "__main__":
    unittest.main()
