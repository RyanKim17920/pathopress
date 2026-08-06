from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from pathopress.matrix import filter_matrix, load_scores, make_matrix


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/evaluation_cost_evidence.json"


def load_cost_plot_module():
    path = ROOT / "scripts/plot_evaluation_cost_evidence.py"
    spec = importlib.util.spec_from_file_location("plot_evaluation_cost_evidence", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluationCostEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
        cls.records = cls.payload["evaluations"]

    def test_registry_exactly_covers_retained_score_protocols(self) -> None:
        matrix, models, evaluation_ids = make_matrix(load_scores(ROOT / "data/scores.csv"))
        _, _, expected = filter_matrix(matrix, models, evaluation_ids)
        expected = sorted(expected)
        actual = [record["evaluation_id"] for record in self.records]
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 187)

    def test_registry_and_figure_denominators_are_consistent(self) -> None:
        total = len(self.records)
        summary = self.payload["summary"]
        self.assertEqual(summary["n_evaluations"], total)
        suite_total = sum(
            row["n_evaluations"] for row in summary["field_coverage_by_suite"].values()
        )
        self.assertEqual(suite_total, total)
        for suite, row in summary["field_coverage_by_suite"].items():
            denominator = row["n_evaluations"]
            for field, count in row.items():
                if field != "n_evaluations":
                    with self.subTest(suite=suite, field=field):
                        self.assertLessEqual(count, denominator)

        copy = load_cost_plot_module().denominator_copy(total)
        self.assertEqual(copy["coverage_title"], f"A. Evidence coverage (n={total:,})")
        self.assertIn(f"0/{total:,}", copy["missingness_footer"])

    def test_every_reported_fact_has_a_resolvable_source_and_locator(self) -> None:
        source_ids = {source["source_id"] for source in self.payload["sources"]}
        for record in self.records:
            for field, fact in record["facts"].items():
                with self.subTest(evaluation=record["evaluation_id"], field=field):
                    if fact["status"] == "reported":
                        self.assertIsNotNone(fact["value"])
                        self.assertIn(fact["source_id"], source_ids)
                        self.assertTrue(fact["source_url"].startswith("https://"))
                        self.assertTrue(fact["locator"])
                        self.assertTrue(fact["evidence_type"])
                        self.assertIn(fact["confidence"], {"high", "medium", "low"})
                        self.assertTrue(fact["scope"])
                    else:
                        self.assertIsNone(fact["value"])
                        self.assertTrue(fact["reason"])
                        self.assertTrue(fact["searched_source_ids"])
                        self.assertTrue(
                            all(source_id in source_ids for source_id in fact["searched_source_ids"])
                        )

    def test_no_numeric_cost_or_observed_runtime_is_invented(self) -> None:
        for record in self.records:
            self.assertEqual(record["facts"]["observed_runtime"]["status"], "not_reported")
            self.assertIsNone(record["facts"]["observed_runtime"]["value"])
            self.assertEqual(record["facts"]["dollar_cost"]["status"], "not_reported")
            self.assertIsNone(record["facts"]["dollar_cost"]["value"])
            self.assertEqual(record["facts"]["annotation_hours"]["status"], "not_reported")
            self.assertFalse(record["numeric_cost_curve_eligible"])
        conclusion = self.payload["summary"]["numeric_cost_curve"]
        self.assertFalse(conclusion["supported"])
        self.assertEqual(conclusion["eligible_evaluations"], 0)

    def test_repository_and_dataset_licenses_are_separate(self) -> None:
        pathobench = next(record for record in self.records if record["suite_id"] == "pathobench")
        self.assertEqual(pathobench["facts"]["software_license"]["value"], "CC-BY-NC-4.0")
        self.assertEqual(pathobench["facts"]["software_license"]["scope"], "benchmark_software")
        self.assertEqual(pathobench["facts"]["dataset_license"]["status"], "not_reported")

        hest = next(record for record in self.records if record["suite_id"] == "hest")
        self.assertEqual(hest["facts"]["dataset_license"]["value"], "CC-BY-NC-SA-4.0")

    def test_raw_split_and_metadata_numbers_are_preserved(self) -> None:
        by_id = {record["evaluation_id"]: record for record in self.records}
        thunder = by_id["thunder.bach.linear_probing"]["facts"]
        self.assertEqual(
            thunder["sample_count"]["value"],
            {"train": 218, "validation": 50, "test": 132, "total": 400},
        )
        self.assertEqual(thunder["acquisition_scale"]["value"]["microns_per_pixel"], 0.42)
        pathorob = by_id["pathorob.camelyon.robustness_index"]["facts"]["sample_count"]
        self.assertEqual(pathorob["value"]["raw_metadata_rows"], 22402)
        self.assertEqual(pathorob["value"]["excluded_ood_rows"], 2002)
        self.assertEqual(pathorob["value"]["evaluated_rows"], 20400)

    def test_pre_error_tiers_are_explicitly_not_cost_claims(self) -> None:
        tiers = self.payload["summary"]["pre_error_feasibility_tier_counts"]
        self.assertEqual(sum(tiers.values()), 187)
        self.assertEqual(tiers["tier_1_direct_small_labeled"], 5)
        for record in self.records:
            definition = record["pre_error_feasibility"]
            self.assertIn("protocol metadata only", definition["derivation_timing"])
            self.assertIn("not measured", definition["claim_boundary"])

    def test_checked_in_registry_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_json = Path(temporary) / "registry.json"
            output_csv = Path(temporary) / "registry.csv"
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts/build_evaluation_cost_evidence.py"),
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            regenerated = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(regenerated, self.payload)
            self.assertEqual(
                output_csv.read_text(encoding="utf-8"),
                (ROOT / "data/evaluation_cost_evidence.csv").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
