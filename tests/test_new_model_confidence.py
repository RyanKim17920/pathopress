import copy
import json
import tempfile
import unittest
from pathlib import Path

from pathopress.new_model_confidence import (
    build_new_model_confidence_artifact,
    calibrated_new_model_interval,
    calibration_probe_count,
)


def fixture_records():
    rows = []
    for model_index in range(7):
        for k in (1, 3, 5, 10):
            for evaluation, suite, offset in (("e1", "s1", 0.0), ("e2", "s2", 1.0)):
                error = 1.0 + model_index / 2 + k / 10 + offset
                rows.append({
                    "target_model_id": f"m{model_index}",
                    "evaluation_id": evaluation,
                    "suite_id": suite,
                    "k": k,
                    "seed": 0,
                    "source": "leave_one_model_out_probe",
                    "actual": 50.0,
                    "predicted": 50.0 + error,
                    "absolute_error": error,
                    "same_suite_probe_count": 1,
                })
    return rows


class NewModelConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.scores = Path(self.temporary.name) / "scores.csv"
        self.scores.write_text("fixture\n", encoding="utf-8")

    def build(self, rows=None):
        return build_new_model_confidence_artifact(
            fixture_records() if rows is None else rows,
            self.scores,
            min_evaluation_models=5,
            min_context_models=5,
        )

    def test_supported_probe_buckets_are_conservative(self):
        self.assertEqual([calibration_probe_count(k) for k in (1, 2, 3, 4, 5, 9, 10, 20)], [1, 1, 3, 3, 5, 5, 10, 10])
        with self.assertRaisesRegex(ValueError, "at least one"):
            calibration_probe_count(0)

    def test_every_crossfit_fold_excludes_its_target_model(self):
        artifact, audited = self.build()
        self.assertEqual(set(artifact["crossfit_group_audit"]), {f"m{i}" for i in range(7)})
        for model, fold in artifact["crossfit_group_audit"].items():
            self.assertTrue(fold["target_absent"])
            self.assertEqual(fold["excluded_target_model"], model)
            self.assertNotIn(model, fold["training_model_ids"])
            self.assertEqual(len(fold["training_model_ids"]), 6)
        for row in audited:
            self.assertEqual(row["target_model_id"], row["calibration_excluded_target_model"])

    def test_target_hidden_values_do_not_change_its_crossfit_interval(self):
        original = fixture_records()
        changed = copy.deepcopy(original)
        for row in changed:
            if row["target_model_id"] == "m0":
                row["absolute_error"] = 99.0
                row["actual"] = 0.0
                row["predicted"] = 99.0
        _, first = self.build(original)
        _, second = self.build(changed)
        first_m0 = [(r["crossfit_risk"], r["crossfit_conformal_scale"]) for r in first if r["target_model_id"] == "m0"]
        second_m0 = [(r["crossfit_risk"], r["crossfit_conformal_scale"]) for r in second if r["target_model_id"] == "m0"]
        self.assertEqual(first_m0, second_m0)

    def test_interval_surfaces_group_counts_and_scope(self):
        artifact, _ = self.build()
        result = calibrated_new_model_interval(50.0, "e1", "s1", ["s1", "s2", "s1"], artifact)
        self.assertEqual(result["confidence_status"], "calibrated_new_model")
        self.assertEqual(result["calibration_k"], 3)
        self.assertEqual(result["calibration_scope"], "evaluation+suite_same_probe")
        self.assertEqual(result["calibration_evaluation_models"], 7)
        self.assertGreater(result["calibration_evaluation_predictions"], 0)
        self.assertGreater(result["calibration_context_models"], 0)
        self.assertLessEqual(result["lower_90"], 50.0)
        self.assertGreaterEqual(result["upper_90"], 50.0)

    def test_unsupported_column_abstention_is_deterministic(self):
        artifact, _ = self.build()
        first = calibrated_new_model_interval(50.0, "unknown", "s1", ["s1"], artifact)
        second = calibrated_new_model_interval(50.0, "unknown", "s1", ["s1"], artifact)
        self.assertEqual(first, second)
        self.assertEqual(first["confidence_status"], "abstained_unsupported_column")
        self.assertEqual(first["abstention_reason"], "evaluation_missing_from_calibration")

    def test_committed_artifact_labels_empirical_not_clinical_coverage(self):
        root = Path(__file__).resolve().parents[1]
        artifact = json.loads((root / "experiments/new_model_confidence_rank1.json").read_text())
        self.assertGreaterEqual(artifact["crossfit_metrics"]["overall"]["interval_coverage"], 0.90)
        self.assertFalse(artifact["applicability"]["clinical_guarantee"])
        self.assertIn("retrospective", " ".join(artifact["limitations"]).lower())


if __name__ == "__main__":
    unittest.main()
