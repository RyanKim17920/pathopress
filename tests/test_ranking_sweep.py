"""Tests for ranking preservation sweep wiring and JSON safety."""

from __future__ import annotations

import importlib.util
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_ranking_preservation", ROOT / "experiments/run_ranking_preservation.py"
)
EXPERIMENT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXPERIMENT)


class MarginSweepWiringTests(unittest.TestCase):
    """BUG 1 - margin sweep is actually computed and non-null in the output artifact."""

    @classmethod
    def setUpClass(cls) -> None:
        output_path = ROOT / "experiments/ranking_preservation_rank1.json"
        cls.payload = json.loads(output_path.read_text())

    def test_margin_sweep_is_not_null(self) -> None:
        """The margin_sweep field must be populated after running main()."""
        sweep = self.payload.get("margin_sweep")
        self.assertIsNotNone(sweep, "margin_sweep must not be null; it should contain sweep results")

    def test_margin_sweep_has_all_points(self) -> None:
        sweep = self.payload["margin_sweep"]
        self.assertEqual(len(sweep["sweep_points"]), 12)
        self.assertEqual(len(sweep["greedy"]), 12)
        self.assertEqual(len(sweep["random"]), 12)

    def test_margin_sweep_greedy_vs_random_gap(self) -> None:
        sweep = self.payload["margin_sweep"]
        for i, sp in enumerate(sweep["sweep_points"]):
            if sp["margin_type"] != "absolute":
                continue
            g_med = sweep["greedy"][i]["median_accuracy"]
            r_med = sweep["random"][i]["median_accuracy"]
            self.assertGreater(
                g_med, r_med,
                f"greedy median ({g_med}) must exceed random median ({r_med}) at margin {sp['margin']}",
            )

    def test_key_absolute_values(self) -> None:
        """Verify specific expected values from the bug report."""
        sweep = self.payload["margin_sweep"]
        # abs 0.0 -> 17,159 pairs, 187 cols, greedy median 0.679
        g0 = sweep["greedy"][0]
        self.assertEqual(g0["n_pairs"], 17159)
        self.assertEqual(g0["n_eligible_columns"], 187)
        self.assertAlmostEqual(g0["median_accuracy"], 0.679, places=2)

        # abs 5.0 -> 6,048 pairs, 148 cols, greedy median 0.878
        g5 = sweep["greedy"][4]
        self.assertEqual(g5["n_pairs"], 6048)
        self.assertEqual(g5["n_eligible_columns"], 148)
        self.assertAlmostEqual(g5["median_accuracy"], 0.878, places=2)

        # sd 1.0 -> 8,232 pairs, greedy median 0.750 (index 10)
        sd1 = sweep["greedy"][10]
        self.assertEqual(sd1["n_pairs"], 8232)
        self.assertAlmostEqual(sd1["median_accuracy"], 0.750, places=2)

        # iqr 1.0 -> 6,076 pairs, greedy median 0.778 (index 11)
        iqr1 = sweep["greedy"][11]
        self.assertEqual(iqr1["n_pairs"], 6076)
        self.assertAlmostEqual(iqr1["median_accuracy"], 0.778, places=2)

    def test_random_baseline_also_swept(self) -> None:
        """The random baseline must also be swept (10 repeats), not just greedy."""
        sweep = self.payload["margin_sweep"]
        self.assertEqual(sweep["mode"], "any_candidate")
        self.assertEqual(sweep["k"], 10)
        for i, r in enumerate(sweep["random"]):
            self.assertIn("median_accuracy", r)
            self.assertIn("pooled_accuracy", r)


class JsonNanSafetyTests(unittest.TestCase):
    """BUG 2 - no NaN floats may reach the JSON artifact."""

    def test_sanitize_for_json_replaces_nan_in_nested_structures(self) -> None:
        import numpy as np
        obj = {
            "a": 1.0,
            "b": {"c": float("nan"), "d": [float("nan"), 2.0, np.float64("nan")]},
        }
        result = EXPERIMENT._sanitize_for_json(obj)
        self.assertIsNone(result["b"]["c"])
        self.assertIsNone(result["b"]["d"][0])
        self.assertEqual(result["b"]["d"][1], 2.0)
        self.assertIsNone(result["b"]["d"][2])

    def test_sanitize_for_json_preserves_finite_values(self) -> None:
        import numpy as np
        obj = {"a": 0.5, "b": np.float64(0.25), "c": [1, 2, 3]}
        result = EXPERIMENT._sanitize_for_json(obj)
        self.assertAlmostEqual(result["a"], 0.5)
        self.assertAlmostEqual(result["b"], 0.25)
        self.assertEqual(result["c"], [1, 2, 3])

    def test_output_artifact_has_no_nan(self) -> None:
        """The actual JSON output file must not contain NaN."""
        output_path = ROOT / "experiments/ranking_preservation_rank1.json"
        text = output_path.read_text()
        # NaN is not valid JSON; the file should parse without error.
        data = json.loads(text)

        def check_no_nan(obj, path=""):
            if isinstance(obj, float) and math.isnan(obj):
                self.fail(f"NaN found at {path}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    check_no_nan(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_no_nan(v, f"{path}[{i}]")

        check_no_nan(data)


if __name__ == "__main__":
    unittest.main()
