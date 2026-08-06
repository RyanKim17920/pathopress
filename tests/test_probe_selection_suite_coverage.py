from __future__ import annotations

import unittest

import numpy as np

from experiments.run_probe_selection import _suite_coverage_summary


class ProbeSelectionSuiteCoverageTests(unittest.TestCase):
    def test_counts_models_with_observations_in_every_represented_suite(self) -> None:
        matrix = np.asarray(
            [
                [80.0, np.nan, 72.0, np.nan],
                [np.nan, 81.0, np.nan, 73.0],
                [80.0, np.nan, np.nan, np.nan],
            ]
        )
        summary = _suite_coverage_summary(
            matrix,
            ["model-a", "model-b", "model-c"],
            ["suite-a.task-1", "suite-a.task-2", "suite-b.task-1", "suite-b.task-2"],
            {
                "suite-a.task-1": "suite-a",
                "suite-a.task-2": "suite-a",
                "suite-b.task-1": "suite-b",
                "suite-b.task-2": "suite-b",
            },
        )

        self.assertEqual(summary["represented_suites"], ["suite-a", "suite-b"])
        self.assertEqual(summary["n_represented_suites"], 2)
        self.assertEqual(
            summary["models_with_all_represented_suites"], ["model-a", "model-b"]
        )
        self.assertEqual(summary["n_models_with_all_represented_suites"], 2)

    def test_rejects_shape_or_metadata_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "matrix shape"):
            _suite_coverage_summary(
                np.asarray([[1.0]]),
                ["model-a", "model-b"],
                ["suite-a.task"],
                {"suite-a.task": "suite-a"},
            )
        with self.assertRaisesRegex(ValueError, "missing suite metadata"):
            _suite_coverage_summary(
                np.asarray([[1.0]]),
                ["model-a"],
                ["suite-a.task"],
                {},
            )


if __name__ == "__main__":
    unittest.main()
