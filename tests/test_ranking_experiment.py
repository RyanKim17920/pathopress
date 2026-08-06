import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_ranking_preservation", ROOT / "experiments/run_ranking_preservation.py"
)
EXPERIMENT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXPERIMENT)


class RankingExperimentAggregationTests(unittest.TestCase):
    def test_pairwise_pools_folds_per_evaluation_before_median(self) -> None:
        common = {"margin": 0.0, "suite_id": "suite", "metric": "auc", "n_predicted_ties": 0}
        rows = [
            {**common, "evaluation_id": "a", "n_pairs": 2, "n_correct": 1},
            {**common, "evaluation_id": "a", "n_pairs": 3, "n_correct": 3},
            {**common, "evaluation_id": "b", "n_pairs": 4, "n_correct": 1},
            {**common, "evaluation_id": "ignored", "n_pairs": 0, "n_correct": 0},
        ]

        summary, evaluations = EXPERIMENT.summarize_pairwise(rows)

        by_id = {row["evaluation_id"]: row for row in evaluations}
        self.assertAlmostEqual(by_id["a"]["accuracy"], 4 / 5)
        self.assertAlmostEqual(by_id["b"]["accuracy"], 1 / 4)
        self.assertEqual(summary["0.0"]["median_accuracy"], 0.525)
        self.assertEqual(summary["0.0"]["pooled_accuracy"], 0.555556)
        self.assertEqual(summary["0.0"]["n_groups"], 3)

    def test_top_recovery_pools_overlap_and_k_per_evaluation(self) -> None:
        common = {"top_fraction": 0.1, "suite_id": "suite", "metric": "auc"}
        rows = [
            {**common, "evaluation_id": "a", "k": 1, "overlap": 1},
            {**common, "evaluation_id": "a", "k": 2, "overlap": 1},
            {**common, "evaluation_id": "b", "k": 4, "overlap": 1},
        ]

        summary, evaluations = EXPERIMENT.summarize_top(rows)

        by_id = {row["evaluation_id"]: row for row in evaluations}
        self.assertAlmostEqual(by_id["a"]["recovery"], 2 / 3)
        self.assertAlmostEqual(by_id["b"]["recovery"], 1 / 4)
        self.assertEqual(summary["0.1"]["median_recovery"], 0.458333)
        self.assertEqual(summary["0.1"]["pooled_recovery"], 0.428571)


if __name__ == "__main__":
    unittest.main()
