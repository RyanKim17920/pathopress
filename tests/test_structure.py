import json
from pathlib import Path
import unittest

import numpy as np

from pathopress.structure import (
    best_neighbor_ols,
    classical_mds_from_correlations,
    complete_submatrix_for_count,
    find_largest_complete_submatrix,
    pairwise_correlations,
    singular_summary,
)


ROOT = Path(__file__).resolve().parents[1]


class CompleteSubmatrixTests(unittest.TestCase):
    def test_top_coverage_selection_produces_a_complete_block(self) -> None:
        matrix = np.asarray(
            [
                [1.0, 2.0, 3.0, np.nan],
                [2.0, 3.0, 4.0, np.nan],
                [3.0, 4.0, np.nan, 5.0],
                [4.0, 5.0, np.nan, 6.0],
            ]
        )
        rows, columns = complete_submatrix_for_count(matrix, 2)
        self.assertEqual(columns, [0, 1])
        self.assertEqual(rows, [0, 1, 2, 3])
        largest_rows, largest_columns = find_largest_complete_submatrix(matrix, min_evaluations=2)
        self.assertTrue(np.isfinite(matrix[np.ix_(largest_rows, largest_columns)]).all())

    def test_rank_one_centered_block_has_stable_rank_one(self) -> None:
        vector = np.arange(1.0, 6.0)[:, None]
        matrix = vector @ np.asarray([[1.0, 2.0, 4.0]]) + np.asarray([[10.0, 20.0, 30.0]])
        summary = singular_summary(matrix)
        self.assertAlmostEqual(summary["stable_rank"], 1.0)
        self.assertAlmostEqual(summary["var_rank1"], 1.0)
        self.assertAlmostEqual(summary["var_rank2"], 1.0)

    def test_probe_mds_annotations_match_top_ten_probe_data(self) -> None:
        annotations = json.loads(
            (ROOT / "experiments/structure_analysis/probe_mds_annotations.json").read_text()
        )["annotations"]
        probes = json.loads(
            (ROOT / "experiments/probe_selection_results_rank1.json").read_text()
        )["all_known_greedy"][:10]
        self.assertEqual([row["rank"] for row in annotations], list(range(1, 11)))
        self.assertEqual(
            [row["evaluation_id"] for row in annotations],
            [row["added_evaluation_id"] for row in probes],
        )
        self.assertEqual(len({tuple(row["offset_points"]) for row in annotations}), 10)
        with np.load(ROOT / "experiments/structure_analysis/correlation_mds.npz") as data:
            coordinates = data["coordinates"]
            evaluation_ids = data["evaluation_ids"].astype(str).tolist()
        by_id = {evaluation: coordinates[index] for index, evaluation in enumerate(evaluation_ids)}
        for row in annotations:
            np.testing.assert_allclose(
                [row["mds_x"], row["mds_y"]], by_id[row["evaluation_id"]]
            )


class CorrelationTests(unittest.TestCase):
    def setUp(self) -> None:
        first = np.linspace(10.0, 80.0, 8)
        second = first.copy()
        third = first[::-1]
        fourth = np.asarray([20.0, 30.0, 25.0, 45.0, 50.0, 65.0, 55.0, 75.0])
        self.matrix = np.column_stack([first, second, third, fourth])
        self.ids = ["first", "second", "third", "fourth"]

    def test_pairwise_correlations_and_best_neighbor_use_absolute_r_squared(self) -> None:
        correlations, counts, _means, _stds = pairwise_correlations(self.matrix, min_shared=5)
        self.assertAlmostEqual(correlations[0, 1], 1.0)
        self.assertEqual(counts[0, 1], 8)
        result = best_neighbor_ols(self.matrix, self.ids, min_shared=5)
        self.assertEqual(result["first"]["best_neighbor"], "second")
        self.assertAlmostEqual(result["first"]["max_r2"], 1.0)
        self.assertAlmostEqual(result["first"]["medae"], 0.0)

    def test_classical_mds_is_finite_centered_and_two_dimensional(self) -> None:
        correlations, _counts, _means, _stds = pairwise_correlations(self.matrix, min_shared=5)
        coordinates = classical_mds_from_correlations(correlations)
        self.assertEqual(coordinates.shape, (4, 2))
        self.assertTrue(np.isfinite(coordinates).all())
        np.testing.assert_allclose(coordinates.mean(axis=0), 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
