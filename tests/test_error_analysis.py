import unittest

import numpy as np

from pathopress.error_analysis import (
    best_neighbor_rows,
    low_rank_r2,
    pairwise_abs_correlation,
    spearman_test,
)


class ErrorAnalysisTests(unittest.TestCase):
    def test_low_rank_r2_has_expected_shapes_and_improves_with_rank(self):
        matrix = np.array([[1.0, 2.0, np.nan], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
        r1_columns = low_rank_r2(matrix, rank=1, axis=0)
        r2_columns = low_rank_r2(matrix, rank=2, axis=0)
        self.assertEqual(r1_columns.shape, (3,))
        self.assertTrue(np.all(r2_columns >= r1_columns - 1e-12))

    def test_best_neighbor_uses_absolute_correlation_and_shared_count(self):
        matrix = np.array(
            [[1.0, 4.0, 1.0], [2.0, 3.0, 2.0], [3.0, 2.0, 3.0], [4.0, 1.0, 4.0]]
        )
        correlation, shared = pairwise_abs_correlation(matrix, axis=0, min_shared=3)
        rows = best_neighbor_rows(correlation, shared)
        self.assertTrue(np.isclose(rows[0]["best_neighbor_abs_r"], 1.0))
        self.assertEqual(rows[0]["best_neighbor_shared"], 4)

    def test_spearman_test_filters_non_finite_pairs(self):
        result = spearman_test(
            np.array([1.0, 2.0, np.nan, 4.0]), np.array([2.0, 4.0, 9.0, 8.0])
        )
        self.assertEqual(result["n"], 3)
        self.assertTrue(np.isclose(result["rho"], 1.0))
