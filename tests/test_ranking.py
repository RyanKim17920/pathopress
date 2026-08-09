import unittest

import numpy as np

from pathopress.ranking import (
    pairwise_ranking_accuracy,
    _column_margins,
    top_fraction_recovery,
)


class PairwiseRankingTests(unittest.TestCase):
    def test_counts_pairs_with_a_holdout_and_treats_predicted_ties_as_wrong(self) -> None:
        actual = np.array([[90.0], [80.0], [70.0], [60.0]])
        predicted = np.array([[90.0], [75.0], [75.0], [60.0]])
        heldout = np.array([[False], [True], [True], [False]])

        result = pairwise_ranking_accuracy(actual, predicted, heldout)
        column = result.columns[0]

        self.assertEqual(column.n_models, 4)
        self.assertEqual(column.n_heldout_models, 2)
        self.assertEqual(column.n_pairs, 5)  # The seen/seen pair is excluded.
        self.assertEqual(column.n_correct, 4)
        self.assertEqual(column.n_predicted_ties, 1)
        self.assertAlmostEqual(column.accuracy, 0.8)
        self.assertAlmostEqual(result.median_accuracy, 0.8)
        self.assertAlmostEqual(result.pooled_accuracy, 0.8)

    def test_accepts_distinct_column_margins(self) -> None:
        actual = np.array(
            [
                [90.0, 90.0],
                [87.0, 87.0],
                [80.0, 80.0],
            ]
        )
        predicted = np.array(
            [
                [85.0, 85.0],
                [87.0, 87.0],
                [80.0, 80.0],
            ]
        )
        heldout = np.array(
            [
                [True, True],
                [False, False],
                [False, False],
            ]
        )

        result = pairwise_ranking_accuracy(actual, predicted, heldout, margin=[0.0, 5.0])

        self.assertEqual(result.columns[0].n_pairs, 2)
        self.assertEqual(result.columns[0].n_correct, 1)
        self.assertEqual(result.columns[1].n_pairs, 1)
        self.assertEqual(result.columns[1].n_correct, 1)
        self.assertAlmostEqual(result.median_accuracy, 0.75)
        self.assertAlmostEqual(result.pooled_accuracy, 2.0 / 3.0)

    def test_reports_ineligible_columns_without_polluting_median(self) -> None:
        actual = np.array([[3.0, 10.0], [2.0, 9.0], [1.0, np.nan]])
        predicted = actual.copy()
        heldout = np.array([[True, False], [False, False], [False, False]])

        result = pairwise_ranking_accuracy(actual, predicted, heldout)

        self.assertEqual(len(result.columns), 2)
        self.assertEqual(result.n_eligible_columns, 1)
        self.assertTrue(np.isnan(result.columns[1].accuracy))
        self.assertEqual(result.median_accuracy, 1.0)

    def test_validates_shapes_and_margins(self) -> None:
        actual = np.ones((2, 2))
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            pairwise_ranking_accuracy(actual, np.ones((2, 1)), np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, "length 2"):
            pairwise_ranking_accuracy(actual, actual, np.ones_like(actual), margin=[0.0])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            pairwise_ranking_accuracy(actual, actual, np.ones_like(actual), margin=-1.0)


class RelativeMarginTests(unittest.TestCase):
    """Tests for dispersion-relative margin modes."""

    def test_relative_margin_mode_produces_different_per_column_margins(self) -> None:
        """SD and IQR modes produce per-column margins that differ from absolute."""
        actual = np.array(
            [
                [100.0, 100.0],
                [90.0, 90.0],
                [80.0, 80.0],
                [70.0, 70.0],
                [60.0, 60.0],
            ]
        )
        predicted = np.array(
            [
                [99.0, 95.0],
                [89.0, 85.0],
                [79.0, 75.0],
                [69.0, 65.0],
                [59.0, 55.0],
            ]
        )
        heldout = np.ones_like(actual, dtype=bool)

        # Absolute margin=5
        res_abs = pairwise_ranking_accuracy(actual, predicted, heldout, margin=5.0)
        # Relative margin=1.0 (1 SD)
        res_sd = pairwise_ranking_accuracy(
            actual, predicted, heldout, margin=1.0, margin_relative_to="sd"
        )
        # Relative margin=1.0 (1 IQR)
        res_iqr = pairwise_ranking_accuracy(
            actual, predicted, heldout, margin=1.0, margin_relative_to="iqr"
        )

        # The absolute margin is the same for both columns
        self.assertEqual(res_abs.columns[0].margin, res_abs.columns[1].margin)

        # The SD-based margins should differ from the absolute value of 5.0
        # Column SD for [100,90,80,70,60] = ~14.14, so margin ~14.14 != 5.0
        self.assertNotEqual(res_sd.columns[0].margin, 5.0)
        self.assertIsInstance(res_sd.columns[0].margin, float)

        # The IQR-based margins should differ from absolute too
        # IQR for [100,90,80,70,60] = 20.0, so margin ~20.0 != 5.0
        self.assertNotEqual(res_iqr.columns[0].margin, 5.0)
        self.assertIsInstance(res_iqr.columns[0].margin, float)

        # With SD ~14.14, the relative margin is larger than 5 so fewer pairs pass
        self.assertLessEqual(res_sd.columns[0].n_pairs, res_abs.columns[0].n_pairs)

    def test_degenerate_zero_dispersion_columns_get_margin_zero(self) -> None:
        """A column with all identical scores has dispersion 0, margin becomes 0.0."""
        actual = np.array(
            [
                [50.0, 100.0],
                [50.0, 90.0],
                [50.0, 80.0],
                [50.0, 70.0],
            ]
        )
        predicted = actual.copy()
        heldout = np.ones_like(actual, dtype=bool)

        result = pairwise_ranking_accuracy(
            actual, predicted, heldout, margin=1.0, margin_relative_to="sd"
        )

        # Column 0 has all identical values (sd=0), so margin should be 0.0
        self.assertEqual(result.columns[0].margin, 0.0)

        # No NaN margins
        for col in result.columns:
            self.assertTrue(np.isfinite(col.margin))

    def test_margin_relative_to_none_is_numerically_identical_to_default(self) -> None:
        """margin_relative_to='none' produces byte-identical results to no relative mode."""
        actual = np.array(
            [
                [100.0, 95.0],
                [90.0, 85.0],
                [80.0, 75.0],
                [70.0, 65.0],
            ]
        )
        predicted = np.array(
            [
                [99.0, 94.0],
                [89.0, 84.0],
                [79.0, 74.0],
                [69.0, 64.0],
            ]
        )
        heldout = np.ones_like(actual, dtype=bool)

        res_default = pairwise_ranking_accuracy(actual, predicted, heldout, margin=3.0)
        res_none = pairwise_ranking_accuracy(
            actual, predicted, heldout, margin=3.0, margin_relative_to="none"
        )

        self.assertEqual(res_default.n_pairs, res_none.n_pairs)
        self.assertEqual(res_default.n_correct, res_none.n_correct)
        self.assertEqual(res_default.n_eligible_columns, res_none.n_eligible_columns)
        self.assertAlmostEqual(res_default.median_accuracy, res_none.median_accuracy)
        self.assertAlmostEqual(res_default.pooled_accuracy, res_none.pooled_accuracy)

        for col_a, col_b in zip(res_default.columns, res_none.columns):
            self.assertEqual(col_a.margin, col_b.margin)
            self.assertEqual(col_a.n_pairs, col_b.n_pairs)
            self.assertEqual(col_a.n_correct, col_b.n_correct)

    def test_column_margins_scalar_mode(self) -> None:
        """_column_margins with scalar margin broadcasts correctly."""
        margins = _column_margins(5.0, 3)
        np.testing.assert_array_equal(margins, [5.0, 5.0, 5.0])

    def test_column_margins_relative_mode_with_degenerate_column(self) -> None:
        """Degenerate column (single finite value) gets margin 0.0."""
        actual = np.array(
            [
                [10.0, np.nan],
                [20.0, np.nan],
                [np.nan, np.nan],
            ]
        )
        margins = _column_margins(
            1.0, 2, margin_relative_to="sd", actual_array=actual
        )
        # Column 1 has no finite values, margin should be 0.0
        self.assertEqual(margins[1], 0.0)
        # No NaN anywhere
        self.assertTrue(np.isfinite(margins).all())


class TopFractionRecoveryTests(unittest.TestCase):
    def test_recovers_top_sets_and_aggregates_by_column_median(self) -> None:
        actual = np.array(
            [
                [100.0, 100.0],
                [90.0, 90.0],
                [80.0, 80.0],
                [70.0, 70.0],
            ]
        )
        predicted = np.array(
            [
                [100.0, 70.0],
                [90.0, 90.0],
                [80.0, 80.0],
                [70.0, 100.0],
            ]
        )
        heldout = np.ones_like(actual, dtype=bool)

        result = top_fraction_recovery(
            actual, predicted, heldout, top_fraction=0.5
        )

        self.assertEqual(result.n_eligible_columns, 2)
        self.assertEqual([column.k for column in result.columns], [2, 2])
        self.assertEqual([column.overlap for column in result.columns], [2, 1])
        self.assertAlmostEqual(result.median_recovery, 0.75)
        self.assertAlmostEqual(result.pooled_recovery, 0.75)
        self.assertEqual(result.total_k, 4)
        self.assertEqual(result.total_overlap, 3)

    def test_uses_ceiling_for_top_set_size_and_requires_a_holdout(self) -> None:
        actual = np.array([[5.0, 5.0], [4.0, 4.0], [3.0, 3.0]])
        predicted = actual.copy()
        heldout = np.array([[True, False], [False, False], [False, False]])

        result = top_fraction_recovery(
            actual, predicted, heldout, top_fraction=0.34
        )

        self.assertEqual(result.columns[0].k, 2)
        self.assertEqual(result.columns[0].recovery, 1.0)
        self.assertEqual(result.columns[1].k, 0)
        self.assertTrue(np.isnan(result.columns[1].recovery))
        self.assertEqual(result.n_eligible_columns, 1)

    def test_validates_fraction(self) -> None:
        matrix = np.ones((2, 2))
        for fraction in (0.0, -0.1, 1.1, float("nan")):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "in \\(0, 1\\]"):
                    top_fraction_recovery(
                        matrix, matrix, np.ones_like(matrix), top_fraction=fraction
                    )


if __name__ == "__main__":
    unittest.main()
