import unittest

import numpy as np

from pathopress.completion import ValidationResult, complete, complete_soft_impute, validate


class CompletionTests(unittest.TestCase):
    def test_complete_preserves_observed_cells_and_bounds_all_predictions(self) -> None:
        matrix = np.array(
            [
                [12.0, 25.0, np.nan, 46.0, 55.0],
                [22.0, np.nan, 38.0, 50.0, 61.0],
                [np.nan, 35.0, 47.0, 59.0, 68.0],
                [41.0, 48.0, 57.0, np.nan, 76.0],
                [53.0, 61.0, 69.0, 78.0, np.nan],
                [64.0, 72.0, np.nan, 87.0, 94.0],
            ]
        )
        observed = np.isfinite(matrix)

        completed = complete(matrix)

        self.assertEqual(completed.shape, matrix.shape)
        np.testing.assert_array_equal(completed[observed], matrix[observed])
        self.assertTrue(np.isfinite(completed).all())
        self.assertTrue(np.all(completed >= 0.0))
        self.assertTrue(np.all(completed <= 100.0))

    def test_complete_rejects_empty_or_non_matrix_inputs(self) -> None:
        invalid_inputs = (
            np.array([]),
            np.array([10.0, 20.0]),
            np.full((2, 2), np.nan),
        )

        for matrix in invalid_inputs:
            with self.subTest(shape=matrix.shape):
                with self.assertRaisesRegex(ValueError, "non-empty 2D array"):
                    complete(matrix)

    def test_complete_rejects_all_missing_rows_or_columns(self) -> None:
        unsupported_matrices = (
            np.array([[50.0, np.nan], [60.0, np.nan]]),
            np.array([[50.0, 60.0], [np.nan, np.nan]]),
        )

        for matrix in unsupported_matrices:
            with self.subTest(matrix=matrix.tolist()):
                with self.assertRaises(ValueError):
                    complete(matrix)

    def test_complete_can_explicitly_allow_an_empty_target_row(self) -> None:
        matrix = np.array(
            [[np.nan, np.nan], [60.0, 70.0], [80.0, 90.0]]
        )

        completed = complete(matrix, rank=0, allow_empty_rows=True)

        self.assertTrue(np.isfinite(completed).all())
        observed = np.isfinite(matrix)
        np.testing.assert_array_equal(completed[observed], matrix[observed])

    def test_complete_supports_bias_only_rank_zero(self) -> None:
        matrix = np.array([[50.0, np.nan], [60.0, 70.0], [80.0, 90.0]])
        completed = complete(matrix, rank=0)
        self.assertTrue(np.isfinite(completed).all())
        np.testing.assert_array_equal(completed[np.isfinite(matrix)], matrix[np.isfinite(matrix)])

    def test_complete_rejects_negative_rank(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            complete(np.array([[50.0]]), rank=-1)

    def test_soft_impute_supports_raw_and_logit_spaces(self) -> None:
        matrix = np.array([[50.0, np.nan], [60.0, 70.0], [80.0, 90.0]])
        observed = np.isfinite(matrix)
        for transform in ("identity", "logit"):
            with self.subTest(transform=transform):
                completed = complete_soft_impute(matrix, rank=1, transform=transform)
                self.assertTrue(np.isfinite(completed).all())
                np.testing.assert_array_equal(completed[observed], matrix[observed])

    def test_soft_impute_validates_rank_and_transform(self) -> None:
        matrix = np.array([[50.0, 60.0], [70.0, 80.0]])
        with self.assertRaisesRegex(ValueError, "at least 1"):
            complete_soft_impute(matrix, rank=0)
        with self.assertRaisesRegex(ValueError, "identity.*logit"):
            complete_soft_impute(matrix, rank=1, transform="sqrt")


class ValidationTests(unittest.TestCase):
    def test_validate_holds_out_observations_and_returns_finite_errors(self) -> None:
        # Dense support ensures every repeat can remove the requested cells while
        # retaining enough row/column observations for matrix completion.
        rows = np.arange(6, dtype=float)[:, None]
        columns = np.arange(5, dtype=float)[None, :]
        matrix = 20.0 + rows * 8.0 + columns * 3.0

        result = validate(matrix, holdout_fraction=0.2, repeats=3, seed=7)

        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.n_predictions, 18)
        self.assertTrue(np.isfinite(result.median_absolute_error))
        self.assertTrue(np.isfinite(result.mean_absolute_error))
        self.assertGreaterEqual(result.median_absolute_error, 0.0)
        self.assertGreaterEqual(result.mean_absolute_error, 0.0)

    def test_validate_rejects_a_matrix_too_sparse_for_any_holdout(self) -> None:
        matrix = np.array(
            [
                [10.0, np.nan, np.nan],
                [np.nan, 20.0, np.nan],
                [np.nan, np.nan, 30.0],
            ]
        )

        with self.assertRaisesRegex(ValueError, "too sparse"):
            validate(matrix, repeats=1)


if __name__ == "__main__":
    unittest.main()
