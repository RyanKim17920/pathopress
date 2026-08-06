from __future__ import annotations

import unittest

import numpy as np

from pathopress.metrics import (
    MEDAPE_EPSILON,
    absolute_percentage_errors,
    median_absolute_percentage_error,
)


class ScoreMetricTests(unittest.TestCase):
    def test_medape_excludes_zero_and_values_at_pinned_epsilon(self) -> None:
        actual = np.array([0.0, MEDAPE_EPSILON, -MEDAPE_EPSILON, 10.0, -20.0])
        predicted = np.array([100.0, 100.0, 100.0, 11.0, -24.0])
        np.testing.assert_allclose(
            absolute_percentage_errors(actual, predicted),
            [10.0, 20.0],
        )
        self.assertEqual(median_absolute_percentage_error(actual, predicted), 15.0)

    def test_medape_excludes_nonfinite_pairs_and_returns_nan_when_empty(self) -> None:
        self.assertTrue(
            np.isnan(
                median_absolute_percentage_error(
                    np.array([0.0, np.nan]), np.array([1.0, 2.0])
                )
            )
        )

    def test_medape_rejects_shape_or_epsilon_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal shape"):
            median_absolute_percentage_error([1.0], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "epsilon"):
            median_absolute_percentage_error([1.0], [1.0], epsilon=-1.0)


if __name__ == "__main__":
    unittest.main()
