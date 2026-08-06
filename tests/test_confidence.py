import unittest

import numpy as np

from pathopress.confidence import (
    conformal_interval,
    coverage_width,
    crossfit_error_risk,
    risk_coverage_curve,
    spearman_uncertainty_error,
    stack_features,
    structural_support_features_for_cells,
    uncertainty_tercile_errors,
)


class ConfidencePrimitiveTests(unittest.TestCase):
    def test_spearman_orders_error_by_uncertainty(self) -> None:
        actual = np.zeros(5)
        predicted = np.arange(5, dtype=float)
        self.assertAlmostEqual(
            spearman_uncertainty_error(actual, predicted, np.arange(5, dtype=float)), 1.0
        )

    def test_risk_coverage_keeps_lowest_uncertainty_first(self) -> None:
        actual = np.zeros(5)
        predicted = np.asarray([1.0, 2.0, 3.0, 20.0, 30.0])
        rows = risk_coverage_curve(actual, predicted, np.arange(5), fractions=(1.0, 0.6))
        self.assertEqual(rows[0]["n"], 5)
        self.assertEqual(rows[1]["n"], 3)
        self.assertEqual(rows[1]["medae"], 2.0)

    def test_conformal_scale_is_fit_without_target_fold(self) -> None:
        actual = np.zeros(12)
        predicted = np.asarray([1.0] * 6 + [10.0] * 6)
        uncertainty = np.ones(12)
        folds = np.asarray([0] * 6 + [1] * 6)
        lower, upper, scale = conformal_interval(actual, predicted, uncertainty, folds)
        np.testing.assert_allclose(scale[:6], 10.0)
        np.testing.assert_allclose(scale[6:], 1.0)
        # Deliberately shifted folds show why leave-fold-out calibration can
        # expose distribution shift rather than calibrating on the test fold.
        self.assertEqual(coverage_width(actual, lower, upper)["coverage"], 0.5)

    def test_stack_features_include_robust_and_classical_spread(self) -> None:
        stack = np.asarray([[1.0, 2.0], [2.0, 4.0], [9.0, 6.0]])
        features = stack_features(stack, np.asarray([3.0, 4.0]))
        self.assertEqual(
            set(features), {"std", "mad", "delta_to_median", "p90_p10_span"}
        )
        np.testing.assert_allclose(features["delta_to_median"], [1.0, 0.0])

    def test_structural_features_use_training_matrix_only(self) -> None:
        matrix = np.asarray(
            [
                [1.0, 2.0, 3.0, np.nan],
                [2.0, 4.0, 6.0, 8.0],
                [1.0, np.nan, 2.0, 3.0],
            ]
        )
        features = structural_support_features_for_cells(matrix, [(0, 3), (2, 1)])
        np.testing.assert_allclose(features["row_obs_count"], [3.0, 3.0])
        np.testing.assert_allclose(features["col_obs_count"], [2.0, 2.0])
        self.assertAlmostEqual(features["row_best_peer_abs_corr"][0], 1.0)

    def test_terciles_are_disjoint_and_cover_all_predictions(self) -> None:
        rows = uncertainty_tercile_errors(
            np.zeros(10), np.arange(10, dtype=float), np.arange(10, dtype=float)
        )
        self.assertEqual(sum(int(row["n"]) for row in rows), 10)
        self.assertEqual([row["bin"] for row in rows], [
            "low_uncertainty", "medium_uncertainty", "high_uncertainty"
        ])

    def test_crossfit_risk_model_returns_every_outer_fold(self) -> None:
        rng = np.random.RandomState(7)
        n = 120
        actual = rng.uniform(40.0, 90.0, size=n)
        feature = rng.uniform(0.0, 5.0, size=n)
        predicted = actual + feature + rng.normal(0.0, 0.1, size=n)
        folds = np.arange(n) % 4
        uncertainty, names, selected = crossfit_error_risk(
            actual,
            predicted,
            folds,
            {"feature": feature, "support": rng.uniform(1.0, 10.0, size=n)},
        )
        self.assertEqual(names, ["feature", "support"])
        self.assertEqual(set(selected), {"0", "1", "2", "3"})
        self.assertTrue(np.all(np.isfinite(uncertainty)))
        self.assertTrue(np.all(uncertainty >= 0.0))


if __name__ == "__main__":
    unittest.main()
