import unittest

import numpy as np

from pathopress.confidence import (
    conformal_interval,
    coverage_width,
    crossfit_trust_probability,
    crossfit_error_risk,
    fit_trust_calibrator,
    predict_serialized_trust,
    RELATIVE_WIDTH_DENOMINATOR_EPSILON,
    relative_width_denominator_mask,
    risk_coverage_curve,
    spearman_uncertainty_error,
    stack_features,
    structural_support_features_for_cells,
    uncertainty_tercile_errors,
)
from pathopress.metrics import MEDAPE_EPSILON


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

    def test_percentage_metrics_use_shared_denominator_boundary(self) -> None:
        just_above = np.nextafter(MEDAPE_EPSILON, np.inf)
        actual = np.asarray([
            MEDAPE_EPSILON, -MEDAPE_EPSILON, just_above, -just_above,
        ])
        predicted = np.zeros(4)
        rows = risk_coverage_curve(actual, predicted, np.ones(4), fractions=(1.0,))
        # Exact-boundary denominators are excluded; both just-above values are
        # one hundred percent errors.
        self.assertEqual(rows[0]["medape"], 100.0)

        lower = -np.abs(actual)
        upper = np.abs(actual)
        interval = coverage_width(actual, lower, upper)
        self.assertAlmostEqual(interval["median_relative_width"], 200.0)

    def test_relative_width_uses_its_distinct_upstream_denominator_boundary(self) -> None:
        just_above = np.nextafter(RELATIVE_WIDTH_DENOMINATOR_EPSILON, np.inf)
        mask = relative_width_denominator_mask(np.asarray([
            RELATIVE_WIDTH_DENOMINATOR_EPSILON,
            -RELATIVE_WIDTH_DENOMINATOR_EPSILON,
            just_above,
            -just_above,
            MEDAPE_EPSILON,
        ]))
        np.testing.assert_array_equal(mask, [False, False, True, True, True])
        # A denominator between 1e-8 and MedAPE's 1e-6 boundary participates
        # in relative interval width, exactly as in upstream BenchPress.
        interval = coverage_width(
            np.asarray([1e-7, 2.0]),
            np.asarray([-0.9999999, 1.0]),
            np.asarray([1.0000001, 3.0]),
        )
        self.assertGreater(interval["median_relative_width"], 1e9)

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

    def test_trust_calibrator_is_monotone_decreasing_in_risk(self) -> None:
        risk = np.arange(100, dtype=float)
        actual = np.zeros(100)
        predicted = np.concatenate([np.zeros(70), np.full(30, 20.0)])
        predictor, metadata = fit_trust_calibrator(risk, actual, predicted, n_bins=10)
        probability = predictor(risk)
        self.assertTrue(np.all(np.diff(probability) <= 1e-12))
        replay = predict_serialized_trust(risk, metadata)
        np.testing.assert_allclose(replay, probability)
        self.assertGreater(probability[0], probability[-1])

    def test_trust_probability_leaves_target_fold_out(self) -> None:
        # Fold 0 always misses the ten-point event and fold 1 always meets it.
        # A proper leave-fold calibrator therefore assigns the opposite fold's
        # prevalence, visibly distinguishing it from in-sample calibration.
        actual = np.zeros(40)
        predicted = np.asarray([20.0] * 20 + [1.0] * 20)
        risk = np.ones(40)
        folds = np.asarray([0] * 20 + [1] * 20)
        probability, metadata = crossfit_trust_probability(
            risk, actual, predicted, folds, n_bins=4
        )
        np.testing.assert_allclose(probability[:20], 1.0)
        np.testing.assert_allclose(probability[20:], 0.0)
        self.assertEqual(set(metadata), {"0", "1"})

    def test_trust_probability_purges_repeated_target_groups(self) -> None:
        folds = np.repeat(np.arange(4), 10)
        groups = np.tile(np.arange(10), 4)
        # Folds 0/1 share groups 0-9 and folds 2/3 share groups 10-19.
        groups[20:] += 10
        risk = np.linspace(0.1, 4.0, 40)
        actual = np.zeros(40)
        predicted = np.linspace(0.0, 20.0, 40)
        probability, metadata = crossfit_trust_probability(
            risk, actual, predicted, folds, group_id=groups, n_bins=4
        )
        self.assertTrue(np.all(np.isfinite(probability)))
        self.assertTrue(all(
            int(row["n_purged_repeated_target_instances"]) == 10
            for row in metadata.values()
        ))


if __name__ == "__main__":
    unittest.main()
