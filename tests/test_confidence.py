import unittest

import numpy as np

from pathopress.confidence import (
    conformal_interval,
    coverage_width,
    crossfit_trust_probability,
    crossfit_error_risk,
    feature_matrix,
    fit_trust_calibrator,
    predict_serialized_trust,
    RELATIVE_WIDTH_DENOMINATOR_EPSILON,
    relative_width_denominator_mask,
    risk_coverage_curve,
    spearman_uncertainty_error,
    stack_features,
    structural_support_features_for_cells,
    summarize_confidence_method,
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

    # -- FIX A: conformal skipped cells visibility --

    def test_conformal_skipped_cells_are_surfaced(self) -> None:
        # To skip a fold, its calibration set (all OTHER folds) must have
        # fewer than 5 valid samples. With 3 folds and 8 total samples,
        # folds 0 and 1 each have 2 cells, so their calibration set only
        # contains the other two folds' 6 cells -- enough. But fold 2 has
        # 4 cells, so its calibration is 4 cells -- too few.
        # A simpler arrangement: 4 samples, 4 folds of 1 each. Every fold's
        # calibration set is 3 samples < 5, so all are skipped.
        actual = np.zeros(4)
        predicted = np.ones(4)
        uncertainty = np.ones(4)
        folds = np.asarray([0, 1, 2, 3])
        summary = summarize_confidence_method(
            actual, predicted, folds, uncertainty
        )
        self.assertEqual(summary["conformal_total_cells"], 4)
        self.assertEqual(summary["conformal_skipped_cells"], 4)
        # No cells contributed to coverage.
        self.assertEqual(summary["conformal_90_interval"]["n"], 0)

    def test_conformal_skipped_partial_fold(self) -> None:
        # Fold 0 has 4 cells (calibration = 1 cell < 5, skipped).
        # Fold 1 has 1 cell (calibration = 4 cells < 5, skipped).
        # But adding fold 2 with 10 cells: fold 2 calibration is 5 cells >= 5.
        actual = np.zeros(15)
        predicted = np.ones(15)
        uncertainty = np.ones(15)
        folds = np.asarray([0] * 4 + [1] * 1 + [2] * 10)
        summary = summarize_confidence_method(
            actual, predicted, folds, uncertainty
        )
        self.assertEqual(summary["conformal_total_cells"], 15)
        # Fold 2 has calibration size 5 >= 5, so it succeeds.
        # Fold 0 calibration = 11 >= 5, succeeds. Fold 1 calibration = 14 >= 5.
        self.assertEqual(summary["conformal_skipped_cells"], 0)

    def test_conformal_skipped_partial_some_folds(self) -> None:
        # 2 folds: fold 0 has 2 cells (calib = 10 >= 5, OK),
        # fold 1 has 10 cells (calib = 2 < 5, SKIPPED).
        actual = np.zeros(12)
        predicted = np.ones(12)
        uncertainty = np.ones(12)
        folds = np.asarray([0] * 2 + [1] * 10)
        summary = summarize_confidence_method(
            actual, predicted, folds, uncertainty
        )
        self.assertEqual(summary["conformal_total_cells"], 12)
        self.assertEqual(summary["conformal_skipped_cells"], 10)
        self.assertEqual(summary["conformal_90_interval"]["n"], 2)

    def test_conformal_skipped_cells_warns_when_nonzero(self) -> None:
        actual = np.zeros(4)
        predicted = np.ones(4)
        uncertainty = np.ones(4)
        folds = np.asarray([0, 0, 1, 1])
        with self.assertWarns(RuntimeWarning):
            summarize_confidence_method(actual, predicted, folds, uncertainty)

    # -- FIX B: all-NaN slice visibility --

    def test_structural_features_include_all_nan_flags(self) -> None:
        matrix = np.asarray([
            [1.0, 2.0, 3.0],
            [np.nan, np.nan, np.nan],  # all-NaN row
            [4.0, 5.0, 6.0],
        ])
        features = structural_support_features_for_cells(matrix, [(0, 0), (1, 0), (2, 2)])
        self.assertIn("row_is_all_nan", features)
        self.assertIn("col_is_all_nan", features)
        np.testing.assert_array_equal(features["row_is_all_nan"], [0.0, 1.0, 0.0])
        # All columns have at least one finite value
        np.testing.assert_array_equal(features["col_is_all_nan"], [0.0, 0.0, 0.0])

    def test_feature_matrix_preserves_boolean_features(self) -> None:
        features = {
            "structural_count": np.asarray([1.0, 2.0]),
            "structural_row_is_all_nan": np.asarray([0.0, 1.0]),
            "structural_col_is_all_nan": np.asarray([1.0, 0.0]),
        }
        matrix, names = feature_matrix(features)
        # Boolean features should pass through without log1p transformation
        row_is_nan_col = names.index("structural_row_is_all_nan")
        col_is_nan_col = names.index("structural_col_is_all_nan")
        np.testing.assert_array_equal(matrix[:, row_is_nan_col], [0.0, 1.0])
        np.testing.assert_array_equal(matrix[:, col_is_nan_col], [1.0, 0.0])
        # Non-boolean features should still be transformed
        count_col = names.index("structural_count")
        np.testing.assert_allclose(
            matrix[:, count_col],
            np.log1p(np.maximum([1.0, 2.0], 0.0))
        )

    def test_safe_stat_warns_on_all_nan_slices(self) -> None:
        # An all-NaN column should trigger a warning via _safe_stat.
        # We test through structural_support_features_for_cells which calls _safe_stat.
        matrix = np.asarray([
            [np.nan, 1.0],
            [np.nan, 2.0],
        ])
        with self.assertWarns(RuntimeWarning):
            structural_support_features_for_cells(matrix, [(0, 0), (1, 0)])

    # -- BUG 5: Inf is silently zeroed without warning --

    def test_safe_stat_warns_on_inf_slices(self) -> None:
        """An Inf column should trigger a RuntimeWarning mentioning 'Inf'."""
        matrix = np.asarray([
            [np.inf, 1.0],
            [np.inf, 2.0],
        ])
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            structural_support_features_for_cells(matrix, [(0, 0), (1, 0)])
        inf_warnings = [w for w in caught if "Inf" in str(w.message)]
        self.assertTrue(inf_warnings, "Expected at least one warning mentioning 'Inf'")

    def test_safe_stat_warns_nan_and_inf_separately(self) -> None:
        """When both NaN and Inf columns exist, the warning should mention both."""
        matrix = np.asarray([
            [np.nan, np.inf, 1.0],
            [np.nan, np.inf, 2.0],
        ])
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            structural_support_features_for_cells(matrix, [(0, 0), (0, 1), (0, 2)])
        combined = [w for w in caught if "NaN" in str(w.message) and "Inf" in str(w.message)]
        self.assertTrue(combined, "Expected a warning mentioning both 'NaN' and 'Inf'")

    # -- FIX C: risk model fallback visibility --

    def test_risk_model_fallback_is_recorded_in_metadata(self) -> None:
        # With n=120 and 4 folds (30 per fold), train=90 >= design.shape[1]+50,
        # so the outer loop enters. But inner splits with mod 5 produce
        # val=24 < 50 and mod 3 also produce val < 50, triggering fallback.
        rng = np.random.RandomState(0)
        n = 120
        actual = rng.uniform(40.0, 90.0, size=n)
        feature = rng.uniform(0.0, 5.0, size=n)
        predicted = actual + feature + rng.normal(0.0, 0.1, size=n)
        folds = np.arange(n) % 4
        uncertainty, names, selected = crossfit_error_risk(
            actual, predicted, folds,
            {"feature": feature, "support": rng.uniform(1.0, 10.0, size=n)},
        )
        # All folds should have selected metadata
        self.assertEqual(set(selected), {"0", "1", "2", "3"})
        # All should show the fallback flag since inner splits are too small
        fallback_folds = [k for k, v in selected.items() if v.get("fallback")]
        self.assertEqual(len(fallback_folds), 4)

    def test_risk_model_fallback_warns(self) -> None:
        rng = np.random.RandomState(0)
        n = 80
        actual = rng.uniform(40.0, 90.0, size=n)
        feature = rng.uniform(0.0, 5.0, size=n)
        predicted = actual + feature + rng.normal(0.0, 0.1, size=n)
        folds = np.arange(n) % 4
        with self.assertWarns(RuntimeWarning):
            crossfit_error_risk(
                actual, predicted, folds,
                {"feature": feature, "support": rng.uniform(1.0, 10.0, size=n)},
            )


if __name__ == "__main__":
    unittest.main()
