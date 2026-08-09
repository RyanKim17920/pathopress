"""Reviewer fixes: leave-one-out skill, leakage-free dispersion, LOFO, controls.

Covers the four conceptual defects the two independent reviews raised:

1. ``skill_score`` was an in-sample oracle ratio, unbounded below, with no
   exclusion for columns whose dispersion sits inside reporting noise.
2. ``predict_all_known`` normalized by a column SD that included the very cells
   it was predicting, making ``medae_normalized`` incomparable to the held-out
   track.
3. A single ``GroupShuffleSplit`` holdout left too few independent validation
   models and swung with the seed.
4. The held-out panel carried no k=0 and no random-probe control, so the
   greedy-vs-random comparison existed only in sample.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathopress.probes import (  # noqa: E402
    MAD_TO_SD_SCALE,
    SKILL_NOISE_FLOOR_DISPERSION,
    compute_column_loo_baseline_medae,
    compute_column_median_baseline_medae,
    compute_column_robust_dispersion,
    compute_column_skill,
    evaluate_global_probes,
    leave_one_family_out_folds,
    summarize_skill_positive_fraction,
)
from pathopress.probe_compression import (  # noqa: E402
    predict_all_known,
    predict_heldout_models,
    score_predictions,
)


def _metadata_csv(families: dict[str, str]) -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    handle.write("model_id,provider,family\n")
    for model_id, family in families.items():
        handle.write(f"{model_id},provider,{family}\n")
    handle.close()
    return handle.name


class LeaveOneOutBaselineTests(unittest.TestCase):
    """FIX 1(b): the baseline must not be fitted on the cell it predicts."""

    def test_loo_baseline_differs_from_in_sample_on_hand_built_column(self) -> None:
        # Column 0 = [10, 20, 30, 40].  In sample the median is 25, so the
        # errors are [15, 5, 5, 15] and the MedAE is 10.
        matrix = np.array([
            [10.0, 1.0],
            [20.0, 2.0],
            [30.0, 4.0],
            [40.0, 8.0],
        ])
        self.assertAlmostEqual(compute_column_median_baseline_medae(matrix, 0), 10.0)
        # Leave-one-out medians are 30, 30, 20, 20 giving errors
        # [20, 10, 10, 20] and a MedAE of 15.
        self.assertAlmostEqual(compute_column_loo_baseline_medae(matrix, 0), 15.0)
        self.assertGreater(
            compute_column_loo_baseline_medae(matrix, 0),
            compute_column_median_baseline_medae(matrix, 0),
        )

    def test_loo_baseline_on_odd_column(self) -> None:
        # Column = [1, 2, 6].  LOO medians are 4, 3.5, 1.5 giving errors
        # [3, 1.5, 4.5]; the median of those is 3.
        matrix = np.array([[1.0, 9.0], [2.0, 9.0], [6.0, 3.0]])
        self.assertAlmostEqual(compute_column_loo_baseline_medae(matrix, 0), 3.0)

    def test_loo_baseline_needs_two_observations(self) -> None:
        matrix = np.array([[5.0, 1.0], [np.nan, 2.0], [np.nan, 6.0]])
        self.assertTrue(np.isnan(compute_column_loo_baseline_medae(matrix, 0)))


class RobustDispersionTests(unittest.TestCase):
    """FIX 1(c): the MAD denominator, scaled to an SD estimate."""

    def test_mad_denominator_matches_manual_calculation(self) -> None:
        # Column = [1, 2, 6, 9]; median 4; deviations [3, 2, 2, 5];
        # MAD = median(3, 2, 2, 5) = 2.5.
        matrix = np.array([[1.0, 0.0], [2.0, 1.0], [6.0, 2.0], [9.0, 3.0]])
        self.assertAlmostEqual(
            compute_column_robust_dispersion(matrix, 0), 2.5 * MAD_TO_SD_SCALE
        )

    def test_mad_scale_constant(self) -> None:
        self.assertAlmostEqual(MAD_TO_SD_SCALE, 1.4826)

    def test_constant_column_has_zero_dispersion(self) -> None:
        matrix = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 6.0]])
        self.assertAlmostEqual(compute_column_robust_dispersion(matrix, 0), 0.0)


class BoundedSkillTests(unittest.TestCase):
    """FIX 1(a): bounded reporting value, raw value preserved separately."""

    MATRIX = np.array([
        [10.0, 1.0],
        [20.0, 2.0],
        [30.0, 4.0],
        [40.0, 8.0],
    ])

    def test_skill_is_clipped_but_raw_is_preserved(self) -> None:
        # LOO baseline for column 0 is 15; a MedAE of 315 gives raw skill -20.
        skill = compute_column_skill(self.MATRIX, 0, 315.0)
        self.assertFalse(skill.excluded_below_noise_floor)
        self.assertAlmostEqual(skill.skill_score_raw, -20.0)
        self.assertAlmostEqual(skill.skill_score, -1.0)

    def test_medae_ratio_is_the_primary_field(self) -> None:
        skill = compute_column_skill(self.MATRIX, 0, 30.0)
        self.assertAlmostEqual(skill.medae_ratio, 2.0)
        self.assertAlmostEqual(skill.skill_score_raw, -1.0)
        self.assertAlmostEqual(skill.skill_score, -1.0)
        # The ratio has no pole at the origin, unlike 1 - ratio's lower tail.
        self.assertAlmostEqual(compute_column_skill(self.MATRIX, 0, 0.0).medae_ratio, 0.0)

    def test_skill_clipped_above_is_impossible_but_bounded_anyway(self) -> None:
        skill = compute_column_skill(self.MATRIX, 0, 0.0)
        self.assertAlmostEqual(skill.skill_score_raw, 1.0)
        self.assertAlmostEqual(skill.skill_score, 1.0)

    def test_skill_zero_when_model_matches_loo_baseline(self) -> None:
        skill = compute_column_skill(self.MATRIX, 0, 15.0)
        self.assertAlmostEqual(skill.skill_score, 0.0)

    def test_noise_floor_exclusion_is_explicit(self) -> None:
        # Column 0 dispersion is 0.02 * 1.4826, far below the floor.
        matrix = np.array([
            [10.00, 1.0],
            [10.02, 2.0],
            [10.04, 4.0],
            [10.06, 8.0],
        ])
        self.assertLess(
            compute_column_robust_dispersion(matrix, 0), SKILL_NOISE_FLOOR_DISPERSION
        )
        skill = compute_column_skill(matrix, 0, 0.001)
        self.assertTrue(skill.excluded_below_noise_floor)
        self.assertEqual(skill.exclusion_reason, "noise_floor")
        self.assertIsNone(skill.skill_score)
        self.assertIsNone(skill.skill_score_raw)

    def test_noise_floor_threshold_is_a_named_constant(self) -> None:
        self.assertIsInstance(SKILL_NOISE_FLOOR_DISPERSION, float)
        self.assertGreater(SKILL_NOISE_FLOOR_DISPERSION, 0.0)

    def test_column_without_model_error_is_excluded_with_its_own_reason(self) -> None:
        skill = compute_column_skill(self.MATRIX, 0, float("nan"))
        self.assertTrue(skill.excluded_below_noise_floor)
        self.assertEqual(skill.exclusion_reason, "no_model_error")


class SkillFractionSummaryTests(unittest.TestCase):
    """FIX 1: the headline statistic is a bounded fraction with a bootstrap CI."""

    MATRIX = np.array([
        [10.0, 1.0, 100.0],
        [20.0, 2.0, 130.0],
        [30.0, 4.0, 160.0],
        [40.0, 8.0, 190.0],
    ])

    def test_fraction_positive_counts_only_scored_columns(self) -> None:
        skills = [
            compute_column_skill(self.MATRIX, 0, 1.0),          # positive
            compute_column_skill(self.MATRIX, 1, 1000.0),       # negative
            compute_column_skill(self.MATRIX, 2, float("nan")), # excluded
        ]
        summary = summarize_skill_positive_fraction(skills, n_bootstrap=200)
        self.assertEqual(summary.n_columns_total, 3)
        self.assertEqual(summary.n_columns_excluded, 1)
        self.assertEqual(summary.n_columns_scored, 2)
        self.assertEqual(summary.n_columns_positive, 1)
        self.assertAlmostEqual(summary.fraction_positive, 0.5)

    def test_bootstrap_interval_brackets_the_point_estimate(self) -> None:
        skills = [compute_column_skill(self.MATRIX, col, 1.0) for col in range(3)]
        summary = summarize_skill_positive_fraction(skills, n_bootstrap=500)
        self.assertLessEqual(summary.ci_lower, summary.fraction_positive)
        self.assertGreaterEqual(summary.ci_upper, summary.fraction_positive)
        self.assertEqual(summary.n_bootstrap, 500)

    def test_bootstrap_is_deterministic_for_a_fixed_seed(self) -> None:
        skills = [
            compute_column_skill(self.MATRIX, 0, 1.0),
            compute_column_skill(self.MATRIX, 1, 1000.0),
            compute_column_skill(self.MATRIX, 2, 1.0),
        ]
        first = summarize_skill_positive_fraction(skills, n_bootstrap=300, seed=7)
        second = summarize_skill_positive_fraction(skills, n_bootstrap=300, seed=7)
        self.assertEqual(first, second)

    def test_all_columns_excluded_yields_no_fraction(self) -> None:
        skills = [
            compute_column_skill(self.MATRIX, col, float("nan")) for col in range(3)
        ]
        summary = summarize_skill_positive_fraction(skills, n_bootstrap=50)
        self.assertEqual(summary.n_columns_scored, 0)
        self.assertTrue(np.isnan(summary.fraction_positive))


class PerColumnHiddenMedaeTests(unittest.TestCase):
    """FIX 1: the skill numerator is denominated per column, like its baseline."""

    def test_probe_columns_have_no_hidden_cells(self) -> None:
        matrix = np.array([
            [10.0, 20.0, 30.0],
            [20.0, 30.0, 40.0],
            [30.0, 40.0, 50.0],
            [40.0, 50.0, 60.0],
        ])
        result = evaluate_global_probes(matrix, [1], rank=1)
        self.assertEqual(len(result.per_column_hidden_medae), 3)
        self.assertTrue(np.isnan(result.per_column_hidden_medae[1]))
        self.assertTrue(np.isfinite(result.per_column_hidden_medae[0]))
        self.assertTrue(np.isfinite(result.per_column_hidden_medae[2]))


class LeakageFreeDispersionTests(unittest.TestCase):
    """FIX 2: no dispersion denominator may see the target cell's own row."""

    MATRIX = np.array([
        [10.0, 20.0],
        [20.0, 30.0],
        [30.0, 40.0],
        [41.0, 50.0],
    ])

    def test_all_known_dispersion_excludes_the_target_row(self) -> None:
        result = predict_all_known(self.MATRIX, [0], rank=1)
        self.assertIsNotNone(result.column_dispersion_by_cell)
        self.assertEqual(result.column_dispersion_by_cell.shape, self.MATRIX.shape)
        for row in range(self.MATRIX.shape[0]):
            for col in range(self.MATRIX.shape[1]):
                others = np.delete(self.MATRIX[:, col], row)
                median = float(np.median(others))
                expected = MAD_TO_SD_SCALE * float(
                    np.median(np.abs(others - median))
                )
                self.assertAlmostEqual(
                    float(result.column_dispersion_by_cell[row, col]),
                    expected,
                    places=9,
                )

    def test_all_known_dispersion_is_not_the_full_matrix_dispersion(self) -> None:
        result = predict_all_known(self.MATRIX, [0], rank=1)
        column = self.MATRIX[:, 0]
        full = MAD_TO_SD_SCALE * float(
            np.median(np.abs(column - float(np.median(column))))
        )
        per_cell = np.asarray(result.column_dispersion_by_cell)[:, 0]
        self.assertFalse(np.allclose(per_cell, full))

    def test_heldout_dispersion_uses_context_rows_only(self) -> None:
        result = predict_heldout_models(
            self.MATRIX, [0], target_model_indices=[2, 3],
            context_model_indices=[0, 1], rank=1,
        )
        dispersion = np.asarray(result.column_dispersion_by_cell)
        for col in range(self.MATRIX.shape[1]):
            context = self.MATRIX[:2, col]
            expected = MAD_TO_SD_SCALE * float(
                np.median(np.abs(context - float(np.median(context))))
            )
            self.assertTrue(
                np.allclose(dispersion[:, col], expected),
                f"column {col} dispersion is not context-only",
            )

    def test_loo_normalized_keys_are_emitted_on_both_tracks(self) -> None:
        all_known = score_predictions(predict_all_known(self.MATRIX, [0], rank=1))
        heldout = score_predictions(
            predict_heldout_models(
                self.MATRIX, [0], target_model_indices=[2, 3],
                context_model_indices=[0, 1], rank=1,
            )
        )
        for metrics in (all_known, heldout):
            self.assertIn("medae_normalized_loo", metrics)
            self.assertIn("medae_normalized_pooled_loo", metrics)
            self.assertTrue(np.isfinite(metrics["medae_normalized_loo"]))

    def test_legacy_keys_are_retained_for_artifact_compatibility(self) -> None:
        metrics = score_predictions(predict_all_known(self.MATRIX, [0], rank=1))
        self.assertIn("medae_normalized", metrics)
        self.assertIn("medae_normalized_pooled", metrics)

    def test_loo_normalized_omitted_when_no_dispersion_available(self) -> None:
        # An empty context leaves every dispersion NaN, so the keys must be
        # dropped rather than serialized as NaN.
        result = predict_heldout_models(
            self.MATRIX, [0, 1], target_model_indices=[0], context_model_indices=[],
            rank=1,
        )
        metrics = score_predictions(result)
        self.assertNotIn("medae_normalized_loo", metrics)


class LeaveOneFamilyOutTests(unittest.TestCase):
    """FIX 3: every model validated exactly once, no family on both sides."""

    FAMILIES = {
        "virchow": "Virchow", "virchow-2": "Virchow",
        "dinov2-b": "DINOv2", "dinov2-l": "DINOv2",
        "dinov3-b": "DINOv3", "dinov3-l": "DINOv3", "dinov3-s": "DINOv3",
        "uni": "UNI", "uni2-h": "UNI",
        "atlas": "",
        "gpfm": "",
        "keep": "",
    }

    def test_every_model_is_validated_exactly_once(self) -> None:
        model_ids = list(self.FAMILIES)
        path = _metadata_csv(self.FAMILIES)
        try:
            folds, info = leave_one_family_out_folds(
                model_ids, model_metadata_path=path
            )
        finally:
            os.unlink(path)

        counts = {index: 0 for index in range(len(model_ids))}
        for fold in folds:
            for index in fold.validation_indices:
                counts[index] += 1
        self.assertEqual(set(counts.values()), {1})
        self.assertEqual(info["aggregate_validation_models"], len(model_ids))

    def test_no_family_is_ever_on_both_sides_of_a_fold(self) -> None:
        model_ids = list(self.FAMILIES)
        path = _metadata_csv(self.FAMILIES)
        try:
            folds, _ = leave_one_family_out_folds(
                model_ids, model_metadata_path=path
            )
        finally:
            os.unlink(path)

        for fold in folds:
            train_families = {
                self.FAMILIES[model_ids[i]] or model_ids[i]
                for i in fold.train_indices
            }
            val_families = {
                self.FAMILIES[model_ids[i]] or model_ids[i]
                for i in fold.validation_indices
            }
            self.assertEqual(
                train_families & val_families,
                set(),
                f"fold {fold.fold} leaks a family",
            )
            self.assertEqual(
                set(fold.train_indices) & set(fold.validation_indices), set()
            )
            self.assertEqual(
                set(fold.train_indices) | set(fold.validation_indices),
                set(range(len(model_ids))),
            )

    def test_multi_model_families_are_held_out_together(self) -> None:
        model_ids = list(self.FAMILIES)
        path = _metadata_csv(self.FAMILIES)
        try:
            folds, _ = leave_one_family_out_folds(
                model_ids, model_metadata_path=path
            )
        finally:
            os.unlink(path)

        by_family = {fold.family: fold for fold in folds}
        dinov3 = by_family["DINOv3"]
        self.assertEqual(
            {model_ids[i] for i in dinov3.validation_indices},
            {"dinov3-b", "dinov3-l", "dinov3-s"},
        )

    def test_fold_count_equals_group_count_and_is_seed_free(self) -> None:
        model_ids = list(self.FAMILIES)
        path = _metadata_csv(self.FAMILIES)
        try:
            folds, info = leave_one_family_out_folds(
                model_ids, model_metadata_path=path
            )
        finally:
            os.unlink(path)
        # 4 real families plus 3 blank-family singletons.
        self.assertEqual(len(folds), 7)
        self.assertEqual(info["n_folds"], 7)
        self.assertEqual(info["split_mode"], "leave_one_family_out")
        self.assertNotIn("seed", info)

    def test_info_reports_per_fold_and_aggregate_validation_sizes(self) -> None:
        model_ids = list(self.FAMILIES)
        path = _metadata_csv(self.FAMILIES)
        try:
            _, info = leave_one_family_out_folds(
                model_ids, model_metadata_path=path
            )
        finally:
            os.unlink(path)
        self.assertEqual(len(info["per_fold"]), info["n_folds"])
        for entry in info["per_fold"]:
            self.assertIn("n_validation_models", entry)
            self.assertIn("n_train_models", entry)
        self.assertEqual(info["min_fold_validation_models"], 1)
        self.assertEqual(info["max_fold_validation_models"], 3)

    def test_blank_families_never_merge(self) -> None:
        families = {"a": "", "b": "", "c": "X", "d": "X"}
        path = _metadata_csv(families)
        try:
            folds, _ = leave_one_family_out_folds(
                list(families), model_metadata_path=path
            )
        finally:
            os.unlink(path)
        singleton_folds = [fold for fold in folds if len(fold.validation_indices) == 1]
        self.assertEqual(len(singleton_folds), 2)

    def test_single_family_is_rejected(self) -> None:
        families = {"a": "X", "b": "X"}
        path = _metadata_csv(families)
        try:
            with self.assertRaises(ValueError):
                leave_one_family_out_folds(list(families), model_metadata_path=path)
        finally:
            os.unlink(path)


class HeldoutControlsTests(unittest.TestCase):
    """FIX 4: the held-out panel must carry k=0 and random-probe controls."""

    def setUp(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
        import run_probe_selection

        self.runner = run_probe_selection

    def _parse(self, *argv: str):
        saved = sys.argv
        sys.argv = ["run_probe_selection.py", *argv]
        try:
            return self.runner.parse_args()
        finally:
            sys.argv = saved

    def test_leave_one_family_out_is_the_default_heldout_protocol(self) -> None:
        args = self._parse()
        self.assertEqual(args.split_mode, "leave_one_family_out")
        self.assertGreaterEqual(args.lofo_max_probes, 1)

    def test_earlier_split_arms_remain_selectable(self) -> None:
        for mode in ("random", "family_blocked"):
            self.assertEqual(self._parse("--split-mode", mode).split_mode, mode)

    def test_heldout_random_control_is_configurable(self) -> None:
        args = self._parse()
        self.assertTrue(hasattr(args, "heldout_random_repeats"))
        self.assertGreaterEqual(args.heldout_random_repeats, 1)

    def test_k0_control_is_a_valid_heldout_evaluation(self) -> None:
        matrix = np.array([
            [10.0, 20.0, 30.0],
            [20.0, 30.0, 40.0],
            [30.0, 40.0, 50.0],
            [41.0, 50.0, 61.0],
            [52.0, 61.0, 70.0],
        ])
        result = self.runner._heldout_evaluate(
            matrix, (), (0, 1, 2), (3, 4), rank=1
        )
        self.assertEqual(result["probe_indices"], [])
        self.assertEqual(result["n_revealed_cells"], 0)
        # With no probes every observed validation cell is hidden.
        self.assertEqual(result["hidden_only"]["n"], 6)
        self.assertIsNotNone(result["hidden_only"]["medae"])

    def test_random_control_uses_the_same_heldout_rows_as_greedy(self) -> None:
        matrix = np.array([
            [10.0, 20.0, 30.0],
            [20.0, 30.0, 40.0],
            [30.0, 40.0, 50.0],
            [41.0, 50.0, 61.0],
            [52.0, 61.0, 70.0],
        ])
        greedy = self.runner._heldout_evaluate(matrix, (0,), (0, 1, 2), (3, 4), 1)
        random_control = self.runner._heldout_evaluate(
            matrix, (2,), (0, 1, 2), (3, 4), 1
        )
        self.assertEqual(
            greedy["parity"]["n"], random_control["parity"]["n"]
        )
        self.assertEqual(
            greedy["hidden_only"]["n"], random_control["hidden_only"]["n"]
        )


if __name__ == "__main__":
    unittest.main()
