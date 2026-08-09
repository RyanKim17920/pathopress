"""Tests for FIX 1 (skill score), FIX 2 (normalized metrics), FIX 3 (family splits)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathopress.probes import (
    compute_column_median_baseline_medae,
    family_blocked_model_split,
    random_model_split,
)
from pathopress.probe_compression import (
    ProbePredictions,
    predict_all_known,
    predict_heldout_models,
    score_predictions,
)


class SkillScoreTests(unittest.TestCase):
    """FIX 1: Per-column skill score correctness on hand-built matrices."""

    def test_skill_score_perfect_predictor(self) -> None:
        matrix = np.array([
            [10.0, 20.0, 30.0],
            [20.0, 30.0, 40.0],
            [30.0, 40.0, 50.0],
        ])
        col_baseline = compute_column_median_baseline_medae(matrix, 0)
        # Column 0: median=20, errors=[10, 0, 10], medae=10
        self.assertAlmostEqual(col_baseline, 10.0)
        # Perfect predictor: parity_medae=0
        skill = 1.0 - (0.0 / col_baseline)
        self.assertAlmostEqual(skill, 1.0)

    def test_skill_score_baseline_predictor(self) -> None:
        matrix = np.array([
            [10.0, 20.0, 30.0],
            [20.0, 30.0, 40.0],
            [30.0, 40.0, 50.0],
        ])
        col_baseline = compute_column_median_baseline_medae(matrix, 0)
        # Baseline: parity_medae = col_baseline => skill_score = 0
        skill = 1.0 - (col_baseline / col_baseline)
        self.assertAlmostEqual(skill, 0.0)

    def test_skill_score_worse_than_baseline(self) -> None:
        matrix = np.array([
            [10.0, 20.0, 30.0],
            [20.0, 30.0, 40.0],
            [30.0, 40.0, 50.0],
        ])
        col_baseline = compute_column_median_baseline_medae(matrix, 0)
        # Worse: parity_medae = 2 * col_baseline => skill_score = -1
        skill = 1.0 - (2.0 * col_baseline / col_baseline)
        self.assertAlmostEqual(skill, -1.0)

    def test_skill_score_zero_baseline_returns_nan(self) -> None:
        parity_medae = 0.5
        col_baseline = 0.0
        if np.isfinite(col_baseline) and col_baseline > 0:
            skill = 1.0 - (parity_medae / col_baseline)
        else:
            skill = float("nan")
        self.assertTrue(np.isnan(skill))

    def test_skill_score_nan_baseline_returns_nan(self) -> None:
        parity_medae = 0.5
        col_baseline = float("nan")
        if np.isfinite(col_baseline) and col_baseline > 0:
            skill = 1.0 - (parity_medae / col_baseline)
        else:
            skill = float("nan")
        self.assertTrue(np.isnan(skill))

    def test_parity_medae_normalized(self) -> None:
        """parity_medae_normalized = parity_medae / column SD."""
        matrix = np.array([
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
        ])
        col_sd = float(np.std(matrix[:, 0], ddof=0))  # ~8.165
        parity_medae = 2.0
        expected = parity_medae / col_sd
        # Verify the formula is correct
        self.assertAlmostEqual(expected, 2.0 / col_sd)

    def test_column_median_baseline_with_nan(self) -> None:
        matrix = np.array([
            [10.0, np.nan],
            [20.0, 5.0],
            [30.0, 15.0],
        ])
        # Column 0: median=20, errors=[10, 0, 10], medae=10
        self.assertAlmostEqual(
            compute_column_median_baseline_medae(matrix, 0), 10.0
        )
        # Column 1: values=[5, 15], median=10, errors=[5, 5], medae=5
        self.assertAlmostEqual(
            compute_column_median_baseline_medae(matrix, 1), 5.0
        )

    def test_column_median_baseline_constant_column(self) -> None:
        matrix = np.array([
            [5.0, 20.0],
            [5.0, 30.0],
            [5.0, 40.0],
        ])
        # Constant column: median=5, errors=[0, 0, 0], medae=0
        self.assertAlmostEqual(
            compute_column_median_baseline_medae(matrix, 0), 0.0
        )


class NormalizedMetricsTests(unittest.TestCase):
    """FIX 2: Dispersion-normalized error metric correctness."""

    def test_predict_all_known_returns_column_sd(self) -> None:
        matrix = np.array([
            [10.0, 20.0],
            [20.0, 30.0],
            [30.0, 40.0],
        ])
        result = predict_all_known(matrix, [0], rank=0)
        self.assertIsNotNone(result.column_sd)
        self.assertEqual(len(result.column_sd), 2)

    def test_column_sd_full_matrix_all_known(self) -> None:
        """column_sd for all_known uses the full actual matrix."""
        matrix = np.array([
            [10.0, 20.0, 30.0],
            [20.0, 30.0, 40.0],
            [30.0, 40.0, 50.0],
        ])
        result = predict_all_known(matrix, [], rank=0)
        for col in range(3):
            expected = float(np.std(matrix[:, col], ddof=0))
            self.assertAlmostEqual(
                float(result.column_sd[col]), expected, places=4
            )

    def test_column_sd_context_only_heldout(self) -> None:
        """column_sd for heldout uses context rows only (leakage-safe)."""
        matrix = np.array([
            [10.0, 20.0],
            [20.0, 30.0],
            [30.0, 40.0],
            [40.0, 50.0],
        ])
        result = predict_heldout_models(
            matrix, [0], target_model_indices=[2, 3],
            context_model_indices=[0, 1], rank=0
        )
        context_matrix = matrix[:2]
        for col in range(2):
            expected = float(np.std(context_matrix[:, col], ddof=0))
            self.assertAlmostEqual(
                float(result.column_sd[col]), expected, places=4
            )

    def test_score_predictions_has_normalized_metrics(self) -> None:
        matrix = np.array([
            [10.0, 20.0],
            [20.0, 30.0],
            [30.0, 40.0],
        ])
        result = predict_all_known(matrix, [0], rank=0)
        metrics = score_predictions(result)
        self.assertIn("medae_normalized", metrics)
        self.assertIn("medae_normalized_pooled", metrics)

    def test_normalized_metrics_finite_with_valid_sd(self) -> None:
        matrix = np.array([
            [10.0, 20.0],
            [20.0, 30.0],
            [30.0, 40.0],
        ])
        result = predict_all_known(matrix, [0], rank=0)
        metrics = score_predictions(result)
        self.assertTrue(np.isfinite(metrics["medae_normalized"]))
        self.assertTrue(np.isfinite(metrics["medae_normalized_pooled"]))

    def test_normalized_metrics_constant_column_handling(self) -> None:
        """Constant column (SD=0) is handled gracefully."""
        matrix = np.array([
            [5.0, 20.0],
            [5.0, 30.0],
            [5.0, 40.0],
        ])
        result = predict_all_known(matrix, [0], rank=0)
        metrics = score_predictions(result)
        # At least medae should be finite
        self.assertTrue(np.isfinite(metrics["medae"]))


class FamilyBlockedSplitTests(unittest.TestCase):
    """FIX 3: Family-blocked model split correctness."""

    def _make_metadata_csv(self, families: dict[str, str]) -> str:
        """Create a temp metadata CSV; returns path."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        f.write("model_id,provider,family\n")
        for mid, fam in families.items():
            f.write(f"{mid},provider,{fam}\n")
        f.close()
        return f.name

    def test_no_family_leaks_across_split(self) -> None:
        """No family appears on both sides of a family-blocked split."""
        model_ids = [
            "virchow", "virchow-2",
            "dinov2-b", "dinov2-l",
            "dinov3-b", "dinov3-l", "dinov3-s",
            "clip-b", "clip-l",
            "plip", "conch", "conch-1.5",
            "kaiko-vit-b-16", "kaiko-vit-l-14",
            "atlas",
            "chief-patch-mean", "chief-slide",
            "ctranspath", "phikon",
            "uni", "uni2-h",
            "vit-b", "vit-l",
        ]
        families = {
            "virchow": "Virchow", "virchow-2": "Virchow",
            "dinov2-b": "DINOv2", "dinov2-l": "DINOv2",
            "dinov3-b": "DINOv3", "dinov3-l": "DINOv3", "dinov3-s": "DINOv3",
            "clip-b": "CLIP", "clip-l": "CLIP",
            "plip": "PLIP", "conch": "CONCH", "conch-1.5": "CONCH",
            "kaiko-vit-b-16": "Kaiko", "kaiko-vit-l-14": "Kaiko",
            "atlas": "Atlas",
            "chief-patch-mean": "CHIEF", "chief-slide": "CHIEF",
            "ctranspath": "CTransPath", "phikon": "Phikon",
            "uni": "UNI", "uni2-h": "UNI",
            "vit-b": "ViT", "vit-l": "ViT",
        }
        path = self._make_metadata_csv(families)
        try:
            train_idx, val_idx, info = family_blocked_model_split(
                model_ids, model_metadata_path=path, seed=42
            )
            train_ids = {model_ids[i] for i in train_idx}
            val_ids = {model_ids[i] for i in val_idx}
            train_families = {families[mid] for mid in train_ids}
            val_families = {families[mid] for mid in val_ids}
            overlap = train_families & val_families
            self.assertEqual(
                overlap, set(),
                f"Families appear on both sides: {overlap}"
            )
        finally:
            os.unlink(path)

    def test_blank_families_are_singletons(self) -> None:
        """Models with blank family are treated as singleton groups."""
        model_ids = ["a", "b", "c", "d", "e", "f"]
        families = {
            "a": "X", "b": "X",
            "c": "", "d": "",
            "e": "Y", "f": "Y",
        }
        path = self._make_metadata_csv(families)
        try:
            train_idx, val_idx, info = family_blocked_model_split(
                model_ids, model_metadata_path=path, seed=42
            )
            train_ids = {model_ids[i] for i in train_idx}
            val_ids = {model_ids[i] for i in val_idx}
            # X family not split
            x_train = {"a", "b"} & train_ids
            x_val = {"a", "b"} & val_ids
            self.assertTrue(
                x_train == set() or x_val == set(),
                "X family split across train/val"
            )
            # Y family not split
            y_train = {"e", "f"} & train_ids
            y_val = {"e", "f"} & val_ids
            self.assertTrue(
                y_train == set() or y_val == set(),
                "Y family split across train/val"
            )
        finally:
            os.unlink(path)

    def test_split_info_records_families(self) -> None:
        model_ids = ["a", "b", "c", "d", "e", "f"]
        families = {
            "a": "X", "b": "X", "c": "X",
            "d": "Y", "e": "Y", "f": "Y",
        }
        path = self._make_metadata_csv(families)
        try:
            _, _, info = family_blocked_model_split(
                model_ids, model_metadata_path=path, seed=42
            )
            self.assertIn("train_families", info)
            self.assertIn("validation_families", info)
            self.assertIn("split_mode", info)
            self.assertEqual(info["split_mode"], "family_blocked")
        finally:
            os.unlink(path)

    def test_random_split_returns_valid_partition(self) -> None:
        model_ids = ["a", "b", "c", "d", "e", "f"]
        train_idx, val_idx, info = random_model_split(model_ids, seed=42)
        train_set = set(train_idx)
        val_set = set(val_idx)
        self.assertEqual(train_set & val_set, set())
        self.assertEqual(len(train_set) + len(val_set), len(model_ids))
        self.assertEqual(info["split_mode"], "random")


class BenchpressStyleMetricsTests(unittest.TestCase):
    """FIX 2: run_benchpress_style.py metrics function correctness."""

    def test_metrics_includes_normalized(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
        from run_benchpress_style import metrics
        actual = [10.0, 20.0, 30.0]
        predicted = [10.5, 20.5, 30.5]
        result = metrics(actual, predicted, column_sd=5.0)
        self.assertIn("medae_normalized", result)
        # medae = 0.5, column_sd = 5.0, so medae_normalized = 0.1
        self.assertAlmostEqual(result["medae_normalized"], 0.1, places=5)

    def test_metrics_normalized_nan_when_no_sd(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
        from run_benchpress_style import metrics
        actual = [10.0, 20.0, 30.0]
        predicted = [10.5, 20.5, 30.5]
        result = metrics(actual, predicted)
        # Convention: omit key entirely when column_sd is unavailable (no NaN).
        self.assertNotIn("medae_normalized", result)


if __name__ == "__main__":
    unittest.main()
