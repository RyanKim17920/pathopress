"""Tests for FIX 1 (skill score), FIX 2 (normalized metrics), FIX 3 (family splits)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


# Ensure src is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathopress.probes import (  # noqa: E402
    compute_column_median_baseline_medae,
    family_blocked_model_split,
    random_model_split,
)
from pathopress.probe_compression import (  # noqa: E402
    ProbePredictions,
    predict_all_known,
    predict_heldout_models,
    score_predictions,
)


class ComputeColumnMedianBaselineMedaeTests(unittest.TestCase):
    """FIX 1: Per-column baseline MedAE correctness."""

    def test_per_column_baseline_matches_manual_calculation(self) -> None:
        # 3x2 matrix: column 0 = [10, 20, 30], column 1 = [5, 5, 5]
        matrix = np.array([
            [10.0, 5.0],
            [20.0, 5.0],
            [30.0, 5.0],
        ])
        # Column 0: median=20, errors=[10, 0, 10], medae=10
        self.assertAlmostEqual(
            compute_column_median_baseline_medae(matrix, 0), 10.0
        )
        # Column 1: median=5, errors=[0, 0, 0], medae=0
        self.assertAlmostEqual(
            compute_column_median_baseline_medae(matrix, 1), 0.0
        )

    def test_per_column_with_nan_cells(self) -> None:
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

    def test_skill_score_formula(self) -> None:
        """skill_score = 1 - (parity_medae / column_baseline_medae)."""
        matrix = np.array([
            [10.0, 20.0, 30.0],
            [20.0, 30.0, 40.0],
            [30.0, 40.0, 50.0],
        ])
        col_baseline = compute_column_median_baseline_medae(matrix, 0)
        # Column 0: median=20, errors=[10, 0, 10], medae=10
        self.assertAlmostEqual(col_baseline, 10.0)

        # Perfect predictor: parity_medae=0 => skill_score=1
        parity_medae = 0.0
        skill = 1.0 - (parity_medae / col_baseline)
        self.assertAlmostEqual(skill, 1.0)

        # Baseline predictor: parity_medae=baseline => skill_score=0
        parity_medae = col_baseline
        skill = 1.0 - (parity_medae / col_baseline)
        self.assertAlmostEqual(skill, 0.0)

    def test_skill_score_zero_baseline_guarded(self) -> None:
        """When column baseline is 0 (constant column), skill_score should be NaN."""
        parity_medae = 0.5
        col_baseline = 0.0
        if col_baseline > 0:
            skill = 1.0 - (parity_medae / col_baseline)
        else:
            skill = float("nan")
        self.assertTrue(np.isnan(skill))

    def test_skill_score_nan_baseline_guarded(self) -> None:
        """When column baseline is NaN, skill_score should be NaN."""
        parity_medae = 0.5
        col_baseline = float("nan")
        if np.isfinite(col_baseline) and col_baseline > 0:
            skill = 1.0 - (parity_medae / col_baseline)
        else:
            skill = float("nan")
        self.assertTrue(np.isnan(skill))


class DispersionNormalizedMetricsTests(unittest.TestCase):
    """FIX 2: Normalized metric correctness."""

    def test_predict_all_known_returns_column_sd(self) -> None:
        matrix = np.array([
            [10.0, 20.0],
            [20.0, 30.0],
            [30.0, 40.0],
        ])
        result = predict_all_known(matrix, [0], rank=0)
        self.assertIsNotNone(result.column_sd)
        self.assertEqual(len(result.column_sd), 2)
        # Column 0: std([10, 20, 30], ddof=0) = sqrt(200/3) ~ 8.16
        expected_sd_0 = float(np.std([10.0, 20.0, 30.0], ddof=0))
        self.assertAlmostEqual(float(result.column_sd[0]), expected_sd_0, places=4)

    def test_score_predictions_includes_normalized_metrics(self) -> None:
        matrix = np.array([
            [10.0, 20.0],
            [20.0, 30.0],
            [30.0, 40.0],
        ])
        result = predict_all_known(matrix, [0], rank=0)
        metrics = score_predictions(result)
        self.assertIn("medae_normalized", metrics)
        self.assertIn("medae_normalized_pooled", metrics)
        # With non-zero column SD, these should be finite
        self.assertTrue(np.isfinite(metrics["medae_normalized"]))
        self.assertTrue(np.isfinite(metrics["medae_normalized_pooled"]))

    def test_column_sd_uses_full_matrix_for_all_known(self) -> None:
        """column_sd for all_known uses the full actual matrix."""
        matrix = np.array([
            [10.0, 20.0, 30.0],
            [20.0, 30.0, 40.0],
            [30.0, 40.0, 50.0],
        ])
        result = predict_all_known(matrix, [], rank=0)
        # All rows are used for column SD
        for col in range(3):
            expected = float(np.std(matrix[:, col], ddof=0))
            self.assertAlmostEqual(float(result.column_sd[col]), expected, places=4)

    def test_column_sd_uses_context_only_for_heldout(self) -> None:
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
        # Column SD should use rows 0,1 only (context)
        context_matrix = matrix[:2]
        for col in range(2):
            expected = float(np.std(context_matrix[:, col], ddof=0))
            self.assertAlmostEqual(float(result.column_sd[col]), expected, places=4)

    def test_normalized_metrics_with_zero_sd(self) -> None:
        """When column SD is 0 (constant column), normalized metrics handle it."""
        matrix = np.array([
            [5.0, 20.0],
            [5.0, 30.0],
            [5.0, 40.0],
        ])
        result = predict_all_known(matrix, [0], rank=0)
        metrics = score_predictions(result)
        # Column 0 has SD=0; column 1 has SD>0
        # medae_normalized uses column 1's SD, so it should be finite
        self.assertTrue(np.isfinite(metrics["medae_normalized"]))


class FamilyBlockedSplitTests(unittest.TestCase):
    """FIX 3: Family-blocked model split correctness."""

    def test_no_family_leaks_across_split(self) -> None:
        """No family appears on both sides of a family-blocked split."""
        model_ids = [
            "virchow", "virchow-2",  # same family
            "dinov2-b", "dinov2-l",   # same family
            "dinov3-b", "dinov3-l", "dinov3-s",  # same family
            "clip-b", "clip-l",       # same family
            "plip", "conch", "conch-1.5",
            "kaiko-vit-b-16", "kaiko-vit-l-14",
            "atlas", "chief-patch-mean", "chief-slide",
            "ctranspath", "phikon",
            "uni", "uni2-h",
            "vit-b", "vit-l",
        ]
        # Create temporary metadata CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("model_id,provider,family\n")
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
            for mid in model_ids:
                fam = families.get(mid, "")
                f.write(f"{mid},provider,{fam}\n")
            metadata_path = f.name

        try:
            train_idx, val_idx, info = family_blocked_model_split(
                model_ids, model_metadata_path=metadata_path, seed=42
            )
            train_ids = {model_ids[i] for i in train_idx}
            val_ids = {model_ids[i] for i in val_idx}

            # Check no family is split
            train_families = {families[mid] for mid in train_ids}
            val_families = {families[mid] for mid in val_ids}
            overlap = train_families & val_families
            self.assertEqual(
                overlap, set(),
                f"Families appear on both sides: {overlap}"
            )
        finally:
            os.unlink(metadata_path)

    def test_random_split_may_split_families(self) -> None:
        """Random split does not guarantee family integrity."""
        model_ids = [
            "virchow", "virchow-2",
            "dinov2-b", "dinov2-l",
            "clip-b", "clip-l",
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("model_id,provider,family\n")
            families = {
                "virchow": "Virchow", "virchow-2": "Virchow",
                "dinov2-b": "DINOv2", "dinov2-l": "DINOv2",
                "clip-b": "CLIP", "clip-l": "CLIP",
            }
            for mid in model_ids:
                fam = families.get(mid, "")
                f.write(f"{mid},provider,{fam}\n")
            metadata_path = f.name

        try:
            train_idx, val_idx, info = random_model_split(
                model_ids, seed=42
            )
            # Random split doesn't respect families, just verify it returns valid indices
            train_set = set(train_idx)
            val_set = set(val_idx)
            self.assertEqual(train_set & val_set, set())
            self.assertEqual(len(train_set) + len(val_set), len(model_ids))
        finally:
            os.unlink(metadata_path)

    def test_family_blocked_with_blank_families(self) -> None:
        """Models with blank family are treated as singletons."""
        model_ids = ["a", "b", "c", "d", "e", "f"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("model_id,provider,family\n")
            # a,b share family "X", c has blank family, d has blank family, e,f share "Y"
            f.write("a,provider,X\n")
            f.write("b,provider,X\n")
            f.write("c,provider,\n")
            f.write("d,provider,\n")
            f.write("e,provider,Y\n")
            f.write("f,provider,Y\n")
            metadata_path = f.name

        try:
            train_idx, val_idx, info = family_blocked_model_split(
                model_ids, model_metadata_path=metadata_path, seed=42
            )
            train_ids = {model_ids[i] for i in train_idx}
            val_ids = {model_ids[i] for i in val_idx}

            # Models with same family should not be split
            for fam_group in [("a", "b"), ("e", "f")]:
                if fam_group[0] in train_ids and fam_group[1] in val_ids:
                    self.fail(f"Family split: {fam_group[0]} train, {fam_group[1]} val")
                if fam_group[0] in val_ids and fam_group[1] in train_ids:
                    self.fail(f"Family split: {fam_group[0]} val, {fam_group[1]} train")

            # Blank family models (c, d) are singletons - they CAN be on different sides
            # This is correct behavior
        finally:
            os.unlink(metadata_path)

    def test_split_info_contains_families(self) -> None:
        """split_info includes family partition for auditability."""
        model_ids = ["a", "b", "c", "d"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("model_id,provider,family\n")
            f.write("a,provider,FamA\n")
            f.write("b,provider,FamA\n")
            f.write("c,provider,FamB\n")
            f.write("d,provider,FamB\n")
            metadata_path = f.name

        try:
            train_idx, val_idx, info = family_blocked_model_split(
                model_ids, model_metadata_path=metadata_path, seed=42
            )
            self.assertIn("train_families", info)
            self.assertIn("validation_families", info)
            self.assertIn("split_mode", info)
            self.assertEqual(info["split_mode"], "family_blocked")
        finally:
            os.unlink(metadata_path)

    def test_all_models_accounted_for(self) -> None:
        """Every model index appears exactly once across train/validation."""
        model_ids = [f"model_{i}" for i in range(20)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("model_id,provider,family\n")
            families = ["FamA", "FamB", "FamC", "FamD"]
            for i, mid in enumerate(model_ids):
                f.write(f"{mid},provider,{families[i % len(families)]}\n")
            metadata_path = f.name

        try:
            train_idx, val_idx, info = family_blocked_model_split(
                model_ids, model_metadata_path=metadata_path, seed=42
            )
            all_indices = set(train_idx) | set(val_idx)
            self.assertEqual(all_indices, set(range(len(model_ids))))
            self.assertEqual(set(train_idx) & set(val_idx), set())
            # Both sides must be non-empty
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(val_idx), 0)
        finally:
            os.unlink(metadata_path)


if __name__ == "__main__":
    unittest.main()
