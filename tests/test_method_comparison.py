from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from pathopress.artifacts import load_fold_artifact, write_fold_artifact
from pathopress.method_comparison import (
    HP_GRIDS,
    METHODS,
    TRANSFORMS,
    UnsupportedMethodError,
    apply_transform,
    complete_bench_knn,
    complete_benchmark_mean,
    complete_mlp,
    complete_model_knn,
    complete_model_mean,
    invert_transform,
    predict_scores,
)


ROOT = Path(__file__).resolve().parents[1]


def _runner_module():
    path = ROOT / "experiments" / "run_method_comparison.py"
    specification = importlib.util.spec_from_file_location("run_method_comparison", path)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class TransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = np.asarray(
            [[10.0, 20.0, np.nan], [25.0, np.nan, 50.0], [40.0, 70.0, 90.0], [60.0, 80.0, 95.0]]
        )

    def test_all_seven_transforms_preserve_observed_and_bound_predictions(self) -> None:
        for transform in TRANSFORMS:
            with self.subTest(transform=transform):
                normalized, state = apply_transform(self.matrix, transform)
                filled = complete_benchmark_mean(normalized)
                restored = invert_transform(filled, self.matrix, transform, state)
                observed = np.isfinite(self.matrix)
                np.testing.assert_array_equal(restored[observed], self.matrix[observed])
                self.assertTrue(np.isfinite(restored).all())
                self.assertTrue(np.all((restored >= 0.0) & (restored <= 100.0)))

    def test_prediction_pipeline_is_deterministic(self) -> None:
        first = predict_scores(self.matrix, "logit", "Benchmark Mean", {})
        second = predict_scores(self.matrix, "logit", "Benchmark Mean", {})
        np.testing.assert_allclose(first, second)

    def test_unidentifiable_fold_column_is_reported_as_noncoverage(self) -> None:
        training = self.matrix.copy()
        training[:, 2] = np.nan
        for method, hyperparameters in (
            ("Soft-Impute", {"rank": 1}),
            ("Benchmark Mean", {}),
        ):
            with self.subTest(method=method):
                predicted = predict_scores(training, "identity", method, hyperparameters)
                self.assertTrue(np.isnan(predicted[:, 2]).all())
                self.assertTrue(np.isfinite(predicted[:, :2]).all())


class CompleterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = np.asarray(
            [
                [0.0, 0.2, np.nan, 0.4, 0.5],
                [0.1, np.nan, 0.3, 0.4, 0.6],
                [0.2, 0.25, 0.35, np.nan, 0.7],
                [0.3, 0.4, 0.5, 0.6, np.nan],
                [0.4, 0.5, 0.6, 0.7, 0.8],
                [0.5, 0.6, 0.7, 0.8, 0.9],
            ]
        )

    def test_mean_and_knn_methods_cover_missing_cells(self) -> None:
        observed = np.isfinite(self.matrix)
        for function in (complete_benchmark_mean, complete_model_mean, complete_bench_knn, complete_model_knn):
            with self.subTest(method=function.__name__):
                output = function(self.matrix)
                np.testing.assert_array_equal(output[observed], self.matrix[observed])
                self.assertTrue(np.isfinite(output).all())

    def test_mlp_dependency_failure_is_explicit(self) -> None:
        real_import = __import__

        def import_without_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("test missing torch")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_without_torch):
            with self.assertRaisesRegex(UnsupportedMethodError, "requires torch"):
                complete_mlp(self.matrix, lr=1e-3, epochs=1, n_seeds=1)


class FoldTests(unittest.TestCase):
    def test_persisted_folds_partition_every_observation_per_seed(self) -> None:
        matrix = np.arange(6 * 7, dtype=float).reshape(6, 7)
        matrix[0, 0] = np.nan
        models = [f"m{i}" for i in range(6)]
        evaluations = [f"e{j}" for j in range(7)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.json"
            write_fold_artifact(path, matrix, models, evaluations, n_seeds=2, n_folds=3)
            folds = load_fold_artifact(path, matrix, models, evaluations)
        self.assertEqual(len(folds), 6)
        for _seed, _fold, training, held in folds:
            self.assertTrue(all(np.isnan(training[row, column]) for row, column in held))

    def test_matrix_identity_drift_is_rejected(self) -> None:
        matrix = np.arange(20, dtype=float).reshape(4, 5)
        models = [f"m{i}" for i in range(4)]
        evaluations = [f"e{j}" for j in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.json"
            write_fold_artifact(path, matrix, models, evaluations)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_fold_artifact(path, matrix, models[:-1] + ["changed"], evaluations)


class GridAndArtifactTests(unittest.TestCase):
    def test_exact_grid_plus_rank_sensitivity_rows(self) -> None:
        self.assertEqual(len(TRANSFORMS), 7)
        self.assertEqual(len(METHODS), 12)
        self.assertEqual(sum(len(HP_GRIDS[method]) for method in METHODS), 49)
        self.assertEqual(len(_runner_module().all_shards(Path("out"))), 343)
        self.assertEqual(HP_GRIDS["Soft-Impute"], [{"rank": 1}, {"rank": 2, "sensitivity": True}])
        self.assertIn({"rank": 2, "lam": 0.1, "sensitivity": True}, HP_GRIDS["Bias ALS"])

    def test_metric_aggregation_is_median_of_fold_metrics(self) -> None:
        runner = _runner_module()
        arrays = {
            "fold_id": np.asarray([0, 0, 1, 1]),
            "actual": np.asarray([10.0, 20.0, 10.0, 20.0]),
            "predicted": np.asarray([11.0, 22.0, 15.0, np.nan]),
        }
        metrics = runner._metrics(arrays)
        self.assertEqual(metrics["coverage"], 0.75)
        self.assertAlmostEqual(metrics["medae_median"], 3.25)
        self.assertAlmostEqual(metrics["medape_median"], 30.0)


if __name__ == "__main__":
    unittest.main()
