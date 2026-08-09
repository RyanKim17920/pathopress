from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.run_soft_impute_rank_sweep import (
    _load_checkpoint,
    _soft_job,
    run_sweep,
)


class SoftImputeRankSweepTests(unittest.TestCase):
    def test_fold_adapter_records_all_missing_column_holdouts(self) -> None:
        matrix = np.asarray(
            [[10.0, 20.0, 30.0], [20.0, 30.0, 40.0], [30.0, 40.0, 50.0]]
        )
        train = matrix.copy()
        held = [(0, 0), (0, 2), (1, 2), (2, 2)]
        for row, column in held:
            train[row, column] = np.nan

        result = _soft_job(("identity", 1, train, held, matrix))

        self.assertEqual(result[0:2], ("identity", 1))
        self.assertEqual(result[2], [10.0])
        self.assertEqual(len(result[3]), 1)
        self.assertEqual(result[5], 0)
        self.assertEqual(result[6], 3)

    def test_resumed_sweep_exactly_matches_uninterrupted_sweep(self) -> None:
        matrix = np.asarray(
            [
                [12.0, 21.0, 35.0, 44.0],
                [18.0, 27.0, 32.0, 49.0],
                [31.0, 38.0, 46.0, 52.0],
                [43.0, 51.0, 58.0, 67.0],
                [55.0, 62.0, 71.0, 78.0],
            ]
        )
        models = [f"model-{index}" for index in range(matrix.shape[0])]
        evaluations = [f"eval-{index}" for index in range(matrix.shape[1])]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            uninterrupted_cache = root / "uninterrupted"
            resumed_cache = root / "resumed"
            common = {
                "matrix": matrix,
                "models": models,
                "evaluations": evaluations,
                "scores_sha256": "fixture-score-sha256",
                "ranks": (1, 2),
                "transforms": ("identity",),
                "seeds": (42, 43),
                "n_folds": 2,
                "workers": 2,
                "progress_every": 100,
            }
            blas_environment_before = {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            }
            uninterrupted = run_sweep(
                **common,
                output_path=root / "uninterrupted.json",
                checkpoint_root=uninterrupted_cache,
                resume=False,
            )
            self.assertEqual(
                {
                    name: os.environ.get(name)
                    for name in blas_environment_before
                },
                blas_environment_before,
            )

            run_directory = next(uninterrupted_cache.iterdir())
            resumed_run_directory = resumed_cache / run_directory.name
            resumed_run_directory.mkdir(parents=True)
            first_checkpoint = sorted(run_directory.glob("*.json"))[0]
            shutil.copy2(first_checkpoint, resumed_run_directory / first_checkpoint.name)
            checkpoint_value = json.loads(first_checkpoint.read_text(encoding="utf-8"))
            wrong_identity = dict(checkpoint_value["run_identity"])
            wrong_identity["code_sha256"] = "wrong-code-sha256"
            self.assertIsNone(
                _load_checkpoint(
                    first_checkpoint,
                    run_identity=wrong_identity,
                    run_identity_sha256=checkpoint_value["run_identity_sha256"],
                    spec=checkpoint_value["job"],
                )
            )

            resumed = run_sweep(
                **common,
                output_path=root / "resumed.json",
                checkpoint_root=resumed_cache,
            )
            self.assertEqual(resumed, uninterrupted)
            self.assertEqual(
                len(list(resumed_run_directory.glob("*.json"))),
                8,
            )

            # A fully resumed run must perform no numerical fits at all.
            with patch(
                "experiments.run_soft_impute_rank_sweep.complete_soft_impute",
                side_effect=AssertionError("cached job was recomputed"),
            ):
                cached = run_sweep(
                    **common,
                    output_path=root / "cached.json",
                    checkpoint_root=resumed_cache,
                )
            self.assertEqual(cached, uninterrupted)


    def test_sweep_result_serializes_without_nan(self) -> None:
        """Regression: the sweep payload must not contain NaN values that would
        fail allow_nan=False JSON serialization."""
        matrix = np.asarray(
            [
                [12.0, 21.0, 35.0, 44.0],
                [18.0, 27.0, 32.0, 49.0],
                [31.0, 38.0, 46.0, 52.0],
                [43.0, 51.0, 58.0, 67.0],
                [55.0, 62.0, 71.0, 78.0],
            ]
        )
        models = [f"model-{index}" for index in range(matrix.shape[0])]
        evaluations = [f"eval-{index}" for index in range(matrix.shape[1])]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = run_sweep(
                matrix=matrix,
                models=models,
                evaluations=evaluations,
                scores_sha256="fixture-score-sha256",
                output_path=root / "result.json",
                checkpoint_root=root / "checkpoints",
                ranks=(1, 2),
                transforms=("identity",),
                seeds=(42,),
                n_folds=2,
                workers=1,
                resume=False,
                progress_every=100,
            )
        # The entire payload (including nested results) must serialize
        # with allow_nan=False — this is the contract enforced by
        # _atomic_write_json.
        from experiments.run_soft_impute_rank_sweep import _canonical_bytes
        _canonical_bytes(payload)  # should not raise

    def test_metrics_omits_normalized_keys_when_column_sd_unavailable(self) -> None:
        """The metrics() function must omit medae_normalized when column_sd
        is None, rather than emitting NaN."""
        from experiments.run_benchpress_style import metrics
        result = metrics([1.0, 2.0, 3.0], [1.1, 2.2, 3.3])
        self.assertNotIn("medae_normalized", result)
        result_with_sd = metrics([1.0, 2.0, 3.0], [1.1, 2.2, 3.3], column_sd=1.0)
        self.assertIn("medae_normalized", result_with_sd)
        self.assertTrue(np.isfinite(result_with_sd["medae_normalized"]))


    def test_sweep_result_serializes_without_nan(self) -> None:
        """Regression: the sweep payload must not contain NaN values that would
        fail allow_nan=False JSON serialization."""
        matrix = np.asarray(
            [
                [12.0, 21.0, 35.0, 44.0],
                [18.0, 27.0, 32.0, 49.0],
                [31.0, 38.0, 46.0, 52.0],
                [43.0, 51.0, 58.0, 67.0],
                [55.0, 62.0, 71.0, 78.0],
            ]
        )
        models = [f"model-{index}" for index in range(matrix.shape[0])]
        evaluations = [f"eval-{index}" for index in range(matrix.shape[1])]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = run_sweep(
                matrix=matrix,
                models=models,
                evaluations=evaluations,
                scores_sha256="fixture-score-sha256",
                output_path=root / "result.json",
                checkpoint_root=root / "checkpoints",
                ranks=(1, 2),
                transforms=("identity",),
                seeds=(42,),
                n_folds=2,
                workers=1,
                resume=False,
                progress_every=100,
            )
        # The entire payload (including nested results) must serialize
        # with allow_nan=False — this is the contract enforced by
        # _atomic_write_json.
        from experiments.run_soft_impute_rank_sweep import _canonical_bytes
        _canonical_bytes(payload)  # should not raise

    def test_metrics_omits_normalized_when_column_sd_unavailable(self) -> None:
        """metrics() must omit medae_normalized when column_sd is None,
        rather than emitting NaN."""
        from experiments.run_benchpress_style import metrics
        result = metrics([1.0, 2.0, 3.0], [1.1, 2.2, 3.3])
        self.assertNotIn("medae_normalized", result)
        result_with_sd = metrics([1.0, 2.0, 3.0], [1.1, 2.2, 3.3], column_sd=1.0)
        self.assertIn("medae_normalized", result_with_sd)
        self.assertTrue(np.isfinite(result_with_sd["medae_normalized"]))


if __name__ == "__main__":
    unittest.main()
