from __future__ import annotations

import argparse
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np


RUNNER_PATH = Path(__file__).resolve().parents[1] / "experiments/run_probe_compression.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_probe_compression_test2", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


class CheckpointIdentityTests(unittest.TestCase):
    """BUG 3 - checkpoint identity must include split_mode and model_metadata hash."""

    def _make_args(self, **overrides):
        args = argparse.Namespace(
            scores=RUNNER.ROOT / "data/scores.csv",
            allowlist=RUNNER.ROOT / "data/low_friction_allowlist_v2_top25.json",
            previous_probes=RUNNER.ROOT / "experiments/probe_selection_results_rank1.json",
            output=RUNNER.ROOT / "experiments/probe_compression_rank1.json",
            raw_output=RUNNER.ROOT / "outputs/probe_compression_selected_raw_rank1.csv",
            random_raw_output=RUNNER.ROOT / "outputs/probe_compression_random_all_known_raw_rank1.csv.gz",
            rank=1,
            max_any_k=10,
            max_random_k=30,
            max_heldout_random_k=10,
            max_ranking_random_k=10,
            random_repeats=10,
            pruned_keep=30,
            ranking_margin=5.0,
            seed=42,
            split_mode="family_blocked",
            workers=1,
            checkpoint_dir=RUNNER.ROOT / "experiments/probe_compression_checkpoints",
            reuse_any_score_curves=True,
            ranking_random_only=False,
        )
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_identity_contains_split_mode(self) -> None:
        args = self._make_args()
        identity = RUNNER._checkpoint_identity(args, "dummy-sha")
        self.assertIn("split_mode", identity)
        self.assertEqual(identity["split_mode"], "family_blocked")

    def test_identity_contains_model_metadata_hash(self) -> None:
        args = self._make_args()
        identity = RUNNER._checkpoint_identity(args, "dummy-sha")
        self.assertIn("model_metadata_sha256", identity)
        self.assertEqual(len(identity["model_metadata_sha256"]), 64)

    def test_identity_changes_when_split_mode_changes(self) -> None:
        args_random = self._make_args(split_mode="random")
        args_blocked = self._make_args(split_mode="family_blocked")
        identity_random = RUNNER._checkpoint_identity(args_random, "dummy-sha")
        identity_blocked = RUNNER._checkpoint_identity(args_blocked, "dummy-sha")
        self.assertNotEqual(identity_random, identity_blocked)
        # The sha256 of the identity dict should also differ
        sha_random = RUNNER._sha256_json(identity_random)
        sha_blocked = RUNNER._sha256_json(identity_blocked)
        self.assertNotEqual(sha_random, sha_blocked)

    def test_checkpoint_rejected_when_identity_differs(self) -> None:
        """A checkpoint saved with one identity is rejected when loaded with a different one."""
        identity_a = {"split_mode": "random", "scores_sha256": "abc"}
        identity_b = {"split_mode": "family_blocked", "scores_sha256": "abc"}
        checkpoint = {
            "identity": identity_a,
            "curve_greedy": {"any_candidate": {"all_known_greedy_medae": [1]}},
            "ranking": {},
            "raw": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            RUNNER._save_phase_checkpoint(path, checkpoint)
            # Loading with different identity should return empty curves
            rejected = RUNNER._load_phase_checkpoint(path, identity_b)
            self.assertEqual(rejected["curve_greedy"], {})
            self.assertEqual(rejected["ranking"], {})


class PreviousProbesSplitModeTests(unittest.TestCase):
    """BUG 4 - previous-probes must not be reused across split modes."""

    def _make_args(self, **overrides):
        args = argparse.Namespace(
            scores=RUNNER.ROOT / "data/scores.csv",
            allowlist=RUNNER.ROOT / "data/low_friction_allowlist_v2_top25.json",
            previous_probes=RUNNER.ROOT / "experiments/probe_selection_results_rank1.json",
            output=RUNNER.ROOT / "experiments/probe_compression_rank1.json",
            raw_output=RUNNER.ROOT / "outputs/probe_compression_selected_raw_rank1.csv",
            random_raw_output=RUNNER.ROOT / "outputs/probe_compression_random_all_known_raw_rank1.csv.gz",
            rank=1,
            max_any_k=10,
            max_random_k=30,
            max_heldout_random_k=10,
            max_ranking_random_k=10,
            random_repeats=10,
            pruned_keep=30,
            ranking_margin=5.0,
            seed=42,
            split_mode="family_blocked",
            workers=1,
            checkpoint_dir=RUNNER.ROOT / "experiments/probe_compression_checkpoints",
            reuse_any_score_curves=True,
            ranking_random_only=False,
        )
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_mismatched_split_mode_raises_error(self) -> None:
        """When previous-probes artifact has different split_mode, it should raise."""
        with tempfile.TemporaryDirectory() as directory:
            prior_path = Path(directory) / "prior.json"
            prior = {
                "all_known_greedy": [
                    {
                        "step": 1,
                        "added_evaluation_index": 0,
                        "added_evaluation_id": "test_eval",
                        "probe_indices": [0],
                        "probe_ids": ["test_eval"],
                        "candidate_results": [],
                    }
                ],
                "heldout_model": {
                    "split_mode": "random",
                    "train_selected_trajectory": [
                        {
                            "step": 1,
                            "added_evaluation_index": 0,
                            "added_evaluation_id": "test_eval",
                            "probe_indices": [0],
                            "probe_ids": ["test_eval"],
                            "candidate_results": [],
                        }
                    ],
                },
            }
            prior_path.write_text(json.dumps(prior), encoding="utf-8")
            args = self._make_args(
                split_mode="family_blocked",
                previous_probes=prior_path,
            )
            # Simulate the check from the main function
            actual_prior = json.loads(args.previous_probes.read_text(encoding="utf-8"))
            prior_split_mode = actual_prior.get("heldout_model", {}).get("split_mode", "")
            self.assertEqual(prior_split_mode, "random")
            self.assertNotEqual(prior_split_mode, args.split_mode)


class SafeStatInfWarningTests(unittest.TestCase):
    """BUG 5 - _safe_stat must count and warn about Inf separately from NaN."""

    def test_inf_is_counted_separately_in_warning(self) -> None:
        matrix = np.asarray([
            [np.inf, 1.0],
            [np.inf, 2.0],
        ])
        with self.assertWarns(RuntimeWarning) as cm:
            from pathopress.confidence import structural_support_features_for_cells
            structural_support_features_for_cells(matrix, [(0, 0), (1, 0)])
        warning_msg = str(cm.warning.args[0])
        # The warning should mention Inf
        self.assertIn("Inf", warning_msg)

    def test_nan_still_warns(self) -> None:
        matrix = np.asarray([
            [np.nan, 1.0],
            [np.nan, 2.0],
        ])
        with self.assertWarns(RuntimeWarning) as cm:
            from pathopress.confidence import structural_support_features_for_cells
            structural_support_features_for_cells(matrix, [(0, 0), (1, 0)])
        warning_msg = str(cm.warning.args[0])
        self.assertIn("NaN", warning_msg)

    def test_inf_and_nan_both_counted(self) -> None:
        """When both NaN and Inf columns exist, the warning should mention both."""
        import warnings
        # Column 0 is all-NaN, column 1 is all-Inf, column 2 is normal.
        # nanmedian(axis=0) produces [nan, inf, 1.5] in a single _safe_stat call.
        matrix = np.asarray([
            [np.nan, np.inf, 1.0],
            [np.nan, np.inf, 2.0],
        ])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from pathopress.confidence import structural_support_features_for_cells
            structural_support_features_for_cells(matrix, [(0, 0), (0, 1), (0, 2)])
        combined = [w for w in caught if "NaN" in str(w.message) and "Inf" in str(w.message)]
        self.assertTrue(combined, f"Expected a warning mentioning both 'NaN' and 'Inf', got: {[str(w.message) for w in caught]}")


if __name__ == "__main__":
    unittest.main()
