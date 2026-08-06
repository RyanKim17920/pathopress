from __future__ import annotations

import math
import csv
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pathopress.probe_compression import (
    ProbePredictions,
    SCORE_RECONSTRUCTION_PAIRWISE_DIAGNOSTIC_MARGIN,
    candidate_prefixes,
    merge_shards,
    objective_value,
    predict_all_known,
    predict_heldout_models,
    rank_prune_trajectory,
    score_predictions,
    sharded_combinations,
)

from scripts.plot_probe_compression import probe_ticks


RUNNER_PATH = Path(__file__).resolve().parents[1] / "experiments/run_probe_compression.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_probe_compression_test", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


MATRIX = np.array(
    [
        [80.0, 70.0, 60.0],
        [75.0, 67.0, 59.0],
        [72.0, 65.0, 55.0],
        [68.0, 62.0, 53.0],
    ]
)


class ProbeCompressionTests(unittest.TestCase):
    def test_phase_checkpoint_roundtrip_and_identity_fail_closed(self) -> None:
        identity = {"scores_sha256": "current", "prediction_rank": 1}
        checkpoint = {
            "identity": identity,
            "curve_greedy": {"any_candidate": {"all_known_greedy_medae": [1]}},
            "ranking": {},
            "raw": [{"k": 1}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            RUNNER._save_phase_checkpoint(path, checkpoint)
            self.assertEqual(RUNNER._load_phase_checkpoint(path, identity), checkpoint)

            rejected = RUNNER._load_phase_checkpoint(
                path, {"scores_sha256": "different", "prediction_rank": 1}
            )
            self.assertEqual(rejected["curve_greedy"], {})
            self.assertEqual(rejected["ranking"], {})
            self.assertEqual(rejected["raw"], [])

    def test_runner_separates_k30_all_known_from_k10_controls(self) -> None:
        with patch.object(sys, "argv", [str(RUNNER_PATH)]):
            args = RUNNER.parse_args()
        self.assertEqual(args.max_random_k, 30)
        self.assertEqual(args.max_heldout_random_k, 10)
        self.assertEqual(args.max_ranking_random_k, 10)

    def test_all_known_random_curve_can_stream_per_cell_raw_rows(self) -> None:
        class ImmediateExecutor:
            @staticmethod
            def map(function, jobs):
                return map(function, jobs)

        fields = [
            "protocol", "candidate_mode", "method", "selection_objective",
            "repeat", "k", "model_id", "evaluation_id",
            "actual_normalized_score", "predicted_normalized_score",
            "is_revealed_probe_cell", "is_hidden_prediction",
        ]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        curves = RUNNER._random_curves(
            ImmediateExecutor(), MATRIX, [0, 1, 2], max_k=3, repeats=2,
            seed=42, rank=1, evaluations=["e0", "e1", "e2"],
            raw_writer=writer, models=["m0", "m1", "m2", "m3"],
            candidate_mode="any_candidate",
        )
        self.assertEqual(len(curves), 6)
        output.seek(0)
        rows = list(csv.DictReader(output))
        self.assertEqual(len(rows), 6 * MATRIX.size)
        self.assertEqual({int(row["k"]) for row in rows}, {1, 2, 3})
        self.assertEqual({row["method"] for row in rows}, {"random_prefix"})

    def test_all_known_reveals_probes_exactly_and_scores_medae_medape(self) -> None:
        predictions = predict_all_known(MATRIX, [1], rank=1)
        np.testing.assert_array_equal(predictions.predicted[:, 1], MATRIX[:, 1])
        self.assertEqual(int(predictions.revealed_mask.sum()), 4)
        self.assertEqual(int(predictions.heldout_mask.sum()), 8)
        metrics = score_predictions(predictions)
        self.assertEqual(metrics["n_target"], 12)
        self.assertEqual(metrics["n_revealed"], 4)
        self.assertGreaterEqual(float(metrics["medae"]), 0)
        self.assertGreaterEqual(float(metrics["medape"]), 0)
        self.assertEqual(
            metrics["pairwise_margin"],
            SCORE_RECONSTRUCTION_PAIRWISE_DIAGNOSTIC_MARGIN,
        )

    def test_medape_excludes_targets_at_or_below_pinned_epsilon(self) -> None:
        actual = np.array([[1e-7, 10.0], [1e-6, 20.0]])
        predicted = np.array([[1.0, 11.0], [2.0, 22.0]])
        target = np.ones_like(actual, dtype=bool)
        result = ProbePredictions(
            (), actual, predicted, target, np.zeros_like(target), target
        )
        metrics = score_predictions(result)
        self.assertAlmostEqual(float(metrics["medape"]), 10.0)
        self.assertAlmostEqual(float(metrics["hidden_medape"]), 10.0)

    def test_heldout_target_is_not_visible_in_context(self) -> None:
        result = predict_heldout_models(MATRIX, [0], [3], [0, 1, 2], rank=1)
        self.assertEqual(int(result.target_mask.sum()), 3)
        self.assertTrue(result.revealed_mask[3, 0])
        self.assertEqual(result.predicted[3, 0], MATRIX[3, 0])
        self.assertTrue(np.isfinite(result.predicted[3, 1:]).all())
        self.assertFalse(result.target_mask[:3].any())

    def test_ranking_losses_are_one_minus_median_metric(self) -> None:
        metrics = score_predictions(predict_all_known(MATRIX, [0], rank=1))
        self.assertAlmostEqual(
            objective_value(metrics, "pairwise_margin_error"),
            1 - float(metrics["pairwise_median_accuracy"]),
        )
        self.assertAlmostEqual(
            objective_value(metrics, "top_fraction_error"),
            1 - float(metrics["top_median_recovery"]),
        )

    def test_ranking_scope_matches_upstream_probe_denominators(self) -> None:
        actual = np.array([[100.0], [90.0], [80.0], [70.0]])
        predicted = np.array([[100.0], [90.0], [70.0], [80.0]])
        target = np.ones_like(actual, dtype=bool)
        revealed = np.array([[True], [True], [False], [False]])
        heldout = target & ~revealed
        result = ProbePredictions(
            (0,), actual, predicted, target, revealed, heldout
        )

        legacy = score_predictions(result, pairwise_margin=5.0)
        all_target = score_predictions(
            result, pairwise_margin=5.0, ranking_scope="all_target"
        )
        hidden_only = score_predictions(
            result, pairwise_margin=5.0, ranking_scope="hidden_only"
        )

        self.assertEqual(legacy["pairwise_n_pairs"], 5)
        self.assertAlmostEqual(float(legacy["pairwise_median_accuracy"]), 4 / 5)
        self.assertEqual(all_target["pairwise_n_pairs"], 6)
        self.assertAlmostEqual(float(all_target["pairwise_median_accuracy"]), 5 / 6)
        self.assertEqual(hidden_only["pairwise_n_pairs"], 1)
        self.assertEqual(float(hidden_only["pairwise_median_accuracy"]), 0.0)

    def test_ranking_scope_rejects_unknown_value(self) -> None:
        result = predict_all_known(MATRIX, [0], rank=1)
        with self.assertRaisesRegex(ValueError, "ranking_scope"):
            score_predictions(result, ranking_scope="probeish")

    def test_candidate_prefixes_are_nested_and_candidate_restricted(self) -> None:
        prefixes = candidate_prefixes([2, 5, 9], max_probes=3, repeats=2, seed=42)
        self.assertEqual(len(prefixes), 2)
        for repeat in prefixes:
            self.assertEqual(repeat[0], repeat[1][:1])
            self.assertEqual(repeat[1], repeat[2][:2])
            self.assertEqual(set(repeat[-1]), {2, 5, 9})

    def test_probe_plot_ticks_cover_upstream_random_k30_range(self) -> None:
        self.assertEqual(probe_ticks(10), list(range(1, 11)))
        self.assertEqual(probe_ticks(30), [1, 5, 10, 15, 20, 25, 30])

    def test_rank_pruning_uses_all_steps_normalized_rank_and_id_ties(self) -> None:
        trajectory = [
            {
                "step": 1,
                "added_evaluation_id": "b",
                "candidate_results": [
                    {"evaluation_id": "b", "parity_medae": 1.0},
                    {"evaluation_id": "a", "parity_medae": 1.0},
                    {"evaluation_id": "c", "parity_medae": 3.0},
                ],
            },
            {
                "step": 2,
                "added_evaluation_id": "c",
                "candidate_results": [
                    {"evaluation_id": "a", "parity_medae": 4.0},
                    {"evaluation_id": "c", "parity_medae": 2.0},
                ],
            },
        ]
        result = rank_prune_trajectory(
            trajectory,
            ["a", "b", "c"],
            keep_count=1,
            score_key="parity_medae",
        )
        self.assertEqual(result["source_steps_used"], 2)
        self.assertEqual(result["ranked_steps"][0]["ranks"][0]["candidate_id"], "a")
        self.assertEqual(result["kept_ids"], ["a"])
        self.assertEqual(result["by_candidate"]["b"]["n_ranked_steps"], 1)

    def test_shards_match_exact_combination_space_and_merge_checks_completeness(self) -> None:
        candidates = [0, 1, 2, 3, 4]
        shards = [
            list(sharded_combinations(candidates, 2, shard_index=index, num_shards=3))
            for index in range(3)
        ]
        merged = merge_shards(shards, math.comb(len(candidates), 2))
        self.assertEqual([ordinal for ordinal, _ in merged], list(range(10)))
        self.assertEqual(len({combo for _, combo in merged}), 10)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            merge_shards(shards[:2], 10)

    def test_wave_and_shard_residue_formula_is_disjoint(self) -> None:
        candidates = range(6)
        pieces = [
            list(
                sharded_combinations(
                    candidates,
                    3,
                    shard_index=shard,
                    num_shards=2,
                    wave_index=wave,
                    num_waves=2,
                )
            )
            for shard in range(2)
            for wave in range(2)
        ]
        merged = merge_shards(pieces, math.comb(6, 3))
        self.assertEqual(len(merged), 20)


if __name__ == "__main__":
    unittest.main()
