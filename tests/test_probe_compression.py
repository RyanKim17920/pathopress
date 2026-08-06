from __future__ import annotations

import math
import unittest

import numpy as np

from pathopress.probe_compression import (
    candidate_prefixes,
    merge_shards,
    objective_value,
    predict_all_known,
    predict_heldout_models,
    score_predictions,
    sharded_combinations,
)


MATRIX = np.array(
    [
        [80.0, 70.0, 60.0],
        [75.0, 67.0, 59.0],
        [72.0, 65.0, 55.0],
        [68.0, 62.0, 53.0],
    ]
)


class ProbeCompressionTests(unittest.TestCase):
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

    def test_candidate_prefixes_are_nested_and_candidate_restricted(self) -> None:
        prefixes = candidate_prefixes([2, 5, 9], max_probes=3, repeats=2, seed=42)
        self.assertEqual(len(prefixes), 2)
        for repeat in prefixes:
            self.assertEqual(repeat[0], repeat[1][:1])
            self.assertEqual(repeat[1], repeat[2][:2])
            self.assertEqual(set(repeat[-1]), {2, 5, 9})

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
