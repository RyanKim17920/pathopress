import unittest

import numpy as np

from pathopress.predictability import (
    aggregate_raw_predictions,
    holdout_half_per_model,
    prediction_error,
)


class PredictabilityTests(unittest.TestCase):
    def test_holdout_half_matches_benchpress_contract(self):
        matrix = np.arange(18, dtype=float).reshape(3, 6)
        matrix[2, 2:] = np.nan
        train, held = holdout_half_per_model(matrix, np.random.RandomState(0))
        self.assertEqual([len(held[i]) for i in range(3)], [3, 3, 0])
        self.assertEqual(sum(np.isnan(train[i, held[i]]).sum() for i in range(3)), 6)
        for i, columns in held.items():
            self.assertTrue(all(np.isfinite(matrix[i, j]) for j in columns))

    def test_prediction_error_uses_median_absolute_percentage_error(self):
        result = prediction_error([50.0, 80.0, 100.0], [55.0, 72.0, 90.0])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["medae"], 8.0)
        self.assertTrue(np.isclose(result["medape"], 10.0))

    def test_aggregate_raw_predictions_groups_and_seed_summaries(self):
        raw = [
            {"seed": 0, "model_id": "a", "actual": 50.0, "predicted": 52.0},
            {"seed": 1, "model_id": "a", "actual": 60.0, "predicted": 57.0},
            {"seed": 1, "model_id": "a", "actual": 70.0, "predicted": 74.0},
        ]
        rows = aggregate_raw_predictions(raw, group_key="model_id", group_ids=["a", "b"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model_id"], "a")
        self.assertEqual(rows[0]["n_test_cells"], 3)
        self.assertEqual(rows[0]["n_seeds"], 2)
        self.assertEqual(rows[0]["medae"], 3.0)
