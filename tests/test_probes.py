import unittest

import numpy as np

from pathopress.probes import (
    evaluate_column_median_baseline,
    evaluate_global_probes,
    evaluate_heldout_model_probes,
    evaluate_random_global_prefixes,
    greedy_probe_selection,
    random_global_probe_prefixes,
)


MATRIX = np.array(
    [
        [10.0, 20.0, 30.0],
        [20.0, 30.0, 40.0],
        [30.0, 40.0, 50.0],
    ]
)


def column_mean_predictor(train: np.ndarray) -> np.ndarray:
    means = np.nanmean(train, axis=0)
    return np.broadcast_to(means, train.shape).copy()


class GlobalProbeEvaluationTests(unittest.TestCase):
    def test_parity_hidden_and_model_average_use_distinct_denominators(self) -> None:
        result = evaluate_global_probes(
            MATRIX, [0], predictor=column_mean_predictor
        )

        self.assertEqual(result.n_target_cells, 9)
        self.assertEqual(result.n_revealed_cells, 3)
        self.assertEqual(result.n_hidden_cells, 6)
        self.assertEqual(result.parity.n, 9)
        self.assertEqual(result.hidden_only.n, 6)
        self.assertEqual(result.model_average.n, 3)
        self.assertLessEqual(
            result.parity.median_absolute_error,
            result.hidden_only.median_absolute_error,
        )

        # Row 0 keeps score 10 and predicts the other-row column means 35/45.
        first = result.model_average_predictions[0]
        self.assertEqual(first.actual_average, 20.0)
        self.assertAlmostEqual(first.predicted_average, 30.0)
        self.assertAlmostEqual(first.absolute_error, 10.0)

    def test_empty_probe_set_allows_an_empty_target_row(self) -> None:
        result = evaluate_global_probes(MATRIX, [], rank=0)

        self.assertEqual(result.n_revealed_cells, 0)
        self.assertEqual(result.n_hidden_cells, 9)
        self.assertTrue(np.isfinite(result.hidden_only.median_absolute_error))

    def test_all_columns_revealed_have_zero_parity_and_no_hidden_metric(self) -> None:
        result = evaluate_global_probes(
            MATRIX, [0, 1, 2], predictor=column_mean_predictor
        )

        self.assertEqual(result.parity.mean_absolute_error, 0.0)
        self.assertEqual(result.model_average.mean_absolute_error, 0.0)
        self.assertEqual(result.hidden_only.n, 0)
        self.assertTrue(np.isnan(result.hidden_only.median_absolute_error))

    def test_column_median_baseline_uses_full_columns(self) -> None:
        result = evaluate_column_median_baseline(MATRIX)

        self.assertEqual(result.probe_indices, ())
        self.assertEqual(result.n_hidden_cells, 9)
        self.assertEqual(result.parity.median_absolute_error, 10.0)

    def test_probe_indices_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            evaluate_global_probes(MATRIX, [0, 0])
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            evaluate_global_probes(MATRIX, [3])


class ProbeSelectionTests(unittest.TestCase):
    @staticmethod
    def informative_probe_predictor(train: np.ndarray) -> np.ndarray:
        predicted = np.broadcast_to(np.nanmean(train, axis=0), train.shape).copy()
        for row in range(train.shape[0]):
            if np.isfinite(train[row, 0]):
                offset = train[row, 0] - 10.0
                predicted[row] = np.array([10.0, 20.0, 30.0]) + offset
        return predicted

    def test_greedy_selection_ranks_the_most_predictive_probe_first(self) -> None:
        steps = greedy_probe_selection(
            MATRIX,
            max_probes=2,
            predictor=self.informative_probe_predictor,
        )

        self.assertEqual([step.added_probe_index for step in steps], [0, 1])
        self.assertEqual(steps[0].probe_indices, (0,))
        self.assertEqual(len(steps[0].candidate_scores), 3)
        self.assertEqual(steps[0].objective_value, 0.0)

    def test_random_prefixes_are_nested_reproducible_and_globally_shared(self) -> None:
        first = random_global_probe_prefixes(5, max_probes=3, repeats=2, seed=7)
        second = random_global_probe_prefixes(5, max_probes=3, repeats=2, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(first[0][0], first[0][1][:1])
        self.assertEqual(first[0][1], first[0][2][:2])
        self.assertNotEqual(first[0], first[1])

    def test_random_prefix_evaluation_returns_repeat_by_budget_grid(self) -> None:
        results = evaluate_random_global_prefixes(
            MATRIX,
            max_probes=2,
            repeats=3,
            seed=4,
            predictor=column_mean_predictor,
        )

        self.assertEqual(len(results), 6)
        self.assertEqual(
            [(result.repeat, result.k) for result in results],
            [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)],
        )
        for repeat in range(3):
            rows = [result for result in results if result.repeat == repeat]
            self.assertEqual(rows[0].probe_indices, rows[1].probe_indices[:1])


class HeldoutModelProbeTests(unittest.TestCase):
    def test_targets_are_isolated_from_each_other_and_context_is_visible(self) -> None:
        matrix = np.array(
            [
                [10.0, 20.0, 30.0],
                [20.0, 30.0, 40.0],
                [30.0, 40.0, 50.0],
                [40.0, 50.0, 60.0],
            ]
        )
        calls: list[np.ndarray] = []

        def checking_predictor(train: np.ndarray) -> np.ndarray:
            calls.append(train.copy())
            means = np.nanmean(train, axis=0)
            return np.broadcast_to(means, train.shape).copy()

        result = evaluate_heldout_model_probes(
            matrix,
            [0],
            target_model_indices=[2, 3],
            context_model_indices=[0, 1],
            predictor=checking_predictor,
        )

        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0][:2], matrix[:2])
        self.assertTrue(np.isfinite(calls[0][2, 0]))
        self.assertTrue(np.isnan(calls[0][2, 1:]).all())
        self.assertTrue(np.isnan(calls[0][3]).all())
        self.assertTrue(np.isnan(calls[1][2]).all())
        self.assertTrue(np.isfinite(calls[1][3, 0]))

        self.assertEqual(result.scorecard.n_target_cells, 6)
        self.assertEqual(result.scorecard.n_revealed_cells, 2)
        self.assertEqual(result.scorecard.n_hidden_cells, 4)
        self.assertEqual(result.primary, result.scorecard.hidden_only)
        self.assertEqual(result.primary.n, 4)

    def test_include_probe_targets_selects_parity_as_primary(self) -> None:
        result = evaluate_heldout_model_probes(
            MATRIX,
            [0],
            target_model_indices=[2],
            context_model_indices=[0, 1],
            include_probe_targets=True,
            predictor=column_mean_predictor,
        )

        self.assertEqual(result.primary, result.scorecard.parity)
        self.assertEqual(result.primary.n, 3)
        self.assertEqual(result.scorecard.hidden_only.n, 2)

    def test_target_with_no_available_probe_can_be_completed(self) -> None:
        matrix = np.array(
            [
                [10.0, 20.0, 30.0],
                [20.0, 30.0, 40.0],
                [30.0, np.nan, 50.0],
            ]
        )

        result = evaluate_heldout_model_probes(
            matrix,
            [1],
            target_model_indices=[2],
            context_model_indices=[0, 1],
            rank=0,
        )

        self.assertEqual(result.scorecard.n_revealed_cells, 0)
        self.assertEqual(result.scorecard.n_hidden_cells, 2)
        self.assertTrue(np.isfinite(result.primary.median_absolute_error))

    def test_target_and_context_rows_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "disjoint"):
            evaluate_heldout_model_probes(
                MATRIX,
                [0],
                target_model_indices=[1],
                context_model_indices=[0, 1],
            )


if __name__ == "__main__":
    unittest.main()
