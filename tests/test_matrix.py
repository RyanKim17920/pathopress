import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pathopress.matrix import Score, filter_matrix, load_scores, make_matrix


FIELDNAMES = (
    "model_id",
    "evaluation_id",
    "value",
    "normalized_score",
    "suite_id",
    "metric",
    "reference_url",
    "audit_status",
)


class LoadScoresTests(unittest.TestCase):
    def _write_csv(self, rows: list[dict[str, str]]) -> Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", suffix=".csv", delete=False
        )
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        with temporary:
            writer = csv.DictWriter(temporary, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        return Path(temporary.name)

    def test_load_scores_parses_values_and_defaults_to_verified_rows(self) -> None:
        path = self._write_csv(
            [
                {
                    "model_id": "model-b",
                    "evaluation_id": "eval-2",
                    "value": "0.83",
                    "normalized_score": "83.0",
                    "suite_id": "suite-a",
                    "metric": "auroc",
                    "reference_url": "https://example.test/paper",
                    "audit_status": "verified",
                },
                {
                    "model_id": "model-a",
                    "evaluation_id": "eval-1",
                    "value": "71",
                    "normalized_score": "71.0",
                    "suite_id": "suite-b",
                    "metric": "accuracy",
                    "reference_url": "https://example.test/report",
                    "audit_status": "pending",
                },
            ]
        )

        scores = load_scores(path)

        self.assertEqual(len(scores), 1)
        self.assertEqual(
            scores[0],
            Score(
                model_id="model-b",
                evaluation_id="eval-2",
                value=0.83,
                normalized_score=83.0,
                suite_id="suite-a",
                metric="auroc",
                reference_url="https://example.test/paper",
                audit_status="verified",
            ),
        )

    def test_load_scores_can_include_unverified_rows(self) -> None:
        path = self._write_csv(
            [
                {
                    "model_id": "model-a",
                    "evaluation_id": "eval-1",
                    "value": "71",
                    "normalized_score": "71.0",
                    "suite_id": "suite-b",
                    "metric": "accuracy",
                    "reference_url": "https://example.test/report",
                    "audit_status": "pending",
                }
            ]
        )

        scores = load_scores(path, verified_only=False)

        self.assertEqual([score.audit_status for score in scores], ["pending"])

    def test_primary_source_parses_are_included_in_prototype_matrix(self) -> None:
        path = self._write_csv(
            [
                {
                    "model_id": "model-a",
                    "evaluation_id": "eval-1",
                    "value": "0.71",
                    "normalized_score": "71.0",
                    "suite_id": "suite-b",
                    "metric": "accuracy",
                    "reference_url": "https://example.test/official-table",
                    "audit_status": "parsed_primary_source",
                }
            ]
        )

        self.assertEqual(len(load_scores(path)), 1)


class MatrixConstructionTests(unittest.TestCase):
    @staticmethod
    def _score(model: str, evaluation: str, normalized_score: float) -> Score:
        return Score(
            model_id=model,
            evaluation_id=evaluation,
            value=normalized_score,
            normalized_score=normalized_score,
            suite_id="suite",
            metric="accuracy",
            reference_url="https://example.test/source",
            audit_status="verified",
        )

    def test_make_matrix_rejects_duplicate_model_evaluation_cells(self) -> None:
        scores = [
            self._score("model-a", "eval-1", 70.0),
            self._score("model-a", "eval-1", 72.0),
        ]

        with self.assertRaisesRegex(ValueError, "duplicate score cell: model-a/eval-1"):
            make_matrix(scores)

    def test_make_matrix_rejects_non_finite_normalized_scores(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    make_matrix([self._score("model-a", "eval-1", value)])

    def test_non_finite_first_duplicate_is_not_silently_overwritten(self) -> None:
        scores = [
            self._score("model-a", "eval-1", np.nan),
            self._score("model-a", "eval-1", 72.0),
        ]

        with self.assertRaises(ValueError):
            make_matrix(scores)

    def test_filter_matrix_repeats_until_support_thresholds_are_stable(self) -> None:
        # First pass drops eval-3 (only one score). Without a second pass model-c
        # would incorrectly remain, despite then having only one supported score.
        matrix = np.array(
            [
                [80.0, 81.0, np.nan],
                [70.0, 71.0, np.nan],
                [60.0, np.nan, 62.0],
            ]
        )

        filtered, models, evaluations = filter_matrix(
            matrix,
            ["model-a", "model-b", "model-c"],
            ["eval-1", "eval-2", "eval-3"],
            min_scores_per_model=2,
            min_models_per_evaluation=2,
        )

        np.testing.assert_array_equal(filtered, [[80.0, 81.0], [70.0, 71.0]])
        self.assertEqual(models, ["model-a", "model-b"])
        self.assertEqual(evaluations, ["eval-1", "eval-2"])


if __name__ == "__main__":
    unittest.main()
