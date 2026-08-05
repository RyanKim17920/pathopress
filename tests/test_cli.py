import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pathopress.cli import main


class CliTests(unittest.TestCase):
    def test_audit_reports_filtered_matrix_dimensions_and_density(self) -> None:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", suffix=".csv", delete=False
        )
        scores_path = Path(temporary.name)
        self.addCleanup(scores_path.unlink, missing_ok=True)
        fieldnames = (
            "model_id",
            "evaluation_id",
            "value",
            "normalized_score",
            "suite_id",
            "metric",
            "reference_url",
            "audit_status",
        )
        with temporary:
            writer = csv.DictWriter(temporary, fieldnames=fieldnames)
            writer.writeheader()
            for model, evaluation, score in (
                ("model-a", "eval-1", "60"),
                ("model-a", "eval-2", "70"),
                ("model-b", "eval-1", "80"),
                ("model-b", "eval-2", "90"),
            ):
                writer.writerow(
                    {
                        "model_id": model,
                        "evaluation_id": evaluation,
                        "value": score,
                        "normalized_score": score,
                        "suite_id": "suite",
                        "metric": "accuracy",
                        "reference_url": "https://example.test/source",
                        "audit_status": "verified",
                    }
                )

        output = io.StringIO()
        argv = [
            "pathopress",
            "audit",
            "--scores",
            str(scores_path),
            "--min-scores-per-model",
            "2",
            "--min-models-per-evaluation",
            "2",
        ]

        with patch("sys.argv", argv), redirect_stdout(output):
            main()

        self.assertEqual(
            output.getvalue().splitlines(),
            ["matrix=2 models x 2 evaluations", "observed=4/4 (100.0%)"],
        )

    def test_audit_handles_a_fully_filtered_matrix(self) -> None:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", suffix=".csv", delete=False
        )
        scores_path = Path(temporary.name)
        self.addCleanup(scores_path.unlink, missing_ok=True)
        with temporary:
            writer = csv.DictWriter(
                temporary,
                fieldnames=(
                    "model_id",
                    "evaluation_id",
                    "value",
                    "normalized_score",
                    "suite_id",
                    "metric",
                    "reference_url",
                    "audit_status",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "model_id": "model-a",
                    "evaluation_id": "eval-1",
                    "value": "60",
                    "normalized_score": "60",
                    "suite_id": "suite",
                    "metric": "accuracy",
                    "reference_url": "https://example.test/source",
                    "audit_status": "verified",
                }
            )

        output = io.StringIO()
        argv = ["pathopress", "audit", "--scores", str(scores_path)]

        with patch("sys.argv", argv), redirect_stdout(output):
            main()

        self.assertEqual(
            output.getvalue().splitlines(),
            ["matrix=0 models x 0 evaluations", "observed=0/0 (0.0%)"],
        )


if __name__ == "__main__":
    unittest.main()
