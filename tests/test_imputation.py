import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pathopress.imputation import (
    IMPUTATION_FIELDS,
    build_imputation_rows,
    to_native_score,
    write_imputations,
)


class ImputationTests(unittest.TestCase):
    def test_build_rows_preserves_observed_and_labels_missing(self) -> None:
        matrix = np.array([[60.0, 70.0], [80.0, np.nan], [75.0, 85.0]])
        rows = build_imputation_rows(
            matrix,
            ["a", "b", "c"],
            ["x", "y"],
            {"x": ("suite", "f1"), "y": ("suite", "f1")},
            rank=1,
        )
        self.assertEqual(len(rows), 6)
        observed = next(r for r in rows if r["model_id"] == "a" and r["evaluation_id"] == "x")
        missing = next(r for r in rows if r["model_id"] == "b" and r["evaluation_id"] == "y")
        self.assertEqual(observed["status"], "observed")
        self.assertEqual(float(observed["normalized_score"]), 60.0)
        self.assertEqual(missing["status"], "imputed")
        self.assertTrue(0.0 <= float(missing["normalized_score"]) <= 100.0)
        self.assertEqual(missing["model_observations"], "1")
        self.assertEqual(missing["evaluation_observations"], "2")
        self.assertEqual(missing["comparison_rank"], "2")
        self.assertNotEqual(missing["rank_sensitivity_absolute_difference"], "")
        self.assertEqual(observed["rank_sensitivity_absolute_difference"], "")

    def test_native_metric_inverse_maps(self) -> None:
        self.assertAlmostEqual(to_native_score(80.0, "pearson_r"), 0.6)
        self.assertAlmostEqual(to_native_score(80.0, "weighted_kappa"), 0.6)
        self.assertAlmostEqual(to_native_score(82.2, "balanced_accuracy"), 0.822)
        self.assertAlmostEqual(to_native_score(82.2, "macro-ovr-auc"), 0.822)
        self.assertAlmostEqual(to_native_score(82.2, "robustness_index"), 0.822)
        self.assertAlmostEqual(to_native_score(65.6, "f1"), 65.6)

    def test_write_imputations_uses_stable_schema(self) -> None:
        handle = tempfile.NamedTemporaryFile(delete=False)
        path = Path(handle.name)
        handle.close()
        self.addCleanup(path.unlink, missing_ok=True)
        row = {field: "value" for field in IMPUTATION_FIELDS}
        write_imputations(path, [row])
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            self.assertEqual(tuple(reader.fieldnames or ()), IMPUTATION_FIELDS)
            self.assertEqual(next(reader), row)


if __name__ == "__main__":
    unittest.main()
