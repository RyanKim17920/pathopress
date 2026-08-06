import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pathopress_confidence_runner", ROOT / "experiments" / "run_confidence_calibration.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class ConfidenceRunnerContractTests(unittest.TestCase):
    def test_generator_contract_is_three_plus_twelve(self) -> None:
        self.assertEqual(len(RUNNER.HP_VARIANTS), 3)
        self.assertEqual(
            [item["hp"]["lam"] for item in RUNNER.HP_VARIANTS],
            [0.01, 0.1, 1.0],
        )
        self.assertTrue(all(item["hp"]["rank"] == 1 for item in RUNNER.HP_VARIANTS))

        completed = []
        results = {}
        for index in range(15):
            transform = f"t{index}"
            method = f"m{index}"
            row = {
                "shard_id": f"s{index}", "status": "completed",
                "transform": transform, "method": method, "hp": {"k": index},
                "coverage": 1.0 if index != 0 else 0.5,
                "medape_median": float(15 - index),
            }
            completed.append(row)
            results.setdefault(transform, {})[method] = dict(row)
        selected = RUNNER._strong_rows(
            results, {row["shard_id"]: row for row in completed}
        )
        self.assertEqual(len(selected), 12)
        self.assertTrue(all(row["coverage"] >= 0.999 for row in selected))
        self.assertEqual(
            [row["medape_median"] for row in selected],
            sorted(row["medape_median"] for row in selected),
        )

    def test_row_alignment_rejects_any_target_coordinate_change(self) -> None:
        reference = {
            "fold_id": np.asarray([0, 1]),
            "test_i": np.asarray([2, 3]),
            "test_j": np.asarray([4, 5]),
            "actual": np.asarray([60.0, 70.0]),
        }
        candidate = {key: value.copy() for key, value in reference.items()}
        RUNNER._assert_aligned(reference, candidate, "valid")
        candidate["test_j"][1] = 6
        with self.assertRaisesRegex(ValueError, "test_j"):
            RUNNER._assert_aligned(reference, candidate, "invalid")

    def test_strong_rows_use_maximum_attainable_coverage(self) -> None:
        completed = []
        results = {}
        for index in range(14):
            coverage = 0.998 if index < 13 else 0.90
            row = {
                "shard_id": f"s{index}", "status": "completed",
                "transform": f"t{index}", "method": f"m{index}",
                "hp": {}, "coverage": coverage,
                "medape_median": float(index),
            }
            completed.append(row)
            results.setdefault(row["transform"], {})[row["method"]] = dict(row)
        selected = RUNNER._strong_rows(
            results, {row["shard_id"]: row for row in completed}
        )
        self.assertEqual(len(selected), 12)
        self.assertTrue(all(row["coverage"] == 0.998 for row in selected))


if __name__ == "__main__":
    unittest.main()
