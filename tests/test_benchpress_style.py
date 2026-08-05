import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "run_benchpress_style.py"
SPEC = importlib.util.spec_from_file_location("run_benchpress_style", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BenchPressStyleFoldTests(unittest.TestCase):
    def test_folds_cover_every_observed_cell_once(self) -> None:
        matrix = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, np.nan],
                [5.0, np.nan, 6.0, 7.0, 8.0],
            ]
        )
        folds = MODULE.make_folds(matrix, n_folds=3, seed=42)
        held = [cell for _, cells in folds for cell in cells]
        expected = [tuple(cell) for cell in np.argwhere(np.isfinite(matrix))]
        self.assertCountEqual(held, expected)
        self.assertEqual(len(held), len(set(held)))
        for train, cells in folds:
            for row, column in cells:
                self.assertTrue(np.isnan(train[row, column]))

    def test_fold_assignment_is_deterministic(self) -> None:
        matrix = np.arange(12, dtype=float).reshape(3, 4)
        first = MODULE.make_folds(matrix, n_folds=3, seed=7)
        second = MODULE.make_folds(matrix, n_folds=3, seed=7)
        self.assertEqual([cells for _, cells in first], [cells for _, cells in second])


if __name__ == "__main__":
    unittest.main()
