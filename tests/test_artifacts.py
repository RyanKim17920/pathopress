import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pathopress.artifacts import (
    build_within_model_folds,
    load_fold_artifact,
    write_fold_artifact,
)


class SharedArtifactTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.arange(24, dtype=float).reshape(4, 6)
        self.models = [f"m{i}" for i in range(4)]
        self.evaluations = [f"e{j}" for j in range(6)]

    def test_each_seed_partitions_every_observed_cell_once(self):
        records = build_within_model_folds(self.matrix, n_seeds=2, n_folds=3, base_seed=42)
        for seed in (42, 43):
            cells = [tuple(cell) for row in records if row["seed"] == seed for cell in row["test_cells"]]
            self.assertEqual(len(cells), self.matrix.size)
            self.assertEqual(len(set(cells)), self.matrix.size)

    def test_artifact_validates_matrix_identity_and_materializes_train(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.json"
            write_fold_artifact(path, self.matrix, self.models, self.evaluations, n_seeds=1)
            folds = load_fold_artifact(path, self.matrix, self.models, self.evaluations)
            self.assertEqual(len(folds), 3)
            for _, _, train, cells in folds:
                self.assertEqual(int(np.isnan(train).sum()), len(cells))
            payload = json.loads(path.read_text())
            payload["configuration"]["models"][0] = "wrong"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_fold_artifact(path, self.matrix, self.models, self.evaluations)
