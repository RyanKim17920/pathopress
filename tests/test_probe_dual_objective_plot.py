from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def load_plot_module():
    path = ROOT / "scripts" / "plot_probe_dual_objective.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLOT = load_plot_module()


class ProbeDualObjectivePlotTests(unittest.TestCase):
    def test_checked_in_curve_is_heldout_only(self) -> None:
        with (ROOT / "outputs" / "probe_dual_objective_rank1.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 20)
        self.assertEqual({row["protocol"] for row in rows}, {"heldout_model"})
        self.assertEqual({int(row["n_train_models"]) for row in rows}, {41})
        self.assertEqual({int(row["n_heldout_models"]) for row in rows}, {18})
        self.assertEqual({int(row["k"]) for row in rows}, set(range(1, 11)))
        self.assertEqual(
            {row["candidate_mode"] for row in rows},
            set(PLOT.MODE_LABELS),
        )
        self.assertEqual({row["random_control_available"] for row in rows}, {"False"})

    def test_figure_discloses_transductive_and_heldout_scopes(self) -> None:
        utility = PLOT.top_utility_rows(
            ROOT / "outputs" / "probe_informativeness_rank1.csv"
        )
        with (ROOT / "outputs" / "probe_dual_objective_rank1.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        figure, utility_ax, prediction_ax = PLOT.build_figure(
            utility,
            rows,
            {"n_train": 41, "n_validation": 18, "matrix_shape": [59, 187]},
        )
        figure.canvas.draw()

        self.assertEqual(figure.axes, [utility_ax, prediction_ax])
        self.assertIn("Retrospective", utility_ax.get_title(loc="left"))
        self.assertIn("mean reported score", prediction_ax.get_title(loc="left"))
        text = " ".join(
            item.get_text()
            for item in [*figure.texts, *utility_ax.texts, *prediction_ax.texts]
        )
        self.assertIn("Transductive", text)
        self.assertIn("41 selection models", text)
        self.assertIn("k=0 and random model-mean controls unavailable", text)
        plt.close(figure)

    def test_fails_closed_without_training_only_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compression = root / "compression.json"
            raw = root / "raw.csv"
            compression.write_text(
                json.dumps({"configuration": {"heldout_semantics": "all-known"}}),
                encoding="utf-8",
            )
            raw.write_text("protocol\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training-only"):
                PLOT.build_heldout_mean_records(compression, raw)


if __name__ == "__main__":
    unittest.main()
