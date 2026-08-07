from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATRIX_PLOT = load_script("plot_benchpress_style.py")
HERO_PLOT = load_script("plot_benchpress_style_hero.py")
TEMPORAL_PLOT = load_script("plot_temporal_deployment.py")


class PublicationFigureLayoutTests(unittest.TestCase):
    def test_hero_is_one_result_first_axis_with_honest_scope_labels(self) -> None:
        compression = json.loads(
            (ROOT / "experiments" / "probe_compression_rank1.json").read_text()
        )
        selection = json.loads(
            (ROOT / "experiments" / "probe_selection_results_rank1.json").read_text()
        )
        values = HERO_PLOT.hero_plot_data(compression, selection)
        self.assertTrue(values["proxy_supported"])
        self.assertEqual(values["source_shape"], [59, 187])
        self.assertEqual(values["n_observed"], 2_122)

        fig, ax = HERO_PLOT.build_hero_figure(values)
        self.assertEqual(fig.axes, [ax])
        self.assertEqual(ax.get_title(), "Retrospective all-known matrix reconstruction")
        disclosure = " ".join(text.get_text() for text in fig.texts)
        self.assertIn("Revealed probes are scored as exact", disclosure)
        self.assertIn("not model-level holdout", disclosure)
        self.assertIn("not measured cost", disclosure)
        self.assertNotIn("ProcessPoolExecutor", (ROOT / "scripts/plot_benchpress_style_hero.py").read_text())
        self.assertFalse(ax.texts)
        plt.close(fig)

    def test_cell_validation_uses_one_axis_and_discloses_repeated_predictions(self) -> None:
        result = json.loads(
            (ROOT / "experiments" / "benchpress_style_results.json").read_text()
        )
        values = MATRIX_PLOT.validation_plot_data(result)
        self.assertEqual(values["selected_rank"], 1)
        self.assertEqual(values["unique_cells"], 2_122)
        self.assertEqual(values["prediction_instances"], 21_181)

        fig, ax = MATRIX_PLOT.build_rank_selection_figure(result)
        self.assertEqual(fig.axes, [ax])
        disclosure = " ".join(text.get_text() for text in fig.texts)
        self.assertIn("2,122 unique reported cells", disclosure)
        self.assertIn("21,181 repeated held-out predictions", disclosure)
        self.assertIn("Cell-level validation", disclosure)
        self.assertIn("selects interaction rank 1", ax.get_title())
        plt.close(fig)

    def test_temporal_figure_is_one_panel_without_obscuring_cohort_overlay(self) -> None:
        payload = json.loads(
            (ROOT / "experiments" / "temporal_deployment_rank1.json").read_text()
        )
        fig, ax = TEMPORAL_PLOT.build_temporal_figure(payload)
        self.assertEqual(fig.axes, [ax])
        self.assertIsNone(ax.get_legend())
        self.assertIn("Parity/reconstruction MedAE", ax.get_ylabel())
        self.assertIn("exact revealed cells", ax.get_ylabel())
        self.assertEqual(
            {text.get_text() for text in ax.texts},
            set(payload["config"]["target_model_ids"]),
        )
        disclosure = " ".join(text.get_text() for text in fig.texts)
        self.assertIn("ten probe seeds", disclosure)
        self.assertIn("supported hidden predictions", disclosure)
        plt.close(fig)

if __name__ == "__main__":
    unittest.main()
