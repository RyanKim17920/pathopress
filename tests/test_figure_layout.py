from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
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
PROBE_PLOT = load_script("plot_probe_selection.py")
HERO_PLOT = load_script("plot_benchpress_style_hero.py")


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

    def test_informativeness_coverage_is_in_a_separate_noncolliding_axis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "informativeness.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "evaluation_id",
                        "suite_id",
                        "improvement_over_column_median",
                        "model_coverage",
                    ),
                )
                writer.writeheader()
                for index in range(15):
                    writer.writerow(
                        {
                            "evaluation_id": f"hest.very_long_evaluation_name_{index}.gene_expression",
                            "suite_id": "hest",
                            "improvement_over_column_median": -0.08 + index * 0.02,
                            "model_coverage": 0.47 + (index % 2) * 0.07,
                        }
                    )

            fig, score_ax, coverage_ax = PROBE_PLOT.build_informativeness_figure(csv_path)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            score_labels = [label.get_window_extent(renderer) for label in score_ax.get_yticklabels()]
            coverage_labels = [text.get_window_extent(renderer) for text in coverage_ax.texts]

            self.assertEqual(len(score_labels), 15)
            self.assertEqual(len(coverage_labels), 15)
            self.assertTrue(all(not left.overlaps(right) for left in score_labels for right in coverage_labels))
            for labels in (score_labels, coverage_labels):
                self.assertTrue(
                    all(
                        not labels[first].overlaps(labels[second])
                        for first in range(len(labels))
                        for second in range(first + 1, len(labels))
                    )
                )
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
