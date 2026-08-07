from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    def test_hero_worker_and_blas_limits_are_hard_capped(self) -> None:
        with patch.object(sys, "argv", ["plot_benchpress_style_hero.py"]):
            self.assertLessEqual(HERO_PLOT.parse_args().workers, 4)
        with patch.object(
            sys,
            "argv",
            ["plot_benchpress_style_hero.py", "--workers", "5"],
        ), self.assertRaises(SystemExit):
            HERO_PLOT.parse_args()
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            self.assertEqual(os.environ[variable], "1")

    def test_hero_probe_labels_are_suite_aware_and_collision_free(self) -> None:
        rows = [
            {"added_evaluation_id": "thunder.spider_skin.linear_probing"},
            {"added_evaluation_id": "thunder.esca.linear_probing"},
            {"added_evaluation_id": "eva.leaderboard.patch_camelyon.test"},
            {"added_evaluation_id": "eva.leaderboard.breakhis.validation"},
            {"added_evaluation_id": "hest.read.gene_expression"},
            {"added_evaluation_id": "pathobench.threads2025.cptac_hnsc.casp8-mutation"},
        ]
        labels = HERO_PLOT._trajectory_labels(rows)
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(labels[0].startswith("THU SPIDER SKIN"))
        self.assertTrue(labels[2].startswith("EVA PCam test"))
        self.assertTrue(labels[4].startswith("HEST READ"))
        self.assertTrue(labels[5].startswith("THR HNSC CASP8"))

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
