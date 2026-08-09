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
    def test_checked_in_curve_schema(self) -> None:
        """Verify the dual-objective CSV has the expected schema.

        Under LOFO there is no single train/validation size, so we assert on
        structural properties rather than hardcoded model counts.
        """
        with (ROOT / "outputs" / "probe_dual_objective_rank1.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))

        # LOFO: 2 candidate modes × lofo_max_probes (5) = 10 rows
        self.assertEqual(
            len(rows), 10,
            "row count should be 2 candidate_modes × lofo_max_probes (5); "
            "if this fails the probe depth changed and the CSV needs regeneration"
        )

        # Every row must declare protocol and candidate_mode
        self.assertEqual(
            {row["protocol"] for row in rows},
            {"heldout_model"},
            "all rows should use heldout_model protocol",
        )
        self.assertEqual(
            {row["candidate_mode"] for row in rows},
            set(PLOT.MODE_LABELS),
            "should contain rows for every candidate mode",
        )

        # K values should be a contiguous range starting at 1
        k_values = {int(row["k"]) for row in rows}
        self.assertEqual(len(k_values), max(k_values))
        self.assertIn(1, k_values)

        # Each mode should have the same k range
        for mode in PLOT.MODE_LABELS:
            mode_ks = sorted(int(row["k"]) for row in rows if row["candidate_mode"] == mode)
            self.assertEqual(
                mode_ks, list(range(1, max(mode_ks) + 1)),
                f"k values for {mode} should be contiguous from 1",
            )

        # Under LOFO the train/val counts are aggregated to median values
        # across folds, so every row carries the same n_train_models and
        # n_heldout_models.  We assert plausible LOFO ranges instead of
        # blind isdigit() -- 59 total models, families range 1-7 members.
        n_train_set = {int(row["n_train_models"]) for row in rows}
        self.assertTrue(all(1 <= n <= 58 for n in n_train_set),
                        f"LOFO train counts {n_train_set} should be in [1, 58] "
                        f"(59 total models minus at least 1 held out)")
        n_heldout_set = {int(row["n_heldout_models"]) for row in rows}
        self.assertTrue(all(1 <= n <= 7 for n in n_heldout_set),
                        f"LOFO held-out counts {n_heldout_set} should be in [1, 7] "
                        f"(min/max family sizes across 34 folds)")

        # random_control_available must be False for LOFO model-average analysis
        # (the current held-out random artifact stores pooled cell metrics, not
        #  per-model predictions or model-average errors).
        self.assertEqual(
            {row["random_control_available"] for row in rows},
            {"False"},
            "random control should not be available for LOFO model-average analysis",
        )

    def test_figure_discloses_scopes(self) -> None:
        """Verify the figure annotates transductive and held-out scopes.

        Instead of hardcoding train/val counts (which change with split mode),
        assert on structural properties derived from the actual artifacts.
        """
        utility = PLOT.top_utility_rows(
            ROOT / "outputs" / "probe_informativeness_rank1.csv"
        )
        with (ROOT / "outputs" / "probe_dual_objective_rank1.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))

        # Read metadata from the compression artifact directly
        compression_path = ROOT / "experiments" / "probe_compression_rank1.json"
        compression_payload = json.loads(compression_path.read_text(encoding="utf-8"))
        matrix_shape = compression_payload["configuration"]["matrix_shape"]

        # For LOFO, n_train / n_validation will reflect fold-level sizes
        split = compression_payload.get("split", {})
        split_mode = split.get("split_mode", "unknown")

        if split_mode == "leave_one_family_out":
            # LOFO stores per_fold entries with n_train_models / n_validation_models.
            # We use median fold sizes (same aggregation as build_heldout_mean_records).
            def _median(lst):
                s = sorted(lst)
                n = len(s)
                return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
            per_fold = split.get("per_fold", [])
            n_train = int(_median(
                [f["n_train_models"] for f in per_fold]
            )) if per_fold else 0
            n_validation = int(_median(
                [f["n_validation_models"] for f in per_fold]
            )) if per_fold else 0
            metadata = {
                "n_train": n_train,
                "n_validation": n_validation,
                "matrix_shape": matrix_shape,
            }
        else:
            # For single-split modes, use the train/validation counts from rows
            n_train = int(rows[0]["n_train_models"])
            n_validation = int(rows[0]["n_heldout_models"])
            metadata = {
                "n_train": n_train,
                "n_validation": n_validation,
                "matrix_shape": matrix_shape,
            }

        figure, utility_ax, prediction_ax = PLOT.build_figure(
            utility,
            rows,
            metadata,
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
        self.assertIn("No k=0/random model-mean controls", text)
        self.assertIn("not measured cost", text)

        # The top-8 utility probes should all be from THUNDER in the current
        # artifact, so the coverage note must say "all 8: THUNDER".
        self.assertIn("all 8: THUNDER", text)

        # LOFO caption sanity: the "Nested prefixes" line should mention a
        # non-zero number of selection models (59 total models, at least one
        # held out per fold).  The exact value is metadata-driven but must
        # reflect the 59-model pool, not stale 41/18 splits.
        self.assertIn("selection models", text,
                       "figure must disclose selection model count in caption")
        self.assertIn("held-out models", text,
                       "figure must disclose held-out model count in caption")
        self.assertNotIn("0 selection models", text,
                         "LOFO caption should show non-zero selection model count; "
                         "if metadata is empty the figure is silently wrong")

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
