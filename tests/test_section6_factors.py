from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.model_metadata import (  # noqa: E402
    build_model_metadata,
    load_model_metadata,
    write_model_metadata,
)
from pathopress.section6_factors import (  # noqa: E402
    grouped_wilcoxon,
    holdout_half_per_benchmark,
    paired_error_record,
    supported_complete,
)
from pathopress.provenance import (  # noqa: E402
    BENCHPRESS_PINNED_COMMIT,
    BENCHPRESS_REPOSITORY,
    benchpress_tree_url,
    validate_benchpress_pin,
)


class Section6FactorTests(unittest.TestCase):
    def test_research_extra_declares_all_experiment_dependencies(self):
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
        extras = pyproject["project"]["optional-dependencies"]
        declared = {
            requirement.split(">=", 1)[0].lower()
            for requirement in extras["research"]
        }
        self.assertEqual(
            declared,
            {"matplotlib", "scipy", "scikit-learn", "torch"},
        )
        self.assertIn(extras["plots"][0], extras["research"])
        self.assertIn(extras["confidence"][0], extras["research"])

    def test_prediction_error_factor_pin_matches_canonical_full_commit(self):
        validate_benchpress_pin()
        self.assertRegex(BENCHPRESS_PINNED_COMMIT, r"^[0-9a-f]{40}$")
        self.assertEqual(
            benchpress_tree_url(),
            f"{BENCHPRESS_REPOSITORY}/tree/{BENCHPRESS_PINNED_COMMIT}",
        )
        artifact = json.loads(
            (ROOT / "experiments/prediction_error_factors_rank1.json").read_text()
        )
        upstream = artifact["protocol"]["upstream_reference"]
        self.assertEqual(upstream["repository"], BENCHPRESS_REPOSITORY)
        self.assertEqual(upstream["pinned_commit"], BENCHPRESS_PINNED_COMMIT)
        provenance = json.loads((ROOT / "data/provenance.json").read_text())
        configured = provenance["repositories"]["benchpress"]
        self.assertEqual(configured["url"], BENCHPRESS_REPOSITORY)
        self.assertEqual(configured["commit"], BENCHPRESS_PINNED_COMMIT)

    def test_benchmark_hide_half_matches_floor_rule(self):
        matrix = np.arange(18, dtype=float).reshape(6, 3)
        test, train = holdout_half_per_benchmark(
            matrix, 1, np.random.RandomState(42), min_test=3
        )
        self.assertEqual(len(test), 3)
        self.assertEqual(len(train), 3)
        self.assertEqual(set(test) | set(train), set(range(6)))

    def test_paired_metrics_use_common_finite_predictions(self):
        result = paired_error_record(
            [50.0, 60.0, 70.0],
            [49.0, 59.0, np.nan],
            [48.0, np.nan, 68.0],
            min_predictions=1,
        )
        self.assertEqual(result["n_test"], 1)
        self.assertEqual(result["base_medae"], 1.0)
        self.assertEqual(result["treat_medae"], 2.0)

    def test_grouped_wilcoxon_uses_one_seed_median_per_target(self):
        records = [
            {"target": "a", "delta_medae": 1.0, "delta_medape": 2.0},
            {"target": "a", "delta_medae": 3.0, "delta_medape": 4.0},
            {"target": "b", "delta_medae": -2.0, "delta_medape": -4.0},
            {"target": "b", "delta_medae": -4.0, "delta_medape": -6.0},
        ]
        result = grouped_wilcoxon(records, group_key="target")
        self.assertEqual(result["medae"]["n"], 2)
        self.assertAlmostEqual(result["medae"]["median_delta"], -0.5)

    def test_supported_completion_retains_unsupported_columns_as_missing(self):
        matrix = np.asarray([[60.0, np.nan, 70.0], [65.0, np.nan, 75.0]])
        predicted = supported_complete(matrix, rank=1)
        self.assertTrue(np.isnan(predicted[:, 1]).all())
        np.testing.assert_allclose(predicted[:, [0, 2]], matrix[:, [0, 2]])

    def test_model_metadata_keeps_unavailable_parameter_counts_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            sources = directory / "sources.csv"
            releases = directory / "releases.csv"
            with sources.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "model_id", "canonical_family", "primary_paper_url",
                    "primary_paper_title",
                ])
                writer.writeheader()
                writer.writerow({
                    "model_id": "uni", "canonical_family": "UNI",
                    "primary_paper_url": "https://example.org/uni",
                    "primary_paper_title": "UNI",
                })
            with releases.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "model_id", "release_date", "verification_status",
                    "primary_source_url", "source_title",
                ])
                writer.writeheader()
                writer.writerow({
                    "model_id": "uni", "release_date": "2024-01-01",
                    "verification_status": "verified",
                    "primary_source_url": "https://example.org/release",
                    "source_title": "UNI release",
                })
                writer.writerow({
                    "model_id": "unknown-slide", "release_date": "2025-01-01",
                    "verification_status": "verified",
                    "primary_source_url": "https://example.org/unknown",
                    "source_title": "Unknown",
                })
            rows = build_model_metadata(
                ["uni", "unknown-slide"],
                model_sources_path=sources,
                release_dates_path=releases,
            )
            self.assertEqual(rows[0]["parameter_count"], 304_000_000)
            self.assertEqual(rows[1]["parameter_count"], "")
            output = directory / "metadata.csv"
            write_model_metadata(output, rows)
            loaded = load_model_metadata(output)
            self.assertEqual(loaded["unknown-slide"]["parameter_basis"], "missing")

    def test_confidence_reliability_plot_uses_percentile_not_risk_magnitude(self):
        spec = importlib.util.spec_from_file_location(
            "confidence_plot", ROOT / "scripts" / "plot_confidence_calibration.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        percentile, observed = module._quantile_calibration(
            np.arange(10, dtype=float), np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 1000.0]),
            bins=5,
        )
        np.testing.assert_allclose(percentile, [10, 30, 50, 70, 90])
        self.assertEqual(observed[-1], 8.5)
        self.assertAlmostEqual(
            module._inverse_risk_maximum(np.asarray([0.0, np.log1p(3.0)])),
            3.0,
        )

    def test_model_factor_stat_annotation_has_one_unambiguous_p_value(self):
        spec = importlib.util.spec_from_file_location(
            "factor_plot", ROOT / "scripts" / "plot_prediction_error_factors.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        label = module._stat_annotation({"rho": 0.48, "p": 0.00252, "n": 53})
        self.assertEqual(label, "$\\rho$=+0.48, n=53\np=0.00252")
        self.assertEqual(label.count("p="), 1)


if __name__ == "__main__":
    unittest.main()
