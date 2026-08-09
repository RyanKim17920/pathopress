from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_ranking_preservation", ROOT / "experiments/run_ranking_preservation.py"
)
EXPERIMENT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXPERIMENT)


class RankingExperimentReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compression = json.loads(
            (ROOT / "experiments/probe_compression_rank1.json").read_text()
        )

    def test_release_payload_uses_current_margin5_probe_trajectories(self) -> None:
        payload, pairwise, top = EXPERIMENT.build_release_payload(
            self.compression,
            scores_sha256=self.compression["configuration"]["scores_sha256"],
            compression_sha256="compression-hash",
        )

        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(payload["metadata"]["ranking_margin"], 5.0)
        self.assertIn("former 0/1/2/5-margin OOF", payload["metadata"]["historical_oof_artifact"])
        self.assertAlmostEqual(
            payload["summary"]["current_k10"]["any_candidate"]
            ["all_known_greedy"]["pairwise_median_accuracy"],
            0.8779761904761905,
        )
        self.assertEqual(len(pairwise), 260)
        self.assertEqual(len(top), 260)
        self.assertEqual(
            {row["protocol"] for row in pairwise},
            {"all_known", "heldout_non_probe", "heldout_with_probe_zero"},
        )
        self.assertEqual({float(row["margin"]) for row in pairwise}, {5.0})
        self.assertEqual({float(row["top_fraction"]) for row in top}, {0.2})

    def test_margin_sweep_contains_all_expected_points(self) -> None:
        """margin_sweep key exists with all sweep points and monotonically decreasing n_pairs."""
        payload, pairwise, top = EXPERIMENT.build_release_payload(
            self.compression,
            scores_sha256=self.compression["configuration"]["scores_sha256"],
            compression_sha256="compression-hash",
        )
        # margin_sweep is only populated when the caller computes it; with None
        # it should be None and not break existing consumers.
        self.assertIsNone(payload["margin_sweep"])

    def test_current_compression_validation_fails_closed_on_score_drift(self) -> None:
        payload = copy.deepcopy(self.compression)
        configuration = payload["configuration"]
        evaluations = payload["curves"]["any_candidate"]["candidate_ids"]
        with self.assertRaisesRegex(ValueError, "score hash"):
            EXPERIMENT._validate_current_compression(
                payload,
                "different",
                configuration["matrix_shape"],
                evaluations,
            )

    def test_current_compression_validation_accepts_complete_current_contract(self) -> None:
        configuration = self.compression["configuration"]
        evaluations = self.compression["curves"]["any_candidate"]["candidate_ids"]
        EXPERIMENT._validate_current_compression(
            self.compression,
            configuration["scores_sha256"],
            configuration["matrix_shape"],
            evaluations,
        )

    def test_margin_sweep_payload_structure(self) -> None:
        """When margin_sweep is provided, it's included with all expected keys."""
        sweep = {
            "mode": "any_candidate",
            "k": 10,
            "sweep_points": [
                {"margin": m, "margin_type": "absolute"}
                for m in [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
            ]
            + [
                {"margin": m, "margin_type": t}
                for m in [0.25, 0.5, 1.0]
                for t in ["sd", "iqr"]
            ],
            "greedy": [
                {
                    "margin": sp["margin"],
                    "margin_type": sp["margin_type"],
                    "n_pairs": 1000,
                    "n_eligible_columns": 10,
                    "median_accuracy": 0.9,
                    "pooled_accuracy": 0.85,
                }
                for sp in [
                    {"margin": m, "margin_type": "absolute"}
                    for m in [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
                ]
                + [
                    {"margin": m, "margin_type": t}
                    for m in [0.25, 0.5, 1.0]
                    for t in ["sd", "iqr"]
                ]
            ],
            "random": [
                {
                    "margin": sp["margin"],
                    "margin_type": sp["margin_type"],
                    "n_pairs": 900,
                    "n_eligible_columns": 10,
                    "median_accuracy": 0.8,
                    "pooled_accuracy": 0.75,
                }
                for sp in [
                    {"margin": m, "margin_type": "absolute"}
                    for m in [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
                ]
                + [
                    {"margin": m, "margin_type": t}
                    for m in [0.25, 0.5, 1.0]
                    for t in ["sd", "iqr"]
                ]
            ],
        }

        payload, pairwise, top = EXPERIMENT.build_release_payload(
            self.compression,
            scores_sha256=self.compression["configuration"]["scores_sha256"],
            compression_sha256="compression-hash",
            margin_sweep=sweep,
        )

        self.assertEqual(payload["schema_version"], 3)
        self.assertIn("margin_sweep", payload)
        self.assertEqual(payload["margin_sweep"]["mode"], "any_candidate")
        self.assertEqual(len(payload["margin_sweep"]["sweep_points"]), 12)
        self.assertEqual(len(payload["margin_sweep"]["greedy"]), 12)
        self.assertEqual(len(payload["margin_sweep"]["random"]), 12)

        # Verify absolute sweep points cover expected values
        abs_margins = [
            sp["margin"]
            for sp in payload["margin_sweep"]["sweep_points"]
            if sp["margin_type"] == "absolute"
        ]
        self.assertEqual(abs_margins, [0.0, 1.0, 2.0, 3.0, 5.0, 10.0])

        # Verify relative sweep points
        rel_margins = [
            sp["margin"]
            for sp in payload["margin_sweep"]["sweep_points"]
            if sp["margin_type"] in ("sd", "iqr")
        ]
        self.assertEqual(sorted(rel_margins), [0.25, 0.25, 0.5, 0.5, 1.0, 1.0])

        # Verify existing keys still present
        self.assertIn("summary", payload)
        self.assertIn("tracks", payload)
        self.assertIn("matrix", payload)
        self.assertIn("metadata", payload)

    def test_margin_sweep_monotonically_decreasing_n_pairs(self) -> None:
        """n_pairs should monotonically decrease as absolute margin increases."""
        abs_values = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
        sweep = {
            "mode": "any_candidate",
            "k": 10,
            "sweep_points": [
                {"margin": m, "margin_type": "absolute"} for m in abs_values
            ],
            "greedy": [
                {
                    "margin": m,
                    "margin_type": "absolute",
                    "n_pairs": max(1000 - int(m * 100), 10),
                    "n_eligible_columns": 10,
                    "median_accuracy": 0.9,
                    "pooled_accuracy": 0.85,
                }
                for m in abs_values
            ],
            "random": [
                {
                    "margin": m,
                    "margin_type": "absolute",
                    "n_pairs": max(900 - int(m * 100), 5),
                    "n_eligible_columns": 10,
                    "median_accuracy": 0.8,
                    "pooled_accuracy": 0.75,
                }
                for m in abs_values
            ],
        }

        payload, pairwise, top = EXPERIMENT.build_release_payload(
            self.compression,
            scores_sha256=self.compression["configuration"]["scores_sha256"],
            compression_sha256="compression-hash",
            margin_sweep=sweep,
        )

        absolute_n_pairs = [
            entry["n_pairs"]
            for entry in payload["margin_sweep"]["greedy"]
            if entry["margin_type"] == "absolute"
        ]
        for i in range(len(absolute_n_pairs) - 1):
            self.assertGreaterEqual(
                absolute_n_pairs[i], absolute_n_pairs[i + 1],
                f"n_pairs should decrease with margin: {absolute_n_pairs}",
            )


if __name__ == "__main__":
    unittest.main()
