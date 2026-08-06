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

        self.assertEqual(payload["schema_version"], 2)
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


if __name__ == "__main__":
    unittest.main()
