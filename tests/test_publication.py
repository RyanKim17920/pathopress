from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PublicationDataTests(unittest.TestCase):
    def test_generated_benchpress_figures_are_consistent(self) -> None:
        benchpress_hero = json.loads(
            (ROOT / "experiments/benchpress_style_hero_summary.json").read_text()
        )
        self.assertEqual(
            benchpress_hero["contract_status"],
            {
                "masking_and_k_budget": "exact",
                "rank_and_domain": "pathology_adapted",
                "exhaustive_25C5_30C5": "not_run_for_current_scores",
            },
        )
        self.assertEqual(benchpress_hero["source_shape"], [59, 187])
        self.assertIsNone(benchpress_hero["inputs"]["exhaustive_sha256"])
        self.assertEqual(benchpress_hero["exact_results"], {})
        self.assertEqual(len(benchpress_hero["examples"]), 4)
        with (ROOT / "outputs/probe_dual_objective_rank1.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            dual = list(csv.DictReader(handle))
        self.assertEqual(len(dual), 20)
        self.assertTrue(all(row["selection_objective"] == "scorecard_medae" for row in dual))


if __name__ == "__main__":
    unittest.main()
