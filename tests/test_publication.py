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
                "exhaustive_25C5_30C5": "not_claimed_in_hero",
            },
        )
        self.assertEqual(benchpress_hero["schema_version"], 2)
        self.assertEqual(benchpress_hero["source_shape"], [59, 187])
        self.assertEqual(benchpress_hero["n_observed"], 2_122)
        self.assertFalse(benchpress_hero["semantics"]["model_level_holdout"])
        self.assertFalse(
            benchpress_hero["semantics"]["low_friction_proxy_is_measured_cost"]
        )
        self.assertEqual(benchpress_hero["semantics"]["outcome_selected_examples"], "omitted")
        with (ROOT / "outputs/probe_dual_objective_rank1.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            dual = list(csv.DictReader(handle))
        self.assertEqual(len(dual), 20)
        objectives = {row["selection_objective"] for row in dual}
        self.assertEqual(len(objectives), 1)
        self.assertIn(objectives.pop(), {"scorecard_medae", "training_scorecard_medae"})


if __name__ == "__main__":
    unittest.main()
