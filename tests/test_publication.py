from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from pathopress.matrix import filter_matrix, load_scores, make_matrix
from pathopress.publication import (
    hero_target_cells,
    metadata_panel_counts,
    read_csv,
    select_hero_target,
    score_source_group,
    top_with_other,
)


ROOT = Path(__file__).resolve().parents[1]


class PublicationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        scores = load_scores(ROOT / "data/scores.csv")
        cls.matrix, cls.models, cls.evaluations = filter_matrix(*make_matrix(scores))

    def test_hero_target_and_keep_k_cells_are_data_determined_and_aligned(self) -> None:
        raw = read_csv(ROOT / "outputs/probe_compression_selected_raw_rank1.csv")
        target = select_hero_target(raw)
        self.assertEqual(target, "h-optimus-0")
        cells = hero_target_cells(raw, target)
        self.assertEqual({k: len(rows) for k, rows in cells.items()}, {1: 145, 3: 145, 10: 145})
        identities = [{row["evaluation_id"] for row in rows} for rows in cells.values()]
        self.assertTrue(all(value == identities[0] for value in identities[1:]))
        self.assertEqual(
            {k: sum(row["is_revealed_probe_cell"] == "True" for row in rows) for k, rows in cells.items()},
            {1: 1, 3: 2, 10: 9},
        )

    def test_metadata_panel_denominators_match_retained_matrix(self) -> None:
        counts = metadata_panel_counts(
            read_csv(ROOT / "data/scores.csv"),
            read_csv(ROOT / "data/tasks.csv"),
            read_csv(ROOT / "data/model_release_dates.csv"),
            set(self.evaluations),
            set(self.models),
        )
        self.assertEqual(counts["n_models"], self.matrix.shape[0])
        self.assertEqual(counts["n_evaluations"], self.matrix.shape[1])
        self.assertEqual(counts["n_observed"], int((self.matrix == self.matrix).sum()))
        self.assertEqual(sum(counts["task_family"].values()), self.matrix.shape[1])
        self.assertEqual(sum(counts["observed_family"].values()), counts["n_observed"])
        self.assertEqual(sum(counts["suite_tasks"].values()), self.matrix.shape[1])
        self.assertEqual(sum(counts["source_provenance"].values()), counts["n_observed"])
        self.assertEqual(
            sum(row["n_models"] for row in counts["coverage_by_release_quarter"].values()),
            counts["n_release_dates"],
        )

    def test_score_source_groups_are_pathology_explicit(self) -> None:
        self.assertEqual(
            score_source_group("https://github.com/example/benchmark"),
            "Official benchmark repository",
        )
        self.assertEqual(
            score_source_group("https://arxiv.org/pdf/1234.5678"),
            "Primary paper or report",
        )

    def test_top_with_other_preserves_exact_total(self) -> None:
        labels, values = top_with_other({"a": 5, "b": 4, "c": 3}, 2)
        self.assertEqual(labels, ["a", "b", "other"])
        self.assertEqual(values, [5, 4, 3])

    def test_generated_publication_summaries_and_inventories_are_consistent(self) -> None:
        hero = json.loads((ROOT / "experiments/publication_hero_summary.json").read_text())
        benchpress_hero = json.loads(
            (ROOT / "experiments/benchpress_style_hero_summary.json").read_text()
        )
        metadata = json.loads((ROOT / "experiments/publication_metadata_summary.json").read_text())
        manifest = json.loads((ROOT / "outputs/tables/manifest.json").read_text())
        self.assertEqual(hero["unrestricted_curve_lengths"], {"medae": 10, "medape": 10})
        self.assertEqual(hero["allowlist_curve_lengths"], {"medae": 10, "medape": 10})
        self.assertNotIn("legacy", json.dumps(hero).lower())
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
        self.assertEqual(metadata["matrix"], {"n_models": 59, "n_evaluations": 168, "n_observed": 2027})
        self.assertEqual(manifest["tables"]["model_inventory"], 59)
        self.assertEqual(manifest["tables"]["evaluation_inventory"], 168)
        for name, expected, identity in (
            ("model_inventory", 59, "model_id"),
            ("evaluation_inventory", 168, "evaluation_id"),
        ):
            with (ROOT / f"outputs/tables/{name}.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), expected)
            self.assertEqual(len({row[identity] for row in rows}), expected)


if __name__ == "__main__":
    unittest.main()
