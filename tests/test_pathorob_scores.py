from __future__ import annotations

import csv
import unittest
from collections import Counter
from pathlib import Path

from pathopress.matrix import filter_matrix, load_scores, make_matrix
from scripts import build_registry
from scripts.extract_pathorob_scores import paper_rows, repository_example_rows


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "source_data/pathorob_nature2026_and_repo_examples.csv"


class PathoROBScoreEvidenceTests(unittest.TestCase):
    def test_snapshot_separates_complete_paper_tables_from_examples(self) -> None:
        with SNAPSHOT.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 122)
        self.assertEqual(Counter(row["source_scope"] for row in rows), {
            "published_paper": 100,
            "repository_example": 22,
        })
        paper = [row for row in rows if row["source_scope"] == "published_paper"]
        examples = [row for row in rows if row["source_scope"] == "repository_example"]
        self.assertEqual(len({row["model_id"] for row in paper}), 20)
        self.assertEqual(len({row["model_id"] for row in examples}), 2)
        self.assertEqual(Counter(row["endpoint"] for row in paper), {
            "apd_id": 20,
            "apd_ood": 20,
            "clustering_score": 60,
        })
        self.assertEqual(
            Counter(row["inclusion_status"] for row in paper),
            {"canonical_analysis_ineligible": 40, "canonical_analysis_eligible": 60},
        )
        atlas_ood = next(
            row for row in paper
            if row["model_id"] == "atlas" and row["endpoint"] == "apd_ood"
        )
        self.assertEqual(atlas_ood["value"], "0.3309109680809159")
        phikon_cam = next(
            row for row in paper
            if row["model_id"] == "phikon-v2"
            and row["evaluation_id"] == "pathorob.nature2026.camelyon.clustering_score"
        )
        self.assertEqual(phikon_cam["value"], "-0.9912678739397616")

    def test_extractor_reproduces_snapshot_when_pinned_inputs_are_available(self) -> None:
        workbook = Path("/tmp/pathorob_source_data.xlsx")
        repository = Path("/tmp/pathopress_sources/pathorob")
        if not workbook.is_file() or not repository.is_dir():
            self.skipTest("pinned PathoROB workbook/repository are unavailable")
        generated = paper_rows(workbook) + repository_example_rows(repository)
        with SNAPSHOT.open(newline="", encoding="utf-8") as handle:
            committed = list(csv.DictReader(handle))
        self.assertEqual(generated, committed)

    def test_registry_ingests_paper_means_and_never_examples(self) -> None:
        source = Path("/tmp/pathopress_sources")
        if not (source / "pathorob").is_dir():
            self.skipTest("pinned upstream source clones are unavailable")
        tasks = [
            *build_registry.build_pathobench(
                source / "pathobench_hf",
                build_registry.git(source / "pathobench_hf", "rev-parse", "HEAD"),
            ),
            *build_registry.build_eva(
                source / "eva", build_registry.git(source / "eva", "rev-parse", "HEAD")
            ),
            *build_registry.build_thunder(
                source / "thunder", build_registry.git(source / "thunder", "rev-parse", "HEAD")
            ),
            *build_registry.build_hest(
                build_registry.git(source / "hest", "rev-parse", "HEAD")
            ),
            *build_registry.build_pathorob(
                build_registry.git(source / "pathorob", "rev-parse", "HEAD")
            ),
        ]
        build_registry.materialize_task_contracts(tasks)
        tasks.extend(build_registry.build_pathorob_nature_protocols(SNAPSHOT, tasks))
        scores, _aliases = build_registry.parse_pathorob_nature_scores(SNAPSHOT, tasks)
        self.assertEqual(len(scores), 100)
        self.assertEqual(Counter(row["audit_status"] for row in scores), {
            "parsed_primary_source": 60,
            "parsed_primary_source_analysis_ineligible": 40,
        })
        self.assertTrue(all(
            row["normalized_score"] == ""
            for row in scores
            if row["metric"] == "average_performance_drop_percent"
        ))
        self.assertTrue(all(
            0.0 <= float(row["normalized_score"]) <= 100.0
            for row in scores
            if row["metric"] == "clustering_score"
        ))
        self.assertFalse(any("repoexample" in row["evaluation_id"] for row in scores))

    def test_committed_registry_retains_only_normalizable_nature_protocols(self) -> None:
        scores = load_scores(ROOT / "data/scores.csv")
        matrix, models, evaluations = filter_matrix(*make_matrix(scores))
        nature = sorted(name for name in evaluations if name.startswith("pathorob.nature2026."))
        self.assertEqual(nature, [
            "pathorob.nature2026.camelyon.clustering_score",
            "pathorob.nature2026.tcga_4x4.clustering_score",
            "pathorob.nature2026.tolkach_esca.clustering_score",
        ])
        self.assertEqual((len(models), len(evaluations), int((matrix == matrix).sum())), (59, 168, 2027))


if __name__ == "__main__":
    unittest.main()
