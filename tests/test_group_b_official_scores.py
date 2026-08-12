from __future__ import annotations

import csv
import unittest
from collections import Counter
from pathlib import Path

from scripts.extract_group_b_official_scores import (
    genbio_rows,
    midnight_rows,
    openmidnight_rows,
)
from tests.pinned_sources import missing_inputs, sources_root


ROOT = Path(__file__).resolve().parents[1]
GENBIO = ROOT / "source_data/genbio_pathfm_official_2026.csv"
MIDNIGHT = ROOT / "source_data/midnight_miccai2025_official_scores.csv"
OPENMIDNIGHT = ROOT / "source_data/openmidnight_technical_report_2025.csv"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class GroupBOfficialScoreEvidenceTests(unittest.TestCase):
    def test_snapshots_have_expected_scope_and_quarantine_counts(self) -> None:
        genbio = read(GENBIO)
        midnight = read(MIDNIGHT)
        openmidnight = read(OPENMIDNIGHT)
        self.assertEqual(len(genbio), 49)
        self.assertEqual(len(midnight), 42)
        self.assertEqual(len(openmidnight), 16)
        self.assertEqual(Counter(row["inclusion_status"] for row in genbio), {
            "canonical_candidate": 15,
            "canonical_candidate_analysis_ineligible": 6,
            "duplicate_alternate_evidence": 12,
            "fine_tuned_excluded": 10,
            "aggregate_excluded": 6,
        })
        self.assertEqual(Counter(row["inclusion_status"] for row in midnight), {
            "canonical_candidate": 24,
            "fine_tuned_excluded": 12,
            "aggregate_excluded": 6,
        })
        self.assertEqual(Counter(row["inclusion_status"] for row in openmidnight), {
            "canonical_candidate": 12,
            "aggregate_excluded": 2,
            "narrative_conflict_excluded": 2,
        })

    def test_protocol_variants_are_not_collapsed(self) -> None:
        genbio = read(GENBIO)
        midnight = read(MIDNIGHT)
        openmidnight = read(OPENMIDNIGHT)
        thunder = [
            row for row in genbio
            if row["suite_id"] == "thunder" and row["inclusion_status"] == "canonical_candidate"
        ]
        self.assertEqual(len(thunder), 12)
        self.assertTrue(all(row["evaluation_id"].endswith(".knn") for row in thunder))
        self.assertTrue(all(row["metric"] == "f1_score" for row in thunder))

        twelve = [
            row for row in midnight
            if row["model_id"] == "midnight" and row["inclusion_status"] == "canonical_candidate"
        ]
        high_resolution = [row for row in midnight if row["model_id"] == "midnight-92k-392"]
        self.assertEqual(len(twelve), 12)
        self.assertTrue(all("clsmean_224" in row["evaluation_id"] for row in twelve))
        self.assertTrue(all(
            row["inclusion_status"] in {"fine_tuned_excluded", "aggregate_excluded"}
            for row in high_resolution
        ))

        own = [row for row in openmidnight if row["inclusion_status"] == "canonical_candidate"]
        self.assertEqual(len(own), 12)
        self.assertTrue(all(".cls." in row["evaluation_id"] for row in own))
        self.assertTrue(all("supporting config snapshot" in row["protocol_variant"] for row in own))

    def test_exact_primary_values_and_known_source_conflicts_are_preserved(self) -> None:
        genbio = read(GENBIO)
        midnight = read(MIDNIGHT)
        openmidnight = read(OPENMIDNIGHT)
        genbio_bach = next(
            row for row in genbio
            if row["evaluation_id"] == "thunder.genbio2026.bach.knn"
        )
        midnight_bach = next(
            row for row in midnight
            if row["model_id"] == "midnight" and row["task_label"] == "BACH"
        )
        open_breakhis = next(
            row for row in openmidnight
            if row["task_label"] == "BreakHis" and row["inclusion_status"] == "canonical_candidate"
        )
        open_cam16 = next(
            row for row in openmidnight
            if row["task_label"] == "Cam16 (small)" and row["inclusion_status"] == "canonical_candidate"
        )
        conflicts = [
            row for row in openmidnight
            if row["inclusion_status"] == "narrative_conflict_excluded"
        ]
        self.assertEqual(genbio_bach["value"], "81.8")
        self.assertEqual(midnight_bach["value"], "0.907")
        self.assertEqual(open_breakhis["value"], "0.873")
        self.assertEqual(open_cam16["value"], "0.946")
        self.assertEqual({(row["task_label"], row["value"]) for row in conflicts}, {
            ("BreakHis", "0.946"),
            ("Cam16 (small)", "0.873"),
        })

    def test_selected_candidates_are_unique_by_checkpoint_and_protocol(self) -> None:
        rows = [*read(GENBIO), *read(MIDNIGHT), *read(OPENMIDNIGHT)]
        selected = [
            row for row in rows
            if row["inclusion_status"] in {
                "canonical_candidate",
                "canonical_candidate_analysis_ineligible",
            }
        ]
        keys = [(row["model_id"], row["evaluation_id"]) for row in selected]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertFalse(any(row["task_label"] == "Average" for row in selected))

    def test_extractor_reproduces_snapshots_when_primary_inputs_are_available(self) -> None:
        genbio_pdf = Path("/tmp/genbio-pathfm.pdf")
        midnight_pdf = Path("/tmp/midnight-4651.pdf")
        openmidnight_html = Path("/tmp/openmidnight_report.html")
        openmidnight_repo = sources_root() / "eva_openmidnight"
        if not openmidnight_repo.is_dir():
            missing_inputs(
                self,
                "the pinned OpenMidnight checkout is unavailable "
                "(scripts/fetch_sources.py --include-extractor-sources)",
            )
        if not all(p.exists() for p in (genbio_pdf, midnight_pdf, openmidnight_html)):
            # data/provenance.json records no retrieval URL for these three
            # publisher artifacts, so nothing in this repository can fetch them.
            missing_inputs(
                self,
                "pinned Group B publisher artifacts are unavailable",
                fetchable=False,
            )
        self.assertEqual(genbio_rows(genbio_pdf), read(GENBIO))
        self.assertEqual(midnight_rows(midnight_pdf), read(MIDNIGHT))
        self.assertEqual(openmidnight_rows(openmidnight_html, openmidnight_repo), read(OPENMIDNIGHT))


if __name__ == "__main__":
    unittest.main()
