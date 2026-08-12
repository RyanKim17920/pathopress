from __future__ import annotations

import csv
import math
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

    # Tight enough that any change with scientific meaning still fails (it is
    # ~4 orders of magnitude above the ~1e-16 double-precision ULP and ~10
    # orders below the smallest difference that could move a reported digit),
    # loose enough to absorb last-ULP drift between supported NumPy/CPython
    # stacks.
    NUMERIC_TOLERANCE = 1e-12

    def _cells_match(self, produced: str, expected: str) -> bool:
        """True when two serialized cells agree up to last-ULP float drift.

        Byte-comparing serialized cells pins the committed CSV to exactly one
        floating-point stack: the extractor's aggregation lands on a different
        last ULP under other supported interpreters, so a bit-exact assertion
        fails for no scientific reason.  Numbers are therefore compared as
        parsed floats within ``NUMERIC_TOLERANCE``; everything else (labels,
        identifiers, key names, and the structure of ``key=value;...``
        payloads such as ``uncertainty``) is still compared byte-for-byte.
        """

        if produced == expected:
            return True
        try:
            return math.isclose(float(produced), float(expected),
                                rel_tol=self.NUMERIC_TOLERANCE, abs_tol=0.0)
        except ValueError:
            pass
        # Structured `key=value;key=value` payloads: keys and ordering must be
        # identical; only the individual values may drift in their last ULP.
        produced_parts = produced.split(";")
        expected_parts = expected.split(";")
        if len(produced_parts) < 2 or len(produced_parts) != len(expected_parts):
            return False
        for produced_part, expected_part in zip(produced_parts, expected_parts):
            if produced_part == expected_part:
                continue
            produced_key, _, produced_number = produced_part.partition("=")
            expected_key, _, expected_number = expected_part.partition("=")
            if produced_key != expected_key:
                return False
            try:
                if not math.isclose(float(produced_number), float(expected_number),
                                    rel_tol=self.NUMERIC_TOLERANCE, abs_tol=0.0):
                    return False
            except ValueError:
                return False
        return True

    def _assert_rows_match_snapshot(
        self, generated: list[dict[str, str]], committed: list[dict[str, str]]
    ) -> None:
        self.assertEqual(len(generated), len(committed))
        for index, (produced, expected) in enumerate(zip(generated, committed)):
            self.assertEqual(sorted(produced), sorted(expected), f"row {index} fields")
            for field, expected_value in expected.items():
                self.assertTrue(
                    self._cells_match(produced[field], expected_value),
                    f"row {index} field {field!r}: {produced[field]!r} does not match "
                    f"snapshot {expected_value!r} within a "
                    f"{self.NUMERIC_TOLERANCE:g} relative tolerance",
                )

    def test_extractor_reproduces_snapshot_when_pinned_inputs_are_available(self) -> None:
        workbook = Path("/tmp/pathorob_source_data.xlsx")
        repository = Path("/tmp/pathopress_sources/pathorob")
        if not workbook.is_file() or not repository.is_dir():
            self.skipTest("pinned PathoROB workbook/repository are unavailable")
        generated = paper_rows(workbook) + repository_example_rows(repository)
        with SNAPSHOT.open(newline="", encoding="utf-8") as handle:
            committed = list(csv.DictReader(handle))
        self._assert_rows_match_snapshot(generated, committed)

    def test_snapshot_comparison_still_rejects_a_real_extraction_regression(self) -> None:
        with SNAPSHOT.open(newline="", encoding="utf-8") as handle:
            committed = list(csv.DictReader(handle))
        # A last-ULP difference is tolerated ...
        nudged = [dict(row) for row in committed]
        nudged[0]["value"] = repr(math.nextafter(float(nudged[0]["value"]), math.inf))
        self.assertNotEqual(nudged[0]["value"], committed[0]["value"])
        self._assert_rows_match_snapshot(nudged, committed)
        # ... a change big enough to matter scientifically is not.
        broken = [dict(row) for row in committed]
        broken[0]["value"] = repr(float(broken[0]["value"]) * (1 + 1e-9))
        with self.assertRaises(AssertionError):
            self._assert_rows_match_snapshot(broken, committed)
        # ... and non-numeric drift is still byte-compared.
        relabelled = [dict(row) for row in committed]
        relabelled[0]["model_id"] = relabelled[0]["model_id"] + "-v2"
        with self.assertRaises(AssertionError):
            self._assert_rows_match_snapshot(relabelled, committed)
        # Structured `key=value;...` payloads get the same treatment.
        payload = "95%_ci_half_width=0.772179915336249;n=60;df=59"
        self.assertTrue(
            self._cells_match("95%_ci_half_width=0.772179915336248;n=60;df=59", payload)
        )
        self.assertFalse(
            self._cells_match("95%_ci_half_width=0.772179925336249;n=60;df=59", payload)
        )
        self.assertFalse(
            self._cells_match("95%_ci_half_width=0.772179915336249;n=61;df=59", payload)
        )
        self.assertFalse(
            self._cells_match("ci_half_width=0.772179915336249;n=60;df=59", payload)
        )

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
        self.assertEqual((len(models), len(evaluations), int((matrix == matrix).sum())), (59, 187, 2122))


if __name__ == "__main__":
    unittest.main()
