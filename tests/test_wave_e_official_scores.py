from __future__ import annotations

import csv
import unittest
from collections import Counter
from pathlib import Path

from scripts.extract_wave_e_official_scores import (
    conch15_rows,
    conch_rows,
    ctranspath_rows,
    phikon_family_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> list[dict[str, str]]:
    with (ROOT / "source_data" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class WaveEOfficialScoresTests(unittest.TestCase):
    def test_snapshot_shapes_and_dispositions(self) -> None:
        conch = read("conch_official_scores_2024.csv")
        conch15 = read("conch15_titan_official_scores_2025.csv")
        phikon = read("phikon_family_official_scores_2023_2024.csv")
        ctranspath = read("ctranspath_official_evidence_2022_2024.csv")
        self.assertEqual((len(conch), len(conch15), len(phikon), len(ctranspath)), (104, 433, 47, 50))
        self.assertEqual(Counter(row["inclusion_status"] for row in conch), {
            "canonical_candidate": 77, "private_internal_excluded": 16, "fine_tuned_excluded": 11,
        })
        self.assertEqual(Counter(row["inclusion_status"] for row in conch15), {
            "canonical_candidate": 297, "private_internal_excluded": 124, "fine_tuned_excluded": 12,
        })
        self.assertEqual(Counter(row["inclusion_status"] for row in phikon), {
            "canonical_candidate": 43, "private_internal_excluded": 4,
        })
        self.assertEqual(Counter(row["inclusion_status"] for row in ctranspath), {
            "secondary_only_excluded": 50,
        })

    def test_exact_representative_values(self) -> None:
        rows = read("conch_official_scores_2024.csv")
        self.assertEqual(next(row for row in rows if row["evaluation_id"] == "conch.natmed2024.t1.0.zero-shot_classification.balanced_accuracy")["value"], "0.913")
        self.assertEqual(next(row for row in rows if row["evaluation_id"] == "conch.natmed2024.t31.1pct.end-to-end_fine-tuning.quadratic_weighted_kappa")["value"], "0.662")

        rows = read("conch15_titan_official_scores_2025.csv")
        self.assertEqual(next(row for row in rows if row["evaluation_id"] == "conch15.titan2025.t22.0.logistic_regression.balanced_accuracy")["value"], "0.779")
        self.assertEqual(next(row for row in rows if row["evaluation_id"] == "conch15.titan2025.t120.retrieval.top3_accuracy")["value"], "0.866")

        rows = read("phikon_family_official_scores_2023_2024.csv")
        self.assertEqual(next(row for row in rows if row["evaluation_id"] == "phikon-v2.phikonv2.t2.metastasis.camelyon16.auroc")["value"], "0.997")
        self.assertEqual(next(row for row in rows if row["evaluation_id"] == "phikon.medrxiv2024.f2.tcga_crc_to_paip.abmil.auroc")["value"], "0.947")

    def test_protocol_and_access_boundaries(self) -> None:
        rows = [
            *read("conch_official_scores_2024.csv"),
            *read("conch15_titan_official_scores_2025.csv"),
            *read("phikon_family_official_scores_2023_2024.csv"),
            *read("ctranspath_official_evidence_2022_2024.csv"),
        ]
        selected = [row for row in rows if row["inclusion_status"] == "canonical_candidate"]
        self.assertFalse(any(row["cohort_access"] != "public" for row in selected))
        self.assertFalse(any("fine-tun" in row["downstream_protocol"] for row in selected))
        self.assertFalse(any(row["model_id"] == "ctranspath" for row in selected))
        keys = [(row["model_id"], row["evaluation_id"]) for row in selected]
        self.assertEqual(len(keys), len(set(keys)))
        private_msi = [row for row in rows if row["model_id"] in {"phikon", "phikon-v2"} and ("Cy1" in row["task_label"] or "NGX1" in row["task_label"])]
        self.assertEqual(len(private_msi), 4)
        self.assertTrue(all(row["inclusion_status"] == "private_internal_excluded" for row in private_msi))

    def test_extractor_reproduces_snapshots_when_sources_are_available(self) -> None:
        paths = {
            "conch": Path("/tmp/conch_supp.pdf"), "titan": Path("/tmp/titan_supp.pdf"),
            "v2": Path("/tmp/phikon_v2.pdf"), "supp": Path("/tmp/phikon_supp.pdf"),
        }
        if not all(path.exists() for path in paths.values()):
            self.skipTest("pinned Wave E source PDFs unavailable")
        self.assertEqual(conch_rows(paths["conch"]), read("conch_official_scores_2024.csv"))
        self.assertEqual(conch15_rows(paths["titan"]), read("conch15_titan_official_scores_2025.csv"))
        self.assertEqual(phikon_family_rows(paths["v2"], paths["supp"]), read("phikon_family_official_scores_2023_2024.csv"))
        self.assertEqual(ctranspath_rows(paths["v2"], paths["supp"], paths["conch"]), read("ctranspath_official_evidence_2022_2024.csv"))


if __name__ == "__main__":
    unittest.main()
