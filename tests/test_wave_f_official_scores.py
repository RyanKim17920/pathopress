from __future__ import annotations

import csv
import importlib.util
import unittest
from collections import Counter
from pathlib import Path

from scripts.extract_wave_f_official_scores import (
    MUSK_ZERO_SHOT_DISPOSITION,
    gpfm_rows,
    hibou_rows,
    musk_rows,
)
from tests.pinned_sources import missing_inputs


ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> list[dict[str, str]]:
    with (ROOT / "source_data" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class WaveFOfficialScoresTests(unittest.TestCase):
    def test_shapes_and_dispositions(self) -> None:
        hibou = read("hibou_official_scores_2024.csv")
        musk = read("musk_official_scores_2025.csv")
        gpfm = read("gpfm_official_scores_2025.csv")
        self.assertEqual((len(hibou), len(musk), len(gpfm)), (38, 135, 224))
        self.assertEqual(Counter(r["inclusion_status"] for r in hibou), {
            "canonical_candidate": 18, "fine_tuned_excluded": 20,
        })
        self.assertEqual(Counter(r["inclusion_status"] for r in musk), {
            "canonical_candidate": 68, "aggregate_excluded": 54,
            "fine_tuned_excluded": 9, "private_internal_excluded": 4,
        })
        self.assertEqual(Counter(r["inclusion_status"] for r in gpfm), {
            "canonical_candidate": 99, "private_internal_excluded": 75,
            "fine_tuned_excluded": 43, "aggregate_excluded": 7,
        })

    def test_representative_values_and_protocols(self) -> None:
        hibou = read("hibou_official_scores_2024.csv")
        self.assertEqual(next(r for r in hibou if r["evaluation_id"] == "hibou-b.hibou2024.t1.crc_100k.top1_accuracy")["value"], "0.955")
        self.assertEqual(next(r for r in hibou if r["evaluation_id"] == "hibou-l.hibou2024.t2.rcc.auroc")["value"], "0.996")
        self.assertTrue(all(r["model_id"] == "hibou-l" for r in hibou if r["inclusion_status"] == "fine_tuned_excluded"))

        musk = read("musk_official_scores_2025.csv")
        self.assertEqual(next(r for r in musk if r["evaluation_id"] == "musk.nature2025.t1.musk.bookset.recall_at_1")["value"], "16.07")
        self.assertEqual(next(r for r in musk if r["evaluation_id"] == "musk.nature2025.t7.musk.muv_idh.auroc")["value"], "0.978")
        self.assertEqual(len([r for r in musk if r["cohort_access"] == "controlled"]), 4)
        pathvqa_large = next(r for r in musk if r["model_id"] == "musk-large" and r["task_label"] == "PathVQA")
        self.assertEqual(pathvqa_large["value"], "73.21")
        self.assertEqual(pathvqa_large["inclusion_status"], "fine_tuned_excluded")
        self.assertIn("whole-model", pathvqa_large["downstream_protocol"])
        variant_ids = {r["model_id"] for r in musk if r["model_alias"] != "MUSK"}
        self.assertEqual(variant_ids, {
            "musk-ablation-final", "musk-ablation-1", "musk-ablation-2",
            "musk-ablation-3", "musk-ablation-4", "musk-base-model5",
            "musk-small", "musk-base", "musk-large",
        })
        self.assertEqual(MUSK_ZERO_SHOT_DISPOSITION, {
            "PatchCamelyon": "exact_table4_leaf", "SkinCancer": "exact_table4_leaf",
            "PanNuke": "exact_table4_leaf", "UniToPatho": "exact_table4_leaf",
            "NCT-CRC": "graph_only_unlocated_exact_value",
            "SICAPv2": "graph_only_unlocated_exact_value",
        })

        gpfm = read("gpfm_official_scores_2025.csv")
        self.assertEqual(next(r for r in gpfm if r["evaluation_id"] == "gpfm.nbe2025.wsi_classification.tcga_nsclc.auroc")["value"], "0.986")
        self.assertEqual(next(r for r in gpfm if r["evaluation_id"] == "gpfm.nbe2025.roi_retrieval.crc_100k_retrieval.top5_accuracy")["value"], "0.995")

    def test_candidate_safety_boundaries(self) -> None:
        rows = [
            *read("hibou_official_scores_2024.csv"),
            *read("musk_official_scores_2025.csv"),
            *read("gpfm_official_scores_2025.csv"),
        ]
        selected = [r for r in rows if r["inclusion_status"] == "canonical_candidate"]
        self.assertTrue(all(r["cohort_access"] == "public" for r in selected))
        self.assertFalse(any("fine-tun" in r["downstream_protocol"] for r in selected))
        self.assertFalse(any("aggregate" in r["downstream_protocol"] for r in selected))
        keys = [(r["model_id"], r["evaluation_id"]) for r in selected]
        self.assertEqual(len(keys), len(set(keys)))
        protocols = {r["downstream_protocol"] for r in selected if r["model_id"] == "musk"}
        self.assertTrue(any("zero-shot" in p for p in protocols))
        self.assertTrue(any("10-shot" in p for p in protocols))
        self.assertTrue(any("linear probe" in p for p in protocols))

    def test_extractors_reproduce_snapshots_when_sources_exist(self) -> None:
        if importlib.util.find_spec("openpyxl") is None:
            self.skipTest("optional evidence dependency openpyxl unavailable")
        base = Path("/tmp/pathopress_wave_f")
        hibou_pdf = base / "hibou" / "2406.05074.pdf"
        musk_pdf = base / "musk" / "41586_2024_8378_MOESM1_ESM.pdf"
        gpfm_pdf = base / "gpfm" / "41551_2025_1488_MOESM1_ESM.pdf"
        if not all(p.exists() for p in (hibou_pdf, musk_pdf, gpfm_pdf)):
            # Publisher supplementary PDFs with no recorded retrieval URL; see
            # tests/pinned_sources.py for why CI cannot provision them.
            missing_inputs(
                self, "pinned Wave F publisher artifacts unavailable", fetchable=False
            )
        self.assertEqual(hibou_rows(hibou_pdf), read("hibou_official_scores_2024.csv"))
        self.assertEqual(musk_rows(musk_pdf), read("musk_official_scores_2025.csv"))
        self.assertEqual(gpfm_rows(gpfm_pdf, base / "gpfm"), read("gpfm_official_scores_2025.csv"))


if __name__ == "__main__":
    unittest.main()
