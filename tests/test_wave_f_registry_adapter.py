from __future__ import annotations

import csv
import unittest
from collections import Counter
from pathlib import Path

from scripts.evidence.wave_f_official_scores import build_protocols, build_scores, selected


ROOT = Path(__file__).resolve().parents[1]
PATHS = [
    ROOT / "source_data/hibou_official_scores_2024.csv",
    ROOT / "source_data/musk_official_scores_2025.csv",
    ROOT / "source_data/gpfm_official_scores_2025.csv",
]


class WaveFRegistryAdapterTests(unittest.TestCase):
    def test_selected_inventory(self) -> None:
        rows = selected(PATHS)
        self.assertEqual(len(rows), 185)
        self.assertEqual(Counter(row["model_id"] for row in rows), {
            "hibou-b": 9,
            "hibou-l": 9,
            "musk": 68,
            "gpfm": 99,
        })

    def test_protocols_normalization_and_identity_links(self) -> None:
        with (ROOT / "data/tasks.csv").open(newline="", encoding="utf-8") as handle:
            tasks = list(csv.DictReader(handle))
        protocols = build_protocols(PATHS, tasks)
        scores, aliases = build_scores(PATHS)
        self.assertEqual(len(protocols), 185)
        self.assertEqual(len(scores), 185)
        self.assertEqual(len({row["task_identity_id"] for row in protocols}), 82)
        self.assertEqual(len(aliases), 4)

        by_id = {row["evaluation_id"]: row for row in protocols}
        self.assertEqual(
            by_id["hibou-b.hibou2024.t1.crc_100k.top1_accuracy"]["task_identity_id"],
            "task.crc100k.nine_class_classification",
        )
        self.assertEqual(
            by_id["musk.nature2025.t5.musk.patchcamelyon.balanced_accuracy"]["task_identity_id"],
            "task.patchcamelyon.binary_metastasis_classification",
        )
        self.assertEqual(
            by_id["musk.nature2025.t5.musk.patchcamelyon.balanced_accuracy"]["metric"],
            "balanced_accuracy_percent",
        )
        gpfm_metrics = {
            row["task_identity_id"]
            for row in protocols
            if row["evaluation_id"].startswith(
                "gpfm.nbe2025.wsi_classification.tcga_nsclc."
            )
        }
        self.assertEqual(len(gpfm_metrics), 1)

        scores_by_id = {row["evaluation_id"]: row for row in scores}
        self.assertEqual(
            scores_by_id["hibou-b.hibou2024.t1.crc_100k.top1_accuracy"][
                "normalized_score"
            ],
            "95.5",
        )
        self.assertEqual(
            scores_by_id["musk.nature2025.t1.musk.bookset.recall_at_1"][
                "normalized_score"
            ],
            "16.07",
        )


if __name__ == "__main__":
    unittest.main()
