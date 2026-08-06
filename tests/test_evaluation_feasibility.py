from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EvaluationFeasibilityTests(unittest.TestCase):
    def test_current_eligible_legacy_and_refreshed_contracts(self) -> None:
        all_eligible = json.loads(
            (ROOT / "data/low_friction_pipeline_eligible_v2_all.json").read_text()
        )
        legacy = json.loads(
            (ROOT / "data/low_friction_allowlist_v2_legacy25.json").read_text()
        )
        refreshed = json.loads(
            (ROOT / "data/low_friction_allowlist_v2_top25.json").read_text()
        )
        self.assertEqual(all_eligible["n_eligible"], 29)
        self.assertEqual(all_eligible["n_task_identities"], 17)
        self.assertEqual(len(legacy["evaluation_ids"]), 25)
        self.assertEqual(len(refreshed["evaluation_ids"]), 25)
        self.assertFalse(refreshed["rule"]["prediction_error_used"])

        with (ROOT / "data/tasks.csv").open(newline="", encoding="utf-8") as handle:
            identities = {
                row["evaluation_id"]: row["task_identity_id"] for row in csv.DictReader(handle)
            }
        first_pass = refreshed["evaluation_ids"][: refreshed["n_identity_representatives"]]
        self.assertEqual(len({identities[value] for value in first_pass}), 17)

        additions = {
            "hoptimus1report2025.cam17_wilds.linear_probe_top1",
            "hoptimus1report2025.crc_no_norm.linear_probe_top1",
            "hoptimus1report2025.mhist.linear_probe_top1",
            "hoptimus1report2025.tcga_uniform.linear_probe_top1",
        }
        self.assertTrue(additions <= set(all_eligible["evaluation_ids"]))
        legacy_identities = {identities[value] for value in legacy["evaluation_ids"]}
        self.assertTrue(all(identities[value] in legacy_identities for value in additions))


if __name__ == "__main__":
    unittest.main()
