from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pathopress.starter_sets import STARTER_SET_SCHEMA, build_starter_sets


class StarterSetTests(unittest.TestCase):
    def test_builds_hash_bound_unrestricted_and_feasibility_sets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowlist = root / "allowlist.json"
            allowlist.write_text(json.dumps({"evaluation_ids": ["f1", "f2"]}))
            digest = hashlib.sha256(allowlist.read_bytes()).hexdigest()
            probe = root / "probe.json"
            probe.write_text(json.dumps({
                "configuration": {"scores_sha256": "a" * 64, "allowlist_sha256": digest},
                "curves": {
                    "any_candidate": {"all_known_greedy_medae": [
                        {"added_evaluation_id": "u1"}, {"added_evaluation_id": "u2"},
                    ]},
                    "pre_error_low_friction_allowlist": {"all_known_greedy_medae": [
                        {"added_evaluation_id": "f1"}, {"added_evaluation_id": "f2"},
                    ]},
                },
            }))
            result = build_starter_sets(probe, allowlist, count=2)
            self.assertEqual(result["schema_version"], STARTER_SET_SCHEMA)
            self.assertEqual(
                result["sets"]["unrestricted"]["evaluation_ids"], ["u1", "u2"]
            )
            self.assertEqual(
                result["sets"]["feasibility"]["evaluation_ids"], ["f1", "f2"]
            )
            self.assertIn("not measured cost", result["sets"]["feasibility"]["semantics"])

    def test_rejects_stale_allowlist_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowlist = root / "allowlist.json"
            allowlist.write_text('{"evaluation_ids": ["f1"]}')
            probe = root / "probe.json"
            probe.write_text(json.dumps({
                "configuration": {"scores_sha256": "a", "allowlist_sha256": "stale"},
                "curves": {},
            }))
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_starter_sets(probe, allowlist, count=1)


if __name__ == "__main__":
    unittest.main()
