from __future__ import annotations

import json
import unittest
from pathlib import Path

import experiments.validate_probe_exhaustive_chunks as chunks
import experiments.validate_probe_exhaustive_merged as merged
import experiments.validate_probe_exhaustive_top as top


ROOT = Path(__file__).resolve().parents[1]


class ProbeExhaustiveValidatorVersionTests(unittest.TestCase):
    def tearDown(self) -> None:
        # Preserve the historical import-time API used by legacy fixture tests.
        chunks.select_runner("legacy")
        top.select_runner("legacy")

    def test_v2_equivalence_contract_uses_stricter_schema(self) -> None:
        payload = json.loads(
            (ROOT / "experiments/probe_exhaustive_fast_equivalence_v2.json").read_text()
        )
        self.assertIs(chunks.select_runner("v2"), chunks.v2_runner)
        tolerances, minimum = chunks._equivalence_limits(payload, "v2")
        self.assertEqual(minimum, 32)
        self.assertEqual(tolerances, payload["requested_tolerances"])
        self.assertIn("execution_backend", chunks.V2_EXPECTED_CONFIG_KEYS)
        self.assertNotIn("execution_backend", chunks.LEGACY_EXPECTED_CONFIG_KEYS)

    def test_top_validator_selects_v2_without_removing_legacy(self) -> None:
        payload = json.loads(
            (ROOT / "experiments/probe_exhaustive_fast_equivalence_v2.json").read_text()
        )
        self.assertIs(top.select_runner("v2"), top.v2_runner)
        _, minimum = top._equivalence_tolerances(payload, "v2")
        self.assertEqual(minimum, 32)
        self.assertIs(top.select_runner("legacy"), top.legacy_runner)

    def test_merged_validator_rejects_cross_schema_validation(self) -> None:
        v2 = {
            "schema_version": 2,
            "config_schema": "pathopress.probe_exhaustive.run.v2",
        }
        legacy = {"schema_version": 1}
        merged.validate_runner_contract(v2, "v2")
        merged.validate_runner_contract(legacy, "legacy")
        with self.assertRaisesRegex(RuntimeError, "schema-v2"):
            merged.validate_runner_contract(legacy, "v2")
        with self.assertRaisesRegex(RuntimeError, "legacy-v1"):
            merged.validate_runner_contract(v2, "legacy")


if __name__ == "__main__":
    unittest.main()
