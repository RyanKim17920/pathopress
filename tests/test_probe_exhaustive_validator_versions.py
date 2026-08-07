from __future__ import annotations

import unittest

import experiments.validate_probe_exhaustive_chunks as chunks
import experiments.validate_probe_exhaustive_merged as merged
import experiments.validate_probe_exhaustive_top as top


class ProbeExhaustiveValidatorVersionTests(unittest.TestCase):
    def test_validators_share_the_schema_v2_runner(self) -> None:
        self.assertIs(chunks.runner, top.runner)
        self.assertEqual(chunks.runner.CONFIG_SCHEMA_VERSION, 2)
        self.assertIn("execution_backend", chunks.EXPECTED_CONFIG_KEYS)

    def test_merged_validator_rejects_cross_schema_validation(self) -> None:
        v2 = {
            "schema_version": 2,
            "config_schema": "pathopress.probe_exhaustive.run.v2",
        }
        legacy = {"schema_version": 1}
        merged.validate_runner_contract(v2)
        with self.assertRaisesRegex(RuntimeError, "schema-v2"):
            merged.validate_runner_contract(legacy)


if __name__ == "__main__":
    unittest.main()
