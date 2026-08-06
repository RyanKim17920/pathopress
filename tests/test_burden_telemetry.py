from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_burden_telemetry.py"
SCHEMA = ROOT / "data/evaluation_burden_measurements.schema.json"
GROUPS = ROOT / "data/evaluation_artifact_cost_groups.csv"
PROFILES = ROOT / "data/evaluation_budget_profiles.json"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_burden_telemetry", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BurdenTelemetryTests(unittest.TestCase):
    def test_schema_declares_complete_status_and_phase_contracts(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        statuses = set(schema["$defs"]["status"]["enum"])
        self.assertEqual(
            statuses,
            {
                "measured",
                "source_reported",
                "configured_ceiling",
                "not_applicable",
                "not_measured",
                "not_reported",
                "inaccessible",
            },
        )
        self.assertEqual(statuses, load_runner_module().ALLOWED_STATUSES)
        measurement = schema["$defs"]["measurement"]
        for key in (
            "model_revision",
            "evaluation_id",
            "run_config_hash",
            "hardware_id",
            "cache_scope",
        ):
            self.assertIn(key, measurement["required"])
        phases = set(measurement["properties"]["phase"]["enum"])
        self.assertEqual(phases, load_runner_module().PHASES)
        required_resources = set(
            measurement["properties"]["resources"]["required"]
        )
        declared_resources = set(
            measurement["properties"]["resources"]["properties"]
        )
        self.assertEqual(required_resources, declared_resources)

    def test_unknown_resource_facts_are_null_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            config = temp / "config.json"
            config.write_text('{"batch_size": 4}\n', encoding="utf-8")
            output = temp / "receipt.json"
            subprocess.run(
                [
                    "python3",
                    str(RUNNER),
                    "--model-revision",
                    "example/model@revision",
                    "--evaluation-id",
                    "eva.leaderboard.bach.validation",
                    "--artifact-group-id",
                    "eva.dataset.bach",
                    "--phase",
                    "per_protocol_evaluation",
                    "--hardware-id",
                    "test-cpu",
                    "--cache-scope",
                    "warm",
                    "--run-config",
                    str(config),
                    "--output",
                    str(output),
                    "--",
                    "python3",
                    "-c",
                    "sum(range(1000))",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            measurement = payload["measurement"]
            schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
            declared = schema["$defs"]["measurement"]
            self.assertEqual(set(measurement), set(declared["required"]))
            self.assertEqual(
                set(measurement["resources"]),
                set(declared["properties"]["resources"]["required"]),
            )
            self.assertEqual(measurement["execution_status"], "completed")
            self.assertEqual(measurement["exit_code"], 0)
            self.assertEqual(measurement["model_revision"], "example/model@revision")
            self.assertEqual(measurement["cache_scope"], "warm")
            self.assertRegex(measurement["run_config_hash"], r"^[0-9a-f]{64}$")
            for field in ("wall_time", "cpu_user_time", "cpu_system_time", "peak_ram"):
                fact = measurement["resources"][field]
                self.assertEqual(fact["status"], "measured")
                self.assertIsInstance(fact["value"], (int, float))
                self.assertGreaterEqual(fact["value"], 0)
            for field in (
                "accelerator_time",
                "peak_vram",
                "download_volume",
                "storage_volume",
                "access_lead_time",
                "access_admin_labor",
                "annotation_labor",
                "pathologist_labor",
                "new_tissue_cases",
                "new_slides",
                "direct_cost",
            ):
                fact = measurement["resources"][field]
                self.assertEqual(fact["status"], "not_measured")
                self.assertIsNone(fact["value"])
                self.assertTrue(fact["reason"])
            for field in (
                "access_class",
                "dataset_license",
                "commercial_use_allowed",
                "redistribution_allowed",
                "new_tissue_required",
            ):
                fact = measurement["constraints"][field]
                self.assertEqual(fact["status"], "not_measured")
                self.assertIsNone(fact["value"])
                self.assertTrue(fact["reason"])

    def test_shared_setup_allows_no_model_or_evaluation(self) -> None:
        module = load_runner_module()
        args = module.parse_args(
            [
                "--artifact-group-id",
                "pathorob.dataset.camelyon",
                "--phase",
                "shared_artifact_setup",
                "--hardware-id",
                "test-host",
                "--cache-scope",
                "cold",
                "--run-config-hash",
                "a" * 64,
                "--output",
                "/tmp/not-written.json",
                "--",
                "true",
            ]
        )
        self.assertIsNone(args.model_revision)
        self.assertIsNone(args.evaluation_id)

    def test_runner_refuses_to_overwrite_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt.json"
            output.write_text("existing\n", encoding="utf-8")
            result = subprocess.run(
                [
                    "python3",
                    str(RUNNER),
                    "--model-revision",
                    "model@revision",
                    "--evaluation-id",
                    "thunder.bach.linear_probing",
                    "--artifact-group-id",
                    "thunder.dataset.bach",
                    "--phase",
                    "per_model_feature_extraction",
                    "--hardware-id",
                    "test-host",
                    "--cache-scope",
                    "cold",
                    "--run-config-hash",
                    "b" * 64,
                    "--output",
                    str(output),
                    "--",
                    "true",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing\n")

    def test_artifact_groups_and_budget_profiles_do_not_impute_cost(self) -> None:
        with GROUPS.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["suite_id"] for row in rows}, {
            "eva", "hest", "hoptimus1_report", "pathobench", "pathorob", "thunder"
        })
        self.assertTrue(all(row["setup_scope"] == "shared_artifact_setup" for row in rows))
        profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
        self.assertFalse(profiles["missingness_policy"]["unknown_is_zero"])
        self.assertIn("no built-in scalar", profiles["missingness_policy"]["scalarization_rule"])
        for profile in profiles["profiles"]:
            self.assertIsNone(profile["budget_values"])
            self.assertIn("requires_user", profile["activation"])


if __name__ == "__main__":
    unittest.main()
