from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "experiments/run_budgeted_probe_selection.py"
SPEC = importlib.util.spec_from_file_location("run_budgeted_probe_selection", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class BudgetedProbeRunnerTests(unittest.TestCase):
    def test_worker_default_and_hard_cap(self):
        args = RUNNER.parse_args([])
        self.assertGreaterEqual(args.workers, 1)
        self.assertLessEqual(args.workers, 4)
        with self.assertRaises(SystemExit):
            RUNNER.parse_args(["--workers", "5"])

    def test_checked_in_nonmeasurement_receipt_fails_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            exit_code = RUNNER.main([
                "--workers", "1",
                "--output", str(output),
            ])
            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "insufficient_cost_coverage")
        self.assertEqual(payload["coverage"]["n_eligible_candidates"], 0)
        self.assertEqual(payload["coverage"]["n_excluded_candidates"], 187)
        self.assertNotIn("selection", payload)
        self.assertIn("No selection or chart is valid", payload["reason"])
        self.assertEqual(payload["configuration"]["worker_cap"], 4)

    def test_complete_path_has_selection_validation_baseline_and_random(self):
        example = json.loads(
            (ROOT / "data/evaluation_burden_measurements_v1.example.json").read_text()
        )
        measurement = example["measurement"]
        for name in ("accelerator_time", "annotation_labor", "direct_cost", "peak_vram"):
            unit = measurement["resources"][name]["unit"]
            measurement["resources"][name] = {
                "status": "measured",
                "value": 1.0,
                "unit": unit,
                "measurement_method": "runner integration fixture",
            }
        measurement["constraints"]["access_class"] = {
            "status": "measured",
            "value": "open_download",
            "measurement_method": "runner integration fixture",
        }
        measurement["constraints"]["commercial_use_allowed"] = {
            "status": "measured",
            "value": True,
            "measurement_method": "runner integration fixture",
        }
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            burden = directory_path / "receipt.json"
            output = directory_path / "result.json"
            burden.write_text(json.dumps(example) + "\n", encoding="utf-8")
            subprocess.run(
                [
                    "python3", str(RUNNER_PATH),
                    "--burden", str(burden),
                    "--output", str(output),
                    "--workers", "1",
                    "--max-probes", "1",
                    "--random-repeats", "1",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["coverage"]["n_eligible_candidates"], 1)
        self.assertEqual(len(payload["selection"]["trajectory"]), 1)
        self.assertEqual(len(payload["heldout_validation"]), 1)
        self.assertEqual(len(payload["random_feasible_baseline"]), 1)
        self.assertIn("all_known", payload["zero_probe_baseline"])
        self.assertIn("heldout", payload["zero_probe_baseline"])


if __name__ == "__main__":
    unittest.main()
