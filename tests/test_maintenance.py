from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pathopress.maintenance import (
    build_freshness_manifest,
    build_result_graph_manifest,
    check_freshness_manifest,
    path_record,
    validate_experiment_set,
    validate_probe_compression_semantics,
)


ROOT = Path(__file__).resolve().parents[1]


class MaintenanceTests(unittest.TestCase):
    def test_probe_semantics_reject_hash_fresh_legacy_margin(self) -> None:
        self.assertEqual(validate_probe_compression_semantics(ROOT), [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            (root / "data").mkdir()
            (root / "data/scores.csv").write_bytes((ROOT / "data/scores.csv").read_bytes())
            (root / "data/low_friction_allowlist_v2_top25.json").write_bytes(
                (ROOT / "data/low_friction_allowlist_v2_top25.json").read_bytes()
            )
            artifact = json.loads((ROOT / "experiments/probe_compression_rank1.json").read_text())
            artifact["configuration"]["ranking_margin"] = 2.0
            (root / "experiments/probe_compression_rank1.json").write_text(json.dumps(artifact))
            failures = validate_probe_compression_semantics(root)
            self.assertIn("expected_margin5", {row["status"] for row in failures})
            artifact["configuration"]["ranking_margin"] = 5.0
            artifact["ranking_aware"]["legacy_diagnostic"] = {}
            (root / "experiments/probe_compression_rank1.json").write_text(json.dumps(artifact))
            failures = validate_probe_compression_semantics(root)
            self.assertIn(
                "ranking_schema_must_contain_only_current_candidate_universes",
                {row["status"] for row in failures},
            )
            artifact["ranking_aware"].pop("legacy_diagnostic")
            artifact["pruning"]["keep_count"] = 29
            (root / "experiments/probe_compression_rank1.json").write_text(json.dumps(artifact))
            failures = validate_probe_compression_semantics(root)
            self.assertIn(
                "expected_pruning_keep_count30", {row["status"] for row in failures}
            )
            artifact["pruning"]["keep_count"] = 30
            artifact["pruning"]["source_steps_used"] = 9
            (root / "experiments/probe_compression_rank1.json").write_text(json.dumps(artifact))
            failures = validate_probe_compression_semantics(root)
            self.assertIn(
                "expected_pruning_source_steps_used10",
                {row["status"] for row in failures},
            )
            artifact["pruning"]["source_steps_used"] = 10
            artifact["curves"]["any_candidate"]["all_known_greedy_medae"][0]["k"] = 0
            (root / "experiments/probe_compression_rank1.json").write_text(json.dumps(artifact))
            failures = validate_probe_compression_semantics(root)
            self.assertIn(
                "any_candidate_all_known_greedy_medae_requires_exact_k1_10",
                {row["status"] for row in failures},
            )

    def test_freshness_manifest_detects_modified_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, artifact = root / "source.txt", root / "artifact.txt"
            source.write_text("source")
            artifact.write_text("artifact")
            manifest = build_freshness_manifest(root, inputs=[source], artifacts=[artifact], kind="test")
            self.assertEqual(check_freshness_manifest(root, manifest), [])
            artifact.write_text("changed")
            self.assertEqual(check_freshness_manifest(root, manifest)[0]["status"], "stale_or_modified")

    def test_result_graph_hashes_directory_dependencies_and_component_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.py").write_text("pass\n")
            (root / "input.txt").write_text("input")
            cache = root / "cache"
            cache.mkdir()
            (cache / "shard.json").write_text("{}")
            (root / "result.json").write_text("{}")
            experiment_set_path = root / "set.json"
            experiment_set = {
                "experiments": [{
                    "name": "graph", "command": "python3 run.py",
                    "inputs": ["input.txt"],
                    "dependencies": [{"path": "cache", "role": "ignored_cache"}],
                    "artifacts": ["result.json"],
                }]
            }
            experiment_set_path.write_text(json.dumps(experiment_set))
            manifest = build_result_graph_manifest(
                root,
                experiment_set_path=experiment_set_path,
                experiment_set=experiment_set,
            )
            self.assertEqual(manifest["dependencies"]["cache"]["kind"], "directory")
            self.assertEqual(manifest["dependencies"]["cache"]["file_count"], 1)
            self.assertEqual(check_freshness_manifest(root, manifest), [])
            (cache / "shard.json").write_text('{"changed": true}')
            failures = check_freshness_manifest(root, manifest)
            self.assertIn(
                {"path": "cache", "status": "stale_or_modified"}, failures
            )

    def test_directory_digest_ignores_runtime_junk_but_tracks_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            module = source / "module.py"
            module.write_text("VALUE = 1\n")
            initial = path_record(source)["sha256"]
            runtime = source / "__pycache__"
            runtime.mkdir()
            (runtime / "module.cpython-312.pyc").write_bytes(b"first")
            (source / ".pytest_cache").mkdir()
            (source / ".pytest_cache" / "state").write_text("changed")
            (source / "writer.tmp").write_text("partial")
            (source / "writer.lock").write_text("locked")
            self.assertEqual(path_record(source)["sha256"], initial)
            (runtime / "module.cpython-312.pyc").write_bytes(b"second")
            self.assertEqual(path_record(source)["sha256"], initial)
            module.write_text("VALUE = 2\n")
            self.assertNotEqual(path_record(source)["sha256"], initial)

    def test_experiment_set_validation_is_read_only_and_flags_missing_inputs(self) -> None:
        ready = validate_experiment_set(
            ROOT,
            {"experiments": [{"name": "ok", "command": "python3 scripts/dry_run_experiment_set.py", "inputs": ["data/scores.csv"], "external_calls": False}]},
        )
        self.assertEqual(ready[0]["status"], "ready")
        blocked = validate_experiment_set(
            ROOT,
            {"experiments": [{"name": "missing", "command": "python3 scripts/dry_run_experiment_set.py", "inputs": ["does-not-exist"], "external_calls": False}]},
        )
        self.assertEqual(blocked[0]["status"], "blocked")

    def test_repository_inventory_covers_completed_result_graphs(self) -> None:
        payload = json.loads((ROOT / "experiments/experiment_set.json").read_text())
        self.assertEqual(payload["schema_version"], 2)
        results = validate_experiment_set(ROOT, payload)
        self.assertTrue(all(row["status"] == "ready" for row in results), results)
        names = {row["name"] for row in results}
        self.assertTrue({
            "method_comparison_grid", "structure_analysis", "probe_exhaustive",
            "ranking_preservation", "confidence_calibration",
            "predictability_and_error_analysis", "prediction_error_factors",
            "temporal_deployment", "publication_tables", "publication_hero",
            "publication_metadata_overview", "public_export",
        }.issubset(names))
        by_name = {row["name"]: row for row in results}
        self.assertGreater(by_name["method_comparison_grid"]["declared_dependencies"], 0)
        self.assertGreater(by_name["prediction_error_factors"]["declared_dependencies"], 0)
        self.assertTrue(all(row["declared_artifacts"] > 0 for row in results))

        ignored = [
            dependency
            for experiment in payload["experiments"]
            for dependency in experiment.get("dependencies", [])
            if isinstance(dependency, dict) and dependency.get("role", "").startswith("ignored")
        ]
        self.assertIn("content-digested", payload["description"])
        self.assertEqual(len(ignored), 5)
        self.assertTrue(all(dependency.get("config") for dependency in ignored))

        manifest = json.loads(
            (ROOT / "experiments/artifact_freshness_manifest.json").read_text()
        )
        self.assertEqual(manifest["schema_version"], 2)
        for dependency in ignored:
            record = manifest["dependencies"][dependency["path"]]
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(record["bytes"], 0)
            if record["kind"] == "directory":
                self.assertGreater(record["file_count"], 0)


if __name__ == "__main__":
    unittest.main()
