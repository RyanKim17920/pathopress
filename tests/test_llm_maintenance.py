from __future__ import annotations

import json
import gzip
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pathopress.llm_baseline import (
    CONDITIONS,
    authorize_external_responses,
    build_request,
    deterministic_mock_response,
    evaluate_cached_responses,
    make_config,
    object_sha256,
    seal_external_response,
    select_peer_examples,
    summarize_real_execution,
    validate_config,
    validate_request,
    validate_response,
)
from pathopress.maintenance import (
    build_freshness_manifest,
    build_result_graph_manifest,
    check_freshness_manifest,
    path_record,
    validate_experiment_set,
    validate_probe_compression_semantics,
)


ROOT = Path(__file__).resolve().parents[1]


def _execution_contract(*, count: int, provider: str, model: str, version: str | None, settings: dict) -> dict:
    from pathopress.llm_baseline import object_sha256
    contract = {
        "schema_version": 1,
        "provider": provider,
        "model_alias": model,
        "model_snapshot": version,
        "response_model": model,
        "required_model_version": version,
        "require_resolved_version_distinct_from_alias": False,
        "required_settings_sha256": object_sha256(settings),
        "transport_kind": "online_chat_completions",
        "expected_response_count": count,
        "config_sha256": "c" * 64,
        "request_pack_sha256": "d" * 64,
        "endpoint_identity_sha256": "e" * 64,
        "adapter_manifest_sha256": "2" * 64,
        "transport_profile_sha256": "3" * 64,
        "capacity_profile_sha256": "4" * 64,
        "approval_manifest_sha256": "f" * 64,
        "preflight_sha256": "1" * 64,
    }
    contract["execution_contract_sha256"] = object_sha256(contract)
    return contract


def _lineage_receipt(contract: dict, **extra) -> dict:
    return {
        **extra,
        "approval_manifest_sha256": contract["approval_manifest_sha256"],
        "execution_preflight_sha256": contract["preflight_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "adapter_manifest_sha256": contract.get("adapter_manifest_sha256"),
        "transport_profile_sha256": contract.get("transport_profile_sha256"),
        "capacity_profile_sha256": contract.get("capacity_profile_sha256"),
        "settings_sha256": contract["required_settings_sha256"],
        "model_sha256": object_sha256(contract["response_model"]),
    }


class LlmBaselineSemanticTests(unittest.TestCase):
    def setUp(self) -> None:
        # Target row 0 has six visible anchors and a hidden final target.
        self.matrix = np.array(
            [
                [10, 20, 30, 40, 50, 60, np.nan],
                [11, 21, 31, 41, 51, 61, 70],
                [60, 50, 40, 30, 20, 10, 25],
                [12, 22, 32, 42, 52, 62, 72],
                [15, 25, 35, 45, 55, np.nan, 75],
                [9, 19, 29, 39, 49, 59, 68],
            ],
            dtype=float,
        )
        self.models = [f"model-{i}" for i in range(self.matrix.shape[0])]
        self.evaluations = [f"evaluation-{j}" for j in range(self.matrix.shape[1])]
        self.tasks = {
            value: {"task_family": "classification", "sample_unit": "image", "metric": "accuracy"}
            for value in self.evaluations
        }
        self.config = make_config(
            scores_sha256="a" * 64,
            folds_sha256="b" * 64,
            models=self.models,
            evaluations=self.evaluations,
            fold_ids=[0],
            cell_limit=1,
        )

    def test_config_hash_detects_semantic_drift(self) -> None:
        validate_config(self.config)
        changed = json.loads(json.dumps(self.config))
        changed["min_shared"] = 4
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_config(changed)

    def test_five_shot_peers_use_visible_cells_minimum_and_stable_tie_break(self) -> None:
        peers = select_peer_examples(self.matrix, 0, 6, n_shots=5, min_shared=5)
        # Row 4 has only five shared values and remains eligible; all peer target
        # scores are visible and the held target itself is never a shared anchor.
        self.assertEqual(len(peers), 5)
        self.assertEqual(
            [(-row["correlation"], row["peer_index"]) for row in peers],
            sorted((-row["correlation"], row["peer_index"]) for row in peers),
        )
        self.assertIn(4, [row["peer_index"] for row in peers])
        self.assertNotIn(6, peers[0]["shared_indices"])
        self.assertTrue(all(len(row["shared_indices"]) >= 5 for row in peers))

    def test_named_and_blind_requests_preserve_numbers_but_hide_identifiers(self) -> None:
        named = build_request(
            config=self.config, condition="five_shot_named", fold_id=0, seed=42, fold=0,
            batch_index=0, train=self.matrix, cells=[(0, 6)], models=self.models,
            evaluations=self.evaluations, task_metadata=self.tasks,
        )
        blind = build_request(
            config=self.config, condition="five_shot_blind", fold_id=0, seed=42, fold=0,
            batch_index=0, train=self.matrix, cells=[(0, 6)], models=self.models,
            evaluations=self.evaluations, task_metadata=self.tasks,
        )
        named_text = "\n".join(message["content"] for message in named["messages"])
        blind_text = "\n".join(message["content"] for message in blind["messages"])
        self.assertIn("model-0", named_text)
        self.assertIn("evaluation-6", named_text)
        self.assertNotIn("model-0", blind_text)
        self.assertNotIn("evaluation-6", blind_text)
        self.assertIn("70.000", named_text)
        self.assertIn("70.000", blind_text)
        validate_request(named, self.config)
        validate_request(blind, self.config)

    def test_mock_response_is_hash_bound_and_never_headline_eligible(self) -> None:
        request = build_request(
            config=self.config, condition="zero_shot_named", fold_id=0, seed=42, fold=0,
            batch_index=0, train=self.matrix, cells=[(0, 6)], models=self.models,
            evaluations=self.evaluations, task_metadata=self.tasks,
        )
        response = deterministic_mock_response(request, self.matrix, self.config)
        validate_response(response, request, self.config)
        self.assertFalse(response["headline_eligible"])
        complete = self.matrix.copy()
        complete[0, 6] = 71.0
        metrics = evaluate_cached_responses([request], [response], complete, self.config)
        self.assertEqual(metrics["result_status"], "mock_contract_validation_only")
        self.assertFalse(metrics["headline_eligible"])

    def test_real_response_import_is_exact_complete_and_range_checked(self) -> None:
        request = build_request(
            config=self.config, condition="zero_shot_named", fold_id=0, seed=42, fold=0,
            batch_index=0, train=self.matrix, cells=[(0, 6)], models=self.models,
            evaluations=self.evaluations, task_metadata=self.tasks,
        )
        settings = {"temperature": 0, "max_tokens": 100}
        contract = _execution_contract(
            count=1, provider="test-provider", model="test-model",
            version="2026-08-01", settings=settings,
        )
        raw = {
            "request_id": request["request_id"],
            "backend_kind": "openai_compatible",
            "provider": "test-provider",
            "model": "test-model",
            "response_text": '{"q0": 71.0}',
            "execution_metadata": {
                "model_version": "2026-08-01",
                "settings": settings,
                "receipt": _lineage_receipt(contract, provider_request_id="receipt-0"),
            },
        }
        response_unapproved = seal_external_response(
            raw, request, self.config, execution_contract=contract
        )
        self.assertFalse(response_unapproved["headline_eligible"])
        response = authorize_external_responses([response_unapproved], contract)[0]
        validate_response(
            response, request, self.config,
            require_real=True, execution_contract=contract,
        )
        complete = self.matrix.copy()
        complete[0, 6] = 71.0
        metrics = evaluate_cached_responses(
            [request], [response], complete, self.config,
            require_complete=True, require_real=True, execution_contract=contract,
        )
        self.assertTrue(metrics["headline_eligible"])
        self.assertTrue(metrics["complete"])
        bad = dict(raw, response_text='{"q0": 101.0}')
        with self.assertRaisesRegex(ValueError, "outside normalized"):
            seal_external_response(bad, request, self.config)
        for invalid_cost in (True, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "cost amount"):
                seal_external_response(
                    dict(raw, cost={"currency": "USD", "amount": invalid_cost}),
                    request, self.config,
                )

    def test_real_pack_requires_one_fixed_identity_and_binds_execution_evidence(self) -> None:
        requests = [
            build_request(
                config=self.config, condition="zero_shot_named", fold_id=0,
                seed=42, fold=0, batch_index=index, train=self.matrix,
                cells=[(0, 6)], models=self.models, evaluations=self.evaluations,
                task_metadata=self.tasks,
            )
            for index in range(2)
        ]
        responses = []
        settings = {"temperature": 0, "seed": 42}
        contract = _execution_contract(
            count=2, provider="fixed-provider", model="fixed-model",
            version="2026-08-01", settings=settings,
        )
        for index, request in enumerate(requests):
            responses.append(seal_external_response({
                "request_id": request["request_id"],
                "backend_kind": "openai_compatible",
                "provider": "fixed-provider",
                "model": "fixed-model",
                "response_text": '{"q0": 71.0}',
                "execution_metadata": {
                    "model_version": "2026-08-01",
                    "settings": settings,
                    "receipt": _lineage_receipt(
                        contract, provider_request_id=f"receipt-{index}"
                    ),
                },
            }, request, self.config, execution_contract=contract))
        evidence = summarize_real_execution(responses)
        self.assertEqual(
            evidence["fixed_identity"],
            {
                "backend_kind": "openai_compatible",
                "provider": "fixed-provider",
                "model": "fixed-model",
            },
        )
        self.assertRegex(evidence["fixed_identity_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(evidence["model_version_reported_responses"], 2)
        self.assertEqual(evidence["settings_reported_responses"], 2)
        self.assertEqual(evidence["receipt_reported_responses"], 2)
        self.assertRegex(evidence["receipt_pack_sha256"], r"^[0-9a-f]{64}$")

        authorized = authorize_external_responses(responses, contract)

        mixed = json.loads(json.dumps(authorized))
        mixed[1]["provider"] = "different-provider"
        mixed[1].pop("response_sha256")
        from pathopress.llm_baseline import object_sha256
        mixed[1]["response_sha256"] = object_sha256(mixed[1])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluate_cached_responses(
                requests, mixed, self.matrix, self.config,
                require_complete=True, require_real=True, execution_contract=contract,
            )

        inconsistent = json.loads(json.dumps(responses))
        inconsistent[1]["execution_metadata_hashes"]["settings_sha256"] = "f" * 64
        inconsistent[1].pop("response_sha256")
        inconsistent[1]["response_sha256"] = object_sha256(inconsistent[1])
        with self.assertRaisesRegex(ValueError, "inconsistent settings_sha256"):
            summarize_real_execution(inconsistent)

    def test_headline_gate_rejects_partial_absent_and_wrong_consistent_evidence(self) -> None:
        requests = [
            build_request(
                config=self.config, condition="zero_shot_named", fold_id=0,
                seed=42, fold=0, batch_index=index, train=self.matrix,
                cells=[(0, 6)], models=self.models, evaluations=self.evaluations,
                task_metadata=self.tasks,
            )
            for index in range(2)
        ]
        approved_settings = {"temperature": 0, "max_tokens": 16384}
        contract = _execution_contract(
            count=2, provider="fixed-provider", model="fixed-model",
            version="fixed-snapshot", settings=approved_settings,
        )

        def make_rows(metadata_by_index, selected_contract=contract):
            return [
                seal_external_response({
                    "request_id": request["request_id"],
                    "backend_kind": "openai_compatible",
                    "provider": "fixed-provider",
                    "model": "fixed-model",
                    "response_text": '{"q0": 71.0}',
                    "execution_metadata": metadata_by_index[index],
                }, request, self.config, execution_contract=selected_contract)
                for index, request in enumerate(requests)
            ]

        complete = {
            "model_version": "fixed-snapshot",
            "settings": approved_settings,
            "receipt": _lineage_receipt(contract, provider_request_id="r"),
        }
        with self.assertRaisesRegex(ValueError, "settings do not match"):
            make_rows([{}, {}])

        with self.assertRaisesRegex(ValueError, "settings do not match"):
            make_rows([complete, {"model_version": "fixed-snapshot", "receipt": {"id": "r2"}}])

        with self.assertRaisesRegex(ValueError, "model_version"):
            make_rows([
                {"settings": approved_settings, "receipt": _lineage_receipt(contract, id="r1")},
                {"settings": approved_settings, "receipt": _lineage_receipt(contract, id="r2")},
            ])

        wrong_settings = {"temperature": 0, "max_tokens": 4096}
        with self.assertRaisesRegex(ValueError, "settings do not match"):
            make_rows([
                {"model_version": "fixed-snapshot", "settings": wrong_settings, "receipt": _lineage_receipt(contract, id="r1")},
                {"model_version": "fixed-snapshot", "settings": wrong_settings, "receipt": _lineage_receipt(contract, id="r2")},
            ])

        with self.assertRaisesRegex(ValueError, "approved snapshot"):
            make_rows([
                {"model_version": "other-snapshot", "settings": approved_settings, "receipt": _lineage_receipt(contract, id="r1")},
                {"model_version": "other-snapshot", "settings": approved_settings, "receipt": _lineage_receipt(contract, id="r2")},
            ])

        wrong_lineage = _lineage_receipt(contract, id="r1")
        wrong_lineage["transport_profile_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "transport_profile_sha256"):
            make_rows([
                {"model_version": "fixed-snapshot", "settings": approved_settings, "receipt": wrong_lineage},
                complete,
            ])

        alias_contract = _execution_contract(
            count=2, provider="fixed-provider", model="fixed-model",
            version=None, settings=approved_settings,
        )
        alias_contract["require_resolved_version_distinct_from_alias"] = True
        alias_contract.pop("execution_contract_sha256")
        alias_contract["execution_contract_sha256"] = object_sha256(alias_contract)
        with self.assertRaisesRegex(ValueError, "dated resolved"):
            make_rows([
                {"model_version": "fixed-model", "settings": approved_settings, "receipt": _lineage_receipt(alias_contract, id="r1")},
                {"model_version": "fixed-model", "settings": approved_settings, "receipt": _lineage_receipt(alias_contract, id="r2")},
            ], alias_contract)

    def test_generated_dry_run_artifacts_mark_every_real_condition_unrun(self) -> None:
        status = json.loads((ROOT / "experiments/llm_baseline/real_run_status.json").read_text())
        self.assertEqual(status["status"], "unrun")
        self.assertFalse(status["headline_eligible"])
        self.assertEqual(status["conditions"], {condition: "unrun" for condition in CONDITIONS})
        self.assertEqual(status["request_count"], 1990)
        self.assertEqual(status["target_prediction_count"], 81080)
        self.assertEqual(status["cost"]["status"], "unknown_until_external_execution")
        index = json.loads((ROOT / "experiments/llm_baseline/requests.jsonl").read_text())
        self.assertEqual(index["request_count"], 1990)
        self.assertEqual(index["shard_count"], 20)
        digest = hashlib.sha256()
        request_ids = set()
        request_counts = {condition: 0 for condition in CONDITIONS}
        target_counts = {condition: 0 for condition in CONDITIONS}
        for row in index["shards"]:
            shard = ROOT / "experiments/llm_baseline/requests" / row["path"]
            self.assertEqual(hashlib.sha256(shard.read_bytes()).hexdigest(), row["sha256"])
            payload = gzip.decompress(shard.read_bytes())
            digest.update(payload)
            for line in payload.decode("utf-8").splitlines():
                request = json.loads(line)
                self.assertNotIn(request["request_id"], request_ids)
                request_ids.add(request["request_id"])
                condition = request["condition"]
                request_counts[condition] += 1
                target_counts[condition] += len(request["targets"])
                if condition.startswith("zero_shot"):
                    self.assertLessEqual(len({target["model_index"] for target in request["targets"]}), 10)
                elif condition == "five_shot_named":
                    self.assertLessEqual(len(request["targets"]), 64)
                else:
                    self.assertLessEqual(len(request["targets"]), 16)
        self.assertEqual(digest.hexdigest(), index["canonical_uncompressed_sha256"])
        self.assertEqual(
            request_counts,
            {
                "zero_shot_named": 180,
                "zero_shot_blind": 180,
                "five_shot_named": 340,
                "five_shot_blind": 1290,
            },
        )
        self.assertEqual(target_counts, {condition: 20270 for condition in CONDITIONS})
        metrics = json.loads((ROOT / "experiments/llm_baseline_smoke/mock_metrics.json").read_text())
        self.assertEqual(metrics["result_status"], "mock_contract_validation_only")
        self.assertFalse(metrics["headline_eligible"])

    def test_response_schema_keeps_mock_valid_and_conditionally_hardens_real_rows(self) -> None:
        schema = json.loads(
            (ROOT / "experiments/llm_baseline/response.schema.json").read_text()
        )
        conditional_required = set(schema["allOf"][0]["then"]["required"])
        self.assertEqual(
            conditional_required,
            {"execution_metadata_hashes", "execution_approval_sha256", "execution_preflight_sha256", "execution_lineage_sha256"},
        )
        self.assertTrue(conditional_required.isdisjoint(schema["required"]))
        mock = json.loads(
            (ROOT / "experiments/llm_baseline_smoke/mock_responses.jsonl")
            .read_text().splitlines()[0]
        )
        self.assertTrue(set(schema["required"]).issubset(mock))
        self.assertEqual(mock["status"], "complete_mock_only")
        self.assertFalse(mock["headline_eligible"])
        self.assertTrue(conditional_required.isdisjoint(mock))

        # A headline real row triggers the conditional and cannot omit the
        # execution evidence/authorization bindings.
        hypothetical_real = dict(mock, status="complete_validated_real", headline_eligible=True)
        self.assertEqual(conditional_required - set(hypothetical_real), conditional_required)


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
            {"experiments": [{"name": "ok", "command": "python3 experiments/run_llm_baseline.py prepare", "inputs": ["data/scores.csv"], "external_calls": False}]},
        )
        self.assertEqual(ready[0]["status"], "ready")
        blocked = validate_experiment_set(
            ROOT,
            {"experiments": [{"name": "missing", "command": "python3 experiments/run_llm_baseline.py prepare", "inputs": ["does-not-exist"], "external_calls": False}]},
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
            "temporal_deployment", "llm_baseline_contract_only",
            "publication_tables", "publication_hero",
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
