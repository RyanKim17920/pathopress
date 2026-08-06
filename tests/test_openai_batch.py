import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from pathopress.llm_baseline import object_sha256
from pathopress.llm_preflight import build_preflight, load_request_pack, verify_pack_index
from pathopress.openai_batch import (
    BATCH_TRANSPORT_PROFILE,
    BUILTIN_SNAPSHOT,
    ENDPOINT,
    EXECUTION_SETTINGS,
    EXPECTED_MESSAGES_SHA256,
    EXPECTED_REQUEST_COUNT,
    OpenAIHTTP,
    approved_contract,
    atomic_json,
    convert_outputs,
    fetch_outputs,
    load_manifest,
    materialize,
    paid_gate,
    profile,
    record_batch,
    submit,
    terminal_ids,
    validate_remote_id,
)


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "experiments/llm_baseline"


class FakeAPI:
    def __init__(self, fail_create=False):
        self.uploads = 0
        self.creates = 0
        self.fail_create = fail_create

    def upload(self, path):
        value = self.uploads
        self.uploads += 1
        return {"id": f"file-{value:04d}", "purpose": "batch"}, f"req-upload-{value}"

    def create_batch(self, file_id, metadata):
        value = self.creates
        self.creates += 1
        if self.fail_create:
            raise RuntimeError("simulated ambiguous create")
        return {
            "id": f"batch-{value:04d}",
            "input_file_id": file_id,
            "endpoint": ENDPOINT,
            "completion_window": "24h",
            "status": "validating",
            "metadata": metadata,
        }, f"req-create-{value}"


class FakeFetchAPI:
    def __init__(self, state, run_dir):
        self.state = state
        self.payloads = {}
        self.by_batch = {}
        for shard in state["shards"]:
            index = shard["shard_index"]
            input_path = run_dir / f"batch-{index:04d}.jsonl"
            lines = [json.loads(line) for line in input_path.read_text().splitlines()]
            output_file_id = f"file-output-{index:04d}"
            output = []
            for offset, line in enumerate(lines):
                output.append({
                    "custom_id": line["custom_id"], "error": None,
                    "response": {
                        "status_code": 200,
                        "request_id": f"req-result-{index}-{offset}",
                        "body": {
                            "id": f"chatcmpl-{index}-{offset}",
                            "model": BUILTIN_SNAPSHOT,
                            "choices": [{"message": {"content": "{\"q0\":50}"}}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                        },
                    },
                })
            self.payloads[output_file_id] = "".join(json.dumps(row) + "\n" for row in output).encode()
            self.by_batch[shard["batch_id"]] = {
                "id": shard["batch_id"],
                "input_file_id": shard["input_file_id"],
                "endpoint": ENDPOINT,
                "completion_window": "24h",
                "metadata": {
                    "pathopress_pack": "e431cfc0fc0a14610fd26c0818ae1cad",
                    "pathopress_shard": f"{index:04d}",
                },
                "status": "completed",
                "output_file_id": output_file_id,
                "error_file_id": None,
                "request_counts": {"total": len(lines), "completed": len(lines), "failed": 0},
                "usage": {"input_tokens": 10 * len(lines), "output_tokens": 2 * len(lines), "total_tokens": 12 * len(lines)},
            }

    def batch(self, batch_id):
        return self.by_batch[batch_id], f"req-status-{batch_id}"

    def file_content(self, file_id):
        return self.payloads[file_id], f"req-file-{file_id}"


class OpenAIBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.temp.name)
        cls.snapshot = profile("chat_snapshot", BUILTIN_SNAPSHOT)
        cls.exact = profile("upstream_exact", None)
        cls.snapshot_manifest = materialize(PACK, cls.run_dir / "snapshot", cls.snapshot)
        cls.exact_manifest = materialize(PACK, cls.run_dir / "exact", cls.exact)

    def contract_stub(self, manifest, marker):
        return {
            "execution_contract_sha256": marker * 64,
            "approval_manifest_sha256": marker * 64,
            "preflight_sha256": marker * 64,
            "adapter_manifest_sha256": manifest["manifest_sha256"],
            "transport_profile_sha256": object_sha256(BATCH_TRANSPORT_PROFILE),
            "capacity_profile_sha256": marker * 64,
        }

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_profiles_are_explicit_and_separate(self):
        with self.assertRaises(ValueError):
            profile("chat_snapshot", None)
        with self.assertRaises(ValueError):
            profile("chat_snapshot", "gpt-5.5")
        with self.assertRaises(ValueError):
            profile("chat_snapshot", "gpt-5.5-2026-05-01")
        custom = profile("chat_custom_snapshot", "gpt-5.5-2026-05-01")
        self.assertEqual(custom.model, "gpt-5.5-2026-05-01")

    def test_exact_wire_contract_and_prompt_hash(self):
        self.assertEqual(self.exact_manifest["pack"]["request_count"], 1990)
        self.assertEqual(len(self.exact_manifest["shards"]), 20)
        rows = []
        for path in sorted((self.run_dir / "exact").glob("batch-*.jsonl")):
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
        self.assertEqual(len(rows), EXPECTED_REQUEST_COUNT)
        self.assertEqual(len({row["custom_id"] for row in rows}), EXPECTED_REQUEST_COUNT)
        self.assertEqual(object_sha256([row["body"]["messages"] for row in rows]), EXPECTED_MESSAGES_SHA256)
        self.assertTrue(all(row["method"] == "POST" and row["url"] == ENDPOINT for row in rows))
        self.assertTrue(all(row["body"]["model"] == "gpt-5.5" for row in rows))
        self.assertTrue(all(row["body"]["temperature"] == 0.0 for row in rows))
        self.assertTrue(all(row["body"]["max_tokens"] == 16384 for row in rows))

    def test_paid_gates(self):
        with self.assertRaises(PermissionError):
            paid_gate(self.snapshot, False, True, "secret")
        with self.assertRaises(PermissionError):
            paid_gate(self.snapshot, True, False, "secret")
        with self.assertRaises(PermissionError):
            paid_gate(self.snapshot, True, True, None)
        with self.assertRaisesRegex(PermissionError, "planning ceiling"):
            paid_gate(self.snapshot, True, True, "secret")
        self.assertEqual(
            paid_gate(
                self.snapshot, True, True, "secret",
                acknowledge_estimated_cost_uncertainty=True,
            ),
            "secret",
        )
        with self.assertRaises(PermissionError):
            paid_gate(self.exact, True, True, "secret")
        self.assertEqual(
            paid_gate(
                self.exact, True, True, "secret", acknowledge_mutable_alias=True,
                acknowledge_estimated_cost_uncertainty=True,
            ),
            "secret",
        )

    def test_manifest_rejects_prompt_tamper_even_if_rehashed(self):
        source = self.run_dir / "snapshot"
        target = self.run_dir / "tamper"
        target.mkdir()
        for path in source.iterdir():
            (target / path.name).write_bytes(path.read_bytes())
        batch = target / "batch-0000.jsonl"
        rows = [json.loads(line) for line in batch.read_text().splitlines()]
        rows[0]["body"]["messages"][0]["content"] += " tampered"
        payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()
        batch.write_bytes(payload)
        manifest_path = target / "preflight.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["shards"][0]["input_sha256"] = __import__("hashlib").sha256(payload).hexdigest()
        manifest["shards"][0]["bytes"] = len(payload)
        manifest.pop("manifest_sha256")
        manifest["manifest_sha256"] = object_sha256(manifest)
        atomic_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "prompts drifted"):
            load_manifest(manifest_path, self.snapshot)

    def test_submit_is_idempotent_after_saved_batch_ids(self):
        manifest_path = self.run_dir / "snapshot" / "preflight.json"
        state_path = self.run_dir / "submit-state.json"
        contract = self.contract_stub(self.snapshot_manifest, "a")
        first = FakeAPI()
        state = submit(manifest_path, state_path, self.snapshot, first, contract)
        self.assertEqual(first.uploads, 20)
        self.assertEqual(first.creates, 20)
        self.assertEqual(state["status"], "submitted")
        self.assertNotIn("secret", state_path.read_text())
        second = FakeAPI()
        submit(manifest_path, state_path, self.snapshot, second, contract)
        self.assertEqual((second.uploads, second.creates), (0, 0))

    def test_ambiguous_create_requires_manual_reconciliation(self):
        manifest_path = self.run_dir / "snapshot" / "preflight.json"
        state_path = self.run_dir / "ambiguous-state.json"
        contract = self.contract_stub(self.snapshot_manifest, "c")
        with self.assertRaisesRegex(RuntimeError, "simulated ambiguous"):
            submit(manifest_path, state_path, self.snapshot, FakeAPI(fail_create=True), contract)
        with self.assertRaisesRegex(RuntimeError, "ambiguous prior create"):
            submit(manifest_path, state_path, self.snapshot, FakeAPI(), contract)
        reconciled = record_batch(manifest_path, state_path, self.snapshot, 0, "batch-reconciled")
        self.assertEqual(reconciled["shards"][0]["batch_id"], "batch-reconciled")

    def test_output_conversion_uniform_evidence(self):
        path = self.run_dir / "two-output.jsonl"
        ids = {"request-a", "request-b"}
        values = []
        for index, custom_id in enumerate(sorted(ids)):
            values.append({
                "custom_id": custom_id,
                "error": None,
                "response": {
                    "status_code": 200,
                    "request_id": f"req-{index}",
                    "body": {
                        "id": f"chatcmpl-{index}",
                        "model": BUILTIN_SNAPSHOT,
                        "choices": [{"message": {"content": "{\"q0\":50}"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                    },
                },
            })
        path.write_text("".join(json.dumps(row) + "\n" for row in values))
        converted = convert_outputs(
            [path], self.snapshot, ids, object_sha256(EXECUTION_SETTINGS),
            {key: "batch-0000" for key in ids}, {
                "approval_manifest_sha256": "e" * 64,
                "preflight_sha256": "f" * 64,
                "execution_contract_sha256": "1" * 64,
                "adapter_manifest_sha256": "2" * 64,
                "transport_profile_sha256": "3" * 64,
                "capacity_profile_sha256": "4" * 64,
            },
        )
        self.assertEqual(len(converted), 2)
        self.assertTrue(all(row["model"] == BUILTIN_SNAPSHOT for row in converted))
        self.assertTrue(all(row["execution_metadata"]["settings"] == EXECUTION_SETTINGS for row in converted))
        self.assertTrue(all(row["execution_metadata"]["receipt"]["approval_manifest_sha256"] == "e" * 64 for row in converted))
        mixed = deepcopy(values)
        mixed[1]["response"]["body"]["model"] = "gpt-5.5-2026-05-01"
        path.write_text("".join(json.dumps(row) + "\n" for row in mixed))
        with self.assertRaisesRegex(ValueError, "immutable submitted snapshot"):
            convert_outputs([path], self.snapshot, ids, "f" * 64, {key: "batch-0000" for key in ids}, None)

    def test_mutable_alias_records_one_resolved_snapshot(self):
        path = self.run_dir / "alias-output.jsonl"
        value = {
            "custom_id": "request-a", "error": None,
            "response": {
                "status_code": 200, "request_id": "req-a",
                "body": {
                    "id": "chatcmpl-a", "model": BUILTIN_SNAPSHOT,
                    "choices": [{"message": {"content": "{\"q0\":50}"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                },
            },
        }
        path.write_text(json.dumps(value) + "\n")
        row = convert_outputs(
            [path], self.exact, {"request-a"}, object_sha256(EXECUTION_SETTINGS),
            {"request-a": "batch-0000"}, {"approval_manifest_sha256": "e" * 64},
        )[0]
        self.assertEqual(row["model"], "gpt-5.5")
        self.assertEqual(row["execution_metadata"]["model_version"], BUILTIN_SNAPSHOT)
        value["response"]["body"]["model"] = "gpt-5.5"
        path.write_text(json.dumps(value) + "\n")
        with self.assertRaisesRegex(ValueError, "dated resolved model"):
            convert_outputs([path], self.exact, {"request-a"}, "x", {"request-a": "batch-0000"}, None)

    def test_remote_ids_are_fail_closed(self):
        validate_remote_id("file-abc123", "file")
        validate_remote_id("batch_abc123", "batch")
        for value in ("../file-secret", "batch/a", "https://evil", "file-abc?x=1"):
            with self.assertRaises(ValueError):
                validate_remote_id(value, "file" if value.startswith("file") else "batch")

    def test_cross_shard_output_id_is_rejected(self):
        payload = json.dumps({
            "custom_id": "request-from-another-shard",
            "error": None,
            "response": {"status_code": 200},
        }).encode() + b"\n"
        with self.assertRaisesRegex(ValueError, "cross-shard"):
            terminal_ids(payload, {"expected-here"}, "output.jsonl", require_success=True)

    def test_fetch_rejects_stale_output_files(self):
        run_dir = self.run_dir / "stale-fetch"
        manifest = materialize(PACK, run_dir, self.snapshot)
        state_path = run_dir / "state.json"
        contract = self.contract_stub(manifest, "7")
        state = submit(run_dir / "preflight.json", state_path, self.snapshot, FakeAPI(), contract)
        (run_dir / "raw_output-9999.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "stale output/error"):
            fetch_outputs(
                run_dir / "preflight.json", state_path, self.snapshot,
                FakeFetchAPI(state, run_dir),
            )

    def test_current_not_ready_approval_rejected_before_api(self):
        with self.assertRaisesRegex(ValueError, "not been approved"):
            approved_contract(
                PACK / "execution_preflight.json",
                PACK / "execution_approval_manifest.json",
                self.exact,
            )

    def test_approved_fixture_materialize_contract_submit(self):
        fixture = self.run_dir / "approved-fixture"
        fixture.mkdir()
        run_manifest = materialize(PACK, fixture / "run", self.snapshot)
        config = json.loads((PACK / "config.json").read_text())
        index = json.loads((PACK / "requests.jsonl").read_text())
        requests, shards = load_request_pack(PACK / "requests", config)
        verify_pack_index(index, requests, shards, PACK / "requests")
        model_profile = json.loads((PACK / "profiles/reproducible_snapshot.model.json").read_text())
        capacity_profile = {
            "schema_version": 1,
            "provider": "OpenAI",
            "model_alias": "gpt-5.5",
            "model_snapshot": BUILTIN_SNAPSHOT,
            "account_scope_label": "nonsecret-test-project",
            "evidence": {
                "source": "synthetic unit-test capacity fixture",
                "retrieved_at": "2026-08-06",
                "active_credential_scope_attested": True,
            },
            "limits": {
                "context_window_tokens": 1000000000,
                "max_output_tokens": 1000000000,
                "max_input_tokens_per_request": 1000000000,
                "max_queued_input_tokens": 1000000000,
            },
        }
        pricing_profile = {
            "schema_version": 1,
            "profile_type": "user_supplied",
            "provider": "OpenAI",
            "model_alias": "gpt-5.5",
            "model_snapshot": BUILTIN_SNAPSHOT,
            "currency": "USD",
            "rates": {
                "batch": {
                    "input_per_million_tokens": 1.0,
                    "output_per_million_tokens": 1.0,
                }
            },
        }
        preflight, _, approval = build_preflight(
            requests=requests,
            shards=shards,
            pack_index=index,
            config=config,
            settings=EXECUTION_SETTINGS,
            transport_profile=BATCH_TRANSPORT_PROFILE,
            model_profile=model_profile,
            capacity_profile=capacity_profile,
            pricing_profile=pricing_profile,
            budget={"currency": "USD", "max_amount": 1000.0},
            adapter_manifest=run_manifest,
            tiktoken_encoding=None,
            acknowledge_cost_estimate_uncertainty=True,
        )
        self.assertEqual(preflight["capacity"]["status"], "pass")
        self.assertEqual(preflight["cost"]["pricing_rate_key"], "batch")
        self.assertEqual(preflight["binding"]["adapter_manifest_sha256"], run_manifest["manifest_sha256"])
        self.assertEqual(approval["approval_status"], "awaiting_human_approval")
        self.assertEqual(approval["blocking_reasons"], [])
        # The only mutation models the documented human authorization action.
        approval.pop("approval_manifest_sha256")
        approval["approval_status"] = "approved"
        approval["execution_authorized"] = True
        approval["approval_manifest_sha256"] = object_sha256(approval)
        preflight_path = fixture / "execution_preflight.json"
        approval_path = fixture / "execution_approval.json"
        atomic_json(preflight_path, preflight)
        atomic_json(approval_path, approval)
        contract = approved_contract(preflight_path, approval_path, self.snapshot)
        fake = FakeAPI()
        state = submit(
            fixture / "run/preflight.json", fixture / "run/state.json",
            self.snapshot, fake, contract,
        )
        self.assertEqual((fake.uploads, fake.creates), (20, 20))
        self.assertEqual(state["execution_contract"]["approval_manifest_sha256"], approval["approval_manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
