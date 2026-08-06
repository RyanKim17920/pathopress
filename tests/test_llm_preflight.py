from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path

from pathopress.llm_baseline import make_config, object_sha256
from pathopress.llm_preflight import (
    ESTIMATOR_SPEC,
    build_approved_execution_contract,
    approve_execution_manifest,
    build_preflight,
    estimate_requests,
    load_request_pack,
    validate_model_profile,
    validate_profile_settings_contract,
    validate_pricing_profile,
    validate_transport_profile,
    verify_pack_index,
    _batch_transport_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def _settings(*, max_output: int = 100) -> dict:
    return {
        "schema_version": 1,
        "max_output_tokens_per_request": max_output,
        "temperature": 0.0,
        "endpoint_api": "chat_completions",
        "provider_parameter_name": "max_tokens",
    }


def _transport(*, batch: bool = False, limits: dict | None = None) -> dict:
    if batch:
        return {
            "schema_version": 1,
            "transport_kind": "openai_batch",
            "protocol_compatibility": "batch_transport_adaptation_not_upstream_exact",
            "pricing_rate_key": "batch",
            "completion_window": "24h",
            "expected_input_file_count": 1,
            "limits": limits or {},
        }
    return {
        "schema_version": 1,
        "transport_kind": "online_chat_completions",
        "protocol_compatibility": "literal_upstream_online_transport",
        "pricing_rate_key": "online",
        "limits": limits or {},
    }


def _model(*, snapshot: str | None = "model-snapshot", limits: dict | None = None) -> dict:
    return {
        "schema_version": 1,
        "profile_type": "user_supplied",
        "protocol_compatibility": "controlled_snapshot_substitution_not_upstream_exact",
        "provider": "test-provider",
        "model_alias": "model-alias",
        "model_snapshot": snapshot,
        "endpoint": {"api": "chat_completions", "base_url": None},
        "limits": limits or {},
    }


def _capacity(*, limits: dict) -> dict:
    return {
        "schema_version": 1,
        "provider": "test-provider",
        "model_alias": "model-alias",
        "model_snapshot": "model-snapshot",
        "account_scope_label": "test-project-nonsecret",
        "evidence": {
            "source": "user-supplied account limits fixture",
            "retrieved_at": "2026-08-06",
            "active_credential_scope_attested": True,
        },
        "limits": limits,
    }


class LlmPreflightUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests = [
            {
                "request_id": "r0",
                "request_sha256": "a" * 64,
                "condition": "zero_shot_named",
                "fold_id": 0,
                "batch_index": 0,
                "targets": [{"query_id": "q0"}, {"query_id": "q1"}],
                "messages": [
                    {"role": "system", "content": "predict pathology scores"},
                    {"role": "user", "content": "q0 and q1"},
                ],
            },
            {
                "request_id": "r1",
                "request_sha256": "b" * 64,
                "condition": "five_shot_blind",
                "fold_id": 0,
                "batch_index": 0,
                "targets": [{"query_id": "q0"}],
                "messages": [{"role": "user", "content": "estimate q0"}],
            },
        ]
        self.config = make_config(
            scores_sha256="c" * 64,
            folds_sha256="d" * 64,
            models=["m"],
            evaluations=["e"],
            fold_ids=[0],
            cell_limit=1,
        )
        self.shards = [
            {
                "shard_index": 0,
                "path": "requests-0000.jsonl.gz",
                "request_count": 2,
                "compressed_bytes": 100,
                "uncompressed_bytes": 500,
                "sha256": "e" * 64,
                "canonical_uncompressed_sha256": "f" * 64,
                "request_ids": ["r0", "r1"],
            }
        ]
        self.index = {
            "pack_sha256": "1" * 64,
            "canonical_uncompressed_sha256": "2" * 64,
        }

    def test_model_agnostic_scenarios_and_output_ceiling_are_deterministic(self) -> None:
        rows = estimate_requests(self.requests, _settings(max_output=321))
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertLessEqual(row["input_tokens"]["best"], row["input_tokens"]["base"])
            self.assertLessEqual(row["input_tokens"]["base"], row["input_tokens"]["worst"])
            self.assertEqual(row["output_tokens"]["worst"], 321)
            self.assertIsNone(row["tiktoken"])
        self.assertRegex(object_sha256(ESTIMATOR_SPEC), r"^[0-9a-f]{64}$")

    def test_optional_tiktoken_requires_an_explicit_encoding(self) -> None:
        try:
            rows = estimate_requests(self.requests, _settings(), tiktoken_encoding="cl100k_base")
        except ValueError as error:
            if "not installed" in str(error):
                self.skipTest("optional tiktoken package is not installed")
            raise
        self.assertEqual(rows[0]["tiktoken"]["encoding"], "cl100k_base")
        self.assertGreater(rows[0]["tiktoken"]["input_content_tokens"], 0)
        self.assertIn("overhead excluded", rows[0]["tiktoken"]["scope"])

    def test_cost_uses_only_exactly_matching_explicit_prices(self) -> None:
        model = _model(
            limits={
                "context_window_tokens": 1_000_000,
                "max_output_tokens": 1000,
                "max_input_tokens_per_request": 1_000_000,
            }
        )
        pricing = {
            "schema_version": 1,
            "profile_type": "user_supplied",
            "provider": "test-provider",
            "model_alias": "model-alias",
            "model_snapshot": "model-snapshot",
            "currency": "USD",
            "rates": {
                "online": {
                    "input_per_million_tokens": 2.0,
                    "output_per_million_tokens": 8.0,
                }
            },
        }
        preflight, details, approval = build_preflight(
            requests=self.requests,
            shards=self.shards,
            pack_index=self.index,
            config=self.config,
            settings=_settings(),
            transport_profile=_transport(),
            model_profile=model,
            capacity_profile=_capacity(limits={
                "context_window_tokens": 1_000_000,
                "max_output_tokens": 1000,
                "max_input_tokens_per_request": 1_000_000,
                "max_queued_input_tokens": 1_000_000,
            }),
            pricing_profile=pricing,
            budget={"currency": "USD", "max_amount": 100.0},
            acknowledge_cost_estimate_uncertainty=True,
            tiktoken_encoding=None,
        )
        expected = (
            preflight["totals"]["input_tokens"]["base"] * 2.0
            + preflight["totals"]["output_tokens"]["base"] * 8.0
        ) / 1_000_000
        self.assertAlmostEqual(preflight["cost"]["scenarios"]["base"]["total"], expected)
        self.assertEqual(preflight["cost"]["status"], "estimated_from_explicit_profile")
        self.assertEqual(approval["approval_status"], "awaiting_human_approval")
        self.assertFalse(approval["execution_authorized"])
        self.assertEqual(preflight["binding"]["request_estimates_canonical_sha256"], object_sha256(details))

        approved = approve_execution_manifest(
            preflight, approval,
            human_review_complete=True,
            acknowledge_estimated_cost_uncertainty=True,
        )
        contract = build_approved_execution_contract(preflight, approved)
        self.assertEqual(contract["required_settings_sha256"], object_sha256(_settings()))
        self.assertEqual(contract["expected_response_count"], 2)

        over_budget = json.loads(json.dumps(approved))
        over_budget.pop("approval_manifest_sha256")
        over_budget["authorized_planning_cost_ceiling"]["max_amount"] = 0.0
        over_budget["approval_manifest_sha256"] = object_sha256(over_budget)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_approved_execution_contract(preflight, over_budget)
        wrong_currency = json.loads(json.dumps(approved))
        wrong_currency.pop("approval_manifest_sha256")
        wrong_currency["authorized_planning_cost_ceiling"]["currency"] = "EUR"
        wrong_currency["approval_manifest_sha256"] = object_sha256(wrong_currency)
        with self.assertRaisesRegex(ValueError, "currency"):
            build_approved_execution_contract(preflight, wrong_currency)

        mismatched = json.loads(json.dumps(pricing))
        mismatched["model_snapshot"] = "another-snapshot"
        with self.assertRaisesRegex(ValueError, "model_snapshot"):
            validate_pricing_profile(mismatched, model)

    def test_capacity_and_batch_limits_fail_closed(self) -> None:
        model = _model(
            limits={
                "context_window_tokens": 1,
                "max_output_tokens": 1,
                "max_input_tokens_per_request": 1,
            }
        )
        preflight, _, approval = build_preflight(
            requests=self.requests,
            shards=self.shards,
            pack_index=self.index,
            config=self.config,
            settings=_settings(),
            transport_profile=_transport(batch=True, limits={
                "max_requests_per_batch_file": 1,
                "max_batch_file_bytes": 1,
            }),
            model_profile=model,
            capacity_profile=_capacity(limits={
                "context_window_tokens": 1,
                "max_output_tokens": 1,
                "max_input_tokens_per_request": 1,
                "max_queued_input_tokens": 1,
            }),
            pricing_profile=None,
            budget=None,
            tiktoken_encoding=None,
        )
        self.assertEqual(preflight["capacity"]["status"], "fail")
        self.assertGreater(preflight["capacity"]["failure_count"], 0)
        self.assertEqual(preflight["capacity"]["batch_queue_feasibility"]["status"], "fail")
        self.assertEqual(approval["approval_status"], "not_ready")
        with self.assertRaisesRegex(ValueError, "not been approved"):
            build_approved_execution_contract(preflight, approval)

    def test_generated_batch_chain_reaches_human_approval_without_preflight_mutation(self) -> None:
        model = _model()
        settings = _settings()
        transport = _transport(batch=True, limits={
            "max_requests_per_batch_file": 100,
            "max_batch_file_bytes": 1_000_000,
        })
        capacity = _capacity(limits={
            "context_window_tokens": 1_000_000,
            "max_output_tokens": 1000,
            "max_input_tokens_per_request": 1_000_000,
            "max_queued_input_tokens": 1_000_000,
        })
        payload = _batch_transport_payload(self.requests, settings, model)
        adapter_plan = [{
            "shard_index": 0,
            "input_sha256": __import__("hashlib").sha256(payload).hexdigest(),
            "bytes": len(payload),
            "request_count": 2,
        }]
        adapter = {
            "schema_version": 1,
            "kind": "pathopress_openai_batch_preflight",
            "pack": {
                "config_sha256": self.config["config_sha256"],
                "pack_sha256": self.index["pack_sha256"],
                "canonical_uncompressed_sha256": self.index["canonical_uncompressed_sha256"],
                "request_count": 2,
                "shard_count": 1,
            },
            "execution_settings_sha256": object_sha256(settings),
            "transport_profile_sha256": object_sha256(transport),
            "materialized_transport_plan_sha256": object_sha256(adapter_plan),
        }
        adapter["manifest_sha256"] = object_sha256(adapter)
        pricing = {
            "schema_version": 1,
            "profile_type": "user_supplied",
            "provider": "test-provider",
            "model_alias": "model-alias",
            "model_snapshot": "model-snapshot",
            "currency": "USD",
            "rates": {"batch": {
                "input_per_million_tokens": 2.0,
                "output_per_million_tokens": 8.0,
            }},
        }
        preflight, _, approval = build_preflight(
            requests=self.requests, shards=self.shards, pack_index=self.index,
            config=self.config, settings=settings, transport_profile=transport,
            model_profile=model, capacity_profile=capacity,
            pricing_profile=pricing, budget={"currency": "USD", "max_amount": 10.0},
            adapter_manifest=adapter,
            acknowledge_cost_estimate_uncertainty=True,
        )
        self.assertEqual(preflight["capacity"]["status"], "pass")
        self.assertEqual(preflight["cost"]["pricing_rate_key"], "batch")
        self.assertEqual(approval["approval_status"], "awaiting_human_approval")
        self.assertEqual(approval["blocking_reasons"], [])
        approved = approve_execution_manifest(
            preflight, approval,
            human_review_complete=True,
            acknowledge_estimated_cost_uncertainty=True,
        )
        contract = build_approved_execution_contract(preflight, approved)
        self.assertEqual(contract["transport_kind"], "openai_batch")
        self.assertEqual(contract["adapter_manifest_sha256"], adapter["manifest_sha256"])
        self.assertEqual(contract["capacity_profile_sha256"], object_sha256(capacity))

    def test_alias_and_snapshot_profiles_cannot_be_conflated(self) -> None:
        upstream = json.loads(
            (ROOT / "experiments/llm_baseline/profiles/upstream_contract.model.json").read_text()
        )
        snapshot = json.loads(
            (ROOT / "experiments/llm_baseline/profiles/reproducible_snapshot.model.json").read_text()
        )
        validate_model_profile(upstream)
        validate_model_profile(snapshot)
        self.assertIsNone(upstream["model_snapshot"])
        self.assertEqual(upstream["protocol_compatibility"], "literal_upstream_contract_mutable_alias")
        self.assertEqual(snapshot["model_snapshot"], "gpt-5.5-2026-04-23")
        self.assertIn("not_upstream_exact", snapshot["protocol_compatibility"])

        exact_settings = json.loads(
            (ROOT / "experiments/llm_baseline/profiles/upstream_contract.settings.json").read_text()
        )
        validate_profile_settings_contract(upstream, exact_settings)
        for key, bad_value in (
            ("top_p", 0.9),
            ("temperature", True),
            ("temperature", float("nan")),
            ("temperature", 0.1),
            ("max_output_tokens_per_request", 4096),
            ("provider_parameter_name", "max_completion_tokens"),
            ("endpoint_api", "responses"),
        ):
            changed = json.loads(json.dumps(exact_settings))
            changed[key] = bad_value
            with self.assertRaisesRegex(ValueError, "upstream_contract|endpoint_api|temperature"):
                validate_profile_settings_contract(upstream, changed)
        validate_transport_profile(json.loads(
            (ROOT / "experiments/llm_baseline/profiles/upstream_online.transport.json").read_text()
        ))
        batch_transport = json.loads(
            (ROOT / "experiments/llm_baseline/profiles/openai_batch_24h.transport.json").read_text()
        )
        validate_transport_profile(batch_transport)
        self.assertIn("not_upstream_exact", batch_transport["protocol_compatibility"])


class LlmPreflightReleaseTests(unittest.TestCase):
    def test_full_pack_preflight_is_hash_bound_and_unapproved(self) -> None:
        directory = ROOT / "experiments/llm_baseline"
        config = json.loads((directory / "config.json").read_text())
        index = json.loads((directory / "requests.jsonl").read_text())
        requests, shards = load_request_pack(directory / "requests", config)
        verify_pack_index(index, requests, shards, directory / "requests")
        self.assertEqual(len(requests), 1990)
        self.assertEqual(sum(len(row["targets"]) for row in requests), 81080)

        preflight = json.loads((directory / "execution_preflight.json").read_text())
        approval = json.loads((directory / "execution_approval_manifest.json").read_text())
        self.assertEqual(preflight["request_pack"]["pack_sha256"], index["pack_sha256"])
        self.assertEqual(preflight["request_pack"]["request_count"], 1990)
        self.assertEqual(preflight["request_pack"]["target_prediction_count"], 81080)
        self.assertEqual(set(preflight["by_condition"]), {
            "zero_shot_named", "zero_shot_blind", "five_shot_named", "five_shot_blind"
        })
        self.assertEqual(len(preflight["by_shard"]), 20)
        self.assertEqual(preflight["execution_settings"]["max_output_tokens_per_request"], 16384)
        self.assertEqual(preflight["model_selection"]["identity"]["model_alias"], "gpt-5.5")
        self.assertEqual(preflight["model_selection"]["identity"]["model_snapshot"], "gpt-5.5-2026-04-23")
        self.assertEqual(preflight["transport_profile"]["transport_kind"], "openai_batch")
        self.assertEqual(preflight["transport_profile"]["completion_window"], "24h")
        self.assertEqual(preflight["transport_profile"]["expected_input_file_count"], 20)
        self.assertEqual(preflight["cost"].get("pricing_rate_key"), None)
        self.assertEqual(preflight["cost"]["status"], "unavailable_no_explicit_pricing_profile")
        self.assertFalse(approval["execution_authorized"])
        self.assertEqual(approval["approval_status"], "not_ready")
        with self.assertRaisesRegex(ValueError, "not been approved"):
            build_approved_execution_contract(preflight, approval)
        evidence = approval["execution_evidence_completeness"]
        self.assertEqual(evidence["expected_response_count"], 1990)
        self.assertEqual(evidence["model_version_evidence_required_responses"], 1990)
        self.assertEqual(evidence["settings_evidence_required_responses"], 1990)
        self.assertEqual(evidence["provider_receipt_required_responses"], 1990)
        self.assertEqual(evidence["required_settings_sha256"], preflight["binding"]["execution_settings_sha256"])
        self.assertEqual(evidence["required_transport_profile_sha256"], preflight["binding"]["transport_profile_sha256"])
        self.assertIsNone(approval["authorized_planning_cost_ceiling"])
        self.assertFalse(approval["cost_estimate_uncertainty_acknowledged"])
        self.assertEqual(approval["preflight_sha256"], preflight["preflight_sha256"])
        unsigned_preflight = dict(preflight)
        supplied = unsigned_preflight.pop("preflight_sha256")
        self.assertEqual(supplied, object_sha256(unsigned_preflight))
        unsigned_approval = dict(approval)
        supplied = unsigned_approval.pop("approval_manifest_sha256")
        self.assertEqual(supplied, object_sha256(unsigned_approval))
        detail = directory / "execution_preflight_requests.jsonl.gz"
        with gzip.open(detail, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(rows), 1990)
        self.assertEqual(object_sha256(rows), preflight["binding"]["request_estimates_canonical_sha256"])


if __name__ == "__main__":
    unittest.main()
