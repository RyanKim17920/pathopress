#!/usr/bin/env python3
"""Materialize or explicitly execute the frozen PathoPress OpenAI Batch pack."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.openai_batch import (  # noqa: E402
    OpenAIHTTP,
    approved_contract,
    cancel_all,
    fetch_outputs,
    materialize,
    online_read_gate,
    paid_gate,
    profile,
    record_batch,
    refresh,
    submit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="preflight", choices=("preflight", "submit", "status", "fetch", "cancel", "record-batch"))
    parser.add_argument("--profile", choices=("chat_snapshot", "chat_custom_snapshot", "upstream_exact"), default="chat_snapshot")
    parser.add_argument("--model", help="required dated model snapshot for chat_snapshot; no alias default")
    parser.add_argument("--pack-dir", type=Path, default=ROOT / "experiments/llm_baseline")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "experiments/llm_baseline/openai_batch")
    parser.add_argument("--submit", action="store_true", help="first paid/external-write gate")
    parser.add_argument("--authorize-paid-run", action="store_true", help="second paid/external-write gate")
    parser.add_argument("--online", action="store_true", help="permit authenticated read-only API calls")
    parser.add_argument("--confirm-cancel", action="store_true", help="required in addition to all paid write gates")
    parser.add_argument("--acknowledge-mutable-alias", action="store_true")
    parser.add_argument("--acknowledge-custom-snapshot", action="store_true")
    parser.add_argument(
        "--acknowledge-estimated-cost-uncertainty", action="store_true",
        help="acknowledge that the approved estimate is not a provider-enforced billing cap",
    )
    parser.add_argument("--approval-manifest", type=Path, default=ROOT / "experiments/llm_baseline/execution_approval_manifest.json")
    parser.add_argument("--execution-preflight", type=Path, default=ROOT / "experiments/llm_baseline/execution_preflight.json")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--batch-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = profile(args.profile, args.model)
    manifest_path = args.run_dir / "preflight.json"
    state_path = args.run_dir / "state.json"
    if args.mode == "preflight":
        result = materialize(args.pack_dir, args.run_dir, selected)
    elif args.mode == "submit":
        key = paid_gate(
            selected, args.submit, args.authorize_paid_run, os.environ.get("OPENAI_API_KEY"),
            acknowledge_mutable_alias=args.acknowledge_mutable_alias,
            acknowledge_custom_snapshot=args.acknowledge_custom_snapshot,
            acknowledge_estimated_cost_uncertainty=args.acknowledge_estimated_cost_uncertainty,
        )
        contract = approved_contract(args.execution_preflight, args.approval_manifest, selected)
        result = submit(manifest_path, state_path, selected, OpenAIHTTP(key), contract)
    elif args.mode in {"status", "fetch"}:
        key = online_read_gate(args.online, os.environ.get("OPENAI_API_KEY"))
        client = OpenAIHTTP(key)
        result = refresh(manifest_path, state_path, selected, client) if args.mode == "status" else fetch_outputs(manifest_path, state_path, selected, client)
    elif args.mode == "cancel":
        if not args.confirm_cancel:
            raise PermissionError("cancel also requires --confirm-cancel")
        key = paid_gate(
            selected, args.submit, args.authorize_paid_run, os.environ.get("OPENAI_API_KEY"),
            acknowledge_mutable_alias=args.acknowledge_mutable_alias,
            acknowledge_custom_snapshot=args.acknowledge_custom_snapshot,
            acknowledge_estimated_cost_uncertainty=args.acknowledge_estimated_cost_uncertainty,
        )
        result = cancel_all(manifest_path, state_path, selected, OpenAIHTTP(key))
    else:
        if args.shard_index is None or args.batch_id is None:
            raise ValueError("record-batch requires --shard-index and --batch-id")
        result = record_batch(manifest_path, state_path, selected, args.shard_index, args.batch_id)
    print(json.dumps({
        "status": result.get("status"),
        "profile": selected.name,
        "request_count": result.get("pack", {}).get("request_count"),
        "api_calls_made": result.get("api_calls_made"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
