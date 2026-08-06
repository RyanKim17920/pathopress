#!/usr/bin/env python3
"""Validate and dry-run an HF publication; network upload is doubly opt-in."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.hf_publication import publish_hf_export  # noqa: E402
from pathopress.public_data import DEFAULT_HF_DATASET_ID  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=ROOT / "exports/pathopress_public")
    parser.add_argument("--repo-id", default=DEFAULT_HF_DATASET_ID)
    parser.add_argument("--upload", action="store_true", help="Request a network upload; default is local dry-run.")
    parser.add_argument(
        "--authorize-upload", action="store_true",
        help="Confirm explicit user authorization; required together with --upload and HF_TOKEN.",
    )
    parser.add_argument("--commit-message", default="Update PathoPress score matrix export")
    args = parser.parse_args()
    result = publish_hf_export(
        args.export_dir, repo_id=args.repo_id, upload=args.upload,
        authorized=args.authorize_upload, token=os.environ.get("HF_TOKEN"),
        commit_message=args.commit_message,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
