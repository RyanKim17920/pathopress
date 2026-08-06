#!/usr/bin/env python3
"""Build and validate the local Hugging Face publication directory; no upload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.hf_publication import validate_hf_export  # noqa: E402
from pathopress.public_data import (  # noqa: E402
    DEFAULT_HF_DATASET_ID,
    build_public_export,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "data/scores.csv")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/tasks.csv")
    parser.add_argument("--suites", type=Path, default=ROOT / "data/suites.csv")
    parser.add_argument("--provenance", type=Path, default=ROOT / "data/provenance.json")
    parser.add_argument("--models", type=Path, default=ROOT / "data/model_metadata.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "exports/pathopress_public")
    parser.add_argument("--repo-id", default=DEFAULT_HF_DATASET_ID)
    parser.add_argument("--parquet", choices=("auto", "yes", "no"), default="auto")
    args = parser.parse_args()
    build_public_export(
        scores_path=args.scores, tasks_path=args.tasks, suites_path=args.suites,
        provenance_path=args.provenance, model_metadata_path=args.models,
        out_dir=args.out_dir, parquet_mode=args.parquet, dataset_id=args.repo_id,
    )
    print(json.dumps(validate_hf_export(args.out_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
