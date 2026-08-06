#!/usr/bin/env python3
"""Freeze the canonical PathoPress analysis matrix and BenchPress fold protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.artifacts import sha256_file, write_fold_artifact  # noqa: E402
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=ROOT / "data" / "scores.csv")
    parser.add_argument("--matrix", type=Path, default=ROOT / "experiments" / "analysis_matrix.npz")
    parser.add_argument(
        "--folds", type=Path, default=ROOT / "experiments" / "folds_s10_f3_bs42.json"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "experiments" / "shared_artifacts_manifest.json"
    )
    parser.add_argument("--min-scores-per-model", type=int, default=3)
    parser.add_argument("--min-models-per-evaluation", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores)
    matrix, models, evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(
        matrix,
        models,
        evaluations,
        min_scores_per_model=args.min_scores_per_model,
        min_models_per_evaluation=args.min_models_per_evaluation,
    )
    args.matrix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.matrix,
        matrix=matrix,
        models=np.asarray(models),
        evaluations=np.asarray(evaluations),
    )
    write_fold_artifact(args.folds, matrix, models, evaluations)
    observed = np.argwhere(np.isfinite(matrix))
    observed_digest = hashlib.sha256(
        b"".join(
            f"{models[i]}\t{evaluations[j]}\t{matrix[i, j]:.12g}\n".encode()
            for i, j in observed
        )
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "description": "Immutable identity and fold contract shared by PathoPress experiments.",
        "filter": {
            "verified_only": True,
            "min_scores_per_model": args.min_scores_per_model,
            "min_models_per_evaluation": args.min_models_per_evaluation,
        },
        "matrix": {
            "path": str(args.matrix.relative_to(ROOT)),
            "sha256": sha256_file(args.matrix),
            "shape": list(matrix.shape),
            "n_observed": int(np.isfinite(matrix).sum()),
            "density": float(np.isfinite(matrix).mean()),
            "observed_triples_sha256": observed_digest,
        },
        "folds": {
            "path": str(args.folds.relative_to(ROOT)),
            "sha256": sha256_file(args.folds),
            "n_seeds": 10,
            "n_folds": 3,
            "base_seed": 42,
            "n_records": 30,
        },
        "source": {
            "scores_path": str(args.scores.relative_to(ROOT)),
            "scores_sha256": sha256_file(args.scores),
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
