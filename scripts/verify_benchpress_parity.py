#!/usr/bin/env python3
"""Compare PathoPress predictions with BenchPress's standalone predictor."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathopress.completion import complete  # noqa: E402
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchpress_source", type=Path)
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "data" / "scores.csv")
    parser.add_argument("--rank", type=int, default=2)
    args = parser.parse_args()
    predictor_path = args.benchpress_source / "website" / "add-model" / "predictor.py"
    specification = importlib.util.spec_from_file_location("benchpress_predictor", predictor_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import {predictor_path}")
    reference = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(reference)

    scores = load_scores(args.scores)
    matrix, models, evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(matrix, models, evaluations)
    ours = complete(matrix, rank=args.rank)
    theirs = reference.predict_benchpress_scores(matrix, rank=args.rank)
    difference = np.abs(ours - theirs)
    print(f"matrix={len(models)}x{len(evaluations)}")
    print(f"rank={args.rank}")
    print(f"max_absolute_difference={float(np.max(difference)):.12g}")
    print(f"mean_absolute_difference={float(np.mean(difference)):.12g}")
    if not np.array_equal(ours, theirs):
        raise SystemExit("predictors differ")


if __name__ == "__main__":
    main()
