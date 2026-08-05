"""Command line entry point for auditing the seed matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .completion import validate
from .matrix import filter_matrix, load_scores, make_matrix


def _matrix(args: argparse.Namespace) -> tuple[np.ndarray, list[str], list[str]]:
    scores = load_scores(args.scores)
    matrix, models, evaluations = make_matrix(scores)
    return filter_matrix(
        matrix,
        models,
        evaluations,
        min_scores_per_model=args.min_scores_per_model,
        min_models_per_evaluation=args.min_models_per_evaluation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a pathology benchmark score matrix")
    parser.add_argument("command", choices=("audit", "validate"))
    parser.add_argument("--scores", type=Path, default=Path("data/scores.csv"))
    parser.add_argument("--min-scores-per-model", type=int, default=3)
    parser.add_argument("--min-models-per-evaluation", type=int, default=5)
    args = parser.parse_args()
    matrix, models, evaluations = _matrix(args)
    observed = int(np.sum(np.isfinite(matrix)))
    total = int(matrix.size)
    print(f"matrix={len(models)} models x {len(evaluations)} evaluations")
    density = observed / total if total else 0.0
    print(f"observed={observed}/{total} ({density:.1%})")
    if args.command == "validate":
        if total == 0:
            parser.error("no matrix cells remain after support filtering")
        result = validate(matrix)
        print(f"held_out_predictions={result.n_predictions}")
        print(f"median_absolute_error={result.median_absolute_error:.3f} points")
        print(f"mean_absolute_error={result.mean_absolute_error:.3f} points")


if __name__ == "__main__":
    main()
