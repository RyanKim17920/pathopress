#!/usr/bin/env python3
"""Build the citation-backed canonical model metadata table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.model_metadata import build_model_metadata, write_model_metadata  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=ROOT / "data" / "scores.csv")
    parser.add_argument("--model-sources", type=Path, default=ROOT / "data" / "model_sources.csv")
    parser.add_argument("--release-dates", type=Path, default=ROOT / "data" / "model_release_dates.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "model_metadata.csv")
    args = parser.parse_args()
    matrix, models, evaluations = make_matrix(load_scores(args.scores))
    _, models, _ = filter_matrix(matrix, models, evaluations)
    rows = build_model_metadata(
        models,
        model_sources_path=args.model_sources,
        release_dates_path=args.release_dates,
    )
    write_model_metadata(args.output, rows)
    n_params = sum(bool(row["parameter_count"]) for row in rows)
    print(f"wrote {args.output}: models={len(rows)} parameter_counts={n_params}")


if __name__ == "__main__":
    main()
