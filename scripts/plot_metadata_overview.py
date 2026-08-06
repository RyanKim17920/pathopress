#!/usr/bin/env python3
"""Plot cited model/task provenance and score-coverage metadata panels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.publication import metadata_panel_counts, read_csv, top_with_other  # noqa: E402


COLORS = ["#1368CE", "#D1495B", "#D89C1D", "#2A9D8F", "#7251B5", "#607D8B", "#ED6A5A", "#669BBC"]


def _barh(ax, counts: dict[str, int], keep: int, title: str, xlabel: str) -> None:
    labels, values = top_with_other(counts, keep)
    labels, values = labels[::-1], values[::-1]
    ax.barh(labels, values, color=COLORS[: len(values)][::-1])
    for y, value in enumerate(values):
        ax.text(value, y, f" {value}", va="center", fontsize=7.5)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=.18)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "data/scores.csv")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/tasks.csv")
    parser.add_argument("--models", type=Path, default=ROOT / "data/model_metadata.csv")
    parser.add_argument("--releases", type=Path, default=ROOT / "data/model_release_dates.csv")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "experiments/publication_metadata_summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/pathopress_metadata_overview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_objects = load_scores(args.scores)
    matrix, model_ids, evaluation_ids = filter_matrix(*make_matrix(score_objects))
    score_rows, task_rows = read_csv(args.scores), read_csv(args.tasks)
    model_rows, release_rows = read_csv(args.models), read_csv(args.releases)
    counts = metadata_panel_counts(
        score_rows, task_rows, release_rows, set(evaluation_ids), set(model_ids)
    )
    summary = {
        "schema_version": 1,
        "matrix": {"n_models": len(model_ids), "n_evaluations": len(evaluation_ids), "n_observed": counts["n_observed"]},
        "counts": counts,
        "semantics": {
            "release_timeline": "verified exact-model primary-source dates only",
            "quarterly_score_coverage": "retained observed score cells grouped by verified model release quarter; not publication-quarter benchmarking volume",
            "task_mix": "retained evaluation identities, not raw source columns",
            "observed_by_category": "retained observed model-evaluation cells",
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.7))
    ax = axes[0, 0]
    model_by_id = {row["model_id"]: row for row in model_rows}
    timeline = [row for row in release_rows if row["model_id"] in set(model_ids) and row["verification_status"] == "verified"]
    types = sorted({model_by_id.get(row["model_id"], {}).get("model_type", "unknown") for row in timeline})
    type_y = {value: index for index, value in enumerate(types)}
    for type_index, model_type in enumerate(types):
        rows = [row for row in timeline if model_by_id.get(row["model_id"], {}).get("model_type", "unknown") == model_type]
        xs = [date.fromisoformat(row["release_date"]) for row in rows]
        ax.scatter(xs, [type_y[model_type]] * len(rows), s=28, color=COLORS[type_index % len(COLORS)], label=model_type.replace("_", " "), alpha=.85)
    ax.set_yticks(range(len(types)), [value.replace("_", " ") for value in types])
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=35)
    type_counts = {model_type: sum(model_by_id.get(row["model_id"], {}).get("model_type", "unknown") == model_type for row in timeline) for model_type in types}
    type_count_text = " · ".join(f"{value} {key.replace('_encoder', '')}" for key, value in type_counts.items())
    ax.set_title(f"A  Verified releases (n={len(timeline)})\n{type_count_text}", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=.2)

    ax = axes[0, 1]
    quarters = sorted(counts["observed_score_quarters"])
    values = [counts["observed_score_quarters"][quarter] for quarter in quarters]
    ax.bar(range(len(quarters)), values, color="#1368CE")
    ax.set_xticks(range(len(quarters)), quarters, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("Observed score cells")
    ax.set_title("B  Score coverage by model release quarter", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=.18)

    _barh(axes[0, 2], counts["task_family"], 7, "C  Retained task-family mix", "Evaluation identities")
    _barh(axes[1, 0], counts["observed_family"], 7, "D  Observed cells by task family", "Observed model-evaluation cells")
    _barh(axes[1, 1], counts["suite_tasks"], 8, "E  Benchmark-suite provenance", "Evaluation identities")
    _barh(axes[1, 2], counts["task_source_domains"], 6, "F  Primary task-metadata sources", "Evaluation identities")
    audit_text = ", ".join(f"{key}: {value}" for key, value in counts["score_audit_status"].items())

    fig.suptitle("PathoPress evidence base: models, tasks, coverage, and provenance", fontsize=16, fontweight="bold")
    fig.text(.5, .012, f"All dates and tasks retain primary-source URLs; missing release metadata is not imputed. Retained score evidence — {audit_text}.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .035, 1, .96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    print(f"wrote {args.summary_output} and {args.output}.png/.pdf")


if __name__ == "__main__":
    main()
