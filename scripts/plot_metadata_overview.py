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

PANEL_NAMES = (
    "pathopress_releases",
    "pathopress_coverage",
    "pathopress_evaluation_mix",
    "pathopress_observed_cells_by_family",
    "pathopress_source_provenance",
)


def _save_pair(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(base.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument(
        "--panel-output-dir", type=Path, default=ROOT / "figures",
        help="Directory for five separately named BenchPress-analogue panels",
    )
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
            "coverage_by_release_quarter": "median and mean retained evaluations per verified model, matching the upstream panel denominator",
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
    plt.close(fig)

    # Five separately named panels mirror the upstream score-matrix figure
    # inventory while retaining pathology-native task/source categories.
    quarters = list(counts["release_quarters"])
    fig, ax = plt.subplots(figsize=(5.1, 3.7))
    ax.bar(range(len(quarters)), [counts["release_quarters"][q] for q in quarters], color="#D1495B")
    ax.set_xticks(range(len(quarters)), quarters, rotation=45, ha="right")
    ax.set_ylabel("Verified model releases")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    _save_pair(fig, args.panel_output_dir / PANEL_NAMES[0])

    coverage = counts["coverage_by_release_quarter"]
    coverage_quarters = list(coverage)
    fig, ax = plt.subplots(figsize=(5.1, 3.7))
    ax.plot(range(len(coverage_quarters)), [coverage[q]["median"] for q in coverage_quarters], "o-", color="#D1495B", linewidth=2.3, label="median")
    ax.plot(range(len(coverage_quarters)), [coverage[q]["mean"] for q in coverage_quarters], "s--", color="#1368CE", linewidth=1.8, label="mean")
    ax.set_xticks(range(len(coverage_quarters)), coverage_quarters, rotation=45, ha="right")
    ax.set_ylabel("Observed evaluations per model")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_pair(fig, args.panel_output_dir / PANEL_NAMES[1])

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    _barh(ax, counts["task_family"], 7, "Retained pathology evaluation mix", "Evaluation identities")
    fig.tight_layout()
    _save_pair(fig, args.panel_output_dir / PANEL_NAMES[2])

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    _barh(ax, counts["observed_family"], 7, "Observed cells by pathology task family", "Observed model-evaluation cells")
    fig.tight_layout()
    _save_pair(fig, args.panel_output_dir / PANEL_NAMES[3])

    source_labels = list(counts["source_provenance"])
    source_values = [counts["source_provenance"][label] for label in source_labels]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    wedges, _, _ = ax.pie(
        source_values, colors=COLORS[:len(source_values)], startangle=90,
        autopct="%1.1f%%", textprops={"fontsize": 8},
    )
    ax.legend(
        wedges, [f"{label} ({value:,})" for label, value in zip(source_labels, source_values)],
        loc="upper center", bbox_to_anchor=(.5, -.02), frameon=False,
    )
    ax.set_title("Reported-cell source provenance")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save_pair(fig, args.panel_output_dir / PANEL_NAMES[4])

    names = ", ".join(PANEL_NAMES)
    print(f"wrote {args.summary_output}, {args.output}.png/.pdf, and panels: {names}")


if __name__ == "__main__":
    main()
