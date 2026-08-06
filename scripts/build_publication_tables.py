#!/usr/bin/env python3
"""Build deterministic CSV, Markdown, and LaTeX publication inventories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402
from pathopress.publication import read_csv  # noqa: E402


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _latex(value: Any) -> str:
    result = _text(value)
    for source, target in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        result = result.replace(source, target)
    return result


def _write_table(base: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: _text(row.get(key)) for key in columns} for row in rows])
    with base.with_suffix(".md").open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(_text(row.get(key)).replace("|", r"\|") for key in columns) + " |\n")
    with base.with_suffix(".tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{longtable}{" + "l" * len(columns) + "}\n")
        handle.write(" & ".join(_latex(value) for value in columns) + r" \\" + "\n\\hline\n\\endfirsthead\n")
        handle.write(" & ".join(_latex(value) for value in columns) + r" \\" + "\n\\hline\n\\endhead\n")
        for row in rows:
            handle.write(" & ".join(_latex(row.get(key)) for key in columns) + r" \\" + "\n")
        handle.write("\\end{longtable}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "data/scores.csv")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/tasks.csv")
    parser.add_argument("--models", type=Path, default=ROOT / "data/model_metadata.csv")
    parser.add_argument("--releases", type=Path, default=ROOT / "data/model_release_dates.csv")
    parser.add_argument("--feasibility", type=Path, default=ROOT / "data/evaluation_feasibility.csv")
    parser.add_argument("--structure", type=Path, default=ROOT / "experiments/structure_analysis")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_objects = load_scores(args.scores)
    matrix, model_ids, evaluation_ids = filter_matrix(*make_matrix(score_objects))
    retained_models, retained_evaluations = set(model_ids), set(evaluation_ids)
    scores = [row for row in read_csv(args.scores) if row["model_id"] in retained_models and row["evaluation_id"] in retained_evaluations and row["audit_status"] == "parsed_primary_source"]
    tasks = {row["evaluation_id"]: row for row in read_csv(args.tasks) if row["evaluation_id"] in retained_evaluations}
    models = {row["model_id"]: row for row in read_csv(args.models) if row["model_id"] in retained_models}
    releases = {row["model_id"]: row for row in read_csv(args.releases) if row["model_id"] in retained_models}
    feasibility = {row["evaluation_id"]: row for row in read_csv(args.feasibility) if row["evaluation_id"] in retained_evaluations}
    if set(tasks) != retained_evaluations or set(feasibility) != retained_evaluations:
        raise ValueError("evaluation metadata does not exactly cover the retained matrix")

    model_counts: dict[str, dict[str, Any]] = defaultdict(lambda: {"evaluations": set(), "suites": set()})
    evaluation_counts: dict[str, dict[str, Any]] = defaultdict(lambda: {"models": set()})
    for row in scores:
        model_counts[row["model_id"]]["evaluations"].add(row["evaluation_id"])
        model_counts[row["model_id"]]["suites"].add(row["suite_id"])
        evaluation_counts[row["evaluation_id"]]["models"].add(row["model_id"])

    model_rows = []
    for model_id in sorted(retained_models):
        metadata, release, observed = models.get(model_id, {}), releases.get(model_id, {}), model_counts[model_id]
        model_rows.append({
            "model_id": model_id,
            "provider": metadata.get("provider", ""),
            "family": metadata.get("family", ""),
            "model_type": metadata.get("model_type", ""),
            "modality": metadata.get("modality", ""),
            "parameter_count": metadata.get("parameter_count", ""),
            "release_date": release.get("release_date", metadata.get("release_date", "")),
            "release_date_basis": release.get("date_basis", "metadata_incomplete"),
            "n_observed_evaluations": len(observed["evaluations"]),
            "n_suites": len(observed["suites"]),
            "primary_source_url": release.get("primary_source_url", metadata.get("primary_source_url", "")),
        })
    evaluation_rows = []
    for evaluation_id in sorted(retained_evaluations):
        task, feasible = tasks[evaluation_id], feasibility[evaluation_id]
        evaluation_rows.append({
            "evaluation_id": evaluation_id,
            "suite_id": task["suite_id"],
            "dataset_id": task["dataset_id"],
            "task_family": task["task_family"],
            "target": task["target"],
            "sample_unit": task["sample_unit"],
            "task_type": task["task_type"],
            "sample_count": feasible["sample_count"],
            "sample_count_status": feasible["sample_count_status"],
            "metric": task["metric"],
            "n_observed_models": len(evaluation_counts[evaluation_id]["models"]),
            "allowlisted_low_friction_proxy": feasible["allowlisted"],
            "reference_url": task["reference_url"],
        })
    _write_table(args.output_dir / "model_inventory", model_rows, list(model_rows[0]))
    _write_table(args.output_dir / "evaluation_inventory", evaluation_rows, list(evaluation_rows[0]))

    threshold = json.loads((args.structure / "threshold_sweep.json").read_text(encoding="utf-8"))
    threshold_rows = []
    for grid_name, rows in sorted(threshold["grids"].items()):
        for row in rows:
            threshold_rows.append({"grid": grid_name, **row})
    threshold_columns = ["grid", "min_scores_per_model", "min_models_per_evaluation", "n_models", "n_evaluations", "n_observations", "fill_rate", "is_pathology_selected"]
    _write_table(args.output_dir / "threshold_sensitivity", threshold_rows, threshold_columns)

    submatrices = json.loads((args.structure / "submatrix_sweep.json").read_text(encoding="utf-8"))
    svd_rows = [{
        "n_models": row["n_models"], "n_evaluations": row["n_evaluations"],
        "stable_rank": f"{row['stable_rank']:.6f}", "rank1_variance": f"{row['var_rank1']:.6f}",
        "rank2_cumulative_variance": f"{row['var_rank2']:.6f}",
        "singular_values": ";".join(f"{value:.6f}" for value in row["singular_values"]),
    } for row in submatrices]
    _write_table(args.output_dir / "svd_submatrix_sensitivity", svd_rows, list(svd_rows[0]))

    manifest = {
        "schema_version": 1,
        "matrix": {"n_models": len(model_ids), "n_evaluations": len(evaluation_ids), "n_observed": len(scores)},
        "tables": {
            "model_inventory": len(model_rows),
            "evaluation_inventory": len(evaluation_rows),
            "threshold_sensitivity": len(threshold_rows),
            "svd_submatrix_sensitivity": len(svd_rows),
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.scores, args.tasks, args.models, args.releases, args.feasibility, args.structure / "threshold_sweep.json", args.structure / "submatrix_sweep.json")
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["tables"], indent=2))


if __name__ == "__main__":
    main()
