#!/usr/bin/env python3
"""Build compact main/appendix tables from Section 6 factor results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "side", "hypothesis", "label", "analysis_type", "feature_or_setting",
    "metric", "estimate", "p_value", "n", "denominator_notes",
]


def _corr_rows(data, side, hypotheses):
    block = data["correlational"][side]
    rows = []
    for hypothesis, label, feature in hypotheses:
        for metric in ("medape", "medae"):
            test = block["tests"][metric][feature]
            notes = "finite feature/error pairs"
            if feature == "log10_parameter_count":
                notes = "models with audited nominal encoder parameter count and OOF error"
            rows.append({
                "side": side, "hypothesis": hypothesis, "label": label,
                "analysis_type": "spearman", "feature_or_setting": feature,
                "metric": metric, "estimate": test["rho"], "p_value": test["p"],
                "n": test["n"], "denominator_notes": notes,
            })
    return rows


def _test_rows(side, hypothesis, label, tests, setting, notes):
    return [
        {
            "side": side, "hypothesis": hypothesis, "label": label,
            "analysis_type": "paired_wilcoxon", "feature_or_setting": setting,
            "metric": metric, "estimate": tests[metric]["median_delta"],
            "p_value": tests[metric]["p_value"], "n": tests[metric]["n"],
            "denominator_notes": notes,
        }
        for metric in ("medape", "medae")
    ]


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "experiments" / "prediction_error_factors_rank1.json")
    parser.add_argument("--main-output", type=Path, default=ROOT / "outputs" / "prediction_error_factor_table_rank1.csv")
    parser.add_argument("--appendix-output", type=Path, default=ROOT / "outputs" / "prediction_error_factor_table_appendix_rank1.csv")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    main_rows = []
    main_rows += _corr_rows(data, "benchmark", [
        ("H1", "Low-rank fit", "rank2_r2"),
        ("H2", "Score level", "median_score"),
        ("H3", "Score dispersion", "score_std"),
    ])
    main_rows += _corr_rows(data, "model", [
        ("H1", "Encoder parameter count", "log10_parameter_count"),
        ("H2", "Slide-encoder model type", "is_slide_model"),
        ("H3", "Score level", "median_score"),
        ("H4", "Low-rank fit", "rank2_r2"),
    ])
    benchmark_labels = {
        "benchmark_h4": ("H4", "Target coverage"),
        "benchmark_h5": ("H5", "Strong-neighbor presence"),
        "benchmark_h6": ("H6", "Strong-neighbor support"),
        "benchmark_h7": ("H7", "Same-task-family evidence"),
    }
    model_labels = {
        "model_h5": ("H5", "Strong-peer presence"),
        "model_h6": ("H6", "Strong-peer support"),
        "model_h7": ("H7", "Same-provider evidence"),
    }
    for key, (hypothesis, label) in benchmark_labels.items():
        block = data["interventions"][key]
        main_rows += _test_rows(
            "benchmark", hypothesis, label, block["tests"],
            block.get("headline_setting", "same_task_family"),
            "one median seed-level delta per evaluation",
        )
    for key, (hypothesis, label) in model_labels.items():
        block = data["interventions"][key]
        main_rows += _test_rows(
            "model", hypothesis, label, block["tests"],
            block.get("headline_setting", "paired headline"),
            "paired model denominator reported by the intervention",
        )
    h8 = data["interventions"]["model_h8"]["by_condition"]
    for condition, label in (("hide_25pct", "Hide 25%"), ("hide_75pct", "Hide 75%")):
        main_rows += _test_rows(
            "model", "H8", "Observation count", h8[condition]["tests"], label,
            "models with pooled baseline and treatment predictions",
        )
    h9 = data["interventions"]["model_h9"]["comparison_A_vs_B"]["10"]
    main_rows += _test_rows(
        "model", "H9", "Training-anchor recency", h9, "k=10; oldest minus middle",
        "paired seed-level aggregate errors",
    )

    appendix_rows = list(main_rows)
    # Direct rank-1 fit counterparts to the upstream rank-2 correlational rows.
    appendix_rows += _corr_rows(data, "benchmark", [("H1-rank1", "Pathology-selected low-rank fit", "rank1_r2")])
    appendix_rows += _corr_rows(data, "model", [("H4-rank1", "Pathology-selected low-rank fit", "rank1_r2")])
    for key, side, label in (
        ("benchmark_h4", "benchmark", "Target coverage"),
        ("benchmark_h5", "benchmark", "Strong-neighbor presence"),
        ("benchmark_h6", "benchmark", "Strong-neighbor support"),
        ("model_h6", "model", "Strong-peer support"),
    ):
        block = data["interventions"][key]
        for setting, tests in block["by_setting"].items():
            if setting == block.get("headline_setting"):
                continue
            appendix_rows += _test_rows(
                side, key, label, tests, setting,
                f"dose-response setting; one median seed delta per {side}",
            )
    temporal = data["interventions"]["model_h9"]["comparison_A_vs_B"]
    for k, metrics in temporal.items():
        if k == "10":
            continue
        appendix_rows += _test_rows(
            "model", "H9", "Training-anchor recency", metrics,
            f"k={k}; oldest minus middle", "paired seed-level aggregate errors",
        )
    _write(args.main_output, main_rows)
    _write(args.appendix_output, appendix_rows)
    print(f"main={len(main_rows)} appendix={len(appendix_rows)}")


if __name__ == "__main__":
    main()
