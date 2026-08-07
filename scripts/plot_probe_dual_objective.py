#!/usr/bin/env python3
"""Plot model-average prediction usefulness of scorecard-selected probes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"any_candidate": "#d33682", "pre_error_low_friction_allowlist": "#268bd2"}
LABELS = {"any_candidate": "Any evaluation", "pre_error_low_friction_allowlist": "25-task feasibility proxy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compression", type=Path, default=ROOT / "experiments/probe_compression_rank1.json")
    parser.add_argument("--raw", type=Path, default=ROOT / "outputs/probe_compression_selected_raw_rank1.csv")
    parser.add_argument("--table", type=Path, default=ROOT / "outputs/probe_dual_objective_rank1.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/probe_dual_objective_rank1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.compression.read_text(encoding="utf-8"))
    with args.raw.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        if row["protocol"] == "all_known" and row["selection_objective"] == "medae":
            groups[(row["candidate_mode"], int(row["k"]), row["model_id"])].append(row)

    records = []
    for mode in ("any_candidate", "pre_error_low_friction_allowlist"):
        for step in payload["curves"][mode]["all_known_greedy_medae"]:
            k = int(step["k"])
            average_errors = []
            for (group_mode, group_k, _), rows in groups.items():
                if group_mode != mode or group_k != k:
                    continue
                actual = float(np.mean([float(row["actual_normalized_score"]) for row in rows]))
                predicted = float(np.mean([float(row["predicted_normalized_score"]) for row in rows]))
                average_errors.append(abs(actual - predicted))
            records.append({
                "candidate_mode": mode,
                "candidate_label": LABELS[mode],
                "k": k,
                "added_evaluation_id": step["added_evaluation_id"],
                "probe_ids": json.dumps(step["probe_ids"], separators=(",", ":")),
                "scorecard_medae": float(step["selection_metrics"]["medae"]),
                "model_average_medae": float(np.median(average_errors)),
                "n_models_for_average": len(average_errors),
                "selection_objective": "scorecard_medae",
                "interpretation": "model_average_medae is an evaluation metric, not the greedy selection objective",
            })
    args.table.parent.mkdir(parents=True, exist_ok=True)
    with args.table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for mode in LABELS:
        rows = [row for row in records if row["candidate_mode"] == mode]
        color = COLORS[mode]
        ax.plot(
            [row["k"] for row in rows],
            [row["model_average_medae"] for row in rows],
            "o-",
            color=color,
            lw=2.2,
            label=LABELS[mode],
        )
    ax.set(
        xlabel="Measured evaluations (k)",
        ylabel="Median absolute model-average error (points)",
        xticks=range(1, 11),
    )
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.suptitle(
        "Predicting a model's average pathology score",
        fontsize=15,
        fontweight="bold",
        y=.97,
    )
    fig.text(
        .5,
        .025,
        "Probe sets were selected for scorecard MedAE; model-average MedAE is an independent evaluation. "
        "The 25-task set is a feasibility proxy, not measured cost.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=.13, right=.98, bottom=.22, top=.84)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.table} and {args.output}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
