#!/usr/bin/env python3
"""Publish compact ranking-preservation tracks from current probe compression."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402


MODES = ("any_candidate", "pre_error_low_friction_allowlist")
MODE_LABELS = {
    "any_candidate": "Any evaluation",
    "pre_error_low_friction_allowlist": "25-task feasibility proxy",
}


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty ranking table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "ranking_scope": metrics["ranking_scope"],
        "pairwise_margin": float(metrics["pairwise_margin"]),
        "pairwise_n_pairs": int(metrics["pairwise_n_pairs"]),
        "pairwise_median_accuracy": float(metrics["pairwise_median_accuracy"]),
        "pairwise_pooled_accuracy": float(metrics["pairwise_pooled_accuracy"]),
        "top_fraction": float(metrics["top_fraction"]),
        "top_total_k": int(metrics["top_total_k"]),
        "top_median_recovery": float(metrics["top_median_recovery"]),
        "top_pooled_recovery": float(metrics["top_pooled_recovery"]),
    }


def _random_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for k in range(1, 11):
        current = [row["metrics"] for row in rows if int(row["k"]) == k]
        if len(current) != 10:
            raise ValueError(f"expected ten random ranking repeats at k={k}")
        pairwise = np.asarray(
            [float(metrics["pairwise_median_accuracy"]) for metrics in current]
        )
        top = np.asarray([float(metrics["top_median_recovery"]) for metrics in current])
        output.append(
            {
                "k": k,
                "n_repeats": len(current),
                "pairwise_median": float(np.median(pairwise)),
                "pairwise_q1": float(np.quantile(pairwise, 0.25)),
                "pairwise_q3": float(np.quantile(pairwise, 0.75)),
                "top_recovery_median": float(np.median(top)),
                "top_recovery_q1": float(np.quantile(top, 0.25)),
                "top_recovery_q3": float(np.quantile(top, 0.75)),
            }
        )
    return output


def _validate_current_compression(
    payload: dict[str, Any], scores_sha256: str, matrix_shape: list[int], evaluations: list[str]
) -> None:
    config = payload["configuration"]
    if config.get("scores_sha256") != scores_sha256:
        raise ValueError("compression score hash does not match current scores")
    if config.get("matrix_shape") != matrix_shape:
        raise ValueError("compression matrix shape does not match current scores")
    if int(config.get("prediction_rank", -1)) != 1:
        raise ValueError("ranking release requires the selected pathology rank 1")
    if float(config.get("ranking_margin", -1)) != 5.0:
        raise ValueError("ranking release requires the dedicated margin-5 objective")

    expected_candidates = {
        "any_candidate": evaluations,
        "pre_error_low_friction_allowlist": payload["allowlist"]["evaluation_ids"],
    }
    for mode in MODES:
        if payload["curves"][mode]["candidate_ids"] != expected_candidates[mode]:
            raise ValueError(f"{mode} candidate identities do not match current inputs")
        ranking = payload["ranking_aware"][mode]
        if len(ranking["all_known_greedy"]) != 10 or len(ranking["heldout_greedy"]) != 10:
            raise ValueError(f"{mode} greedy ranking trajectories must be complete through k=10")
        if len(ranking["all_known_random"]) != 100:
            raise ValueError(f"{mode} random ranking trajectory must contain 10x10 rows")
        if {int(row["k"]) for row in ranking["all_known_random"]} != set(range(1, 11)):
            raise ValueError(f"{mode} random ranking k coverage is incomplete")
        if {int(row["repeat"]) for row in ranking["all_known_random"]} != set(range(10)):
            raise ValueError(f"{mode} random ranking repeat coverage is incomplete")


def build_release_payload(
    compression: dict[str, Any], *, scores_sha256: str, compression_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pairwise_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    tracks: dict[str, Any] = {}

    def append_metrics(
        *, protocol: str, mode: str, method: str, repeat: int | None, k: int,
        added_evaluation_id: str | None, metrics: dict[str, Any]
    ) -> None:
        common = {
            "protocol": protocol,
            "candidate_mode": mode,
            "method": method,
            "repeat": "" if repeat is None else repeat,
            "k": k,
            "added_evaluation_id": added_evaluation_id or "",
            "ranking_scope": metrics["ranking_scope"],
        }
        pairwise_rows.append(
            {
                **common,
                "margin": float(metrics["pairwise_margin"]),
                "n_pairs": int(metrics["pairwise_n_pairs"]),
                "median_accuracy": float(metrics["pairwise_median_accuracy"]),
                "pooled_accuracy": float(metrics["pairwise_pooled_accuracy"]),
            }
        )
        top_rows.append(
            {
                **common,
                "top_fraction": float(metrics["top_fraction"]),
                "total_k": int(metrics["top_total_k"]),
                "median_recovery": float(metrics["top_median_recovery"]),
                "pooled_recovery": float(metrics["top_pooled_recovery"]),
            }
        )

    for mode in MODES:
        ranking = compression["ranking_aware"][mode]
        all_known = []
        for row in ranking["all_known_greedy"]:
            metrics = _compact_metrics(row["selection_metrics"])
            compact = {
                "k": int(row["k"]),
                "added_evaluation_id": row["added_evaluation_id"],
                "probe_ids": row["probe_ids"],
                "metrics": metrics,
            }
            all_known.append(compact)
            append_metrics(
                protocol="all_known", mode=mode, method="greedy", repeat=None,
                k=compact["k"], added_evaluation_id=compact["added_evaluation_id"],
                metrics=metrics,
            )

        for row in ranking["all_known_random"]:
            append_metrics(
                protocol="all_known", mode=mode, method="random_prefix",
                repeat=int(row["repeat"]), k=int(row["k"]), added_evaluation_id=None,
                metrics=row["metrics"],
            )

        heldout = []
        for row in ranking["heldout_greedy"]:
            non_probe = _compact_metrics(row["validation_non_probe"])
            with_probe = _compact_metrics(row["validation_with_probe_zero"])
            compact = {
                "k": int(row["k"]),
                "added_evaluation_id": row["added_evaluation_id"],
                "probe_ids": row["probe_ids"],
                "validation_non_probe": non_probe,
                "validation_with_probe_zero": with_probe,
            }
            heldout.append(compact)
            for protocol, metrics in (
                ("heldout_non_probe", non_probe),
                ("heldout_with_probe_zero", with_probe),
            ):
                append_metrics(
                    protocol=protocol, mode=mode, method="train_selected_greedy",
                    repeat=None, k=compact["k"],
                    added_evaluation_id=compact["added_evaluation_id"], metrics=metrics,
                )

        random_summary = _random_summary(ranking["all_known_random"])
        tracks[mode] = {
            "label": MODE_LABELS[mode],
            "candidate_count": len(compression["curves"][mode]["candidate_ids"]),
            "all_known_greedy": all_known,
            "all_known_random_summary": random_summary,
            "heldout_greedy": heldout,
        }

    current_k10 = {}
    for mode in MODES:
        track = tracks[mode]
        current_k10[mode] = {
            "all_known_greedy": track["all_known_greedy"][-1]["metrics"],
            "all_known_random_median": track["all_known_random_summary"][-1],
            "heldout_non_probe": track["heldout_greedy"][-1]["validation_non_probe"],
            "heldout_with_probe_zero": track["heldout_greedy"][-1]["validation_with_probe_zero"],
        }

    payload = {
        "schema_version": 2,
        "metadata": {
            "experiment": "Current-score probe ranking preservation",
            "source_protocol": "probe_compression_ranking_aware_margin5_v1",
            "prediction_rank": 1,
            "ranking_margin": 5.0,
            "top_fraction": 0.2,
            "scores_sha256": scores_sha256,
            "compression_sha256": compression_sha256,
            "semantics": (
                "All-known tracks count revealed probes as exact and use every observed target; "
                "held-out non-probe tracks rank hidden validation cells only; the with-probe-zero "
                "diagnostic includes revealed validation probes. Greedy sets optimize median "
                "pairwise accuracy at a true normalized-score gap of at least five points."
            ),
            "historical_oof_artifact": (
                "The former 0/1/2/5-margin OOF ranking release was bound to an earlier score "
                "snapshot and is superseded by these current compression-derived trajectories."
            ),
        },
        "matrix": compression["configuration"] | {
            "candidate_modes": {
                mode: len(compression["curves"][mode]["candidate_ids"]) for mode in MODES
            }
        },
        "tracks": tracks,
        "summary": {"current_k10": current_k10},
    }
    return payload, pairwise_rows, top_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "data/scores.csv")
    parser.add_argument(
        "--compression", type=Path,
        default=PROJECT_ROOT / "experiments/probe_compression_rank1.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "experiments/ranking_preservation_rank1.json",
    )
    parser.add_argument(
        "--pairwise-csv", type=Path,
        default=PROJECT_ROOT / "outputs/ranking_preservation_pairwise_rank1.csv",
    )
    parser.add_argument(
        "--top-csv", type=Path,
        default=PROJECT_ROOT / "outputs/ranking_preservation_top_fraction_rank1.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores_sha256 = hashlib.sha256(args.scores.read_bytes()).hexdigest()
    matrix, _, evaluations = filter_matrix(*make_matrix(load_scores(args.scores)))
    compression = json.loads(args.compression.read_text(encoding="utf-8"))
    _validate_current_compression(
        compression, scores_sha256, list(matrix.shape), evaluations
    )
    payload, pairwise_rows, top_rows = build_release_payload(
        compression,
        scores_sha256=scores_sha256,
        compression_sha256=hashlib.sha256(args.compression.read_bytes()).hexdigest(),
    )
    _write_json_atomic(args.output, payload)
    _write_csv_atomic(args.pairwise_csv, pairwise_rows)
    _write_csv_atomic(args.top_csv, top_rows)
    print(
        f"wrote {args.output}, {args.pairwise_csv}, and {args.top_csv}; "
        f"current unrestricted k=10 margin-5 accuracy="
        f"{payload['summary']['current_k10']['any_candidate']['all_known_greedy']['pairwise_median_accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
