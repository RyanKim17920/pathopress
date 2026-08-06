#!/usr/bin/env python3
"""Run the pinned BenchPress confidence experiment on the pathology matrix.

The experiment is deliberately prediction-cache first, matching Microsoft
BenchPress at commit ``0a684b6``.  It consumes the exact 30-fold Section-4
method-comparison shards only after validating score, fold, matrix, and
row-level identities.  No held-out target is used to build its own point-risk
features, risk estimate, interval scale, or trust probability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.artifacts import load_fold_artifact  # noqa: E402
from pathopress.confidence import (  # noqa: E402
    DEFAULT_RISK_MODEL_GRID,
    DEFAULT_TRUST_BINS,
    DEFAULT_TRUST_THRESHOLD,
    confidence_feature_sets,
    conformal_interval,
    crossfit_error_risk,
    crossfit_trust_probability,
    fit_trust_calibrator,
    stack_features,
    structural_support_features_for_cells,
    summarize_confidence_method,
    trust_probability_summary,
)
from pathopress.matrix import filter_matrix, load_scores, make_matrix  # noqa: E402


METHODS = ("disagreement", "structural_support", "combined_risk_model")
RAW_DISAGREEMENT_METHODS = (
    "bias_als_hp_disagreement",
    "strong_method_disagreement",
)
UPSTREAM_COMMIT = "0a684b63ee0e4a401cb907a3827a82ea997d74c4"
FOLD_PROTOCOL = {"n_seeds": 10, "n_folds": 3, "base_seed": 42, "min_scores": 1}
TARGET_SPEC = {"transform": "logit", "method": "Bias ALS", "hp": {"rank": 1, "lam": 0.1}}
HP_VARIANTS = (
    {"transform": "logit", "method": "Bias ALS", "hp": {"rank": 1, "lam": 0.01}},
    TARGET_SPEC,
    {"transform": "logit", "method": "Bias ALS", "hp": {"rank": 1, "lam": 1.0}},
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_identity(matrix: np.ndarray, models: list[str], evaluations: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(models, separators=(",", ":")).encode())
    digest.update(json.dumps(evaluations, separators=(",", ":")).encode())
    digest.update(np.asarray(np.isfinite(matrix), dtype=np.uint8).tobytes())
    digest.update(np.nan_to_num(matrix, nan=-1.23456789e300).astype("<f8").tobytes())
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _matching_row(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    matches = [
        row for row in rows
        if row.get("status") == "completed"
        and row.get("transform") == spec["transform"]
        and row.get("method") == spec["method"]
        and row.get("hp") == spec["hp"]
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one completed method cache for {spec}, found {len(matches)}")
    return matches[0]


def _strong_rows(results: dict[str, Any], rows_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Upstream selection over the maximum attainable fold coverage."""

    candidates: list[dict[str, Any]] = []
    for transform, methods in results.items():
        for method, payload in methods.items():
            row = rows_by_id.get(str(payload.get("shard_id")))
            if row is None:
                raise ValueError(f"results row missing from validated manifest: {payload.get('shard_id')}")
            if transform == TARGET_SPEC["transform"] and method == TARGET_SPEC["method"]:
                continue
            candidates.append(row)
    if not candidates:
        raise ValueError("method results contain no alternative confidence generators")
    attainable = max(float(row.get("coverage", 0.0)) for row in candidates)
    full_attainable = [
        row for row in candidates
        if abs(float(row.get("coverage", 0.0)) - attainable) <= 1e-12
    ]
    selected = sorted(full_attainable, key=lambda row: float(row["medape_median"]))[:12]
    if len(selected) != 12:
        raise ValueError(
            f"need 12 maximum-attainable-coverage strong alternatives, found {len(selected)} "
            f"at coverage={attainable:.12f}"
        )
    return selected


def _load_prediction_cache(
    row: dict[str, Any], *, scores_hash: str, folds_hash: str,
    matrix_shape: tuple[int, int], matrix_identity: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    path = ROOT / row["prediction_file"]
    if not path.is_file():
        raise FileNotFoundError(
            f"required confidence generator cache missing: {path}; regenerate the exact "
            "method grid with `python3 experiments/run_method_comparison.py --merge`"
        )
    with np.load(path, allow_pickle=False) as data:
        required = {"fold_id", "test_i", "test_j", "actual", "predicted", "metadata_json"}
        if not required.issubset(data.files):
            raise ValueError(f"prediction cache lacks required arrays: {path}")
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(str(arrays["metadata_json"]))
    expected = {
        "status": "completed",
        "scores_sha256": scores_hash,
        "folds_sha256": folds_hash,
        "matrix_shape": list(matrix_shape),
        "matrix_identity_sha256": matrix_identity,
        "fold_protocol": FOLD_PROTOCOL,
        "upstream_commit": UPSTREAM_COMMIT,
        "shard_id": row["shard_id"],
        "transform": row["transform"],
        "method": row["method"],
        "hp": row["hp"],
    }
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"prediction-cache identity mismatch for {path}: {actual} != {expected}")
    audit = {
        "path": _display_path(path),
        "sha256": _sha256(path),
        "shard_id": row["shard_id"],
        "transform": row["transform"],
        "method": row["method"],
        "hp": row["hp"],
        "medape_median": row["medape_median"],
        "coverage": row["coverage"],
        "identity_fields_verified": sorted(expected),
    }
    return arrays, metadata, audit


def _assert_aligned(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], label: str) -> None:
    for key in ("fold_id", "test_i", "test_j", "actual"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"confidence generator {label} is not row-aligned on {key}")


def _structural_features(
    reference: dict[str, np.ndarray], folds: list[tuple[int, int, np.ndarray, list[tuple[int, int]]]],
) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {}
    for fold_id, (_seed, _fold, training, held) in enumerate(folds):
        selected = np.flatnonzero(reference["fold_id"].astype(int) == fold_id)
        cells = [
            (int(reference["test_i"][index]), int(reference["test_j"][index]))
            for index in selected
        ]
        if cells != [(int(row), int(column)) for row, column in held]:
            raise ValueError(f"fold-cache cell order mismatch in fold {fold_id}")
        current = structural_support_features_for_cells(training, cells)
        for name, values in current.items():
            parts.setdefault(name, []).append(values)
    return {name: np.concatenate(values) for name, values in parts.items()}


def _generator_stacks(
    *, matrix: np.ndarray, models: list[str], evaluations: list[str], scores: Path,
    folds_path: Path, method_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _load_json(method_dir / "manifest.json")
    results = _load_json(method_dir / "results.json")
    if manifest.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("method-grid upstream commit does not match pinned confidence contract")
    if manifest.get("counts") != {"completed": 343, "missing": 0, "unsupported": 0}:
        raise ValueError(f"confidence requires the complete 343-shard method grid: {manifest.get('counts')}")
    rows = manifest["completed"]
    rows_by_id = {str(row["shard_id"]): row for row in rows}
    hp_rows = [_matching_row(rows, spec) for spec in HP_VARIANTS]
    strong_rows = _strong_rows(results, rows_by_id)
    scores_hash = _sha256(scores)
    folds_hash = _sha256(folds_path)
    identity = _matrix_identity(matrix, models, evaluations)

    loaded: dict[str, tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]] = {}
    for row in [*hp_rows, *strong_rows]:
        loaded[row["shard_id"]] = _load_prediction_cache(
            row, scores_hash=scores_hash, folds_hash=folds_hash,
            matrix_shape=matrix.shape, matrix_identity=identity,
        )
    target_row = _matching_row(rows, TARGET_SPEC)
    reference = loaded[target_row["shard_id"]][0]
    for row in [*hp_rows, *strong_rows]:
        _assert_aligned(reference, loaded[row["shard_id"]][0], row["shard_id"])
    hp_stack = np.stack([loaded[row["shard_id"]][0]["predicted"].astype(float) for row in hp_rows])
    strong_stack = np.stack([loaded[row["shard_id"]][0]["predicted"].astype(float) for row in strong_rows])
    audits = {row["shard_id"]: loaded[row["shard_id"]][2] for row in [*hp_rows, *strong_rows]}
    return reference, hp_stack, strong_stack, [audits[row["shard_id"]] for row in hp_rows], [audits[row["shard_id"]] for row in strong_rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "data" / "scores.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "experiments" / "folds_s10_f3_bs42.json")
    parser.add_argument("--method-dir", type=Path, default=ROOT / "experiments" / "method_comparison")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "confidence_calibration_rank1.json")
    parser.add_argument("--cells", type=Path, default=ROOT / "experiments" / "confidence_cells_rank1.csv")
    parser.add_argument("--base-seed", type=int, default=42)
    # Retained for command compatibility; cache loading is I/O-bound and risk
    # fits are intentionally deterministic and sequential, as upstream.
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores)
    matrix, models, evaluations = make_matrix(scores)
    matrix, models, evaluations = filter_matrix(matrix, models, evaluations)
    metadata_by_evaluation: dict[str, tuple[str, str]] = {}
    for score in scores:
        if score.evaluation_id in evaluations:
            metadata_by_evaluation.setdefault(score.evaluation_id, (score.suite_id, score.metric))
    folds = load_fold_artifact(args.folds, matrix, models, evaluations)
    reference, hp_stack, strong_stack, hp_audit, strong_audit = _generator_stacks(
        matrix=matrix, models=models, evaluations=evaluations, scores=args.scores,
        folds_path=args.folds, method_dir=args.method_dir,
    )
    structural_features = _structural_features(reference, folds)
    target_supported = np.isfinite(reference["predicted"].astype(float))
    generator_masks = [
        *[np.isfinite(values) for values in hp_stack],
        *[np.isfinite(values) for values in strong_stack],
    ]
    if any(not np.array_equal(target_supported, mask) for mask in generator_masks):
        raise ValueError("maximum-attainable confidence generators disagree on predictable cells")
    structural_supported = np.logical_and.reduce(
        [np.isfinite(values) for values in structural_features.values()]
    )
    supported = target_supported & structural_supported
    n_prediction_instances_total = int(len(target_supported))
    n_unidentifiable_instances = int((~supported).sum())
    fold_ids = reference["fold_id"].astype(int)[supported]
    rows = reference["test_i"].astype(int)[supported]
    columns = reference["test_j"].astype(int)[supported]
    actual = reference["actual"].astype(float)[supported]
    predicted = reference["predicted"].astype(float)[supported]
    hp_stack = hp_stack[:, supported]
    strong_stack = strong_stack[:, supported]
    structural_features = {
        name: values[supported] for name, values in structural_features.items()
    }
    hp_features = stack_features(hp_stack, predicted)
    strong_features = stack_features(strong_stack, predicted)
    all_features = confidence_feature_sets(hp_features, strong_features, structural_features)

    uncertainties: dict[str, np.ndarray] = {}
    intervals: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    trust_probabilities: dict[str, np.ndarray] = {}
    risk_metadata: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    trust_metadata: dict[str, Any] = {}
    target_cell_group = rows.astype(np.int64) * len(evaluations) + columns.astype(np.int64)
    raw_uncertainties = {
        "bias_als_hp_disagreement": hp_features["mad"],
        "strong_method_disagreement": strong_features["mad"],
    }
    for method in RAW_DISAGREEMENT_METHODS:
        summaries[method] = summarize_confidence_method(
            actual, predicted, fold_ids, raw_uncertainties[method]
        )
    for method in METHODS:
        uncertainty, feature_names, selected = crossfit_error_risk(
            actual, predicted, fold_ids, all_features[method],
            seed=args.base_seed, label=method, verbose=True,
        )
        if not np.all(np.isfinite(uncertainty)):
            raise ValueError(f"cross-fit uncertainty is incomplete for {method}")
        lower, upper, scale = conformal_interval(actual, predicted, uncertainty, fold_ids)
        trust, trust_by_fold = crossfit_trust_probability(
            uncertainty, actual, predicted, fold_ids,
            group_id=target_cell_group,
            threshold=DEFAULT_TRUST_THRESHOLD, n_bins=DEFAULT_TRUST_BINS,
        )
        if not np.all(np.isfinite(trust)):
            raise ValueError(f"cross-fit trust probability is incomplete for {method}")
        _full_predictor, full_calibrator = fit_trust_calibrator(
            uncertainty, actual, predicted,
            threshold=DEFAULT_TRUST_THRESHOLD, n_bins=DEFAULT_TRUST_BINS,
        )
        uncertainties[method] = uncertainty
        intervals[method] = (lower, upper, scale)
        trust_probabilities[method] = trust
        risk_metadata[method] = {
            "feature_names": feature_names,
            "selected_risk_model_by_fold": selected,
        }
        summaries[method] = summarize_confidence_method(actual, predicted, fold_ids, uncertainty)
        summaries[method]["trust_probability"] = trust_probability_summary(
            actual, predicted, trust, threshold=DEFAULT_TRUST_THRESHOLD,
        )
        trust_metadata[method] = {
            "crossfit_calibrator_by_held_out_fold": trust_by_fold,
            "full_heldout_calibrator_for_deployment": full_calibrator,
        }

    raw_feature_columns: dict[str, np.ndarray] = {}
    for method in ("disagreement", "structural_support"):
        for name, values in all_features[method].items():
            raw_feature_columns[f"{method}_{name}"] = values
    cell_fields = [
        "crossfit_fold_id", "seed", "fold", "model_id", "evaluation_id",
        "suite_id", "metric", "actual_normalized_score", "predicted_normalized_score",
        "absolute_error", *sorted(raw_feature_columns),
        *[f"{method}_risk" for method in RAW_DISAGREEMENT_METHODS],
    ]
    for method in METHODS:
        cell_fields.extend([
            f"{method}_risk", f"{method}_lower_90", f"{method}_upper_90",
            f"{method}_conformal_scale", f"{method}_trust_probability",
        ])
    args.cells.parent.mkdir(parents=True, exist_ok=True)
    with args.cells.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cell_fields, lineterminator="\n")
        writer.writeheader()
        for cell_index in range(len(actual)):
            evaluation = evaluations[int(columns[cell_index])]
            suite, metric = metadata_by_evaluation[evaluation]
            fold_id = int(fold_ids[cell_index])
            row: dict[str, Any] = {
                "crossfit_fold_id": fold_id,
                "seed": 42 + fold_id // 3,
                "fold": fold_id % 3,
                "model_id": models[int(rows[cell_index])],
                "evaluation_id": evaluation,
                "suite_id": suite,
                "metric": metric,
                "actual_normalized_score": f"{actual[cell_index]:.6f}",
                "predicted_normalized_score": f"{predicted[cell_index]:.6f}",
                "absolute_error": f"{abs(predicted[cell_index] - actual[cell_index]):.6f}",
                **{name: f"{values[cell_index]:.6f}" for name, values in raw_feature_columns.items()},
                **{
                    f"{method}_risk": f"{raw_uncertainties[method][cell_index]:.6f}"
                    for method in RAW_DISAGREEMENT_METHODS
                },
            }
            for method in METHODS:
                lower, upper, scale = intervals[method]
                row.update({
                    f"{method}_risk": f"{uncertainties[method][cell_index]:.6f}",
                    f"{method}_lower_90": f"{lower[cell_index]:.6f}",
                    f"{method}_upper_90": f"{upper[cell_index]:.6f}",
                    f"{method}_conformal_scale": f"{scale[cell_index]:.6f}",
                    f"{method}_trust_probability": f"{trust_probabilities[method][cell_index]:.8f}",
                })
            writer.writerow(row)

    payload = {
        "schema_version": 2,
        "description": "Pinned BenchPress confidence calibration port for pathology matrix completion.",
        "upstream": {
            "repository": "https://github.com/microsoft/benchpress",
            "commit": UPSTREAM_COMMIT,
            "source": "experiments/sec6_trust/confidence_calibration/run.py and benchpress/methods/confidence.py",
        },
        "matrix": {
            "n_models": len(models), "n_evaluations": len(evaluations),
            "n_observed": int(np.isfinite(matrix).sum()), "density": float(np.isfinite(matrix).mean()),
        },
        "configuration": {
            "target_predictor": "logit bias ALS rank=1 regularization=0.1",
            "fold_protocol": FOLD_PROTOCOL,
            "n_prediction_instances": len(actual),
            "n_prediction_instances_total": n_prediction_instances_total,
            "n_unidentifiable_fold_instances_excluded": n_unidentifiable_instances,
            "attainable_prediction_coverage": len(actual) / n_prediction_instances_total,
            "risk_target": "log1p(abs(predicted_normalized_score - actual_normalized_score))",
            "confidence_methods": list(METHODS),
            "raw_disagreement_diagnostics": list(RAW_DISAGREEMENT_METHODS),
            "risk_model_grid": [
                {"model": model, "hidden_layers": list(hidden)}
                for model, hidden in DEFAULT_RISK_MODEL_GRID
            ],
            "hp_disagreement_variants": hp_audit,
            "strong_method_variants": strong_audit,
            "strong_method_selection": "top 12 maximum-attainable-coverage best-HP transform/method rows by Section-4 fold-median MedAPE, excluding the target transform/method",
            "structural_feature_count": len(structural_features),
            "structural_features": sorted(structural_features),
            "conformal_protocol": "leave the target point-prediction fold out when fitting the 90% scale",
            "trust_event": "abs(predicted_normalized_score - actual_normalized_score) <= 10",
            "trust_threshold_justification": "Ten points is one decile of the shared normalized 0-100 pathology score scale and exactly preserves BenchPress's public trust event; it is an engineering tolerance, not a clinical threshold.",
            "trust_protocol": "leave the target point-prediction fold out and purge every repeated-seed instance of its model-evaluation targets before fitting the decreasing binned-isotonic mapping",
        },
        "risk_models": risk_metadata,
        "trust_calibration": trust_metadata,
        "confidence_methods": summaries,
        "input": {
            "scores_path": _display_path(args.scores), "scores_sha256": _sha256(args.scores),
            "folds_path": _display_path(args.folds), "folds_sha256": _sha256(args.folds),
            "method_manifest_path": _display_path(args.method_dir / "manifest.json"),
            "method_manifest_sha256": _sha256(args.method_dir / "manifest.json"),
            "method_results_path": _display_path(args.method_dir / "results.json"),
            "method_results_sha256": _sha256(args.method_dir / "results.json"),
            "cells_path": _display_path(args.cells), "cells_sha256": _sha256(args.cells),
        },
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "script_sha256": _sha256(Path(__file__)),
        },
        "pathology_adaptations": [
            "Pathology-selected rank 1 replaces upstream rank 2 while preserving the exact target family, transform, lambda, folds, feature definitions, risk-model grid, and calibration logic.",
            "The twelve strong generators are pathology's own top full-coverage Section-4 alternatives under the exact upstream selection rule.",
            "Fold instances whose benchmark column has no remaining training observations are excluded from confidence fitting; all selected generators share the same predictable-cell mask.",
            "Errors, interval widths, and the ten-point trust event use the shared normalized pathology 0-100 scale, not clinical utility units.",
            "Each observed cell appears once per seed, matching BenchPress's repeated within-model fold experiment; this is not external-institution validation.",
        ],
        "leakage_guards": [
            "All 15 generator shards must share exact score, fold, matrix, protocol, target coordinates, and actual-value identities before stacking.",
            "Each point prediction is read from the same held-out fold cache used by the Section-4 comparison.",
            "Each risk estimate is fit without its target point-prediction fold.",
            "Each serialized evaluation trust probability is calibrated without its target point-prediction fold or any repeated-seed instance of the same model-evaluation target.",
            "The full held-out trust mapping is serialized only for outcome-free deployment lookup.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": _display_path(args.output),
        "cells": _display_path(args.cells),
        "n_hp_generators": len(hp_audit),
        "n_strong_generators": len(strong_audit),
        "methods": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
