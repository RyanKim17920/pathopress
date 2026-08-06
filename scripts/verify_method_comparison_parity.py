#!/usr/bin/env python3
"""Numerically compare PathoPress method-grid primitives with pinned BenchPress."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress import method_comparison as local  # noqa: E402


def load_upstream(checkout: Path):
    benchpress = types.ModuleType("benchpress")
    benchpress.__path__ = []
    methods = types.ModuleType("benchpress.methods")
    methods.__path__ = []
    harness = types.ModuleType("benchpress.evaluation_harness")

    def normalize(matrix):
        means = np.nanmean(matrix, axis=0)
        stds = np.nanstd(matrix, axis=0)
        stds[stds < 1e-8] = 1.0
        return (matrix - means) / stds, means, stds

    harness.col_normalize = normalize
    harness.col_denormalize = lambda matrix, means, stds: matrix * stds + means
    globals_stub = {
        "M_FULL": np.empty((0, 0)), "OBSERVED": np.empty((0, 0), dtype=bool),
        "N_MODELS": 0, "N_BENCH": 0, "MODEL_IDS": [], "BENCH_IDS": [],
        "MODEL_NAMES": [], "BENCH_NAMES": [], "MODEL_PROVIDERS": [],
        "MODEL_REASONING": [], "MODEL_OPEN": [], "MODEL_PARAMS": [],
        "MODEL_ACTIVE": [], "BENCH_CATS": [],
    }
    for name, value in globals_stub.items():
        setattr(harness, name, value)
    sys.modules.update(
        {"benchpress": benchpress, "benchpress.methods": methods, "benchpress.evaluation_harness": harness}
    )

    def load(name: str, relative: str):
        specification = importlib.util.spec_from_file_location(name, checkout / relative)
        if specification is None or specification.loader is None:
            raise RuntimeError(f"cannot load {relative}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
        return module

    transforms = load("benchpress.methods.transforms", "benchpress/methods/transforms.py")
    completers = load("benchpress.methods.completers", "benchpress/methods/completers.py")
    return transforms, completers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    upstream_transforms, upstream = load_upstream(args.checkout)
    matrix = np.asarray(
        [
            [0.0, 0.2, np.nan, 0.4, 0.5, 0.6],
            [0.1, np.nan, 0.3, 0.4, 0.6, 0.7],
            [0.2, 0.25, 0.35, np.nan, 0.7, 0.8],
            [0.3, 0.4, 0.5, 0.6, np.nan, 0.9],
            [0.4, 0.5, 0.6, 0.7, 0.8, 0.95],
            [0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        ]
    )
    upstream.N_MODELS, upstream.N_BENCH = matrix.shape
    comparisons = {
        "Benchmark Mean": (local.complete_benchmark_mean(matrix), upstream.complete_benchmark_mean(matrix)),
        "Model Mean": (local.complete_model_mean(matrix), upstream.complete_model_mean(matrix)),
        "Bench-KNN": (local.complete_bench_knn(matrix, k=3), upstream.complete_bench_knn(matrix, k=3)),
        "Model-KNN": (local.complete_model_knn(matrix, k=3), upstream.complete_model_knn(matrix, k=3)),
        "BenchReg": (local.complete_benchreg(matrix, top_k=3, min_r2=0.1), upstream.complete_benchreg(matrix, top_k=3, min_r2=0.1)),
        "ModelReg": (local.complete_modelreg(matrix, top_k=3, min_r2=0.1), upstream.complete_modelreg(matrix, top_k=3, min_r2=0.1)),
        "Soft-Impute": (local.complete_soft_impute(matrix, rank=1), upstream.complete_soft_impute(matrix, rank=1, normalize=False)),
        "Bias ALS rank 1": (local.complete_bias_als(matrix, rank=1, lam=0.1), upstream.complete_bias_als(matrix, rank=1, lam=0.1, normalize=False)),
        "Bias ALS rank 2": (local.complete_bias_als(matrix, rank=2, lam=0.1), upstream.complete_bias_als(matrix, rank=2, lam=0.1, normalize=False)),
        "NMF": (local.complete_nmf(matrix, rank=1), upstream.complete_nmf(matrix, rank=1, normalize=False)),
        "PMF": (local.complete_pmf(matrix, rank=1), upstream.complete_pmf(matrix, rank=1, normalize=False)),
        "Nuclear Norm": (local.complete_nuclear_norm(matrix, lam=0.1), upstream.complete_nuclear_norm(matrix, lam=0.1, normalize=False)),
    }
    results = {}
    for name, (left, right) in comparisons.items():
        difference = float(np.nanmax(np.abs(left - right)))
        results[name] = difference
        if difference > args.tolerance or not np.array_equal(np.isnan(left), np.isnan(right)):
            raise AssertionError(f"{name} parity failed: {difference}")

    raw_matrix = matrix * 100.0
    for transform in local.TRANSFORMS:
        local_z, state = local.apply_transform(raw_matrix, transform)
        to_function, from_function, percentage_only = upstream_transforms.TRANSFORMS[transform]
        upstream_z, observed, is_percentage, means, stds = upstream_transforms.apply_transform(
            raw_matrix, to_function, percentage_only
        )
        forward_difference = float(np.nanmax(np.abs(local_z - upstream_z)))
        predicted_z = local.complete_benchmark_mean(local_z)
        local_raw = local.invert_transform(predicted_z, raw_matrix, transform, state)
        upstream_raw = upstream_transforms.invert_transform(
            predicted_z, raw_matrix, to_function, from_function, percentage_only,
            observed, is_percentage, means, stds,
        )
        inverse_difference = float(np.nanmax(np.abs(local_raw - upstream_raw)))
        results[f"transform:{transform}:forward"] = forward_difference
        results[f"transform:{transform}:inverse"] = inverse_difference
        if max(forward_difference, inverse_difference) > args.tolerance:
            raise AssertionError(f"{transform} transform parity failed")
    print(json.dumps({"tolerance": args.tolerance, "max_abs_differences": results}, indent=2))


if __name__ == "__main__":
    main()
