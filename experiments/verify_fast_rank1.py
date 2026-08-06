#!/usr/bin/env python3
"""Compile and verify the optional exact-search rank-1 execution backend."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import run_probe_exhaustive_v2 as runner  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_sha256(function: object) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "data" / "scores.csv")
    parser.add_argument(
        "--source", type=Path, default=ROOT / "experiments" / "fast_rank1_v2.cpp"
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("/tmp/libpathopress_fast_rank1.so"),
        help=(
            "naming/location hint; the verified library is written beneath a "
            "private content-addressed sibling directory"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "probe_exhaustive_fast_equivalence_v2.json",
    )
    parser.add_argument("--samples", type=int, default=runner.FAST_MIN_COMPARISONS)
    parser.add_argument(
        "--cell-tolerance", type=float, default=runner.FAST_CELL_DELTA_CAP
    )
    parser.add_argument(
        "--metric-tolerance", type=float, default=runner.FAST_METRIC_DELTA_CAP
    )
    return parser.parse_args()


def _content_addressed_build(source: Path, hint: Path) -> tuple[Path, dict[str, object]]:
    compiler_name = shutil.which("g++")
    if compiler_name is None:
        raise FileNotFoundError("g++ was not found on PATH")
    compiler = Path(compiler_name).resolve(strict=True)
    source = source.resolve(strict=True)
    source_hash = sha256(source)
    directory = hint.parent.resolve() / f".{hint.stem}-{source_hash[:16]}"
    try:
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError:
        metadata = directory.lstat()
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(f"Build directory is not a real directory: {directory}")
        if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise RuntimeError(f"Build directory is not private to this user: {directory}")
    os.chmod(directory, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".build-", suffix=".so", dir=directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    command = [
        str(compiler),
        *runner.FAST_COMPILE_FLAGS,
        str(source),
        "-o",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        library_hash = sha256(temporary)
        library = directory / f"libfast_rank1-{library_hash}.so"
        if library.exists():
            if library.is_symlink() or not library.is_file() or sha256(library) != library_hash:
                raise RuntimeError(f"Content-addressed library collision: {library}")
            temporary.unlink()
        else:
            os.replace(temporary, library)
        os.chmod(library, 0o500)
    finally:
        if temporary.exists():
            temporary.unlink()
    compiler_version = subprocess.run(
        [str(compiler), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    return library, {
        "path": str(compiler),
        "sha256": sha256(compiler),
        "version": compiler_version,
    }


def main() -> int:
    args = parse_args()
    if args.samples < runner.FAST_MIN_COMPARISONS:
        raise ValueError(
            f"--samples must be at least {runner.FAST_MIN_COMPARISONS}"
        )
    if (
        not math.isfinite(args.cell_tolerance)
        or args.cell_tolerance < 0.0
        or args.cell_tolerance > runner.FAST_CELL_DELTA_CAP
    ):
        raise ValueError(
            f"--cell-tolerance must be within [0, {runner.FAST_CELL_DELTA_CAP}]"
        )
    if (
        not math.isfinite(args.metric_tolerance)
        or args.metric_tolerance < 0.0
        or args.metric_tolerance > runner.FAST_METRIC_DELTA_CAP
    ):
        raise ValueError(
            f"--metric-tolerance must be within [0, {runner.FAST_METRIC_DELTA_CAP}]"
        )
    library, compiler = _content_addressed_build(args.source, args.library)
    matrix, _, evaluation_ids = runner._load_matrix(args.scores)
    if matrix.shape[1] < 5:
        raise ValueError("At least five evaluation columns are required")
    rng = np.random.RandomState(20260806)
    combinations: list[tuple[int, ...]] = []
    for probes in (
        tuple(range(5)),
        tuple(range(matrix.shape[1] - 5, matrix.shape[1])),
    ):
        if probes not in combinations and len(combinations) < args.samples:
            combinations.append(probes)
    while len(combinations) < args.samples:
        probes = tuple(
            sorted(int(value) for value in rng.choice(matrix.shape[1], 5, replace=False))
        )
        if probes not in combinations:
            combinations.append(probes)
    jobs = [
        (sample_index, probes, "medae")
        for sample_index, probes in enumerate(combinations)
    ]

    started = time.time()
    runner._init_worker(matrix, tuple(evaluation_ids), runner.SEED)
    scalar = [runner._evaluate_combo(job) for job in jobs]
    runner._init_worker(matrix, tuple(evaluation_ids), runner.SEED, str(library))
    accelerated = [runner._evaluate_combo(job) for job in jobs]
    comparisons = []
    for reference, candidate in zip(scalar, accelerated):
        cell_delta = float(
            np.max(
                np.abs(
                    np.asarray(reference["predictions"]["pred"])
                    - np.asarray(candidate["predictions"]["pred"])
                )
            )
        )
        comparisons.append(
            {
                "combo_index": int(reference["combo_index"]),
                "probe_set": reference["probe_set"],
                "max_absolute_cell_delta": cell_delta,
                "absolute_medae_delta": abs(
                    float(reference["medae"]) - float(candidate["medae"])
                ),
                "absolute_medape_delta": abs(
                    float(reference["medape"]) - float(candidate["medape"])
                ),
            }
        )
    max_cell = max(row["max_absolute_cell_delta"] for row in comparisons)
    max_metric = max(
        max(row["absolute_medae_delta"], row["absolute_medape_delta"])
        for row in comparisons
    )
    passed = max_cell <= args.cell_tolerance and max_metric <= args.metric_tolerance
    payload = {
        "schema_version": runner.FAST_EQUIVALENCE_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "scientific_engine": {
            "rank": runner.PREDICTOR_RANK,
            "regularization": runner.PREDICTOR_REGULARIZATION,
            "iterations": 40,
            "ensembles": 10,
            "seeds": list(range(42, 52)),
        },
        "inputs": {
            "scores_path": runner._display_path(args.scores),
            "scores_sha256": sha256(args.scores),
            "source_path": runner._display_path(args.source),
            "source_sha256": sha256(args.source),
            "library_path": str(library),
            "library_sha256": sha256(library),
            "runner_path": runner._display_path(Path(runner.__file__)),
            "runner_sha256": sha256(Path(runner.__file__)),
            "execution_function_sha256": {
                "_init_worker": function_sha256(runner._init_worker),
                "_predict_all_known_fast": function_sha256(
                    runner._predict_all_known_fast
                ),
                "_evaluate_combo": function_sha256(runner._evaluate_combo),
            },
            "compiler": compiler,
            "compile_flags": list(runner.FAST_COMPILE_FLAGS),
            "platform": runner._platform_identity(),
        },
        "hard_caps": {
            "max_absolute_cell_delta": runner.FAST_CELL_DELTA_CAP,
            "max_absolute_metric_delta": runner.FAST_METRIC_DELTA_CAP,
            "minimum_comparisons": runner.FAST_MIN_COMPARISONS,
        },
        "requested_tolerances": {
            "max_absolute_cell_delta": args.cell_tolerance,
            "max_absolute_metric_delta": args.metric_tolerance,
        },
        "observed": {
            "sample_combinations": len(comparisons),
            "max_absolute_cell_delta": max_cell,
            "max_absolute_metric_delta": max_metric,
            "elapsed_seconds": time.time() - started,
        },
        "comparisons": comparisons,
    }
    runner._write_json_atomic(args.output, payload, indent=2)
    print(json.dumps(payload["observed"], indent=2))
    print(f"library -> {library}")
    print(f"{payload['status']} -> {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
