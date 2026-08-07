from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

import experiments.run_probe_exhaustive_v2 as runner
from experiments.run_probe_exhaustive_v2 import (
    ROOT,
    _assignment_residue,
    _config_at_location,
    _cleanup_staged_fast_library,
    _config_for_chunk,
    _expected_combo_indices,
    _init_worker,
    _load_json,
    _predict_all_known_fast,
    _stage_fast_library,
    _valid_chunk_payload,
    _validate_fast_equivalence,
    _write_json_atomic,
    merge,
)
from pathopress.probe_compression import predict_all_known, score_predictions


def _base_config() -> dict[str, object]:
    model_ids = ["model.0", "model.1"]
    evaluation_ids = [f"eval.{index}" for index in range(5)]
    return {
        "schema_version": 2,
        "config_schema": "pathopress.probe_exhaustive.run.v2",
        "eval_protocol": "all_known_probe_bruteforce_v1",
        "upstream_reference_commit": "0a684b63ee0e4a401cb907a3827a82ea997d74c4",
        "k": 1,
        "metric": "medae",
        "seed": 42,
        "n_models": 2,
        "n_evaluations": 5,
        "n_observed": 10,
        "n_target_cells": 10,
        "predictor_rank": 1,
        "predictor_regularization": 0.1,
        "scores_sha256": "scores",
        "model_ids_hash": "models",
        "evaluation_ids_hash": "evaluations",
        "model_ids_sha256": "model-sha256",
        "evaluation_ids_sha256": "evaluation-sha256",
        "candidate_ids_sha256": "candidate-sha256",
        "fixed_probe_ids_sha256": "fixed-sha256",
        "remaining_candidate_ids_sha256": "remaining-sha256",
        "model_ids": model_ids,
        "evaluation_ids": evaluation_ids,
        "candidate_ids": evaluation_ids,
        "fixed_probe_ids": [],
        "remaining_candidate_ids": evaluation_ids,
        "candidate_hash": "candidates",
        "fixed_probe_hash": None,
        "remaining_candidate_hash": "remaining",
        "choose_size_after_fixed": 1,
        "total_combinations": 5,
        "num_waves": 2,
        "num_shards": 2,
        "assignment_modulus": 4,
        "chunk_size": 2,
        "execution_backend": {"kind": "test-native-backend"},
    }


def _record(combo_index: int) -> dict[str, object]:
    rows = [0] * 5 + [1] * 5
    columns = list(range(5)) * 2
    values = [float(index + 1) for index in range(10)]
    return {
        "combo_index": combo_index,
        "probe_set": [f"eval.{combo_index}"],
        "probe_names": [f"eval.{combo_index}"],
        "score": 0.0,
        "medape": 0.0,
        "medae": 0.0,
        "n": 10,
        "elapsed_s": 0.01,
        "predictions": {
            "i": rows,
            "j": columns,
            "true": values,
            "pred": values,
        },
    }


class ProbeExhaustiveRunnerTests(unittest.TestCase):
    def test_staged_fast_library_fd_survives_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "backend.so"
            original = b"first verified native inode"
            source.write_bytes(original)
            digest = hashlib.sha256(original).hexdigest()
            descriptor, directory, staged, observed = _stage_fast_library(source, digest)
            try:
                self.assertEqual(observed, digest)
                replacement = Path(temporary) / "replacement.so"
                replacement.write_bytes(b"different bytes")
                os.replace(replacement, staged)
                os.lseek(descriptor, 0, os.SEEK_SET)
                self.assertEqual(os.read(descriptor, len(original) + 1), original)
            finally:
                _cleanup_staged_fast_library(descriptor, directory)

    def test_compiled_rank1_backend_is_numerically_equivalent(self) -> None:
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ is unavailable")
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary) / "fast_rank1.so"
            subprocess.run(
                [
                    compiler,
                    "-O3",
                    "-std=c++17",
                    "-fPIC",
                    "-shared",
                    "-ffp-contract=off",
                    str(root / "experiments" / "fast_rank1_v2.cpp"),
                    "-o",
                    str(library),
                ],
                check=True,
            )
            matrix = np.asarray(
                [
                    [51.0, 62.0, 73.0, 84.0, 65.0, 76.0],
                    [53.0, 64.0, 75.0, 86.0, 67.0, 78.0],
                    [55.0, 66.0, 77.0, 88.0, 69.0, 80.0],
                    [57.0, 68.0, 79.0, 90.0, 71.0, 82.0],
                ]
            )
            scalar = predict_all_known(matrix, (0, 3), rank=1, regularization=0.1)
            _init_worker(
                matrix,
                tuple(f"evaluation.{index}" for index in range(matrix.shape[1])),
                42,
                str(library),
            )
            accelerated = _predict_all_known_fast(matrix, (0, 3))
            np.testing.assert_allclose(
                accelerated.predicted,
                scalar.predicted,
                rtol=0.0,
                atol=2e-12,
                equal_nan=True,
            )
            self.assertAlmostEqual(
                float(score_predictions(accelerated)["medae"]),
                float(score_predictions(scalar)["medae"]),
                places=12,
            )

            native = ctypes.CDLL(str(library)).complete_target_rank1
            vector = np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
            )
            native.argtypes = [
                vector,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                vector,
                vector,
                vector,
            ]
            native.restype = ctypes.c_int
            flat = np.ascontiguousarray(matrix).ravel()
            initial_rows = np.zeros(10 * matrix.shape[0], dtype=np.float64)
            initial_columns = np.zeros(10 * matrix.shape[1], dtype=np.float64)
            output = np.empty(matrix.shape[1], dtype=np.float64)
            self.assertEqual(
                native(
                    flat,
                    matrix.shape[0],
                    matrix.shape[1],
                    -1,
                    initial_rows,
                    initial_columns,
                    output,
                ),
                11,
            )
            out_of_range = flat.copy()
            out_of_range[0] = 101.0
            self.assertEqual(
                native(
                    out_of_range,
                    matrix.shape[0],
                    matrix.shape[1],
                    0,
                    initial_rows,
                    initial_columns,
                    output,
                ),
                14,
            )
            nonfinite_initial = initial_rows.copy()
            nonfinite_initial[0] = np.nan
            self.assertEqual(
                native(
                    flat,
                    matrix.shape[0],
                    matrix.shape[1],
                    0,
                    nonfinite_initial,
                    initial_columns,
                    output,
                ),
                13,
            )

    def test_schema_v2_chunk_binds_backend_and_deep_validates_records(self) -> None:
        evaluation_ids = [f"eval.{index}" for index in range(5)]
        config = {
            **_base_config(),
            "schema_version": 2,
            "config_schema": "pathopress.probe_exhaustive.run.v2",
            "model_ids_sha256": "model-sha",
            "evaluation_ids_sha256": "evaluation-sha",
            "candidate_ids_sha256": "candidate-sha",
            "fixed_probe_ids_sha256": "fixed-sha",
            "remaining_candidate_ids_sha256": "remaining-sha",
            "model_ids": ["model.0"],
            "evaluation_ids": evaluation_ids,
            "fixed_probe_ids": [],
            "remaining_candidate_ids": evaluation_ids,
            "execution_backend": {"kind": "native_rank1", "library_sha256": "lib-a"},
            "n_models": 1,
            "n_observed": 5,
            "n_target_cells": 5,
        }
        config = _config_at_location(config, 0, 0)
        record = {
            "combo_index": 0,
            "probe_set": ["eval.0"],
            "probe_names": ["eval.0"],
            "score": 0.0,
            "medape": 0.0,
            "medae": 0.0,
            "n": 5,
            "elapsed_s": 0.01,
            "predictions": {
                "i": [0, 0, 0, 0, 0],
                "j": [0, 1, 2, 3, 4],
                "true": [10.0, 20.0, 30.0, 40.0, 50.0],
                "pred": [10.0, 20.0, 30.0, 40.0, 50.0],
            },
        }
        payload = {
            "config": _config_for_chunk(config),
            "combo_indices": [0],
            "records": [record],
        }
        self.assertEqual(_valid_chunk_payload(payload, config, [0]), (True, "ok"))

        forged = json.loads(json.dumps(payload))
        forged["records"][0]["predictions"]["pred"][1] = 101.0
        self.assertIn(
            "prediction range mismatch", _valid_chunk_payload(forged, config, [0])[1]
        )
        changed_backend = {**config, "execution_backend": {"kind": "native_rank1", "library_sha256": "lib-b"}}
        self.assertEqual(
            _valid_chunk_payload(payload, changed_backend, [0])[1],
            "chunk config mismatch",
        )

    def test_fast_evidence_rejects_self_escalated_or_forged_deltas(self) -> None:
        compiler_name = shutil.which("g++")
        if compiler_name is None:
            self.skipTest("g++ is unavailable")
        compiler = Path(compiler_name).resolve()
        compiler_version = subprocess.run(
            [str(compiler), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            library = directory / "backend.so"
            library.write_bytes(b"hash-bound test fixture")
            comparisons = [
                {
                    "combo_index": index,
                    "probe_set": [f"eval.{index}.{column}" for column in range(5)],
                    "max_absolute_cell_delta": 0.0,
                    "absolute_medae_delta": 0.0,
                    "absolute_medape_delta": 0.0,
                }
                for index in range(runner.FAST_MIN_COMPARISONS)
            ]
            payload = {
                "schema_version": runner.FAST_EQUIVALENCE_SCHEMA_VERSION,
                "status": "passed",
                "scientific_engine": {
                    "rank": 1,
                    "regularization": 0.1,
                    "iterations": 40,
                    "ensembles": 10,
                    "seeds": list(range(42, 52)),
                },
                "inputs": {
                    "scores_sha256": runner._sha256_bytes(ROOT / "data" / "scores.csv"),
                    "source_sha256": runner._sha256_bytes(runner.FAST_SOURCE),
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "runner_sha256": runner._sha256_bytes(Path(runner.__file__)),
                    "execution_function_sha256": {
                        "_init_worker": runner._function_sha256(runner._init_worker),
                        "_predict_all_known_fast": runner._function_sha256(
                            runner._predict_all_known_fast
                        ),
                        "_evaluate_combo": runner._function_sha256(runner._evaluate_combo),
                    },
                    "compiler": {
                        "path": str(compiler),
                        "sha256": runner._sha256_bytes(compiler),
                        "version": compiler_version,
                    },
                    "compile_flags": list(runner.FAST_COMPILE_FLAGS),
                    "platform": runner._platform_identity(),
                },
                "hard_caps": {
                    "max_absolute_cell_delta": runner.FAST_CELL_DELTA_CAP,
                    "max_absolute_metric_delta": runner.FAST_METRIC_DELTA_CAP,
                    "minimum_comparisons": runner.FAST_MIN_COMPARISONS,
                },
                "observed": {
                    "sample_combinations": len(comparisons),
                    "max_absolute_cell_delta": 0.0,
                    "max_absolute_metric_delta": 0.0,
                },
                "comparisons": comparisons,
            }
            evidence = directory / "evidence.json"
            evidence.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                _validate_fast_equivalence(
                    ROOT / "data" / "scores.csv", library, evidence
                )["status"],
                "passed",
            )
            payload["hard_caps"]["max_absolute_cell_delta"] = 1e9
            evidence.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hard_caps"):
                _validate_fast_equivalence(
                    ROOT / "data" / "scores.csv", library, evidence
                )
            payload["hard_caps"]["max_absolute_cell_delta"] = runner.FAST_CELL_DELTA_CAP
            payload["comparisons"][0]["max_absolute_cell_delta"] = 1.0
            evidence.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "cell_max_recomputed|cell_hard_cap"):
                _validate_fast_equivalence(
                    ROOT / "data" / "scores.csv", library, evidence
                )

    def test_wave_shard_residues_and_chunk_indices_cover_space_once(self) -> None:
        total = math.comb(5, 1)
        pieces: list[int] = []
        for shard in range(2):
            for wave in range(2):
                residue = _assignment_residue(wave, 2, shard, 2)
                pieces.extend(_expected_combo_indices(total, residue, 4, 2, 0))
        self.assertEqual(sorted(pieces), list(range(total)))
        self.assertEqual(len(pieces), len(set(pieces)))

    def test_chunk_validation_checks_config_indices_and_record_order(self) -> None:
        config = _config_at_location(_base_config(), 0, 0)
        indices = [0, 4]
        payload = {
            "config": _config_for_chunk(config),
            "combo_indices": indices,
            "records": [_record(index) for index in indices],
        }
        self.assertEqual(_valid_chunk_payload(payload, config, indices), (True, "ok"))

        wrong_order = dict(payload)
        wrong_order["records"] = list(reversed(payload["records"]))
        self.assertIn(
            "record combo_index mismatch",
            _valid_chunk_payload(wrong_order, config, indices)[1],
        )
        wrong_config = dict(payload)
        wrong_config["config"] = {**payload["config"], "metric": "medape"}
        self.assertEqual(
            _valid_chunk_payload(wrong_config, config, indices)[1],
            "chunk config mismatch",
        )

    def test_merge_is_complete_by_default_and_writes_compact_top_n(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            base = _base_config()
            _write_json_atomic(out_dir / "config.json", base, indent=2)
            total = int(base["total_combinations"])
            for shard in range(2):
                for wave in range(2):
                    config = _config_at_location(base, wave, shard)
                    residue = int(config["assignment_residue"])
                    indices = _expected_combo_indices(total, residue, 4, 2, 0)
                    if not indices:
                        continue
                    path = (
                        out_dir
                        / "shards"
                        / f"wave_{wave:02d}"
                        / f"shard_{shard:03d}"
                        / "chunk_000000.json.gz"
                    )
                    _write_json_atomic(
                        path,
                        {
                            "config": _config_for_chunk(config),
                            "combo_indices": indices,
                            "records": [_record(index) for index in indices],
                        },
                    )

            merge(Namespace(out_dir=out_dir, top_n=2, allow_incomplete=False))
            result = _load_json(out_dir / "merged_summary.json.gz")
            self.assertTrue(result["complete"])
            self.assertEqual(result["n_records"], 5)
            self.assertEqual([row["combo_index"] for row in result["top"]], [0, 1])
            self.assertNotIn("predictions", result["top"][0])

            # Remove one materialized chunk: strict merge must fail while the
            # explicitly diagnostic merge records incompleteness.
            missing = out_dir / "shards" / "wave_00" / "shard_000" / "chunk_000000.json.gz"
            missing.unlink()
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                merge(Namespace(out_dir=out_dir, top_n=2, allow_incomplete=False))
            merge(Namespace(out_dir=out_dir, top_n=2, allow_incomplete=True))
            partial = _load_json(out_dir / "merged_summary.json.gz")
            self.assertFalse(partial["complete"])
            self.assertTrue(partial["missing_chunks"])


if __name__ == "__main__":
    unittest.main()
