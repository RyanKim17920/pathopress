from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_probe_exhaustive_chunks",
    ROOT / "experiments/validate_probe_exhaustive_chunks.py",
)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)
MERGED_SPEC = importlib.util.spec_from_file_location(
    "validate_probe_exhaustive_merged",
    ROOT / "experiments/validate_probe_exhaustive_merged.py",
)
assert MERGED_SPEC and MERGED_SPEC.loader
MERGED_VALIDATOR = importlib.util.module_from_spec(MERGED_SPEC)
sys.modules[MERGED_SPEC.name] = MERGED_VALIDATOR
MERGED_SPEC.loader.exec_module(MERGED_VALIDATOR)


class ProbeExhaustiveIntegrityTests(unittest.TestCase):
    def test_combination_unranking_matches_itertools(self) -> None:
        for n in range(1, 9):
            for k in range(n + 1):
                expected = list(itertools.combinations(range(n), k))
                observed = [
                    VALIDATOR._unrank_combination(n, k, rank)
                    for rank in range(len(expected))
                ]
                self.assertEqual(observed, expected)
        with self.assertRaises(ValueError):
            VALIDATOR._unrank_combination(5, 2, -1)
        with self.assertRaises(ValueError):
            VALIDATOR._unrank_combination(5, 2, 10)

    def _fixture(self):
        matrix = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        evaluations = ("a", "b", "c")
        VALIDATOR._init_worker(matrix, evaluations)
        rows, columns = np.where(np.isfinite(matrix))
        true = matrix[rows, columns]
        record = {
            "combo_index": 1,
            "probe_set": ["b"],
            "probe_names": ["b"],
            "score": 0.0,
            "medape": 0.0,
            "medae": 0.0,
            "n": 6,
            "elapsed_s": 0.1,
            "predictions": {
                "i": rows.tolist(),
                "j": columns.tolist(),
                "true": true.tolist(),
                "pred": true.tolist(),
            },
        }
        config = {
            "fixed_probe_ids": [],
            "remaining_candidate_ids": list(evaluations),
            "choose_size_after_fixed": 1,
            "n_target_cells": 6,
            "metric": "medae",
        }
        return record, config

    def test_record_validation_accepts_exact_payload(self) -> None:
        record, config = self._fixture()
        result = VALIDATOR._validate_record(record, 1, config)
        self.assertEqual(result[:2], (0.0, 0.0))

    def test_record_validation_rejects_probe_and_metric_tampering(self) -> None:
        record, config = self._fixture()
        wrong_probe = copy.deepcopy(record)
        wrong_probe["probe_set"] = ["a"]
        with self.assertRaisesRegex(ValueError, "probe identity"):
            VALIDATOR._validate_record(wrong_probe, 1, config)
        wrong_metric = copy.deepcopy(record)
        for index in (0, 2, 3, 5):
            wrong_metric["predictions"]["pred"][index] = 0.0
        with self.assertRaisesRegex(ValueError, "recomputed MedAE"):
            VALIDATOR._validate_record(wrong_metric, 1, config)

    def test_record_validation_rejects_target_and_bounds_tampering(self) -> None:
        record, config = self._fixture()
        wrong_truth = copy.deepcopy(record)
        wrong_truth["predictions"]["true"][0] = 11.0
        with self.assertRaisesRegex(ValueError, "target truth"):
            VALIDATOR._validate_record(wrong_truth, 1, config)
        out_of_bounds = copy.deepcopy(record)
        out_of_bounds["predictions"]["pred"][0] = 101.0
        with self.assertRaisesRegex(ValueError, "outside"):
            VALIDATOR._validate_record(out_of_bounds, 1, config)

    def test_record_validation_rejects_types_indices_reveals_and_medape(self) -> None:
        record, config = self._fixture()
        for bad_value in (True, "10.0", float("nan"), float("inf")):
            tampered = copy.deepcopy(record)
            tampered["predictions"]["pred"][0] = bad_value
            with self.assertRaisesRegex(ValueError, "non-numeric/non-finite"):
                VALIDATOR._validate_record(tampered, 1, config)
        wrong_index = copy.deepcopy(record)
        wrong_index["predictions"]["j"][0] = 2
        with self.assertRaisesRegex(ValueError, "target indices"):
            VALIDATOR._validate_record(wrong_index, 1, config)
        wrong_reveal = copy.deepcopy(record)
        wrong_reveal["predictions"]["pred"][1] = 19.0
        with self.assertRaisesRegex(ValueError, "revealed probe"):
            VALIDATOR._validate_record(wrong_reveal, 1, config)
        wrong_medape = copy.deepcopy(record)
        wrong_medape["medape"] = 0.1
        with self.assertRaisesRegex(ValueError, "recomputed MedAPE"):
            VALIDATOR._validate_record(wrong_medape, 1, config)

    def test_nonstandard_json_and_symlinks_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "constant"):
            VALIDATOR._reject_json_constant("NaN")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            target = directory / "target"
            target.write_text("x", encoding="utf-8")
            link = directory / "link"
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "non-symlink regular"):
                VALIDATOR._require_regular(link, "test")

    def test_full_config_invariants_fail_before_chunk_scan(self) -> None:
        matrix = np.arange(125, dtype=float).reshape(5, 25) % 101
        models = [f"model.{index}" for index in range(matrix.shape[0])]
        evaluations = [f"evaluation.{index}" for index in range(matrix.shape[1])]
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            scores = fixture / "scores.csv"
            scores.write_text("self-contained-validator-fixture\n", encoding="utf-8")
            allowlist = fixture / "allowlist.json"
            allowlist.write_text(
                json.dumps({"evaluation_ids": evaluations}), encoding="utf-8"
            )
            config = {
                "schema_version": VALIDATOR.runner.CONFIG_SCHEMA_VERSION,
                "config_schema": "pathopress.probe_exhaustive.run.v2",
                "eval_protocol": VALIDATOR.runner.EVAL_PROTOCOL,
                "upstream_reference_commit": (
                    VALIDATOR.runner.UPSTREAM_REFERENCE_COMMIT
                ),
                "k": 5,
                "metric": "medae",
                "seed": VALIDATOR.runner.SEED,
                "n_models": len(models),
                "n_evaluations": len(evaluations),
                "n_observed": int(np.isfinite(matrix).sum()),
                "n_target_cells": int(np.isfinite(matrix).sum()),
                "eval_scope": "all_observed_cells",
                "predictor_rank": VALIDATOR.runner.PREDICTOR_RANK,
                "predictor_regularization": (
                    VALIDATOR.runner.PREDICTOR_REGULARIZATION
                ),
                "prediction_engine": VALIDATOR.EXPECTED_ENGINE_TEXT,
                "scores_path": str(scores),
                "scores_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
                "model_ids_hash": VALIDATOR.runner._short_text_hash(models),
                "evaluation_ids_hash": VALIDATOR.runner._short_text_hash(evaluations),
                "model_ids": models,
                "evaluation_ids": evaluations,
                "model_ids_sha256": VALIDATOR.runner._sha256_strings(models),
                "evaluation_ids_sha256": VALIDATOR.runner._sha256_strings(evaluations),
                "candidate_allowlist_path": str(allowlist),
                "candidate_allowlist_sha256": hashlib.sha256(
                    allowlist.read_bytes()
                ).hexdigest(),
                "candidate_limit": None,
                "candidate_ids": evaluations,
                "candidate_hash": VALIDATOR.runner._short_text_hash(evaluations),
                "candidate_ids_sha256": VALIDATOR.runner._sha256_strings(evaluations),
                "fixed_probe_ids": [],
                "fixed_probe_hash": None,
                "fixed_probe_ids_sha256": VALIDATOR.runner._sha256_strings([]),
                "remaining_candidate_ids": evaluations,
                "remaining_candidate_hash": VALIDATOR.runner._short_text_hash(evaluations),
                "remaining_candidate_ids_sha256": VALIDATOR.runner._sha256_strings(evaluations),
                "choose_size_after_fixed": 5,
                "total_combinations": math.comb(25, 5),
                "num_waves": 10,
                "num_shards": 8,
                "assignment_modulus": 80,
                "chunk_size": VALIDATOR.runner.DEFAULT_CHUNK_SIZE,
                "cell_masking": VALIDATOR.EXPECTED_MASKING_TEXT,
                "execution_backend": {"kind": "test-native-backend"},
            }
            cases = (
                ("assignment_modulus", 81, "scientific contract"),
                ("model_ids_hash", "tampered", "model ID hash"),
                ("total_combinations", 53129, "combination total"),
            )
            for key, value, message in cases:
                with self.subTest(key=key):
                    run_dir = fixture / f"run-{key}"
                    run_dir.mkdir()
                    tampered = copy.deepcopy(config)
                    tampered[key] = value
                    (run_dir / "config.json").write_text(
                        json.dumps(tampered), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(RuntimeError, message):
                        VALIDATOR.validate_run(
                            run_dir,
                            matrix,
                            models,
                            evaluations,
                            scores,
                            workers=1,
                            expected_backend=config["execution_backend"],
                        )

    def test_merged_order_validation_rejects_top_and_provenance_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            run_dir = directory / "run"
            run_dir.mkdir()
            config = {
                "schema_version": 2,
                "config_schema": "pathopress.probe_exhaustive.run.v2",
                "total_combinations": 2,
                "metric": "medae",
            }
            config_path = run_dir / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            top = [
                {
                    "combo_index": 0, "probe_set": ["a"], "score": 1.0,
                    "medape": 2.0, "medae": 1.0, "n": 4, "elapsed_s": 0.1,
                },
                {
                    "combo_index": 1, "probe_set": ["b"], "score": 2.0,
                    "medape": 3.0, "medae": 2.0, "n": 4, "elapsed_s": 0.2,
                },
            ]
            aggregate = "a" * 64
            integrity = {
                "status": "passed",
                "runs": [{
                    "config_sha256": config_hash,
                    "expected_top": top,
                    "expected_top_count": 2,
                    "chunk_digest_aggregate_sha256": aggregate,
                }],
            }
            integrity_path = directory / "integrity.json"
            integrity_path.write_text(json.dumps(integrity), encoding="utf-8")
            provenance = {
                "path": MERGED_VALIDATOR.display(integrity_path),
                "sha256": hashlib.sha256(integrity_path.read_bytes()).hexdigest(),
                "config_sha256": config_hash,
                "chunk_digest_aggregate_sha256": aggregate,
            }

            def write_merged(rows, source):
                payload = {
                    "config": config,
                    "complete": True,
                    "n_records": 2,
                    "missing_chunks": [],
                    "invalid_chunks": [],
                    "integrity_manifest": source,
                    "best": rows[0],
                    "top": rows,
                }
                with gzip.open(run_dir / "merged_summary.json.gz", "wt") as handle:
                    json.dump(payload, handle)

            write_merged(top, provenance)
            result = MERGED_VALIDATOR.validate_run(
                run_dir, integrity_path, integrity
            )
            self.assertEqual(result["top_rows_validated"], 2)
            write_merged(list(reversed(top)), provenance)
            with self.assertRaisesRegex(RuntimeError, "all-record ordering"):
                MERGED_VALIDATOR.validate_run(run_dir, integrity_path, integrity)
            bad_provenance = dict(provenance, sha256="0" * 64)
            write_merged(top, bad_provenance)
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                MERGED_VALIDATOR.validate_run(run_dir, integrity_path, integrity)


if __name__ == "__main__":
    unittest.main()
