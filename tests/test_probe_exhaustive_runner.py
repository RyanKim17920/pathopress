from __future__ import annotations

import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from experiments.run_probe_exhaustive import (
    _assignment_residue,
    _config_at_location,
    _config_for_chunk,
    _expected_combo_indices,
    _load_json,
    _valid_chunk_payload,
    _write_json_atomic,
    merge,
)


def _base_config() -> dict[str, object]:
    return {
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
        "candidate_hash": "candidates",
        "fixed_probe_hash": None,
        "remaining_candidate_hash": "remaining",
        "choose_size_after_fixed": 1,
        "total_combinations": 5,
        "num_waves": 2,
        "num_shards": 2,
        "assignment_modulus": 4,
        "chunk_size": 2,
    }


def _record(combo_index: int) -> dict[str, object]:
    return {
        "combo_index": combo_index,
        "probe_set": [f"eval.{combo_index}"],
        "score": float(combo_index),
        "medape": float(combo_index + 1),
        "medae": float(combo_index),
        "n": 10,
        "elapsed_s": 0.01,
        "predictions": {
            "i": [0],
            "j": [combo_index],
            "true": [1.0],
            "pred": [1.0],
        },
    }


class ProbeExhaustiveRunnerTests(unittest.TestCase):
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
        self.assertEqual(
            _valid_chunk_payload(wrong_order, config, indices)[1],
            "record combo_index mismatch",
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
