"""The probe-compression artifact's on-disk form must be losslessly reversible.

``experiments/probe_compression_rank1.json`` stores fold-invariant LOFO curves
once at the mode level instead of repeating them inside all 34 folds.  These
tests pin that the hoisting is purely a storage transformation: every consumer
that loads the artifact sees exactly the values a fully materialised payload
would have carried.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.probe_compression import (  # noqa: E402
    FOLD_INVARIANT_CURVE_KEYS,
    FOLD_INVARIANT_KEYS_FIELD,
    dump_probe_compression,
    expand_fold_invariant_curves,
    hoist_fold_invariant_curves,
    load_probe_compression,
)

ARTIFACT = ROOT / "experiments" / "probe_compression_rank1.json"


def _payload(fold_curve: dict[str, object], n_folds: int = 3) -> dict[str, object]:
    return {
        "curves": {
            "any_candidate": {
                "lofo": {
                    str(fold): {"any_candidate": json.loads(json.dumps(fold_curve))}
                    for fold in range(n_folds)
                },
                "candidate_ids": list(fold_curve["candidate_ids"]),
                "all_known_greedy_medae": json.loads(
                    json.dumps(fold_curve["all_known_greedy_medae"])
                ),
            }
        }
    }


class FoldInvariantHoistingTests(unittest.TestCase):
    def test_hoisting_then_expanding_is_the_identity(self) -> None:
        fold_curve = {
            "candidate_ids": ["a", "b"],
            "all_known_greedy_medae": [{"k": 1, "medae": 1.5}],
            "heldout_greedy_medae": [{"k": 1, "medae": 2.5}],
        }
        original = _payload(fold_curve)
        stored = hoist_fold_invariant_curves(json.loads(json.dumps(original)))

        block = stored["curves"]["any_candidate"]
        self.assertEqual(
            block[FOLD_INVARIANT_KEYS_FIELD],
            ["candidate_ids", "all_known_greedy_medae"],
        )
        for fold in block["lofo"].values():
            self.assertNotIn("candidate_ids", fold["any_candidate"])
            self.assertIn("heldout_greedy_medae", fold["any_candidate"])

        self.assertEqual(expand_fold_invariant_curves(stored), original)

    def test_a_fold_specific_value_is_never_hoisted(self) -> None:
        original = _payload(
            {
                "candidate_ids": ["a"],
                "all_known_greedy_medae": [{"k": 1, "medae": 1.5}],
            }
        )
        # Perturb one fold so the block is no longer fold-invariant.
        original["curves"]["any_candidate"]["lofo"]["1"]["any_candidate"][
            "all_known_greedy_medae"
        ] = [{"k": 1, "medae": 9.0}]

        stored = hoist_fold_invariant_curves(json.loads(json.dumps(original)))
        block = stored["curves"]["any_candidate"]
        self.assertEqual(block[FOLD_INVARIANT_KEYS_FIELD], ["candidate_ids"])
        for fold in block["lofo"].values():
            self.assertIn("all_known_greedy_medae", fold["any_candidate"])
        self.assertEqual(expand_fold_invariant_curves(stored), original)

    def test_expanding_an_unhoisted_payload_is_a_no_op(self) -> None:
        original = _payload(
            {
                "candidate_ids": ["a"],
                "all_known_greedy_medae": [{"k": 1, "medae": 1.5}],
            }
        )
        self.assertEqual(
            expand_fold_invariant_curves(json.loads(json.dumps(original))), original
        )


class PublishedArtifactStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_probe_compression(ARTIFACT)

    def test_published_artifact_is_stored_in_hoisted_form(self) -> None:
        raw = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        modes = [
            mode
            for mode, block in raw["curves"].items()
            if isinstance(block, dict) and "lofo" in block
        ]
        self.assertTrue(modes)
        for mode in modes:
            block = raw["curves"][mode]
            self.assertIn(FOLD_INVARIANT_KEYS_FIELD, block)
            hoisted = block[FOLD_INVARIANT_KEYS_FIELD]
            self.assertTrue(set(hoisted) <= set(FOLD_INVARIANT_CURVE_KEYS))
            for fold in block["lofo"].values():
                for key in hoisted:
                    self.assertNotIn(key, fold[mode])

    def test_loading_materialises_every_fold_invariant_curve(self) -> None:
        for mode, block in self.loaded["curves"].items():
            if not isinstance(block, dict) or "lofo" not in block:
                continue
            self.assertNotIn(FOLD_INVARIANT_KEYS_FIELD, block)
            for fold in block["lofo"].values():
                curve = fold[mode]
                for key in FOLD_INVARIANT_CURVE_KEYS:
                    if key in block:
                        self.assertEqual(curve[key], block[key])

    def test_serialising_the_loaded_payload_reproduces_the_file(self) -> None:
        self.assertEqual(
            dump_probe_compression(self.loaded),
            ARTIFACT.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
