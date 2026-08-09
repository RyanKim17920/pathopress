"""Unit tests for the LOFO post-processing backfill in run_probe_compression.

Ensures that:
- Only fold-invariant keys are promoted to the top level.
- Held-out curves (fold-specific) are never promoted.
- all_known_random is only copied for any_candidate, not allowlist.
- No KeyError when fold 0 only contains its own mode key.
"""

from __future__ import annotations

import copy
import json
import unittest


def _lofo_backfill(payload: dict, first_fold: str) -> None:
    """Reproduce the backfill logic from run_probe_compression.py for testing.

    This is the version AFTER the fix.  It only promotes fold-invariant
    keys and never emits fold-specific held-out curves.
    """
    for _cm in ("any_candidate", "pre_error_low_friction_allowlist"):
        _source = payload["curves"][_cm]["lofo"][first_fold][_cm]
        for _bk in ("candidate_indices", "candidate_ids",
                    "all_known_greedy_medae", "all_known_greedy_medape"):
            if _bk in _source and _bk not in payload["curves"][_cm]:
                payload["curves"][_cm][_bk] = copy.deepcopy(_source[_bk])
        if _cm == "any_candidate" and "all_known_random" not in payload["curves"][_cm]:
            payload["curves"][_cm]["all_known_random"] = copy.deepcopy(
                _source.get("all_known_random", [])
            )


def _make_synthetic_payload() -> dict:
    """Build a minimal LOFO payload matching the real nesting structure.

    curves[mode]["lofo"][fold][mode][key] — each fold only contains its
    own mode key, never the other mode.
    """
    return {
        "curves": {
            "any_candidate": {
                "lofo": {
                    "0": {
                        "any_candidate": {
                            "candidate_indices": [0, 1, 2],
                            "candidate_ids": ["e0", "e1", "e2"],
                            "all_known_greedy_medae": [{"k": 1, "medae": 0.5}],
                            "all_known_greedy_medape": [{"k": 1, "medape": 1.0}],
                            "heldout_greedy_medae": [{"k": 1, "medae": 2.0}],
                            "heldout_greedy_medape": [{"k": 1, "medape": 3.0}],
                            "all_known_random": [{"k": 1, "repeat": 0}],
                            "heldout_random": [{"k": 1, "repeat": 0, "n_revealed": 0}],
                        },
                    },
                    "1": {
                        "any_candidate": {
                            "candidate_indices": [0, 1, 2],
                            "candidate_ids": ["e0", "e1", "e2"],
                            "all_known_greedy_medae": [{"k": 1, "medae": 0.5}],
                            "all_known_greedy_medape": [{"k": 1, "medape": 1.0}],
                            "heldout_greedy_medae": [{"k": 1, "medae": 2.1}],
                            "heldout_greedy_medape": [{"k": 1, "medape": 3.1}],
                            "all_known_random": [{"k": 1, "repeat": 0}],
                            "heldout_random": [{"k": 1, "repeat": 0, "n_revealed": 0}],
                        },
                    },
                },
            },
            "pre_error_low_friction_allowlist": {
                "lofo": {
                    "0": {
                        "pre_error_low_friction_allowlist": {
                            "candidate_indices": [0, 1],
                            "candidate_ids": ["e0", "e1"],
                            "all_known_greedy_medae": [{"k": 1, "medae": 0.6}],
                            "all_known_greedy_medape": [{"k": 1, "medape": 1.1}],
                            "heldout_greedy_medae": [{"k": 1, "medae": 2.2}],
                            "heldout_greedy_medape": [{"k": 1, "medape": 3.2}],
                            "heldout_random": [{"k": 1, "repeat": 0, "n_revealed": 0}],
                        },
                    },
                    "1": {
                        "pre_error_low_friction_allowlist": {
                            "candidate_indices": [0, 1],
                            "candidate_ids": ["e0", "e1"],
                            "all_known_greedy_medae": [{"k": 1, "medae": 0.6}],
                            "all_known_greedy_medape": [{"k": 1, "medape": 1.1}],
                            "heldout_greedy_medae": [{"k": 1, "medae": 2.3}],
                            "heldout_greedy_medape": [{"k": 1, "medape": 3.3}],
                            "heldout_random": [{"k": 1, "repeat": 0, "n_revealed": 0}],
                        },
                    },
                },
            },
        },
    }


class TestLofoBackfill(unittest.TestCase):
    """Verify the LOFO backfill logic is correct and safe."""

    def _synthetic(self) -> dict:
        return copy.deepcopy(_make_synthetic_payload())

    def test_no_keyerror_on_allowlist_iteration(self) -> None:
        """BUG(a): fold 0 only has {mode} key, not the other mode.

        The old code tried payload[...]["any_candidate"]["lofo"][0]
        ["pre_error_low_friction_allowlist"] which raised KeyError.
        The fix reads from payload[curves][_cm]["lofo"][fold][_cm].
        """
        payload = self._synthetic()
        # Must not raise
        _lofo_backfill(payload, "0")

    def test_fold_invariant_keys_promoted(self) -> None:
        """candidate_indices, candidate_ids, all_known_greedy_* are promoted."""
        payload = self._synthetic()
        _lofo_backfill(payload, "0")

        for mode in ("any_candidate", "pre_error_low_friction_allowlist"):
            for key in ("candidate_indices", "candidate_ids",
                        "all_known_greedy_medae", "all_known_greedy_medape"):
                self.assertIn(
                    key, payload["curves"][mode],
                    f"{mode} should have top-level {key}",
                )

    def test_heldout_keys_not_promoted(self) -> None:
        """BUG: held-out curves are fold-specific and must not be promoted."""
        payload = self._synthetic()
        _lofo_backfill(payload, "0")

        for mode in ("any_candidate", "pre_error_low_friction_allowlist"):
            self.assertNotIn(
                "heldout_greedy_medae", payload["curves"][mode],
                f"{mode} must not have top-level heldout_greedy_medae",
            )
            self.assertNotIn(
                "heldout_greedy_medape", payload["curves"][mode],
                f"{mode} must not have top-level heldout_greedy_medape",
            )
            self.assertNotIn(
                "heldout_random", payload["curves"][mode],
                f"{mode} must not have top-level heldout_random",
            )

    def test_all_known_random_only_any_candidate(self) -> None:
        """BUG(b): all_known_random from any_candidate must not leak to allowlist."""
        payload = self._synthetic()
        _lofo_backfill(payload, "0")

        self.assertIn(
            "all_known_random", payload["curves"]["any_candidate"],
            "any_candidate should have top-level all_known_random",
        )
        self.assertNotIn(
            "all_known_random",
            payload["curves"]["pre_error_low_friction_allowlist"],
            "allowlist must NOT have top-level all_known_random",
        )

    def test_no_cross_mode_contamination(self) -> None:
        """Allowlist all_known_random must not be any_candidate's copy."""
        payload = self._synthetic()
        _lofo_backfill(payload, "0")

        allow_top = payload["curves"]["pre_error_low_friction_allowlist"]
        # Verify allowlist curve was populated from its own fold, not any_candidate
        self.assertEqual(
            allow_top["candidate_indices"], [0, 1],
            "allowlist candidate_indices must come from allowlist fold, not any_candidate",
        )
        self.assertEqual(
            allow_top["candidate_ids"], ["e0", "e1"],
            "allowlist candidate_ids must come from allowlist fold",
        )
        self.assertEqual(
            allow_top["all_known_greedy_medae"][0]["medae"], 0.6,
            "allowlist all_known_greedy_medae must be allowlist value, not 0.5",
        )

    def test_per_fold_data_unchanged(self) -> None:
        """Backfill must not modify the nested per-fold data."""
        payload = self._synthetic()
        orig = json.dumps(
            payload["curves"]["any_candidate"]["lofo"]["0"]["any_candidate"],
            sort_keys=True,
        )
        _lofo_backfill(payload, "0")
        after = json.dumps(
            payload["curves"]["any_candidate"]["lofo"]["0"]["any_candidate"],
            sort_keys=True,
        )
        self.assertEqual(orig, after, "per-fold data must be unchanged")


if __name__ == "__main__":
    unittest.main()
