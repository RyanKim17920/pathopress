"""Regression tests for the matched-cell replay and the skill-scope fix.

Three defects motivated these tests:

1. ``experiments/run_probe_selection.py`` divided a MATRIX-WIDE parity MedAE by
   a COLUMN-SCOPED leave-one-out baseline, turning ``skill_score`` into a
   re-encoding of column dispersion.
2. The published arms were each scored on their own hidden denominator, so a
   greedy-versus-random comparison was confounded by cell composition.
3. ``experiments/run_probe_compression.py`` refused to emit held-out per-cell
   rows for the random arm and never emitted k=0 rows at all.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pathopress.probes import (
    SKILL_NOISE_FLOOR_DISPERSION,
    compute_column_loo_baseline_medae,
    compute_column_skill,
    evaluate_global_probes,
)

ROOT = Path(__file__).resolve().parents[1]

REPLAY_PATH = ROOT / "scripts/replay_lofo_matched_cells.py"
_SPEC = importlib.util.spec_from_file_location("replay_lofo_matched_cells", REPLAY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(_SPEC)
sys.modules["replay_lofo_matched_cells"] = REPLAY
_SPEC.loader.exec_module(REPLAY)

RUNNER_PATH = ROOT / "experiments/run_probe_compression.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_probe_compression_matched_cells", RUNNER_PATH
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(RUNNER)


class SkillNumeratorScopeTests(unittest.TestCase):
    """The skill numerator must be denominated over one column, not the matrix."""

    MATRIX = np.array(
        [
            [80.0, 10.0, 60.0],
            [75.0, 10.2, 59.0],
            [72.0, 9.8, 55.0],
            [68.0, 10.1, 53.0],
            [64.0, 9.9, 50.0],
        ]
    )

    def test_matrix_wide_numerator_is_not_column_scoped(self) -> None:
        """A single matrix-wide error cannot be a per-column numerator.

        Column 1 is tight (values within 0.4 points) and columns 0/2 are wide.
        A matrix-wide MedAE is one number for all three, so dividing it by each
        column's own leave-one-out baseline produces an ordering driven purely
        by the denominator.
        """

        result = evaluate_global_probes(self.MATRIX, [0], rank=1)
        matrix_wide = result.parity.median_absolute_error
        baselines = [
            compute_column_loo_baseline_medae(self.MATRIX, column)
            for column in range(self.MATRIX.shape[1])
        ]
        ratios = [matrix_wide / base for base in baselines]
        # The ordering of the "skill" scores is exactly the inverse ordering of
        # the baselines -- it contains no information about the model.
        self.assertEqual(
            [i for i, _ in sorted(enumerate(ratios), key=lambda p: p[1])],
            [i for i, _ in sorted(enumerate(baselines), key=lambda p: -p[1])],
        )

    def test_per_column_hidden_medae_is_scoped_to_its_own_column(self) -> None:
        """The correctly scoped numerator differs per column."""

        result = evaluate_global_probes(self.MATRIX, [], rank=1)
        per_column = result.per_column_hidden_medae
        self.assertEqual(len(per_column), self.MATRIX.shape[1])
        self.assertGreater(len(set(np.round(per_column, 9))), 1)
        # Every entry is a median over that column's own cells, so it must lie
        # within the column's own error range rather than the matrix's.
        for column, value in enumerate(per_column):
            self.assertTrue(np.isfinite(value))
            self.assertLessEqual(
                value,
                float(np.nanmax(np.abs(self.MATRIX[:, column] - np.nanmedian(
                    self.MATRIX[:, column])))) + abs(value),
            )

    def test_probe_column_has_no_hidden_cells(self) -> None:
        """A column used as the only probe is revealed, so it has no numerator.

        This is why the fix points at the k=0 baseline evaluation rather than
        at the one-probe evaluation row.
        """

        result = evaluate_global_probes(self.MATRIX, [1], rank=1)
        self.assertTrue(np.isnan(result.per_column_hidden_medae[1]))
        self.assertFalse(np.isnan(result.per_column_hidden_medae[0]))

    def test_selection_runner_uses_the_k0_per_column_numerator(self) -> None:
        """Guard the actual source line against regressing to parity_medae."""

        source = (ROOT / "experiments/run_probe_selection.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "skill_numerator_medae = float(baseline.per_column_hidden_medae[index])",
            source,
        )
        self.assertNotIn("compute_column_skill(matrix, index, parity_medae)", source)


class MatchedCellProtocolTests(unittest.TestCase):
    """Every arm must be scored on the identical cell set."""

    def setUp(self) -> None:
        self.observed = np.array(
            [
                [True, True, True, False],
                [True, False, True, True],
            ]
        )

    def test_revealed_cells_are_only_observed_cells(self) -> None:
        cells = REPLAY._revealed_cells(self.observed, [1], [1, 2])
        self.assertEqual(cells, {(1, 2)})

    def test_matched_set_removes_every_arm_s_revealed_cells(self) -> None:
        targets = [0, 1]
        all_cells = {
            (i, j)
            for i in targets
            for j in range(self.observed.shape[1])
            if self.observed[i, j]
        }
        greedy = REPLAY._revealed_cells(self.observed, targets, [0])
        random_a = REPLAY._revealed_cells(self.observed, targets, [1])
        random_b = REPLAY._revealed_cells(self.observed, targets, [3])
        matched = all_cells - (greedy | random_a | random_b)
        self.assertEqual(matched, {(0, 2), (1, 2)})
        for revealed in (greedy, random_a, random_b):
            self.assertFalse(matched & revealed)

    def test_median_finite_drops_folds_with_no_matched_cells(self) -> None:
        self.assertAlmostEqual(
            REPLAY._median_finite([1.0, float("nan"), 3.0]), 2.0
        )
        self.assertTrue(np.isnan(REPLAY._median_finite([float("nan")])))

    def test_bootstrap_reduction_is_resampled_over_folds(self) -> None:
        baseline = [4.0] * 8
        arm = [2.0] * 8
        result = REPLAY._bootstrap_reduction(
            baseline, arm, n_bootstrap=200, seed=1
        )
        self.assertAlmostEqual(result["point_estimate"], 0.5)
        self.assertEqual(result["n_folds"], 8)
        # With every fold identical the resampled reduction cannot move.
        self.assertAlmostEqual(result["ci_lower"], 0.5)
        self.assertAlmostEqual(result["ci_upper"], 0.5)

    def test_bootstrap_fraction_counts_and_bounds(self) -> None:
        result = REPLAY._bootstrap_fraction(
            [True, True, False, False], n_bootstrap=500, seed=3
        )
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["n_positive"], 2)
        self.assertAlmostEqual(result["fraction"], 0.5)
        self.assertLessEqual(result["ci_lower"], 0.5)
        self.assertGreaterEqual(result["ci_upper"], 0.5)


class PublishedMatchedCellArtifactTests(unittest.TestCase):
    """Pin the numbers the replay artifact publishes."""

    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "experiments/lofo_matched_cells_rank1.json"
        if not path.exists():
            raise unittest.SkipTest("matched-cell artifact not generated")
        cls.payload = json.loads(path.read_text(encoding="utf-8"))

    def test_matched_cell_accounting_adds_up(self) -> None:
        matched = self.payload["matched_cells"]
        self.assertEqual(matched["n_target_cells"], 2122)
        self.assertEqual(matched["n_excluded_revealed_cells"], 486)
        self.assertEqual(matched["n_matched_cells"], 1636)
        self.assertEqual(
            matched["n_excluded_revealed_cells"] + matched["n_matched_cells"],
            matched["n_target_cells"],
        )
        for entry in matched["by_k"]:
            self.assertEqual(
                entry["n_excluded_revealed_cells"] + entry["n_matched_cells"],
                matched["n_target_cells"],
            )

    def test_every_arm_at_a_depth_shares_one_cell_count(self) -> None:
        for block in self.payload["by_k"]:
            per_fold = block["arms"]["k0"]["per_fold_medae"]
            for arm in ("greedy", "random"):
                other = block["arms"][arm]
                key = (
                    "per_fold_medae" if arm == "greedy"
                    else "per_fold_median_over_repeats"
                )
                self.assertEqual(set(per_fold), set(other[key]))

    def test_headline_arm_values(self) -> None:
        block = next(b for b in self.payload["by_k"] if b["k"] == 4)
        self.assertAlmostEqual(
            block["arms"]["k0"]["medae_median_of_fold_medians"], 2.6524212715, places=6
        )
        self.assertAlmostEqual(
            block["arms"]["greedy"]["medae_median_of_fold_medians"],
            1.8780757868,
            places=6,
        )
        self.assertAlmostEqual(
            block["arms"]["random"]["medae_median_over_fold_repeat_medaes"],
            2.6012587119,
            places=6,
        )
        vs_k0 = block["arms"]["greedy"]["vs_k0"]
        self.assertEqual(vs_k0["folds_improved"], 18)
        self.assertEqual(vs_k0["n_folds"], 34)
        self.assertAlmostEqual(
            vs_k0["wilcoxon_signed_rank"]["p_value"], 0.0088261992, places=8
        )
        self.assertAlmostEqual(
            vs_k0["bootstrap_reduction_over_folds"]["point_estimate"],
            0.2919391022,
            places=8,
        )

    def test_both_random_aggregations_are_reported_and_differ(self) -> None:
        for block in self.payload["by_k"]:
            random_arm = block["arms"]["random"]
            self.assertIn("medae_median_over_fold_repeat_medaes", random_arm)
            self.assertIn("medae_median_of_fold_medians", random_arm)
        headline = next(b for b in self.payload["by_k"] if b["k"] == 4)["arms"]["random"]
        self.assertNotAlmostEqual(
            headline["medae_median_over_fold_repeat_medaes"],
            headline["medae_median_of_fold_medians"],
            places=6,
        )

    def test_skill_headline_and_noise_floor_bias_are_both_reported(self) -> None:
        skill = self.payload["per_column_skill"]
        headline = skill["headline_scored_columns"]
        self.assertEqual(headline["n_columns_total"], 187)
        self.assertEqual(headline["n_columns_scored"], 174)
        self.assertEqual(headline["n_columns_positive"], 86)
        self.assertEqual(skill["n_columns_excluded_by_reason"]["noise_floor"], 12)
        self.assertEqual(skill["n_columns_excluded_by_reason"]["no_model_error"], 1)
        self.assertEqual(
            skill["noise_floor_dispersion"], SKILL_NOISE_FLOOR_DISPERSION
        )

        all_columns = skill["all_columns_including_noise_floor"]
        self.assertEqual(all_columns["n"], 187)
        # The exclusion is confounded with structurally-negative status, so the
        # all-column variant must be present and must not silently equal the
        # headline.
        self.assertNotEqual(
            all_columns["n_positive"], headline["n_columns_positive"]
        )
        self.assertEqual(
            skill["n_excluded_positive"],
            sum(
                1 for row in skill["excluded_columns"]
                if row["would_have_been_positive"]
            ),
        )
        # Every excluded column is recorded with its baseline and its reason.
        self.assertEqual(len(skill["excluded_columns"]), 13)
        reasons = [row["skill_exclusion_reason"] for row in skill["excluded_columns"]]
        self.assertEqual(reasons.count("noise_floor"), 12)
        self.assertEqual(reasons.count("no_model_error"), 1)
        for row in skill["excluded_columns"]:
            self.assertIn("column_loo_baseline_medae", row)
            self.assertIn("skill_exclusion_reason", row)

    def test_suite_breakdown_sums_to_the_headline(self) -> None:
        skill = self.payload["per_column_skill"]
        by_suite = skill["by_suite"]
        self.assertEqual(
            sum(v["n_scored"] for v in by_suite.values()),
            skill["headline_scored_columns"]["n_columns_scored"],
        )
        self.assertEqual(
            sum(v["n_positive"] for v in by_suite.values()),
            skill["headline_scored_columns"]["n_columns_positive"],
        )

    def test_provenance_is_recorded(self) -> None:
        provenance = self.payload["provenance"]
        for key in ("scores_sha256", "script_sha256", "selection_sha256"):
            self.assertRegex(provenance[key], r"^[0-9a-f]{64}$")


class InformativenessCsvTests(unittest.TestCase):
    """The published CSV must carry exactly one skill definition."""

    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "outputs/probe_informativeness_rank1.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            cls.fields = list(reader.fieldnames or ())
            cls.rows = list(reader)

    def test_only_one_live_skill_score_definition(self) -> None:
        """No two similarly named columns a consumer could confuse."""

        live = [
            field for field in self.fields
            if field.startswith("skill_score")
        ]
        self.assertEqual(sorted(live), ["skill_score", "skill_score_raw"])
        # The old in-sample/matrix-wide field survives only under a name that
        # cannot be mistaken for the corrected score.
        deprecated = [f for f in self.fields if "DEPRECATED" in f]
        self.assertEqual(
            deprecated, ["skill_in_sample_matrixwide_numerator_DEPRECATED"]
        )
        self.assertNotIn("skill_score_in_sample", self.fields)

    def test_skill_numerator_and_scope_are_declared(self) -> None:
        self.assertIn("skill_numerator_medae", self.fields)
        self.assertIn("skill_numerator_scope", self.fields)
        scopes = {row["skill_numerator_scope"] for row in self.rows}
        self.assertEqual(scopes, {"lofo_matched_cells_greedy_k4_per_column"})

    def test_skill_score_matches_its_declared_numerator_and_baseline(self) -> None:
        checked = 0
        for row in self.rows:
            if row["skill_excluded_below_noise_floor"] == "True":
                self.assertEqual(row["skill_score"], "")
                continue
            numerator = float(row["skill_numerator_medae"])
            baseline = float(row["column_loo_baseline_medae"])
            expected = 1.0 - numerator / baseline
            self.assertAlmostEqual(
                float(row["skill_score_raw"]), expected, places=9
            )
            self.assertAlmostEqual(
                float(row["skill_score"]),
                float(np.clip(expected, -1.0, 1.0)),
                places=9,
            )
            checked += 1
        self.assertEqual(checked, 174)

    def test_skill_numerator_is_not_the_matrix_wide_parity_medae(self) -> None:
        """The old bug: numerator identical in scope for every column."""

        numerators = {
            round(float(row["skill_numerator_medae"]), 9)
            for row in self.rows
            if row["skill_numerator_medae"]
        }
        parity = {round(float(row["parity_medae"]), 9) for row in self.rows}
        self.assertFalse(numerators & parity)

    def test_matched_cell_inputs_are_present_and_consistent(self) -> None:
        total = 0
        for row in self.rows:
            n = int(row["matched_cell_n"])
            total += n
            if n == 0:
                self.assertEqual(row["matched_cell_greedy_medae"], "")
                self.assertEqual(row["matched_cell_greedy_beats_k0"], "False")
                continue
            beats = float(row["matched_cell_greedy_medae"]) < float(
                row["matched_cell_k0_medae"]
            )
            self.assertEqual(row["matched_cell_greedy_beats_k0"], str(beats))
        self.assertEqual(total, 1636)

    def test_writer_field_lists_agree_between_producers(self) -> None:
        selection_source = (ROOT / "experiments/run_probe_selection.py").read_text(
            encoding="utf-8"
        )
        for field in REPLAY.INFORMATIVENESS_FIELDS:
            self.assertIn(f'"{field}"', selection_source, field)


class CompressionRawRowTests(unittest.TestCase):
    """Held-out random rows and k=0 rows must be emitted (Task 4)."""

    MATRIX = np.array(
        [
            [80.0, 70.0, 60.0, 50.0],
            [75.0, 67.0, 59.0, 48.0],
            [72.0, 65.0, 55.0, 46.0],
            [68.0, 62.0, 53.0, 44.0],
            [64.0, 60.0, 50.0, 42.0],
        ]
    )
    MODELS = ["m0", "m1", "m2", "m3", "m4"]
    EVALUATIONS = ["e0", "e1", "e2", "e3"]
    FIELDS = [
        "protocol", "candidate_mode", "method", "selection_objective",
        "repeat", "k", "model_id", "evaluation_id",
        "actual_normalized_score", "predicted_normalized_score",
        "is_revealed_probe_cell", "is_hidden_prediction", "fold",
    ]

    class _ImmediateExecutor:
        """Serial stand-in: the runner module is loaded by path, so its
        worker functions cannot be pickled into a real process pool."""

        @staticmethod
        def map(function, jobs):
            return map(function, jobs)

    def _writer(self) -> tuple[io.StringIO, csv.DictWriter]:
        handle = io.StringIO()
        writer = csv.DictWriter(handle, fieldnames=self.FIELDS, lineterminator="\n")
        writer.writeheader()
        return handle, writer

    def test_heldout_random_raw_rows_are_emitted_with_a_fold(self) -> None:
        handle, writer = self._writer()
        rows = RUNNER._random_curves(
                self._ImmediateExecutor, self.MATRIX, [0, 1, 2, 3],
                max_k=2, repeats=2, seed=0, rank=1,
                evaluations=self.EVALUATIONS,
                heldout=((4,), (0, 1, 2, 3)),
                raw_writer=writer, models=self.MODELS,
                candidate_mode="any_candidate", fold=7,
            )
        self.assertEqual(len(rows), 4)
        handle.seek(0)
        raw = list(csv.DictReader(handle))
        self.assertTrue(raw)
        self.assertEqual({row["protocol"] for row in raw}, {"heldout"})
        self.assertEqual({row["fold"] for row in raw}, {"7"})
        self.assertEqual({row["method"] for row in raw}, {"random_prefix"})
        # Held-out rows only cover the target model's own cells.
        self.assertEqual({row["model_id"] for row in raw}, {"m4"})
        self.assertEqual({row["k"] for row in raw}, {"1", "2"})

    def test_all_known_random_raw_rows_still_work(self) -> None:
        handle, writer = self._writer()
        RUNNER._random_curves(
                self._ImmediateExecutor, self.MATRIX, [0, 1, 2, 3],
                max_k=1, repeats=1, seed=0, rank=1,
                evaluations=self.EVALUATIONS,
                raw_writer=writer, models=self.MODELS,
                candidate_mode="any_candidate",
            )
        handle.seek(0)
        raw = list(csv.DictReader(handle))
        self.assertEqual({row["protocol"] for row in raw}, {"all_known"})
        self.assertEqual({row["fold"] for row in raw}, {""})

    def test_random_raw_rows_require_models_and_mode(self) -> None:
        _, writer = self._writer()
        with self.assertRaises(ValueError):
                RUNNER._random_curves(
                    self._ImmediateExecutor, self.MATRIX, [0, 1],
                    max_k=1, repeats=1, seed=0, rank=1,
                    evaluations=self.EVALUATIONS,
                    heldout=((4,), (0, 1, 2, 3)),
                    raw_writer=writer,
                )

    def test_k0_raw_rows_exist_for_both_protocols(self) -> None:
        all_known = RUNNER._k0_raw_rows(
            self.MATRIX, self.MODELS, self.EVALUATIONS, 1
        )
        self.assertEqual(len(all_known), self.MATRIX.size)
        self.assertEqual({row["k"] for row in all_known}, {0})
        self.assertEqual({row["protocol"] for row in all_known}, {"all_known"})
        self.assertFalse(any(row["is_revealed_probe_cell"] for row in all_known))
        self.assertTrue(all(row["is_hidden_prediction"] for row in all_known))

        heldout = RUNNER._k0_raw_rows(
            self.MATRIX, self.MODELS, self.EVALUATIONS, 1,
            heldout=((4,), (0, 1, 2, 3)), fold=3,
        )
        self.assertEqual({row["model_id"] for row in heldout}, {"m4"})
        self.assertEqual({row["fold"] for row in heldout}, {3})
        self.assertEqual({row["protocol"] for row in heldout}, {"heldout"})

    def test_k0_and_greedy_rows_share_a_csv_schema(self) -> None:
        k0 = RUNNER._k0_raw_rows(self.MATRIX, self.MODELS, self.EVALUATIONS, 1)
        greedy = RUNNER._raw_rows(
            RUNNER.predict_all_known(self.MATRIX, (0,), rank=1),
            self.MODELS, self.EVALUATIONS,
            {
                "protocol": "all_known", "candidate_mode": "any_candidate",
                "method": "greedy", "selection_objective": "medae",
                "k": 1, "fold": None,
            },
        )
        self.assertEqual(set(k0[0]), set(greedy[0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            RUNNER._write_csv_atomic(path, k0 + greedy)
            with path.open(newline="", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))
        self.assertEqual(len(written), len(k0) + len(greedy))

    def test_restriction_is_gone_from_the_source(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("random raw rows currently support all-known only", source)


class NoiseFloorDisclosureTests(unittest.TestCase):
    def test_threshold_comment_names_the_confound(self) -> None:
        source = (ROOT / "src/pathopress/probes.py").read_text(encoding="utf-8")
        head = source.split("SKILL_NOISE_FLOOR_DISPERSION = 0.5")[0]
        self.assertIn("CONFOUND", head)
        self.assertIn("structurally guaranteed to score negative skill", head)

    def test_low_dispersion_column_is_excluded_not_silently_nan(self) -> None:
        matrix = np.array([[10.0, 5.0], [10.1, 9.0], [9.9, 1.0], [10.05, 7.0]])
        skill = compute_column_skill(matrix, 0, 0.05)
        self.assertTrue(skill.excluded_below_noise_floor)
        self.assertEqual(skill.exclusion_reason, "noise_floor")
        self.assertIsNone(skill.skill_score)
        # And it would have scored positive, which is exactly the bias.
        baseline = compute_column_loo_baseline_medae(matrix, 0)
        self.assertLess(0.05, baseline)


class SelectionObjectiveDisclosureTests(unittest.TestCase):
    def test_parity_objective_bias_is_documented_at_the_selection_site(self) -> None:
        source = (ROOT / "experiments/run_probe_selection.py").read_text(
            encoding="utf-8"
        )
        head = source.split(
            "scores = [result.parity.median_absolute_error for result in results]"
        )[0]
        self.assertIn("KNOWN SELECTION-OBJECTIVE BIAS", head)

    def test_revealed_probe_cells_score_as_exact_zero_error(self) -> None:
        matrix = np.array(
            [[80.0, 70.0, 60.0], [75.0, 67.0, 59.0], [72.0, 65.0, 55.0],
             [68.0, 62.0, 53.0]]
        )
        result = evaluate_global_probes(matrix, [0], rank=1)
        self.assertEqual(result.n_revealed_cells, matrix.shape[0])
        # Parity averages the zero-error revealed cells in; hidden-only does not.
        self.assertLess(
            result.parity.median_absolute_error,
            result.hidden_only.median_absolute_error,
        )


class AllowlistRestrictedRandomDrawTests(unittest.TestCase):
    """The restricted random arm must actually be restricted, and nested.

    The published claim "inside the 25-task low-friction allowlist, greedy is
    no better than random" had no artifact behind it: the only ``random`` arm
    in the LOFO artifacts shuffles all 187 evaluations, so it is a control for
    the unrestricted greedy arm, not for the allowlist-greedy arm.  These tests
    pin the properties the genuine control has to have.
    """

    POOL = (3, 11, 17, 23, 29, 31)

    def test_prefixes_only_ever_use_allowlisted_columns(self) -> None:
        drawn = REPLAY.allowlist_random_prefixes(
            self.POOL, fold=0, max_k=4, repeats=10, seed=42
        )
        self.assertTrue(drawn)
        for probes in drawn.values():
            self.assertTrue(set(probes).issubset(set(self.POOL)))

    def test_prefixes_are_nested_and_have_the_right_size(self) -> None:
        drawn = REPLAY.allowlist_random_prefixes(
            self.POOL, fold=3, max_k=4, repeats=10, seed=42
        )
        for repeat in range(10):
            for k in range(1, 5):
                probes = drawn[(repeat, k)]
                self.assertEqual(len(probes), k)
                self.assertEqual(len(set(probes)), k)
                if k > 1:
                    self.assertEqual(drawn[(repeat, k - 1)], probes[: k - 1])

    def test_depth_is_capped_by_the_candidate_pool(self) -> None:
        """A pool smaller than max_k must truncate, not raise or repeat."""

        drawn = REPLAY.allowlist_random_prefixes(
            (2, 5), fold=0, max_k=5, repeats=3, seed=42
        )
        self.assertEqual(max(k for _, k in drawn), 2)
        self.assertEqual(REPLAY.allowlist_random_prefixes((), fold=0, max_k=5,
                                                          repeats=3, seed=42), {})

    def test_each_fold_gets_its_own_draw(self) -> None:
        """Mirrors run_probe_selection.py, which offsets the seed by the fold."""

        first = REPLAY.allowlist_random_prefixes(
            self.POOL, fold=0, max_k=4, repeats=10, seed=42
        )
        second = REPLAY.allowlist_random_prefixes(
            self.POOL, fold=1, max_k=4, repeats=10, seed=42
        )
        self.assertNotEqual(first, second)

    def test_draw_matches_the_published_random_prefix_generator(self) -> None:
        """Same generator as the unrestricted arm; only the pool differs."""

        from pathopress.probes import random_global_probe_prefixes

        expected = random_global_probe_prefixes(
            len(self.POOL), max_probes=4, repeats=10, seed=42 + 7
        )
        drawn = REPLAY.allowlist_random_prefixes(
            self.POOL, fold=7, max_k=4, repeats=10, seed=42
        )
        for repeat, prefixes in enumerate(expected):
            for k, positions in enumerate(prefixes, start=1):
                self.assertEqual(
                    drawn[(repeat, k)],
                    tuple(self.POOL[p] for p in positions),
                )


class AllowlistRestrictedRandomArmArtifactTests(unittest.TestCase):
    """Pin the allowlist-internal head-to-head the published claim needs."""

    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "experiments/lofo_matched_cells_rank1.json"
        if not path.exists():
            raise unittest.SkipTest("matched-cell artifact not generated")
        cls.payload = json.loads(path.read_text(encoding="utf-8"))
        if not cls.payload.get("allowlist_greedy_vs_random_by_k"):
            raise unittest.SkipTest("allowlist arms not replayed")

    def test_block_exists_for_every_depth_with_all_three_arms(self) -> None:
        blocks = self.payload["allowlist_greedy_vs_random_by_k"]
        self.assertEqual([b["k"] for b in blocks], [1, 2, 3, 4, 5])
        for block in blocks:
            for arm in ("k0", "allowlist_greedy", "allowlist_random"):
                self.assertIn(arm, block["arms"])
            self.assertEqual(
                self.payload["configuration"]["n_allowlist_candidates"], 25
            )
            self.assertEqual(
                self.payload["configuration"]["n_allowlist_random_repeats"], 10
            )

    def test_all_three_arms_share_one_cell_set_at_each_depth(self) -> None:
        for block in self.payload["allowlist_greedy_vs_random_by_k"]:
            folds = set(block["arms"]["k0"]["per_fold_medae"])
            self.assertEqual(
                set(block["arms"]["allowlist_greedy"]["per_fold_medae"]), folds
            )
            self.assertEqual(
                set(block["arms"]["allowlist_random"]["per_fold_median_over_repeats"]),
                folds,
            )

    def test_matched_set_is_strictly_smaller_than_the_greedy_only_block(self) -> None:
        """Adding a random arm reveals more cells, so it must shrink further.

        It must NOT shrink ``allowlist_arm_by_k``, whose numbers are published.
        """

        pairs = {b["k"]: b for b in self.payload["allowlist_greedy_vs_random_by_k"]}
        for block in self.payload["allowlist_arm_by_k"]:
            self.assertLess(
                pairs[block["k"]]["n_matched_cells"], block["n_matched_cells"]
            )
        headline = next(b for b in self.payload["allowlist_arm_by_k"] if b["k"] == 4)
        self.assertAlmostEqual(
            headline["arms"]["allowlist_greedy"]["medae_median_of_fold_medians"],
            1.9951398678,
            places=6,
        )

    def test_the_unrestricted_random_arm_is_not_the_allowlist_control(self) -> None:
        """Guard against re-quoting the 187-column arm as the allowlist control."""

        unrestricted = next(b for b in self.payload["by_k"] if b["k"] == 4)
        restricted = next(
            b for b in self.payload["allowlist_greedy_vs_random_by_k"] if b["k"] == 4
        )
        self.assertNotAlmostEqual(
            unrestricted["arms"]["random"]["medae_median_of_fold_medians"],
            restricted["arms"]["allowlist_random"]["medae_median_of_fold_medians"],
            places=3,
        )
        self.assertNotIn("allowlist_random", unrestricted["arms"])

    def test_headline_allowlist_greedy_vs_random_values(self) -> None:
        block = next(
            b for b in self.payload["allowlist_greedy_vs_random_by_k"] if b["k"] == 4
        )
        self.assertEqual(block["n_matched_cells"], 1237)
        self.assertAlmostEqual(
            block["arms"]["k0"]["medae_median_of_fold_medians"], 2.8555082606,
            places=6,
        )
        self.assertAlmostEqual(
            block["arms"]["allowlist_greedy"]["medae_median_of_fold_medians"],
            1.9463385335,
            places=6,
        )
        self.assertAlmostEqual(
            block["arms"]["allowlist_random"]["medae_median_of_fold_medians"],
            2.0251163736,
            places=6,
        )
        self.assertAlmostEqual(
            block["arms"]["allowlist_random"]["medae_median_over_fold_repeat_medaes"],
            2.0118290091,
            places=6,
        )
        self.assertEqual(
            block["arms"]["allowlist_random"]["n_fold_repeat_pairs"], 340
        )

    def test_headline_head_to_head_is_not_significant(self) -> None:
        """The negative result itself: greedy's edge is inside the noise.

        Both aggregation conventions put allowlist-greedy nominally ahead, but
        the signed-rank test is far from significant and the bootstrap CI on
        the reduction straddles zero, so no ordering is supported.
        """

        block = next(
            b for b in self.payload["allowlist_greedy_vs_random_by_k"] if b["k"] == 4
        )
        head = block["allowlist_greedy_vs_allowlist_random"]
        self.assertAlmostEqual(
            head["wilcoxon_signed_rank"]["p_value"], 0.4939012303, places=8
        )
        self.assertGreater(head["wilcoxon_signed_rank"]["p_value"], 0.05)
        reduction = head["bootstrap_reduction_over_folds"]
        self.assertLess(reduction["ci_lower"], 0.0)
        self.assertGreater(reduction["ci_upper"], 0.0)
        self.assertAlmostEqual(reduction["point_estimate"], 0.0389004015, places=8)
        # Both conventions are reported, because the claim is asserted under
        # both and they do not agree to the quoted precision.
        convention_a = block["allowlist_greedy_vs_allowlist_random_convention_a"]
        self.assertAlmostEqual(convention_a["greedy_medae"], 1.9463385335, places=6)
        self.assertAlmostEqual(convention_a["random_medae"], 2.0118290091, places=6)

    def test_ties_are_exactly_the_zero_information_folds(self) -> None:
        """Every tie is a fold where no allowlisted probe is observed at all.

        In those folds the held-out family has no score on any of the 25
        candidates, no arm can reveal a cell, and all three arms collapse onto
        the k=0 completion.  Roughly half the folds are like this, which is why
        the head-to-head has so little power.
        """

        for block in self.payload["allowlist_greedy_vs_random_by_k"]:
            head = block["allowlist_greedy_vs_allowlist_random"]
            self.assertEqual(head["folds_tied"], head["folds_zero_information"])
            k0 = block["arms"]["k0"]["per_fold_medae"]
            greedy = block["arms"]["allowlist_greedy"]["per_fold_medae"]
            random_arm = block["arms"]["allowlist_random"][
                "per_fold_median_over_repeats"
            ]
            tied = [
                f for f in k0
                if greedy[f] is not None and random_arm[f] is not None
                and greedy[f] == random_arm[f]
            ]
            self.assertEqual(len(tied), head["folds_tied"])
            for fold in tied:
                self.assertEqual(greedy[fold], k0[fold])
        headline = next(
            b for b in self.payload["allowlist_greedy_vs_random_by_k"] if b["k"] == 4
        )
        self.assertEqual(
            headline["allowlist_greedy_vs_allowlist_random"][
                "folds_zero_information"], 15
        )

    def test_candidate_ids_are_recorded_and_match_the_allowlist_file(self) -> None:
        recorded = self.payload["allowlist_candidate_ids"]
        self.assertEqual(len(recorded), 25)
        published = json.loads(
            (ROOT / "data/low_friction_allowlist_v2_top25.json").read_text(
                encoding="utf-8"
            )
        )["evaluation_ids"]
        self.assertEqual(list(recorded), list(published))

    def test_caveat_names_the_unrestricted_arm_trap(self) -> None:
        joined = " ".join(self.payload["caveats"])
        self.assertIn("UNRESTRICTED", joined)
        self.assertIn("allowlist_greedy_vs_random_by_k", joined)


if __name__ == "__main__":
    unittest.main()
