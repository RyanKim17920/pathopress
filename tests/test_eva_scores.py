import tempfile
import unittest
from pathlib import Path

from tests.pinned_sources import missing_inputs, sources_root

from scripts.evidence.eva_scores import (
    merge_scores,
    parse_midnight_scores,
    parse_repository_scores,
    required_additional_protocols,
)


class EvaScoreExtractionTests(unittest.TestCase):
    def test_official_pinned_sources_have_expected_shape(self) -> None:
        eva = sources_root() / "eva"
        midnight = sources_root() / "eva_midnight"
        if not eva.is_dir() or not midnight.is_dir():
            missing_inputs(self, "pinned upstream source clones are unavailable")
        repository = parse_repository_scores(eva, "e43e74a99b75660b0014f790f25a33dd9f11e121")
        report = parse_midnight_scores(midnight, "adc6b15679c981cce6f9b018bbad09d16eeeda9f")
        selected, duplicates = merge_scores(repository, report)
        self.assertEqual(len(repository), 195)
        self.assertEqual(len(report), 180)
        self.assertEqual(len(selected), 265)
        self.assertEqual(len(duplicates), 110)
        self.assertEqual(len({(row.model_id, row.evaluation_id) for row in selected}), 265)
        self.assertEqual(len(required_additional_protocols()), 15)
        self.assertEqual(
            {row.metric for row in selected if "consep" not in row.evaluation_id and "monusac" not in row.evaluation_id},
            {"balanced_accuracy"},
        )

    def test_protocol_columns_are_not_collapsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = root / "tools/data/leaderboards"
            relative.mkdir(parents=True)
            columns = ["model", *(
                "bach,breakhis,crc,gleason_arvaniti,mhist,patch_camelyon,patch_camelyon/test,"
                "camelyon16_small,camelyon16_small/test,panda_small,panda_small/test,consep,monusac"
            ).split(",")]
            (relative / "pathology.csv").write_text(
                ",".join(columns) + "\n" + "paige_virchow2," + ",".join(["0.8"] * 13) + "\n",
                encoding="utf-8",
            )
            scores = parse_repository_scores(root, "deadbeef", strict=False)
        ids = {row.evaluation_id for row in scores}
        self.assertIn("eva.leaderboard.patch_camelyon.validation", ids)
        self.assertIn("eva.leaderboard.patch_camelyon.test", ids)
        self.assertIn("eva.leaderboard.camelyon16_small.validation", ids)
        self.assertIn("eva.leaderboard.camelyon16_small.test", ids)
        self.assertIn("eva.leaderboard.mhist.test", ids)
        self.assertIn("eva.leaderboard.monusac.test", ids)

    def test_current_repository_wins_duplicate_without_losing_audit(self) -> None:
        eva = sources_root() / "eva"
        midnight = sources_root() / "eva_midnight"
        if not eva.is_dir() or not midnight.is_dir():
            missing_inputs(self, "pinned upstream source clones are unavailable")
        repository = parse_repository_scores(eva, "repo")
        report = parse_midnight_scores(midnight, "report")
        selected, duplicates = merge_scores(repository, report)
        chosen = next(
            row for row in selected
            if row.model_id == "virchow-2" and row.evaluation_id == "eva.leaderboard.bach.validation"
        )
        alternate = next(
            row for row in duplicates
            if row["model_id"] == "virchow-2" and row["evaluation_id"] == "eva.leaderboard.bach.validation"
        )
        self.assertEqual(chosen.value, 0.883)
        self.assertEqual(alternate["alternate_value"], 0.890)


if __name__ == "__main__":
    unittest.main()
