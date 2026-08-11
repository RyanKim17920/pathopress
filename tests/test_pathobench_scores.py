import csv
import importlib.util
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PathoBenchScoreEvidenceTests(unittest.TestCase):
    def test_exaone_snapshot_is_complete_and_protocol_specific(self) -> None:
        path = ROOT / "source_data/exaone_path_2_5_pathobench_2512.14019v1.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 560)
        self.assertEqual(len({row["evaluation_id"] for row in rows}), 80)
        self.assertEqual(len({row["model_alias"] for row in rows}), 7)
        self.assertTrue(all(row["evaluation_id"].startswith("pathobench.exaone2025.") for row in rows))
        self.assertTrue(all(row["evaluation_id"] != row["base_evaluation_id"] for row in rows))
        self.assertEqual(Counter(row["metric"] for row in rows), {"macro-ovr-auc": 504, "cindex": 56})
        anchor = next(
            row for row in rows
            if row["source_task"] == "panda_isup_grade" and row["model_alias"] == "EXAONE Path 2.5"
        )
        self.assertEqual(anchor["value"], "0.956")
        self.assertEqual(anchor["metric"], "macro-ovr-auc")

    def test_threads_public_snapshot_excludes_internal_protocols(self) -> None:
        path = ROOT / "source_data/threads_pathobench_2501.16652v1.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 336)
        self.assertEqual(len({row["evaluation_id"] for row in rows}), 42)
        self.assertEqual(len({row["model_alias"] for row in rows}), 8)
        self.assertTrue(all(row["evaluation_id"].startswith("pathobench.threads2025.") for row in rows))
        self.assertNotIn("27", {row["source_table"] for row in rows})

    def test_registry_parser_emits_all_exaone_cells(self) -> None:
        registry = load_script("build_registry_test", ROOT / "scripts/build_registry.py")
        source = Path("/tmp/pathopress_sources")
        if not (source / "pathobench_hf").is_dir():
            self.skipTest("pinned upstream source clones are unavailable")
        tasks = [
            *registry.build_pathobench(source / "pathobench_hf", registry.git(source / "pathobench_hf", "rev-parse", "HEAD")),
            *registry.build_eva(source / "eva", registry.git(source / "eva", "rev-parse", "HEAD")),
            *registry.build_thunder(source / "thunder", registry.git(source / "thunder", "rev-parse", "HEAD")),
            *registry.build_hest(registry.git(source / "hest", "rev-parse", "HEAD")),
            *registry.build_pathorob(registry.git(source / "pathorob", "rev-parse", "HEAD")),
        ]
        registry.materialize_task_contracts(tasks)
        snapshot = ROOT / "source_data/exaone_path_2_5_pathobench_2512.14019v1.csv"
        tasks.extend(registry.build_exaone_pathobench_protocols(snapshot, tasks))
        scores, aliases = registry.parse_exaone_pathobench_scores(snapshot, tasks)
        self.assertEqual((len(scores), len(aliases)), (560, 7))


if __name__ == "__main__":
    unittest.main()
