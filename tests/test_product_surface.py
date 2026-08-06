import csv
import io
import json
import mimetypes
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np

from pathopress.cli import main as cli_main
from pathopress.prediction import (
    build_deployment_confidence_artifact,
    load_confidence_artifact,
    load_prediction_dataset,
    parse_known_scores,
    predict_new_model,
)
from pathopress.new_model_confidence import build_new_model_confidence_artifact
from pathopress.public_data import (
    build_public_export,
    download_public_export,
    load_public_export,
)


SCORE_FIELDS = (
    "model_id",
    "reported_model_alias",
    "model_revision",
    "evaluation_id",
    "value",
    "normalized_score",
    "suite_id",
    "metric",
    "reference_url",
    "source_locator",
    "extraction_date",
    "review_status",
    "uncertainty",
    "lineage",
    "audit_status",
)


class ProductFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scores = self.root / "scores.csv"
        rows = []
        for model_index, model in enumerate(("m1", "m2", "m3", "m4")):
            for evaluation_index, evaluation in enumerate(("e1", "e2", "e3")):
                if model == "m1" and evaluation == "e3":
                    continue
                value = 20 + model_index * 10 + evaluation_index * 5
                rows.append(
                    {
                        "model_id": model,
                        "reported_model_alias": model,
                        "model_revision": "r1",
                        "evaluation_id": evaluation,
                        "value": value / 100,
                        "normalized_score": value,
                        "suite_id": "suite",
                        "metric": "auc",
                        "reference_url": f"https://example.test/{model}/{evaluation}",
                        "source_locator": "table=1",
                        "extraction_date": "2026-01-01",
                        "review_status": "machine_parsed_single_source",
                        "uncertainty": "not_reported",
                        "lineage": "fixture",
                        "audit_status": "parsed_primary_source",
                    }
                )
        rows.append(
            {
                **rows[0],
                "model_id": "external-only",
                "reported_model_alias": "external-only",
                "audit_status": "reported_external",
            }
        )
        with self.scores.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        self.tasks = self.root / "tasks.csv"
        task_fields = [
            "evaluation_id", "suite_id", "dataset_id", "task_name", "task_family",
            "target", "sample_unit", "task_type", "num_samples", "endpoint", "metric",
            "direction", "protocol", "reference_url", "audit_status",
        ]
        with self.tasks.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=task_fields)
            writer.writeheader()
            for evaluation in ("e1", "e2", "e3"):
                writer.writerow(
                    {
                        "evaluation_id": evaluation, "suite_id": "suite", "dataset_id": "d",
                        "task_name": evaluation, "task_family": "classification", "target": "label",
                        "sample_unit": "case", "task_type": "classification", "num_samples": 10,
                        "endpoint": "classification", "metric": "auc", "direction": "higher",
                        "protocol": "fixture", "reference_url": "https://example.test/task",
                        "audit_status": "parsed_primary_source",
                    }
                )
        self.suites = self.root / "suites.csv"
        with self.suites.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("suite_id", "name", "scope", "task_count", "reference_url", "protocol", "audit_notes")
            )
            writer.writeheader()
            writer.writerow(
                {"suite_id": "suite", "name": "Fixture Suite", "scope": "test", "task_count": 3,
                 "reference_url": "https://example.test/suite", "protocol": "fixture", "audit_notes": "test"}
            )
        self.provenance = self.root / "provenance.json"
        self.provenance.write_text(
            json.dumps({"schema_version": 1, "repositories": {"suite": {"url": "https://example.test", "commit": "abc", "local_path": "/tmp/private"}}}),
            encoding="utf-8",
        )
        self.models = self.root / "model_metadata.csv"
        with self.models.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("model_id", "provider", "family", "model_type", "modality", "parameter_count", "release_date", "primary_source_url", "verification_status"),
            )
            writer.writeheader()
            for model in ("m1", "m2", "m3", "m4"):
                writer.writerow(
                    {"model_id": model, "provider": "lab", "family": "f", "model_type": "tile_encoder",
                     "modality": "vision", "parameter_count": "", "release_date": "2025-01-01",
                     "primary_source_url": "https://example.test/model", "verification_status": "verified"}
                )


class PredictionProductTests(ProductFixture):
    def test_parse_and_predict_new_model_preserves_known_scores(self) -> None:
        known = parse_known_scores(["e1=25", "e2=35"])
        dataset = load_prediction_dataset(
            self.scores, min_scores_per_model=2, min_models_per_evaluation=2
        )
        prediction = predict_new_model(dataset, known)
        self.assertEqual(prediction.shape, (3,))
        self.assertEqual(prediction[dataset.evaluation_index["e1"]], 25)
        self.assertEqual(prediction[dataset.evaluation_index["e2"]], 35)
        self.assertTrue(np.isfinite(prediction).all())

    def test_known_score_parser_rejects_duplicates_and_out_of_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_known_scores(["e1=20,e1=30"])
        with self.assertRaisesRegex(ValueError, r"\[0, 100\]"):
            parse_known_scores(["e1=120"])

    def test_deployment_confidence_collapses_repeated_cells_and_is_hash_bound(self) -> None:
        cells = self.root / "cells.csv"
        with cells.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("model_id", "evaluation_id", "suite_id", "absolute_error"),
            )
            writer.writeheader()
            for error in (1, 3, 5):
                writer.writerow({"model_id": "m1", "evaluation_id": "e1", "suite_id": "suite", "absolute_error": error})
            writer.writerow({"model_id": "m2", "evaluation_id": "e1", "suite_id": "suite", "absolute_error": 7})
        artifact = build_deployment_confidence_artifact(cells, self.scores)
        self.assertEqual(artifact["calibration_cells"]["n_unique_cells"], 2)
        path = self.root / "confidence.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        loaded = load_confidence_artifact(path, self.scores)
        self.assertFalse(loaded["applicability"]["new_model_rows"])
        self.scores.write_text(self.scores.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_confidence_artifact(path, self.scores)

    def _cli(self, *arguments: str) -> str:
        output = io.StringIO()
        argv = [
            "pathopress", *arguments, "--scores", str(self.scores),
            "--min-scores-per-model", "2", "--min-models-per-evaluation", "2",
            "--format", "json",
        ]
        with patch("sys.argv", argv), redirect_stdout(output):
            cli_main()
        return output.getvalue()

    def test_cli_lists_and_predicts_product_modes_as_json(self) -> None:
        models = json.loads(self._cli("list-models"))
        evaluations = json.loads(self._cli("list-evaluations"))
        cell = json.loads(self._cli("predict", "--model", "m1", "--evaluation", "e3"))
        completed = json.loads(self._cli("complete-model", "--model", "m1"))
        records = []
        for index, model in enumerate(("m1", "m2", "m3", "m4")):
            for evaluation in ("e1", "e2", "e3"):
                records.append({
                    "target_model_id": model, "evaluation_id": evaluation,
                    "suite_id": "suite", "k": 1, "source": "leave_one_model_out_probe",
                    "actual": 50.0, "predicted": 51.0 + index,
                    "absolute_error": 1.0 + index, "same_suite_probe_count": 1,
                })
        artifact, _ = build_new_model_confidence_artifact(
            records, self.scores, min_evaluation_models=3, min_context_models=3
        )
        artifact_path = self.root / "new_model_confidence.json"
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        added = json.loads(self._cli(
            "add-model", "--model", "new", "--known-score", "e1=30,e2=40", "--confidence",
            "--new-model-confidence-artifact", str(artifact_path),
        ))
        self.assertEqual(len(models), 4)
        self.assertEqual(len(evaluations), 3)
        self.assertEqual(cell[0]["status"], "predicted")
        self.assertEqual(len(completed), 1)
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["confidence_status"], "calibrated_new_model")
        self.assertIn("lower_90", added[0])
        self.assertEqual(added[0]["calibration_k"], 1)


class PublicExportTests(ProductFixture):
    def _build(self, directory: str) -> Path:
        out = self.root / directory
        build_public_export(
            scores_path=self.scores,
            tasks_path=self.tasks,
            suites_path=self.suites,
            provenance_path=self.provenance,
            model_metadata_path=self.models,
            out_dir=out,
            min_scores_per_model=2,
            min_models_per_evaluation=2,
        )
        return out

    def test_export_is_deterministic_filtered_and_loadable(self) -> None:
        first, second = self._build("first"), self._build("second")
        one, two = load_public_export(first), load_public_export(second)
        self.assertEqual((len(one.models), len(one.evaluations), len(one.scores)), (4, 3, 11))
        self.assertEqual(one.manifest["files"], two.manifest["files"])
        provenance = json.loads((first / "provenance.json").read_text())
        self.assertNotIn("local_path", provenance["repositories"]["suite"])
        self.assertIn("do **not** relicense", (first / "LICENSES.md").read_text())

    def test_downloader_uses_manifest_and_verifies_hashes(self) -> None:
        source = self._build("source")
        destination = self.root / "download"
        release = download_public_export(source.as_uri() + "/", destination)
        self.assertEqual(len(release.scores), 11)
        (destination / "data" / "scores_paper.csv").write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_public_export(destination)


class StaticWebsiteTests(unittest.TestCase):
    def test_generated_data_semantics_and_client_only_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        data = json.loads((root / "website" / "data.json").read_text())
        self.assertEqual(data["schema_version"], "pathopress-static-predictor-v1")
        self.assertEqual((len(data["models"]), len(data["evaluations"])), (59, 165))
        self.assertTrue(any(value is None for row in data["observed"] for value in row))
        for i, row in enumerate(data["observed"]):
            for j, value in enumerate(row):
                if value is not None:
                    self.assertIsNone(data["prediction_intervals"][i][j])
        javascript = (root / "website" / "app.js").read_text()
        html = (root / "website" / "index.html").read_text()
        self.assertIn("function completeRank1", javascript)
        self.assertIn('fetch("data.json")', javascript)
        self.assertNotIn("pyodide", html.lower())
        self.assertNotIn("<form", html.lower())

    def test_site_assets_serve_over_plain_http(self) -> None:
        root = Path(__file__).resolve().parents[1]
        handler = partial(SimpleHTTPRequestHandler, directory=str(root))
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        except PermissionError:
            # Restricted CI sandboxes can deny even loopback sockets. Retain a
            # static-server fallback check instead of hiding the product test.
            self.assertEqual(mimetypes.guess_type(root / "website" / "index.html")[0], "text/html")
            self.assertEqual(mimetypes.guess_type(root / "website" / "data.json")[0], "application/json")
            self.assertTrue((root / "website" / "index.html").is_file())
            return
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/website/") as response:
            self.assertIn(b"PathoPress", response.read())
        with urlopen(base + "/website/data.json") as response:
            self.assertEqual(json.load(response)["meta"]["observations"], 1967)


if __name__ == "__main__":
    unittest.main()
