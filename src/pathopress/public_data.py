"""Deterministic public exports, loader, downloader, and website data builder."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping
from urllib.parse import urljoin
from urllib.request import urlopen

import numpy as np

from .prediction import (
    DEFAULT_RANK,
    DEFAULT_REGULARIZATION,
    calibrated_interval,
    complete_dataset,
    load_confidence_artifact,
    load_prediction_dataset,
    sha256_file,
)
from .new_model_confidence import load_new_model_confidence_artifact


PUBLIC_SCHEMA_VERSION = "pathopress-public-tables-v1"
WEBSITE_SCHEMA_VERSION = "pathopress-static-predictor-v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: list[str]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _clean_provenance(payload: object) -> object:
    """Remove workstation paths while retaining URLs, commits, hashes, and notes."""
    if isinstance(payload, dict):
        return {
            key: _clean_provenance(value)
            for key, value in payload.items()
            if not (
                key == "local_path"
                or key.endswith("_path")
                or key.endswith("_paths")
            )
        }
    if isinstance(payload, list):
        return [_clean_provenance(value) for value in payload]
    return payload


def _model_rows(
    model_ids: Iterable[str], raw_scores: list[dict[str, str]], metadata_path: Path
) -> list[dict[str, object]]:
    metadata = {
        row["model_id"]: row for row in _read_csv(metadata_path)
    } if metadata_path.exists() else {}
    output = []
    for model_id in sorted(set(model_ids)):
        scores = [row for row in raw_scores if row["model_id"] == model_id]
        item = metadata.get(model_id, {})
        output.append(
            {
                "model_id": model_id,
                "provider": item.get("provider", ""),
                "family": item.get("family", ""),
                "model_type": item.get("model_type", ""),
                "modality": item.get("modality", ""),
                "parameter_count": item.get("parameter_count", ""),
                "release_date": item.get("release_date", ""),
                "primary_source_url": item.get("primary_source_url", ""),
                "verification_status": item.get("verification_status", ""),
                "observed_scores": len(scores),
                "represented_suites": ";".join(sorted({row["suite_id"] for row in scores})),
            }
        )
    return output


MODEL_FIELDS = [
    "model_id",
    "provider",
    "family",
    "model_type",
    "modality",
    "parameter_count",
    "release_date",
    "primary_source_url",
    "verification_status",
    "observed_scores",
    "represented_suites",
]


EVALUATION_FIELDS = [
    "evaluation_id",
    "suite_id",
    "dataset_id",
    "task_name",
    "task_family",
    "target",
    "sample_unit",
    "task_type",
    "num_samples",
    "endpoint",
    "metric",
    "direction",
    "protocol",
    "reference_url",
    "audit_status",
    "observed_models",
]


SCORE_FIELDS = [
    "model_id",
    "evaluation_id",
    "value",
    "normalized_score",
    "suite_id",
    "metric",
    "reference_url",
    "source_locator",
    "extraction_date",
    "review_status",
    "audit_status",
    "lineage",
]


def _evaluation_rows(
    evaluation_ids: Iterable[str], raw_scores: list[dict[str, str]], tasks_path: Path
) -> list[dict[str, object]]:
    tasks = {row["evaluation_id"]: row for row in _read_csv(tasks_path)}
    output = []
    for evaluation_id in sorted(set(evaluation_ids)):
        item = tasks.get(evaluation_id, {})
        scores = [row for row in raw_scores if row["evaluation_id"] == evaluation_id]
        exemplar = scores[0] if scores else {}
        output.append(
            {
                key: (
                    evaluation_id
                    if key == "evaluation_id"
                    else len(scores)
                    if key == "observed_models"
                    else item.get(key, exemplar.get(key, ""))
                )
                for key in EVALUATION_FIELDS
            }
        )
    return output


def _score_rows(
    raw_scores: list[dict[str, str]],
    *,
    model_ids: set[str] | None = None,
    evaluation_ids: set[str] | None = None,
    cell_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, object]]:
    output = []
    for row in raw_scores:
        if model_ids is not None and row["model_id"] not in model_ids:
            continue
        if evaluation_ids is not None and row["evaluation_id"] not in evaluation_ids:
            continue
        if cell_keys is not None and (row["model_id"], row["evaluation_id"]) not in cell_keys:
            continue
        output.append({field: row.get(field, "") for field in SCORE_FIELDS})
    return sorted(output, key=lambda row: (str(row["model_id"]), str(row["evaluation_id"])))


def _licenses_text(suite_rows: list[dict[str, str]]) -> str:
    suite_links = "\n".join(
        f"- {row['name']} (`{row['suite_id']}`): {row['reference_url']}"
        for row in suite_rows
    )
    return f"""# Data provenance and licenses

PathoPress source code is MIT-licensed; see `LICENSE` in the project repository.

The exported tables contain factual identifiers, protocol metadata, citations,
and reported numeric results. They do **not** relicense benchmark datasets,
pathology images, labels, model weights, or source publications. Each upstream
artifact keeps its own license, terms, access restrictions, and attribution
requirements. Consult the linked upstream project before redistribution or
commercial use.

## Upstream benchmark suites

{suite_links}

`provenance.json` records pinned source revisions, report hashes, normalization
rules, and audit caveats. `scores_all.csv` includes prototype and external rows;
`scores_paper.csv` is restricted to accepted evidence and the supported paper
matrix. Machine-parsed primary-source evidence is not dual human verification.
"""


def _export_readme_text() -> str:
    return """# PathoPress public score-matrix export

The `data/` directory contains model, evaluation, and long score tables for the
full source registry (`*_all.csv`) and the supported publication matrix
(`*_paper.csv`). `score_matrix_paper_wide.csv` is the same accepted paper cells
in model-by-evaluation form. Current row counts are:

| Table layer | Models | Evaluations | Score rows |
|---|---:|---:|---:|
| Full registry | 60 | 287 | 1,976 |
| Fixed paper matrix | 59 | 165 | 1,967 |

The paper filter accepts `verified` and `parsed_primary_source` evidence and
iteratively requires at least three scores per model and five models per
evaluation. It excludes nine external-report rows. Machine-parsed primary
evidence has not necessarily received two independent human reviews.

Every file is byte-counted and SHA-256-bound by `manifest.json`. Load and verify
a local copy with:

```python
from pathopress.public_data import load_public_export
release = load_public_export("exports/pathopress_public")
```

Download an HTTP/file mirror reproducibly with:

```bash
PYTHONPATH=src python3 scripts/download_public_release.py BASE_URL DESTINATION
```

Use `--force` to refresh an existing local manifest. The downloader performs no
upload and rejects unsafe paths, unsupported schemas, missing files, and hash
mismatches. Local source paths are removed from the public provenance payload.

This package contains reported facts and protocol metadata, not benchmark
images, labels, model weights, or a license grant for upstream data. Read
`LICENSES.md` and `provenance.json` before redistribution. Building or
downloading this export does not upload or deploy it.
"""


def build_public_export(
    *,
    scores_path: str | Path,
    tasks_path: str | Path,
    suites_path: str | Path,
    provenance_path: str | Path,
    model_metadata_path: str | Path,
    out_dir: str | Path,
    min_scores_per_model: int = 3,
    min_models_per_evaluation: int = 5,
) -> dict[str, object]:
    """Build deterministic all/paper/wide CSVs and a hash manifest."""
    scores_path = Path(scores_path)
    tasks_path = Path(tasks_path)
    suites_path = Path(suites_path)
    provenance_path = Path(provenance_path)
    model_metadata_path = Path(model_metadata_path)
    out_dir = Path(out_dir)
    data_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_scores = _read_csv(scores_path)
    tasks = _read_csv(tasks_path)
    suites = _read_csv(suites_path)
    dataset = load_prediction_dataset(
        scores_path,
        min_scores_per_model=min_scores_per_model,
        min_models_per_evaluation=min_models_per_evaluation,
    )
    all_models = _model_rows(
        (row["model_id"] for row in raw_scores), raw_scores, model_metadata_path
    )
    paper_models = [row for row in all_models if row["model_id"] in set(dataset.models)]
    all_evaluations = _evaluation_rows(
        (row["evaluation_id"] for row in tasks), raw_scores, tasks_path
    )
    paper_evaluations = [
        row for row in all_evaluations if row["evaluation_id"] in set(dataset.evaluations)
    ]
    all_scores = _score_rows(raw_scores)
    paper_scores = _score_rows(
        raw_scores,
        model_ids=set(dataset.models),
        evaluation_ids=set(dataset.evaluations),
        cell_keys=set(dataset.score_by_cell),
    )

    row_counts: dict[str, int] = {}
    row_counts["data/models_all.csv"] = _write_csv(
        data_dir / "models_all.csv", all_models, MODEL_FIELDS
    )
    row_counts["data/models_paper.csv"] = _write_csv(
        data_dir / "models_paper.csv", paper_models, MODEL_FIELDS
    )
    row_counts["data/evaluations_all.csv"] = _write_csv(
        data_dir / "evaluations_all.csv", all_evaluations, EVALUATION_FIELDS
    )
    row_counts["data/evaluations_paper.csv"] = _write_csv(
        data_dir / "evaluations_paper.csv", paper_evaluations, EVALUATION_FIELDS
    )
    row_counts["data/scores_all.csv"] = _write_csv(
        data_dir / "scores_all.csv", all_scores, SCORE_FIELDS
    )
    row_counts["data/scores_paper.csv"] = _write_csv(
        data_dir / "scores_paper.csv", paper_scores, SCORE_FIELDS
    )

    wide_fields = ["model_id", *dataset.evaluations]
    wide_rows = []
    for row_index, model in enumerate(dataset.models):
        row: dict[str, object] = {"model_id": model}
        for column, evaluation in enumerate(dataset.evaluations):
            value = dataset.matrix[row_index, column]
            row[evaluation] = f"{value:.10g}" if np.isfinite(value) else ""
        wide_rows.append(row)
    row_counts["data/score_matrix_paper_wide.csv"] = _write_csv(
        data_dir / "score_matrix_paper_wide.csv", wide_rows, wide_fields
    )

    provenance = _clean_provenance(
        json.loads(provenance_path.read_text(encoding="utf-8"))
    )
    _json_write(out_dir / "provenance.json", provenance)
    (out_dir / "LICENSES.md").write_text(_licenses_text(suites), encoding="utf-8")
    (out_dir / "README.md").write_text(_export_readme_text(), encoding="utf-8")

    tracked = [
        *sorted(row_counts),
        "provenance.json",
        "LICENSES.md",
        "README.md",
    ]
    files = [
        {
            "path": relative,
            "sha256": sha256_file(out_dir / relative),
            "bytes": (out_dir / relative).stat().st_size,
            **({"rows": row_counts[relative]} if relative in row_counts else {}),
        }
        for relative in tracked
    ]
    manifest: dict[str, object] = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "description": "Public PathoPress pathology benchmark score-matrix tables.",
        "inputs": {
            "scores_sha256": sha256_file(scores_path),
            "tasks_sha256": sha256_file(tasks_path),
            "suites_sha256": sha256_file(suites_path),
            "provenance_sha256": sha256_file(provenance_path),
            "model_metadata_sha256": (
                sha256_file(model_metadata_path) if model_metadata_path.exists() else None
            ),
        },
        "paper_filter": {
            "accepted_audit_statuses": ["verified", "parsed_primary_source"],
            "min_scores_per_model": min_scores_per_model,
            "min_models_per_evaluation": min_models_per_evaluation,
            "models": len(dataset.models),
            "evaluations": len(dataset.evaluations),
            "observations": int(np.isfinite(dataset.matrix).sum()),
        },
        "rows": {
            "models_all": len(all_models),
            "models_paper": len(paper_models),
            "evaluations_all": len(all_evaluations),
            "evaluations_paper": len(paper_evaluations),
            "scores_all": len(all_scores),
            "scores_paper": len(paper_scores),
        },
        "files": files,
    }
    _json_write(out_dir / "manifest.json", manifest)
    return manifest


@dataclass(frozen=True)
class PublicExport:
    root: Path
    manifest: dict[str, object]
    models: list[dict[str, str]]
    evaluations: list[dict[str, str]]
    scores: list[dict[str, str]]


def _safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe export path: {value!r}")
    return Path(*pure.parts)


def load_public_export(root: str | Path, *, verify: bool = True) -> PublicExport:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PUBLIC_SCHEMA_VERSION:
        raise ValueError("unsupported public export schema")
    if verify:
        for item in manifest["files"]:
            path = root / _safe_relative_path(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise ValueError(f"public export hash mismatch: {item['path']}")
    models = _read_csv(root / "data" / "models_paper.csv")
    evaluations = _read_csv(root / "data" / "evaluations_paper.csv")
    scores = _read_csv(root / "data" / "scores_paper.csv")
    expected = manifest["paper_filter"]
    if (len(models), len(evaluations), len(scores)) != (
        expected["models"],
        expected["evaluations"],
        expected["observations"],
    ):
        raise ValueError("public export row counts disagree with manifest")
    return PublicExport(root, manifest, models, evaluations, scores)


def _download_atomic(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, tempfile.NamedTemporaryFile(
        dir=target.parent, delete=False
    ) as temporary:
        shutil.copyfileobj(response, temporary)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, target)


def download_public_export(
    base_url: str, destination: str | Path, *, force: bool = False
) -> PublicExport:
    """Download a manifest and its exact files, then verify all hashes."""
    destination = Path(destination)
    manifest_path = destination / "manifest.json"
    if force or not manifest_path.exists():
        _download_atomic(urljoin(base_url.rstrip("/") + "/", "manifest.json"), manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PUBLIC_SCHEMA_VERSION:
        raise ValueError("unsupported public export schema")
    for item in manifest["files"]:
        relative = _safe_relative_path(item["path"])
        target = destination / relative
        if force or not target.exists() or sha256_file(target) != item["sha256"]:
            _download_atomic(
                urljoin(base_url.rstrip("/") + "/", item["path"]), target
            )
    return load_public_export(destination)


def build_website_data(
    *,
    scores_path: str | Path,
    tasks_path: str | Path,
    model_metadata_path: str | Path,
    output_path: str | Path,
    confidence_artifact_path: str | Path | None = None,
    new_model_confidence_artifact_path: str | Path | None = None,
    min_scores_per_model: int = 3,
    min_models_per_evaluation: int = 5,
) -> dict[str, object]:
    dataset = load_prediction_dataset(
        scores_path,
        min_scores_per_model=min_scores_per_model,
        min_models_per_evaluation=min_models_per_evaluation,
    )
    completed = complete_dataset(dataset)
    raw_scores = _read_csv(Path(scores_path))
    models = _model_rows(
        dataset.models, raw_scores, Path(model_metadata_path)
    )
    evaluations = _evaluation_rows(
        dataset.evaluations, raw_scores, Path(tasks_path)
    )
    confidence = None
    if confidence_artifact_path is not None:
        confidence = load_confidence_artifact(
            confidence_artifact_path,
            scores_path,
            rank=DEFAULT_RANK,
            regularization=DEFAULT_REGULARIZATION,
        )
    new_model_confidence = None
    if new_model_confidence_artifact_path is not None:
        loaded_new_model_confidence = load_new_model_confidence_artifact(
            new_model_confidence_artifact_path,
            scores_path,
            rank=DEFAULT_RANK,
            regularization=DEFAULT_REGULARIZATION,
        )
        # The browser needs deploy lookups and empirical summaries, but not the
        # verbose cross-fit group membership audit retained in the JSON artifact.
        new_model_confidence = {
            key: value for key, value in loaded_new_model_confidence.items()
            if key != "crossfit_group_audit"
        }
    observed: list[list[float | None]] = []
    predictions: list[list[float]] = []
    sources: list[list[dict[str, str] | None]] = []
    intervals: list[list[list[float] | None]] = []
    for row_index, model in enumerate(dataset.models):
        observed_row: list[float | None] = []
        predicted_row: list[float] = []
        source_row: list[dict[str, str] | None] = []
        interval_row: list[list[float] | None] = []
        for column, evaluation in enumerate(dataset.evaluations):
            value = dataset.matrix[row_index, column]
            predicted = float(completed[row_index, column])
            score = dataset.score_by_cell.get((model, evaluation))
            observed_row.append(float(value) if np.isfinite(value) else None)
            predicted_row.append(round(predicted, 6))
            source_row.append(
                {
                    "url": score.reference_url,
                    "audit_status": score.audit_status,
                }
                if score is not None
                else None
            )
            if confidence is not None and not np.isfinite(value):
                suite = evaluations[column]["suite_id"]
                lower, upper, _ = calibrated_interval(predicted, suite, confidence)
                interval_row.append([round(lower, 6), round(upper, 6)])
            else:
                interval_row.append(None)
        observed.append(observed_row)
        predictions.append(predicted_row)
        sources.append(source_row)
        intervals.append(interval_row)
    payload: dict[str, object] = {
        "schema_version": WEBSITE_SCHEMA_VERSION,
        "models": models,
        "evaluations": evaluations,
        "observed": observed,
        "predictions": predictions,
        "sources": sources,
        "prediction_intervals": intervals,
        "new_model_confidence": new_model_confidence,
        "meta": {
            "point_method": "logit + evaluation z-score + bias ALS rank=1 lambda=0.1",
            "models": len(dataset.models),
            "evaluations": len(dataset.evaluations),
            "observations": int(np.isfinite(dataset.matrix).sum()),
            "scores_sha256": sha256_file(scores_path),
            "confidence": (
                "90% suite-conditional held-out-cell absolute-residual intervals"
                if confidence is not None
                else None
            ),
            "new_model_confidence": (
                "group-balanced empirical 90% intervals from leave-one-model-out sparse-probe and temporal residuals; unsupported columns abstain"
                if new_model_confidence is not None
                else None
            ),
        },
    }
    _json_write(Path(output_path), payload)
    return payload
