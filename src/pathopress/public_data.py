"""Deterministic public exports, loader, downloader, and website data builder."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
from importlib import metadata as importlib_metadata
import json
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping
from urllib.parse import urljoin
from urllib.request import urlopen

import numpy as np

from pathopress.probe_compression import load_probe_compression

from .prediction import (
    DEFAULT_RANK,
    DEFAULT_REGULARIZATION,
    calibrated_interval,
    calibrated_trust_probability,
    complete_dataset,
    load_confidence_artifact,
    load_prediction_dataset,
    sha256_file,
)
from .new_model_confidence import load_new_model_confidence_artifact


PUBLIC_SCHEMA_VERSION = "pathopress-public-tables-v1"
WEBSITE_SCHEMA_VERSION = "pathopress-static-predictor-v1"
HF_DATASET_SCHEMA_VERSION = "pathopress-hf-table-schema-v1"
DEFAULT_HF_DATASET_ID = "pathopress/pathopress-score-matrix"
PINNED_BENCHPRESS_COMMIT = "0a684b63ee0e4a401cb907a3827a82ea997d74c4"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UV_LOCK_PATH = PROJECT_ROOT / "uv.lock"


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


def parquet_available() -> bool:
    """Return whether the declared deterministic Parquet backend is available."""

    return importlib.util.find_spec("pyarrow") is not None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _exporter_identity(*, write_parquet: bool) -> dict[str, object]:
    """Return the lock/code/backend identity needed for byte-level rebuilds."""

    if not UV_LOCK_PATH.is_file():
        raise FileNotFoundError(
            f"Reproducible public export requires the repository lockfile: {UV_LOCK_PATH}"
        )
    pyarrow_version = _distribution_version("pyarrow") if write_parquet else None
    if write_parquet and pyarrow_version is None:
        raise RuntimeError("Parquet export selected but the PyArrow version is unavailable")
    return {
        "implementation": "pathopress.public_data.build_public_export",
        "implementation_sha256": sha256_file(Path(__file__)),
        "uv_lock_sha256": sha256_file(UV_LOCK_PATH),
        "parquet_backend": "pyarrow" if write_parquet else None,
        "pyarrow_version": pyarrow_version,
        "pandas_used": False,
        "pandas_version": _distribution_version("pandas"),
    }


def _parquet_scalar(value: object, logical_type: str) -> object:
    if value in (None, ""):
        return None
    if logical_type == "float64":
        return float(value)
    if logical_type == "int64":
        return int(value)
    if logical_type == "boolean":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() == "true"
    return str(value)


def _write_parquet(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fields: list[str],
    logical_types: Mapping[str, str],
) -> int:
    """Write byte-stable PyArrow Parquet with an explicit field order/schema."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - guarded by parquet_available
        raise RuntimeError(
            "Parquet export requires pyarrow; install `pathopress[hf]`."
        ) from exc
    arrow_types = {
        "string": pa.string(), "float64": pa.float64(),
        "int64": pa.int64(), "boolean": pa.bool_(),
    }
    schema = pa.schema([
        pa.field(field, arrow_types[logical_types.get(field, "string")])
        for field in fields
    ])
    materialized = [
        {
            field: _parquet_scalar(row.get(field), logical_types.get(field, "string"))
            for field in fields
        }
        for row in rows
    ]
    table = pa.Table.from_pylist(materialized, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table, path, compression="zstd", version="2.6", use_dictionary=False,
        write_statistics=True, data_page_version="1.0",
    )
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


def _export_readme_text(
    *, dataset_id: str, all_models: int, all_evaluations: int, all_scores: int,
    paper_models: int, paper_evaluations: int, paper_scores: int,
    parquet_written: bool,
) -> str:
    parquet_note = (
        "CSV and deterministic Parquet mirrors are included."
        if parquet_written else
        "CSV tables are included; install `pathopress[hf]` and rebuild with `--parquet yes` for Parquet."
    )
    table_extension = "parquet" if parquet_written else "csv"
    return f"""---
pretty_name: PathoPress Pathology Foundation-Model Score Matrix
license: other
license_name: mixed-upstream-terms
task_categories:
- tabular-classification
configs:
- config_name: scores_paper
  data_files: data/scores_paper.{table_extension}
- config_name: scores_all
  data_files: data/scores_all.{table_extension}
- config_name: models
  data_files: data/models.{table_extension}
- config_name: benchmarks
  data_files: data/benchmarks.{table_extension}
---

# PathoPress public score-matrix export

Intended dataset repository: `{dataset_id}`. This is a local publication build;
building it does not upload or create a remote repository.

The `data/` directory contains model, evaluation, and long score tables for the
full source registry (`*_all.csv`) and the supported publication matrix
(`*_paper.csv`). `score_matrix_paper_wide.csv` is the same accepted paper cells
in model-by-evaluation form. {parquet_note} Current row counts are:

| Table layer | Models | Evaluations | Score rows |
|---|---:|---:|---:|
| Full registry | {all_models} | {all_evaluations} | {all_scores} |
| Fixed paper matrix | {paper_models} | {paper_evaluations} | {paper_scores} |

The paper filter accepts `verified` and `parsed_primary_source` evidence and
iteratively requires at least three scores per model and five models per
evaluation. Machine-parsed primary evidence has not necessarily received two
independent human reviews.

Every distributed file is byte-counted and SHA-256-bound by `manifest.json`.
`schema.json` defines ordered columns and logical types; `metadata.json` records
the pinned BenchPress maintenance mapping and matrix counts. Load and verify a
local copy with:

```python
from pathopress.public_data import load_public_export
release = load_public_export("exports/pathopress_public")
```

Download an HTTP/file mirror reproducibly with:

```bash
PYTHONPATH=src python3 scripts/download_public_release.py BASE_URL DESTINATION
```

The downloader performs no upload and rejects unsafe paths, unsupported schemas,
missing files, and hash mismatches. Local source paths are removed from the
public provenance payload.

This package contains reported facts and protocol metadata, not benchmark
images, labels, model weights, or a license grant for upstream data. Read
`LICENSES.md` and `provenance.json` before redistribution. Building or
downloading this export does not upload or deploy it.
"""


def _logical_types(fields: Iterable[str], numeric: Mapping[str, str]) -> dict[str, str]:
    return {field: numeric.get(field, "string") for field in fields}


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
    parquet_mode: str = "auto",
    dataset_id: str = DEFAULT_HF_DATASET_ID,
) -> dict[str, object]:
    """Build deterministic CSV/Parquet publication tables and hash manifest."""
    scores_path = Path(scores_path)
    tasks_path = Path(tasks_path)
    suites_path = Path(suites_path)
    provenance_path = Path(provenance_path)
    model_metadata_path = Path(model_metadata_path)
    out_dir = Path(out_dir)
    data_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    if parquet_mode not in {"auto", "yes", "no"}:
        raise ValueError("parquet_mode must be one of: auto, yes, no")
    write_parquet = parquet_mode == "yes" or (
        parquet_mode == "auto" and parquet_available()
    )
    if parquet_mode == "yes" and not parquet_available():
        raise RuntimeError(
            "Parquet export requested but pyarrow is unavailable; install `pathopress[hf]`."
        )
    exporter = _exporter_identity(write_parquet=write_parquet)
    existing_parquet = sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []
    if write_parquet:
        # Every one of these is rewritten below, so removing them first only
        # clears Parquet produced by an older build; nothing is lost.
        for stale_parquet in existing_parquet:
            stale_parquet.unlink()
    elif existing_parquet:
        # A CSV-only build must never silently delete Parquet it cannot
        # regenerate: those files are committed artifacts of this repository.
        if parquet_mode == "auto":
            raise RuntimeError(
                f"{len(existing_parquet)} Parquet file(s) already exist in {data_dir} "
                "but pyarrow is unavailable, so this build cannot regenerate them. "
                "Refusing to delete them. Install `pathopress[hf]` to rebuild the "
                "Parquet tables, or pass parquet_mode='no' to build a CSV-only "
                "export and leave the existing Parquet files in place."
            )
        warnings.warn(
            f"CSV-only export requested; {len(existing_parquet)} pre-existing Parquet "
            f"file(s) in {data_dir} were left untouched and are NOT covered by this "
            "manifest. Treat them as stale and remove them before publishing.",
            RuntimeWarning,
            stacklevel=2,
        )

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
    # Pinned BenchPress maintenance compatibility aliases. In pathology,
    # `benchmarks` are protocol-level evaluation identities.
    row_counts["data/models.csv"] = _write_csv(
        data_dir / "models.csv", all_models, MODEL_FIELDS
    )
    row_counts["data/benchmarks.csv"] = _write_csv(
        data_dir / "benchmarks.csv", all_evaluations, EVALUATION_FIELDS
    )
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

    table_definitions: dict[str, tuple[list[dict[str, object]], list[str], dict[str, str]]] = {
        "models": (all_models, MODEL_FIELDS, _logical_types(MODEL_FIELDS, {"observed_scores": "int64"})),
        "benchmarks": (all_evaluations, EVALUATION_FIELDS, _logical_types(EVALUATION_FIELDS, {"observed_models": "int64"})),
        "models_all": (all_models, MODEL_FIELDS, _logical_types(MODEL_FIELDS, {"observed_scores": "int64"})),
        "models_paper": (paper_models, MODEL_FIELDS, _logical_types(MODEL_FIELDS, {"observed_scores": "int64"})),
        "evaluations_all": (all_evaluations, EVALUATION_FIELDS, _logical_types(EVALUATION_FIELDS, {"observed_models": "int64"})),
        "evaluations_paper": (paper_evaluations, EVALUATION_FIELDS, _logical_types(EVALUATION_FIELDS, {"observed_models": "int64"})),
        "scores_all": (all_scores, SCORE_FIELDS, _logical_types(SCORE_FIELDS, {"value": "float64", "normalized_score": "float64"})),
        "scores_paper": (paper_scores, SCORE_FIELDS, _logical_types(SCORE_FIELDS, {"value": "float64", "normalized_score": "float64"})),
        "score_matrix_paper_wide": (wide_rows, wide_fields, _logical_types(wide_fields, {field: "float64" for field in wide_fields if field != "model_id"})),
    }
    if write_parquet:
        for name, (rows, fields, logical_types) in table_definitions.items():
            relative = f"data/{name}.parquet"
            row_counts[relative] = _write_parquet(
                out_dir / relative, rows, fields, logical_types
            )

    provenance = _clean_provenance(
        json.loads(provenance_path.read_text(encoding="utf-8"))
    )
    _json_write(out_dir / "provenance.json", provenance)
    (out_dir / "LICENSES.md").write_text(_licenses_text(suites), encoding="utf-8")
    (out_dir / "README.md").write_text(
        _export_readme_text(
            dataset_id=dataset_id,
            all_models=len(all_models), all_evaluations=len(all_evaluations),
            all_scores=len(all_scores), paper_models=len(paper_models),
            paper_evaluations=len(paper_evaluations), paper_scores=len(paper_scores),
            parquet_written=write_parquet,
        ),
        encoding="utf-8",
    )

    schema_tables = {}
    for name, (_rows, fields, logical_types) in table_definitions.items():
        schema_tables[name] = {
            "csv": f"data/{name}.csv",
            "parquet": f"data/{name}.parquet" if write_parquet else None,
            "fields": [
                {"name": field, "logical_type": logical_types[field], "nullable": True}
                for field in fields
            ],
        }
    schema_tables["models"]["upstream_name"] = "models"
    schema_tables["benchmarks"]["upstream_name"] = "benchmarks"
    _json_write(out_dir / "schema.json", {
        "schema_version": HF_DATASET_SCHEMA_VERSION,
        "pathology_adaptation": "BenchPress benchmark_id maps to PathoPress evaluation_id; normalized_score preserves direction on a 0-100 fit scale.",
        "exporter": exporter,
        "tables": schema_tables,
    })
    metadata = {
        "schema_version": "public-table-export-v1",
        "pathopress_manifest_schema": PUBLIC_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "parquet_written": write_parquet,
        "exporter": exporter,
        "upstream": {
            "repository": "https://github.com/microsoft/benchpress",
            "commit": PINNED_BENCHPRESS_COMMIT,
            "maintenance_export": "maintenance/export_hf_dataset.py",
        },
        "rows": {
            "models": len(all_models), "benchmarks": len(all_evaluations),
            "scores_all": len(all_scores), "scores_paper": len(paper_scores),
        },
        "paper_matrix": {
            "models": len(dataset.models), "benchmarks": len(dataset.evaluations),
            "observations": int(np.isfinite(dataset.matrix).sum()),
            "fill_rate": float(np.isfinite(dataset.matrix).mean()),
            "m_threshold": min_scores_per_model,
            "b_threshold": min_models_per_evaluation,
        },
        "files": sorted(row_counts),
        "notes": [
            "scores_all is the public pre-filter score table.",
            "scores_paper is the paper-filtered pathology long table.",
            "models.csv and benchmarks.csv reproduce pinned upstream maintenance names.",
            "Rich private audit traces are not implied by this public export.",
        ],
    }
    _json_write(out_dir / "metadata.json", metadata)

    tracked = [
        *sorted(row_counts),
        "provenance.json",
        "schema.json",
        "metadata.json",
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
        "dataset_id": dataset_id,
        "parquet_written": write_parquet,
        "hf_table_schema_version": HF_DATASET_SCHEMA_VERSION,
        "pinned_benchpress_commit": PINNED_BENCHPRESS_COMMIT,
        "exporter": exporter,
        "inputs": {
            "scores_sha256": sha256_file(scores_path),
            "tasks_sha256": sha256_file(tasks_path),
            "suites_sha256": sha256_file(suites_path),
            "provenance_sha256": sha256_file(provenance_path),
            "model_metadata_sha256": (
                sha256_file(model_metadata_path) if model_metadata_path.exists() else None
            ),
            "uv_lock_sha256": exporter["uv_lock_sha256"],
            "exporter_implementation_sha256": exporter["implementation_sha256"],
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
    file_paths = [item["path"] for item in manifest.get("files", [])]
    if len(file_paths) != len(set(file_paths)):
        raise ValueError("public export manifest contains duplicate paths")
    if verify:
        for item in manifest["files"]:
            path = root / _safe_relative_path(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise ValueError(f"public export hash mismatch: {item['path']}")
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if schema.get("schema_version") != HF_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported Hugging Face table schema")
    if metadata.get("schema_version") != "public-table-export-v1":
        raise ValueError("unsupported upstream-compatible metadata schema")
    if metadata.get("dataset_id") != manifest.get("dataset_id"):
        raise ValueError("dataset ID disagrees between metadata and manifest")
    if bool(metadata.get("parquet_written")) != bool(manifest.get("parquet_written")):
        raise ValueError("Parquet status disagrees between metadata and manifest")
    exporter = manifest.get("exporter")
    if not isinstance(exporter, dict) or not exporter:
        raise ValueError("public export is missing exporter provenance")
    if metadata.get("exporter") != exporter or schema.get("exporter") != exporter:
        raise ValueError("exporter provenance disagrees across public metadata")
    if manifest.get("inputs", {}).get("uv_lock_sha256") != exporter.get("uv_lock_sha256"):
        raise ValueError("lockfile provenance disagrees with exporter identity")
    if (
        manifest.get("inputs", {}).get("exporter_implementation_sha256")
        != exporter.get("implementation_sha256")
    ):
        raise ValueError("exporter code provenance disagrees with manifest inputs")
    for required in ("models", "benchmarks", "scores_all", "scores_paper"):
        table = schema.get("tables", {}).get(required, {})
        if not table.get("csv") or not isinstance(table.get("fields"), list):
            raise ValueError(f"missing public table schema: {required}")
        if manifest.get("parquet_written") and not table.get("parquet"):
            raise ValueError(f"missing Parquet schema path: {required}")
    if not (root / "README.md").read_text(encoding="utf-8").startswith("---\n"):
        raise ValueError("Hugging Face dataset card metadata is missing")
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
    if metadata["rows"]["scores_paper"] != len(scores):
        raise ValueError("upstream-compatible metadata row count disagrees with export")
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
    probe_compression_path: str | Path | None = None,
    confidence_artifact_path: str | Path | None = None,
    new_model_confidence_artifact_path: str | Path | None = None,
    feasibility_allowlist_paths: Mapping[str, str | Path] | None = None,
    min_scores_per_model: int = 3,
    min_models_per_evaluation: int = 5,
) -> dict[str, object]:
    scores_path = Path(scores_path)
    scores_sha256 = sha256_file(scores_path)
    probe_compression_sha256 = None
    if probe_compression_path is not None:
        probe_compression_path = Path(probe_compression_path)
        probe_compression = load_probe_compression(probe_compression_path)
        if probe_compression.get("configuration", {}).get("scores_sha256") != scores_sha256:
            raise ValueError("probe-compression artifact does not match website score matrix")
        probe_compression_sha256 = sha256_file(probe_compression_path)
    feasibility_allowlists = {}
    for name, value in sorted((feasibility_allowlist_paths or {}).items()):
        path = Path(value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        evaluation_ids = payload.get("evaluation_ids", [])
        if not isinstance(evaluation_ids, list):
            raise ValueError(f"feasibility allowlist {name!r} lacks evaluation_ids")
        feasibility_allowlists[name] = {
            "sha256": sha256_file(path),
            "evaluation_count": len(evaluation_ids),
        }
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
    trust_probabilities: list[list[float | None]] = []
    trust_statuses: list[list[str | None]] = []
    for row_index, model in enumerate(dataset.models):
        observed_row: list[float | None] = []
        predicted_row: list[float] = []
        source_row: list[dict[str, str] | None] = []
        interval_row: list[list[float] | None] = []
        trust_row: list[float | None] = []
        trust_status_row: list[str | None] = []
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
                lower, upper, _ = calibrated_interval(
                    predicted, suite, confidence,
                    model_id=model, evaluation_id=evaluation,
                )
                interval_row.append([round(lower, 6), round(upper, 6)])
                trust = calibrated_trust_probability(model, evaluation, confidence)
                probability = trust.get("trust_probability")
                trust_row.append(
                    round(float(probability), 6)
                    if isinstance(probability, (int, float)) else None
                )
                trust_status_row.append(str(trust["trust_probability_status"]))
            else:
                interval_row.append(None)
                trust_row.append(None)
                trust_status_row.append(
                    "not_applicable_observed" if np.isfinite(value) else None
                )
        observed.append(observed_row)
        predictions.append(predicted_row)
        sources.append(source_row)
        intervals.append(interval_row)
        trust_probabilities.append(trust_row)
        trust_statuses.append(trust_status_row)
    payload: dict[str, object] = {
        "schema_version": WEBSITE_SCHEMA_VERSION,
        "models": models,
        "evaluations": evaluations,
        "observed": observed,
        "predictions": predictions,
        "sources": sources,
        "prediction_intervals": intervals,
        "trust_probabilities": trust_probabilities,
        "trust_probability_status": trust_statuses,
        "new_model_confidence": new_model_confidence,
        "meta": {
            "point_method": "logit + evaluation z-score + bias ALS rank=1 lambda=0.1",
            "models": len(dataset.models),
            "evaluations": len(dataset.evaluations),
            "observations": int(np.isfinite(dataset.matrix).sum()),
            "scores_sha256": scores_sha256,
            "probe_compression_sha256": probe_compression_sha256,
            "feasibility_allowlists": feasibility_allowlists,
            "confidence": (
                "90% hybrid-risk conformal intervals plus calibrated P(abs error <= 10 normalized points); unsupported cells abstain"
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
