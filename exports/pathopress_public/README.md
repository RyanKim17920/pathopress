---
pretty_name: PathoPress Pathology Foundation-Model Score Matrix
license: other
license_name: mixed-upstream-terms
task_categories:
- tabular-classification
configs:
- config_name: scores_paper
  data_files: data/scores_paper.parquet
- config_name: scores_all
  data_files: data/scores_all.parquet
- config_name: models
  data_files: data/models.parquet
- config_name: benchmarks
  data_files: data/benchmarks.parquet
---

# PathoPress public score-matrix export

Intended dataset repository: `pathopress/pathopress-score-matrix`. This is a local publication build;
building it does not upload or create a remote repository.

The `data/` directory contains model, evaluation, and long score tables for the
full source registry (`*_all.csv`) and the supported publication matrix
(`*_paper.csv`). `score_matrix_paper_wide.csv` is the same accepted paper cells
in model-by-evaluation form. CSV and deterministic Parquet mirrors are included. Current row counts are:

| Table layer | Models | Evaluations | Score rows |
|---|---:|---:|---:|
| Full registry | 60 | 292 | 2076 |
| Fixed paper matrix | 59 | 168 | 2027 |

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
