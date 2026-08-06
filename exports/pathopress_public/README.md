# PathoPress public score-matrix export

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
