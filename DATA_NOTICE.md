# Data notice

The MIT license in [`LICENSE`](LICENSE) applies to PathoPress code. It does not
relicense the benchmark datasets, model weights, upstream source tables, or
other third-party materials cited by this repository.

The files under `data/` contain factual identifiers, short protocol metadata,
source links, and reported numeric results extracted from the upstream sources
listed in [`data/provenance.json`](data/provenance.json). Each upstream artifact
retains its own license, terms of use, attribution requirements, and access
restrictions. PathoPress does not redistribute pathology images, benchmark
labels, or model weights.

Before publishing, mirroring, or using these artifacts commercially, review the
terms of every underlying source and obtain legal advice where appropriate.

## What is covered by which terms

PathoPress deliberately separates three layers, because they do not share a
single license:

| Layer | Examples | Terms |
|---|---|---|
| Extraction and analysis code | `src/pathopress/`, `scripts/`, `experiments/`, `website/` | MIT ([`LICENSE`](LICENSE)); adapted BenchPress components are additionally covered by [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) |
| PathoPress-authored provenance and curation metadata | [`data/provenance.json`](data/provenance.json), `data/model_sources.csv`, `data/model_metadata.csv`, `data/model_aliases.csv`, `data/deduplication.csv`, `data/suites.csv`, `data/tasks.csv`, the review ledger and schema files | MIT, with a CC-BY-style expectation that reuse cites PathoPress (see "Citation and attribution" below) |
| Reported benchmark scores | the numeric `score` cells in `data/scores.csv` and every artifact derived from them | **Per-score**: governed by the terms of the individual upstream publication or leaderboard the score was parsed from — see "Per-score terms of use" |

The distinction matters because a single file mixes layers. `data/scores.csv` is
one table, but its rows are machine-parsed from roughly twenty separately
licensed publications and leaderboards. The row structure, normalization, task
identity resolution, deduplication, and audit status are PathoPress work; the
underlying reported numbers are not.

## Per-score terms of use

Every row in `data/scores.csv` is traceable to exactly one pinned upstream
source. Resolve a row's terms as follows:

1. Read the row's `suite_id`, `reference_url`, and `source_locator` columns in
   `data/scores.csv`. Together they name the exact upstream table, figure, or
   leaderboard the value came from.
2. Look the source up in [`data/provenance.json`](data/provenance.json) — under
   `repositories` for benchmark suites (pinned by git or Hugging Face commit) or
   under `source_reports` for primary papers and official model reports (pinned
   by URL plus a SHA-256 of the archived source and its snapshot).
3. Apply the license and terms of use published by that source. They are not
   uniform: the repository-hosted suites carry open-source or dataset licenses,
   whereas the primary-paper sources are journal or preprint articles whose
   tables are governed by the publisher's copyright and reuse policy. Some
   sources additionally restrict redistribution of the underlying cohorts.
4. Where a score is marked quarantined, analysis-ineligible, or
   alternate-evidence, see [`docs/score-source-coverage.md`](docs/score-source-coverage.md)
   for why. Those markers are eligibility statements about scientific
   comparability, not license grants.

The upstream sources currently represented — with their pinned revisions — are
enumerated in [`exports/pathopress_public/LICENSES.md`](exports/pathopress_public/LICENSES.md),
which ships with the public export so that the list travels with the data.
[`data/LICENSE`](data/LICENSE) restates these terms next to the tables
themselves.

PathoPress asserts no ownership over any reported number. Extracting a published
figure into a machine-readable table does not create a new copyright in that
figure, and nothing in this repository should be read as granting rights the
upstream publishers did not grant.

## Model weights and datasets

`docs/model-sources.md` and `data/model_sources.csv` link primary papers and
official model releases for each evaluated foundation model. Those links are
provided for attribution and verification only. Model checkpoints and the
pathology datasets they are evaluated on are obtained directly from their
publishers under their own licenses, several of which are non-commercial,
gated, or require a data use agreement. PathoPress ships none of them.

By policy the ledger links only author/publisher primary papers and official
model or code releases; community mirrors are excluded. See the "Source policy"
section of [`docs/model-sources.md`](docs/model-sources.md).

## Citation and attribution

If you use the PathoPress tables, the provenance record, or the completion
results, please cite PathoPress **and** the upstream sources of any scores you
rely on. Citing PathoPress alone is not sufficient attribution for the reported
numbers: the obligation to credit the original benchmark authors passes through
to you. The per-source citation information needed to do this is in
`data/provenance.json` and `data/model_sources.csv`.

## Related documents

- [`LICENSE`](LICENSE) — MIT license for PathoPress code.
- [`data/LICENSE`](data/LICENSE) — the same terms, stated next to the data.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — adapted Microsoft
  BenchPress components and their MIT notice.
- [`data/provenance.json`](data/provenance.json) — pinned revisions, source
  digests, normalization rules, and audit caveats.
- [`docs/score-source-coverage.md`](docs/score-source-coverage.md) — which
  reported numbers enter the registry and which are excluded or quarantined.
- [`docs/model-sources.md`](docs/model-sources.md) — component-level model
  source ledger and source policy.
- [`exports/pathopress_public/LICENSES.md`](exports/pathopress_public/LICENSES.md)
  — upstream suite list with pinned revisions, shipped with the public export.
