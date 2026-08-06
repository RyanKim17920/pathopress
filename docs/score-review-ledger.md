# Automated score provenance review

`data/score_review_ledger.csv` is a one-row-per-score audit trail for all 4,013
registry rows. It is an automated review, not a human review: every row fixes
`reviewer_type=automated_agent_review` and records the deterministic review time,
pinned evidence file and SHA-256, source locator, checks, canonical-setting
decision, preserved eligibility status, and review outcome.

## What was checked

The builder reparses the exact pinned upstream score files using the same
source-specific extractors that build the registry, then requires an exact match
for every field other than workflow-only `review_status`. Each cell therefore
passes all of the following before a positive outcome is recorded:

- source value and source locator;
- metric identity, direction, and normalization scale;
- reported model alias and canonical model mapping;
- protocol/canonical-setting contract; and
- split or source revision pinned in `data/provenance.json`.

Rows whose evaluation occurs in `data/deduplication.csv` additionally check the
deduplication membership and link-only/keep-separate contract. They receive
`source_locator_crosschecked`; other primary-source rows receive
`source_locator_validated`. These outcomes live in the ledger and do not rewrite
the registry's original extraction metadata. In the summary field
`locator_reachable_rows`, “reachable” means that the locator resolves inside the
exact pinned local evidence file used by the automated review. It does not mean
that a live web URL was contacted or remains online.

## Eligibility is deliberately unchanged

The ledger is evidence-bounded and cannot promote a row into analysis:

- 3,952 bounded, normalized primary-source cells remain retained;
- 46 APD cells remain
  `parsed_primary_source_analysis_ineligible`, with no invented normalization;
- 6 public cells whose source does not specify an endpoint metric remain
  analysis-ineligible, with no inferred metric or normalization;
- 9 values transcribed by PathoROB from external publications remain
  `reported_external` and `promotion_eligible=false`.

Thus a reachable PathoROB table locator validates the transcription but does not
misrepresent the external result as author-validated primary evidence.

## Rebuild and validate

With the pinned repository checkouts at `/tmp/pathopress_sources`:

```bash
python3 scripts/build_score_review_ledger.py
```

The five small score-bearing repository files are also frozen under
`source_data/pinned/`; together with the three existing paper snapshots, this
makes validation independent of network or `/tmp`:

```bash
python3 scripts/validate_score_review_ledger.py
pytest -q tests/test_score_review.py
```

The validator enforces exact 4,013-row coverage, immutable evidence hashes,
automated-only reviewer identity, duplicate-group membership, promotion rules,
and preservation of the 3,952/52/9 eligibility boundary. Aggregate counts and
the ledger hash are in `data/score_review_summary.json`; the row contract is in
`data/score_review_ledger.schema.json`.
