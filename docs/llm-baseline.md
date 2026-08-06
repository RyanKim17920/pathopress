# BenchPress LLM-completer parity

PathoPress has a complete offline request pack for the four LLM conditions in
Microsoft BenchPress Section 5.3. No provider call has been made, so this is an
audited experiment specification, not an LLM result.

## Pinned upstream protocol

The implementation is pinned to BenchPress commit
`0a684b63ee0e4a401cb907a3827a82ea997d74c4`:

- zero-shot named and blind: the model sees the full fold-specific sparse score
  matrix and predicts held-out cells, with target models batched 10 at a time;
- five-shot named and blind: each target cell receives up to five peer models;
  candidates must expose the target score and share at least five visible
  training cells with the target model, then are ordered by descending Pearson
  correlation with peer-index tie-breaking;
- the pinned five-shot prompt intentionally keeps the first 12 visible target
  anchors and first four shared anchors per peer; named batches contain 64
  cells and blind batches contain 16.

Pathology adaptations are explicit: all retained scores use the audited
0–100 orientation, model `reasoning` flags are omitted, task definitions replace
general-LLM benchmark metadata, and the prompt says rank 1 because matched
pathology validation selected rank 1. It does not copy BenchPress's empirical
rank-2 statement.

## Complete request pack

The [pack index](../experiments/llm_baseline/requests.jsonl) binds 20
deterministic gzip JSONL shards by compressed and canonical-uncompressed
SHA-256. It contains all 30 shared folds:

| Condition | Requests | Target predictions |
|---|---:|---:|
| zero-shot named | 180 | 20,270 |
| zero-shot blind | 180 | 20,270 |
| five-shot named | 340 | 20,270 |
| five-shot blind | 1,290 | 20,270 |
| **Total** | **1,990** | **81,080** |

Each of the 2,027 observed matrix cells is held out once per seed and requested
under every condition. The [manifest](../experiments/llm_baseline/dry_run_manifest.json)
records exact counts, input hashes, batching, and request-pack hashes. A separate
[smoke fixture](../experiments/llm_baseline_smoke/dry_run_manifest.json) has four
requests and 32 deterministic mock predictions; mock accuracy is never
headline-eligible.

Prepare or validate/materialize the full pack without network access:

```bash
PYTHONPATH=src python3 experiments/run_llm_baseline.py prepare --scope full
PYTHONPATH=src python3 experiments/run_llm_baseline.py materialize --scope full \
  --materialized-output /tmp/pathopress-llm-requests
```

## Sole external action

Execute every provider-neutral request with one fixed genuine LLM and return
one raw JSONL record per `request_id`, containing `backend_kind`, `provider`,
`model`, `response_text`, and optional token usage. This repository neither
reads credentials nor implements a paid provider client. Cost and token usage
remain unknown until execution.

The real-response gate requires exactly one fixed
`(backend_kind, provider, model)` tuple across the complete 1,990-response
pack. A mixed provider or model pack is rejected before metrics or headline
status. Executors may also attach an `execution_metadata` object with a
`model_version` string, fixed `settings` object, and per-request `receipt`.
These values are SHA-256-bound into the sealed responses and import status;
receipt contents need not be published. Supplied model-version and settings
hashes must be consistent across the pack.

Genuine raw and sealed response JSONL files and provider-receipt directories
are ignored by default. Publishing response evidence therefore requires an
explicit content review and deliberate force-add. Never commit credentials,
secret values, or raw provider receipts; retain only the validated hashes when
receipt provenance is needed. Provider-neutral request shards, schemas, and
the deterministic smoke mock remain ordinary tracked reproducibility assets.

Import is fail-closed: duplicate, missing, or unexpected request IDs; malformed
JSON; missing or extra query IDs; nonnumeric/nonfinite values; and scores
outside 0–100 are rejected. Successful import hash-seals responses and only a
complete real-backend pack becomes headline-eligible:

```bash
PYTHONPATH=src python3 experiments/run_llm_baseline.py import-real --scope full \
  --raw-responses /path/to/raw-provider-responses.jsonl
python3 scripts/plot_llm_baseline.py
```

The [real-run status](../experiments/llm_baseline/real_run_status.json) is the
current source of truth: zero genuine responses, external cost unknown. Once
responses are imported, the chart reports median per-fold MedAE/MedAPE for all
four conditions and the hash-matched rank-1 pathology comparator.
