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

## Offline execution preflight

The checked [execution preflight](../experiments/llm_baseline/execution_preflight.json)
validates the complete pack and estimates its execution envelope without a
network call. Its [request ledger](../experiments/llm_baseline/execution_preflight_requests.jsonl.gz)
contains one hash-bound row per prepared request, and the
[approval manifest](../experiments/llm_baseline/execution_approval_manifest.json)
is self-hashed but currently `not_ready` and `execution_authorized=false`.

The literal upstream invocation contract is now explicit rather than inferred:
OpenAI Chat Completions, mutable alias `gpt-5.5`, `temperature=0.0`, and
`max_tokens=16384`. The runner rejects any profile labelled
`upstream_contract` if one of those fields changes. BenchPress recorded no
resolved immutable model snapshot. Accordingly, the
[literal profile](../experiments/llm_baseline/profiles/upstream_contract.model.json)
is labelled a mutable, unfrozen alias contract. The separate
[`gpt-5.5-2026-04-23` profile](../experiments/llm_baseline/profiles/reproducible_snapshot.model.json)
is a controlled reproducibility adaptation and is explicitly **not** an exact
historical upstream invocation.

The checked default is the controlled snapshot plus the 24-hour OpenAI Batch
transport adaptation. First materialize the actual provider JSONL plan, then
bind that exact 20-file hash/byte plan into the capacity, price, and approval
preflight:

```bash
PYTHONPATH=src python3 experiments/run_openai_batch.py preflight \
  --profile chat_snapshot --model gpt-5.5-2026-04-23
PYTHONPATH=src python3 experiments/run_llm_baseline.py preflight --scope full \
  --adapter-manifest experiments/llm_baseline/openai_batch/preflight.json \
  --capacity-profile /path/to/reviewed-capacity.json \
  --pricing-profile /path/to/reviewed-pricing.json \
  --budget-currency USD --max-budget REVIEWED_PLANNING_CEILING \
  --acknowledge-estimated-cost-uncertainty
```

The capacity profile supplies model/account context and queued-token limits;
it is hash-bound separately from the immutable Batch protocol profile. The
profile must identify a nonsecret account/project scope and attest that the
credential used for submission belongs to that same scope:

```json
{
  "schema_version": 1,
  "provider": "OpenAI",
  "model_alias": "gpt-5.5",
  "model_snapshot": "gpt-5.5-2026-04-23",
  "account_scope_label": "reviewed-project-label-no-secrets",
  "evidence": {
    "source": "reviewed account limits page or administrator attestation",
    "retrieved_at": "YYYY-MM-DD",
    "active_credential_scope_attested": true
  },
  "limits": {
    "context_window_tokens": 0,
    "max_output_tokens": 0,
    "max_input_tokens_per_request": 0,
    "max_queued_input_tokens": 0
  }
}
```

Replace every zero with a reviewed positive limit. Never include an API key,
organization secret, or project secret. The
budget is a human planning ceiling checked against the conservative estimated
scenario. It is not a provider-enforced billing cap: tokenization, retries,
provider accounting, or pricing changes can make actual billed cost differ or
exceed it. The uncertainty acknowledgement is therefore required both in the
approval and again at paid submission. Literal alias analysis remains an
explicit path that also selects the literal online transport rather than the
default Batch adaptation:

```bash
PYTHONPATH=src python3 experiments/run_llm_baseline.py preflight --scope full \
  --model-profile experiments/llm_baseline/profiles/upstream_contract.model.json \
  --transport-profile experiments/llm_baseline/profiles/upstream_online.transport.json \
  --acknowledge-mutable-alias
```

That example still requires separately reviewed capacity/pricing/budget inputs
before approval; it only makes the upstream model and transport choice
unambiguous.

When every technical check is ready, perform the separate offline human state
transition. This validates the current preflight and manifest bindings,
records the two explicit acknowledgements, and atomically rewrites only the
self-hashed approval manifest; it makes no provider call and reads no key:

```bash
PYTHONPATH=src python3 experiments/run_llm_baseline.py approve --scope full \
  --human-review-complete --acknowledge-estimated-cost-uncertainty
```

The model-agnostic estimator serializes each messages array deterministically
and reports best/base/worst scenarios. Across this pack it estimates
19,343,573 / 25,799,752 / 77,521,688 input tokens. Compact-output estimates are
166,350 / 306,800 tokens; the worst case is the configured ceiling of
32,604,160 output tokens (`1,990 × 16,384`). These are planning scenarios, not
provider-billed token counts. The largest request is estimated at 35,987 /
47,987 / 144,019 input tokens. An explicitly selected `tiktoken` encoding can
add a content-only cross-check; it never guesses an encoding from a model
alias and does not claim to reconstruct a proprietary chat template:

```bash
PYTHONPATH=src python3 experiments/run_llm_baseline.py preflight --scope full \
  --tiktoken-encoding cl100k_base
```

Current source transport facts are exact: 1,990 requests, 20 shards, 8,586,412
compressed bytes, and 113,055,766 canonical-uncompressed bytes. The adapter
separately hashes the actual provider Batch JSONL bytes after adding the
`custom_id`/method/URL/model/settings wrapper; preflight rejects any mismatch
between its independently reconstructed plan and those 20 materialized files.
The literal upstream call used online Chat Completions, so asynchronous Batch
transport is explicitly labelled an adaptation. Its inner Chat request is
validated by exact equality of the model, messages, temperature, and
`max_tokens` fields; no raw-byte identity with the upstream SDK's HTTP
serialization is claimed.

Cost remains unavailable by construction. The tool accepts prices only from
an explicit JSON profile whose provider, alias, and snapshot exactly match the
selected model profile. A pricing profile has this shape (illustrative values
must be replaced with reviewed values):

```json
{
  "schema_version": 1,
  "profile_type": "user_supplied",
  "provider": "OpenAI",
  "model_alias": "gpt-5.5",
  "model_snapshot": "gpt-5.5-2026-04-23",
  "currency": "USD",
  "rates": {
    "batch": {
      "input_per_million_tokens": 0,
      "output_per_million_tokens": 0
    }
  }
}
```

Pass it with `--pricing-profile`. A checked official profile additionally must
carry source URL, title, retrieval date, and effective date. No latest,
cheaper, or similarly named model price is substituted. Every estimate is
bound to the compressed and canonical pack hashes, configuration, model
identity, endpoint, settings, tokenizer choice, pricing profile, and detailed
request ledger. The approval manifest expects all 1,990 returned records to
carry consistent settings, model-version evidence, and provider receipts.

## Gated OpenAI Batch execution

Execute every provider-neutral request with one fixed genuine LLM and return
one raw JSONL record per `request_id`, containing `backend_kind`, `provider`,
`model`, `response_text`, complete `execution_metadata`, and token usage. The
provider-neutral runner still reads no credentials. The separate
[`run_openai_batch.py`](../experiments/run_openai_batch.py) adapter reads
`OPENAI_API_KEY` only after explicit gates and never persists or prints it.
No OpenAI call has been made for this release.

Local preparation is the default. It validates the fixed config and pack
hashes, reconstructs all 1,990 prompts, and emits 20 traceable OpenAI Batch
JSONL files without reading a key or making a network call:

```bash
PYTHONPATH=src python3 experiments/run_openai_batch.py preflight \
  --profile chat_snapshot --model gpt-5.5-2026-04-23
```

The controlled snapshot profile preserves the upstream Chat Completions body
(`temperature=0.0`, `max_tokens=16384`, and unchanged messages) but replaces
the mutable alias with the explicitly named snapshot. It is therefore a
reproducibility adaptation, not an exact historical invocation. The
`upstream_exact` profile materializes the literal `gpt-5.5` alias/body. Paying
for that mutable-alias experiment additionally requires
`--acknowledge-mutable-alias`; a custom dated GPT-5.5 snapshot uses the
separate `chat_custom_snapshot` profile and
`--acknowledge-custom-snapshot`.

Submission fails before any upload unless the independent capacity/cost
preflight passes, the self-hashed approval manifest has no blockers and is
human-authorized, its model/settings/pack identity matches the selected run,
and every external-write/uncertainty gate is present:

```bash
# OPENAI_API_KEY must already be set securely in the process environment.
PYTHONPATH=src python3 experiments/run_openai_batch.py submit \
  --profile chat_snapshot --model gpt-5.5-2026-04-23 \
  --submit --authorize-paid-run --acknowledge-estimated-cost-uncertainty
```

The checked approval is currently `not_ready`, so that command intentionally
fails closed today. When an authorized run exists, read-only polling and
retrieval require `--online`; cancellation requires both paid-write flags and
`--confirm-cancel`. Each shard is state-saved after upload. Immediately before
batch creation it is marked `submission_pending`; if a create response is
ambiguous, automatic resubmission stops until an operator reconciles the job
in the OpenAI dashboard and records its ID with `record-batch --shard-index N
--batch-id batch-...`. This prevents a blind duplicate paid run.

```bash
PYTHONPATH=src python3 experiments/run_openai_batch.py status \
  --profile chat_snapshot --model gpt-5.5-2026-04-23 --online
PYTHONPATH=src python3 experiments/run_openai_batch.py fetch \
  --profile chat_snapshot --model gpt-5.5-2026-04-23 --online
PYTHONPATH=src python3 experiments/run_openai_batch.py cancel \
  --profile chat_snapshot --model gpt-5.5-2026-04-23 \
  --submit --authorize-paid-run --confirm-cancel \
  --acknowledge-estimated-cost-uncertainty
```

The adapter implements the documented [Batch](https://platform.openai.com/docs/api-reference/batch),
[Files](https://platform.openai.com/docs/api-reference/files), and
[Chat Completions](https://platform.openai.com/docs/api-reference/chat/create)
contracts: per-line `custom_id`/`POST`/`/v1/chat/completions`, upload purpose
`batch`, and completion window `24h`. Output/error files and converted genuine
responses remain under the ignored execution directory. Runtime state retains
only nonsecret file/batch/request IDs, statuses, usage, and hash-bound
settings/approval receipts.

Authenticated GET and file-download operations retry transient 408/409/429
and 5xx responses with bounded backoff. Upload, create, and cancel writes are
never blindly retried. The provider error file is downloaded, hashed, retained
beside the output file, and reflected as an incomplete execution; failed or
expired requests require a separately reviewed retry rather than being folded
silently into the original result. The provider-neutral importer requires all
1,990 successful rows, so partial output can never become headline evidence.

The real-response gate requires exactly one fixed
`(backend_kind, provider, model)` tuple across the complete 1,990-response
pack. A mixed provider or model pack is rejected before metrics or headline
status. Every row must attach an `execution_metadata` object with a
`model_version` string, the full fixed `settings` object, and per-request `receipt`.
These values are SHA-256-bound into the sealed responses and import status;
receipt contents need not be published. Supplied model-version and settings
hashes must be consistent across the pack. Consistency alone is insufficient:
the settings hash must equal the approved preflight settings hash, and a
selected snapshot must exactly equal the reported model version.

Genuine raw and sealed response JSONL files and provider-receipt directories
are ignored by default. Publishing response evidence therefore requires an
explicit content review and deliberate force-add. Never commit credentials,
secret values, or raw provider receipts; retain only the validated hashes when
receipt provenance is needed. Provider-neutral request shards, schemas, and
the deterministic smoke mock remain ordinary tracked reproducibility assets.

Import is fail-closed: a malformed, unapproved, or unauthorized manifest; a preflight, config, or
request-pack hash mismatch; duplicate, missing, or unexpected request IDs; malformed
JSON; missing or extra query IDs; nonnumeric/nonfinite values; and scores
outside 0–100 are rejected. Partial or absent settings/model-version/receipt
evidence and wrong-but-consistent settings are also rejected. Raw responses
are initially sealed as non-headline evidence; only whole-pack validation
against a human-approved manifest promotes them. A complete evidence-matched
real-backend pack then becomes headline-eligible:

```bash
PYTHONPATH=src python3 experiments/run_llm_baseline.py import-real --scope full \
  --raw-responses /path/to/raw-provider-responses.jsonl \
  --approval-manifest /path/to/human-approved-execution-manifest.json
python3 scripts/plot_llm_baseline.py
```

The [real-run status](../experiments/llm_baseline/real_run_status.json) is the
current source of truth: zero genuine responses, external cost unknown. Once
responses are imported, the chart reports median per-fold MedAE/MedAPE for all
four conditions and the hash-matched rank-1 pathology comparator.
