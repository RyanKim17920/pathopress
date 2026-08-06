# Measured-budget probe selection

PathoPress's 25-protocol `low_friction` set is a pipeline-feasibility proxy,
not a cost model. This independent runner selects probes only when burden is
measured or explicitly accepted from source-reported/configured-ceiling facts. It never converts sample count,
dataset size, or a qualitative feasibility tier into money or runtime.

The checked-in example is an explicit canonical non-measurement receipt. The current evidence
audit has no observed per-protocol runtime, annotation labor, or dollar cost,
so its correct result is `insufficient_cost_coverage`, with no recommendation
or chart.

## Evidence and accounting contract

[`evaluation_burden_measurements_v1.example.json`](../data/evaluation_burden_measurements_v1.example.json)
is a canonical non-measurement receipt that demonstrates the fail-safe path.
For a real panel, pass a directory containing canonical receipts. Facts use the
same status vocabulary as
[`evaluation_burden_measurements.schema.json`](../data/evaluation_burden_measurements.schema.json):

- `measured`: observed inside the declared scenario and run boundary;
- `source_reported`: a traceable source value;
- `configured_ceiling`: a traceable conservative configuration bound;
- `not_applicable`, `not_measured`, `not_reported`, and `inaccessible`: an
  explicit null value plus a reason.

Every numeric fact has an explicit unit. Active budget fields must use exactly
the same unit. Negative, non-finite, unitless, or invented `estimated` values
are rejected.

Canonical phases separate shared artifact setup, per-model feature extraction,
protocol-head fitting, and protocol evaluation. A
`shared_artifact_setup` receipt is charged once per `artifact_group_id`, even
when several selected protocols reference it. Additive dimensions such as
accelerator-seconds, person-hours, input bytes, and marginal USD are summed.
Peak accelerator memory is a capacity constraint and is maximized, not summed.
The canonical `constraints` facts make access class, dataset license,
commercial use, redistribution, and new-tissue requirements categorical hard
constraints instead of ordinal scores pretending to be costs.

Budget profiles are scenario-specific. For example, `reuse_existing_labels`
may accept a canonical annotation-labor fact as `not_applicable` only when its
reason establishes that existing labels are reused. A greenfield acquisition
profile must measure or source the relevant labor and tissue facts.

## Missing evidence

`--missing-policy error` fails on the first retained protocol lacking an active
fact. `--missing-policy exclude` removes incomplete or disallowed protocols and
records every reason in the output coverage ledger. If no candidate survives,
the runner writes `status: insufficient_cost_coverage` and stops before matrix
completion. It never silently falls back to the old sample-count proxy.

By default, only one protocol may be selected for each deduplicated
`task_identity_id`. Use `--no-one-per-task-identity` only when protocol variants
are intentionally separate purchases.

## Search and validation

The runner imports the existing `predict_all_known`, `predict_heldout_models`,
`score_predictions`, and `objective_value` implementations. It does not
duplicate or modify the existing top-k compression experiment.

Budget-constrained greedy search evaluates every feasible one-protocol
addition and chooses the lowest predictive loss. It is deterministic but is
not called globally optimal. Ties within `1e-12` use lower normalized budget
pressure, the canonical cost vector, then lexicographic evaluation IDs.

Exact search enumerates every feasible subset through `--max-probes`. Before
doing work it computes the unconstrained combination-count upper bound and
refuses a run above `--max-subsets`. A completed exact artifact is globally
exact only for its candidate universe, active constraints, objective, and
maximum probe count.

MedAE score reconstruction and margin-5 ranking error are separate objectives;
they are not blended with arbitrary weights. Selected sets receive isolated
held-out-model validation. Deterministic random, identity-unique,
budget-feasible prefixes provide the baseline.

## Commands

The non-measurement example demonstrates the fail-safe behavior:

```bash
PYTHONPATH=src python3 experiments/run_budgeted_probe_selection.py \
  --burden data/evaluation_burden_measurements_v1.example.json \
  --budget data/probe_budget_v1.example.json \
  --missing-policy exclude \
  --workers 4 \
  --output /tmp/pathopress-budget-check.json
```

After collecting audited receipts into a directory, run the two objectives independently:

```bash
PYTHONPATH=src python3 experiments/run_budgeted_probe_selection.py \
  --burden data/evaluation_burden_receipts/ \
  --budget data/probe_budget_v1.json \
  --objective medae --search greedy --max-probes 10 --workers 4

PYTHONPATH=src python3 experiments/run_budgeted_probe_selection.py \
  --burden data/evaluation_burden_receipts/ \
  --budget data/probe_budget_v1.json \
  --objective pairwise_margin_error --ranking-margin 5 \
  --search exact --max-probes 5 --max-subsets 200000 --workers 4
```

The worker count is hard-capped at four and every worker is limited to one
BLAS thread. Exact-search caps are checked before the process pool starts.

## Output contract

The JSON artifact binds the score, task, burden, budget, and runner hashes. It
records the matrix shape, objective, rank, model split, evidence coverage,
every excluded candidate, accepted evidence statuses, active additive and
capacity limits, categorical constraints, search exactness, deterministic tie
rule, selected trajectory or optimum, cumulative burden and newly charged
assets, held-out metrics, and random feasible baseline.

No budget chart or website starter set should be published until an artifact
has `status: complete` and adequate measured coverage. At that point a single
budget frontier should replace—not accompany—redundant feasibility-proxy
charts.

## Verification

```bash
UV_CACHE_DIR=/tmp/pathopress-uv-cache uv run --frozen --offline pytest -q \
  tests/test_budgeted_probes.py tests/test_budgeted_probe_runner.py
```

The tests cover canonical schema and unit rejection, reported/ceiling opt-in,
null not-applicable facts, missing fail-closed and exclusion modes,
shared-phase charging,
additive and peak constraints, access and licensing, task-identity conflicts,
deterministic greedy and random paths, exact optimality and preflight refusal,
the four-worker cap, and the current non-measurement-receipt result.
