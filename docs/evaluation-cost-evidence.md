# Evaluation cost evidence

PathoPress now has a source-backed cost-evidence registry for all 187 retained
evaluation protocols. Its main conclusion is negative but important: a true
numeric cost curve is **not currently supportable**. None of the retained
protocol sources reports both observed evaluation runtime and dollar cost; in
fact, observed runtime, hardware make/model, annotation hours, and dollar cost
are each available for **0 of 187** protocols.

The machine-readable records are in
[`data/evaluation_cost_evidence.json`](../data/evaluation_cost_evidence.json),
with a flat audit view in
[`data/evaluation_cost_evidence.csv`](../data/evaluation_cost_evidence.csv).

## Evidence policy

The registry follows the rules in Microsoft BenchPress's pinned
[`benchmark_cost_evidence.README.md`](https://github.com/microsoft/benchpress/blob/0a684b63ee0e4a401cb907a3827a82ea997d74c4/benchpress/data/benchmark_cost_evidence.README.md):

1. Preserve source numbers and their original units.
2. Treat step limits, epoch limits, batch sizes, device counts, and download
   sizes as configuration or budget evidence—not observed use.
3. Do not convert descriptions such as “fast,” “large,” or “requires GPU” into
   numeric cost.
4. Keep benchmark-software licenses separate from underlying dataset licenses.
5. Give every reported fact a primary source URL, locator, evidence type,
   confidence, and scope. Record an explicit reason and searched sources for
   every missing fact.

No numeric values are imputed. Sample count is never treated as money,
runtime, compute, or annotation labor.

## Measured-burden infrastructure

The source registry describes what publications and benchmark repositories
say. A separate [measurement schema](../data/evaluation_burden_measurements.schema.json)
now defines what a controlled local run must record. Receipts are keyed by
model revision, evaluation identity, run-configuration hash, hardware
identity, and cache scope. They also carry an artifact group and one of four
non-overlapping phases:

1. shared dataset/artifact setup, charged once per artifact group;
2. per-model feature extraction;
3. per-protocol head fitting; and
4. per-protocol evaluation.

The [artifact grouping rules](../data/evaluation_artifact_cost_groups.csv)
prevent validation/test or protocol variants from repeatedly charging the same
dataset download and staging work. The
[budget profiles](../data/evaluation_budget_profiles.json) are deliberately
templates: users must provide the actual limits and access policy. There is no
built-in conversion between GPU time, labor, storage, tissue, and dollars.

Every resource fact uses one of `measured`, `source_reported`,
`configured_ceiling`, `not_applicable`, `not_measured`, `not_reported`, or
`inaccessible`. The latter four states carry `null`, never numeric zero.
Unknown dimensions fail closed only when a selected budget constrains that
dimension; otherwise they remain visible in measurement-coverage reporting.

The telemetry wrapper currently measures wall time, child CPU time, and peak
child resident memory using the local operating system. It leaves accelerator
time, VRAM, data transfer/storage, access lead time/labor,
annotation/pathologist labor, tissue/slides, access and license constraints,
and direct dollars explicitly unmeasured until dedicated instrumentation or
billing evidence is attached. A receipt can be collected as:

```bash
python3 scripts/run_burden_telemetry.py \
  --model-revision MODEL@REVISION \
  --evaluation-id EVALUATION_ID \
  --artifact-group-id SUITE.dataset.DATASET_ID \
  --phase per_protocol_evaluation \
  --hardware-id HOST_OR_CLUSTER_ID \
  --cache-scope warm \
  --run-config path/to/config.yaml \
  --output measurements/receipt.json \
  -- command --and its --arguments
```

The runner invokes the command directly without a shell, writes atomically,
refuses to overwrite a receipt unless `--force` is explicit, and propagates the
child exit code after preserving failed-run telemetry.

Audited receipts feed the independent
[budget-constrained probe selector](budgeted-probe-selection.md). The checked-in
[preflight artifact](../experiments/budgeted_probe_selection_rank1.json) binds
the current 59 × 187 matrix and records `insufficient_cost_coverage`; it contains
no selected set or numeric cost frontier.

## What the sources support

| Evidence field | Any sourced context | Evaluation-specific | Interpretation |
|---|---:|---:|---|
| Sample count | 147/187 | 147/187 | Exact task counts, split counts, or evaluated metadata rows |
| Sample unit | 187/187 | 187/187 | Image, patch, slide, case, or spatial spot |
| Access evidence | 177/187 | 31/187 | Family-level access notes are not blanket dataset-access claims |
| Supplied label artifact | 187/187 | 187/187 | Label type only; not annotation labor |
| Acquisition scale | 16/187 | 16/187 | THUNDER mpp and image dimensions; no magnification conversion |
| Stain | 18/187 | 0/187 | HEST family explicitly states H&E |
| Compute configuration | 177/187 | 15/187 | Mostly family/default context; EVA has task-specific configs |
| Dataset license | 28/187 | 28/187 | Only when the benchmark source explicitly states it |
| Hardware make/model | 0/187 | 0/187 | Missing |
| Observed runtime | 0/187 | 0/187 | Missing |
| Annotation hours | 0/187 | 0/187 | Missing |
| Dollar cost | 0/187 | 0/187 | Missing |

The exact sample-count coverage comprises all 122 retained Patho-Bench rows,
all 16 THUNDER rows, five of the six PathoROB rows, and four of the ten
H-Optimus-1 report rows. The remaining 40 protocols (15 EVA, 18 HEST, one
PathoROB protocol, and six H-Optimus-1 report protocols) do not have an
evaluation-specific count in the audited source. HEST's source does report
1,276 paired samples and a corpus larger than
2 TB, but those corpus-level values are not substituted for per-organ spatial
spot counts.

## Primary source snapshots

The audit uses committed, official repositories and the task/report URL
already attached to every retained protocol:

- [Patho-Bench README at `660e770`](https://github.com/mahmoodlab/Patho-Bench/blob/660e77044640e3d7d2f1150cc6721e97454993bf/README.md)
  documents automatic split/config download, required Trident features, and
  GPU load-balanced large runs. Its
  [repository license](https://github.com/mahmoodlab/Patho-Bench/blob/660e77044640e3d7d2f1150cc6721e97454993bf/LICENSE)
  is recorded separately from dataset terms.
- [eva offline pathology configs at `e43e74a`](https://github.com/kaiko-ai/eva/tree/e43e74a99b75660b0014f790f25a33dd9f11e121/configs/vision/pathology/offline)
  provide run counts, step/epoch limits, batch sizes, resize dimensions,
  accelerator mode, and device count. These are budgets/defaults. Dataset
  licenses are retained only where a task config states one. eva's software
  license is [Apache-2.0](https://github.com/kaiko-ai/eva/blob/e43e74a99b75660b0014f790f25a33dd9f11e121/LICENSE).
- [THUNDER dataset configs at `3d1cc95`](https://github.com/MICS-Lab/thunder/tree/3d1cc9513fb2cfd8c4afb0d7bb9f5c4f6b69117f/src/thunder/config/dataset)
  provide train/validation/test counts, mpp, and image sizes. The
  [README](https://github.com/MICS-Lab/thunder/blob/3d1cc9513fb2cfd8c4afb0d7bb9f5c4f6b69117f/README.md)
  describes SLURM orchestration and a 32 GB VRAM minimum for segmentation;
  that requirement is not assigned to the retained linear-probe rows.
- [HEST at `3ddb5ea`](https://github.com/mahmoodlab/HEST/blob/3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3/README.md)
  reports 1,276 H&E/spatial-transcriptomics pairs, free subset access, and a
  corpus larger than 2 TB. Its
  [benchmark config](https://github.com/mahmoodlab/HEST/blob/3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3/bench_config/bench_config.yaml)
  declares batch size 128 and one worker; those are not runtime measurements.
- [PathoROB at `6583cf0`](https://github.com/bifold-pathomics/PathoROB/blob/6583cf0b0d902c8cc032308262fa3a3befdc0687/README.md)
  reports an approximately 100,000-image, 2 GB feature-extraction download and
  per-dataset licenses. Exact evaluated patch counts come from its pinned
  metadata CSVs and OOD-exclusion code, while its
  [feature-extraction defaults](https://github.com/bifold-pathomics/PathoROB/blob/6583cf0b0d902c8cc032308262fa3a3befdc0687/pathorob/features/extract_features.py)
  remain configuration evidence.
- The [H-Optimus-1 launch report](https://www.bioptimus.com/news/bioptimus-launches-h-optimus-1)
  is the primary source attached to ten retained report-table protocols. The
  registry preserves its reported task counts where available but does not
  infer a benchmark software license, access mechanism, or executable compute
  configuration from the report.

## Pre-error feasibility strata

The registry assigns four deterministic strata before looking at any
prediction error:

| Stratum | Rule | Protocols |
|---|---|---:|
| 1: direct, small, labeled | Image/patch classification with a directly reported total ≤10,000 | 5 |
| 2: direct, labeled | Other image/patch classification | 24 |
| 3: aggregated or WSI | Case/slide classification, survival, or retrieval | 132 |
| 4: specialized | Segmentation, spatial regression, robustness, or another specialized protocol | 26 |

These are feasibility strata, not measured cost tiers. In particular, they do
not silently turn sample count into a price or ignore access and licensing
differences.

## Why there is no numeric cost curve

A valid numeric curve needs a common auditable denominator—such as observed
GPU-hours on a specified device, measured elapsed time under a fixed harness,
or directly reported dollars. The current sources instead mix sample counts,
configuration ceilings, qualitative access information, and family-level
storage requirements. Combining those into one number would require invented
exchange rates and hardware assumptions.

The supported outputs are therefore:

- source coverage and missingness;
- raw task/configuration evidence in its original units;
- pre-error feasibility-stratified analyses with an explicit proxy label.

The unsupported output is a “dollars versus error” or “runtime versus error”
curve. That becomes supportable only after a controlled benchmark run records
hardware identity, software environment, wall time, accelerator time, energy
or cloud billing where applicable, and whether feature extraction/data staging
are included.

## Reproduction

```bash
python3 scripts/build_evaluation_cost_evidence.py
PYTHONPATH=src python3 -m unittest \
  tests.test_evaluation_cost_evidence \
  tests.test_burden_telemetry -v
```

The tests verify exact 187-protocol coverage, source resolution for every
non-null fact, explicit missingness, raw THUNDER/PathoROB numbers, separation
of software and dataset licenses, zero invented numeric costs, and byte-stable
CSV plus semantically stable JSON regeneration. Telemetry tests additionally
verify the full status vocabulary, phase/key contract, null unknowns, shared
artifact grouping, non-numeric budget templates, and overwrite protection.
