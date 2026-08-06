# Evaluation cost evidence

PathoPress now has a source-backed cost-evidence registry for all 165 retained
evaluation protocols. Its main conclusion is negative but important: a true
numeric cost curve is **not currently supportable**. None of the retained
protocol sources reports both observed evaluation runtime and dollar cost; in
fact, observed runtime, hardware make/model, annotation hours, and dollar cost
are each available for **0 of 165** protocols.

The machine-readable records are in
[`data/evaluation_cost_evidence.json`](../data/evaluation_cost_evidence.json),
with a flat audit view in
[`data/evaluation_cost_evidence.csv`](../data/evaluation_cost_evidence.csv).
The [coverage figure](../figures/evaluation_cost_evidence_coverage.png) shows
what is and is not known.

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

## What the sources support

| Evidence field | Any sourced context | Evaluation-specific | Interpretation |
|---|---:|---:|---|
| Sample count | 141/165 | 141/165 | Exact task counts, split counts, or evaluated metadata rows |
| Sample unit | 165/165 | 165/165 | Image, patch, slide, case, or spatial spot |
| Access evidence | 165/165 | 31/165 | Family-level access notes are not blanket dataset-access claims |
| Supplied label artifact | 165/165 | 165/165 | Label type only; not annotation labor |
| Acquisition scale | 16/165 | 16/165 | THUNDER mpp and image dimensions; no magnification conversion |
| Stain | 9/165 | 0/165 | HEST family explicitly states H&E |
| Compute configuration | 165/165 | 15/165 | Mostly family/default context; EVA has task-specific configs |
| Dataset license | 17/165 | 17/165 | Only when the benchmark source explicitly states it |
| Hardware make/model | 0/165 | 0/165 | Missing |
| Observed runtime | 0/165 | 0/165 | Missing |
| Annotation hours | 0/165 | 0/165 | Missing |
| Dollar cost | 0/165 | 0/165 | Missing |

The exact sample-count coverage comprises all 122 retained Patho-Bench rows,
all 16 THUNDER rows, and all three PathoROB rows. The remaining 24 protocols
(15 EVA and nine HEST) do not have an evaluation-specific count in the audited
source. HEST's source does report 1,276 paired samples and a corpus larger than
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

## Pre-error feasibility strata

The registry assigns four deterministic strata before looking at any
prediction error:

| Stratum | Rule | Protocols |
|---|---|---:|
| 1: direct, small, labeled | Image/patch classification with a directly reported total ≤10,000 | 4 |
| 2: direct, labeled | Other image/patch classification | 21 |
| 3: aggregated or WSI | Case/slide classification, survival, or retrieval | 126 |
| 4: specialized | Segmentation, spatial regression, robustness, or another specialized protocol | 14 |

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
python3 scripts/plot_evaluation_cost_evidence.py
PYTHONPATH=src python3 -m unittest tests.test_evaluation_cost_evidence -v
```

The tests verify exact 165-protocol coverage, source resolution for every
non-null fact, explicit missingness, raw THUNDER/PathoROB numbers, separation
of software and dataset licenses, zero invented numeric costs, and byte-stable
CSV plus semantically stable JSON regeneration.
