# Feasibility of pathology benchmark compression

_Evidence snapshot: 2026-08-06. Upstream suites are living projects; registry
releases should continue to pin revisions and extraction dates._

## Bottom line

Applying the BenchPress idea to pathology is technically tractable and the
current repository demonstrates an end-to-end implementation. The completion
algorithm was the easy part. The difficult work is maintaining trustworthy
protocol identities, extracting exact checkpoint-level scores, measuring probe
cost, and validating predictions under new-model, new-family, new-time, and
new-institution shifts.

PathoPress currently provides:

- a five-suite registry with 287 protocols over 145 task identities;
- 1,976 reported score rows, of which 1,967 form a fixed 59 × 165 research
  matrix at 20.2054% density;
- a hash-bound matrix and ten-seed, three-fold artifact;
- direct numerical parity for the BenchPress point completer and core classical
  methods;
- a 343-configuration, 12-method × 7-transform comparison;
- low-rank, correlation, MDS, probe, ranking, confidence, temporal, and error
  factor analyses;
- a prediction CLI, manifest-verified public export, and static browser app.

That establishes engineering and retrospective research feasibility. It does
not establish clinical validity or prospective generalization.

## What is being compressed

[BenchPress](https://github.com/microsoft/benchpress) compresses a sparse
model × evaluation scorecard: evaluate a model on a small selected panel, then
infer supported missing scores. It does not compress the benchmark's images,
reduce patch counts, or make different benchmark runners interchangeable.

A pathology implementation therefore has four separate entities:

| Layer | Identity | Why it must remain separate |
|---|---|---|
| Dataset artifact | versioned cohort/files/splits/license | a dataset name is not a reproducible sample set |
| Task identity | artifact + target + granularity | supports conservative conceptual deduplication |
| Evaluation protocol | split + preprocessing + head/pooling + metric + aggregation | defines a comparable matrix column |
| Score observation | exact model revision + protocol + value + source/audit status | defines one measured matrix cell |

Deduplicate tasks for catalog navigation, but preserve protocols as columns.
Two papers using PANDA, BACH, or TCGA may change splits, label maps,
magnification, aggregation, head, metric, or hyperparameter selection enough
that their scores cannot be merged.

## Suite extraction and overlap

The audited registry covers:

| Suite | Registry role | Primary project |
|---|---|---|
| Patho-Bench | classification/survival protocols and paper-specific variants | [GitHub](https://github.com/mahmoodlab/patho-bench) |
| EVA | patch/slide frozen-encoder evaluations | [GitHub](https://github.com/kaiko-ai/eva) |
| THUNDER | classification/segmentation configurations | [GitHub](https://github.com/MICS-Lab/thunder) |
| HEST | morphology-to-expression tasks | [GitHub](https://github.com/mahmoodlab/HEST) |
| PathoROB | robustness endpoints | [GitHub](https://github.com/bifold-pathomics/PathoROB) |

Among Patho-Bench, EVA, and THUNDER, a conservative dataset × target ×
granularity key finds 122 code-backed task identities and only seven exact
cross-suite overlaps: EVA/THUNDER BACH, BRACS, four-class BreakHis, CRC-100K,
MHIST, PatchCamelyon, and Patho-Bench/EVA full PANDA slide-level ISUP grading.
The intuition that “lots are repeated” is more true at the dataset-family or
cancer-type level than at the exact evaluation level.

The machine-readable evidence is in [tasks.csv](../data/tasks.csv),
[deduplication.csv](../data/deduplication.csv), and
[provenance.json](../data/provenance.json). Score extraction and exclusions are
documented in [score-source-coverage.md](score-source-coverage.md). The score
pool contains Patho-Bench 896, EVA 265, HEST 234, THUNDER 512, and PathoROB 69
rows. Nine `reported_external` rows are excluded from the fixed paper matrix.

## Does the matrix support completion?

Yes, retrospectively, with material heterogeneity.

- Matched rank-1 bias ALS: 3.005264 MAE / 1.603026 MedAE versus column median
  4.092133/2.477500 over 19,670 OOF predictions.
- A largest complete 32 × 16 block has stable rank 1.431; its first one/two
  components explain 69.88%/87.57% of variance.
- All 165 retained evaluations have a neighbor sharing at least five models;
  the median best absolute correlation is 0.918881.
- Random-cell rank 1 is 2.834996/1.526795, but leave-one-suite-block-out rank 1
  degrades to 5.612789/3.525174 and prefers rank 5 at 4.952972/3.055638.
- Sparse-new-model rank 1 is 3.190380/1.817465 over 4,896 prediction instances.

The result is useful structure, not one universal low-dimensional pathology
ability. Suite-block missingness, author-selected evaluations, model-family
clustering, and native endpoint differences can all create apparent low rank.

## Does a small panel work?

Retrospectively, yes. From a 1.900 all-known scorecard MedAE baseline, greedy
rank-1 selection reaches 1.481124 with five probes and 1.196456 with ten.
Strict hidden-only values are 1.612112 and 1.539134; isolated held-out-model
values are 1.951271 and 1.879857.

Those protocols answer different questions. The all-known curve is
transductive and includes exact revealed cells. Held-out-model validation is
closer to deployment but still chooses from published retrospective rows.

The current four-protocol “low-friction” allowlist is a reproducible pre-error
metadata rule, not a measured cost model. A defensible practical panel still
needs runtime, GPU/memory, sample acquisition, staining, annotation, tissue,
licensing, and runner-reproducibility audits. Exhaustive results are exact only
for the declared four-item allowlist and a 12-candidate pruned `k=2` space, not
all subsets of 165 evaluations.

## What remains difficult

| Remaining issue | Difficulty | What credible completion requires |
|---|---:|---|
| Dual review and living extraction | High, ongoing | two-source/second-review checks, correction log, pinned releases, exact table locations |
| Real probe-cost model | High | measured compute/runtime plus data, tissue, label, access, and license constraints |
| Prospective validation | High | preregistered newer models and external institutions not used during selection |
| Family/site leakage analysis | High | stronger canonical family, pretraining, cohort, and institution metadata |
| Confidence for unseen rows | High | calibration designed around genuinely new models, with abstention |
| Real LLM baselines | Low engineering, nonzero cost | provider execution for four named/blind conditions; current runner has no network client |
| Hosted release/maintenance | Moderate, ongoing | deployment, versioned snapshots, CI freshness, review/contribution workflow |

The real LLM baseline is not a scientific blocker: upstream treats it as an
experimental comparator, while the classical grid contains the core evidence.
PathoPress has hash-bound request/response schemas, deterministic mock contract
tests, and merge logic. [Real-run status](../experiments/llm_baseline/real_run_status.json)
is explicitly `unrun`; mock errors are not headline eligible.

## Release gates

A defensible living release should enforce:

1. Every score has an exact checkpoint, protocol, source location, and audit
   status; automated extraction creates a candidate, not verified truth.
2. Matrix/filter/fold hashes fail closed when inputs drift.
3. Coverage accompanies every method and probe result.
4. Random-cell, held-out-row, suite/family/site, and time-aware evaluations are
   reported separately.
5. Native metrics, normalized values, reported values, and predictions remain
   distinguishable.
6. Intervals use held-out residuals and unsupported/new populations abstain.
7. Probe selection is constrained by measured feasibility before it is called
   “cheap.”
8. No prediction is presented as evidence of diagnostic safety, subgroup
   performance, external-site validity, or clinical utility.

## Reproduce the evidence

```bash
PYTHONPATH=src python3 scripts/build_shared_artifacts.py
PYTHONPATH=src python3 experiments/run_benchpress_style.py
PYTHONPATH=src python3 experiments/run_structure_analysis.py
PYTHONPATH=src python3 experiments/run_probe_compression.py
PYTHONPATH=src python3 experiments/run_probe_exhaustive.py --workers 8
PYTHONPATH=src python3 experiments/run_confidence_calibration.py
PYTHONPATH=src python3 experiments/run_temporal_deployment.py
```

See [the experiment index](../experiments/README.md) before rerunning expensive
or sharded jobs, [the full parity audit](full-parity-audit.md) for the complete
status matrix, and [the public export notice](../exports/pathopress_public/LICENSES.md)
for redistribution boundaries.
