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

- a 20-suite evidence registry with 1,638 protocols over 394 task identities;
- 4,013 reported score rows, of which 2,122 form a fixed 59 × 187 research
  matrix at 19.2332% density;
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
| H0-mini / UNI2-h companions | EVA, HEST, PLISM, and UNI-repository evaluations | pinned paper and repository snapshots |
| Virchow2-family / TITAN | tile, slide, survival, zero-shot, and retrieval protocols | pinned primary-paper extracts |
| H-Optimus-1 / Virchow2 / Virchow / UNI | report- and paper-specific frozen-feature protocols | pinned first-party reports and papers |
| CONCH / Phikon families | patch- and slide-level primary-paper protocols | pinned primary-paper extracts |
| Hibou / MUSK / GPFM | public frozen-feature protocols | pinned papers and publisher workbooks |

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
pool contains 4,013 rows across the 20 source suites. Its largest blocks are
Patho-Bench 896, TITAN paper 629, THUNDER 524, HEST 377, EVA 317, TITAN patch
encoder 297, and UNI paper 227. Forty-six signed APD rows and six public rows
without a source-specified endpoint metric remain in the raw registry but are
analysis-ineligible because no bounded common-scale mapping can be justified; nine
`reported_external` rows are also excluded from the fixed paper matrix.

## Does the matrix support completion?

Yes, retrospectively, with material heterogeneity.

- Matched rank-1 bias ALS: 3.134532 MAE / 1.609006 MedAE versus column median
  4.151756/2.400000 over 21,181 OOF predictions.
- Random-cell rank 1 is 2.937385/1.568476 over 4,240 predictions.
- Leave-one-suite-block-out rank 1 degrades to 5.788534/3.599615 and improves
  through tested rank 6 at 5.099363/3.203260; 95 column targets lack training
  support and are reported as unsupported rather than silently scored.
- Sparse-new-model rank 1 is 3.285791/1.753653 over 5,331 prediction instances.
- The existing complete-submatrix/correlation structure artifact predates the
  187-column registry expansion and must be regenerated before reuse.

The result is useful structure, not one universal low-dimensional pathology
ability. Suite-block missingness, author-selected evaluations, model-family
clustering, and native endpoint differences can all create apparent low rank.

## Does a small panel work?

Retrospectively, yes. From a 1.935 all-known scorecard MedAE baseline, greedy
rank-1 selection reaches 1.474879 with five probes and 1.270529 with ten.
Strict hidden-only values are 1.637639 and 1.538607; isolated held-out-model
values are 2.126261 and 2.142613.

Those protocols answer different questions. The all-known curve is
transductive and includes exact revealed cells. Held-out-model validation is
closer to deployment but still chooses from published retrospective rows.

The v2 25-protocol “low-friction” allowlist is a reproducible pre-error
input/label pipeline rule, not a measured cost model. It matches BenchPress's
candidate count while adapting the task identities to pathology. A defensible
practical panel still needs runtime, GPU/memory, sample acquisition, staining,
annotation, tissue, and licensing audits. Exact choose-five plans cover 53,130
pre-error and 142,506 error-informed combinations. Both searches are complete:
their certified MedAE optima are 1.485944 and 1.427339, respectively. This does
not turn the 25-task proxy into measured cost, and neither result is globally
exhaustive over the legacy 168-column snapshot. The current 187-column probe
artifacts must be regenerated before those values are used as current results.

## What remains difficult

| Remaining issue | Difficulty | What credible completion requires |
|---|---:|---|
| Dual review and living extraction | High, ongoing | two-source/second-review checks, correction log, pinned releases, exact table locations |
| Real probe-cost model | High | measured compute/runtime plus data, tissue, label, access, and license constraints |
| Prospective validation | High | preregistered newer models and external institutions not used during selection |
| Family/site leakage analysis | High | stronger canonical family, pretraining, cohort, and institution metadata |
| Confidence for unseen rows | High | calibration designed around genuinely new models, with abstention |
| Hosted release/maintenance | Moderate, ongoing | deployment, versioned snapshots, CI freshness, review/contribution workflow |

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
PYTHONPATH=src python3 experiments/build_probe_pruning.py
# Validate the completed hash-bound exact-search artifacts.
PYTHONPATH=src python3 experiments/validate_probe_exhaustive_chunks.py
PYTHONPATH=src python3 experiments/validate_probe_exhaustive_merged.py
PYTHONPATH=src python3 experiments/validate_probe_exhaustive_top.py
PYTHONPATH=src python3 experiments/build_probe_exhaustive_summary.py
PYTHONPATH=src python3 experiments/run_confidence_calibration.py
PYTHONPATH=src python3 experiments/run_temporal_deployment.py
```

See [the experiment index](../experiments/README.md) before rerunning expensive
or sharded jobs, [the full parity audit](full-parity-audit.md) for the complete
status matrix, and [the public export notice](../exports/pathopress_public/LICENSES.md)
for redistribution boundaries.
