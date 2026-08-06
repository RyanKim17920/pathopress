# PathoPress

Citation-backed score-matrix completion for pathology foundation-model
benchmarks, adapted from [Microsoft BenchPress](https://github.com/microsoft/benchpress).

## What it does

PathoPress does not compress pathology images or replace benchmark runners. It
builds an auditable model × evaluation score matrix, predicts supported missing
cells, and studies which small evaluation panels preserve score and ranking
information. The source registry covers Patho-Bench, EVA, THUNDER, HEST, and
PathoROB while keeping dataset identity, task identity, evaluation protocol, and
score observation distinct.

The full registry contains 287 protocol rows over 145 task identities and 1,976
reported score cells. The fixed research matrix accepts primary-source-parsed
evidence and iteratively requires at least three scores per model and five models
per evaluation. It retains **59 models × 165 evaluations, 1,967 reported cells,
and 20.2054% density**. The retained columns are Patho-Bench 122, EVA 15,
THUNDER 16, HEST 9, and PathoROB 3.

This is a research prototype. The scores are machine-parsed from pinned primary
sources, not dual-human-verified; normalized points mix several native metrics;
and retrospective interpolation is not prospective clinical validation.

## Main findings

- The logit, per-evaluation-standardized bias-ALS implementation matches the
  pinned BenchPress numerical primitive to floating-point precision. Pathology
  cross-validation selects interaction rank 1; BenchPress deploys rank 2.
- On the shared 10-seed × 3-fold artifact, rank-1 bias ALS gives 3.005264 MAE,
  1.603026 MedAE, and 1.609435 median fold MedAE. The column-median baseline is
  4.092133/2.477500 MAE/MedAE.
- The complete 7-transform × 12-method comparison ran 343/343 configurations.
  The best MedAPE row is log BenchReg at 1.8144, but it covers only 71.0% of
  held-out cells. Coverage is therefore part of every method result; a partial
  regression row does not replace the full-coverage selected predictor.
- A largest complete 32 × 16 submatrix has stable rank 1.431; its first and
  first two components explain 69.88% and 87.57% of variance. All 165
  evaluations have a correlation neighbor with at least five shared models;
  median best absolute correlation is 0.9189.
- With the all-known BenchPress denominator, greedy rank-1 scorecard MedAE is
  1.481124 at five probes and 1.196456 at ten. Hidden-only values are 1.612112
  and 1.539134. These are transductive reconstruction results, not estimates of
  a new model's clinical utility.
- OOF ranking preservation has median pairwise accuracy 0.7622/0.7954/0.8337/
  0.9049 at normalized-score margins 0/1/2/5, and median top-set recovery
  0.6786/0.7759/0.8133 at top fractions 10/20/30%.
- Cross-fitted structural-support uncertainty correlates with absolute error at
  Spearman 0.5980; its leave-fold-out 90% intervals cover 89.995% of 19,670 OOF
  instances. The separate deploy artifact gives suite-conditioned held-out-cell
  intervals only for supported existing rows, not genuinely unseen models.
- A hard-rule temporal experiment predicts seven 2025 model releases using
  strictly earlier models and 1/5/10 revealed cells across ten seeds. It is a
  small retrospective release cohort, not external or prospective validation.

Detailed protocol distinctions and remaining gaps are in
[the parity note](docs/benchpress-parity.md) and
[the full audit](docs/full-parity-audit.md).

## Install and quick start

The core CLI needs only the base package:

```bash
python3 -m pip install -e .
pathopress audit --scores data/scores.csv
pathopress validate --scores data/scores.csv
```

Install `.[research]` before regenerating analyses or figures. This extra
includes the compatible PyTorch dependency used by the MLP method-grid units;
the grid records dependency failures rather than silently dropping units.

The product CLI uses canonical IDs and supports CSV or JSON:

```bash
pathopress list-models --format csv
pathopress list-evaluations --format json
pathopress predict \
  --model atlas \
  --evaluation eva.leaderboard.bach.validation \
  --confidence --format json
pathopress complete-model --model atlas --output atlas-missing.csv
pathopress add-model --model my-model \
  --known-score eva.leaderboard.bach.validation=72 \
  --known-score eva.leaderboard.bracs.validation=68 \
  --format json --output my-model-predictions.json
```

`status=observed`, `provided`, and `predicted` remain distinct. Confidence is
hash-bound to the source scores and exact point recipe. New-model predictions
are explicitly marked `not_applicable_new_model` for confidence because the
calibration population contains existing model rows only.

## Reproduce the compact release

Build the shared matrix/folds, public tables, confidence artifact, and static
site data from the repository root:

```bash
PYTHONPATH=src python3 scripts/build_shared_artifacts.py
PYTHONPATH=src python3 scripts/build_public_release.py
python3 -m http.server 8000
```

Then open <http://localhost:8000/website/>. The site is static: lookup uses
generated JSON, add-model completion runs in the browser, and no score is
uploaded. See [website/README.md](website/README.md).

The public export at [exports/pathopress_public/](exports/pathopress_public/)
contains all/paper model, evaluation, and score tables; a 59 × 165 wide matrix;
sanitized provenance; license caveats; and a file-hash manifest. Load or fetch a
mirror with `pathopress.public_data` or:

```bash
PYTHONPATH=src python3 scripts/download_public_release.py BASE_URL DESTINATION
```

No build or download command performs deployment or upload.

## Artifact map

| Layer | Primary evidence |
|---|---|
| Registry and deduplication | [suites](data/suites.csv), [tasks](data/tasks.csv), [deduplication](data/deduplication.csv), [scores](data/scores.csv), [provenance](data/provenance.json) |
| Canonical substrate | [matrix/fold manifest](experiments/shared_artifacts_manifest.json), [matrix NPZ](experiments/analysis_matrix.npz), [folds](experiments/folds_s10_f3_bs42.json) |
| Point estimates and rank | [imputations](outputs/imputations_rank1.csv), [bias-ALS CV](experiments/benchpress_style_results.json), [Soft-Impute sweep](experiments/soft_impute_rank_sweep_results.json) |
| Full classical grid | [manifest](experiments/method_comparison/manifest.json), [results](experiments/method_comparison/results.json), [top table](experiments/method_comparison/top_methods.md), [grid figure](figures/method_comparison_grid.png) |
| Structure | [structure manifest](experiments/structure_analysis/manifest.json), [stable rank](experiments/structure_analysis/stable_rank_results.json), [MDS coordinates](experiments/structure_analysis/mds_coordinates.csv) |
| Probe compression | [selection](experiments/probe_selection_results_rank1.json), [compression](experiments/probe_compression_rank1.json), [bounded exhaustive search](experiments/probe_exhaustive_rank1.json), [hero](figures/pathopress_hero_rank1.png) |
| Ranking and time | [ranking](experiments/ranking_preservation_rank1.json), [temporal](experiments/temporal_deployment_rank1.json) |
| Trust and error factors | [confidence](experiments/confidence_calibration_rank1.json), [deploy intervals](experiments/deployment_confidence_rank1.json), [predictability](experiments/predictability_results_rank1.json), [factor analysis](experiments/prediction_error_factors_rank1.json) |
| Publication outputs | [table manifest](outputs/tables/manifest.json), [metadata summary](experiments/publication_metadata_summary.json), [figure gallery](figures/README.md) |
| Product surface | [CLI](src/pathopress/cli.py), [public export](exports/pathopress_public/README.md), [static site](website/README.md) |

Large resumable caches remain local. The 343 method NPZ shards occupy about 439
MiB under `experiments/method_comparison/predictions/` and are narrowly ignored
by Git; the merged manifest records their count, path, and size. Section 6 unit
shards and its 29 MiB raw prediction CSV are also ignored, while compact merged
records, tables, and figures remain tracked.

## Experiments

The concise command and artifact index is [experiments/README.md](experiments/README.md).
Core regeneration commands are:

```bash
PYTHONPATH=src python3 experiments/run_method_comparison.py --prepare-folds
# Run independent shards, then:
PYTHONPATH=src python3 experiments/run_method_comparison.py --merge
PYTHONPATH=src python3 experiments/run_structure_analysis.py
PYTHONPATH=src python3 experiments/run_probe_compression.py
PYTHONPATH=src python3 experiments/run_probe_exhaustive.py --workers 8
PYTHONPATH=src python3 experiments/run_ranking_preservation.py
PYTHONPATH=src python3 experiments/run_confidence_calibration.py
PYTHONPATH=src python3 experiments/run_temporal_deployment.py
PYTHONPATH=src python3 experiments/run_prediction_error_factors.py --workers 8
```

Some commands are intentionally expensive or sharded; consult the experiment
README before rerunning. The real named/blind matrix and five-shot LLM baselines
have provider-neutral, no-network request/cache scaffolding, but no real-provider
responses and therefore no headline-eligible results.

## Scientific and legal boundaries

Normalized points are direction-preserving display/fitting values, not a common
clinical unit. HEST Pearson `r` maps to `50 × (r + 1)`, PathoROB RI and 0–1
metrics map to `100 × value`, weighted kappa maps to `50 × (kappa + 1)`, and
THUNDER's reported 0–100 F1 is unchanged. See
[docs/imputation.md](docs/imputation.md).

Random-cell and within-model folds share model/suite context. Suite-block,
held-out-row, and temporal experiments are stronger stress tests but still use
published retrospective evidence. Pathology pretraining/evaluation overlap,
publication selection, protocol drift, and institutional shortcuts remain
possible. Predictions prioritize real evaluations; they do not establish
diagnostic safety, subgroup performance, external-site validity, or clinical
utility.

PathoPress code is [MIT-licensed](LICENSE), including attributed BenchPress
adaptations described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Registry facts do not relicense benchmark data, source publications, images,
labels, or model weights; see [DATA_NOTICE.md](DATA_NOTICE.md) and the public
export's [license notice](exports/pathopress_public/LICENSES.md).
