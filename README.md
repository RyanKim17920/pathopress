# PathoPress

Citation-backed score-matrix completion for pathology foundation-model
benchmarks, adapted from [Microsoft BenchPress](https://github.com/microsoft/benchpress).

## What it does

PathoPress does not compress pathology images or replace benchmark runners. It
builds an auditable model × evaluation score matrix, predicts supported missing
cells, and studies which small evaluation panels preserve score and ranking
information. The source registry covers 20 paper/report/repository suites,
including Patho-Bench, EVA, THUNDER, HEST, PathoROB, and H-Optimus-1, while
keeping dataset identity, task identity, evaluation protocol, and score
observation distinct.

The full registry contains 1,638 protocol rows over 394 task identities and
4,013 reported score rows. The fixed research matrix accepts primary-source-parsed
evidence and iteratively requires at least three scores per model and five models
per evaluation. It retains **59 models × 187 evaluations, 2,122 reported cells,
and 19.2332% density**. The retained columns are Patho-Bench 122, EVA 15,
THUNDER 16, H-Optimus-1 report 10, HEST 18, and PathoROB 6.

This is a research prototype. The scores are machine-parsed from pinned primary
sources, not dual-human-verified; normalized points mix several native metrics;
and retrospective interpolation is not prospective clinical validation.

## Main findings

- The logit, per-evaluation-standardized bias-ALS implementation matches the
  pinned BenchPress numerical primitive to floating-point precision. Pathology
  cell-level cross-validation selects interaction rank 1; this is a completion
  hyperparameter, not the literal matrix rank. BenchPress deploys rank 2.
- On the shared 10-seed × 3-fold artifact, rank-1 bias ALS gives 3.134532 MAE,
  1.609006 MedAE, and 1.608566 median fold MedAE over 21,181 supported
  repeated held-out prediction instances from 2,122 unique reported cells.
  Other scores from the same model may remain visible. The column-median
  baseline is 4.151756/2.400000 MAE/MedAE. Both raw and logit Soft-Impute
  sweeps independently select rank 1.
- The current 59×187 7-transform × 12-method comparison ran 343/343 configurations.
  The best MedAPE row is logit BenchReg at 1.9023, but it covers only 68.3% of
  held-out cells. Coverage is therefore part of every method result; a partial
  regression row does not replace the full-coverage selected predictor.
- With the all-known BenchPress denominator, greedy rank-1 scorecard MedAE is
  1.397334 at five probes and 1.213706 at ten. Hidden-only values are 1.548536
  and 1.493709. These are transductive reconstruction results, not estimates of
  a new model's clinical utility.
- The margin-5 ranking-aware greedy trajectory reaches 0.7321 pairwise accuracy
  at five unrestricted probes and 0.8780 at ten. The 25-task pipeline-feasibility
  proxy reaches 0.4000 at ten; it is not a measured cost set. No exhaustive
  choose-five search has been run for the current 59×187 scores.
- Prefixes selected on 41 training models predict the mean reported normalized
  score of 18 held-out models. Median model-mean error for unrestricted probes
  is 1.9039 at one probe, 0.7835 at five, and 0.8921 at ten; the 25-task proxy
  is 1.2254/0.9247 at five/ten. These non-monotone held-out values include exact
  revealed probes and supported hidden predictions. There is no committed
  held-out `k=0` or random model-mean control.
- The standalone ranking-preservation release is now derived from the current
  compression artifact: at `k=10`, unrestricted all-known margin-5 pairwise
  accuracy is 0.8780 versus a 0.6029 random-prefix median, and hidden-only
  held-out-model accuracy is 0.7750. It supersedes the earlier-snapshot OOF
  margin-sensitivity release.
- The pinned BenchPress 3+12 confidence generator contract is reproduced with
  pathology rank 1, eight structural features, nested ridge/MLP risk selection,
  conformal intervals, and cross-fitted P(|error| <= 10 normalized points).
  Cross-fitted structural-support uncertainty correlates with absolute error at
  Spearman 0.6022; its leave-fold-out 90% intervals cover 89.982% of 21,181 OOF
  instances. A separate unseen-model artifact calibrates 31,163 of 33,272 nested
  leave-model-out/temporal predictions; its nominal-90% empirical coverage is
  94.80% with 14.72-point median width, and unsupported columns abstain.
- A hard-rule temporal experiment predicts seven 2025 model releases using
  strictly earlier models and 1/5/10 revealed cells across ten seeds. It is a
  small retrospective release cohort, not external or prospective validation.

Detailed protocol distinctions and remaining gaps are in
[the parity note](docs/benchpress-parity.md).

## Install and quick start

The core CLI needs only the base package:

```bash
python3 -m pip install -e .
pathopress audit --scores data/scores.csv
pathopress validate --scores data/scores.csv
```

Install `.[research]` before regenerating analyses or figures. Install
`.[research,mlp]` only when rerunning the optional PyTorch MLP method-grid
units; the grid records dependency failures rather than silently dropping them.

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
hash-bound to the source scores and exact point recipe. Existing supported cells
also report calibrated P(|error| <= 10 normalized points); unsupported cells
abstain. Existing-row and unseen-model intervals use separate populations, and
unseen models explicitly abstain from the existing-row trust probability. New-model
output reports the conservative k bucket, risk, interval, fallback scope,
calibration group/prediction counts, or an explicit unsupported-context
abstention. These are retrospective empirical intervals, not clinical guarantees.

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
contains all/paper model, evaluation, and score tables. Its checked-in wide
matrix predates the current 59 × 187 analysis refresh. It also includes
sanitized provenance, license caveats, and a file-hash manifest. Load or fetch a
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
| Full classical grid | [manifest](experiments/method_comparison/manifest.json), [results](experiments/method_comparison/results.json), [top table](experiments/method_comparison/top_methods.md) |
| Probe compression | [selection](experiments/probe_selection_results_rank1.json), [compression](experiments/probe_compression_rank1.json), [top-30 pruning](experiments/probe_pruning_rank1_top30.json), [BenchPress-style hero](figures/pathopress_benchpress_hero_rank1.png), [task utility and held-out mean figure](figures/probe_dual_objective_rank1.png), [dual-objective table](outputs/probe_dual_objective_rank1.csv) |
| Cost and feasibility evidence | [source-backed registry](data/evaluation_cost_evidence.json), [measured-burden contract](docs/budgeted-probe-selection.md), [current fail-closed preflight](experiments/budgeted_probe_selection_rank1.json) |
| Ranking and time | [ranking](experiments/ranking_preservation_rank1.json), [temporal](experiments/temporal_deployment_rank1.json) |
| Trust | [confidence](experiments/confidence_calibration_rank1.json), [existing-row intervals](experiments/deployment_confidence_rank1.json), [unseen-model intervals](experiments/new_model_confidence_rank1.json), [new-model method](docs/new-model-confidence.md) |
| Figures | [figure gallery](figures/README.md) |
| Product surface | [CLI](src/pathopress/cli.py), [public export](exports/pathopress_public/README.md), [static site](website/README.md) |

Large method and exact-search shard caches are rebuildable and are not retained
in the compact repository. The checked-in method manifest and results preserve
the completed 343-unit current-matrix summary.

## Experiments

The concise command and artifact index is [experiments/README.md](experiments/README.md).
Core regeneration commands are:

```bash
PYTHONPATH=src python3 experiments/run_method_comparison.py --prepare-folds
# Run independent shards, then:
PYTHONPATH=src python3 experiments/run_method_comparison.py --merge
PYTHONPATH=src python3 experiments/run_probe_compression.py
PYTHONPATH=src python3 experiments/build_probe_pruning.py
# New matrices require new schema-v2 run directories and a full exact rerun.
# Replace NEW_SCORE_SHA12 after the registry/matrix refresh.
CHEAP_RUN=experiments/probe_exhaustive_runs/cheap25_medae_k5_mNEW_SCORE_SHA12
PRUNED_RUN=experiments/probe_exhaustive_runs/pruned30_medae_k5_mNEW_SCORE_SHA12
PYTHONPATH=src:experiments python3 experiments/verify_fast_rank1.py
# Run all declared v2 cheap/pruned residues before these certification steps.
PYTHONPATH=src:experiments python3 experiments/validate_probe_exhaustive_chunks.py \
  "$CHEAP_RUN" "$PRUNED_RUN"
PYTHONPATH=src:experiments python3 experiments/run_probe_exhaustive_v2.py merge \
  --out-dir "$CHEAP_RUN" --top-n 1001 \
  --integrity-manifest experiments/probe_exhaustive_integrity_manifest.json
PYTHONPATH=src:experiments python3 experiments/run_probe_exhaustive_v2.py merge \
  --out-dir "$PRUNED_RUN" --top-n 1001 \
  --integrity-manifest experiments/probe_exhaustive_integrity_manifest.json
PYTHONPATH=src:experiments python3 experiments/validate_probe_exhaustive_merged.py \
  "$CHEAP_RUN" "$PRUNED_RUN"
PYTHONPATH=src:experiments python3 experiments/validate_probe_exhaustive_top.py \
  "$CHEAP_RUN" "$PRUNED_RUN"
PYTHONPATH=src:experiments python3 experiments/build_probe_exhaustive_summary.py \
  --cheap-run "$CHEAP_RUN" --pruned-run "$PRUNED_RUN" \
  --fast-equivalence experiments/probe_exhaustive_fast_equivalence_v2.json
PYTHONPATH=src python3 experiments/run_ranking_preservation.py
PYTHONPATH=src python3 experiments/run_confidence_calibration.py
PYTHONPATH=src python3 experiments/run_new_model_confidence.py
PYTHONPATH=src python3 experiments/run_temporal_deployment.py
python3 scripts/build_evaluation_cost_evidence.py
python3 scripts/plot_benchpress_style.py
python3 scripts/plot_benchpress_style_hero.py
python3 scripts/plot_probe_dual_objective.py
python3 scripts/plot_temporal_deployment.py
```

Some commands are intentionally expensive or sharded; consult the experiment
README before rerunning.

The final `59×187` public tables can be rebuilt and locally validated in the
pinned BenchPress Hugging Face maintenance layout without uploading anything:

```bash
python3 -m pip install -e '.[hf]'
PYTHONPATH=src python3 scripts/build_hf_dataset.py --parquet yes
PYTHONPATH=src python3 scripts/publish_hf_dataset.py
```

The second command is a dry run. Remote publication is intentionally doubly
opt-in and additionally requires `HF_TOKEN`; see the
[public dataset card](exports/pathopress_public/README.md).

## Scientific and legal boundaries

Normalized points are direction-preserving display/fitting values, not a common
clinical unit. HEST Pearson `r` maps to `50 × (r + 1)`, PathoROB RI and 0–1
metrics map to `100 × value`, weighted kappa maps to `50 × (kappa + 1)`, and
THUNDER's reported 0–100 F1 is unchanged. See
[docs/imputation.md](docs/imputation.md).

The cost-evidence audit finds no directly reported observed runtime, hardware
make/model, annotation hours, or dollar cost for any of the 187 retained
protocols. Its pre-error feasibility strata are auditable metadata proxies,
not a numeric cost curve. The measured-budget runner therefore reports
`insufficient_cost_coverage` instead of recommending a fake cheapest set; see
[the evidence note](docs/evaluation-cost-evidence.md) and
[budgeted-selection contract](docs/budgeted-probe-selection.md).

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
