# PathoPress–BenchPress parity audit

This is an evidence-backed audit against Microsoft BenchPress commit
[`0a684b63ee0e4a401cb907a3827a82ea997d74c4`](https://github.com/microsoft/benchpress/tree/0a684b63ee0e4a401cb907a3827a82ea997d74c4).
It distinguishes exact numerical parity, pathology-adapted scientific choices,
bounded computations, product analogues, and genuinely unrun experiments.

## Bottom line

PathoPress now implements the complete practical pipeline: a provenance-aware
registry, immutable matrix/folds, point completion, broad classical comparison,
structure analysis, score and ranking probe selection, confidence, temporal and
factor analyses, publication assets, CLI, public export, and static browser
predictor. This is much more than an algorithm sketch.

It is not accurate to call the repository 100% identical to BenchPress.
Pathology uses different native metric mappings, selects rank 1 instead of the
upstream deployed rank 2, uses a narrower confidence ensemble, bounds expensive
probe enumeration, substitutes a metadata feasibility proxy for measured cost,
has no hosted deployment/upload, and has not run the real LLM baselines.

## Status vocabulary

- **Exact algorithmic parity:** same core computation/parameters, with direct
  reference evidence.
- **Pathology-adapted:** same scientific role with declared domain-specific
  inputs, normalization, selection, or validation.
- **Bounded:** complete only within an explicit tractability-limited search
  space.
- **Engineering analogue:** comparable usable surface, not an upstream-identical
  deployment.
- **Unrun:** scaffolding exists but no scientific result has been produced.

## Canonical evidence

The registry contains 287 protocols over 145 task identities and 1,976 score
rows across Patho-Bench, EVA, HEST, THUNDER, and PathoROB. The fixed paper
matrix retains 59 models × 165 evaluations and 1,967 observed cells at 20.2054%
density. Its immutable contract is:

- source score SHA-256 `4d3518465869aae57b354886d0fbf974559ac3e7d38994baceb1eda5f75c76a9`;
- [matrix artifact](../experiments/analysis_matrix.npz), SHA-256
  `8144fbbd46076e1626ddf5d28c408de152ece267e23ef119b287347f1e3251ee`;
- [fold artifact](../experiments/folds_s10_f3_bs42.json), SHA-256
  `6363c9426ca55397548a8e4234bc22d4a22733758d0e06221140464acd0618e5`;
- [shared manifest](../experiments/shared_artifacts_manifest.json), which fixes
  accepted statuses, thresholds, ordering, shape, counts, and hashes.

## Parity matrix

| BenchPress layer | PathoPress evidence | Status | Qualification |
|---|---|---|---|
| Canonical score matrix and folds | [manifest](../experiments/shared_artifacts_manifest.json), matrix NPZ, folds JSON | Pathology-adapted | Protocol columns and permissive 3/5 support thresholds reflect domain sparsity |
| Point predictor | [`completion.py`](../src/pathopress/completion.py), [`verify_benchpress_parity.py`](../scripts/verify_benchpress_parity.py) | **Exact algorithmic parity**, parameterized by rank | Max difference ~`4.26e-14` rank 1, `0` rank 2; pathology selects rank 1 |
| Transform/method primitives | [`verify_method_comparison_parity.py`](../scripts/verify_method_comparison_parity.py) | **Exact algorithmic parity** | Seven transforms and core classical methods match reference fixtures to floating-point precision |
| Matched 10×3 fold CV | [CV result](../experiments/benchpress_style_results.json) | Exact split contract; pathology-adapted reporting | 19,670 predictions; rank-1 MAE/MedAE 3.005264/1.603026 |
| Raw/logit Soft-Impute rank sweep | [rank sweep](../experiments/soft_impute_rank_sweep_results.json) | Exact algorithm; pathology-adapted result | Both tracks select rank 1 |
| Broad method grid | [manifest](../experiments/method_comparison/manifest.json), [results](../experiments/method_comparison/results.json) | Exact core methods; expanded grid | 343/343 units versus upstream 329; partial-coverage winners require coverage caveat |
| Threshold and complete-submatrix SVD | [structure manifest](../experiments/structure_analysis/manifest.json), [tables](../outputs/tables/manifest.json) | Pathology-adapted | Largest complete block 32 × 16; stable rank 1.431046 |
| Correlation and MDS | [pairwise stats](../experiments/structure_analysis/pairwise_ols_stats.json), [coordinates](../experiments/structure_analysis/mds_coordinates.csv) | Pathology-adapted | All 165 columns have a neighbor at `min_shared=5` |
| All-known and held-out score probes | [selection](../experiments/probe_selection_results_rank1.json), [compression](../experiments/probe_compression_rank1.json) | Pathology-adapted, **bounded** | Any/proxy-feasible, MedAE/MedAPE, random, held-out, pruned, and ranking tracks; unrestricted curves capped at `k=10` |
| Low-cost probes | [feasibility data](../data/evaluation_feasibility.csv), [allowlist](../data/low_friction_allowlist_v1.json) | Pathology-adapted proxy | Four protocols selected before errors by metadata; cost is not measured |
| Exhaustive subsets | [merged result](../experiments/probe_exhaustive_rank1.json) | **Bounded exact search** | All subsets of four-item allowlist for `k=1..4`; all 66 `k=2` subsets of 12-item pruned set, not 165-column global search |
| Ranking preservation | [JSON](../experiments/ranking_preservation_rank1.json), [CSVs](../outputs/ranking_preservation_pairwise_rank1.csv) | Pathology-adapted | Same 0/1/2/5 margins and 10/20/30% fractions; normalized margins have endpoint-specific native meanings |
| Ranking-aware probes | [compression result](../experiments/probe_compression_rank1.json) | Pathology-adapted, bounded | Implemented within declared candidate/probe limits |
| Confidence calibration | [JSON](../experiments/confidence_calibration_rank1.json), [cells](../experiments/confidence_cells_rank1.csv) | Pathology-adapted | Cross-fitted risk/conformal pipeline; six full-coverage ALS/Soft-Impute variants are narrower than upstream top-12 diversity |
| Deployment intervals | [deployment artifact](../experiments/deployment_confidence_rank1.json) | Engineering analogue | Suite-calibrated for supported existing rows only; not new-model or external-site calibration |
| Temporal deployment | [JSON](../experiments/temporal_deployment_rank1.json), [raw CSV](../outputs/temporal_deployment_raw_rank1.csv) | Pathology-adapted | Hard prior-release rule; only seven retrospective 2025 targets |
| Per-column/model predictability | [JSON](../experiments/predictability_results_rank1.json), [raw CSV](../experiments/predictability_predictions_rank1.csv) | Pathology-adapted | 9,605 predictions, 165 evaluation and 53 model summaries |
| Benchmark/model error factors | [factor JSON](../experiments/prediction_error_factors_rank1.json), [manifest](../experiments/prediction_error_factor_manifest.json) | Pathology-adapted | Nine intervention groups; metadata-dependent denominators and correlational interpretation |
| Publication tables/figures | [table manifest](../outputs/tables/manifest.json), [gallery](../figures/README.md) | Engineering analogue | Deterministic result-first generation |
| Prediction CLI | [`cli.py`](../src/pathopress/cli.py) | Engineering analogue | List, lookup, completion, add-model, CSV/JSON, existing-row confidence |
| Public dataset export | [export README](../exports/pathopress_public/README.md), [manifest](../exports/pathopress_public/manifest.json) | Engineering analogue | All/paper/wide CSVs, sanitized provenance, licenses, hashes, loader/downloader; no upload |
| Static website | [site README](../website/README.md) | Engineering analogue | Browser-side exact rank-1 recipe, no server submission/analytics; not deployed here |
| Maintenance contract | [freshness manifest](../experiments/artifact_freshness_manifest.json), [dry run](../experiments/experiment_set_dry_run.json) | Engineering analogue | Hash/freshness and smoke checks exist; living score review remains human work |
| Matrix and five-shot LLM baselines | [config/status](../experiments/llm_baseline/real_run_status.json) | **Unrun** | Four provider-neutral conditions prepared; no provider client or real responses; mock output is contract-only |

## Exactness boundaries

The point predictor is exact only after supplying a shared normalized matrix and
rank. PathoPress's native metric mappings—such as `50 × (r + 1)` for HEST,
`100 × RI` for PathoROB, and `50 × (kappa + 1)`—are pathology adaptations.
They ensure valid direction/range handling but do not create one clinical unit.

Likewise, choosing rank 1 is evidence-based adaptation, not failure to copy the
upstream rank-2 default. Rank 1 wins matched pathology folds; suite-block
stress testing prefers rank 5, so no universal intrinsic-rank claim is made.

The method comparison's best MedAPE row, log BenchReg at 1.8144, covers only
71.0% of expected held-out cells. PathoPress retains full-coverage rank-1 ALS
for product predictions rather than selecting a partial method on error alone.

## Tractability and cache boundaries

Some upstream-style experiments are computationally combinatorial. PathoPress
makes the bounds explicit:

- unrestricted greedy probe selection stops at ten probes;
- exact all-known enumeration covers 15 nonempty subsets of the four-item
  pre-error allowlist;
- the separate pruned enumeration covers all 66 two-probe subsets of 12
  error-informed candidates;
- neither result implies global exhaustive search across 165 columns.

Large prediction-first caches are reproducibility intermediates, not compact
release assets. `experiments/method_comparison/predictions/` contains 343 NPZ
files totaling about 439 MiB. The factor experiment has 5,795 unit-cache files
around 21 MiB and a 279,487-row raw prediction CSV around 29 MiB. These paths
are narrowly Git-ignored; their compact manifests, merged results, tables, and
figures are tracked. “Ignored” does not mean missing from the completed local
run.

## Trust boundaries

Cross-fitted structural-support uncertainty has Spearman 0.598017 with absolute
error. Nominal 90% leave-fold-out intervals cover 89.9949% of 19,670 OOF
instances with median width 9.7677 normalized points. Retaining the lowest-risk
20% reduces MedAE to 0.606970.

This is retrospective selective prediction. The confidence ensemble is less
diverse than BenchPress's upstream stack, and the deployment intervals are not
validated for a new model row. The CLI/site explicitly return no interval for
new-model completion.

Temporal results enforce that every training model predates its target, but
target eligibility yields only seven 2025 models. It is evidence against simple
time leakage, not prospective validation. Factor analyses have incomplete
metadata—for example parameter count for 39/53 supported models and
provider/family for 52/53—and should be read as associations or controlled
retrospective interventions, not causal claims.

## Genuinely incomplete items

1. **Real LLM comparison:** all named/blind matrix and five-shot real-provider
   conditions are `unrun`. Deterministic mock metrics are not headline eligible.
2. **Measured evaluation cost:** the current allowlist is a pre-error metadata
   feasibility proxy, not runtime/compute/access/tissue/license measurement.
3. **Prospective and external validation:** no preregistered future model or
   external institution cohort has been evaluated.
4. **New-model confidence:** existing-row residual calibration is not a valid
   interval guarantee for a genuinely unseen model.
5. **Dual human review/living maintenance:** accepted rows are parsed from
   primary sources but have not all received two independent human reviews.
6. **Hosted deployment/upload:** the export and site are local static assets;
   no Hugging Face upload, GitHub Pages publication, or other deployment occurs
   from the build scripts.
7. **Globally exhaustive probe search:** intentionally not attempted because
   the 165-column subset space is intractable.

## Reproduction and verification

Install the research extra, which includes PyTorch for the MLP method-grid
units.

```bash
python3 -m pip install -e '.[research]'
PYTHONPATH=src python3 scripts/build_shared_artifacts.py
PYTHONPATH=src python3 scripts/check_artifact_freshness.py check
PYTHONPATH=src python3 -m unittest discover -s tests -v
node --check website/app.js
git diff --check
```

For direct upstream verification:

```bash
PYTHONPATH=src python3 scripts/verify_benchpress_parity.py \
  /path/to/benchpress --rank 1
PYTHONPATH=src python3 scripts/verify_benchpress_parity.py \
  /path/to/benchpress --rank 2
PYTHONPATH=src python3 scripts/verify_method_comparison_parity.py \
  /path/to/benchpress
```

The concise protocol explanation is in
[benchpress-parity.md](benchpress-parity.md); the runnable artifact index is
[experiments/README.md](../experiments/README.md).
