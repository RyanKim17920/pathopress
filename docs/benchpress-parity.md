# BenchPress parity in PathoPress

This note maps PathoPress to Microsoft BenchPress at pinned commit
[`0a684b63ee0e4a401cb907a3827a82ea997d74c4`](https://github.com/microsoft/benchpress/tree/0a684b63ee0e4a401cb907a3827a82ea997d74c4).
“Parity” has three different meanings here: an exact numerical algorithm, an
adapted pathology experiment, or an engineering analogue with different data.

## What is exact

The point completer uses BenchPress's logit transform, column standardization,
global/model/evaluation biases, ridge `0.1`, 40 alternating-least-squares
iterations, and ten starts with seeds 42–51. Direct comparison against the
pinned checkout gives maximum absolute differences of approximately
`4.26e-14` at rank 1 and `0` at rank 2. The seven transforms and the classical
mean, KNN, regression, Soft-Impute, bias-ALS, NMF, PMF, and nuclear-norm
implementations also agree with the reference verification fixture to
floating-point precision.

```bash
PYTHONPATH=src python3 scripts/verify_benchpress_parity.py \
  /path/to/benchpress --rank 1
PYTHONPATH=src python3 scripts/verify_benchpress_parity.py \
  /path/to/benchpress --rank 2
PYTHONPATH=src python3 scripts/verify_method_comparison_parity.py \
  /path/to/benchpress
```

The algorithm is exact *when given the same normalized matrix and rank*.
PathoPress selects interaction rank 1 from pathology cross-validation, while
BenchPress's deployed LLM predictor uses rank 2. PathoPress also maps native
pathology metrics to a direction-preserving 0–100 fitting scale; those mappings
are domain adaptations, not upstream semantics.

## Shared experimental substrate

The fixed paper matrix has 59 models, 165 protocol-level evaluations, 1,967
observations, and 20.2054% density. It draws from Patho-Bench (122 retained
evaluations), EVA (15), THUNDER (16), HEST (9), and PathoROB (3). All experiments
can consume the same ordered matrix and ten-seed, three-fold split contract:

- [shared manifest](../experiments/shared_artifacts_manifest.json)
- [matrix NPZ](../experiments/analysis_matrix.npz)
- [fold assignments](../experiments/folds_s10_f3_bs42.json)

The matched rank-1 fold result is 3.005264 MAE and 1.603026 MedAE over 19,670
predictions; the column-median baseline is 4.092133/2.477500. Random-cell,
suite-block, and sparse-new-model tests are additional pathology stress tests,
not substitutes for the matched upstream fold protocol. Suite-block rank 1 is
5.612789/3.525174 and prefers rank 5 overall, demonstrating that the selected
rank and error depend on the deployment question.

## What benchmark “compression” means

BenchPress does not compress benchmark images or examples. It asks whether a
small globally selected set of evaluation columns can reconstruct published
model scorecards. With an all-known probe set, every observed score is hidden
except the selected probes, the matrix is completed, and revealed probe cells
remain exact in the denominator. PathoPress additionally reports hidden-only
error, which excludes those zero-error revealed cells.

On the current matrix, the zero-probe scorecard baseline is 1.900 MedAE.
Greedy all-known rank-1 completion reaches 1.481124 at five probes and 1.196456
at ten; hidden-only values are 1.612112 and 1.539134. The held-out-row protocol
selects probes on training models and evaluates each validation model in
isolation; its hidden-cell MedAE is 1.951271 and 1.879857 at five and ten.

The expanded [compression artifact](../experiments/probe_compression_rank1.json)
adds:

- any-evaluation and four-evaluation pre-error feasibility tracks;
- MedAE and MedAPE objectives;
- nested random orders;
- held-out-model and ranking-aware objectives; and
- a separately labeled error-informed pruned search.

The pre-error allowlist contains only THUNDER BACH, BRACS, BreakHis, and MHIST.
It is generated from protocol metadata—image/patch classification with reported
sample count at most 10,000—and is only a low-friction proxy. It does not measure
runtime, compute, tissue access, label burden, or licensing cost.

Exact subset enumeration is deliberately bounded. The
[exhaustive artifact](../experiments/probe_exhaustive_rank1.json) covers every
subset of the four-item pre-error allowlist for `k=1..4` and all 66 `k=2`
subsets of a 12-candidate error-informed pruned set. It is exact within those
declared spaces, not exhaustive across all 165 evaluations. Unrestricted greedy
curves stop at ten because each candidate evaluation requires completing every
target row.

## Other parity layers

| Layer | Evidence | Classification |
|---|---|---|
| Raw/logit rank sweep | [Soft-Impute results](../experiments/soft_impute_rank_sweep_results.json) | Exact algorithm; pathology data/rank choice |
| Classical method grid | [343-shard manifest](../experiments/method_comparison/manifest.json), [results](../experiments/method_comparison/results.json) | Exact core algorithms; expanded pathology grid |
| Complete-submatrix rank, thresholds, correlations, MDS | [structure manifest](../experiments/structure_analysis/manifest.json) | Adapted to protocol columns |
| Score-probe search | [compression results](../experiments/probe_compression_rank1.json) | Adapted objectives and bounded search |
| Ranking preservation | [ranking results](../experiments/ranking_preservation_rank1.json) | Matched margins/fractions; normalized-score interpretation |
| Confidence calibration | [OOF calibration](../experiments/confidence_calibration_rank1.json) | Adapted six-predictor stack; narrower diversity than upstream top-12 |
| Temporal deployment | [temporal results](../experiments/temporal_deployment_rank1.json) | Adapted, retrospective seven-model cohort |
| Error factors | [factor results](../experiments/prediction_error_factors_rank1.json) | Adapted pathology metadata and intervention groups |
| Product/export | [CLI](../src/pathopress/cli.py), [export](../exports/pathopress_public/README.md), [site](../website/README.md) | Local engineering analogue; no hosted deployment |
| LLM baselines | [real-run status](../experiments/llm_baseline/real_run_status.json) | Request/cache contract only; real provider runs unrun |

The method grid completed 343/343 configurations: 12 method families over
seven transforms, including pathology rank-sensitivity additions beyond the
upstream 329-shard grid. Its top MedAPE configuration is log BenchReg at 1.8144
with only 71.0% prediction coverage. It must not displace the full-coverage
rank-1 ALS product predictor on that aggregate alone.

Confidence uses cross-fitted residuals. Structural support obtains Spearman
0.5980 with absolute error and 89.995% empirical coverage for nominal 90%
leave-fold-out intervals. The predictor stack contains six full-coverage
ALS/Soft-Impute variants rather than upstream's more diverse top-12 stack, so
this is a pathology adaptation. The deployment interval artifact is calibrated
for held-out cells in supported existing model rows; it is not a guarantee for
new models, new sites, or external cohorts.

Temporal deployment selects seven verified 2025 targets using a date window
and observed-count rule fixed before target errors, then trains only on earlier
models and reveals 1/5/10 target scores over ten seeds. This enforces the hard
time rule but remains a small retrospective study.

## Interpretation rules

- Deduplicate dataset/task identities for catalog navigation; never collapse
  protocol columns merely because they share a dataset name.
- Report coverage beside completion error. Sparse suite blocks can make a probe
  or method look useful only for a subset of cells.
- Treat normalized points as fitting/display units, not interchangeable clinical
  utility across AUC, kappa, Pearson correlation, Dice, F1, or robustness.
- Label reported, provided, and predicted cells distinctly.
- Do not describe the feasibility proxy as measured cost or the bounded subset
  search as globally exhaustive.
- Do not use deterministic mock LLM metrics as scientific evidence. The four
  named/blind matrix and five-shot real-provider conditions are still unrun.

The comprehensive evidence/status matrix is in
[full-parity-audit.md](full-parity-audit.md); metric mappings and error semantics
are in [imputation.md](imputation.md).
