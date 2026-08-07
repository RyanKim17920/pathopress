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
`1.07e-13` at rank 1 and `0` at rank 2. The seven transforms and the classical
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
BenchPress deploys rank 2. PathoPress also maps native
pathology metrics to a direction-preserving 0–100 fitting scale; those mappings
are domain adaptations, not upstream semantics.

## Shared experimental substrate

The fixed paper matrix has 59 models, 187 protocol-level evaluations, 2,122
observations, and 19.2332% density. It draws from Patho-Bench (122 retained
evaluations), EVA (15), THUNDER (16), H-Optimus-1 report (10), HEST (18), and
PathoROB (6). All experiments
can consume the same ordered matrix and ten-seed, three-fold split contract:

- [shared manifest](../experiments/shared_artifacts_manifest.json)
- [matrix NPZ](../experiments/analysis_matrix.npz)
- [fold assignments](../experiments/folds_s10_f3_bs42.json)

The matched rank-1 fold result is 3.134532 MAE and 1.609006 MedAE over 21,181
supported predictions; the column-median baseline is 4.151756/2.400000. Random-cell,
suite-block, and sparse-new-model tests are additional pathology stress tests,
not substitutes for the matched upstream fold protocol. Suite-block rank 1 is
5.788534/3.599615 and improves through tested rank 6 at 5.099363/3.203260,
demonstrating that the
selected rank and error depend on the deployment question.

## What benchmark “compression” means

BenchPress does not compress benchmark images or examples. It asks whether a
small globally selected set of evaluation columns can reconstruct published
model scorecards. With an all-known probe set, every observed score is hidden
except the selected probes, the matrix is completed, and revealed probe cells
remain exact in the denominator. PathoPress additionally reports hidden-only
error, which excludes those zero-error revealed cells.

On the current matrix, the zero-probe scorecard baseline is 1.900 MedAE.
Greedy all-known rank-1 completion reaches 1.397334 at five probes and 1.213706
at ten; hidden-only values are 1.548536 and 1.493709. The held-out-row protocol
selects probes on training models and evaluates each validation model in
isolation; its hidden-cell MedAE is 1.885364 and 2.008051 at five and ten.

The compression runner now implements:

- any-evaluation and 25-evaluation pre-error pipeline-proxy tracks;
- MedAE and MedAPE objectives;
- nested random orders;
- held-out-model and margin-5 ranking-aware objectives through `k=10`; and
- all-ten-step aggregate-rank pruning to an error-informed 30 candidates.

Metric names require one important scope distinction. The `pairwise_margin=2`
values nested inside ordinary `curves` are ancillary ordering diagnostics
computed alongside score reconstruction; they do not select those MedAE/MedAPE
probe sets and are not the ranking result. The dedicated `ranking_aware`
trajectories are selected and evaluated with the pinned margin of 5 normalized
points. The artifact records both margins and this distinction explicitly.

The [pathology hero](../figures/pathopress_benchpress_hero_rank1.png) shows the
retrospective all-known scorecard trajectory. Revealed probes are exact, and
selection/evaluation use the same model population, so it is not model-level
holdout. The machine-readable [ranking result](../experiments/ranking_preservation_rank1.json)
preserves the random and greedy margin-5 trajectories without a separate public
chart. At `k=10`, unrestricted all-known pairwise accuracy is 0.8780, versus
0.4000 for the 25-task feasibility proxy. The
[task-utility and held-out-mean figure](../figures/probe_dual_objective_rank1.png)
separates transductive single-task utility from mean-score prediction on 18
held-out models using prefixes selected on 41 training models. Revealed probe
values are exact, supported hidden cells are predicted, and no held-out `k=0`
or random model-mean control is available.

The v2 pre-error allowlist contains the 25 retained image/patch classification
evaluations. This exactly matches the pinned upstream candidate count, but not
its task identities or cost semantics. It is only a low-friction input/label
pipeline proxy; several datasets are large, and it does not measure runtime,
compute, tissue access, annotation labor, or licensing cost.

No exhaustive choose-five result is checked in for the current 59×187 matrix.
The retained schema-v2 runner and validators can produce a new hash-bound result;
historical outputs from earlier matrix snapshots are intentionally excluded.

## Other parity layers

| Layer | Evidence | Classification |
|---|---|---|
| Raw/logit rank sweep | [Soft-Impute results](../experiments/soft_impute_rank_sweep_results.json) | Exact algorithm; pathology data/rank choice |
| Classical method grid | [343-shard manifest](../experiments/method_comparison/manifest.json), [results](../experiments/method_comparison/results.json) | Exact core algorithms; expanded pathology grid |
| Score-probe search | [compression results](../experiments/probe_compression_rank1.json) | Adapted objectives and bounded search |
| Ranking preservation | [ranking results](../experiments/ranking_preservation_rank1.json) | Current compression-derived margin-5 all-known/random and held-out trajectories through `k=10` |
| Confidence calibration | [OOF calibration](../experiments/confidence_calibration_rank1.json), [method](confidence-trust.md) | Exact upstream experiment contract; pathology rank/data adaptation |
| Temporal deployment | [temporal results](../experiments/temporal_deployment_rank1.json) | Adapted, retrospective seven-model cohort |
| Product/export | [CLI](../src/pathopress/cli.py), [export](../exports/pathopress_public/README.md), [site](../website/README.md) | Local engineering analogue; no hosted deployment |

The method grid completed 343/343 configurations: 12 method families over
seven transforms, including pathology rank-sensitivity additions beyond the
upstream 329-shard grid. Its top MedAPE configuration is logit BenchReg at 1.9023
with only 68.3% prediction coverage. It must not displace the full-coverage
rank-1 ALS product predictor on that aggregate alone.

Confidence uses cross-fitted residuals. Its generator contract now matches
upstream: three same-family Logit Bias-ALS lambda variants plus the top twelve
full-coverage Section-4 alternatives, with strict cache identity checks. The
eight structural features, three learned risks, nested ridge/MLP selection,
leave-fold conformal diagnostics, and five retained confidence methods follow
the pinned code. Pathology rank 1 and pathology's own top-twelve roster are the
documented data adaptations. Calibrated trust means P(|error| <= 10 normalized
points); unsupported or unseen-model populations abstain.

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
Metric mappings and error semantics are in [imputation.md](imputation.md).
