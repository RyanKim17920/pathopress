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
BenchPress's deployed LLM predictor uses rank 2. PathoPress also maps native
pathology metrics to a direction-preserving 0–100 fitting scale; those mappings
are domain adaptations, not upstream semantics.

## Shared experimental substrate

The fixed paper matrix has 59 models, 168 protocol-level evaluations, 2,027
observations, and 20.4500% density. It draws from Patho-Bench (122 retained
evaluations), EVA (15), THUNDER (16), HEST (9), and PathoROB (6). All experiments
can consume the same ordered matrix and ten-seed, three-fold split contract:

- [shared manifest](../experiments/shared_artifacts_manifest.json)
- [matrix NPZ](../experiments/analysis_matrix.npz)
- [fold assignments](../experiments/folds_s10_f3_bs42.json)

The matched rank-1 fold result is 3.222008 MAE and 1.647585 MedAE over 20,270
predictions; the column-median baseline is 4.275274/2.500000. Random-cell,
suite-block, and sparse-new-model tests are additional pathology stress tests,
not substitutes for the matched upstream fold protocol. Suite-block rank 1 is
5.688229/3.537207 and improves through tested rank 6, demonstrating that the
selected rank and error depend on the deployment question.

## What benchmark “compression” means

BenchPress does not compress benchmark images or examples. It asks whether a
small globally selected set of evaluation columns can reconstruct published
model scorecards. With an all-known probe set, every observed score is hidden
except the selected probes, the matrix is completed, and revealed probe cells
remain exact in the denominator. PathoPress additionally reports hidden-only
error, which excludes those zero-error revealed cells.

On the current matrix, the zero-probe scorecard baseline is 1.935 MedAE.
Greedy all-known rank-1 completion reaches 1.474879 at five probes and 1.270529
at ten; hidden-only values are 1.637639 and 1.538607. The held-out-row protocol
selects probes on training models and evaluates each validation model in
isolation; its hidden-cell MedAE is 2.126261 and 2.142613 at five and ten.

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

The upstream-shaped [pathology hero](../figures/pathopress_benchpress_hero_rank1.png)
reconstructs the four target examples and overall score-prediction panel; the
[ranking panel](../figures/pathopress_benchpress_ranking_rank1.png) shows the
random and greedy margin-5 trajectories. At `k=10`, unrestricted all-known
pairwise accuracy is 0.8889, versus 0.3333 for the 25-task feasibility proxy.
The [dual-objective table](../outputs/probe_dual_objective_rank1.csv) also asks
how well those scorecard-selected probes predict each model's average observed
score. It reports that quantity separately and does not imply it was optimized.

The v2 pre-error allowlist contains the 25 retained image/patch classification
evaluations. This exactly matches the pinned upstream candidate count, but not
its task identities or cost semantics. It is only a low-friction input/label
pipeline proxy; several datasets are large, and it does not measure runtime,
compute, tissue access, annotation labor, or licensing cost.

The [exhaustive execution status](../experiments/probe_exhaustive_execution_status.json)
binds the upstream-equivalent plans: `C(25,5)=53,130` for the pre-error proxy and
`C(30,5)=142,506` after error-informed pruning. The operational runner matches
the wave/shard residue, gzip chunk, raw-prediction, validated-resume, and strict
merge contracts. Both searches are complete and scalar-certified: the five-probe
MedAE optima are 1.485944 in the 25-task proxy and 1.427339 in the error-informed
30-task universe. These are exact within those candidate sets, not globally over
all 168 evaluations, and they optimize MedAE rather than the separate ranking
objective. Legacy-v1 chunk configs did not bind the generator binary; the audit
therefore establishes numerical backend compatibility, not generator attribution.

## Other parity layers

| Layer | Evidence | Classification |
|---|---|---|
| Raw/logit rank sweep | [Soft-Impute results](../experiments/soft_impute_rank_sweep_results.json) | Exact algorithm; pathology data/rank choice |
| Classical method grid | [343-shard manifest](../experiments/method_comparison/manifest.json), [results](../experiments/method_comparison/results.json) | Exact core algorithms; expanded pathology grid |
| Complete-submatrix rank, thresholds, correlations, MDS | [structure manifest](../experiments/structure_analysis/manifest.json) | Adapted to protocol columns |
| Score-probe search | [compression results](../experiments/probe_compression_rank1.json) | Adapted objectives and bounded search |
| Ranking preservation | [ranking results](../experiments/ranking_preservation_rank1.json) | Matched margins/fractions; normalized-score interpretation |
| Confidence calibration | [OOF calibration](../experiments/confidence_calibration_rank1.json), [method](confidence-trust.md) | Exact upstream experiment contract; pathology rank/data adaptation |
| Temporal deployment | [temporal results](../experiments/temporal_deployment_rank1.json) | Adapted, retrospective seven-model cohort |
| Error factors | [factor results](../experiments/prediction_error_factors_rank1.json) | Adapted pathology metadata and intervention groups |
| Product/export | [CLI](../src/pathopress/cli.py), [export](../exports/pathopress_public/README.md), [site](../website/README.md) | Local engineering analogue; no hosted deployment |
| LLM baselines | [protocol](llm-baseline.md), [real-run status](../experiments/llm_baseline/real_run_status.json) | Complete 1,990-request/81,080-target offline pack; real provider run unrun |

The method grid completed 343/343 configurations: 12 method families over
seven transforms, including pathology rank-sensitivity additions beyond the
upstream 329-shard grid. Its top MedAPE configuration is logit BenchReg at 1.9077
with only 71.9% prediction coverage. It must not displace the full-coverage
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
- Do not use deterministic mock LLM metrics as scientific evidence. The four
  named/blind zero-shot matrix and five-shot real-provider conditions are still unrun; all 30-fold requests are prepared.

The comprehensive evidence/status matrix is in
[full-parity-audit.md](full-parity-audit.md); metric mappings and error semantics
are in [imputation.md](imputation.md).
