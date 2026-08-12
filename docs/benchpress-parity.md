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
isolation; its hidden-cell MedAE is **3.137815** at five probes (n = 321 hidden
cells) and **3.058947** at ten (n = 307), read from
[`probe_selection_results_rank1.json`](../experiments/probe_selection_results_rank1.json)
under `heldout_model.validation[step].hidden_only.medae`.

Three disclosures travel with that pair. First, it comes from a **single 70/30
`split_mode = family_blocked` holdout — 48 training models and 11 validation
models** — not from the 34-fold LOFO protocol used elsewhere in this note, and a
single split at N = 11 validation models is seed-sensitive. Second, the hidden
cell count shrinks with k (334 at k=0 down to 307 at k=10), so the curve is not
scored on a fixed denominator. Third, and most importantly, **the sign of the
story is the opposite of what was previously published here**: held-out error
does not fall below the 1.900 zero-probe scorecard baseline — it is roughly 65%
worse than it, and it is also worse than this protocol's own k=0 hidden-cell
baseline of 2.933142 (n = 334). Revealing probes on a held-out model row does not
improve reconstruction of that row under this split. The earlier figures
(1.885364 at five, 2.008051 at ten) are withdrawn; they trace to no current
artifact.

Arm-to-arm comparison needs a matched denominator. Each arm removes the cells it
reveals, so a greedy arm, a random arm, and a k=0 arm are otherwise scored on
three different cell sets. The matched-cell leave-one-family-out replay excludes,
per fold and per depth, the union of the cells revealed by the greedy prefix and
by all ten random repeats, then scores every arm on the identical remainder. At
four probes that excludes 486 of 2,122 cells and leaves 1,636 matched: greedy
1.8781, k=0 2.6524 (both medians of the 34 fold medians), random 2.6013 under
convention A (median over all 340 fold × repeat MedAEs) or 2.6260 under
convention B (median of the 34 per-fold medians); the convention must be named
whenever the random arm is quoted. Greedy beats k=0 in 18 of 34 folds
(Wilcoxon p = 0.0088) and random in 22 of 34 (p = 0.0151); the reduction versus
k=0 has a bootstrap-over-folds 95% CI of [2.8%, 58.7%] and is therefore not
estimable to three significant figures. Note the deliberate mismatch: the
22-of-34 test and the [3.4%, 53.5%] CI against random are computed on
**convention B** fold medians (2.6260), because a paired test needs one random
value per fold, whereas the 2.6013 figure usually quoted alongside them is
**convention A**. Table value and test statistic use different aggregations, and
that is stated rather than left implicit. Two scope notes travel with this: the
greedy selector optimizes `parity.median_absolute_error`, which scores revealed
probe cells as literal 0.0 and is about 6.4% optimistic at four probes (1.5026
with the 85 revealed cells counted at zero error, against 1.6055 on the 2,037
hidden cells of the same completion); and per-evaluation utility on this protocol
is null,
at 86 of 174 scored columns (49.4%, 95% CI [42.0%, 56.9%]). Reproduce with
`scripts/replay_lofo_matched_cells.py`, artifact
[matched-cell LOFO results](../experiments/lofo_matched_cells_rank1.json).

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
preserves the random and greedy trajectories across all tested margins without a
separate public chart. At `k=10` and margin 0 (all 17,159 pairs, all 187
columns), unrestricted all-known greedy accuracy is 0.679 versus 0.552 random.
At the margin-5 threshold — which retains 6,048 of 17,159 pairs (35%) across
148 of 187 columns, because the median score gap between model pairs is only
3.10 normalized points — the figures are 0.878 greedy versus 0.603 random.
Greedy exceeds random at every tested margin, both absolute (0–10 points) and
relative (0.25–1.0 × column SD or IQR); relative margins retain all 187
columns and are the stronger evidence. The 25-task proxy reaches 0.4000 at
`k=10`. The
[task-utility and held-out-mean figure](../figures/probe_dual_objective_rank1.png)
separates transductive single-task utility from mean-score prediction on
held-out models under a leave-one-family-out protocol with 34 family folds;
all 59 models are held out exactly once, with 1 validation model per fold at
the median (min 1, max 7) and 58 training models. Revealed probe
values are exact, supported hidden cells are predicted, and no held-out `k=0`
or random model-mean control is available.

### Pairwise accuracy margin sweep

The table below records greedy (k=10, rank 1, any\_candidate, all-known track)
and random-baseline (10 repeats) pairwise accuracy across all tested margins.
The pair count and column count show how many comparisons survive each
threshold. Relative margins are expressed as multiples of the per-column SD or
IQR and always retain all 187 columns; they are therefore the stronger evidence
for the robustness of the greedy-beats-random finding.

| Margin type | Margin | Pairs | Cols | Greedy median | Greedy pooled | Random median | Random pooled |
|---|---|---:|---:|---:|---:|---:|---:|
| absolute | 0.0 | 17,159 | 187 | 0.679 | 0.771 | 0.552 | 0.607 |
| absolute | 1.0 | 13,763 | 187 | 0.708 | 0.811 | 0.567 | 0.623 |
| absolute | 2.0 | 10,973 | 178 | 0.746 | 0.836 | 0.571 | 0.633 |
| absolute | 3.0 | 8,845 | 168 | 0.785 | 0.849 | 0.585 | 0.638 |
| absolute | 5.0 | 6,048 | 148 | 0.878 | 0.861 | 0.603 | 0.647 |
| absolute | 10.0 | 2,722 | 100 | 1.000 | 0.883 | 0.663 | 0.657 |
| relative | 0.25 × SD | 14,397 | 187 | 0.696 | 0.804 | 0.559 | 0.619 |
| relative | 0.25 × IQR | 13,657 | 187 | 0.696 | 0.810 | 0.562 | 0.622 |
| relative | 0.50 × SD | 11,957 | 187 | 0.712 | 0.829 | 0.567 | 0.629 |
| relative | 0.50 × IQR | 10,751 | 187 | 0.712 | 0.838 | 0.575 | 0.635 |
| relative | 1.00 × SD | 8,232 | 187 | 0.750 | 0.862 | 0.567 | 0.642 |
| relative | 1.00 × IQR | 6,076 | 187 | 0.778 | 0.863 | 0.592 | 0.655 |

Greedy exceeds the random baseline at every row. The median pathology score gap
between model pairs is 3.10 normalized points; 64.8% of pairs differ by less
than 5 points. The margin-5 row (0.878 greedy, 0.603 random) therefore
represents the most favorable point on the absolute curve, not a typical
operating point. The margin-0 row (0.679 greedy, 0.552 random) is the
unconditional figure.

The v2 pre-error allowlist contains the 25 retained image/patch classification
evaluations. This exactly matches the pinned upstream candidate count, but not
its task identities or cost semantics. It is only a low-friction input/label
pipeline proxy; several datasets are large, and it does not measure runtime,
compute, tissue access, annotation labor, or licensing cost.

The allowlist arm also carries a negative selection result. On its own matched
cell set at four probes (1,581 cells), allowlist greedy reaches 1.9951 against
1.7234 for greedy over any candidate, so restricting candidates to the
feasibility pool costs accuracy.

Within that pool there is also no demonstrable selection signal. Against a
genuine allowlist-restricted random control, scored on the cell set the two
allowlist arms share at four probes (1,237 matched cells over 34 LOFO folds),
allowlist greedy is 1.9463 and allowlist random is 2.0251 under
median-of-fold-medians (2.0118 under median over all 340 fold × repeat MedAEs);
the k=0 arm on those cells is 2.8555. Greedy leads nominally but wins only
10 of 34 folds, ties 15 and loses 9, with Wilcoxon p = 0.4939 and a
bootstrap-over-folds 95% CI on the reduction of [-11.5%, +20.7%]. The result is
the same at every tested depth k=1..5, where p ranges from 0.05 to 0.49 and
every CI straddles zero.

The accompanying disclosure is not optional: the test is underpowered by design.
The 15 ties are exactly the 15 folds where the allowlist arms are zero-information —
the held-out family has no observed score on any of the 25 allowlist evaluations, so
both arms produce literally identical predictions (not merely "too close to call") and
reduce to the k=0 arm. This identity holds at every k=1..5. Restricting to the 19
informative folds alone (10 wins vs 9 losses) is still non-significant, so the null
is not an artifact of tie-handling. Only 19 folds carry signal. Read this as "no advantage demonstrated in a pool too sparse to test it",
not as evidence that greedy is worse than random; the unrestricted 187-candidate
arm on the same protocol beats random in 22 of 34 folds at p = 0.0151. The
previously published pair (allowlist greedy 2.0404, allowlist random 2.0109) is
withdrawn: those values occur in no artifact and the comparison's sign was
inverted.

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
| Matched-cell LOFO comparison | [matched-cell results](../experiments/lofo_matched_cells_rank1.json), replay `scripts/replay_lofo_matched_cells.py` | Pathology-specific correction; no upstream analogue |
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
