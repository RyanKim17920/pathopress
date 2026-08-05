# Imputation and error metrics

## What MAE and MedAE mean

Validation starts with a published cell whose normalized score is known. The
cell is hidden, the remaining matrix is completed, and its prediction is
compared with the hidden score:

```text
absolute error = |imputed normalized score - reported normalized score|
MAE            = arithmetic mean of all absolute errors
MedAE          = median of all absolute errors
```

MAE gives extra weight to occasional large misses. MedAE describes a typical
prediction instance: a MedAE of 1.083 means half of evaluated imputations were
within 1.083 normalized points and half were farther away. Neither value is a
confidence interval, clinical error, or universal accuracy percentage.

The common display scale is metric-specific:

| Suite | Source metric | Normalized scale | Meaning of one point |
| --- | --- | --- | --- |
| THUNDER | F1 reported on 0–100 | unchanged | one F1 point |
| HEST | Pearson correlation `r` | `50 × (r + 1)` | `0.02 r` |
| PathoROB | Robustness Index | `100 × RI` | `0.01 RI` |

Pooling these units is useful for fitting a shared matrix, but it does not make
them clinically interchangeable. Suite-specific errors must accompany pooled
errors.

## BenchPress-style validation

The reproduction assigns every observed score within each model to one of
three folds, repeats that assignment for seeds 42 through 51, and predicts the
held-out cells from the remaining scores. Every source cell is therefore tested
once per seed, giving 8,060 rank-1 prediction instances from 806 distinct cells.

For the selected rank 1:

| Aggregation | Result |
| --- | ---: |
| Pooled MAE | 2.427 |
| Pooled MedAE | 1.038 |
| Median of 30 fold MedAEs | 1.048 |
| Within 1 point | 48.5% |
| Within 3 points | 81.7% |
| Within 5 points | 88.8% |
| Within 10 points | 95.4% |

The matching task-column-median baseline has MAE 4.470 and MedAE 2.400, versus
rank 1's 2.427 and 1.038. This is evidence that the model is using cross-task
structure, although it is not yet a prospective or family-held-out test.

This matches BenchPress's fold construction and reports both pooled and
fold-median aggregation. It does not make the numeric error directly comparable
to BenchPress's language-model matrix because the tasks, score mappings, and
missingness differ.

## Figures

- [Observed/missing matrix](../figures/matrix_observation_pattern.png)
- [Reported versus rank-1 completed matrix](../figures/matrix_completed_rank1.png)
- [Rank sweep, parity plot, suite errors, and error distribution](../figures/benchpress_style_validation_rank1.png)

The parity plot uses only out-of-fold predictions. In the completed heatmap,
reported cells are solid and point estimates are translucent.

## Imputation artifact

[`outputs/imputations_rank1.csv`](../outputs/imputations_rank1.csv) contains one
row for every supported model/evaluation pair. `status=observed` means the score
was reported and is preserved exactly. `status=imputed` marks one of the 510
rank-1 point estimates. The file includes source-metric estimates, row/column
support counts, a rank-2 comparison, and the absolute rank-1/rank-2 difference.

The rank difference is a sensitivity diagnostic, not an uncertainty interval.
The ten internal ALS starts stabilize numerical optimization; they are not ten
independent samples. A public prediction service should add calibrated
uncertainty and abstain for weakly supported or out-of-family cells.

The remaining catalog-only tasks cannot be completed: an entirely unobserved
column has no learned task bias or latent factor. Patho-Bench and eva result
tables must be extracted before their columns become imputable.

## Is the pathology matrix rank 1?

For the final bias-ALS predictor, **one latent interaction factor is preferred**.
On identical matched folds, rank 1 has MAE/MedAE
2.427/1.038, versus 2.553/1.101 for bias-only rank 0, 2.541/1.083 for rank 2,
and worse errors for every tested rank through 10.

That is not a claim that the raw data matrix has exact algebraic rank 1. The
model is

```text
global mean + model bias + evaluation bias + rank-r interaction
```

so an interaction rank of 1 can have algebraic rank up to roughly 3 in the
transformed score space. Inverse logit and per-column inverse transforms can
raise the algebraic rank again. With incomplete, noisy, heterogeneous data
there is no uniquely observable “true rank”; cross-validation only says that
additional latent interaction dimensions overfit this seed matrix. Individual
suites may still have different effective ranks, and rank should be selected
again as Patho-Bench and eva score columns are added. In the separate
leave-one-suite-block-out experiment, rank 2 beats rank 1 by only 0.048 MAE;
that small reversal is another reason to retain rank sensitivity rather than
declare a universal intrinsic rank.

## How exact is the BenchPress reproduction?

For the final logit + z-score + bias-ALS predictor, it is exact on this input
matrix. Running `scripts/verify_benchpress_parity.py` against the pinned
BenchPress checkout compares PathoPress with BenchPress's standalone browser
predictor and currently returns a maximum absolute difference of `0.0` at rank
2. The shared details are ridge `0.1`, 40 ALS iterations, ten initializations
with seeds 42–51, and preservation of observed cells.

The pathology data contract is necessarily adapted: HEST Pearson correlation
is mapped with `50 × (r + 1)`, PathoROB RI with `100 × RI`, and THUNDER F1 is
already on 0–100. All resulting columns satisfy BenchPress's percentage-column
heuristic, so its logit transform is applied exactly. For HEST, that logit is a
scaled Fisher transform after column standardization.

BenchPress's *rank-discovery figure* is a different algorithm from its final
predictor: iterative truncated-SVD Soft-Impute in raw and logit spaces. The
matching pathology reproduction is in
[`soft_impute_rank_sweep_results.json`](../experiments/soft_impute_rank_sweep_results.json)
and [its graph](../figures/soft_impute_rank_sweep.png). By BenchPress's plotted
MedAPE criterion, rank 2 narrowly wins in both spaces. By pooled MAE, pooled
MedAE in logit space, and bias-ALS cross-validation, rank 1 wins. The difference
between ranks 1 and 2 is small enough that the responsible conclusion is
“effective interaction rank around 1–2,” not an exact intrinsic rank.

## Probe-policy figures

![BenchPress-style scorecard and literal-average curves](../figures/probe_selection_rank1.png)

The left panel follows the upstream hero protocol: the magenta greedy curve is
selected and evaluated on the all-known matrix, and exact probe cells remain in
the denominator. The gray curve is the median of 10 nested global random probe
orders with an interquartile band. The blue curve selects probes using 70% of
model rows, then evaluates hidden non-probe cells on each held-out model in
isolation. It is deliberately not joined to the full-matrix zero-probe
baseline, because its model population is different.

The right panel answers the separate literal-average question by averaging the
true and reconstructed values over each model's own fixed published target
cells. Its unit is still a mixture of normalized endpoint scales, so it should
not be interpreted as clinical utility.

![Single-evaluation informativeness](../figures/probe_informativeness_rank1.png)

This ranking is the first step of greedy selection: lower one-probe scorecard
MedAE means greater conditional predictive utility, and the chart expresses it
as improvement over the 2.100-point column-median baseline. Coverage is printed
on every bar because a probe with no published result for a target model
reveals nothing. The all-THUNDER top of the list reflects the current block
structure and 68% THUNDER row coverage.

Machine-readable results are in
[`experiments/probe_selection_results_rank1.json`](../experiments/probe_selection_results_rank1.json)
and [`outputs/probe_informativeness_rank1.csv`](../outputs/probe_informativeness_rank1.csv).
The exact protocol and its limitations are documented in
[`benchpress-parity.md`](benchpress-parity.md).
