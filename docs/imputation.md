# Imputation, validation, and uncertainty

PathoPress predicts missing model × evaluation cells on a direction-preserving
0–100 normalized scale. A validation cell is first hidden, predicted from the
remaining matrix, and compared with its reported value:

```text
absolute error = |predicted normalized score - reported normalized score|
MAE            = mean absolute error
MedAE          = median absolute error
MedAPE         = median absolute percentage error
```

A MedAE of 1.609006 means that half of the matched-CV predictions are within
1.609006 normalized points. It is not a confidence interval, clinical error,
or universal accuracy percentage.

## Native metric mappings

| Source | Native endpoint | Normalized score |
|---|---|---|
| Patho-Bench | AUC, balanced accuracy, c-index in 0–1 | `100 × value` |
| Patho-Bench | weighted kappa in −1–1 | `50 × (value + 1)` |
| EVA | balanced accuracy or Dice in 0–1 | `100 × value` |
| THUNDER | F1 reported in 0–100 | unchanged |
| HEST | Pearson correlation in −1–1 | `50 × (value + 1)` |
| PathoROB | Robustness Index in 0–1 | `100 × value` |

Column standardization allows these endpoints to be fitted jointly, but it
does not make them biologically or clinically interchangeable. Pooled errors
must be read alongside suite/protocol errors.

## Selected point predictor

The selected model is BenchPress's logit, column-standardized bias ALS with one
latent interaction factor. It includes a global mean plus model and evaluation
biases, so “rank 1” is not a claim that the observed raw matrix has exact
algebraic rank one.

The canonical fold artifact assigns every observed cell to one of three folds
for seeds 42–51. Across 2,122 observed cells and ten seeds, the supported-fold
adapter yields 21,181 out-of-fold predictions; 39 held targets occur in a
column with no remaining training observation and are recorded as unsupported
rather than imputed.

| Method/rank | Pooled MAE | Pooled MedAE |
|---|---:|---:|
| Column median | 4.151756 | 2.400000 |
| Bias-only rank 0 | 3.284259 | 1.724449 |
| Bias ALS rank 1 | **3.134532** | **1.609006** |
| Bias ALS rank 2 | 3.265505 | 1.640863 |

The median of the 30 rank-1 fold MedAEs is 1.608566. Its errors are within one,
three, five, and ten points for 35.9898%, 69.1469%, 81.9366%, and 93.8577% of
instances. Raw and logit Soft-Impute sweeps independently choose rank 1. At
rank 1 their pooled MAE/MedAE values are 3.378152/1.793787 in raw space and
3.344836/1.737833 in logit space.

This choice is protocol-specific. In a harder leave-one-suite-block-out test,
rank 1 gives 5.788534 MAE and 3.599615 MedAE on 1,051 supported predictions,
while error improves through tested rank 6 at 5.099363/3.203260. Another 95
suite-block targets are unsupported because their columns become empty. This
adverse shift sensitivity is why rank 1 is a matched-completion product choice,
not a universal pathology rank claim.

## Completed matrix

[The imputation table](../outputs/imputations_rank1.csv) contains one row for
every supported model/evaluation pair: 2,122 reported values and 8,911 point
estimates. `status=observed` cells are preserved exactly; `status=imputed` cells
are estimates. Rank-2 differences are sensitivity diagnostics, not intervals.
Catalog-only evaluations with no retained observations cannot be completed
because they have no learned evaluation bias or factor.

The point implementation directly matches the pinned BenchPress reference to
floating-point precision. Reproduce that check with:

```bash
PYTHONPATH=src python3 scripts/verify_benchpress_parity.py \
  /path/to/benchpress --rank 1
```

The algorithm is exact for a shared input matrix and rank; native pathology
metric normalization and the selected rank are documented adaptations.

## Confidence is a separate cross-fitted experiment

[Confidence calibration](../experiments/confidence_calibration_rank1.json)
uses 21,181 OOF cells and compares disagreement, structural-support, and a
combined risk model. Structural support has Spearman 0.602190 with absolute
error. Its nominal 90% leave-fold-out conformal intervals achieve 89.9816%
coverage with median width 9.9293 points. The combined method achieves 89.9721%
coverage with 9.9131-point median width; keeping its lowest-risk 20% reduces
MedAE from 1.609006 to 0.611151.

Those figures measure retrospective selective prediction. The generator stack
follows BenchPress's exact selection contract: three same-family lambda
variants plus the twelve strongest full-coverage Section-4 alternatives. The
eight structural features and learned disagreement/support/hybrid risks are
unchanged; pathology's rank and method roster are data adaptations. This does
not establish calibration for a new institution, patient population, or
endpoint.

[Deployment confidence](../experiments/deployment_confidence_rank1.json) is a
separate, hash-bound lookup artifact. It combines model- and evaluation-level
median hybrid risk for an existing supported cell, applies the held-out
conformal scale, and maps risk to calibrated P(|error| <= 10 normalized points).
It abstains without both histories and does not apply this trust probability to
new model rows. See [the confidence/trust protocol](confidence-trust.md).

[Unseen-model confidence](new-model-confidence.md) is calibrated separately
from 33,272 genuinely hidden sparse-row and temporal predictions. Its nested
leave-target-model-out evaluation obtains 94.80% empirical coverage at nominal
90% and 14.7223-point median width. The CLI and website expose the risk, interval,
probe bucket, fallback scope, and calibration group counts, or deterministically
abstain for unsupported columns. This remains retrospective and is not a
clinical or future-domain coverage guarantee.

## Probe and ranking validation

All-known greedy reconstruction gives MedAE 1.397334 with five probes and
1.213706 with ten; hidden-only values are 1.548536 and 1.493709. Revealed
probes contribute exact zero-error cells only to the all-known denominator, so
these quantities must not be merged.

The held-out-model protocol, which selects probes on training models and then
completes each validation row in isolation, gives hidden-cell MedAE **3.137815**
at five probes (n = 321) and **3.058947** at ten (n = 307). These come from
[`probe_selection_results_rank1.json`](../experiments/probe_selection_results_rank1.json)
at `heldout_model.validation[step].hidden_only.medae`, and they replace the
previously quoted pair (1.885364 and 2.008051), which traces to no artifact.

Three scope conditions are mandatory whenever they are quoted. They come from a
**single 70/30 `split_mode = family_blocked` holdout with 48 training models and
11 validation models**, not from the 34-fold LOFO protocol used below, so the
estimate is seed-sensitive at this validation size. The hidden-cell denominator
shrinks as k grows (334 at k=0, 321 at k=5, 307 at k=10), so the curve is not
scored on a fixed cell set. And the direction of the result is the reverse of
what was previously stated here: held-out error is **worse** than the 1.900
all-known zero-probe scorecard baseline and worse than this protocol's own k=0
hidden-cell baseline of 2.933142 (n = 334). Revealing probes on an unseen model's
row does not improve reconstruction of that row under this split.

Comparing probe-selection arms requires a further step, because each arm
otherwise removes its own revealed cells and is therefore scored on a different
denominator. Under the matched-cell leave-one-family-out protocol, the union of
the cells revealed by the greedy prefix and by all ten random repeats is excluded
per fold and per depth, and every arm is scored on the identical remainder. At
four probes that leaves 1,636 of 2,122 cells, on which greedy is 1.8781, the k=0
baseline 2.6524 (both being medians of the 34 fold medians), and the random
control 2.6013 under convention A, the median over all 340 fold × repeat MedAEs
(2.6260 under convention B, median of fold medians; the convention must always
be named). Greedy beats k=0 in 18 of 34 folds (Wilcoxon p = 0.0088) and random
in 22 of 34 (p = 0.0151) — that random test, and the CI against random, are
computed on **convention B** fold medians, not on the convention-A value quoted
in the same sentence, because a paired test needs one random value per fold.
The reduction versus k=0 has a
bootstrap-over-folds 95% CI of [2.8%, 58.7%], so its point estimate is not
precise and should not be quoted to three significant figures.

Two disclosures belong with these figures. Greedy selects on
`parity.median_absolute_error`, which scores revealed probe cells as literal 0.0,
so the selection objective is about 15.9% optimistic at four probes (1.5142
against a held-out 1.7994). And per-evaluation utility is null: 86 of 174 scored
columns (49.4%, 95% CI [42.0%, 56.9%]) improve at four greedy probes over their
own k=0 baseline. Reproduce all of these with
`scripts/replay_lofo_matched_cells.py`, which writes
[the matched-cell artifact](../experiments/lofo_matched_cells_rank1.json).

The current standalone ranking release is derived from the dedicated
probe-compression objective. At `k=10` and margin 0 (all 17,159 model pairs,
all 187 columns), unrestricted all-known greedy pairwise accuracy is 0.679
versus 0.552 random. At a margin of 5 normalized points — which retains 6,048
of 17,159 pairs (35%) across 148 of 187 columns, because the median true score
gap between model pairs is only 3.10 normalized points and 64.8% of pairs
differ by less than 5 points — greedy accuracy is 0.878 versus 0.603 random.
Greedy exceeds random at every tested margin (absolute 0–10 points; relative
0.25–1.0 × column SD or IQR); relative-margin rows retain all 187 columns and
are therefore the more robust evidence. Hidden-only held-out-model accuracy at
margin 5 is 0.804 under the LOFO protocol (pairwise_n_pairs = 1 at k=10);
the estimate rests on too few independent pairs to support precise inference
and should not be read as an improvement over the prior single-split value.
A five-point normalized difference has different native
meaning across endpoint types; see the full margin sweep in
[docs/benchpress-parity.md](benchpress-parity.md).

Primary artifacts:

- [shared matrix/folds](../experiments/shared_artifacts_manifest.json)
- [bias-ALS CV](../experiments/benchpress_style_results.json)
- [Soft-Impute sweep](../experiments/soft_impute_rank_sweep_results.json)
- [probe compression](../experiments/probe_compression_rank1.json)
- [ranking preservation](../experiments/ranking_preservation_rank1.json)
- [matched-cell LOFO comparison](../experiments/lofo_matched_cells_rank1.json),
  regenerated by `scripts/replay_lofo_matched_cells.py`
