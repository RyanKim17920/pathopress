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
from 30,992 genuinely hidden sparse-row and temporal predictions. Its nested
leave-target-model-out evaluation obtains 94.98% empirical coverage at nominal
90% and 15.25-point median width. The CLI and website expose the risk, interval,
probe bucket, fallback scope, and calibration group counts, or deterministically
abstain for unsupported columns. This remains retrospective and is not a
clinical or future-domain coverage guarantee.

## Probe and ranking validation

All-known greedy reconstruction gives MedAE 1.397334 with five probes and
1.213706 with ten; hidden-only values are 1.548536 and 1.493709. Held-out-model
hidden-cell MedAE is 1.885364 and 2.008051. Revealed probes contribute exact
zero-error cells only to the all-known denominator, so these quantities must
not be merged.

The current standalone ranking release is derived from the dedicated margin-5
probe-compression objective. At `k=10`, unrestricted all-known median pairwise
accuracy is 0.877976 (random-prefix median 0.602858), and hidden-only held-out
model accuracy is 0.775000. A five-point normalized difference still has
different native meaning across endpoint types.

Primary artifacts:

- [shared matrix/folds](../experiments/shared_artifacts_manifest.json)
- [bias-ALS CV](../experiments/benchpress_style_results.json)
- [Soft-Impute sweep](../experiments/soft_impute_rank_sweep_results.json)
- [probe compression](../experiments/probe_compression_rank1.json)
- [ranking preservation](../experiments/ranking_preservation_rank1.json)
- [confidence calibration figure](../figures/confidence_calibration_rank1.png)
