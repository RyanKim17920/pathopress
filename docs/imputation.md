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

A MedAE of 1.647585 means that half of the matched-CV predictions are within
1.647585 normalized points. It is not a confidence interval, clinical error,
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
for seeds 42–51. Every one of 2,027 cells is tested once per seed, yielding
20,270 out-of-fold predictions.

| Method/rank | Pooled MAE | Pooled MedAE |
|---|---:|---:|
| Column median | 4.275274 | 2.500000 |
| Bias-only rank 0 | 3.360017 | 1.787446 |
| Bias ALS rank 1 | **3.222008** | **1.647585** |
| Bias ALS rank 2 | 3.341028 | 1.700758 |

The median of the 30 rank-1 fold MedAEs is 1.662339. Its errors are within one,
three, five, and ten points for 35.1801%, 68.1944%, 81.0705%, and 93.6112% of
instances. Raw and logit Soft-Impute rank sweeps also choose rank 1.

This choice is protocol-specific. In a harder leave-one-suite-block-out test,
rank 1 gives 5.688229 MAE and 3.537207 MedAE, while error improves through
tested rank 6 at 5.093822/3.175723. This adverse shift sensitivity is why rank
1 is a matched-completion product choice, not a universal pathology rank claim.

## Completed matrix

[The imputation table](../outputs/imputations_rank1.csv) contains one row for
every supported model/evaluation pair: 2,027 reported values and 7,885 point
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
uses 20,270 OOF cells and compares disagreement, structural-support, and a
combined risk model. Structural support has Spearman 0.606612 with absolute
error. Its nominal 90% leave-fold-out conformal intervals achieve 89.9803%
coverage with median width 10.1435 points. The hybrid method achieves 89.9901%
coverage with 10.0770-point median width; keeping its lowest-risk 20% reduces
MedAE from 1.647585 to 0.607845.

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

All-known greedy reconstruction gives MedAE 1.474879 with five probes and
1.270529 with ten; hidden-only values are 1.637639 and 1.538607. Held-out-model
hidden-cell MedAE is 2.126261 and 2.142613. Revealed probes contribute exact
zero-error cells only to the all-known denominator, so these quantities must
not be merged.

OOF pairwise ordering accuracy increases from median 0.754237 at zero margin to
0.903077 at a five-point margin. Median top-set recovery is 0.694444,
0.784524, and 0.809762 for the top 10%, 20%, and 30%. A five-point difference
still has different native meaning across endpoint types.

Primary artifacts:

- [shared matrix/folds](../experiments/shared_artifacts_manifest.json)
- [bias-ALS CV](../experiments/benchpress_style_results.json)
- [Soft-Impute sweep](../experiments/soft_impute_rank_sweep_results.json)
- [probe compression](../experiments/probe_compression_rank1.json)
- [ranking preservation](../experiments/ranking_preservation_rank1.json)
- [confidence calibration figure](../figures/confidence_calibration_rank1.png)
