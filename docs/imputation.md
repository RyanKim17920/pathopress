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

A MedAE of 1.603026 means that half of the evaluated predictions are within
1.603026 normalized points. It is not a confidence interval, clinical error,
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
for seeds 42–51. Every one of 1,967 cells is tested once per seed, yielding
19,670 out-of-fold predictions.

| Method/rank | Pooled MAE | Pooled MedAE |
|---|---:|---:|
| Column median | 4.092133 | 2.477500 |
| Bias-only rank 0 | 3.056250 | 1.711456 |
| Bias ALS rank 1 | **3.005264** | **1.603026** |
| Bias ALS rank 2 | 3.117610 | 1.632035 |

The median of the 30 rank-1 fold MedAEs is 1.609435. Its errors are within one,
three, five, and ten points for 35.9532%, 69.6492%, 82.7758%, and 94.4331% of
instances. Raw and logit Soft-Impute rank sweeps also choose rank 1.

This choice is protocol-specific. In a harder leave-one-suite-block-out test,
rank 1 gives 5.612789 MAE and 3.525174 MedAE, while rank 5 is best at
4.952972/3.055638. Rank-1 suite-block MAE ranges from 1.4860 for HEST to
14.4385 for PathoROB. The latter has only three retained evaluations and must
not be generalized to the full suite.

## Completed matrix

[The imputation table](../outputs/imputations_rank1.csv) contains one row for
every supported model/evaluation pair: 1,967 reported values and 7,768 point
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
uses 19,670 OOF cells and compares disagreement, structural-support, and a
combined risk model. Structural support has Spearman 0.598017 with absolute
error. Its nominal 90% leave-fold-out conformal intervals achieve 89.9949%
coverage with median width 9.7677 points. Keeping the lowest-risk 20% reduces
MedAE from 1.603026 to 0.606970.

Those figures measure retrospective selective prediction. The six-predictor
stack consists of full-coverage ALS and Soft-Impute variants and is narrower
than BenchPress's top-12 diverse stack. It does not establish calibration for a
new institution, patient population, or endpoint.

[Deployment confidence](../experiments/deployment_confidence_rank1.json) is a
separate, hash-bound lookup artifact. It collapses OOF residuals to one value
per unique observed cell and conditions intervals by suite. Empirical
containment on supported rows is approximately 0.902–0.916 across the five
suites and applies only to existing supported rows.

[Unseen-model confidence](new-model-confidence.md) is calibrated separately
from 30,182 genuinely hidden sparse-row and temporal predictions. Its nested
leave-target-model-out evaluation obtains 94.77% empirical coverage at nominal
90% and 14.32-point median width. The CLI and website expose the risk, interval,
probe bucket, fallback scope, and calibration group counts, or deterministically
abstain for unsupported columns. This remains retrospective and is not a
clinical or future-domain coverage guarantee.

## Probe and ranking validation

All-known greedy reconstruction gives MedAE 1.481124 with five probes and
1.196456 with ten; hidden-only values are 1.612112 and 1.539134. Held-out-model
hidden-cell MedAE is 1.951271 and 1.879857. Revealed probes contribute exact
zero-error cells only to the all-known denominator, so these quantities must
not be merged.

OOF pairwise ordering accuracy increases from median 0.762195 at zero margin to
0.904907 at a five-point margin. Median top-set recovery is 0.678571,
0.775862, and 0.813333 for the top 10%, 20%, and 30%. A five-point difference
still has different native meaning across endpoint types.

Primary artifacts:

- [shared matrix/folds](../experiments/shared_artifacts_manifest.json)
- [bias-ALS CV](../experiments/benchpress_style_results.json)
- [Soft-Impute sweep](../experiments/soft_impute_rank_sweep_results.json)
- [probe compression](../experiments/probe_compression_rank1.json)
- [ranking preservation](../experiments/ranking_preservation_rank1.json)
- [confidence calibration figure](../figures/confidence_calibration_rank1.png)
