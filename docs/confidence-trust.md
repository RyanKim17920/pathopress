# Confidence and calibrated trust probability

PathoPress ports the confidence-calibration experiment from Microsoft
BenchPress commit `0a684b63ee0e4a401cb907a3827a82ea997d74c4`. The pathology adaptation changes
the selected latent rank from two to one and uses pathology's own method-grid
results. It does not change the experiment's folds, generator-selection rule,
features, risk learners, or diagnostics.

## Exact generator contract

The target is Logit Bias ALS, rank 1, lambda 0.1. Two stacks are built from the
same 30 persisted Section-4 folds:

- same-family sensitivity: Logit Bias ALS rank 1 at lambda 0.01, 0.1, and 1.0;
- strong alternatives: the twelve lowest-MedAPE, full-coverage best-HP
  transform/method rows, excluding the target transform/method.

Every generator is loaded from the method-comparison prediction cache. Before
stacking, the runner verifies the score hash, fold hash, matrix identity, matrix
shape, fold protocol, upstream commit, method specification, held-out cell
coordinates, and actual values. A mismatched or missing cache fails closed.

The two prediction stacks each yield four features: standard deviation, scaled
median absolute deviation, target-to-ensemble-median distance, and the 90th-to-
10th percentile span. Matrix support contributes the upstream eight structural
features: row and column observation counts, row and column medians, column
dispersion, strongest row-peer correlation and overlap, and strongest column-
neighbor correlation.

## Leakage-free evaluation

Three uncertainty models use disagreement features, structural-support
features, or their union. For each point-prediction fold, the risk learner is
fit on all other folds and predicts `log1p(abs(error))`. Its inner selection
chooses between ridge and the upstream MLP widths `(16)`, `(32)`, and `(64,32)`
using training folds only.

As in upstream, the unlearned same-family MAD and strong-method MAD are also
retained as diagnostic confidence methods; the three learned methods are the
ones emphasized in the main figure.

The artifact reports the same BenchPress diagnostics:

- Spearman correlation between predicted risk and realized absolute error;
- risk-coverage at 100%, 80%, 60%, 40%, and 20% retained predictions;
- low-, medium-, and high-risk terciles;
- normal-theory and leave-fold-out conformal 90% interval coverage and width.

The two percentage denominator guards deliberately follow different pinned
upstream primitives: MedAPE excludes `abs(actual) <= 1e-6`, while relative
interval width excludes only `abs(actual) <= 1e-8`. They are separate named
constants and tests so a future metric refactor cannot silently conflate them.

The raw cell table retains fold, model, evaluation, all input features, each
risk estimate, interval endpoints, conformal scale, and trust probability.

## What “trust” means

Trust is the calibrated event probability

`P(abs(predicted normalized score - actual normalized score) <= 10)`.

Ten points is one decile of the shared normalized 0-100 pathology score scale
and exactly preserves BenchPress's public trust event. It is an engineering
tolerance for the compressed benchmark matrix, not a clinical threshold and
not evidence of diagnostic safety.

The risk-to-probability map follows BenchPress's deterministic binned isotonic
procedure: sort by risk, form twenty equal-count bins, measure the event rate,
and apply weighted pool-adjacent-violators so trust cannot rise with risk. For
reported evaluation metrics, the mapping is itself leave-fold-out and
group-purged: a cell's probability is produced by a calibrator that sees
neither its point-prediction fold nor any repeated-seed instance of the same
model-evaluation target. The JSON records fold-specific calibration bins, event
prevalence, Brier score, log loss, expected calibration error, and a reliability
curve.

For an unreported cell belonging to an existing supported model, the deploy
artifact follows BenchPress's website estimator: average that model's and that
evaluation's median cross-fitted hybrid risk, then apply the full held-out
monotone mapping. It abstains unless both histories exist. A genuinely unseen
model row always abstains from this probability because the calibration
population is different; its separate group-conformal interval remains
available.

## Reproduction

After the canonical score matrix and all 343 exact method-comparison caches are
current:

```bash
PYTHONPATH=src python3 experiments/run_confidence_calibration.py
MPLCONFIGDIR=/tmp/pathopress-mpl PYTHONPATH=src python3 scripts/plot_confidence_calibration.py
PYTHONPATH=src python3 scripts/build_public_release.py
```

The primary artifacts are
`experiments/confidence_calibration_rank1.json`,
`experiments/confidence_cells_rank1.csv`,
`experiments/deployment_confidence_rank1.json`, and
`figures/confidence_calibration_rank1.{png,pdf}`.
