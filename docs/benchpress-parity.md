# What full BenchPress parity means for PathoPress

This note is based on a code-level audit of Microsoft BenchPress at commit
`0a684b63ee0e4a401cb907a3827a82ea997d74c4`. It distinguishes features that
look similar in a figure but answer different scientific questions.

## The GitHub hero chart

The panel titled **Overall score prediction** does not predict a model's
arithmetic mean score. It asks how accurately the whole observed scorecard can
be reconstructed after measuring a small, globally selected set of benchmark
columns.

For a probe set P and each target model i, BenchPress starts with the complete
published matrix, hides every observed score in row i except scores in P, runs
matrix completion once, and evaluates the row's published cells. If a target
cell is itself in P, its exact reported value is copied into the prediction and
therefore contributes zero error. Errors are pooled over the same set of known
model-by-benchmark cells at every probe count. Greedy forward selection adds
the candidate column that produces the lowest pooled median absolute error
(MedAE). The plotted curves are:

- the benchmark-column-median baseline at zero probes;
- random global probe orders, summarized across seeds with an interquartile
  band;
- a greedy sequence selected from every candidate benchmark; and
- a second greedy sequence selected from a manually audited low-cost
  allowlist.

PathoPress reproduces the scorecard-reconstruction quantity, but also reports
hidden-only error. Hidden-only excludes the exact revealed probe cells from the
denominator. The latter is more honest about the ability to predict scores that
were not actually measured.

## What “informativeness” means

The first greedy step evaluates every individual benchmark as the sole probe.
Ranking those candidates by resulting scorecard MedAE is the natural
BenchPress-style informativeness ranking. Later positions are conditional:
the second probe is the best addition to the first, the third is the best
addition to the first two, and so on. This is not a causal importance score and
it is not a claim about the biological value of a dataset. It measures
redundancy and predictive coverage in the assembled score matrix.

Coverage must accompany the ranking. A selected probe supplies information
only for models that actually have a published value in that column. This is
especially important here because the current 47-by-28 pathology matrix is
suite-blocked: many models occur in only HEST, THUNDER, or PathoROB, and only
11 models span all three. A probe can consequently appear informative because
it identifies a well-covered suite block rather than because its dataset is a
universally useful assay.

## Arithmetic-average prediction is separate

For clarity, PathoPress also computes a true per-model average diagnostic. For
each target row and probe set, it compares the mean of all that model's
published normalized scores with the mean of the reconstructed values at those
same columns. This answers the user's literal “predict the average” question,
but it is not the quantity in the BenchPress hero chart. It also has an
important limitation: averaging F1, Pearson correlation, and robustness index
after mapping them to a nominal 0–100 scale is a convenience summary, not a
validated common pathology utility measure.

## Validation layers in the original repository

Reproducing one figure is not full system parity. BenchPress has several
separate evaluation layers:

1. **Within-model cross-validation.** Ten seeds and three folds hide cells
   within every model row and compare completion ranks and baselines.
2. **All-known probe selection.** The optimistic/transductive hero protocol
   described above selects global probe sets.
3. **Held-out-model probe validation.** A deterministic 70/30 row split selects
   probes using training models only. Each validation model is then added in
   isolation with only its probe scores visible; the primary validation metric
   excludes the probe cells.
4. **Ranking preservation.** For each benchmark column, true and reconstructed
   pairwise model orderings are compared when at least one score was held out.
   Results are summarized as the median benchmark-level accuracy at several
   score-gap margins. Top-10/20/30-percent set recovery is a separate metric.
5. **Confidence calibration.** Cross-fitted risk models combine disagreement
   across alternate predictors with row/column support features. Reliability
   is evaluated by error-risk correlation, risk-coverage curves, uncertainty
   strata, and leave-fold-out conformal intervals. A rank sensitivity curve is
   not a confidence interval.
6. **Temporal deployment.** Newer target models are predicted using only
   strictly earlier releases and a limited number of revealed target scores.
7. **Error analysis and ablation.** Benchmark/model support, score spread,
   neighbor correlation, category/provider overlap, and metadata are tested as
   correlates or controlled ablations.

The current repository exactly matches Microsoft's standalone default
logit-plus-bias-ALS point predictor, implements the matched within-row
cross-validation and Soft-Impute rank sweep, and now adds the probe-selection
and ranking primitives. Confidence calibration and temporal validation remain
future work; the latter cannot be run until model release dates are collected.

## Pathology-specific changes required for credible use

- Keep evaluation-protocol columns distinct even when two suites use the same
  underlying dataset. Dataset deduplication is not score-column collapse.
- Publish all-known and held-out-model probe results together. Add
  leave-one-family or leave-one-institution validation when model metadata is
  available.
- Report probe coverage and complete-case sensitivity, because suite-block
  missingness can dominate candidate rankings.
- Do not copy BenchPress's universal five-point ranking margin. Five normalized
  points means different things for F1, Pearson r, and robustness index. Use
  protocol-specific meaningful differences or within-column standardized
  gaps.
- Do not call a probe set “low cost” until runtime, sample count, compute,
  tissue access, and licensing requirements have been audited.
- Add Patho-Bench and eva score anchors before making broad claims. Their task
  catalogs are inventoried, but their score columns are absent from the current
  supported completion matrix.

## Upstream implementation map

- Hero rendering: `experiments/sec1_intro/hero_figure/plot.py`
- Probe evaluator: `benchpress/evaluation_harness.py`
- All-known greedy/random probes:
  `experiments/sec5_findings/optimal_probe/all_known/`
- Held-out-model probes:
  `experiments/sec5_findings/optimal_probe/holdout/`
- Ranking preservation:
  `experiments/sec5_findings/ranking_preservation/`
- Confidence calibration: `benchpress/methods/confidence.py` and
  `experiments/sec6_trust/confidence_calibration/`
- Temporal deployment:
  `experiments/sec5_findings/temporal_deployment/`

The clean upstream checkout does not include every cached `results/*.json.gz`
file consumed by the hero plotting script, so the published hero cannot be
rerendered from source alone without regenerating its experiments. PathoPress
stores its generated result tables explicitly.
