# PathoPress retrospective validation

`run_validation.py` evaluates the current primary-source-parsed `data/scores.csv` matrix
with three deterministic protocols. It sweeps ALS ranks 1 through 6 on the
same split definitions and writes the machine-readable results to
`results.json`.

Run from the repository root:

```bash
python3 experiments/run_validation.py
```

Use `--workers 1` for a simple serial run. Parallel and serial runs are
numerically deterministic because `pathopress.completion.complete` uses fixed
ensemble seeds and all validation splits are constructed before rank fitting.
Worker jobs carry the source matrix explicitly, so `--workers` is portable to
both fork- and spawn-based Python multiprocessing platforms.

## Protocols

- **Random cell:** ten fixed 20% holdouts, greedily preserving at least two
  training observations per model and three per evaluation. This is a useful
  interpolation check but is leaky because other scores for the same model and
  suite remain visible.
- **Leave one suite block out:** for each score suite, all of that suite's
  scores are hidden for models that have scores in another suite. Unsupported
  all-missing rows/columns are excluded and counted in the JSON rather than
  passed to the core completion API.
- **Sparse new-model probes:** each eligible model is evaluated independently.
  Exactly 3, 5, or 10 fixed random scores are revealed and its other published
  scores are predicted. Models with no remaining target after revealing `k`
  scores are ineligible.

MAE and MedAE are expressed in **normalized-score points on the 0–100 scale**.
The source metrics include differently scaled F1, Pearson correlation, and
robustness aggregates. Consequently, one normalized point is not necessarily
one accuracy percentage point or a common clinical-utility increment.
Reported `n` values are pooled prediction instances: random repeats can select
the same source cell, and separate sparse-`k` simulations can target it again.

These are retrospective feasibility checks, not claims of prospective or
clinical validity. Random-cell splits share suite context; suite-block splits
still train target columns on other models; sparse probes retain all other
published models. None prevents model-family, publication-selection,
pretraining-data, temporal, or institutional leakage.

`run_probe_selection.py` reproduces the BenchPress all-known global-probe
protocol with pooled MedAE, 10 nested random probe orders, and deterministic
greedy forward selection through 10 columns. It additionally exports strict
hidden-cell metrics, per-model average error, one-probe informativeness, and a
70/30 isolated held-out-model validation. Its generated JSON records the
audited upstream commit, configuration, input hash, selected trajectories, and
candidate scores.

`run_benchpress_style.py` adds the closer BenchPress reproduction: ten random
within-model fold assignments × three folds and a bias-only/rank 1–10 sweep. It
writes aggregate and fold-level metrics to `benchpress_style_results.json` and
selected rank-1 out-of-fold diagnostics to
`benchpress_style_predictions_rank1.csv`. Unlike the original feasibility
protocols below, every observed cell is tested once per seed. Rank 1 currently
wins by pooled MAE, pooled MedAE, and median fold MedAE.

`run_soft_impute_rank_sweep.py` separately reproduces the method behind
BenchPress's published rank U-curve: iterative truncated-SVD completion at
ranks 1–10 in raw and logit spaces. Both spaces choose rank 1 on the expanded
matrix. This is intentionally kept distinct from the bias-ALS default
predictor.

## Results for the current matrix

The primary-source-parsed, support-filtered matrix has 59 models, 165
evaluations, and 1,967 observed cells (20.2054% density). These cells have not
yet received dual human review. Rank 1 is selected by the matched BenchPress
within-model CV and is strongest among ranks 1–6 for random-cell validation.
The suite-block stress test instead prefers rank 5.

Matched-fold rank-1 bias-ALS gives 3.005264 MAE, 1.603026 MedAE, and 1.609435
median fold MedAE, versus 4.092133/2.477500 for the column-median baseline.
Rank 0 is 3.056250/1.711456 and rank 2 is 3.117610/1.632035. Both raw and logit
Soft-Impute sweeps independently select rank 1.

| Protocol | Best rank by MAE | n | MAE | MedAE |
| --- | ---: | ---: | ---: | ---: |
| Fixed random-cell holdout | 1 | 3,930 | 2.834996 | 1.526795 |
| Leave-one-suite-block-out | 5 | 1,009 | 4.952972 | 3.055638 |
| Sparse new-model probes, pooled | 1 | 4,896 | 3.190380 | 1.817465 |

For comparison at the selected completion rank 1, suite-block MAE is 4.2643 for
Patho-Bench, 8.8757 for EVA, 1.4860 for HEST, 14.4385 for PathoROB, and 6.3854
for THUNDER. Overall rank-1 suite-block error is 5.612789 MAE and 3.525174
MedAE. PathoROB contributes only three supported columns, so its result is a
small, structurally different block and should not be generalized to its full
task inventory.

At rank 1, sparse probes give MAE/MedAE of 3.436124/1.883527 for `k=3`
(`n=1,790`), 3.116649/1.852156 for `k=5` (`n=1,684`), and
2.968354/1.717575 for `k=10` (`n=1,422`). These
rows do not form a controlled learning curve: increasing `k` changes both the
eligible target cells and their sample count, and the script uses one fixed
probe draw per model. Compare ranks within a row, not errors across `k`, unless
a future experiment fixes a common eligible-model/target set and repeats probe
draws.

The probe-selection artifact records the exact cross-suite probe lists.
All-known scorecard MedAE is 1.481124 at five probes and 1.196456 at ten;
hidden-only MedAE is 1.612112 and 1.539134. Held-out-model hidden-cell MedAE is
1.951271 and 1.879857. Literal-average MAE is 3.203489 and 2.908513 for the
all-known protocol, and 1.684778 and 1.231976 for held-out models.
