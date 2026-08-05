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
ranks 1–10 in raw and logit spaces. Its MedAPE criterion narrowly chooses rank
2 in both spaces; raw-space MedAE also chooses rank 2, while raw/logit MAE and
logit-space MedAE choose rank 1. This is intentionally kept distinct from the
bias-ALS default predictor.

## Results for the current matrix

The primary-source-parsed, support-filtered matrix has 47 models, 28 evaluations,
and 806 observed cells (61.25% density). These cells have not yet received dual
human review. Rank 1 is the strongest simple choice in the
random-cell and sparse-probe aggregates; rank 2 has the lowest suite-block MAE.

| Protocol | Best rank by MAE | n | MAE | MedAE |
| --- | ---: | ---: | ---: | ---: |
| Fixed random-cell holdout | 1 | 1,610 | 2.326 | 1.013 |
| Leave-one-suite-block-out | 2 | 485 | 4.312 | 2.320 |
| Sparse new-model probes, pooled | 1 | 1,633 | 2.398 | 1.113 |

Suite-block difficulty is highly uneven at rank 2: HEST is 1.268/1.080
MAE/MedAE (`n=171`), THUNDER is 4.964/3.905 (`n=272`), and PathoROB is
12.475/9.344 (`n=42`). PathoROB currently contributes only three published
aggregate score columns, so this is a small, structurally different block and
should not be generalized to its full task inventory.

At rank 1, sparse probes give MAE/MedAE of 2.400/1.112 for `k=3` (`n=665`),
2.545/1.099 for `k=5` (`n=583`), and 2.171/1.143 for `k=10` (`n=385`). These
rows do not form a controlled learning curve: increasing `k` changes both the
eligible target cells and their sample count, and the script uses one fixed
probe draw per model. Compare ranks within a row, not errors across `k`, unless
a future experiment fixes a common eligible-model/target set and repeats probe
draws.
