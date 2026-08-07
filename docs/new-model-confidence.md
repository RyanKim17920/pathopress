# Confidence for a genuinely unseen model

PathoPress can attach empirical 90% intervals to a newly appended model row.
This calibration population and artifact are separate from the held-out-cell
intervals for models already represented in the matrix; existing-row residuals
are never reused for new-row confidence.

## Calibration design

For every eligible source model, the experiment removes that model from the
training matrix, appends a new row containing only `k` known scores, and hides
the rest. It repeats this at `k = 1, 3, 5, 10` over three deterministic probe
seeds. It also includes pinned temporal-release simulations, where each target
was predicted using only models released before its cutoff.

The raw audit contains 30,992 hidden predictions from 59 target models:
20,412 leave-one-model-out sparse-probe predictions and 9,770 temporal-release
predictions. Risk combines a model-balanced evaluation residual with a
suite/same-suite-probe context. Evaluation risk requires at least five distinct
target-model groups. Context falls back from suite plus same-suite probe, to
suite, then global `k`; unsupported evaluations or contexts abstain.

Probe counts between calibrated values use the largest supported count not
exceeding the submitted evidence. Two scores use `k=1`, for example, while
eight use `k=5`. This avoids assigning an unvalidated, more optimistic bucket.

## Leakage control

Cross-fitted metrics use nested model-group exclusion. For target model `T`,
all risk lookups exclude `T`. When a residual from calibration model `C` helps
estimate `T`'s conformal scale, its risk estimate excludes both `T` and `C`.
Changing `T`'s hidden values therefore cannot change `T`'s audited interval.
The artifact records every excluded target and each fold's training-model IDs.

After evaluation, the deployable lookup is fitted on every held-out group. A
genuinely new model is absent from those groups by definition. Repeated probe
contexts are model-balanced so densely reported source models do not dominate.

## Empirical results and interpretation

At a nominal 90% level, nested held-out empirical coverage is 94.98% overall
with a 15.25-point median width. Coverage is 96.30% at `k=1`, 93.62% at `k=3`,
94.81% at `k=5`, and 94.49% at `k=10`.
94.22% at `k=5`, and 94.12% at `k=10`. The leave-one-model-out population
covers 94.02%; the temporal population covers 96.33%. Risk ordering is useful:
retaining lower-risk predictions progressively reduces held-out MedAE.

These are retrospective empirical estimates over the available pathology
foundation-model population, not distribution-free guarantees under arbitrary
model-family or temporal shift and not clinical guarantees. Intervals are
symmetric in normalized-score space and clipped to `[0, 100]`.

## Interfaces and artifacts

- [`new_model_confidence_rank1.json`](../experiments/new_model_confidence_rank1.json)
  contains deploy lookups, group/prediction counts, fallback levels, scales,
  cross-fit membership, empirical coverage, and limitations.
- [`new_model_confidence_predictions_rank1.csv`](../experiments/new_model_confidence_predictions_rank1.csv)
  contains each probe set, actual, prediction, residual, cross-fitted risk,
  interval, excluded target, and calibration scope.
- `pathopress add-model ... --confidence` and the static site return the
  interval, risk, bucket, fallback scope, group/prediction counts, or explicit
  abstention. Browser calculations stay local.

Reproduce with:

```bash
PYTHONPATH=src python3 experiments/run_new_model_confidence.py --workers 4
```
