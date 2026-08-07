# Generated figure gallery

Figures are result-first: plotting scripts read committed compact JSON/CSV/NPZ
artifacts and do not implicitly rerun expensive experiments. PNGs are review
assets; matching PDFs are publication/vector assets.

The fixed research matrix is 59 models × 187 protocol-level evaluations with
2,122 observed cells (19.2332% density) and 8,911 rank-1 point estimates.

## Canonical main figures

These six figures carry the main result. Other retained figures are diagnostic
or supplementary and should not be presented as alternative headline charts.

| Figure | What it shows | Source artifact |
|---|---|---|
| [BenchPress-style pathology hero](pathopress_benchpress_hero_rank1.png) | four target-cell examples plus random, most-predictive, and 25-task scorecard trajectories | [`benchpress_style_hero_summary.json`](../experiments/benchpress_style_hero_summary.json) |
| [Completed matrix](matrix_completed_rank1.png) | reported values and translucent rank-1 estimates | [`imputations_rank1.csv`](../outputs/imputations_rank1.csv) |
| [Matched validation](benchpress_style_validation_rank1.png) | rank sweep, OOF parity, suite errors, error distribution | [`benchpress_style_results.json`](../experiments/benchpress_style_results.json) |
| [One-probe informativeness](probe_informativeness_rank1.png) | first-step greedy utility with observed-model coverage | [`probe_informativeness_rank1.csv`](../outputs/probe_informativeness_rank1.csv) |
| [Model-average prediction](probe_dual_objective_rank1.png) | model-average MedAE for scorecard-selected probe sets | [`probe_dual_objective_rank1.csv`](../outputs/probe_dual_objective_rank1.csv) |
| [Ranking preservation](ranking_preservation_rank1.png) | current margin-5 all-known/random and held-out probe trajectories | [`ranking_preservation_rank1.json`](../experiments/ranking_preservation_rank1.json) |

The informativeness chart is a predictive-information ranking, not a cost
ranking. Its sample-count metadata does not measure runtime, memory, annotation,
access, licensing, acquisition, or dollars.

## Supplementary figures

These figures retain unique robustness, deployment, or cost detail
without duplicating the canonical narrative.

| Figure | What it shows | Source artifact |
|---|---|---|
| [Soft-Impute rank sweep](soft_impute_rank_sweep.png) | raw/logit MedAE and MedAPE rank sensitivity | [`soft_impute_rank_sweep_results.json`](../experiments/soft_impute_rank_sweep_results.json) |
| [Confidence calibration](confidence_calibration_rank1.png) | risk correlation, retention, strata, conformal intervals | [`confidence_calibration_rank1.json`](../experiments/confidence_calibration_rank1.json) |
| [Unseen-model confidence](new_model_confidence_rank1.png) | empirical risk–coverage, coverage–width, suite coverage, abstention support | [`new_model_confidence_rank1.json`](../experiments/new_model_confidence_rank1.json) |
| [Temporal deployment](temporal_deployment_rank1.png) | seven 2025 target releases with prior-only training | [`temporal_deployment_rank1.json`](../experiments/temporal_deployment_rank1.json) |
| [Evaluation cost evidence](evaluation_cost_evidence_coverage.png) | source coverage, missingness, and explicitly non-monetary pre-error feasibility strata | [`evaluation_cost_evidence.json`](../data/evaluation_cost_evidence.json) |

Matched rank-1 error is 3.134532 MAE / 1.609006 MedAE. All-known greedy
scorecard MedAE is 1.397334 and 1.213706 at five and ten probes; hidden-only is
1.548536 and 1.493709. These are normalized-score retrospective quantities,
not clinical accuracy or confidence intervals.

The current pre-error feasibility curve uses 25 image/patch classification
protocols selected without outcome errors. It is not measured cost. The
checked-in greedy curves are complete through `k=10`; all-known random baselines
cover unrestricted `k=30` and every feasibility candidate through `k=25`.
Held-out and ranking random controls cover `k=10`. No exhaustive `C(25,5)` or
`C(30,5)` result is claimed for the current 59 × 187 matrix.

## Regeneration

Install the research dependencies first. PyTorch is isolated in the optional
`mlp` extra and is not required to render the checked-in figures.

```bash
python3 -m pip install -e '.[research]'
python3 scripts/plot_benchpress_style.py
python3 scripts/plot_probe_selection.py
python3 scripts/plot_ranking_preservation.py
python3 scripts/plot_confidence_calibration.py
python3 scripts/plot_new_model_confidence.py
python3 scripts/plot_temporal_deployment.py
python3 scripts/plot_benchpress_style_hero.py
python3 scripts/plot_probe_dual_objective.py
python3 scripts/plot_evaluation_cost_evidence.py
```

The exact experiment commands and local-cache boundaries are in
[the experiment index](../experiments/README.md).
