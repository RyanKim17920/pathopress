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
| [Dual probe objective](probe_dual_objective_rank1.png) | scorecard MedAE and separately evaluated model-average MedAE for the same selected sets | [`probe_dual_objective_rank1.csv`](../outputs/probe_dual_objective_rank1.csv) |
| [Ranking preservation](ranking_preservation_rank1.png) | current margin-5 all-known/random and held-out probe trajectories | [`ranking_preservation_rank1.json`](../experiments/ranking_preservation_rank1.json) |

The informativeness chart is a predictive-information ranking, not a cost
ranking. Its sample-count metadata does not measure runtime, memory, annotation,
access, licensing, acquisition, or dollars.

## Supplementary figures

These figures retain unique robustness, deployment, inventory, or method detail
without duplicating the canonical narrative.

| Figure | What it shows | Source artifact |
|---|---|---|
| [PathoPress publication summary](pathopress_hero_rank1.png) | combined observation/completion, compression, ranking, and uncertainty overview | [`publication_hero_summary.json`](../experiments/publication_hero_summary.json) |
| [Soft-Impute rank sweep](soft_impute_rank_sweep.png) | raw/logit MedAE and MedAPE rank sensitivity | [`soft_impute_rank_sweep_results.json`](../experiments/soft_impute_rank_sweep_results.json) |
| [Method grid](method_comparison_grid.png) | seven transforms × 12 classical method families | [`method_comparison/results.json`](../experiments/method_comparison/results.json) |
| [Probe compression](probe_compression_curves_rank1.png) | any/proxy-feasible/random/held-out/pruned/ranking tracks | [`probe_compression_rank1.json`](../experiments/probe_compression_rank1.json) |
| [Confidence calibration](confidence_calibration_rank1.png) | risk correlation, retention, strata, conformal intervals | [`confidence_calibration_rank1.json`](../experiments/confidence_calibration_rank1.json) |
| [Unseen-model confidence](new_model_confidence_rank1.png) | empirical risk–coverage, coverage–width, suite coverage, abstention support | [`new_model_confidence_rank1.json`](../experiments/new_model_confidence_rank1.json) |
| [Temporal deployment](temporal_deployment_rank1.png) | seven 2025 target releases with prior-only training | [`temporal_deployment_rank1.json`](../experiments/temporal_deployment_rank1.json) |
| [Metadata overview](pathopress_metadata_overview.png) | release/source/category coverage and inventory | [`publication_metadata_summary.json`](../experiments/publication_metadata_summary.json) |
| [Evaluation cost evidence](evaluation_cost_evidence_coverage.png) | source coverage, missingness, and explicitly non-monetary pre-error feasibility strata | [`evaluation_cost_evidence.json`](../data/evaluation_cost_evidence.json) |

Matched rank-1 error is 3.134532 MAE / 1.609006 MedAE. All-known greedy
scorecard MedAE is 1.397334 and 1.213706 at five and ten probes; hidden-only is
1.548536 and 1.493709. These are normalized-score retrospective quantities,
not clinical accuracy or confidence intervals.

## Structure and probe diagnostics

- [Best-neighbor correlations](benchmark_best_neighbor_correlations.png):
  nearest-neighbor correlation support for retained evaluations.
- [Probe-overlay correlation MDS](benchmark_correlation_mds_probes.png):
  classical MDS of the pairwise correlation-distance representation with the
  selected probes overlaid.
- [Legacy probe selection](probe_selection_rank1.png): direct upstream-style
  all-known and isolated held-out-model scorecard curves plus median absolute
  model-row-average error, explicitly separated by protocol.

The current pre-error feasibility curve uses 25 image/patch classification
protocols selected without outcome errors. It is not measured cost. The
checked-in greedy curves are complete through `k=10`; all-known random baselines
cover unrestricted `k=30` and every feasibility candidate through `k=25`.
Held-out and ranking random controls cover `k=10`. Exact `C(25,5)` and `C(30,5)`
MedAE searches were completed only for the historical 59 × 168 snapshot. They
are not displayed or claimed for the current 59 × 187 BenchPress-style hero.

## Predictability and error factors

- [Benchmark predictability](benchmark_predictability_rank1.png) and
  [model predictability](model_predictability_rank1.png) summarize the 10,007
  raw hide-half predictions from the historical 168-evaluation analysis over
  59 supported models.
- [Benchmark factors](predictability_factors_benchmark_rank1.png) and
  [model factors](predictability_factors_model_rank1.png) are headline Section
  6 panels.
- `predictability_factors_*_appendix_rank1` files contain the complete
  hypothesis panels; `predictability_factors_rank1` is the combined overview.

Factor associations are correlational and use variable-specific denominators.
For example, parameter count is present for 42 of 59 model-error rows; provider
and family are present for 57/59.

## Regeneration

Install the research dependencies first; the extra includes PyTorch for the
optional MLP grid unit.

```bash
python3 -m pip install -e '.[research]'
python3 scripts/plot_benchpress_style.py
python3 scripts/plot_method_comparison.py
python3 scripts/plot_structure_analysis.py
python3 scripts/plot_probe_compression.py
python3 scripts/plot_ranking_preservation.py
python3 scripts/plot_confidence_calibration.py
python3 scripts/plot_new_model_confidence.py
python3 scripts/plot_temporal_deployment.py
python3 scripts/plot_prediction_error_factors.py
python3 scripts/plot_publication_hero.py
python3 scripts/plot_benchpress_style_hero.py
python3 scripts/plot_probe_dual_objective.py
python3 scripts/plot_metadata_overview.py
python3 scripts/plot_evaluation_cost_evidence.py
```

The exact experiment commands and local-cache boundaries are in
[the experiment index](../experiments/README.md).
