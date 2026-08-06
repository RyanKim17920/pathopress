# Generated figure gallery

Figures are result-first: plotting scripts read committed compact JSON/CSV/NPZ
artifacts and do not implicitly rerun expensive experiments. PNGs are review
assets; matching PDFs are publication/vector assets.

The fixed research matrix is 59 models × 168 protocol-level evaluations with
2,027 observed cells (20.4500% density) and 7,885 rank-1 point estimates.

## Main figures

| Figure | What it shows | Source artifact |
|---|---|---|
| [PathoPress hero](pathopress_hero_rank1.png) | observation/completion, probe compression, ranking, and uncertainty summary | [`publication_hero_summary.json`](../experiments/publication_hero_summary.json) |
| [BenchPress-style pathology hero](pathopress_benchpress_hero_rank1.png) | four target-cell examples plus random, most-predictive, and 25-task scorecard trajectories | [`benchpress_style_hero_summary.json`](../experiments/benchpress_style_hero_summary.json) |
| [BenchPress-style ranking](pathopress_benchpress_ranking_rank1.png) | random and greedy margin-5 pairwise recovery through ten probes | [`probe_compression_rank1.json`](../experiments/probe_compression_rank1.json) |
| [Dual probe objective](probe_dual_objective_rank1.png) | scorecard MedAE and separately evaluated model-average MedAE for the same selected sets | [`probe_dual_objective_rank1.csv`](../outputs/probe_dual_objective_rank1.csv) |
| [Observation matrix](matrix_observation_pattern.png) | reported versus missing cells | [`analysis_matrix.npz`](../experiments/analysis_matrix.npz) |
| [Completed matrix](matrix_completed_rank1.png) | reported values and translucent rank-1 estimates | [`imputations_rank1.csv`](../outputs/imputations_rank1.csv) |
| [Matched validation](benchpress_style_validation_rank1.png) | rank sweep, OOF parity, suite errors, error distribution | [`benchpress_style_results.json`](../experiments/benchpress_style_results.json) |
| [Soft-Impute rank sweep](soft_impute_rank_sweep.png) | raw/logit MedAE and MedAPE rank sensitivity | [`soft_impute_rank_sweep_results.json`](../experiments/soft_impute_rank_sweep_results.json) |
| [Method grid](method_comparison_grid.png) | seven transforms × 12 classical method families | [`method_comparison/results.json`](../experiments/method_comparison/results.json) |
| [Probe compression](probe_compression_curves_rank1.png) | any/proxy-feasible/random/held-out/pruned/ranking tracks | [`probe_compression_rank1.json`](../experiments/probe_compression_rank1.json) |
| [Ranking preservation](ranking_preservation_rank1.png) | pairwise margins and top-fraction recovery | [`ranking_preservation_rank1.json`](../experiments/ranking_preservation_rank1.json) |
| [Confidence calibration](confidence_calibration_rank1.png) | risk correlation, retention, strata, conformal intervals | [`confidence_calibration_rank1.json`](../experiments/confidence_calibration_rank1.json) |
| [Unseen-model confidence](new_model_confidence_rank1.png) | empirical risk–coverage, coverage–width, suite coverage, abstention support | [`new_model_confidence_rank1.json`](../experiments/new_model_confidence_rank1.json) |
| [Temporal deployment](temporal_deployment_rank1.png) | seven 2025 target releases with prior-only training | [`temporal_deployment_rank1.json`](../experiments/temporal_deployment_rank1.json) |
| [Metadata overview](pathopress_metadata_overview.png) | release/source/category coverage and inventory | [`publication_metadata_summary.json`](../experiments/publication_metadata_summary.json) |
| [Evaluation cost evidence](evaluation_cost_evidence_coverage.png) | source coverage, missingness, and explicitly non-monetary pre-error feasibility strata | [`evaluation_cost_evidence.json`](../data/evaluation_cost_evidence.json) |

After the next canonical-matrix refresh, `plot_metadata_overview.py` also emits
five separately named BenchPress-analogue pathology assets:
`pathopress_releases`, `pathopress_coverage`,
`pathopress_evaluation_mix`, `pathopress_observed_cells_by_family`, and
`pathopress_source_provenance` in both PNG and PDF. The first four preserve the
upstream denominators; provenance uses pathology-native primary-paper/report
and official-repository categories.

Matched rank-1 error is 3.222008 MAE / 1.647585 MedAE. All-known greedy
scorecard MedAE is 1.474879 and 1.270529 at five and ten probes; hidden-only is
1.637639 and 1.538607. These are normalized-score retrospective quantities,
not clinical accuracy or confidence intervals.

## Structure and probe diagnostics

- [Best-neighbor correlations](benchmark_best_neighbor_correlations.png): all
  168 evaluations have a neighbor sharing at least five models; median best
  absolute correlation is 0.916362.
- [Correlation MDS](benchmark_correlation_mds.png) and
  [probe-overlay MDS](benchmark_correlation_mds_probes.png): classical MDS of
  the pairwise correlation-distance representation.
- [Legacy probe selection](probe_selection_rank1.png): direct upstream-style
  all-known and isolated held-out-model scorecard curves plus literal-average
  diagnostic.
- [One-probe informativeness](probe_informativeness_rank1.png): first-step
  greedy utility with observed-model coverage.

The current pre-error feasibility curve uses 25 image/patch classification
protocols selected without outcome errors. It is not measured cost. The
checked-in greedy curves are complete through `k=10`; all-known random baselines
cover unrestricted `k=30` and every feasibility candidate through `k=25`.
Held-out and ranking random controls cover `k=10`. Exact `C(25,5)` and `C(30,5)`
MedAE searches are complete and displayed as stars in the BenchPress-style hero:
1.485944 for the 25-task proxy and 1.427339 for the error-informed 30-task set.
They are exact only inside those candidate universes and do not optimize the
separate ranking objective.

## Predictability and error factors

- [Benchmark predictability](benchmark_predictability_rank1.png) and
  [model predictability](model_predictability_rank1.png) summarize 10,007 raw
  hide-half predictions over 168 evaluations and 59 supported models.
- [Benchmark factors](predictability_factors_benchmark_rank1.png) and
  [model factors](predictability_factors_model_rank1.png) are headline Section
  6 panels.
- `predictability_factors_*_appendix_rank1` files contain the complete
  hypothesis panels; `predictability_factors_rank1` is the combined overview.

Factor associations are correlational and use variable-specific denominators.
For example, parameter count is present for 42 of 59 model-error rows; provider
and family are present for 57/59.

## LLM status figure

[LLM baseline status](llm_baseline_status.png) is a status/contract figure, not
a result comparison. All four real named/blind zero-shot matrix and five-shot conditions
remain `unrun`. Deterministic mock metrics only validate request, response, and
merge plumbing and are not headline eligible.

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
python3 scripts/plot_llm_baseline.py
python3 scripts/plot_evaluation_cost_evidence.py
```

The exact experiment commands and local-cache boundaries are in
[the experiment index](../experiments/README.md).
