# Generated figure gallery

The public gallery contains exactly four result figures. PNG files are review
assets; matching PDFs are publication/vector assets. Plotters consume committed
JSON/CSV results and do not rerun the expensive experiments.

The fixed research matrix is 59 models × 187 protocol-level evaluations with
2,122 observed cells (19.2332% density).

| Figure | Evidence and scope | Source artifact |
|---|---|---|
| [Cell-level rank validation](benchpress_style_validation_rank1.png) | Rank sweep over 2,122 unique reported cells and 21,181 repeated held-out prediction instances from ten seeds × three folds. Other scores from the same model may remain visible; this is not model-level holdout. | [`benchpress_style_results.json`](../experiments/benchpress_style_results.json) |
| [BenchPress-style pathology hero](pathopress_benchpress_hero_rank1.png) | Retrospective all-known cell reconstruction. Revealed probes are exact, and selection/evaluation use the same model population. The 25-task track is a low-friction proxy, not measured cost. | [`benchpress_style_hero_summary.json`](../experiments/benchpress_style_hero_summary.json) |
| [Task utility and held-out mean prediction](probe_dual_objective_rank1.png) | Panel A is transductive single-task utility, not causal task importance. Panel B uses prefixes selected on 41 training models to predict the mean reported normalized score of 18 held-out models; revealed probe values are exact and supported hidden cells are predicted. No held-out `k=0` or random model-mean control is available. | [`probe_dual_objective_rank1.csv`](../outputs/probe_dual_objective_rank1.csv) |
| [Temporal deployment](temporal_deployment_rank1.png) | Seven 2025 target models trained from strictly prior releases. Each trajectory is the target-level median over ten probe seeds; MedAE includes `k` exact revealed cells plus supported hidden predictions. | [`temporal_deployment_rank1.json`](../experiments/temporal_deployment_rank1.json) |

The interaction rank is a fitted completion hyperparameter, not the literal
rank of the pathology data. The ranking, confidence, Soft-Impute, and cost
audits remain available as machine-readable experiment artifacts without
additional public charts. In particular, the cost registry reports no measured
runtime/dollar curve; sample count and the 25-task proxy are not cost models.

## Regeneration

Install the research dependencies, then run only these lightweight plotters:

```bash
python3 -m pip install -e '.[research]'
python3 scripts/plot_benchpress_style.py
python3 scripts/plot_benchpress_style_hero.py
python3 scripts/plot_probe_dual_objective.py
python3 scripts/plot_temporal_deployment.py
```

The experiment runners and local-cache boundaries are documented in the
[experiment index](../experiments/README.md).
