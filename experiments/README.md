# Experiment and artifact index

All commands below run from the repository root. Install the research stack
before regenerating analyses:

```bash
python3 -m pip install -e '.[research]'
```

The fixed substrate is 59 models × 165 protocol-level evaluations with 1,967
observations (20.2054% density). The accepted source rows are Patho-Bench 896,
EVA 265, HEST 234, THUNDER 512, and PathoROB 60 after nine external-report rows
are excluded by the fixed paper filter. The registry still retains all 69
PathoROB score rows and 1,976 total rows.

## Shared substrate and selected predictor

```bash
PYTHONPATH=src python3 scripts/build_shared_artifacts.py
PYTHONPATH=src python3 experiments/run_benchpress_style.py
PYTHONPATH=src python3 experiments/run_soft_impute_rank_sweep.py
PYTHONPATH=src python3 experiments/run_validation.py
```

- [shared manifest](shared_artifacts_manifest.json), [matrix](analysis_matrix.npz),
  and [folds](folds_s10_f3_bs42.json) fix ordered identities, filters, hashes,
  and ten seeds × three folds.
- [matched CV](benchpress_style_results.json) selects rank-1 bias ALS at
  3.005264 MAE / 1.603026 MedAE over 19,670 predictions; column median is
  4.092133/2.477500.
- [Soft-Impute](soft_impute_rank_sweep_results.json) reproduces the separate
  raw/logit rank-discovery algorithm; both tracks choose rank 1.
- [pathology stress tests](results.json) give rank-1 random-cell
  2.834996/1.526795, suite-block 5.612789/3.525174, and pooled sparse-new-model
  3.190380/1.817465 MAE/MedAE. Suite-block prefers rank 5 overall at
  4.952972/3.055638.

## Classical method comparison

The grid contains 343 deterministic units across seven transforms and 12
method families. Units write compressed local NPZ predictions and merge only
when every expected shard validates.

```bash
PYTHONPATH=src python3 experiments/run_method_comparison.py --prepare-folds
PYTHONPATH=src python3 experiments/run_method_comparison.py --list-shards
PYTHONPATH=src python3 experiments/run_method_comparison.py \
  --run-range 0 343 --workers 8
PYTHONPATH=src python3 experiments/run_method_comparison.py --merge
python3 scripts/plot_method_comparison.py
```

[The manifest](method_comparison/manifest.json) records 343 completed, zero
missing, and zero unsupported units. The methods are benchmark/model means,
benchmark/model KNN, benchmark/model regression, Soft-Impute, bias ALS, NMF,
PMF, nuclear norm, and MLP over identity/log/logit/asinh/sqrt/probit/quantile
transforms. The best MedAPE row is log BenchReg at 1.8144 with 71.0% coverage;
coverage-filtered results must not be compared as though they predicted every
held-out cell.

The 343 NPZ caches under `method_comparison/predictions/` total about 439 MiB
and are narrowly Git-ignored. Compact [results](method_comparison/results.json),
[top methods](method_comparison/top_methods.md), manifest, and figure are the
tracked merge products.

## Matrix structure

```bash
PYTHONPATH=src python3 experiments/run_structure_analysis.py
python3 scripts/plot_structure_analysis.py
python3 scripts/build_publication_tables.py
```

[The structure manifest](structure_analysis/manifest.json) links threshold
sensitivity, complete-submatrix stable rank/SVD, pairwise correlations, and
classical MDS. The largest complete block is 32 × 16 with stable rank 1.431046;
the first one/two components explain 69.8789%/87.5715%. All 165 evaluations
have a neighbor sharing at least five models, with median best absolute
correlation 0.918881. Publication CSV/Markdown/LaTeX tables are under
[`outputs/tables/`](../outputs/tables/).

## Probe compression and ranking

```bash
PYTHONPATH=src python3 experiments/run_probe_selection.py
PYTHONPATH=src python3 experiments/run_probe_compression.py
PYTHONPATH=src python3 experiments/run_probe_exhaustive.py --workers 8
PYTHONPATH=src python3 experiments/run_ranking_preservation.py
python3 scripts/plot_probe_compression.py
python3 scripts/plot_ranking_preservation.py
```

- [Legacy selection](probe_selection_results_rank1.json) preserves the direct
  all-known/held-out comparison.
- [Compression](probe_compression_rank1.json) adds any-evaluation, pre-error
  low-friction proxy, MedAE/MedAPE, nested random, held-out-row, pruned, and
  ranking-aware tracks through tractable bounds.
- [Exhaustive search](probe_exhaustive_rank1.json) enumerates every subset of
  the four-item pre-error allowlist for `k=1..4` and all 66 `k=2` subsets of a
  12-candidate error-informed space. “Exhaustive” applies only to these declared
  spaces, not all subsets of 165 evaluations.
- [Ranking](ranking_preservation_rank1.json) reports pairwise accuracy at
  margins 0/1/2/5 and top-set recovery at 10/20/30% from OOF predictions.

All-known greedy MedAE is 1.481124/1.196456 at five/ten probes; hidden-only is
1.612112/1.539134. The four-item feasibility allowlist is a protocol-metadata
proxy, not measured compute, access, or licensing cost.

## Confidence, time, and error factors

```bash
PYTHONPATH=src python3 experiments/run_confidence_calibration.py
PYTHONPATH=src python3 experiments/run_temporal_deployment.py
PYTHONPATH=src python3 experiments/run_predictability.py
PYTHONPATH=src python3 experiments/run_error_analysis.py
PYTHONPATH=src python3 experiments/run_prediction_error_factors.py --workers 8
python3 scripts/build_prediction_error_factor_tables.py
```

- [Confidence](confidence_calibration_rank1.json) contains 19,670 cross-fitted
  cells, risk–coverage curves, strata, and leave-fold-out conformal results.
  Structural support has Spearman 0.598017 and nominal-90% coverage 0.899949.
  Its six full-coverage ALS/Soft-Impute variants are less diverse than the
  upstream top-12 stack.
- [Deployment intervals](deployment_confidence_rank1.json) are separately
  suite-calibrated for existing supported rows. They do not cover genuinely
  new models.
- [Temporal deployment](temporal_deployment_rank1.json) evaluates seven
  verified 2025 targets using strictly earlier models and 1/5/10 revealed
  scores over ten seeds. This is a small retrospective cohort.
- [Predictability](predictability_results_rank1.json) contains 9,605 raw
  predictions, 165 evaluation summaries, and 53 model summaries.
- [Error factors](prediction_error_factors_rank1.json) merge 11,535 compact
  records and nine intervention groups. Metadata are incomplete for some
  variables (for example parameter count is known for 39/53 supported models),
  so results are correlational and denominators vary.

The factor experiment's 5,795 local unit-cache files (~21 MiB) and 279,487-row
raw prediction CSV (~29 MiB) are Git-ignored. The manifest, merged records,
tables, and figures remain tracked.

## LLM baseline contract

```bash
PYTHONPATH=src python3 experiments/run_llm_baseline.py all-mock
# After an authorized external provider creates contract-compliant JSONL:
PYTHONPATH=src python3 experiments/run_llm_baseline.py merge-real \
  --real-responses /path/to/provider_responses.jsonl
```

The runner prepares named/blind matrix and named/blind five-shot requests,
validates schemas, supports deterministic mock contract tests, and merges real
provider responses. It implements no provider client and makes no network
calls. [Real-run status](llm_baseline/real_run_status.json) is `unrun` for all
four conditions. [Mock metrics](llm_baseline/merged_mock_metrics.json) are
explicitly `headline_eligible=false` and are not scientific comparisons.

## Publication and product artifacts

```bash
PYTHONPATH=src python3 scripts/build_publication_tables.py
python3 scripts/plot_publication_hero.py
python3 scripts/plot_metadata_overview.py
PYTHONPATH=src python3 scripts/build_public_release.py
```

The publication summaries, tables, and figures are compact derivatives of
experiment artifacts. The public build produces a hash-verified all/paper/wide
export and `website/data.json`; it performs no upload or deployment. See the
[figure gallery](../figures/README.md), [export README](../exports/pathopress_public/README.md),
and [static-site README](../website/README.md).

## Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/check_artifact_freshness.py check
git diff --check
node --check website/app.js
```

Direct upstream parity additionally requires a local checkout of the pinned
BenchPress commit; see [the parity note](../docs/benchpress-parity.md).
