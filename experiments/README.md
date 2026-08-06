# Experiment and artifact index

All commands below run from the repository root. Install the research stack
before regenerating analyses:

```bash
python3 -m pip install -e '.[research]'
```

The fixed substrate is 59 models × 168 protocol-level evaluations with 2,027
observations (20.4500% density). The accepted source rows are Patho-Bench 896,
EVA 265, HEST 234, THUNDER 512, and PathoROB 120. The registry additionally
retains 40 analysis-ineligible signed APD rows and nine external-report rows,
for 169 PathoROB rows and 2,076 total rows.

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
  3.222008 MAE / 1.647585 MedAE over 20,270 predictions; column median is
  4.275274/2.500000.
- [Soft-Impute](soft_impute_rank_sweep_results.json) reproduces the separate
  raw/logit rank-discovery algorithm; both tracks choose rank 1.
- [pathology stress tests](results.json) give rank-1 random-cell
  3.050584/1.603529, suite-block 5.688229/3.537207, and pooled sparse-new-model
  3.503746/1.894207 MAE/MedAE. In the tested rank-1-through-6 sweep, suite-block
  prefers rank 6 at 5.093822/3.175723; sparse-new-model also has its lowest
  tested pair at rank 6, 3.351160/1.873395.

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
transforms. The best MedAPE row is logit BenchReg at 1.9077 with 71.9% coverage;
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
the first one/two components explain 69.8789%/87.5715%. All 168 evaluations
have a neighbor sharing at least five models, with median best absolute
correlation 0.916362. Publication CSV/Markdown/LaTeX tables are under
[`outputs/tables/`](../outputs/tables/).

## Probe compression and ranking

```bash
PYTHONPATH=src python3 experiments/run_probe_selection.py
PYTHONPATH=src python3 experiments/run_probe_compression.py
PYTHONPATH=src python3 experiments/build_probe_pruning.py
# Validate all 800 exact-search chunks, merged order, and scalar top candidates:
PYTHONPATH=src python3 experiments/validate_probe_exhaustive_chunks.py
PYTHONPATH=src python3 experiments/validate_probe_exhaustive_merged.py
PYTHONPATH=src python3 experiments/validate_probe_exhaustive_top.py
PYTHONPATH=src python3 experiments/build_probe_exhaustive_summary.py
PYTHONPATH=src python3 experiments/run_ranking_preservation.py
python3 scripts/plot_probe_compression.py
python3 scripts/plot_ranking_preservation.py
python3 scripts/plot_benchpress_style_hero.py
python3 scripts/plot_probe_dual_objective.py
```

- [Legacy selection](probe_selection_results_rank1.json) preserves the direct
  all-known/held-out comparison.
- The compression runner uses any-evaluation and 25-task pre-error proxy
  candidates, MedAE/MedAPE, nested random, held-out rows, and exact upstream
  margin-5 ranking budgets through `k=10`. Its all-known score-reconstruction
  random baseline is configured through `k=30` for unrestricted candidates and
  through the full `k=25` feasibility-proxy universe; held-out and ranking
  random controls remain at the upstream-comparable `k=10`. Per-cell all-known
  random predictions stream to a deterministic gzip CSV. The canonical artifact
  contains all expanded curves, 10×10 random ranking baselines, 107,360 selected
  prediction rows, and 1,114,850 random-prefix prediction rows.
  Its `curves.*.pairwise_margin=2` values are ancillary score-reconstruction
  diagnostics; only `ranking_aware` is the dedicated margin-5 ranking objective.
- [Top-30 pruning](probe_pruning_rank1_top30.json) uses all ten source MedAE
  greedy contexts and exact normalized-rank aggregation.
- [Exhaustive status](probe_exhaustive_execution_status.json) records completed
  `C(25,5)=53,130` and `C(30,5)=142,506` MedAE searches. All 195,636 candidates
  across 800 chunks were checked record-by-record, the merged top order was
  reconstructed from raw chunks, and the leading candidates were recomputed
  with the scalar reference implementation. The [compact result](probe_exhaustive_rank1.json)
  contains the certified top lists. Exactness is limited to each declared
  candidate universe; the 25-task universe remains a pre-error pipeline proxy,
  not a measured-cost set.
- The frozen legacy-v1 audit continues to bind
  `run_probe_exhaustive.py`, `fast_rank1.cpp`, and
  `probe_exhaustive_fast_equivalence.json` byte-for-byte. New native searches
  use `run_probe_exhaustive_v2.py` with `fast_rank1_v2.cpp`: schema-v2 chunk
  configs bind the runner, native library, equivalence evidence, compiler,
  flags, platform, and full candidate identities. The verifier requires at
  least 32 unique scalar/native comparisons under fixed `1e-10` cell and
  `1e-11` metric caps, builds into a private content-addressed directory, and
  the runner loads a no-follow staged inode through `/proc/self/fd` for the
  lifetime of one reusable worker pool. Regenerate the host-bound evidence with
  `PYTHONPATH=src:experiments python3 experiments/verify_fast_rank1.py` before
  starting a new v2 native run.
- [Ranking](ranking_preservation_rank1.json) reports pairwise accuracy at
  margins 0/1/2/5 and top-set recovery at 10/20/30% from OOF predictions.

All-known greedy MedAE is 1.474879/1.270529 at five/ten probes; hidden-only is
1.637639/1.538607. The 25-task feasibility allowlist is an input/label pipeline
proxy, not measured compute, access, or licensing cost. The faithful
[BenchPress-style summary](benchpress_style_hero_summary.json) separates exact
masking/search budgets, the pathology rank-1 adaptation, and the two completed
candidate-bounded exact searches. The [dual-objective table](../outputs/probe_dual_objective_rank1.csv)
reports model-average prediction error without pretending it was optimized.

## Confidence, time, and error factors

```bash
PYTHONPATH=src python3 experiments/run_confidence_calibration.py
PYTHONPATH=src python3 experiments/run_new_model_confidence.py
PYTHONPATH=src python3 experiments/run_temporal_deployment.py
PYTHONPATH=src python3 experiments/run_predictability.py
PYTHONPATH=src python3 experiments/run_error_analysis.py
PYTHONPATH=src python3 experiments/run_prediction_error_factors.py --workers 8
python3 scripts/build_prediction_error_factor_tables.py
```

- [Confidence](confidence_calibration_rank1.json) contains 20,270 cross-fitted
  cells, risk–coverage curves, strata, and leave-fold-out conformal results.
  Structural support has Spearman 0.606612 and nominal-90% coverage 0.899803.
  Its prediction-cache-first generator contract matches upstream: three Logit
  Bias-ALS lambda variants plus the top twelve full-coverage Section-4
  alternatives. Per-cell cross-fitted P(|error| <= 10 normalized points), fold
  calibration metadata, Brier/log-loss/ECE, and the full deploy mapping are
  serialized; see [the protocol](../docs/confidence-trust.md).
- [Deployment intervals](deployment_confidence_rank1.json) are separately
  hybrid-risk calibrated for existing supported rows and include the trust
  mapping. Genuinely new models use a separate interval population and abstain
  from this trust probability.
- [Unseen-model confidence](new_model_confidence_rank1.json) and its
  [30,992-row audit](new_model_confidence_predictions_rank1.csv) use only
  leave-one-model-out sparse-probe and temporal residuals. Nested target-group
  exclusion prevents hidden-score leakage; unsupported contexts abstain.
- [Temporal deployment](temporal_deployment_rank1.json) evaluates seven
  verified 2025 targets using strictly earlier models and 1/5/10 revealed
  scores over ten seeds. This is a small retrospective cohort.
- [Predictability](predictability_results_rank1.json) contains 10,007 raw
  predictions, 168 evaluation summaries, and 59 model summaries.
- [Error factors](prediction_error_factors_rank1.json) merge 11,990 compact
  records and nine intervention groups. Metadata are incomplete for some
  variables (for example parameter count is known for 42/59 model-error rows),
  so results are correlational and denominators vary.

The factor experiment's 6,030 local unit-cache files (~22 MiB) and 289,681-row
raw prediction CSV (~30 MiB) are Git-ignored. The manifest, merged records,
tables, and figures remain tracked.

## LLM baseline contract

```bash
PYTHONPATH=src python3 experiments/run_llm_baseline.py prepare --scope full
PYTHONPATH=src python3 experiments/run_llm_baseline.py all-mock --scope smoke
# After an authorized external provider creates raw provider-neutral JSONL:
PYTHONPATH=src python3 experiments/run_llm_baseline.py import-real --scope full \
  --raw-responses /path/to/provider_responses.jsonl
```

The full pack contains 1,990 requests and 81,080 target predictions across all
30 folds: named/blind full-matrix zero-shot and named/blind five-shot. It is
stored as 20 deterministic gzip JSONL shards with compressed and canonical
uncompressed hashes. The separate four-request [smoke fixture](llm_baseline_smoke/)
is explicitly `headline_eligible=false`. The runner implements no provider
client and reads no credentials; external cost remains unknown. See the
[protocol and import contract](../docs/llm-baseline.md).

## Publication and product artifacts

```bash
PYTHONPATH=src python3 scripts/build_publication_tables.py
python3 scripts/plot_publication_hero.py
python3 scripts/plot_metadata_overview.py
python3 scripts/build_evaluation_cost_evidence.py
python3 scripts/plot_evaluation_cost_evidence.py
PYTHONPATH=src python3 scripts/build_public_release.py
PYTHONPATH=src python3 scripts/build_hf_dataset.py --parquet auto
PYTHONPATH=src python3 scripts/publish_hf_dataset.py  # validation + dry run; no network
```

The publication summaries, tables, and figures are compact derivatives of
experiment artifacts. The public build produces a hash-verified all/paper/wide
export and `website/data.json`; it performs no upload or deployment. See the
[figure gallery](../figures/README.md), [export README](../exports/pathopress_public/README.md),
and [static-site README](../website/README.md).

The local Hugging Face build reproduces the pinned BenchPress maintenance table
names (`models`, `benchmarks`, `scores_all`, `scores_paper`, and the paper-wide
matrix), retains richer PathoPress all/paper tables, and writes a dataset card,
ordered logical schema, upstream-compatible metadata, and SHA-256 manifest.
With `pyarrow` installed (`pip install -e '.[hf]'`), `--parquet auto` emits
deterministic typed Parquet mirrors; `--parquet yes` fails closed if the backend
is unavailable. `publish_hf_dataset.py` is a local validation/dry run by
default. A network upload requires all three of `--upload`,
`--authorize-upload`, and a nonempty `HF_TOKEN`; no publication command in the
experiment inventory supplies those capabilities.

The [cost-evidence registry](../data/evaluation_cost_evidence.json) and
[audit note](../docs/evaluation-cost-evidence.md) cover all 168 retained
protocols. They preserve source configuration/count evidence and explicit
missingness; they do not impute a numeric evaluation cost.

## Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/check_artifact_freshness.py check
git diff --check
node --check website/app.js
```

Direct upstream parity additionally requires a local checkout of the pinned
BenchPress commit; see [the parity note](../docs/benchpress-parity.md).
