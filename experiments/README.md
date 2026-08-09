# Experiment and artifact index

All commands below run from the repository root. Install the research stack
before regenerating analyses:

```bash
python3 -m pip install -e '.[research]'
# Add `mlp` only when running the optional PyTorch method-grid units:
python3 -m pip install -e '.[research,mlp]'
```

## Artifact storage forms

Two artifacts are stored in a compressed-but-lossless form so the repository
stays inside GitHub's 100 MB per-file hard limit. Neither change alters a single
recorded value.

- `probe_compression_rank1.json` (108 MB → 26 MB) stores the fold-invariant LOFO
  curves — `candidate_ids`, `candidate_indices`, `all_known_greedy_medae`,
  `all_known_greedy_medape`, `all_known_random` — once per candidate mode
  instead of repeating them byte-identically inside all 34 folds, and is
  serialised without indentation. The hoisted keys are named in
  `curves.<mode>._fold_invariant_keys`. Read it with
  `pathopress.probe_compression.load_probe_compression`, which restores the
  fully materialised per-fold payload; `tests/test_probe_compression_storage.py`
  pins the round trip. A plain `json.load` still parses the file but leaves the
  fold-invariant curves hoisted.
- `../outputs/probe_compression_selected_raw_rank1.csv.gz` (397 MB → 49 MB) is
  the same per-cell selected-probe stream, gzipped with `mtime=0` so repeated
  runs are byte-identical. This matches the pre-existing
  `probe_compression_random_all_known_raw_rank1.csv.gz`. Both are written and
  read transparently by `run_probe_compression.py` and
  `scripts/plot_probe_dual_objective.py`; regenerate them with
  `PYTHONPATH=src python3 experiments/run_probe_compression.py` (a multi-hour
  run — the committed artifacts exist so figures and the matched-cell replay can
  be reproduced without it).

The fixed substrate is 59 models × 187 protocol-level evaluations with 2,122
observations (19.2332% density). The source registry contains 4,013 score rows:
3,952 retained primary rows, 52 analysis-ineligible rows, and nine reported-
external rows. The supported analysis columns span Patho-Bench (122), EVA (15),
THUNDER (16), H-Optimus-1 report (10), HEST (18), and PathoROB (6).

## Shared substrate and selected predictor

```bash
PYTHONPATH=src python3 scripts/build_shared_artifacts.py
PYTHONPATH=src python3 experiments/run_benchpress_style.py
PYTHONPATH=src python3 experiments/run_soft_impute_rank_sweep.py \
  --workers 8 --blas-threads 1
PYTHONPATH=src python3 experiments/run_validation.py
```

- [shared manifest](shared_artifacts_manifest.json), [matrix](analysis_matrix.npz),
  and [folds](folds_s10_f3_bs42.json) fix ordered identities, filters, hashes,
  and ten seeds × three folds.
- [matched CV](benchpress_style_results.json) selects rank-1 bias ALS at
  3.134532 MAE / 1.609006 MedAE over 21,181 supported predictions; column
  median is 4.151756/2.400000. Thirty-nine held targets whose columns become
  empty in-fold are explicitly recorded as unsupported.
- [Soft-Impute](soft_impute_rank_sweep_results.json) reproduces the separate
  raw/logit rank-discovery algorithm; both tracks choose rank 1. Its 600
  transform/rank/fold jobs checkpoint atomically under the ignored
  `soft_impute_rank_sweep_checkpoints/` directory and resume by default.
  Checkpoint identities bind score, ordered matrix, folds, numerical config,
  and implementation hashes; `--merge-only` verifies and merges a complete
  compatible cache without fitting. Keep `--blas-threads 1` when using
  multiple workers to avoid BLAS oversubscription.
- [pathology stress tests](results.json) give rank-1 random-cell
  2.937385/1.568476, suite-block 5.788534/3.599615, and pooled sparse-new-model
  3.285791/1.753653 MAE/MedAE. In the tested rank-1-through-6 sweep, suite-block
  prefers rank 6 at 5.099363/3.203260. Sparse-new-model rank 1 has the best MAE;
  rank 6 has a marginally lower MedAE (1.751528 versus 1.753653). These are
  stress diagnostics, not the primary matched-CV rank selector.

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
```

[The manifest](method_comparison/manifest.json) records 343 completed, zero
missing, and zero unsupported units. The methods are benchmark/model means,
benchmark/model KNN, benchmark/model regression, Soft-Impute, bias ALS, NMF,
PMF, nuclear norm, and MLP over identity/log/logit/asinh/sqrt/probit/quantile
transforms. The best MedAPE row is logit BenchReg at 1.9023 with 68.3% coverage;
coverage-filtered results must not be compared as though they predicted every
held-out cell.

The NPZ prediction shards are rebuildable local cache and are not retained.
Compact [results](method_comparison/results.json), [top methods](method_comparison/top_methods.md),
and the manifest are the tracked merge products.

## Probe compression and ranking

```bash
PYTHONPATH=src python3 experiments/run_probe_selection.py
PYTHONPATH=src python3 experiments/run_probe_compression.py
PYTHONPATH=src python3 experiments/build_probe_pruning.py
# For a newly generated schema-v2 matrix, first run all declared residues with
# run_probe_exhaustive_v2.py. Then certify the two explicit run directories:
# Replace NEW_SCORE_SHA12 in both values with the first 12 hex characters of
# the refreshed data/scores.csv SHA-256 before running these commands.
CHEAP_RUN=experiments/probe_exhaustive_runs/cheap25_medae_k5_mNEW_SCORE_SHA12
PRUNED_RUN=experiments/probe_exhaustive_runs/pruned30_medae_k5_mNEW_SCORE_SHA12
PYTHONPATH=src:experiments python3 experiments/verify_fast_rank1.py
FAST_LIBRARY=$(python3 -c 'import json; print(json.load(open("experiments/probe_exhaustive_fast_equivalence_v2.json"))["inputs"]["library_path"])')
for wave in $(seq 0 9); do
  for shard in $(seq 0 7); do
    PYTHONPATH=src:experiments python3 experiments/run_probe_exhaustive_v2.py run-shard \
      --candidate-allowlist data/low_friction_allowlist_v2_top25.json \
      --k 5 --metric medae --num-waves 10 --wave-index "$wave" \
      --num-shards 8 --shard-index "$shard" \
      --fast-library "$FAST_LIBRARY" \
      --fast-equivalence experiments/probe_exhaustive_fast_equivalence_v2.json \
      --out-dir "$CHEAP_RUN"
  done
done
for wave in $(seq 0 19); do
  PYTHONPATH=src:experiments python3 experiments/run_probe_exhaustive_v2.py run-shard \
    --candidate-allowlist data/error_informed_probe_allowlist_rank1_top30.json \
    --k 5 --metric medae --num-waves 20 --wave-index "$wave" \
    --num-shards 1 --shard-index 0 \
    --fast-library "$FAST_LIBRARY" \
    --fast-equivalence experiments/probe_exhaustive_fast_equivalence_v2.json \
    --out-dir "$PRUNED_RUN"
done
PYTHONPATH=src:experiments python3 experiments/validate_probe_exhaustive_chunks.py \
  "$CHEAP_RUN" "$PRUNED_RUN"
PYTHONPATH=src:experiments python3 experiments/run_probe_exhaustive_v2.py merge \
  --out-dir "$CHEAP_RUN" --top-n 1001 \
  --integrity-manifest experiments/probe_exhaustive_integrity_manifest.json
PYTHONPATH=src:experiments python3 experiments/run_probe_exhaustive_v2.py merge \
  --out-dir "$PRUNED_RUN" --top-n 1001 \
  --integrity-manifest experiments/probe_exhaustive_integrity_manifest.json
PYTHONPATH=src:experiments python3 experiments/validate_probe_exhaustive_merged.py \
  "$CHEAP_RUN" "$PRUNED_RUN"
PYTHONPATH=src:experiments python3 experiments/validate_probe_exhaustive_top.py \
  "$CHEAP_RUN" "$PRUNED_RUN"
PYTHONPATH=src:experiments python3 experiments/build_probe_exhaustive_summary.py \
  --cheap-run "$CHEAP_RUN" --pruned-run "$PRUNED_RUN" \
  --fast-equivalence experiments/probe_exhaustive_fast_equivalence_v2.json
PYTHONPATH=src python3 experiments/run_ranking_preservation.py
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
  contains all expanded curves, 10×10 random ranking baselines, 111,920 selected
  prediction rows, and 1,167,100 random-prefix prediction rows.
  Its `curves.*.pairwise_margin=2` values are ancillary score-reconstruction
  diagnostics; only `ranking_aware` is the dedicated margin-5 ranking objective.
- [Top-30 pruning](probe_pruning_rank1_top30.json) uses all ten source MedAE
  greedy contexts and exact normalized-rank aggregation.
- No exhaustive choose-five search has been run for the current 59×187 scores.
  The 25-task universe remains a pre-error pipeline proxy, not a measured-cost
  set. New native searches use `run_probe_exhaustive_v2.py` with
  `fast_rank1_v2.cpp`: schema-v2 chunk
  configs bind the runner, native library, equivalence evidence, compiler,
  flags, platform, and full candidate identities. The verifier requires at
  least 32 unique scalar/native comparisons under fixed `1e-10` cell and
  `1e-11` metric caps, builds into a private content-addressed directory, and
  the runner loads a no-follow staged inode through `/proc/self/fd` for the
  lifetime of one reusable worker pool. Regenerate the host-bound evidence with
  `PYTHONPATH=src:experiments python3 experiments/verify_fast_rank1.py` before
  starting a new v2 native run.
- The three validators are schema-v2-only and require explicit run directories.
- [Ranking](ranking_preservation_rank1.json) is rebuilt directly from the current
  compression artifact's dedicated margin-5 tracks. It reports unrestricted and
  25-task all-known/random trajectories plus held-out-model validation through
  `k=10`; unrestricted all-known k10 accuracy is 0.877976.

All-known greedy MedAE is 1.397334/1.213706 at five/ten probes; hidden-only is
1.548536/1.493709. The 25-task feasibility allowlist is an input/label pipeline
proxy, not measured compute, access, or licensing cost. The
[BenchPress-style summary](benchpress_style_hero_summary.json) records the
retrospective all-known scope, exact revealed probes, pathology rank-1
adaptation, and absence of a current exhaustive-search claim. The
[dual-objective table](../outputs/probe_dual_objective_rank1.csv) reports mean
reported-score prediction under a leave-one-family-out protocol with 34
family folds; all 59 models are held out exactly once, with 1 validation
model per fold at the median (min 1, max 7) and 58 training models. It
includes exact revealed probes and supported hidden
predictions; no held-out `k=0` or random model-mean control is available.

## Matched-cell LOFO comparison

```bash
PYTHONPATH=src python3 scripts/replay_lofo_matched_cells.py
```

Each probe-selection arm hides a different set of cells, so the per-arm MedAEs in
[probe compression](probe_compression_rank1.json) and
[legacy selection](probe_selection_results_rank1.json) use different denominators
and are not directly comparable to one another. This replay re-derives the
held-out per-cell predictions from the recorded LOFO probe sets and scores every
arm on one matched cell set: per fold and per depth `k`, it excludes the union of
the cells revealed by the greedy prefix and by all ten random repeats. At `k=4`
that excludes 486 of 2,122 cells and leaves 1,636 matched.

- [Matched-cell results](lofo_matched_cells_rank1.json) is the reproduction path
  for every corrected LOFO number. At `k=4`: greedy 1.8781, `k=0` 2.6524, random
  2.6013 under the median-over-340-fold-×-repeat-MedAEs convention and 2.6260
  under median-of-fold-medians. The two random conventions differ at every `k`,
  so any quoted random value must name its convention.
- Paired per-fold results are the well-supported part: greedy beats `k=0` in 18 of
  34 folds (Wilcoxon `p = 0.0088`) and random in 22 of 34 (`p = 0.0151`). The
  bootstrap-over-folds 95% CI on the reduction is [2.8%, 58.7%] versus `k=0` and
  [3.4%, 53.5%] versus random, so the 29.2% point estimate must not be quoted to
  three significant figures.
- Per-column skill is null: 86 of 174 scored columns (49.4%, CI [42.0%, 56.9%])
  have a matched greedy `k=4` MedAE below their matched `k=0` MedAE. Including
  all 187 columns gives 94/187 (50.3%, CI [43.3%, 57.2%]); 8 of the 12
  noise-floor-excluded columns (`SKILL_NOISE_FLOOR_DISPERSION = 0.5`) are
  positive, and 1 further column has no matched cells.
- The `allowlist_arm_by_k` block records the 25-task negative result: at `k=4` on
  its own 1,581 matched cells, allowlist greedy is 1.9951 against 1.7234 for
  greedy over any candidate.
- The artifact is hash-bound through `provenance`, which pins the score CSV, the
  replay script, and the selection and compression artifacts it reads.

## Confidence and time

```bash
PYTHONPATH=src python3 experiments/run_confidence_calibration.py
PYTHONPATH=src python3 experiments/run_new_model_confidence.py
PYTHONPATH=src python3 experiments/run_temporal_deployment.py
```

- [Confidence](confidence_calibration_rank1.json) contains 21,181 cross-fitted
  cells, risk–coverage curves, strata, and leave-fold-out conformal results.
  Structural support has Spearman 0.602190 and nominal-90% coverage 0.899816.
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
  scores over ten seeds. Its parity MedAE includes exact revealed cells plus
  supported hidden predictions. This is a small retrospective cohort.

## Publication and product artifacts

```bash
python3 scripts/build_evaluation_cost_evidence.py
python3 scripts/plot_benchpress_style.py
python3 scripts/plot_benchpress_style_hero.py
python3 scripts/plot_probe_dual_objective.py
python3 scripts/plot_temporal_deployment.py
PYTHONPATH=src python3 scripts/build_public_release.py
PYTHONPATH=src python3 scripts/build_hf_dataset.py --parquet auto
PYTHONPATH=src python3 scripts/publish_hf_dataset.py  # validation + dry run; no network
```

The four public figure pairs and their summaries are compact derivatives of
experiment artifacts. Cost evidence remains a registry and prose audit, not a
chart. The public build produces a hash-verified all/paper/wide
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
[audit note](../docs/evaluation-cost-evidence.md) cover all 187 retained
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
