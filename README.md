# PathoPress

Citation-backed score-matrix completion for pathology foundation-model benchmarks.

## Verdict

The BenchPress idea is applicable to pathology, but it is a **moderate-to-hard data and validation project**, not mainly an engineering port. A registry and auditable score matrix are realistic in 2–4 weeks; a useful completion model needs roughly 6–10 weeks of extraction, protocol normalization, and retrospective validation; a defensible public release is more plausibly a 3–6 month effort. The schedule is driven by score provenance and comparability, not the matrix-factorization code.

[Microsoft BenchPress](https://github.com/microsoft/benchpress) does not compress benchmark datasets or replace their runners. It collects published model × evaluation scores and predicts missing cells with calibrated low-rank matrix completion. Its paper-canonical matrix has 84 models, 133 evaluations, and 2,604 observed cells (23.3% filled). PathoPress tests whether pathology results have enough overlap and low-rank structure for the same strategy.

## What the initial audit found

The current inventory covers [Patho-Bench](https://github.com/mahmoodlab/patho-bench), [eva](https://github.com/kaiko-ai/eva), [THUNDER](https://github.com/MICS-Lab/thunder), [HEST](https://github.com/mahmoodlab/HEST), and [PathoROB](https://github.com/bifold-pathomics/PathoROB).

- Patho-Bench: 95 tasks from 33 sources.
- eva: 13 canonical pathology task identities in current code; its dataset documentation shows 14 when the `PANDASmall` variant is counted separately.
- THUNDER: 21 current task configurations (17 classification and 4 segmentation), versus 20 documented tasks and 16 datasets in the paper.
- HEST: 9 morphology-to-gene-expression tasks.
- PathoROB: 4 dataset configurations × 3 robustness metrics = 12 endpoints.
- Across Patho-Bench, eva, and THUNDER, deduplication yields **122 code-backed dataset × target × granularity identities**, or **121 using only documented tasks** and excluding THUNDER's code-present `STARC9` entry.

Only seven exact cross-suite task overlaps were found: six patch tasks shared by eva and THUNDER (`BACH`, `BRACS`, the selected four-class `BreakHis` task, `CRC-100K`, `MHIST`, and `PatchCamelyon`) and the full-slide PANDA ISUP task shared by Patho-Bench and eva. These are shared task identities, not automatically interchangeable score columns: preprocessing, splits, adaptation method, and metric still have to match.

The full evidence, inventory drift, feasibility gates, roadmap, and licensing risks are in [docs/feasibility.md](docs/feasibility.md).

## Current implementation

The repository contains an MVP for the numerical core:

- a citation-backed CSV score loader;
- iterative support filtering for sparse model × evaluation matrices;
- logit transform, per-evaluation standardization, and configurable bias-decomposed alternating least squares;
- random-cell holdout smoke validation; and
- `audit` and `validate` CLI commands.

The generated seed registry currently contains 5 suites, 150 evaluation endpoints, and 815 citation-backed score cells. The score pool covers HEST (234 cells), THUNDER (512), and PathoROB (69 cells from the three currently reported Robustness Index dataset columns); Patho-Bench and eva score extraction remains an expansion target. The generated artifacts are evidence metadata, not bundled benchmark images:

- [`data/suites.csv`](data/suites.csv) — suite-level scope and provenance;
- [`data/tasks.csv`](data/tasks.csv) — complete endpoint inventory;
- [`data/deduplication.csv`](data/deduplication.csv) — exact links and related-but-distinct tasks;
- [`data/model_aliases.csv`](data/model_aliases.csv) — reported names to canonical model IDs;
- [`data/scores.csv`](data/scores.csv) — measured score cells; and
- [`data/provenance.json`](data/provenance.json) — pinned upstream revisions and generation record.

Regenerate them from pinned local upstream clones with `python3 scripts/build_registry.py --sources <directory> --output data`. Generated files should be reviewed as evidence artifacts; automation does not replace source verification.

Install and run:

```bash
python3 -m pip install -e .
pathopress audit --scores data/scores.csv
pathopress validate --scores data/scores.csv
```

The default CLI support thresholds (three evaluations per model and five models per evaluation) are intentionally permissive smoke-test defaults. They are **not** evidence that a matrix is publication-ready. A public completion model must pass the stricter gates in the feasibility report, including leave-one-model and time-aware validation, protocol audit, low-rank diagnostics, and calibrated abstention.

On the present mixed seed pool, the default support filter retains 47 models × 28 evaluations and 806 primary-source-parsed cells (61.2%). These cells are machine extracted from pinned official tables but have not received dual human review. Random-cell validation gives a 1.046-point median absolute error and 2.450-point mean absolute error. Treat those numbers only as an executable smoke test: the pool combines endpoint families, missingness is structured, and random held-out cells are much easier than predicting a genuinely new release.

A fixed rank sweep in [`experiments/`](experiments/) reinforces that warning. Rank 1 performs best on random-cell and sparse-probe aggregates, while rank 2 performs best when an entire suite block is hidden (4.312 MAE / 2.320 MedAE). PathoROB block prediction is much weaker (12.475 MAE) than HEST (1.268) or THUNDER (4.964), so BenchPress's global rank-2 finding does not transfer automatically.

## BenchPress-style imputation and figures

The repository also reproduces BenchPress's 10-seed × 3-fold within-model
validation design and sweeps latent interaction ranks 0 through 10. Rank 1 is
best on all three matched-fold error summaries: pooled MAE 2.427, pooled MedAE
1.038, and median-of-fold MedAE 1.048. The task-column-median baseline is
4.470/2.400 MAE/MedAE; bias-only rank 0 is 2.553/1.101 and inherited rank 2 is
2.541/1.083. The primary point-estimate export therefore uses rank 1, not
BenchPress's rank 2.

The separate Soft-Impute SVD sweep used for BenchPress's published rank figure
is more nuanced: its MedAPE criterion narrowly selects rank 2 in both raw and
logit space, while pooled MAE and logit-space MedAE select rank 1. The current
evidence therefore supports an effective interaction rank around 1–2, not a
known exact matrix rank. Direct comparison with Microsoft's standalone default
predictor produces identical rank-2 outputs on this matrix.

Generate the complete point-estimate table and figures with:

```bash
pathopress impute --scores data/scores.csv --rank 1 --output outputs/imputations_rank1.csv
python3 experiments/run_benchpress_style.py
python3 experiments/run_soft_impute_rank_sweep.py
python3 scripts/plot_benchpress_style.py
```

The current supported matrix has 806 reported and 510 imputed cells. Fully
unobserved Patho-Bench and eva columns cannot yet be imputed; they need score
anchors extracted from papers or model cards first. See
[`docs/imputation.md`](docs/imputation.md) for metric definitions, the figure
gallery, and the distinction between a point estimate and a confidence
interval.

## Probe selection and benchmark informativeness

PathoPress now reproduces the logic behind BenchPress's GitHub hero curve. It
greedily selects evaluation columns that best reconstruct the fixed set of 806
published cells, compares them with 10 nested random probe orders, and repeats
selection on a 70% model-training split before isolated evaluation on the held
out 30% of models.

The faithful all-known curve falls from a 2.100-point column-median baseline to
0.634 MedAE with five probes and 0.241 with ten. Those headline values include
the measured probe cells as exact zero-error predictions, as BenchPress does.
When those cells are excluded, the corresponding errors are 0.907 and 0.792.
Held-out-model hidden-cell MedAE is 0.968 at five probes and 0.906 at ten.

The companion table ranks every single evaluation by its one-probe reduction
in scorecard MedAE and includes model coverage. The first ten entries are all
THUNDER columns, which is a warning about the current suite-blocked matrix, not
evidence that those datasets are intrinsically the ten most useful pathology
datasets. A second panel reports literal per-model mean-score prediction; that
is an additional diagnostic and is not what BenchPress calls “overall score
prediction.”

Generate the artifacts with:

```bash
python3 experiments/run_probe_selection.py
python3 scripts/plot_probe_selection.py
```

See [`docs/benchpress-parity.md`](docs/benchpress-parity.md) for the audited
upstream protocol, the exact meaning of informativeness, and the remaining
work for confidence, temporal, low-cost, and pathology-family validation.

## Data contract

PathoPress keeps four concepts distinct:

1. **Dataset artifact** — cohort/version, access route, license, unit of observation, and sample identifiers.
2. **Task identity** — dataset + prediction target + granularity. This is the level used to find conceptual duplicates.
3. **Evaluation protocol** — split, label map, preprocessing, magnification, pooling/adaptation method, metric, and aggregation. This is the matrix-column identity.
4. **Score observation** — exact model/checkpoint + protocol + value, uncertainty, source URL, and audit status. This is an observed matrix cell.

Collapsing layers 2 and 3 would silently combine results that answer different experimental questions. PathoPress deduplicates the catalog while retaining protocol-specific evaluations.

## Scope and non-goals

PathoPress is intended to answer: “Given a model's measured results and similar models' results, which missing evaluation scores may be predictable, with what uncertainty, and which real benchmark should be run next?”

It is not:

- a substitute for running the original pathology benchmarks;
- a new clinical validation study;
- a license to redistribute source images, labels, or gated model weights;
- proof that a model generalizes to an unseen institution; or
- a remedy for pretraining/evaluation leakage.

Predictions should be labeled as estimates, cite their supporting observations, carry uncertainty, and abstain when support is weak.

## Primary sources

- BenchPress: [code](https://github.com/microsoft/benchpress), [paper](https://arxiv.org/abs/2606.24020), [score matrix](https://huggingface.co/datasets/microsoft/benchpress-score-matrix)
- Patho-Bench: [code](https://github.com/mahmoodlab/patho-bench), [task splits](https://huggingface.co/datasets/MahmoodLab/Patho-Bench), [paper](https://arxiv.org/abs/2502.06750)
- eva: [code](https://github.com/kaiko-ai/eva), [datasets](https://kaiko-ai.github.io/eva/main/datasets/), [paper](https://openreview.net/forum?id=FNBQOPj18N)
- THUNDER: [code](https://github.com/MICS-Lab/thunder), [paper](https://papers.nips.cc/paper_files/paper/2025/hash/e3a2bd22ef74970b2fff74a16f806237-Abstract-Datasets_and_Benchmarks_Track.html)
- HEST: [code and benchmark](https://github.com/mahmoodlab/HEST), [benchmark data](https://huggingface.co/datasets/MahmoodLab/hest-bench), [paper](https://arxiv.org/abs/2406.16192)
- PathoROB: [code](https://github.com/bifold-pathomics/PathoROB), [data collection](https://huggingface.co/collections/bifold-pathomics/pathorob), [paper](https://arxiv.org/abs/2507.17845)

## License and attribution

PathoPress code is [MIT-licensed](LICENSE). The adapted numerical method is attributed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The factual registry and score metadata are covered by [DATA_NOTICE.md](DATA_NOTICE.md): upstream code, reported scores, model weights, and datasets retain their own licenses and access conditions. See [docs/feasibility.md](docs/feasibility.md#licensing-access-and-leakage) before redistributing any artifact.
