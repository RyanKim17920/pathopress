# Feasibility of BenchPress-style completion for pathology

_Audit date: 2026-08-05. Counts describe the cited live repositories and documents on that date; these are living projects and will drift._

## Bottom line

**Feasible, with an important qualification:** the easy part is porting the completion algorithm; the hard part is creating enough trustworthy, comparable score cells to show that pathology evaluation matrices are predictable at all.

A practical effort has three milestones:

| Deliverable | Difficulty | Plausible elapsed time | What “done” means |
|---|---:|---:|---|
| Audited registry MVP | Moderate | 2–4 weeks | Suite/task/protocol/model identities, source URLs, licenses, and extracted score candidates are machine-readable and reviewable. |
| Research-grade matrix pilot | Hard | 6–10 weeks | A sufficiently connected score matrix passes identity, metric, low-rank, and held-out prediction checks on at least one coherent stratum. |
| Defensible living release | Hard | 3–6 months | Repeatable extraction, dual review, confidence calibration, temporal validation, leakage annotations, abstention, and maintenance policy are in place. |

These estimates assume one engineer/researcher with domain review and reuse of reported scores. Re-running many missing experiments, acquiring controlled cohorts, or harmonizing WSI pipelines can extend the work materially.

## What BenchPress actually compresses

[BenchPress](https://github.com/microsoft/benchpress) is best understood as **score-matrix completion and evaluation selection**, not benchmark-item compression. Its paper starts from a sparse matrix whose rows are exact LLM releases and columns are benchmark settings. The living repository currently describes a raw audit pool of 189 models, 316 evaluations, and 4,903 numeric scores; its paper-canonical filter contains 84 models × 133 evaluations with 2,604 measured cells (23.3% density). After transforming percentage scores to logit space and standardizing each column, its default method fits model and evaluation biases plus rank-2 latent factors. It reports a 4.60-point median absolute error on its fold protocol and shows that a small probe set can predict much of a model profile.

A pathology analogue would therefore:

1. extract reported results from primary papers, supplements, repositories, leaderboards, and Hugging Face model/dataset cards;
2. resolve model, task, and protocol identities;
3. construct a sparse model × evaluation matrix;
4. test whether a low-rank assumption holds within coherent pathology strata;
5. estimate supported missing cells with calibrated uncertainty; and
6. recommend the next real evaluations with the highest information value.

It would **not** combine the images from HEST and PANDA, shorten the task datasets, or make the Patho-Bench/EVA/THUNDER runners interchangeable. Those remain upstream execution systems.

## Audited suite inventory

The counts below intentionally distinguish paper claims, documentation, and executable configuration. A living registry should record all three and pin a revision.

| Suite | Current audited inventory | Current-versus-paper/documentation drift | Primary sources |
|---|---|---|---|
| Patho-Bench | 95 tasks across 33 public data sources: 85 classification and 10 survival; 83 case-level and 12 slide-level. Primary metrics: 57 macro one-vs-rest AUC, 19 balanced accuracy, 9 quadratic-weighted kappa, and 10 concordance index. | The live README says 95 tasks/33 sources. The THREADS paper/HF artifacts are useful provenance but must not override current split/config files without a versioned record. | [GitHub](https://github.com/mahmoodlab/patho-bench), [Hugging Face](https://huggingface.co/datasets/MahmoodLab/Patho-Bench), [Patho-Bench paper](https://arxiv.org/abs/2502.06750), [THREADS paper](https://arxiv.org/abs/2501.16652) |
| EVA | 13 canonical pathology identities: 11 patch-level tasks and two slide-level tasks. The pathology set covers BACH, BRACS, BreakHis (selected four-class task), CRC-100K, GleasonArvaniti, MHIST, PatchCamelyon, UniToPatho, MoNuSAC, CoNSeP, BCSS, Camelyon16, and PANDA. | The dataset documentation displays 14 because `PANDASmall` is listed beside full PANDA. It is a subset/setting variant, not a new dataset × target × granularity identity. | [GitHub](https://github.com/kaiko-ai/eva), [dataset documentation](https://kaiko-ai.github.io/eva/main/datasets/), [paper](https://openreview.net/forum?id=FNBQOPj18N) |
| THUNDER | 21 current task configurations. Classification (17): BACH, BRACS, BreakHis, CCRCC, CRC-100K, ESCA, MHIST, PatchCamelyon, Camelyon17-WILDS, TCGA CRC-MSI, TCGA TILs, TCGA Uniform, SPIDER Breast, Colorectal, Skin, and Thorax, plus code-present/pending STARC9. Segmentation (4): OCELOT, PanNuke, SegPath Epithelial, and SegPath Lymphocytes. | Current documentation exposes 20; executable code exposes 21; the NeurIPS paper describes 16 datasets. This is normal living-suite drift and demonstrates why commits and extraction dates belong in the registry. | [GitHub](https://github.com/MICS-Lab/thunder), [API inventory](https://mics-lab.github.io/thunder/api/), [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2025/hash/e3a2bd22ef74970b2fff74a16f806237-Abstract-Datasets_and_Benchmarks_Track.html) |
| HEST-Benchmark | Nine morphology-to-expression tasks, each predicting 50 highly variable genes at 112 × 112 μm regions: IDC, PRAD, PAAD, SKCM, COAD, READ, CCRCC, LUNG, and LYMPH_IDC. | The live leaderboard has grown from 11 evaluated models in the original benchmark description to 25 public models as of its 2026-04-03 result table; tasks remain nine. | [GitHub](https://github.com/mahmoodlab/HEST), [Hugging Face benchmark data](https://huggingface.co/datasets/MahmoodLab/hest-bench), [paper](https://arxiv.org/abs/2406.16192) |
| PathoROB | Four dataset configurations × three metrics = 12 robustness endpoints. The configurations are TCGA 2×2, TCGA 4×4, CAMELYON, and Tolkach ESCA; the metrics are Robustness Index, Average Performance Drop, and Clustering Score. Together they cover 28 biological classes and 34 centers. | The user-facing code uses a shared `tcga` handle for the two TCGA configurations. The live Robustness Index leaderboard displays TCGA 2×2, CAMELYON, Tolkach ESCA, and an average; that average is a derived summary, not a fourth dataset or independent matrix column. Preserve TCGA 2×2 and 4×4 as separate protocols. | [GitHub](https://github.com/bifold-pathomics/PathoROB), [Hugging Face collection](https://huggingface.co/collections/bifold-pathomics/pathorob), [paper](https://arxiv.org/abs/2507.17845) |

HEST and PathoROB should initially be modeled as separate strata. HEST's gene-wise regression aggregate and PathoROB's representation/robustness endpoints are not ordinary classification or survival columns. They are valuable precisely because they probe different capabilities, but they may not share the same latent geometry.

## Deduplication result

For Patho-Bench, EVA, and THUNDER, an exact identity key of **dataset × prediction target × observation granularity** produces:

- **122 unique code-backed identities**, including THUNDER's `STARC9` configuration;
- **121 unique documented identities** when that unlisted/pending entry is excluded; and
- **seven exact cross-suite overlaps**.

The overlaps are:

| Suites | Shared task identity | Important caveat |
|---|---|---|
| EVA ↔ THUNDER | BACH patch classification | Confirm class map, resampling, split, and metric before aligning scores. |
| EVA ↔ THUNDER | BRACS patch classification | Same named dataset does not imply the same split or adaptation head. |
| EVA ↔ THUNDER | BreakHis selected four-class patch classification | BreakHis also appears as an eight-class task elsewhere; those targets are not duplicates. |
| EVA ↔ THUNDER | CRC-100K nine-class patch classification | Dataset aliases such as CRC/NCT-CRC-HE must resolve to a versioned artifact. |
| EVA ↔ THUNDER | MHIST binary patch classification | Verify official split versus generated folds. |
| EVA ↔ THUNDER | PatchCamelyon binary patch classification | Do not confuse this with Camelyon16 slide-level classification. |
| Patho-Bench ↔ EVA | Full PANDA slide-level ISUP grading | `PANDASmall` is a subset/protocol variant; kappa, label aggregation, and split must remain explicit. |

Thus the intuition that “lots are repeated” is only partly borne out. There are many recurring **dataset families, cancer types, and outcome concepts**, but only seven exact task identities under a deliberately conservative key. Broader fuzzy clustering is useful for navigation and meta-analysis, not safe deduplication.

The machine-readable registry materializes this audit in [`data/tasks.csv`](../data/tasks.csv) and [`data/deduplication.csv`](../data/deduplication.csv), with pinned upstream commits in [`data/provenance.json`](../data/provenance.json). It currently contains 287 protocols over 145 task identities. The complete 95-identity Patho-Bench inventory and its paper-specific protocol variants live there rather than being copied into this narrative report.

## Current expanded score matrix

The generated score pool has **1,976 measured cells**: 896 Patho-Bench, 265
EVA, 234 HEST, 512 THUNDER, and 69 PathoROB. The exact paper/Hugging Face
sources, protocol boundaries, THREADS internal-cohort quarantine, and 110 EVA
alternate-source conflicts are documented in
[`score-source-coverage.md`](score-source-coverage.md) and
[`data/eva_source_conflicts.csv`](../data/eva_source_conflicts.csv).

With the primary support filter, 59 models × 165 evaluations and 1,967
observations remain (20.2054% filled). Matched-fold rank-1 bias-ALS gives
3.005264 MAE, 1.603026 MedAE, and 1.609435 median fold MedAE, versus
4.092133/2.477500 for the column-median baseline. Rank 0 is
3.056250/1.711456 and rank 2 is 3.117610/1.632035. Raw and logit Soft-Impute
both select rank 1. The rank-1 completed artifact contains 1,967 reported and
7,768 imputed cells.

The more demanding fixed-split experiment in [`experiments/`](../experiments/)
finds rank-1 random-cell error of 2.834996/1.526795 MAE/MedAE and pooled sparse
new-model error of 3.190380/1.817465. Hiding an entire suite raises rank-1 error
to 5.612789/3.525174; rank 5 is best overall on that stress test at
4.952972/3.055638. Rank-1 suite-block MAE is 4.2643 for Patho-Bench, 8.8757 for
EVA, 1.4860 for HEST, 14.4385 for PathoROB, and 6.3854 for THUNDER. This is
direct evidence against assuming one global rank or one error guarantee across
pathology endpoint families.

Probe selection is useful but remains retrospective. From a 1.900 MedAE
full-matrix baseline, all-known scorecard MedAE reaches 1.481124 with five
probes and 1.196456 with ten; hidden-only values are 1.612112 and 1.539134.
Held-out-model hidden-cell MedAE is 1.951271 and 1.879857. Literal-average MAE
is 3.203489/2.908513 all-known and 1.684778/1.231976 held out. The exact selected
probe lists are in the result artifact and span multiple suites.

## The canonical four-layer model

The registry needs four layers because “benchmark” is overloaded:

| Layer | Canonical identity and fields | Why it exists |
|---|---|---|
| 1. Dataset artifact | Stable artifact ID; source/version; cohort/site; raw-vs-derived status; sample unit and IDs; stain/modality; access; license; checksum or revision. | Distinguishes a dataset name from a reproducible set of examples. |
| 2. Task identity | Dataset artifact + target definition + granularity (patch/region/slide/case/patient/gene). Include label ontology and prediction family. | Finds genuine conceptual overlap without discarding experimental detail. |
| 3. Evaluation protocol | Task ID + split/folds + preprocessing/magnification/patching + representation/pooling + adaptation/head + metric/aggregation + seed policy. | Defines a comparable matrix column. Protocol changes create new columns or an explicit equivalence decision. |
| 4. Score observation | Exact model/checkpoint/revision + protocol ID + point estimate + uncertainty/runs + source location + extractor/reviewer + audit status. | Defines a measured cell with enough evidence to reproduce or challenge it. |

Model identity needs its own normalized entity behind the fourth layer: architecture, parameter count, checkpoint revision, tile versus slide encoder, input resolution, pretraining corpus/time, and weights license. Names like “UNI,” “CONCH,” or “GigaPath” are not sufficient identifiers by themselves.

### Why protocols cannot be merged

Even for an exact task overlap, the following can change the meaning of a score:

- official versus regenerated or patient-stratified splits;
- patch-, slide-, case-, or patient-level aggregation;
- label remapping, excluded classes, and binary versus multiclass targets;
- stain normalization, tissue segmentation, patch size, magnification, and encoder input resize;
- frozen k-nearest-neighbor, linear probe, attention MIL, slide pooling, partial tuning, or full fine-tuning;
- hyperparameter selection and validation leakage;
- AUC variant, balanced accuracy, kappa, concordance index, Pearson correlation, Dice, or rank-sum aggregation; and
- single-run point estimates versus multi-run means and confidence intervals.

Deduplicate layer 2 for the catalog, but construct matrix columns from layer 3. Two protocols may later be declared equivalent only through a documented crosswalk and sensitivity analysis. Never average them merely because the dataset name matches.

## Matrix feasibility and release gates

The expanded numerical pilot is enough to test matrix completion; it is not
enough to claim prospective BenchPress-like performance. Each proposed matrix
stratum should pass all of these gates.

### 1. Identity and provenance gate

Every observed cell has an exact checkpoint, protocol, numeric table location, primary URL, and audit status. OCR or automated extraction creates a candidate, not a verified score. Duplicate papers and model cards must not turn one experiment into multiple observations.

### 2. Connectivity and support gate

The model–evaluation bipartite graph must have a useful connected core. Each retained model needs multiple observations and each evaluation multiple independently reported models. BenchPress's paper filter used at least 15 scores per model and eight models per benchmark; PathoPress's CLI defaults of three and five are only permissive smoke-test thresholds. Final thresholds should be selected before looking at completion performance and accompanied by coverage curves.

### 3. Comparable-scale gate

Metrics must have an explicit direction and mathematically valid transform. Percentage AUC/accuracy and 0–1 c-index can be mapped to a consistent display scale, but a HEST Pearson correlation or PathoROB robustness measure is not semantically “the same score.” Column standardization permits differently difficult evaluations; it does not justify mixing incompatible or oppositely oriented observations without metadata and stratified checks.

### 4. Low-rank evidence gate

Show singular-value spectra on sufficiently complete submatrices, compare against degree-preserving missingness/null baselines, and test stability across suites and task families. A low-rank fit to a highly structured missingness pattern can be an artifact: model families are often evaluated by their own authors on the same favorite suites.

### 5. Honest prediction gate

Random-cell holdout is a software smoke test only. Report at least:

- repeated per-model folds comparable to BenchPress;
- leave-one-model/checkpoint-out prediction;
- time-aware evaluation in which later model releases are never used to predict earlier availability states;
- suite-held-out and protocol-family-held-out stress tests;
- baselines such as evaluation mean, model mean, nearest neighbor, regularized additive biases, and higher/lower ranks; and
- MAE/median AE, rank preservation, interval coverage, and coverage at each abstention threshold.

### 6. Calibration and abstention gate

Prediction intervals must be calibrated on out-of-fold residuals and widen for sparse models, sparse evaluations, distant peers, and out-of-family checkpoints. The system should decline to predict isolated cells. A predicted pathology score must never be displayed indistinguishably from a measured result.

### 7. Usefulness gate

Demonstrate that a small, pre-declared probe set predicts the rest better than simple baselines and preserves the model-selection decisions users care about. If the matrix has no stable low-rank structure, the registry and deduplication work are still useful; completion should simply not ship.

## Phased roadmap

### Phase 0 — Freeze scope and schema (days 1–3)

Pin repository commits and paper versions. Adopt the four-layer schema, controlled metric vocabulary, exact model ID rules, audit states, and a policy for corrections. Choose one initial coherent stratum, likely frozen patch-encoder classification, rather than mixing every endpoint immediately.

### Phase 1 — Generate and audit the registry (weeks 1–4)

Generate inventories from current code/configuration, reconcile them with papers and documentation, and record drift instead of hiding it. Extract tables from primary sources and HF reports into candidate observations. Resolve aliases and suspected duplicates, then have a second reviewer verify numeric values and protocols. Publish source URLs and extraction notes, not copyrighted paper tables or raw benchmark images.

### Phase 2 — Build the observed matrix (weeks 3–7)

Normalize exact checkpoint IDs, choose canonical score representations, keep protocol variants separate, and compute graph coverage. Start with measured, directly comparable cells. Add HEST, survival, segmentation, and robustness as separate matrices only after each has adequate overlap.

### Phase 3 — Establish feasibility (weeks 6–10)

Run missingness analysis, complete-submatrix/SVD diagnostics, simple baselines, rank/regularization sweeps, and retrospective holdouts. Pre-register release gates. Investigate errors by task, suite, architecture family, pretraining overlap, and protocol distance.

### Phase 4 — Calibrate and select probes (months 2–4)

Fit out-of-fold uncertainty, establish abstention rules, and search for a low-cost probe set subject to access, compute, and licensing constraints. Verify it on temporally newer releases and on model families absent from training.

### Phase 5 — Living release (months 3–6)

Ship measured/predicted labeling, provenance views, versioned snapshots, contribution templates, automated schema checks, human review, and a correction log. Rebuild and recalibrate after material updates. Predictions should be reproducible from a named snapshot, never silently change underneath a citation.

## Licensing, access, and leakage

### Licensing and access

- [BenchPress code](https://github.com/microsoft/benchpress) is MIT; its public score-matrix dataset is CDLA-Permissive-2.0. An implementation can be adapted with attribution, but pathology source data do not inherit those terms.
- [Patho-Bench's HF dataset card](https://huggingface.co/datasets/MahmoodLab/Patho-Bench) identifies a non-commercial Creative Commons license for its split/config artifacts; its repository does not redistribute all raw WSI data. Each underlying cohort keeps its own access terms.
- [HEST](https://github.com/mahmoodlab/HEST/blob/main/LICENSE.md) is CC BY-NC-SA 4.0 and its full data are terabyte-scale. Commercial redistribution/use requires separate analysis.
- [PathoROB code](https://github.com/bifold-pathomics/PathoROB) is BSD-3-Clause, but its subsampled datasets retain source licenses: CAMELYON is CC0, TCGA-UT is CC BY-NC-SA 4.0, and Tolkach ESCA is CC BY-SA 4.0 with a PathoROB-specific grant noted by the authors.
- [EVA](https://github.com/kaiko-ai/eva/blob/main/LICENSE) is Apache-2.0 and [THUNDER](https://github.com/MICS-Lab/thunder/blob/main/LICENSE) is CC BY 4.0 at repository level. Dataset and model licenses remain separate. Many supported Hugging Face weights are gated and require acceptance of usage conditions.

The safest public artifact is a provenance-rich facts table containing identifiers, reported numeric results, and short factual protocol metadata. Do not mirror images, labels, model weights, or substantial copyrighted tables unless their licenses and access terms explicitly permit it. Record the source license per artifact and obtain legal review for a commercial product.

### Leakage and scientific validity

Pathology foundation models are frequently pretrained on TCGA and other cohorts that reappear downstream. A high predicted or measured score can therefore reflect pretraining exposure, near-duplicate slides, institutional shortcuts, or selection on a public leaderboard rather than deployable clinical generalization.

The registry should record known pretraining cohorts, date ranges, site overlap, public/private status, and any author disclosure of evaluation leakage. Add exclusion/sensitivity views for contaminated model–task pairs. Hash-based image checking is useful where lawful access exists, but absence of a detected duplicate is not proof of independence.

Matrix completion introduces an additional leakage path: a later model card may copy a score from an earlier paper, or several papers may report the same upstream run. Provenance lineage must keep these as one measured cell. Time-aware validation must use only information that existed at the prediction cutoff.

Finally, estimates are research metadata, not medical-device evidence. They cannot establish diagnostic safety, subgroup performance, external-site validity, or prospective clinical utility. The purpose is to prioritize which real evaluations to run, not to avoid the evaluations needed for clinical claims.

## Recommendation

Proceed, but stage the bet. First ship the audited, versioned registry and a measured-score browser; those are valuable regardless of the modeling outcome. Then attempt completion on one well-connected, protocol-coherent matrix and make low-rank structure, temporal prediction accuracy, interval calibration, and abstention hard release gates. If those gates fail, keep PathoPress as the deduplicated evidence layer rather than forcing a pathology analogue of BenchPress where the data do not support it.
