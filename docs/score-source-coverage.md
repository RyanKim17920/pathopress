# Numerical score-source coverage

This note defines which EVA and Patho-Bench numbers enter the registry, which
reported numbers remain excluded or quarantined, and what cannot safely be
inferred from either category. A score being present means that it was parsed
from a pinned first-party report and passed structural checks. It does **not**
mean that the score was independently reproduced, that the model checkpoint
revision is known, or that the cell is automatically safe to use as a
completion target.

## Coverage at a glance

| Evidence block | Extracted source cells | Registry-selected cells | Status |
|---|---:|---:|---|
| EVA repository pathology leaderboard | 195 = 15 models × 13 columns | 195 before source reconciliation | Ingested |
| Kaiko Midnight Hugging Face report | 180 = 15 models × 12 columns | 70 additional cells | Ingested; 110 overlapping cells retained as alternate evidence |
| EVA reconciled block | 375 observations across two sources | 265 unique model × protocol cells, 19 models, 15 protocols | Ingested |
| EXAONE Path 2.5, Patho-Bench Table 4 | 560 = 80 tasks × 7 models | 560 protocol-specific cells | Ingested |
| THREADS public Patho-Bench-compatible results | 336 = 42 tasks × 8 frozen representation rows | 336 protocol-specific cells | Ingested |
| THREADS internal results | 96 = 12 tasks × 8 frozen representation rows | 0 | Not extracted into the snapshot and not eligible for the public registry |

After the three ingested additions, `data/scores.csv` contains 1,976 score cells:
265 EVA, 896 Patho-Bench (560 EXAONE plus 336 THREADS), 234 HEST, 512 THUNDER,
and 69 PathoROB. The generated counts and source hashes are recorded in
[`data/provenance.json`](../data/provenance.json).

## EVA: 265 selected numerical cells

The extraction has two first-party inputs:

1. The pinned EVA pathology leaderboard
   [`pathology.csv`](https://github.com/kaiko-ai/eva/blob/e43e74a99b75660b0014f790f25a33dd9f11e121/tools/data/leaderboards/pathology.csv)
   at revision `e43e74a99b75660b0014f790f25a33dd9f11e121`. It contributes
   195 fully populated cells: 15 model rows by 13 result columns.
2. The pinned Kaiko Midnight Hugging Face
   [model report](https://huggingface.co/kaiko-ai/midnight/blob/adc6b15679c981cce6f9b018bbad09d16eeeda9f/README.md)
   at revision `adc6b15679c981cce6f9b018bbad09d16eeeda9f`. Its results table
   contributes 180 observations: 15 model rows by 12 result columns.

The sources overlap on 110 model × protocol cells. The registry does not average
them. For an exact duplicate key it selects the current EVA repository value,
and writes the Midnight value, both references, and their absolute difference
to [`data/eva_source_conflicts.csv`](../data/eva_source_conflicts.csv). The 70
non-overlapping Midnight observations are added, producing 265 selected cells.
The executable reconciliation is in
[`scripts/evidence/eva_scores.py`](../scripts/evidence/eva_scores.py), with shape
and precedence tests in
[`tests/test_eva_scores.py`](../tests/test_eva_scores.py).

The 15 protocol columns are deliberately more specific than a dataset name:

- validation and test are separate for PatchCamelyon, Camelyon16 Small, and
  PANDA Small;
- PatchCamelyon 10-shot test is separate from full-data PatchCamelyon;
- BACH, BRACS, BreakHis, CRC, and Gleason Arvaniti use their reported
  validation result, while MHIST uses its configured official test split;
- CoNSeP validation and MoNuSAC's configured official test split remain
  segmentation protocols rather than being mixed with classification columns.

EVA's [leaderboard documentation](https://github.com/kaiko-ai/eva/blob/e43e74a99b75660b0014f790f25a33dd9f11e121/docs/leaderboards.md)
defines balanced accuracy for classification and Dice excluding background for
segmentation. It reports means over five runs for patch classification and
segmentation and over 20 runs for slide tasks. The numerical sources provide
means but no per-cell dispersion, so the registry must not manufacture standard
errors or confidence intervals. Some training configs monitor ordinary accuracy
for checkpoint selection; that selection metric does not replace the documented
leaderboard metric. BACH remains a valid score column even though EVA excludes
it from its overall average because of its unusual post-resize resolution.

Other EVA caveats are material:

- the leaderboard CSV is not the same 13-task set as the canonical config
  inventory: it includes reduced-data variants and separate validation/test
  columns while omitting several full tasks;
- the Midnight report adds BRACS and PatchCamelyon 10-shot evidence and models
  absent from the current CSV, but it is a second report, not an independent
  rerun by this project;
- aggregate `AVG` and HEST summary columns are not independent cells and are
  excluded;
- reported model names are mapped by an explicit alias table, but exact model
  checkpoint revisions are not reported by the score tables.

## Patho-Bench: 560 EXAONE cells ingested

Patho-Bench's code and Hugging Face repository define tasks and splits, but do
not publish a current machine-readable numerical leaderboard. In particular,
files such as `k=all.tsv` are split definitions, not result tables; the runner
produces a local `results_summary.csv` ([pinned runner](https://github.com/mahmoodlab/Patho-Bench/blob/660e77044640e3d7d2f1150cc6721e97454993bf/patho_bench/Runner.py#L237-L312)).
The task crosswalk therefore uses the pinned
[95-task Hugging Face manifest](https://huggingface.co/datasets/MahmoodLab/Patho-Bench/blob/60fde3a9138b2fb27a163ed6f3e2cf0ef7e8f387/available_splits.yaml)
and its per-task configs at revision
`60fde3a9138b2fb27a163ed6f3e2cf0ef7e8f387`, while numerical values come from a
separate primary report.

The ingested source is EXAONE Path 2.5
[Table 4](https://arxiv.org/pdf/2512.14019v1): 80 exact Patho-Bench task names by
seven model columns (`CHIEF`, `GigaPath`, `PRISM`, `TITAN`, `H-optimus-0`,
`UNI2-h`, and `EXAONE Path 2.5`). The source-archive member
`tabs/pathobench_result.tex` is pinned by SHA-256
`0c479164dfab7ac48a1e1876649ef73efe9f457e064c3ab00ee960856d35a268`.
The reproducible extractor and committed evidence snapshot are
[`scripts/extract_exaone_pathobench.py`](../scripts/extract_exaone_pathobench.py)
and
[`source_data/exaone_path_2_5_pathobench_2512.14019v1.csv`](../source_data/exaone_path_2_5_pathobench_2512.14019v1.csv).

The resulting 560 point estimates comprise:

- 504 macro one-vs-rest AUROC cells: 72 classification tasks × 7 models;
- 56 concordance-index cells: 8 survival tasks × 7 models.

These are stored under `pathobench.exaone2025.*` protocol IDs rather than being
attached directly to the current generic task rows. This is necessary because
EXAONE applies its shared paper protocol and reports macro one-vs-rest AUROC for
every classification task, including tasks whose current Hugging Face config
specifies balanced accuracy or weighted kappa. The 80 protocol rows share task
identity and dataset-artifact links with their current Patho-Bench counterparts,
but their scores must not overwrite or be averaged with results from a different
probe, split realization, endpoint, or metric. Table 4 reports point estimates
without cell-level uncertainty. Its derived average row is excluded.

EXAONE Table 4 covers 80 of the 95 current Hugging Face tasks. The other 15
remain unobserved by this source; no values are imputed and no fuzzy task-name
matches are allowed:

- `pathobench.cervical_subtype.subtype`
- `pathobench.cptac_all.organ`
- `pathobench.cptac_ccrcc_dhmc.oncotreecode`
- `pathobench.cptac_lung.subtype`
- `pathobench.crc_outcomes.os-valentino`
- `pathobench.crc_outcomes.pfs-valentino`
- `pathobench.crc_outcomes.braf-grading`
- `pathobench.crc_outcomes.braf-lymphovas-invasion`
- `pathobench.crc_outcomes.braf-mmr`
- `pathobench.crc_outcomes.braf-synapto`
- `pathobench.crc_outcomes.braf-tils`
- `pathobench.crc_outcomes.valentino-molecular-subtype`
- `pathobench.crc_outcomes.valentino-msi`
- `pathobench.dhmc_luad.label`
- `pathobench.ucla_lung.progression-regression`

## THREADS: 336 public cells ingested and 96 internal cells quarantined

The THREADS paper ([arXiv 2501.16652v1](https://arxiv.org/abs/2501.16652v1))
reports a larger Patho-Bench-era result block. Its eight frozen representation
rows supply the ingested public model × evaluation block:

1. Virchow mean pooling
2. GigaPath patch encoder with mean pooling
3. CHIEF patch encoder with mean pooling
4. CONCH v1.5 mean pooling
5. PRISM slide encoder
6. GigaPath slide encoder
7. CHIEF slide encoder
8. THREADS slide encoder

Extended Data Tables 11–26 and 28–37 contain 42 tasks that crosswalk exactly to
public Patho-Bench artifacts. Their 42 × 8 = 336 cells have been extracted into
[`source_data/threads_pathobench_2501.16652v1.csv`](../source_data/threads_pathobench_2501.16652v1.csv)
by
[`scripts/extract_threads_pathobench.py`](../scripts/extract_threads_pathobench.py).
The parsed arXiv HTML is pinned by SHA-256
`a6c7af63c1f527eba692f83b362651e0e1d96d07e303520f90cd08f34b00c92f`,
and the corresponding source archive by
`3d8b3f6779b9b0eae21be12e8917bd6f0bab26e3c7943470e378383d20a1de4f`.
All 336 public cells are ingested into `data/scores.csv` under versioned
`pathobench.threads2025.*` protocol IDs. Registry presence records the paper's
evidence; it does not by itself make the cells leakage-safe completion targets.

The remaining 12 THREADS tasks are internal cohorts. Tables 9 and 10 contain 11
MGB Breast/Lung tasks, and Table 27 contains one internal GBM-treatment task.
Their 12 × 8 = 96 cells are neither public Patho-Bench task evidence nor part of
the committed public snapshot. They remain quarantined and must not be
silently counted as public coverage.

The public extraction also excludes supervised and fine-tuned rows,
aggregate figures, alternative-regularization ablations, few-shot experiments,
transfer tests, and retrieval tables because those are distinct protocols or
derived summaries. Where the frozen rows report uncertainty, the snapshot
preserves its type: standard error for fold/Monte Carlo means and 95% bootstrap
confidence intervals for single-fold tasks. It does not convert one into the
other.

The ingested model mappings keep GigaPath mean pooling (`prov-gigapath`)
distinct from the GigaPath slide encoder (`prov-gigapath-slide`), and CHIEF mean
pooling (`chief-patch-mean`) distinct from the CHIEF slide encoder
(`chief-slide`). The paper-specific balanced linear-probe and CoxNet settings
likewise have versioned protocol IDs; they do not inherit a current generic
Patho-Bench protocol merely because the dataset and target match.

## Deduplication and validation boundaries

Deduplication happens at explicit levels:

- identical source evidence for the same `model_id × evaluation_id` is resolved
  by a declared precedence rule and retained as a conflict record, as with the
  110 EVA overlaps;
- the same dataset, target, and observation granularity links task identities,
  but distinct splits, probes, preprocessing, and metrics remain distinct
  evaluation protocols;
- aliases may identify the same base model, but a patch encoder plus mean
  pooling is not collapsed into a slide encoder with the same family name;
- averages, rankings, and other derived summaries are excluded rather than
  treated as additional observations.

Every ingested score is currently `machine_parsed_single_source`. Structural
validation checks source hashes, table shape, exact task and model sets,
duplicate keys, finite values, metric domains, and source locators. It does not
establish experimental reproduction or resolve unreported checkpoint commits.

Randomly holding out individual cells would substantially overstate imputation
quality. Cells from one paper share preprocessing, probe code, task splits, and
reporting choices; model variants share architecture and pretraining data; and
the same cohort can appear under multiple suites or protocol variants. Models
may also have seen some benchmark-family data during pretraining, which cannot
be ruled out from a score table alone. Completion experiments should therefore
report source-block, model-family, task-identity/dataset-artifact, and temporal
holdouts in addition to any random-cell diagnostic. The 96 internal THREADS
cells must remain excluded from training and evaluation unless their data
status and protocol contracts are separately approved. The 336 public THREADS
cells should still be evaluated with source-block and pretraining-overlap
sensitivity analyses rather than treated as independent random observations.
