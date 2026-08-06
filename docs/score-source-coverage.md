# Numerical score-source coverage

This note defines which pathology benchmark numbers enter the registry, which
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
| PathoROB Nature Source Data | 100 = 20 models × (2 APD + 3 clustering protocols) | 100 raw registry cells; 60 clustering cells analysis-eligible | Ingested with endpoint-specific eligibility |
| PathoROB pinned repository examples | 22 = two models across per-dataset/aggregate APD and clustering | 0 | Extracted to the audit snapshot; quarantined as version-different examples |
| H0-mini arXiv v3 public leaf tables | 46 = 2 models × (8 EVA + 9 HEST + 6 PLISM) | 46 versioned cells | Ingested; four EVA/HEST means excluded |
| PLISM official repository leaderboard | 10 observations = 2 models × (4 leaf metrics + 1 mean) | 8 versioned cells | Ingested; two derived means/ranks excluded and paper/repository versions remain separate |
| UNI official repository, UNI2-h row | 8 observations = 6 public leaf + 1 HEST mean + 1 internal IHC | 6 raw-registry-only public cells | Ingested without normalization; endpoint metrics unspecified, aggregate excluded, internal row quarantined |
| H0-mini BreastBm private table | 18 H0-mini cells | 0 | Extracted to the audit snapshot; private cohorts quarantined |
| GenBio-PathFM official PDF | 49 observations | 15 analysis-eligible leaves; 6 APD leaves raw-only | Ingested under versioned kNN/PathoROB protocols; 12 rounded alternates, 10 HEST-joint cells and 6 aggregates are not selected |
| Midnight MICCAI 2025 Table 2 | 42 observations = 3 checkpoints × (12 EVA leaves + 2 aggregates) | 24 base-checkpoint leaves | Ingested under CLS+Mean protocols; 12 high-resolution post-trained leaves and all 6 aggregates are quarantined |
| OpenMidnight technical report | 16 observations | 12 own-model EVA leaves | Ingested under CLS-only protocols; two aggregates and two contradictory prose claims are excluded |
| CONCH / CONCHv1.5 / Phikon-family official papers | 584 primary observations | 417 public leaves | Ingested under exact versioned protocols; 151 private/internal and 16 fine-tuned cells quarantined |
| CTransPath comparator audit | 50 secondary observations | 0 | Staged as quarantine-only; original publisher result tables remain inaccessible |
| Hibou / MUSK / GPFM official papers and GPFM workbooks | 397 observations | 185 public leaf cells | Ingested; 212 private, fine-tuned, controlled-access, or aggregate cells quarantined |
| Virchow2-family / Prov-GigaPath / TITAN primary papers | 1,083 exact numeric observations | 737 public protocol-specific cells | Ingested by evidence adapter; 346 private, fine-tuned, or aggregate cells quarantined |

After the ingested additions, `data/scores.csv` contains 4,013 score cells:
896 Patho-Bench, 317 EVA, 524 THUNDER, 377 HEST, 178 PathoROB, 26 H0-mini
companion-repository cells, 737 Virchow2-family/TITAN cells, 356 Wave-D report
and paper cells, 417 CONCH/Phikon-family cells, and 185 Hibou/MUSK/GPFM cells.
Of these, 3,952 have an analysis-eligible primary-source normalization, 46 APD
cells and six public cells whose endpoint metric is unspecified remain
raw-registry-only, and nine PathoROB RI cells are external-publication reports.
The support-filtered research matrix retains 2,122 observations over 59 models
and 187 protocols. The generated counts and source hashes are recorded in
[`data/provenance.json`](../data/provenance.json).

## Virchow2-family, Prov-GigaPath, and TITAN: 737 public leaves

The pinned primary-paper extract contains 108 public Virchow2-family comparator
cells and 629 public TITAN/TITAN-V cells. Virchow2G, Virchow2, and Virchow each
have the same 36 public leaves: eight OOD tile tasks and ten HEST tasks for each
of CLS-only and CLS+Mean. HEST uses the paper's random-forest protocol, not the
ridge/PCA protocol used by other HEST sources. The 36 PanMSK and aggregate
observations are retained only in quarantine; PanMSK is an internal MSK cohort.

TITAN coverage spans the public headline cohorts plus the broader morphology,
grading, molecular, six-cohort TCGA survival, zero-shot, Rare-Cancers-Public,
and TCGA Slide-Reports retrieval tables. TITAN-V and multimodal TITAN remain
separate model identities, and logistic regression, SimpleShot, k-NN,
few-shot-K, zero-shot, slide retrieval, and cross-modal retrieval remain
separate protocols. The TITAN quarantine contains 284 cells from OT108, CRANE,
renal-allograft, MGB/MGH, private rare-cancer cohorts, fine-tuned rows, and
reported aggregates.

Prov-GigaPath contributes no eligible frozen public leaf from the audited
paper tables. Its 26 exact Supplementary Table 2 values are task-specific
slide-encoder fine-tuning results; 21 use Providence data and the other five use
TCGA-LUAD but are still fine-tuned. The report-aligned zero-shot experiment also
uses a proprietary Providence holdout. Its main-paper and extended-data figures
show bars without exact numeric labels, and neither the supplement nor pinned
official repository provides the underlying result table, so approximate
visual readings are deliberately not entered. The deterministic extractor is
[`scripts/extract_group_c_official_scores.py`](../scripts/extract_group_c_official_scores.py),
with public and quarantine ledgers in
[`source_data/virchow2g_gigapath_titan_official_scores_2024_2025.csv`](../source_data/virchow2g_gigapath_titan_official_scores_2024_2025.csv)
and
[`source_data/virchow2g_gigapath_titan_official_quarantine_2024_2025.csv`](../source_data/virchow2g_gigapath_titan_official_quarantine_2024_2025.csv).
Pinned hashes, repository revisions, and the Prov-GigaPath graphical-only
limitation are recorded in
[`source_data/group_c_source_audit.md`](../source_data/group_c_source_audit.md).

The tile encoder has a separate exhaustive no-eligible-evidence audit. The
Nature tile-pretraining ablation is not an isolated tile-feature evaluation:
the DINOv2/MAE/SimCLR/ImageNet variants feed the downstream whole-slide stack.
The official PCam materials are reproducibility inputs rather than reported
results—the repository computes metrics into a local `results.txt`, while the
Hugging Face dataset contains embeddings but no checked result. Accordingly,
external HEST/PathoROB/THUNDER reports may still contain scores for the model,
but the primary-paper ledger must not claim that the authors published a
frozen public tile-only benchmark. The machine-checked disposition ledger and
validator are
[`source_data/prov_gigapath_tile_evidence_audit_2024_2026.csv`](../source_data/prov_gigapath_tile_evidence_audit_2024_2026.csv)
and
[`scripts/audit_prov_gigapath_tile_evidence.py`](../scripts/audit_prov_gigapath_tile_evidence.py).

## H0-mini and UNI2-h: 60 public leaf cells, without cross-version merging

The pinned H0-mini source is [arXiv 2501.16239v3](https://arxiv.org/abs/2501.16239v3),
whose source archive has SHA-256
`222798059c15b554528d61f8caa04de8fcc2d5cc23997607dc25d851282a6f08`.
Its Tables 1–3 contain 46 public leaf cells: for each of H0-mini and UNI2-h,
eight EVA tasks, nine HEST tasks, and six PLISM robustness metrics. This is the
origin of the earlier “23 cells per model” count. The complete official-source
audit adds 14 more public leaf cells: eight refreshed cells from the pinned
[PLISM repository](https://github.com/owkin/plism-benchmark/tree/5ec9511893af993f6faa099f093d1924b291aed2)
and six UNI2-h cells from the pinned
[UNI repository](https://github.com/mahmoodlab/UNI/tree/42715efc11722a496e0a67f3369505a8f277206c),
for 60 public leaf cells total. By model, this is 27 H0-mini cells and 33 UNI2-h
cells; by suite/report it is 16 EVA, 18 HEST, 20 PLISM, and six UNI-repository
cells.

The audit snapshot contains 86 observations in all: the 60 public leaf cells,
seven derived means, and 19 private/internal cells. The seven means are four
paper EVA/HEST means, two PLISM repository leaderboard means, and one UNI
repository HEST aggregate. The private/internal quarantine consists of 18
H0-mini BreastBm cells and one UNI2-h IHC ER/PR cell. None enters the public
score registry.

Paper and repository PLISM scores are not reconciled or averaged. The repository
leaderboard is a later dated report with different values and a distinct
all-pairs cosine endpoint; it receives `plism.repo2025.*` protocol IDs. Its three
top-10 endpoints share task identities with the corresponding paper endpoints
only so deduplication can expose the repeated information while retaining both
protocol-specific values.

The six public UNI-repository cells are deliberately raw-registry-only. The
README states the global-representation/no-TTA recipe and learning-rate sweep,
but it does not identify each column's endpoint metric. The values therefore use
`reported_performance_score`, have blank `normalized_score`, and are excluded
from factor analysis rather than being silently treated as accuracy or AUROC.
The reproducible evidence validator is
[`scripts/extract_h0mini_uni2h_scores.py`](../scripts/extract_h0mini_uni2h_scores.py),
and the full disposition ledger is
[`source_data/h0mini_uni2h_official_scores_2025.csv`](../source_data/h0mini_uni2h_official_scores_2025.csv).
Pinned model-card revisions, file hashes, and source-level adjudications are in
[`source_data/h0mini_uni2h_source_audit.md`](../source_data/h0mini_uni2h_source_audit.md).

## GenBio-PathFM, Midnight, and OpenMidnight: protocol-preserving integration

The Group B audit preserves 107 observations in three model-specific evidence
snapshots. It does not reinterpret later leaderboard values as if they were the
same runs:

- The official [GenBio-PathFM paper](https://genbio.ai/papers/genbio-pathfm.pdf)
  contributes 49 observations. Twenty-one are new public leaf candidates: 12
  THUNDER kNN/F1 cells, three PathoROB balanced-accuracy cells, and six signed
  APD endpoints. The APD endpoints remain raw-only because their signed metric
  has no source-defined bounded normalization. Nine rounded HEST cells and
  three PathoROB RI cells are alternate evidence for higher-precision suite
  leaderboard rows. Ten HEST-joint observations and six aggregates are
  quarantined.
- The official [MICCAI 2025 Midnight paper](https://papers.miccai.org/miccai-2025/paper/4651_paper.pdf)
  contributes 42 observations: 12 EVA leaves for each of Midnight-12k,
  Midnight-92k, and Midnight-92k/392, plus two aggregates per checkpoint. The
  24 base-checkpoint leaves are candidates under a versioned CLS+Mean protocol;
  the paper disables early stopping except for Camelyon16 and PANDA. The 392px
  checkpoint is explicitly high-resolution post-training and is quarantined,
  as are all aggregates. These values differ from later EVA repository and
  Hugging Face report snapshots, so none is silently overwritten or averaged.
- The official [OpenMidnight technical report](https://sophont.med/blog/openmidnight)
  contributes 12 own-model EVA leaves under a CLS-only protocol, supported by
  the pinned [OpenMidnight evaluation configs](https://github.com/MedARC-AI/OpenMidnight/tree/4c3e4a83802010f47dc68bb2d25629f2b6f58eea/eval_configs).
  Its HEST and cross-suite means are excluded. The results prose reverses the
  labeled BreakHis and Cam16-small values; both prose claims are retained as
  conflict evidence and excluded in favor of the unambiguous table columns.
  Prior-model rows are secondary transcriptions and are not imported.

The reproducible extractor is
[`scripts/extract_group_b_official_scores.py`](../scripts/extract_group_b_official_scores.py).
The disposition ledgers are
[`source_data/genbio_pathfm_official_2026.csv`](../source_data/genbio_pathfm_official_2026.csv),
[`source_data/midnight_miccai2025_official_scores.csv`](../source_data/midnight_miccai2025_official_scores.csv),
and
[`source_data/openmidnight_technical_report_2025.csv`](../source_data/openmidnight_technical_report_2025.csv).
Each selected leaf carries a versioned evaluation ID plus a deduplication key.
The registry adapter adds 57 cells over 45 protocols; the six signed APD cells
remain raw-only. At the current ≥3 scores/model and ≥5 models/evaluation matrix
filters, these source-specific protocols add no matrix columns yet because each
has fewer than five reported models.

## Hibou, MUSK and GPFM: 185 integrated public leaf cells

Wave F preserves 397 official-source observations without folding incompatible
protocols together. Hibou contributes 18 public frozen leaves (nine each for
Hibou-B and Hibou-L); its 20 PanNuke CellViT-Hibou-L values are task-fine-tuned
quarantine. MUSK contributes 68 public retrieval, zero-shot, 10-shot, linear,
biomarker and prognosis leaves. Four immunotherapy results are controlled-access
and 54 model-size/ablation cells are aggregates. Nine exact PathVQA cells are
whole-model-fine-tuned quarantine, including MUSK-base 73.01 and MUSK-large
73.21; small/base/large and ablations use distinct component IDs. Of six named
zero-shot datasets, four have exact Table 4 leaves, while NCT-CRC and SICAPv2
are graph-only/unlocated. Melanoma-relapse leaves likewise remain graph-only.
None is visually approximated.

GPFM contributes 99 public frozen-feature leaves across WSI ABMIL, survival
ABMIL, ROI linear classification and ROI retrieval. All eleven publisher
source-data workbooks were digest-checked and sheet-inventoried. Seventy-five
cells use internal or public-status-unlisted cohorts, 43 VQA/report-generation
cells are task-fine-tuned, and seven cross-dataset averages are excluded. The
1,000-row bootstrap sheets support uncertainty intervals but are not counted as
independent tasks. The deterministic extractor and complete adjudication are in
[`scripts/extract_wave_f_official_scores.py`](../scripts/extract_wave_f_official_scores.py)
and
[`source_data/wave_f_source_audit.md`](../source_data/wave_f_source_audit.md).

## CONCH, CONCHv1.5, Phikon and Phikon-v2: 417 integrated public leaves

The Wave E audit preserves 634 observations and integrates only the 417 public
primary-source leaves. CONCH's Nature Medicine supplement contributes 104 cells: 77 public
zero-shot classification, frozen supervised classification, public retrieval,
and zero-shot segmentation leaves; 16 private Source A/B retrieval cells; and
11 captioning or end-to-end fine-tuned cells. Prompt ensembles and single-prompt
zero-shot runs receive different evaluation IDs.

CONCHv1.5 is treated as the TITAN patch encoder, not as CONCH v1 or as the TITAN
slide encoder. Its supplement contributes 433 cells: 297 public candidates from
logistic regression, SimpleShot, kNN, ABMIL, few-shot and slide-retrieval
protocols; 124 OT108, renal/CRANE, MGB/MGH or other internal cells; and 12
task-finetuned cells. Each probe, cohort and shot count remains explicit.

The Phikon-family snapshot contributes 47 observations. Forty-three are public:
Phikon's ABMIL slide tasks, PAIP external validation and patch linear-probe
cells, plus public cohort-specific Phikon/Phikon-v2 values from the Phikon-v2
paper. Four Cy1/NGX1 cells are private and quarantined. Table 4's repeated Cy1
and PAIP entries are not duplicated.

The separate CTransPath snapshot contains 50 public-dataset comparator cells
reported by the official CONCH and Phikon-family papers. They are all marked
secondary-only, and the nine end-to-end Gleason cells are additionally
fine-tuned. The official CTransPath repository exposes the checkpoint and code
but no numeric results, while the original publisher tables could not be
acquired as a first-party artifact in this environment; no comparator cell is
silently promoted to primary evidence. Details, source hashes, and the exact
blocker are recorded in
[`source_data/wave_e_source_audit.md`](../source_data/wave_e_source_audit.md).

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

## PathoROB: complete APD and clustering paper tables recovered

The published Nature Communications article supplies a 20.2 MB
[Source Data workbook](https://pmc.ncbi.nlm.nih.gov/articles/instance/13260997/bin/41467_2026_73923_MOESM4_ESM.xlsx)
through PMC. The workbook is pinned by SHA-256
`07456f3ffc5270ea1d8d48a8f82c08a5be396c88f99cc0227968dad721943047`.
The reproducible standard-library extractor is
[`scripts/extract_pathorob_scores.py`](../scripts/extract_pathorob_scores.py),
and its 122-row audit snapshot is
[`source_data/pathorob_nature2026_and_repo_examples.csv`](../source_data/pathorob_nature2026_and_repo_examples.csv).

Two complete published score blocks were previously missing:

- `Fig-3d-correlation_apds_x_ri` provides 20 models by two distinct aggregate
  endpoints, APD-ID and APD-OOD: 40 canonical means. APD is the signed relative
  accuracy change from the balanced training split, averaged across the
  nonbaseline correlation splits, 20 repetitions, and Camelyon, TCGA 4×4, and
  Tolkach ESCA. Zero means no performance drop; the paper states that
  increasingly negative values indicate worse drops and describes
  higher/closer-to-zero values as better. The workbook's 60 corrected
  observations per model/endpoint reproduce every summary mean and the reported
  95% confidence interval rule.
- `Fig-6b-clustering` provides 20 models by three dataset protocols: 60
  canonical means and standard deviations over 50 random initializations. The
  score is `ARI(biological labels) - ARI(medical-center labels)` and the paper
  declares an approximate `[-1,1]` domain with higher values better. These cells
  therefore use the auditable normalization `(score + 1) × 50`.

The APD means are fully retained in `data/scores.csv`, with their exact signed
percent values and explicit APD-ID/APD-OOD protocol rows. They are not admitted
to the factor matrix: neither the paper nor project policy defines a bounded
common-scale normalization, so their `normalized_score` is intentionally blank
and their audit status is `parsed_primary_source_analysis_ineligible`. No
clipping, empirical rescaling, or arbitrary logistic scale is introduced.

The pinned PathoROB repository contains 22 additional example results for only
UNI2-h and Phikon-v2: 12 per-dataset APD values, four aggregate APD values, and
six clustering values. Several differ slightly from the final paper workbook.
They remain source- and version-specific rows in the audit snapshot under
`pathorob.repoexample2026.*`; they are not merged with the final paper values or
inserted into `data/scores.csv`.

The resulting raw registry is 4,013 cells. After the complete evidence-wave
ingestion and the fixed support filter, the accepted factor matrix is 59 models
× 187 protocols with 2,122 observed cells. The PathoROB APD protocols and other
analysis-ineligible rows remain raw-registry-only. This inclusion decision is
made before downstream completion experiments are regenerated.

## Deduplication and validation boundaries

### Focused Virchow2G / Prov-GigaPath / TITAN primary-paper staging

The official-PDF extractor
[`scripts/extract_group_c_official_scores.py`](../scripts/extract_group_c_official_scores.py)
produces a 737-cell public snapshot and a 346-cell quarantine ledger.
Its source digests are pinned for Virchow2 (`41054d…8125`), the Prov-GigaPath
Nature supplement (`b82791…866b`), and the TITAN Nature Medicine supplement
(`26321e…1a70`). The public rows comprise 108 Virchow2-family cells and 629
TITAN/TITAN-V cells. They are included in the raw-registry total above; the
fixed audit and support filters determine which protocols enter the matrix.

The staging boundary is deliberately conservative: private PanMSK and OT108,
all reported aggregates, and task-specific fine-tuning are excluded from the
public snapshot. Prov-GigaPath contributes zero eligible primary-paper cells
because all 26 Supplementary Table 2 values fine-tune the LongNet slide
encoder; its five public TCGA-LUAD values remain fully transcribed in quarantine
rather than being silently dropped. TITAN-V and TITAN are distinct model IDs,
and Virchow2G CLS/CLS+Mean plus random-forest HEST are distinct protocols. The
114 public evaluation IDs share task identities where dataset and target match,
so secondary metrics and probe variants can be deduplicated during benchmark
selection without overwriting their source values.

Deduplication happens at explicit levels:

- identical source evidence for the same `model_id × evaluation_id` is resolved
  by a declared precedence rule and retained as a conflict record, as with the
  110 EVA overlaps;
- the same dataset, target, and observation granularity links task identities,
  but distinct splits, probes, preprocessing, and metrics remain distinct
  evaluation protocols;
- aliases may identify the same base model, but a patch encoder plus mean
  pooling is not collapsed into a slide encoder with the same family name;
- convenience leaderboard averages, rankings, and other derived summaries are
  excluded unless the paper explicitly defines the aggregate as a primary
  endpoint, as it does for PathoROB APD-ID/APD-OOD.

Every ingested score is marked as either `machine_parsed_single_source` or
`machine_parsed_single_primary_source`, depending on its source adapter.
Structural validation checks source hashes, table shape, exact task and model sets,
duplicate keys, finite values, metric domains, and source locators. It does not
establish experimental reproduction or resolve unreported checkpoint commits.

### Wave D exhaustive UNI and comparator extraction

The pinned Wave D snapshots add 481 public leaves: 227 from every active UNI
primary-paper result row, 95 from the first-party H-optimus-1 report, 144 from
Virchow2 Tables 2-3 across its four released comparator models, and 15 from the
original Virchow paper. A further 179 cells remain visible in quarantine or as
excluded aggregates. The UNI audit expands 83 source rows to 277 metric leaves
and keeps all resolution, stain normalization, KNN, retrieval, cohort, ABMIL,
MI-SimpleShot, and metric variants separate. Exact task identities link known
CRC-100K, BACH, UniToPatho, TCGA-MSI, TCGA-Uniform, TCGA-TILs, CAMELYON16,
BRACS, and PANDA overlaps without overwriting their suite-specific values.

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
