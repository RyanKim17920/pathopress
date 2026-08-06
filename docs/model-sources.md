# Foundation-model source and extraction backlog

`data/model_sources.csv` is the component-level source ledger. It deliberately separates families from runnable components: UNI from UNI2-h, Virchow2 from Virchow2G, CONCH v1 from v1.5, Prov-GigaPath tile from LongNet slide, and TITAN-V vision-only from TITAN vision-language. A row marked `seed_table_extracted` means this repository already contains at least one result for that model from HEST, PathoROB or THUNDER; it does **not** mean the primary paper is fully extracted. `paper_backlog` means there is no component-specific seed result yet. The `primary_paper_*` statuses record focused extraction or a completed eligibility audit and name remaining work explicitly.

The focused Virchow2-family / Prov-GigaPath / TITAN audit now preserves 1,083
exact primary-paper cells: 737 public protocol-specific cells enter the registry
and 346 are quarantined. Virchow2G, Virchow2, and Virchow each contribute 36
public comparator leaves, with CLS and CLS+Mean separated and HEST explicitly
labeled random forest. TITAN and TITAN-V contribute 629 cells across public
morphology, grading, molecular, survival, full/few/zero-shot classification,
slide retrieval, rare-cancer retrieval, and TCGA report/slide cross-modal
retrieval protocols. All 26 Prov-GigaPath Supplementary Table 2 cells remain in
quarantine because they use task-specific slide-encoder fine-tuning; 21 also use
private Providence cohorts. The report-aligned zero-shot experiment is likewise
Providence-only and reported only as unlabeled graphical bars, so no exact
numeric leaves can be extracted without inventing precision. No official
Virchow2G checkpoint was found, so its revision remains
`paper_only_unreleased_checkpoint`.

The Prov-GigaPath tile-encoder primary-source audit is also closed with no
eligible official numeric leaf. Supplementary Figure 4 varies the tile
pretraining method on public TCGA-LUAD, but evaluates it through a downstream
LongNet whole-slide stack and publishes graphical bars without exact labels.
The five exact public TCGA-LUAD cells in Supplementary Table 2 freeze the tile
encoder while task-specifically fine-tuning LongNet, ABMIL, and the classifier;
they therefore remain slide-stack quarantine evidence. The official repository
and Hugging Face card provide a runnable PCam frozen-feature linear probe and a
2.33 GB embedding archive, but neither publishes an expected score or result
artifact. The exhaustive 11-item disposition ledger is
[`source_data/prov_gigapath_tile_evidence_audit_2024_2026.csv`](../source_data/prov_gigapath_tile_evidence_audit_2024_2026.csv).

The UNI2-h and H0-mini P0 source audit is complete for the current official
model cards, H0-mini arXiv v3, PLISM repository leaderboard, and UNI repository
benchmark tables. It preserves 86 observations: 60 public leaf cells enter the
raw registry, seven derived aggregates are excluded, and 19 private/internal
cells are quarantined. The six UNI-repository leaf values remain
analysis-ineligible because the report does not name their endpoint metrics.

The Wave D official-source audit is complete for UNI, Virchow, Virchow2,
H-optimus-0, and H-optimus-1. The UNI primary-paper source contains 83 active
UNI result rows, expanded into 277 metric leaves: 227 public frozen-feature
cells enter the registry, 24 explicitly fine-tuned Mask2Former cells and 23
in-house BWH cells are quarantined, and three averages are excluded. The
Bioptimus report preserves 170 cells (95 public, 60 internal/access-ambiguous,
15 aggregates); Virchow2 preserves 192 (144 public, 24 PanMSK, 24 aggregates);
and the original Virchow paper preserves 21 (15 public, six internal or
non-downloadable). Exact resolution, stain, KNN, retrieval, ABMIL, cohort,
embedding-recipe, metric, and HEST estimator settings remain separate.

The GenBio-PathFM, Midnight, and OpenMidnight P0 evidence audit is also
complete. It preserves 107 observations from the official GenBio PDF, MICCAI
Midnight paper, and OpenMidnight technical report: 51 analysis-eligible
versioned candidate leaf cells, six signed APD leaves that remain
analysis-ineligible, 12 rounded alternate-evidence cells, and 38 quarantined fine-tuned, aggregate, or
internally contradictory observations. The staged rows are intentionally not
collapsed into current suite leaderboard protocols: GenBio reports THUNDER
kNN, Midnight reports CLS+Mean with a modified early-stopping policy, and
OpenMidnight reports CLS-only results from its EVA fork.

The Wave E CONCH/Phikon audit stages 634 additional observations: 417 public
primary-paper leaf candidates and 217 quarantined cells. CONCH contributes 77
public leaves, CONCHv1.5 contributes 297, and the Phikon family contributes 43.
Private Source A/B, OT108, renal/CRANE, MGB/MGH, Cy1 and NGX1 cohorts are
excluded, as are task-finetuned rows. Fifty CTransPath comparator cells are
preserved only as secondary evidence because its official repository contains
no numeric result table and the original publisher table artifact could not be
acquired; none is promoted to the public candidate set.

The Wave F Hibou/MUSK/GPFM audit stages 397 observations: 185 public
protocol-specific candidates and 212 quarantined cells. Hibou-B and Hibou-L
remain distinct and contribute nine frozen leaves each; all 20 CellViT-Hibou-L
cells are task-fine-tuned quarantine. MUSK contributes 68 public retrieval,
zero-shot, 10-shot, linear, biomarker and prognosis leaves; four controlled
immunotherapy cells, 54 model-size/ablation aggregates, and nine exact
whole-model-fine-tuned PathVQA cells are excluded. Small/base/large and ablation
component IDs remain distinct. Four of six zero-shot datasets have exact Table
4 leaves; NCT-CRC and SICAPv2 are explicitly graph-only/unlocated. GPFM
contributes 99 public frozen WSI, survival, ROI-linear and retrieval leaves.
Its audit exhausts all eleven publisher source-data workbooks; 75 unlisted or
internal-cohort cells, 43 task-fine-tuned VQA/report cells, and seven aggregates
remain quarantined. Raw bootstrap draws are uncertainty provenance, not tasks.

## Extraction order

1. **TITAN audit complete for exact public leaves.** The broad morphology, grading, molecular, survival, zero-shot, rare-cancer, and cross-modal retrieval tables are extracted. OT-108, renal-allograft, CRANE, MGB/MGH, private rare-cancer, fine-tuned rows, and aggregates remain quarantined; report generation stays analysis-ineligible because the public checkpoint omits the decoder.
2. **Prov-GigaPath exact-number limitation recorded.** Supplementary Table 2 is fully accounted for but has no eligible frozen public rows. The report-aligned zero-shot variant uses the proprietary Providence holdout and publishes graphical bars without exact labels; it must not be approximated into the registry.
3. **Virchow2-family comparator audit complete.** Virchow2G, Virchow2, and Virchow columns from Tables 1-3 are preserved without overwriting suite-specific rows. PanMSK stays private; CLS, CLS+Mean, weighted-F1 and random-forest HEST remain separate from ridge/PCA HEST.
4. **CONCH and Phikon-family audit complete.** Exact CONCH, CONCHv1.5, Phikon and Phikon-v2 leaves are staged under versioned prompt/probe/MIL/few-shot/retrieval protocols. Cy1 and NGX1 are private despite appearing beside public MSI cohorts. CTransPath secondary comparators remain quarantined pending an accessible first-party original table.
5. **Hibou, MUSK and GPFM audit complete.** Exact checkpoint identities and protocol families are staged. MUSK PathVQA values are exact but whole-model-fine-tuned quarantine; NCT-CRC/SICAPv2 zero-shot and melanoma-relapse leaves remain graph-only/unlocated and are not approximated. GPFM's eleven-workbook inventory and public-cohort boundary are explicit.
6. **Remaining older canonical papers.** CTransPath's original publisher tables remain blocked. Re-audit MUSK only if first-party graph source data become available.

## Hard comparison rules

- A model family name is not a score-bearing identity. Every result needs the exact checkpoint/component, input resolution or magnification, embedding recipe, downstream head, split and metric.
- Tile, ROI and slide encoders are separate components. In particular, an external ABMIL model trained on Prov-GigaPath tile features is not the pretrained Prov-GigaPath LongNet slide encoder.
- Zero-shot prompting, few-shot prototypes, frozen linear probes, k-NN, random forests, ABMIL and fine-tuning are different protocols even on the same dataset.
- Public-data pretraining is not automatically leakage-free. TCGA, PAIP, CPTAC and public web/image-text corpora recur in both pretraining and evaluation. Prefer released slide/tile manifests and hashes; otherwise mark contamination `unknown` or `possible`.
- Internal benchmarks may be catalogued, but they should not be compressed into the same auditable leaderboard as downloadable public tasks.
- Aggregate means and ranks are views, not tasks. Extract leaf results first and recompute aggregates only over an explicit eligible set.

## Source policy

The ledger links only author/publisher primary papers and official model/code releases. Community Hugging Face mirrors are excluded. Exact values enter only from a named table or machine-readable official artifact with a pinned digest and provenance record.
