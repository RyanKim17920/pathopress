# Wave F official-source audit: Hibou, MUSK, and GPFM

Audit date: 2026-08-06. This is an evidence-layer extraction only; it does not rebuild the shared registry, matrix, figures, website, or public release.

## Result

| Snapshot | Rows | Canonical candidates | Quarantined |
|---|---:|---:|---:|
| `hibou_official_scores_2024.csv` | 38 | 18 | 20 task-fine-tuned |
| `musk_official_scores_2025.csv` | 135 | 68 | 54 aggregate/ablation; 9 PathVQA fine-tuned; 4 controlled-access |
| `gpfm_official_scores_2025.csv` | 224 | 99 | 75 private/internal or unlisted; 43 task-fine-tuned; 7 aggregate |
| **Total** | **397** | **185** | **212** |

Canonical candidates are public, exact benchmark leaves with a frozen foundation encoder (a trained linear, attention-pooling, survival, or nearest-neighbour downstream component is allowed and named). Cross-dataset averages, raw bootstrap replicates, task-specific end-to-end systems, and non-public cohorts are not canonical candidates.

## Pinned first-party sources

- Hibou paper: `arXiv:2406.05074v1`, SHA-256 `5c4086cad4dfa47ae6699a53149362a9a593785830b56dcc4971a72fe95d5fe5`; official repository revision `c453bbe4dab0fec6f7df343b09ea87048629c58d`.
- MUSK Nature supplementary PDF (`41586_2024_8378_MOESM1_ESM.pdf`), SHA-256 `f96a44a9e3b531472a166a0a06dc3dec241b3f75fc455a21479c798f6028b770`; official repository revision `714b666969c1911e5efe70d991140a21030f4ef3`.
- GPFM Nature Biomedical Engineering supplementary PDF (`41551_2025_1488_MOESM1_ESM.pdf`), SHA-256 `7fb834ee12f33fcdd369fe8f218c82a13523f12401f3bac8d24c10bee2f77b2f`; official repository revision `4f8be2b1f163f99ba35931f35abe60dc836f17da`.

Official source URLs are retained on every row. Repository revisions validate model/protocol identity; publisher tables/workbooks control the reported values.

## GPFM workbook exhaustion

The publisher page exposes **eleven**, not seven, source-data XLSX files. All were downloaded, digest-checked, and sheet-inventoried:

| Workbook | SHA-256 | Audit role |
|---|---|---|
| MOESM4 | `0431fdab9f8c5f410fb0c5ae77103a86158ecce961d6631a291f8f086a2bfb97` | Fig. 1 task/radar summary; aggregate/rank sheets |
| MOESM5 | `145f5a913a36d79a8fd13b9b4783817e5cad3dcdbeecf4fc69857cdceff184c4` | Fig. 2 task means and ranks |
| MOESM6 | `97742abaaefebc70435d04948eaa9d4d4d6b9447b982baf26d587fb85d5f0d76` | Fig. 3; 36 WSI cohorts and detailed source sheets |
| MOESM7 | `544dbdc7a0e2cccf0642da377af682b3b979aee765e88d78e6c13e1d3965fb2d` | Fig. 4; 15 survival cohorts plus bootstrap source sheets |
| MOESM8 | `0f62bd4e74a8736286a82d562d122603bce49ea2b37daac2f84bcb4706f2dfd9` | Fig. 5; 16 ROI cohorts plus bootstrap source sheets |
| MOESM9 | `71aa5171037b97c36d8b9a1f1222c11ed6a0f79671393ac6265a218a207c76c9` | Fig. 6; retrieval, PathVQA, and report generation |
| MOESM10 | `10b4f275bfb98762c883b89b173c87d187e6a32c75f35352b19f44b3d9530b71` | Extended Data Fig. 1 |
| MOESM11 | `acabd754d8a63594314bce1ae3281041fd5a2401da602929a6723a7886f93122` | Extended Data Fig. 2 |
| MOESM12 | `71777ead89466460982c295109dfd3ecf0e7a9770265b2259ffc715b02387a67` | Extended Data Fig. 3 |
| MOESM13 | `5d33aa355859a25b7f133865904851e87aa58beb4afcdb259dc0bb36e56996f3` | Extended Data Fig. 5; WSI-VQA |
| MOESM14 | `56a270d8e4dcce21a63e98d540b47e3a0b97c68cca7e838595138cca75ab7fa1` | Extended Data Fig. 8; organ-specific report generation and human scores |

The source workbooks contain both published means and 1,000-row bootstrap distributions. Bootstrap draws are uncertainty provenance, not independent benchmark observations. Published means are taken from the workbooks and aligned to the corresponding supplement confidence intervals. Fig. 1/Fig. 2 rank and mean sheets are audited but not duplicated as canonical performance rows.

## Model- and protocol-specific decisions

### Hibou

- Hibou-B and Hibou-L remain separate model IDs and checkpoint families.
- Table 1 contributes six public patch-level linear-probe leaves per checkpoint.
- Table 2 contributes three public slide-level frozen-encoder attention-pooling leaves per checkpoint.
- Tables 3–4 contribute 20 exact CellViT-Hibou-L segmentation cells, all quarantined because Hibou-L is trained as a task-specific CellViT backbone.
- Reported `AVG` rows are not independent tasks and are not emitted as benchmark leaves.

### MUSK

- Tables 1–3 preserve text-to-image, image-to-text, and image-to-image retrieval as different protocols and metrics.
- Zero-shot classification, 10-shot classification, and linear probing remain separate measurement families.
- Table 7 keeps biomarker AUC and prognosis c-index leaves distinct. The four lung/gastro-oesophageal immunotherapy leaves are exact but controlled-access and quarantined.
- Tables 13–15 and 17 contribute 63 model-size/ablation observations. Fifty-four cross-dataset cells are aggregates. Nine exact PathVQA cells—including MUSK-base 73.01 and MUSK-large 73.21—are separately quarantined because the methods explicitly fine-tune the complete backbone and classification head. Small, base, large, and each ablation use distinct component IDs.
- The zero-shot classification experiment names six datasets. Table 4 provides exact leaves for PatchCamelyon, SkinCancer, PanNuke, and UniToPatho. NCT-CRC and SICAPv2 are named in the zero-shot experiment but their exact leaf values are graph-only/unlocated in the supplement and official repository; they are explicitly dispositioned by `MUSK_ZERO_SHOT_DISPOSITION` and no bar is visually approximated.
- Exact melanoma-relapse values remain graph-only in the publisher material and no publisher source workbook is available. The repository contains evaluation code/example data but no first-party numeric result file. No visually estimated bar value is promoted.

### GPFM

- WSI ABMIL, survival ABMIL, ROI linear classification, and ROI nearest-neighbour retrieval use frozen GPFM features and are protocol-separated.
- Public access is controlled by Supplementary Table 46. Named NFH/QFS/SAL/FUSAN/FUYI cohorts and other cohorts absent from that public-data inventory are quarantined.
- PathVQA, WSI-VQA, and report generation values are preserved but quarantined as task-specific fine-tuned/generative systems.
- Supplementary Tables 1, 19, and 24 cross-dataset means are retained as `aggregate_excluded` and never treated as independent tasks.

## Reproduction

Run:

```bash
python3 scripts/extract_wave_f_official_scores.py \
  --hibou-pdf /tmp/pathopress_wave_f/hibou/2406.05074.pdf \
  --musk-supplement /tmp/pathopress_wave_f/musk/41586_2024_8378_MOESM1_ESM.pdf \
  --gpfm-supplement /tmp/pathopress_wave_f/gpfm/41551_2025_1488_MOESM1_ESM.pdf \
  --gpfm-workbook-dir /tmp/pathopress_wave_f/gpfm \
  --output-dir source_data
```

The extractor fails closed on source digest or expected-table-shape changes. `tests/test_wave_f_official_scores.py` checks row/status counts, representative cells, checkpoint separation, protocol boundaries, public-access safety, unique selected IDs, and byte-for-byte regeneration when the pinned artifacts are present.
