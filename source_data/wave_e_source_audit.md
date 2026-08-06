# Wave E official-source score audit

This audit stages evidence for CONCH, CONCHv1.5, Phikon, Phikon-v2, and
CTransPath without rebuilding the shared score registry, matrix, factor model,
or figures. Every row has an exact model identity, task/protocol identifier,
metric, cohort-access disposition, and source locator. Repeated protocols remain
separate observations until the coordinated deduplication pass.

## Pinned sources

- CONCH: [Nature Medicine article and Supplementary Information](https://www.nature.com/articles/s41591-024-02856-4), supplementary PDF SHA-256
  `4940cf23c1f341791fd16db84e6f22a83d90d1c67657c26cb118ad8fcbbef457`.
- CONCHv1.5: [TITAN Nature Medicine article and Supplementary Information](https://www.nature.com/articles/s41591-025-03982-3), supplementary PDF SHA-256
  `26321e4018bec7b80f2fe7ea7cc497139c83b44fb60df5128417623ad1f71a70`;
  official TITAN repository revision `9e34c66ff66445c6c590da0dbf7acc103d39a40b` was used to cross-check the
  public/internal labels shown in its benchmark table.
- Phikon: [MedRxiv v3 article](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v3)
  and the first-party supplementary PDF asset, SHA-256
  `dd73798e6b34e317a205bec036318388d60c33da3af4326690f51ca357b96faa`.
  The supplement supplies ABMIL-specific Tables F1/F2 and linear ROC-AUC Table
  G3, avoiding the ambiguous headline “best across five MIL algorithms” table.
- Phikon-v2: [arXiv 2409.09173v1](https://arxiv.org/abs/2409.09173v1), PDF SHA-256
  `52e26d832d27a6d51ce2b68da462f6a6ad851236fc1efafb9b1258359fc6b633`.
- CTransPath: [Medical Image Analysis DOI](https://doi.org/10.1016/j.media.2022.102559)
  and official repository revision `26c61940312dbb239351eff46fe08c884e9e5c3e`.
  The repository contains code/checkpoint metadata but no machine-readable
  primary result table. The publisher-hosted full table artifact could not be
  acquired in this environment. Consequently, the CTransPath snapshot contains
  only clearly labeled secondary comparator cells from the official CONCH and
  Phikon-family papers, and all 50 are quarantined.

## Disposition summary

| Snapshot | Observations | Public primary candidates | Quarantined |
|---|---:|---:|---:|
| `conch_official_scores_2024.csv` | 104 | 77 | 16 private Source A/B; 11 fine-tuned |
| `conch15_titan_official_scores_2025.csv` | 433 | 297 | 124 private/internal; 12 fine-tuned |
| `phikon_family_official_scores_2023_2024.csv` | 47 | 43 | 4 private Cy1/NGX1 cells |
| `ctranspath_official_evidence_2022_2024.csv` | 50 | 0 | 50 secondary-only (9 also fine-tuned) |

The 634 staged observations therefore contain 417 candidate public primary
leaves and 217 quarantined observations. “Candidate” is not a claim of leakage
freedom: CONCH prompt-based zero-shot, supervised frozen-feature, retrieval and
segmentation rows remain distinct; CONCHv1.5 logistic regression, SimpleShot,
kNN, ABMIL, few-shot, and retrieval protocols remain distinct; and Phikon TCGA
downstreams retain their pretraining-overlap caveat.

For Phikon-v2 Table 2, ER, PR, HER2, IDH1, ISUP, Camelyon16 metastasis, PAIP MSI,
and DHMC RCC cells are public. Cy1 is explicitly private. Table 4 contributes
NGX1 only after removing its repeated Cy1/PAIP leaves, but NGX1 is also private,
so both Phikon and Phikon-v2 NGX1 cells are quarantined.

The executable extraction and digest checks are in
[`scripts/extract_wave_e_official_scores.py`](../scripts/extract_wave_e_official_scores.py),
with structural and exact-value tests in
[`tests/test_wave_e_official_scores.py`](../tests/test_wave_e_official_scores.py).
