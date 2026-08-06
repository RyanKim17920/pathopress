# Wave D official-source audit

This audit covers UNI, Virchow, Virchow2, H-optimus-0, and H-optimus-1. Only exact numeric leaves from first-party reports, official model cards, and primary-paper source archives are materialized. Fine-tuned, internal, access-ambiguous, and derived aggregate cells remain in the evidence snapshots but outside the compression matrix.

| Source | Pinned evidence | Public leaves | Quarantined / excluded |
|---|---|---:|---:|
| H-optimus-1 first-party report | HTML SHA-256 `a46ad75f…997bf3`; H1 card `43e14486…dbc45e`; H0 card `a9110f5b…4077c` | 95 across H1, H0, UNI2-h, Virchow2, and UNI | 60 internal/access-ambiguous; 15 means |
| Virchow2 arXiv 2408.00738 | source archive SHA-256 `1ee7317a…f3fda` | 144 across Virchow2, Virchow, H0, and UNI | 24 PanMSK; 24 averages |
| Virchow arXiv 2309.07778 | source archive SHA-256 `b2c31918…ba6b0` | 15 Virchow public tile leaves | 6 PanMSK/non-downloadable biomarker leaves |
| UNI arXiv 2308.15474 | source archive SHA-256 `26da25ce…4e60` | 227 UNI frozen-feature leaves | 24 Mask2Former fine-tuned; 23 in-house BWH; 3 averages |

The UNI extractor accounts for all 83 active `UNI & ...` result rows in the official TeX source and expands them to 277 metric leaves. It preserves patch linear-probe, 1-NN, 20-NN, stain-normalization, resolution, image-retrieval, slide ABMIL, cohort, MI-SimpleShot, top-K, and metric variants as distinct protocols. The eight SegPath task rows are quarantined because the paper explicitly fine-tunes the encoder with Mask2Former; the SegPath average is not an independent task. OT-43, OT-108, and BWH endomyocardial results are quarantined because the paper identifies them as in-house data.

The Virchow2 adapter keeps CLS-only and CLS+Mean separate and never merges random-forest HEST with ridge/PCA HEST. The H-optimus report adapter accepts only cohorts whose public artifact status is clear from the official report; SLN-Breast, SURGEN, YALE-HER2, IMPRESS, BCNB, and FR-CRC-Bio remain quarantined rather than inferred public.
