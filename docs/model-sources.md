# Foundation-model source and extraction backlog

`data/model_sources.csv` is the component-level source ledger. It deliberately separates families from runnable components: UNI from UNI2-h, Virchow2 from Virchow2G, CONCH v1 from v1.5, and the Prov-GigaPath tile encoder from its LongNet slide encoder. A row marked `seed_table_extracted` means this repository already contains at least one result for that model from HEST, PathoROB or THUNDER; it does **not** mean the primary paper is fully extracted. `paper_backlog` means there is no component-specific seed result yet.

## Extraction order

1. **Prov-GigaPath Supplementary Table 2 (and Tables 6-9).** It is the highest-value missing block: 26 slide tasks, but the current seed rows represent the tile encoder on third-party suites. Extract nine subtype tasks and 17 pathomics tasks as distinct dataset/protocol endpoints. Keep Providence and TCGA evaluations separate, and keep the report-aligned zero-shot variant separate from the image-only model.
2. **TITAN main and supplementary tables.** Extract public TCGA-UT-8K, TCGA-OT and EBRAINS first; mark OT-108 and renal-allograft AMR internal. Record linear, few-shot, zero-shot, retrieval and report generation as different protocols. CONCH v1.5 is the patch encoder; TITAN is the slide encoder. The public checkpoint omits the report decoder, so report-generation results need a reproducibility flag.
3. **Virchow2 Tables 1-3.** Add Virchow2G, then audit existing Virchow/2 comparator results. Store PanMSK magnifications as separate tasks; store PCam, CRC, CRC-no-normalization, WILDS, TILS, MHIST, DLBCL and MIDOG separately. Preserve `CLS`, `CLS+Mean`, weighted-F1 and random-forest HEST protocol fields rather than mixing these values with suite leaderboard rows.
4. **Phikon-v2 Table 2 and supplement.** This is a clean, compact ten-cohort block covering eight clinical endpoints. The paper deliberately uses external validation to avoid its public pretraining cohorts. Do not collapse the two MSI cohorts or treat endpoint and cohort as synonyms. Preserve its explicit warning that CTransPath's PAIP result may be inflated by overlap.
5. **MUSK Nature tables and repository benchmark outputs.** Its 23 benchmarks span retrieval, VQA, zero/few-shot classification, linear probes and multimodal outcomes. They should enter separate protocol families, never one averaged endpoint. The repository provides the canonical dataset list and runnable benchmark configuration.
6. **GPFM source-data workbook.** The headline 39 tasks across six clinical types can substantially expand coverage, but the source-data tables must be normalized carefully. Tag teacher provenance (CONCH, Phikon and UNI), public-pretraining overlap, and internal/external validation cohorts.
7. **Older canonical papers.** Extract UNI's 34-task supplement, CONCH's 14-task supplement, H-optimus-0's official software-report tables, Hibou Tables 1-4, Phikon's 17-task supplement and CTransPath's original tables. These are useful for historical breadth after the component-confusion risks above are resolved.

## Hard comparison rules

- A model family name is not a score-bearing identity. Every result needs the exact checkpoint/component, input resolution or magnification, embedding recipe, downstream head, split and metric.
- Tile, ROI and slide encoders are separate components. In particular, an external ABMIL model trained on Prov-GigaPath tile features is not the pretrained Prov-GigaPath LongNet slide encoder.
- Zero-shot prompting, few-shot prototypes, frozen linear probes, k-NN, random forests, ABMIL and fine-tuning are different protocols even on the same dataset.
- Public-data pretraining is not automatically leakage-free. TCGA, PAIP, CPTAC and public web/image-text corpora recur in both pretraining and evaluation. Prefer released slide/tile manifests and hashes; otherwise mark contamination `unknown` or `possible`.
- Internal benchmarks may be catalogued, but they should not be compressed into the same auditable leaderboard as downloadable public tasks.
- Aggregate means and ranks are views, not tasks. Extract leaf results first and recompute aggregates only over an explicit eligible set.

## Source policy

The ledger links only author/publisher primary papers and official model/code releases. Community Hugging Face mirrors are excluded. No scores were transcribed into this backlog; exact values should be imported only from a named table or machine-readable official artifact with a provenance record.
