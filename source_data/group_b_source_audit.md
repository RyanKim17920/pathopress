# Group B official-source audit

This ledger records the primary sources used for GenBio-PathFM, Midnight, and
OpenMidnight. The numerical CSVs are evidence snapshots, not claims that these
runs are interchangeable with current suite leaderboards.

## Pinned sources

| Model | Official source | Pinned identity | Numerical locator | Disposition |
|---|---|---|---|---|
| GenBio-PathFM | [Official paper PDF](https://genbio.ai/papers/genbio-pathfm.pdf) | SHA-256 `814ac10253ae55d737e02e1aad549e1995f05d90fd96c1fc2bc28a669996d9a1`; PDF created 2026-03-17 | Tables A1-A5 | Own-checkpoint public leaves extracted; rounded duplicates, aggregates, and HEST-joint results labeled separately |
| GenBio-PathFM | [Official GitHub repository](https://github.com/genbio-ai/genbio-pathfm) | commit `1d65d002f28f6a4b481ca6cc434267b453c30b60` | README/model implementation | Confirms release identity; repository has no numerical result table |
| Midnight | [MICCAI 2025 proceedings PDF](https://papers.miccai.org/miccai-2025/paper/4651_paper.pdf) | SHA-256 `29dce4004f4f4c2e8f75cb5fc2c03c934d97e20099329bb9682901c647bc07cb`; PDF modified 2025-08-25 | Table 2 and evaluation-method paragraphs | Three own checkpoint rows extracted; aggregates and high-resolution post-trained checkpoint quarantined |
| Midnight | [Official Hugging Face report](https://huggingface.co/kaiko-ai/midnight/blob/adc6b15679c981cce6f9b018bbad09d16eeeda9f/README.md) | revision `adc6b15679c981cce6f9b018bbad09d16eeeda9f` | Results Summary | Already handled by the EVA reconciliation; retained as a different report revision, not used to overwrite the MICCAI table |
| OpenMidnight | [Official technical report](https://sophont.med/blog/openmidnight) | `SOPHONT-TR-2025-001`, DOI `10.5281/zenodo.20711012`, published 2025-11-14, modified 2026-06-16; captured HTML SHA-256 `bf31b3c716dd59da9e230e75abd913b4809bd864ff89e38b752bd9b89f4c402b` | Performance table and Results prose | Only the own-model row is extracted; derived means and contradictory prose values are excluded |
| OpenMidnight | [Official GitHub repository](https://github.com/MedARC-AI/OpenMidnight) | commit `4c3e4a83802010f47dc68bb2d25629f2b6f58eea` | `eval_configs/*.yaml` | Twelve task configs pin the supporting EVA-fork protocol; per-file hashes are embedded in the evidence rows |
| OpenMidnight | [Official Hugging Face model](https://huggingface.co/SophontAI/OpenMidnight) | gated; exact public card revision could not be fetched anonymously | Public web card repeats the report table | Not used as the reproducibility anchor because anonymous raw access returns 401 |

## Protocol adjudication

- GenBio Table A3 is explicitly a frozen-encoder **kNN** evaluation reporting
  F1. It is not the same protocol as the current THUNDER linear-probe
  leaderboard. The 12 leaves therefore use `thunder.genbio2026.*.knn` IDs.
- GenBio Table A1 and Table A4 RI values agree with current HEST/PathoROB rows
  only at the paper's rounded precision. They remain alternate evidence, never
  averaged with the higher-precision leaderboard cells.
- GenBio Table A2 jointly trains across HEST training sets. Those observations
  are downstream-trained variants rather than frozen base-checkpoint evidence.
- Midnight Table 2 uses CLS+Mean embeddings. It disables EVA early stopping for
  every task except Camelyon16 and PANDA. The paper values differ from later
  EVA/Hugging Face reports, so the MICCAI protocol has its own IDs.
- Midnight-92k/392 is described as 120k iterations of high-resolution
  post-training/fine-tuning. Its leaves remain in the evidence CSV but are not
  candidates for the frozen base-checkpoint matrix.
- OpenMidnight reports CLS-only embeddings and ships a customized EVA fork.
  Its versioned report IDs must not be collapsed into either the current EVA
  leaderboard or the MICCAI Midnight CLS+Mean protocol.
- The OpenMidnight table labels BreakHis as `0.873` and Cam16-small as `0.946`.
  The adjacent prose assigns `0.946` and `0.873` respectively. The labeled
  table cells are unambiguous; the swapped prose claims are retained with
  `narrative_conflict_excluded` status.
- OpenMidnight states that all prior-model comparison rows come from Karasikov
  et al. Those secondary transcriptions are not imported; only the report's own
  model row is eligible.

## Reproduction

With the three pinned source files and the OpenMidnight repository available,
run:

```bash
python3 scripts/extract_group_b_official_scores.py
PYTHONPATH=src python3 -m unittest tests.test_group_b_official_scores -v
```

The extractor fails closed on source hashes, report anchors, repository commit,
and the exact 12-file OpenMidnight config inventory.
