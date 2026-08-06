# H0-mini and UNI2-h official-source audit

Extraction date: 2026-08-06. Only first-party model cards, papers, and
repositories were used. The numeric audit snapshot is
`h0mini_uni2h_official_scores_2025.csv`.

| Source | Pinned revision | Acquired artifact SHA-256 | Numeric disposition |
|---|---|---|---|
| [H0-mini arXiv v3](https://arxiv.org/abs/2501.16239v3) | `arxiv:2501.16239v3` | source archive `222798059c15b554528d61f8caa04de8fcc2d5cc23997607dc25d851282a6f08`; `main_preprint.tex` `861293a774f0cb8cd2a5971c4aa9b1f4682c3fb60867a22509d4b84f6506825e` | Tables 1–3: 46 public leaf cells and four excluded means. Table 4: 18 private BreastBm cells quarantined. |
| [PLISM benchmark repository](https://github.com/owkin/plism-benchmark/tree/5ec9511893af993f6faa099f093d1924b291aed2) | `5ec9511893af993f6faa099f093d1924b291aed2` | `README.md` `d1715234a41f8da728ad669560bbcfc5253680db3403b400673d6a40a3955a64` | Eight public leaf cells accepted under repository-version protocols; two leaderboard means and ranks excluded. |
| [UNI repository](https://github.com/mahmoodlab/UNI/tree/42715efc11722a496e0a67f3369505a8f277206c) | `42715efc11722a496e0a67f3369505a8f277206c` | `README.md` `4ac024c83dbcdc39987a81f4983474b0e6c6f15352226677809a6fb492f9cdb8` | Six public leaf cells retained raw without inferred metrics; one HEST mean excluded; one internal IHC cell quarantined. |
| [H0-mini model card](https://huggingface.co/bioptimus/H0-mini/tree/5b5cc0505d19ae558270045eb0df8c34df4d9609) | `5b5cc0505d19ae558270045eb0df8c34df4d9609` | `README.md` `b7c25b15da884cee3439d8eb54aa2f0528adc4f2da2ebbc9d7ea545849f00a5d`; model API JSON `63d2165a80aa8082979ddb9ae9215443acb1293fdc90194ad5a9482310ced94b` | No numeric benchmark table. Used to pin component identity and link the paper. |
| [UNI2-h model card](https://huggingface.co/MahmoodLab/UNI2-h/tree/d517a8dd47902dd7c308b3c36f63bce47e7b9a43) | `d517a8dd47902dd7c308b3c36f63bce47e7b9a43` | `README.md` `d0c283892c0fddc0b5571c02d37ff7c923647b6381c3f2fcd970a3484322b244`; model API JSON `9975d8e39bfbb673ce165b128f6a37a1c10e3b07be17bad212d66f8e1ea8e29b` | No numeric benchmark table. Used to pin component identity and official repository link. |

Important adjudications:

- The H0-mini paper contributes 23 public leaf cells per model, 46 total. The
  later PLISM repository adds eight public cells, and the UNI repository adds
  six, explaining the complete 60-cell public total.
- PLISM paper and repository values are separate report versions. They are not
  averaged or selected by precedence. Only the three matched top-10 endpoints
  share task identities for link-only deduplication.
- The final top-10 header in the paper contains a down-arrow inconsistent with
  the metric definition, prose, and bolded winners. Direction is recorded as
  higher-is-better, and the source typo remains documented.
- BreastBm is explicitly private. The UNI IHC column is explicitly internal.
  Both remain audit-only.
- UNI's README reports a selection recipe but does not name endpoint metrics.
  Its six public leaf values are retained with blank normalization and cannot
  enter the factor matrix.
