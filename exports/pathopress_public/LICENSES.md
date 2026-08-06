# Data provenance and licenses

PathoPress source code is MIT-licensed; see `LICENSE` in the project repository.

The exported tables contain factual identifiers, protocol metadata, citations,
and reported numeric results. They do **not** relicense benchmark datasets,
pathology images, labels, model weights, or source publications. Each upstream
artifact keeps its own license, terms, access restrictions, and attribution
requirements. Consult the linked upstream project before redistribution or
commercial use.

## Upstream benchmark suites

- Patho-Bench (`pathobench`): https://huggingface.co/datasets/MahmoodLab/Patho-Bench/tree/60fde3a9138b2fb27a163ed6f3e2cf0ef7e8f387
- EVA (`eva`): https://github.com/kaiko-ai/eva/tree/e43e74a99b75660b0014f790f25a33dd9f11e121
- THUNDER (`thunder`): https://github.com/MICS-Lab/thunder/tree/3d1cc9513fb2cfd8c4afb0d7bb9f5c4f6b69117f
- HEST-Benchmark (`hest`): https://github.com/mahmoodlab/HEST/tree/3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3
- PathoROB (`pathorob`): https://github.com/bifold-pathomics/PathoROB/tree/6583cf0b0d902c8cc032308262fa3a3befdc0687
- PLISM robustness benchmark (`plism`): https://github.com/owkin/plism-benchmark/tree/5ec9511893af993f6faa099f093d1924b291aed2
- Official UNI 2 benchmark report (`uni2_benchmark`): https://github.com/mahmoodlab/UNI/tree/42715efc11722a496e0a67f3369505a8f277206c
- Virchow2G primary-paper benchmarks (`virchow2g_paper`): https://arxiv.org/abs/2408.00738
- TITAN primary-paper benchmarks (`titan_paper`): https://www.nature.com/articles/s41591-025-03982-3
- H-optimus-1 official benchmark report (`hoptimus1_report`): https://www.bioptimus.com/news/bioptimus-launches-h-optimus-1
- Virchow2 primary-paper benchmarks (`virchow2_paper`): https://arxiv.org/abs/2408.00738
- Virchow primary-paper benchmarks (`virchow_paper`): https://arxiv.org/abs/2309.07778
- UNI primary-paper benchmarks (`uni_paper`): https://arxiv.org/abs/2308.15474
- CONCH primary-paper benchmarks (`conch_primary`): https://www.nature.com/articles/s41591-024-02856-4
- CONCHv1.5 TITAN-paper benchmarks (`titan_patch_encoder`): https://www.nature.com/articles/s41591-025-03982-3
- Phikon primary-paper benchmarks (`phikon_primary`): https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v3
- Phikon-v2 external-cohort benchmarks (`phikon_v2_external`): https://arxiv.org/abs/2409.09173v1
- Hibou primary-paper benchmarks (`hibou_primary`): https://arxiv.org/abs/2406.05074
- MUSK primary-paper benchmarks (`musk_primary`): https://www.nature.com/articles/s41586-024-08378-w
- GPFM primary-paper benchmarks (`gpfm_primary`): https://www.nature.com/articles/s41551-025-01488-4

`provenance.json` records pinned source revisions, report hashes, normalization
rules, and audit caveats. `scores_all.csv` includes prototype and external rows;
`scores_paper.csv` is restricted to accepted evidence and the supported paper
matrix. Machine-parsed primary-source evidence is not dual human verification.
