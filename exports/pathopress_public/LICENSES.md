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

`provenance.json` records pinned source revisions, report hashes, normalization
rules, and audit caveats. `scores_all.csv` includes prototype and external rows;
`scores_paper.csv` is restricted to accepted evidence and the supported paper
matrix. Machine-parsed primary-source evidence is not dual human verification.
