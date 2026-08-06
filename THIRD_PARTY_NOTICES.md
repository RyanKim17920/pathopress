# Third-party notices

PathoPress is licensed under the project license in [`LICENSE`](LICENSE).
Portions of its code and experimental design are adapted from the following
third-party project. This notice describes the relationship without implying
that Microsoft authored, endorses, or is responsible for PathoPress or its
pathology data and conclusions.

## Microsoft BenchPress

- Project: Microsoft BenchPress
- Source: <https://github.com/microsoft/benchpress>
- Audited revision:
  [`0a684b63ee0e4a401cb907a3827a82ea997d74c4`](https://github.com/microsoft/benchpress/tree/0a684b63ee0e4a401cb907a3827a82ea997d74c4)
- Copyright: Microsoft Corporation
- License: MIT

The following PathoPress components contain adapted implementations of
BenchPress algorithmic behavior, rewritten around local NumPy arrays,
pathology protocol identifiers, and PathoPress artifact contracts:

- `src/pathopress/completion.py`: percentage-logit normalization,
  column standardization, bias-decomposed alternating least squares, and
  iterative truncated-SVD/Soft-Impute completion, based principally on the
  upstream website predictor and completion methods;
- `src/pathopress/method_comparison.py`: the seven transforms, classical
  completer families, and method/hyperparameter grid based on
  `benchpress/methods/transforms.py`, `benchpress/methods/completers.py`, and
  the upstream method-comparison experiment; and
- `src/pathopress/confidence.py`: disagreement, structural-support,
  cross-fitted error-risk, risk–coverage, stratum, and leave-fold-out conformal
  primitives based on `benchpress/methods/confidence.py` and the upstream
  confidence-calibration experiment; and
- `website/app.js`: an independently written JavaScript implementation of the
  same adapted rank-1 point-completion recipe, including NumPy-compatible seeded
  initialization so browser results agree with PathoPress's Python predictor.

Other modules independently reimplement or pathology-adapt BenchPress
*experiment contracts* rather than reproducing upstream source text verbatim:

- `src/pathopress/artifacts.py` and the experiment runners use the persisted
  ten-seed, three-fold within-model holdout design;
- `src/pathopress/probes.py` and `src/pathopress/probe_compression.py` implement
  all-known, isolated held-out-model, random, greedy, pruned, and bounded
  exhaustive probe protocols;
- `src/pathopress/ranking.py` implements the pairwise-margin and top-fraction
  leaderboard metrics used by the ranking-preservation experiments;
- `src/pathopress/predictability.py` and `src/pathopress/temporal.py` implement
  the hide-half predictability and prior-release-only deployment protocols; and
- structure, error-factor, plotting, table, maintenance, export, CLI, and the
  remainder of the site code follow analogous scientific roles where
  documented, but use
  PathoPress-specific implementations and data contracts.

Pathology-specific work—including benchmark extraction and deduplication,
native metric mappings, rank selection, feasibility metadata, model metadata,
confidence population restrictions, public export, and browser product—is not
represented as Microsoft code. Direct parity tests compare selected numerical
behavior with the pinned revision; they do not change ownership or licensing.

PathoPress does not redistribute the BenchPress score-matrix dataset. The
PathoPress registry instead contains factual identifiers, protocol metadata,
citations, and reported pathology scores from separately licensed upstream
benchmark sources; see [`DATA_NOTICE.md`](DATA_NOTICE.md) and the public
export's [`LICENSES.md`](exports/pathopress_public/LICENSES.md).

### Microsoft BenchPress MIT license

MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
