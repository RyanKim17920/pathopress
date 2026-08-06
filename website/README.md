# PathoPress static predictor

This directory is a static, no-backend interface for three PathoPress workflows:

- look up an existing model × evaluation cell, preserving reported/predicted status, source links, applicable calibrated intervals, and calibrated P(|error| <= 10 normalized points);
- append a new model from one or more known normalized scores and run the selected rank-1 bias-ALS recipe entirely in the browser.
- browse the complete reported/predicted matrix with model, evaluation, and cell-kind filters; every cell links back to its lookup details.

Generated `data.json` contains the fixed 59-model × 168-evaluation paper matrix,
2,027 reported cells, rank-1 estimates, source links, existing-row intervals and
trust probabilities (with explicit abstention statuses),
and the compact unseen-model confidence lookup. The browser implementation uses the same logit, per-evaluation
standardization, ridge `0.1`, 40 ALS iterations, and seeded ten-start recipe as
the Python predictor. Automated parity tests compare its new-row results with
Python output.

`starter_sets.json` is a separate hash-bound deploy artifact built from the
completed unrestricted and pre-error-feasibility all-known greedy MedAE
trajectories. Build it only after the canonical score matrix and probe artifact
are current:

```bash
PYTHONPATH=src python3 scripts/build_website_starter_sets.py
```

The browser refuses starter sets whose score-matrix or probe-compression hash
differs from `data.json`, and requires the declared number of unique supported
evaluations in each trajectory. The feasibility button remains explicitly
labeled as a pipeline proxy, not measured monetary or runtime cost. Both starter buttons only prefill
evaluation identities; users supply their own normalized scores, after which
the existing local completion and unseen-model interval logic is used.

Existing-row trust uses the pinned BenchPress 3+12 generator experiment and a
leave-fold-out decreasing isotonic calibration. Its ten-point tolerance is one
decile of the normalized 0-100 scale, not a clinical threshold. The deploy
lookup averages model- and evaluation-level median hybrid risk and abstains
unless both are supported. New model rows use the separate interval artifact
below and explicitly abstain from this existing-row trust probability.

New-model intervals use a separate group-balanced artifact built only from
leave-one-model-out sparse-probe and temporal-release residuals at k=1/3/5/10.
The UI shows its risk, fallback scope, model-group/prediction counts, and an
explicit abstention for unsupported columns. Its 94.98% held-out coverage at a
nominal 90% level is a retrospective empirical result, not a prospective,
distribution-free, or clinical guarantee.

Regenerate the data before serving:

```bash
PYTHONPATH=src python3 scripts/build_public_release.py
```

To rebuild only the current pathology registry, matrix predictions, and public
tables while probe/confidence experiments are still pending, use:

```bash
PYTHONPATH=src python3 scripts/build_public_release.py --core-only
PYTHONPATH=src python3 scripts/build_website_starter_sets.py --pending
```

This mode records the current versioned feasibility-allowlist hashes and makes
the optional starter/confidence UI explicitly unavailable; it never embeds a
stale probe or calibration artifact.

Serve from the repository root so both the site and relative data links resolve:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/website/>. Opening `index.html` directly with a
`file://` URL will usually fail because browsers block `fetch("data.json")` from
local files.

The site has no forms that submit to a server, no analytics, no external runtime,
and no automatic upload or deployment action. The build command only refreshes
local export and website artifacts. Reported, provided, and predicted values
remain visually and structurally distinct.

[`NOTICE.txt`](NOTICE.txt) travels with the static assets and preserves the
Microsoft BenchPress copyright and MIT terms applicable to the adapted browser
completion recipe. The repository-level component map is in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

The source scores are machine-parsed primary evidence, not universally
dual-human-verified. Native endpoint mappings mix AUC, balanced accuracy,
kappa, correlation, F1, Dice, survival, and robustness semantics; a normalized
point is not a common clinical unit.
