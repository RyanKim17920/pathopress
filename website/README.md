# PathoPress static predictor

This directory is a static, no-backend interface for two PathoPress workflows:

- look up an existing model × evaluation cell, preserving reported/predicted status, source links, and applicable calibrated intervals;
- append a new model from one or more known normalized scores and run the selected rank-1 bias-ALS recipe entirely in the browser.

Generated `data.json` contains the fixed 59-model × 165-evaluation paper matrix,
1,967 reported cells, rank-1 estimates, source links, and applicable existing-row
intervals. The browser implementation uses the same logit, per-evaluation
standardization, ridge `0.1`, 40 ALS iterations, and seeded ten-start recipe as
the Python predictor. Automated parity tests compare its new-row results with
Python output.

New-model confidence is deliberately marked unavailable. The deploy-time interval
artifact was calibrated on held-out cells from existing supported model rows, not
on genuinely unseen models. Intervals are normalized-score uncertainty
diagnostics, not clinical guarantees.

Regenerate the data before serving:

```bash
PYTHONPATH=src python3 scripts/build_public_release.py
```

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
