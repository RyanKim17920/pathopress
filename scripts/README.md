# `scripts/` layout

Every file here is a standalone entry point run as `PYTHONPATH=src python3
scripts/<name>.py`. Those literal paths are recorded in
`experiments/experiment_set.json` and re-hashed in
`experiments/artifact_freshness_manifest.json`, and many are cited from
`README.md` and `docs/`, so **paths are part of the published record and are not
moved casually**. This file supplies the grouping that a directory split would
have provided, without invalidating a single documented command.

After `pip install -e .` the same workflows are also reachable as
`pathopress-run <workflow>` (see `pathopress-run --list`).

## Pipeline steps (registered in `experiments/experiment_set.json`)

| Script | Role |
| --- | --- |
| `build_shared_artifacts.py` | Freezes the analysis matrix and the BenchPress fold protocol. |
| `dry_run_experiment_set.py` | Validates the experiment set without executing it. |
| `replay_lofo_matched_cells.py` | Re-derives held-out per-cell predictions and scores all arms on one matched cell set. |
| `build_evaluation_cost_evidence.py` | Builds the evaluation-cost evidence table. |
| `build_public_release.py`, `build_website_starter_sets.py`, `publish_hf_dataset.py` | Assemble the public export, the static site payload, and the Hugging Face dataset. |

## Figures

`plot_benchpress_style.py`, `plot_benchpress_style_hero.py`,
`plot_probe_dual_objective.py`, `plot_temporal_deployment.py` — each writes into
`figures/` and is cited from `figures/README.md`.

## Registry and input construction

`build_registry.py` is the sole producer of `data/scores.csv` and
`data/provenance.json`; it imports every module in `scripts/evidence/` (by that
exact import path, so `scripts/evidence/` must not move).
`build_model_metadata.py`, `build_evaluation_feasibility.py`, and
`build_score_review_ledger.py` produce the remaining committed `data/` inputs.

## Verification

`check_artifact_freshness.py`, `verify_benchpress_parity.py`,
`verify_method_comparison_parity.py`, `validate_score_review_ledger.py`,
`audit_prov_gigapath_tile_evidence.py`, `run_burden_telemetry.py`.

## Source extraction and evidence validators (`extract_*.py`)

These are the provenance layer for `data/scores.csv` and split into two kinds:

* **Extractors** parse pinned upstream PDFs, spreadsheets, and pages into the
  committed `source_data/*.csv` snapshots: `extract_group_b_official_scores.py`,
  `extract_group_c_official_scores.py`, `extract_wave_e_official_scores.py`,
  `extract_wave_f_official_scores.py`, `extract_pathorob_scores.py`,
  `extract_exaone_pathobench.py`, `extract_threads_pathobench.py`,
  `extract_wave_d_uni_paper.py`. Because their outputs are committed,
  `data/scores.csv` is regenerable without re-downloading the sources.
* **Validators** re-check hand-authored snapshots against their recorded hashes
  and print the disposition ledger: `extract_h0mini_uni2h_scores.py`,
  `extract_wave_d_hoptimus_report.py`, `extract_wave_d_virchow_paper.py`,
  `extract_wave_d_virchow2_paper.py`. They write nothing. They are the only
  executable statement of that check outside `tests/`, and
  `docs/score-source-coverage.md` cites them by path, so they are kept in place
  rather than archived.

## `scripts/evidence/`

Importable modules (not entry points) consumed by `build_registry.py`; each has
a dedicated test under `tests/`.
