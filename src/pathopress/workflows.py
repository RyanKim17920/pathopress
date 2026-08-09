"""Console entry points for the user-facing PathoPress workflows.

The reproduction pipeline is a set of standalone scripts under ``scripts/`` and
``experiments/`` (their paths are recorded verbatim in
``experiments/experiment_set.json`` and in the freshness manifest, so they must
keep working exactly as ``python3 scripts/<name>.py``).  This module adds a
packaged front end on top of them: after ``pip install -e .`` the same commands
are reachable as ``pathopress-run <workflow> [args...]`` — or through the
dedicated shortcuts declared in ``[project.scripts]`` — without the caller
needing to know where the checkout lives.

Nothing here re-implements a workflow.  Each entry point locates the script in
the source checkout and executes it in ``__main__`` context, so behaviour,
arguments, and artifacts are identical to invoking the file directly.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

__all__ = [
    "WORKFLOWS",
    "repository_root",
    "workflow_path",
    "run_workflow",
    "main",
]


#: Workflow name -> repository-relative script path.  Only workflows a reader
#: of the paper would plausibly re-run are listed; the long-running selection
#: and compression sweeps are included for completeness but are documented as
#: multi-hour jobs in ``experiments/README.md``.
WORKFLOWS: dict[str, str] = {
    # Inputs and shared artifacts
    "build-shared-artifacts": "scripts/build_shared_artifacts.py",
    "build-registry": "scripts/build_registry.py",
    # Analyses
    "benchpress-style": "experiments/run_benchpress_style.py",
    "method-comparison": "experiments/run_method_comparison.py",
    "probe-selection": "experiments/run_probe_selection.py",
    "probe-compression": "experiments/run_probe_compression.py",
    "replay-lofo-matched-cells": "scripts/replay_lofo_matched_cells.py",
    "ranking-preservation": "experiments/run_ranking_preservation.py",
    "confidence-calibration": "experiments/run_confidence_calibration.py",
    "new-model-confidence": "experiments/run_new_model_confidence.py",
    "temporal-deployment": "experiments/run_temporal_deployment.py",
    "budgeted-probe-selection": "experiments/run_budgeted_probe_selection.py",
    # Figures
    "plot-benchpress-style": "scripts/plot_benchpress_style.py",
    "plot-hero": "scripts/plot_benchpress_style_hero.py",
    "plot-probe-dual-objective": "scripts/plot_probe_dual_objective.py",
    "plot-temporal-deployment": "scripts/plot_temporal_deployment.py",
    # Release and maintenance
    "build-public-release": "scripts/build_public_release.py",
    "build-website-starter-sets": "scripts/build_website_starter_sets.py",
    "download-public-release": "scripts/download_public_release.py",
    "check-freshness": "scripts/check_artifact_freshness.py",
    "dry-run-experiment-set": "scripts/dry_run_experiment_set.py",
}


class WorkflowNotAvailable(RuntimeError):
    """Raised when a workflow script cannot be located in the checkout."""


def repository_root() -> Path:
    """Return the source checkout that owns this package.

    Works for the supported ``pip install -e .`` src-layout install, where the
    package lives at ``<root>/src/pathopress``.
    """

    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "experiments" / "experiment_set.json").is_file():
        return candidate
    raise WorkflowNotAvailable(
        "PathoPress workflow scripts are only available from a source checkout "
        "(install with `pip install -e .` from the cloned repository); "
        f"looked for the experiment set under {candidate}"
    )


def workflow_path(name: str) -> Path:
    try:
        relative = WORKFLOWS[name]
    except KeyError:
        raise SystemExit(
            f"unknown workflow {name!r}; choose one of: "
            + ", ".join(sorted(WORKFLOWS))
        ) from None
    path = repository_root() / relative
    if not path.is_file():
        raise WorkflowNotAvailable(f"workflow script is missing: {path}")
    return path


def run_workflow(name: str, argv: list[str] | None = None) -> None:
    """Execute a workflow script in ``__main__`` context."""

    path = workflow_path(name)
    sys.argv = [str(path), *(argv or [])]
    runpy.run_path(str(path), run_name="__main__")


def _make_entry_point(name: str):
    def entry_point() -> None:
        run_workflow(name, sys.argv[1:])

    entry_point.__name__ = f"run_{name.replace('-', '_')}"
    entry_point.__doc__ = f"Run the {name} workflow ({WORKFLOWS[name]})."
    return entry_point


# Dedicated console scripts declared in pyproject's [project.scripts].
replay_lofo_matched_cells = _make_entry_point("replay-lofo-matched-cells")
check_freshness = _make_entry_point("check-freshness")
build_public_release = _make_entry_point("build-public-release")
download_public_release = _make_entry_point("download-public-release")


def main(argv: list[str] | None = None) -> None:
    """``pathopress-run`` dispatcher."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "--list"}:
        print("usage: pathopress-run <workflow> [script arguments...]\n")
        print("workflows:")
        width = max(len(name) for name in WORKFLOWS)
        for name in sorted(WORKFLOWS):
            print(f"  {name.ljust(width)}  {WORKFLOWS[name]}")
        return
    run_workflow(args[0], args[1:])


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    main()
