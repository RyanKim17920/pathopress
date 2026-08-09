"""PathoPress: pathology benchmark score-matrix tooling.

The public API is intentionally small and is re-exported lazily, so importing
``pathopress`` stays cheap and free of import cycles; the heavier analysis
modules are only loaded when one of their names is actually touched.

Typical use::

    from pathopress import load_scores, make_matrix, complete

    scores = load_scores("data/scores.csv")
    matrix, models, evaluations = make_matrix(scores)
    filled = complete(matrix, rank=1)
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

#: name -> defining submodule, resolved on first attribute access.
_EXPORTS: dict[str, str] = {
    # Score matrix construction
    "load_scores": "pathopress.matrix",
    "make_matrix": "pathopress.matrix",
    "filter_matrix": "pathopress.matrix",
    # Low-rank completion and imputation
    "complete": "pathopress.completion",
    "complete_soft_impute": "pathopress.completion",
    "validate": "pathopress.completion",
    "build_imputation_rows": "pathopress.imputation",
    "write_imputations": "pathopress.imputation",
    # Metrics and ranking
    "absolute_percentage_errors": "pathopress.metrics",
    "pairwise_ranking_accuracy": "pathopress.ranking",
    "top_fraction_recovery": "pathopress.ranking",
    # Probe selection and compression
    "predict_all_known": "pathopress.probe_compression",
    "predict_heldout_models": "pathopress.probe_compression",
    "score_predictions": "pathopress.probe_compression",
    "load_probe_compression": "pathopress.probe_compression",
    "dump_probe_compression": "pathopress.probe_compression",
    # Artifacts, provenance and maintenance
    "sha256_file": "pathopress.artifacts",
    "build_freshness_manifest": "pathopress.maintenance",
    "check_freshness_manifest": "pathopress.maintenance",
    "validate_probe_compression_semantics": "pathopress.maintenance",
    # Workflow front end
    "WORKFLOWS": "pathopress.workflows",
    "run_workflow": "pathopress.workflows",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
