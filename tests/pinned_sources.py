"""Shared guard for tests whose inputs are pinned upstream sources.

A handful of tests in this suite do not read repository files: they re-run an
extractor against the *original* upstream artifact and compare the result with
the committed ``source_data/*.csv`` snapshot (or with ``data/scores.csv``
itself, via ``scripts/build_registry.py``).  That comparison is the only thing
in the suite that checks the project's central claim -- that the published
scores really derive from the cited sources.

Those inputs are deliberately not vendored, so the tests skip when they are
absent.  Skipping is right for a laptop and for the pull-request gate, but it
made the claim unverifiable in CI: the tests were *always* skipped there, so a
regressed extractor would have gone green.  ``PATHOPRESS_REQUIRE_PINNED_SOURCES``
inverts that for a run which has deliberately provisioned the inputs -- a
missing input then fails instead of skipping.

Two classes of input, because only one of them can be provisioned
automatically:

``fetchable``
    Git checkouts pinned by ``data/provenance.json`` and materialised by
    ``scripts/fetch_sources.py``.  ``PATHOPRESS_REQUIRE_PINNED_SOURCES=1``
    requires these.

``publisher``
    Publisher supplementary PDFs, a Nature source-data workbook and one
    vendor HTML report.  ``data/provenance.json`` records no retrieval URL for
    the Group B / Wave E / Wave F artifacts at all, and the ones it does record
    sit behind publisher endpoints that refuse automated clients, so no script
    in this repository can materialise them.  They are required only under
    ``PATHOPRESS_REQUIRE_PINNED_SOURCES=all``, which is meaningful on a
    workstation that has them and is not usable from CI.

Environment
-----------
``PATHOPRESS_SOURCES``
    Root of the pinned checkouts.  Defaults to ``/tmp/pathopress_sources``,
    which is also ``scripts/fetch_sources.py`` and ``scripts/build_registry.py
    --sources``' default.

``PATHOPRESS_REQUIRE_PINNED_SOURCES``
    Unset/``0``  -- absent inputs skip (the default; unchanged behaviour).
    ``1``        -- absent *fetchable* inputs fail.
    ``all``      -- absent inputs of either class fail.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

DEFAULT_SOURCES = Path("/tmp/pathopress_sources")
SOURCES_ENV = "PATHOPRESS_SOURCES"
REQUIRE_ENV = "PATHOPRESS_REQUIRE_PINNED_SOURCES"

_OFF = {"", "0", "false", "no", "off"}


def sources_root() -> Path:
    """Where the pinned upstream checkouts live for this run."""
    override = os.environ.get(SOURCES_ENV, "").strip()
    return Path(override) if override else DEFAULT_SOURCES


def _requirement() -> str:
    value = os.environ.get(REQUIRE_ENV, "").strip().lower()
    if value in _OFF:
        return "none"
    return "all" if value == "all" else "fetchable"


def requires_fetchable() -> bool:
    """True when git checkouts pinned by provenance must be present."""
    return _requirement() in {"fetchable", "all"}


def requires_publisher() -> bool:
    """True when the non-retrievable publisher artifacts must be present too."""
    return _requirement() == "all"


def missing_inputs(case: unittest.TestCase, reason: str, *, fetchable: bool = True) -> None:
    """Skip -- unless this run asserted the inputs are present, then fail.

    ``fetchable`` selects which class of input is missing; see the module
    docstring.  Never returns.
    """
    required = requires_fetchable() if fetchable else requires_publisher()
    if not required:
        case.skipTest(reason)
    expected = "1" if fetchable else "all"
    hint = (
        f"run `python3 scripts/fetch_sources.py {sources_root()}` first"
        if fetchable
        else "these artifacts cannot be fetched automatically; see tests/pinned_sources.py"
    )
    raise AssertionError(
        f"{reason}, but {REQUIRE_ENV}={os.environ.get(REQUIRE_ENV)!r} asserts they are "
        f"present (this run requires the "
        f"{'pinned git checkouts' if fetchable else 'publisher artifacts'}, "
        f"{REQUIRE_ENV}={expected}). Refusing to skip a provenance check that was "
        f"asked for: {hint}."
    )
