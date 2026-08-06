"""Canonical external-source pins shared by generators and verification tests."""

from __future__ import annotations

import re


BENCHPRESS_REPOSITORY = "https://github.com/microsoft/benchpress"
BENCHPRESS_PINNED_COMMIT = "0a684b63ee0e4a401cb907a3827a82ea997d74c4"


def validate_benchpress_pin(commit: str = BENCHPRESS_PINNED_COMMIT) -> None:
    """Validate a full Git identity against the configured canonical pin."""

    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("BenchPress pin must be a full 40-character lowercase Git SHA")
    if commit != BENCHPRESS_PINNED_COMMIT:
        raise ValueError("BenchPress pin differs from the configured canonical value")


def benchpress_tree_url(commit: str = BENCHPRESS_PINNED_COMMIT) -> str:
    validate_benchpress_pin(commit)
    return f"{BENCHPRESS_REPOSITORY}/tree/{commit}"
