"""Build hash-bound static-site starter probe sets from completed experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


STARTER_SET_SCHEMA = "pathopress-static-starter-sets-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_starter_sets(
    probe_compression_path: str | Path,
    feasibility_allowlist_path: str | Path,
    *,
    count: int = 10,
) -> dict[str, Any]:
    """Extract unrestricted and pre-error-feasibility greedy trajectories."""

    if count < 1:
        raise ValueError("count must be positive")
    probe_path = Path(probe_compression_path)
    allowlist_path = Path(feasibility_allowlist_path)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
    if probe["configuration"]["allowlist_sha256"] != _sha256(allowlist_path):
        raise ValueError("probe-compression artifact does not match feasibility allowlist")

    def trajectory(mode: str) -> list[str]:
        rows = probe["curves"][mode]["all_known_greedy_medae"]
        result = [str(row["added_evaluation_id"]) for row in rows[:count]]
        if len(result) != count or len(set(result)) != count:
            raise ValueError(f"{mode} does not provide {count} unique greedy probes")
        return result

    unrestricted = trajectory("any_candidate")
    feasibility = trajectory("pre_error_low_friction_allowlist")
    allowed = set(allowlist["evaluation_ids"])
    if not set(feasibility).issubset(allowed):
        raise ValueError("feasibility starter trajectory escapes the pre-error allowlist")
    return {
        "schema_version": STARTER_SET_SCHEMA,
        "matrix_scores_sha256": probe["configuration"]["scores_sha256"],
        "probe_compression_sha256": _sha256(probe_path),
        "feasibility_allowlist_sha256": _sha256(allowlist_path),
        "default_visible_count": min(5, count),
        "sets": {
            "unrestricted": {
                "label": "Most predictive unrestricted set",
                "semantics": "All-known greedy MedAE trajectory; selected using prediction errors.",
                "evaluation_ids": unrestricted,
            },
            "feasibility": {
                "label": "Pre-error feasibility-proxy set",
                "semantics": (
                    "All-known greedy MedAE trajectory restricted to the pre-error "
                    "image/patch classification pipeline proxy; not measured cost."
                ),
                "evaluation_ids": feasibility,
            },
        },
    }
