"""Hash-bound artifact freshness and experiment-set dry-run utilities."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any, Iterable

from pathopress.matrix import filter_matrix, load_scores, make_matrix
from pathopress.probe_compression import load_probe_compression


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _directory_record(path: Path) -> dict[str, Any]:
    """Return a stable content digest without materializing a cache manifest."""

    def stable_file(candidate: Path) -> bool:
        relative = candidate.relative_to(path)
        ignored_directories = {"__pycache__", ".pytest_cache"}
        ignored_suffixes = {".pyc", ".pyo", ".tmp", ".lock", ".swp"}
        return (
            candidate.is_file()
            and not candidate.is_symlink()
            and ignored_directories.isdisjoint(relative.parts)
            and candidate.suffix not in ignored_suffixes
            and not candidate.name.endswith("~")
            and candidate.name != ".DS_Store"
        )

    digest = hashlib.sha256()
    files = sorted((candidate for candidate in path.rglob("*") if stable_file(candidate)),
                   key=lambda candidate: candidate.relative_to(path).as_posix())
    total_bytes = 0
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        size = candidate.stat().st_size
        total_bytes += size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "kind": "directory",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "bytes": total_bytes,
    }


def path_record(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"kind": "file", "sha256": file_sha256(path), "bytes": path.stat().st_size}
    if path.is_dir():
        return _directory_record(path)
    raise ValueError(f"manifest path does not exist: {path}")


def _path_value(value: str | dict[str, Any]) -> str:
    if isinstance(value, str):
        return value
    path = value.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("path records require a non-empty 'path' string")
    return path


def _record_paths(root: Path, paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    output = {}
    for path in sorted(paths, key=lambda value: str(value)):
        output[_relative_path(root, path)] = path_record(path)
    return output


def build_freshness_manifest(
    root: Path,
    *,
    inputs: Iterable[Path],
    artifacts: Iterable[Path],
    kind: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "kind": kind,
        "inputs": _record_paths(root, inputs),
        "dependencies": {},
        "artifacts": _record_paths(root, artifacts),
    }
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(unsigned).hexdigest()
    return payload


def build_result_graph_manifest(
    root: Path,
    *,
    experiment_set_path: Path,
    experiment_set: dict[str, Any],
    kind: str = "pathopress_completed_result_graph",
) -> dict[str, Any]:
    """Hash every declared node in the completed compact-result graph."""

    groups: dict[str, set[Path]] = {
        "inputs": set(),
        "dependencies": set(),
        "artifacts": set(),
    }
    components: dict[str, dict[str, Any]] = {}
    for experiment in experiment_set.get("experiments", []):
        name = experiment.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("experiment names must be non-empty strings")
        component: dict[str, Any] = {"command": experiment.get("command", "")}
        for group in groups:
            declared = experiment.get(group, [])
            normalized = []
            for value in declared:
                relative = _path_value(value)
                normalized.append(value if isinstance(value, dict) else relative)
                groups[group].add(root / relative)
            component[group] = normalized
        components[name] = component

    payload = {
        "schema_version": 2,
        "kind": kind,
        "experiment_set": {
            "path": _relative_path(root, experiment_set_path),
            "sha256": file_sha256(experiment_set_path),
        },
        "components": components,
        **{group: _record_paths(root, paths) for group, paths in groups.items()},
    }
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(unsigned).hexdigest()
    return payload


def _expected_digest(value: str | dict[str, Any]) -> str:
    if isinstance(value, str):
        return value
    digest = value.get("sha256")
    if not isinstance(digest, str):
        raise ValueError("manifest records require sha256")
    return digest


def check_freshness_manifest(root: Path, manifest: dict[str, Any]) -> list[dict[str, str]]:
    supplied = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    expected = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    failures = []
    if supplied != expected:
        failures.append({"path": "<manifest>", "status": "manifest_hash_mismatch"})
    experiment_set = manifest.get("experiment_set")
    if experiment_set:
        relative = experiment_set.get("path", "")
        path = root / relative
        if not path.is_file():
            failures.append({"path": relative, "status": "missing"})
        elif file_sha256(path) != experiment_set.get("sha256"):
            failures.append({"path": relative, "status": "stale_or_modified"})
    for group in ("inputs", "dependencies", "artifacts"):
        for relative, expected_record in manifest.get(group, {}).items():
            path = root / relative
            if not path.exists():
                failures.append({"path": relative, "status": "missing"})
            elif path_record(path)["sha256"] != _expected_digest(expected_record):
                failures.append({"path": relative, "status": "stale_or_modified"})
    for component in manifest.get("components", {}).values():
        for group in ("inputs", "dependencies", "artifacts"):
            registry = manifest.get(group, {})
            for value in component.get(group, []):
                relative = _path_value(value)
                if relative not in registry:
                    failures.append({"path": relative, "status": f"undeclared_{group}_node"})
    return failures


def validate_probe_compression_semantics(root: Path) -> list[dict[str, str]]:
    """Reject hash-fresh but semantically obsolete public probe artifacts."""
    artifact_path = root / "experiments/probe_compression_rank1.json"
    scores_path = root / "data/scores.csv"
    allowlist_path = root / "data/low_friction_allowlist_v2_top25.json"
    for path in (artifact_path, scores_path, allowlist_path):
        if not path.is_file():
            return [{"path": str(path.relative_to(root)), "status": "missing"}]
    try:
        payload = load_probe_compression(artifact_path)
        allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
        config = payload["configuration"]
        pruning = payload["pruning"]
        curves = payload["curves"]
        ranking = payload["ranking_aware"]
        current_matrix, _, current_evaluations = filter_matrix(
            *make_matrix(load_scores(scores_path))
        )
        expected_candidates = {
            "any_candidate": current_evaluations,
            "pre_error_low_friction_allowlist": allowlist["evaluation_ids"],
        }
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        return [{"path": str(artifact_path.relative_to(root)), "status": f"invalid_schema:{exc}"}]
    failures: list[dict[str, str]] = []

    def require(condition: bool, status: str) -> None:
        if not condition:
            failures.append({"path": str(artifact_path.relative_to(root)), "status": status})

    require(config.get("scores_sha256") == file_sha256(scores_path), "score_hash_semantic_mismatch")
    require(config.get("allowlist_sha256") == file_sha256(allowlist_path), "allowlist_hash_semantic_mismatch")
    require(config.get("matrix_shape") == list(current_matrix.shape), "matrix_shape_semantic_mismatch")
    require(config.get("prediction_rank") == 1, "expected_pathology_rank1")
    require(float(config.get("ranking_margin", -1)) == 5.0, "expected_margin5")
    require(
        float(config.get("score_curve_pairwise_diagnostic_margin", -1)) == 2.0,
        "expected_ancillary_score_curve_pairwise_diagnostic_margin2",
    )
    require(
        "ancillary diagnostics only" in config.get("score_curve_pairwise_diagnostic_semantics", ""),
        "score_curve_pairwise_diagnostic_semantics_missing",
    )
    require(config.get("medape_epsilon") == 1e-6, "expected_medape_epsilon_1e-6")
    require(pruning.get("keep_count") == 30, "expected_pruning_keep_count30")
    require(pruning.get("source_steps_used") == 10, "expected_pruning_source_steps_used10")
    require(len(pruning.get("evaluation_ids", [])) == 30, "expected_30_pruned_evaluation_ids")

    exact_k = list(range(1, 11))

    def has_exact_k(rows: list[dict[str, Any]], expected_k: list[int] | None = None) -> bool:
        target = expected_k if expected_k is not None else exact_k
        return [row.get("k") for row in rows] == target

    def has_exact_random_grid(rows: list[dict[str, Any]], max_k: int = 10) -> bool:
        expected_k = list(range(1, max_k + 1))
        return len(rows) == 10 * max_k and all(
            [row.get("k") for row in rows if row.get("repeat") == repeat] == expected_k
            for repeat in range(10)
        )

    is_lofo = payload.get("split", {}).get("split_mode") == "leave_one_family_out"
    heldout_max_k = (
        config.get("lofo_max_probes", 10) if is_lofo else 10
    )
    heldout_exact_k = list(range(1, heldout_max_k + 1))

    for mode, expected_ids in expected_candidates.items():
        curve = curves.get(mode, {})
        rank = ranking.get(mode, {})
        require(
            curve.get("candidate_ids") == expected_ids,
            f"{mode}_candidate_ids_mismatch_current_inputs",
        )
        expected = len(expected_ids)
        for key in ("all_known_greedy_medae", "all_known_greedy_medape"):
            require(has_exact_k(curve.get(key, [])), f"{mode}_{key}_requires_exact_k1_10")

        # Under LOFO, held-out curves live per-fold, not at top level.
        if is_lofo:
            lofo_folds = curve.get("lofo", {})
            _fold_keys = sorted(lofo_folds.keys(), key=lambda x: int(x))
            # Top-level held-out keys must NOT exist under LOFO
            for key in ("heldout_greedy_medae", "heldout_greedy_medape",
                        "heldout_random"):
                require(
                    key not in curve,
                    f"{mode}_{key}_must_not_exist_at_top_level_in_lofo",
                )
            # Validate each fold's held-out curves
            for _fk in _fold_keys:
                _fold_curve = lofo_folds[_fk].get(mode, {})
                for key in ("heldout_greedy_medae", "heldout_greedy_medape"):
                    require(
                        has_exact_k(_fold_curve.get(key, []), heldout_exact_k),
                        f"{mode}_{key}_lofo_fold{_fk}_requires_exact_k1_{heldout_max_k}",
                    )
                require(
                    has_exact_random_grid(
                        _fold_curve.get("heldout_random", []), heldout_max_k,
                    ),
                    f"{mode}_heldout_random_lofo_fold{_fk}_requires_exact_10x_k1_{heldout_max_k}",
                )
        else:
            for key in ("heldout_greedy_medae", "heldout_greedy_medape"):
                require(
                    has_exact_k(curve.get(key, []), heldout_exact_k),
                    f"{mode}_{key}_requires_exact_k1_{heldout_max_k}",
                )
            require(
                has_exact_random_grid(curve.get("heldout_random", []), heldout_max_k),
                f"{mode}_heldout_random_requires_exact_10x_k1_{heldout_max_k}",
            )

        all_known_max = min(
            expected, int(config.get("all_known_random_curve_limit", 10))
        )
        # In LOFO mode, all_known_random is only computed for any_candidate
        # (fold-invariant; identical across all 34 folds).  Allowlist modes
        # do not have it, so skip the check for non-any_candidate LOFO modes.
        if is_lofo and mode != "any_candidate":
            # Validate the per-fold copy of all_known_random for allowlist
            # (absent by design—no failure expected, but catches future regressions).
            lofo_folds = curve.get("lofo", {})
            if lofo_folds:
                _fold_keys = sorted(lofo_folds.keys(), key=lambda x: int(x))
                for _fk in _fold_keys:
                    _fold_curve = lofo_folds[_fk].get(mode, {})
                    require(
                        "all_known_random" not in _fold_curve,
                        f"{mode}_all_known_random_should_not_exist_in_lofo_fold{_fk}",
                    )
        elif not is_lofo or mode == "any_candidate":
            require(
                has_exact_random_grid(curve.get("all_known_random", []), all_known_max),
                f"{mode}_all_known_random_requires_exact_10x_k1_{all_known_max}",
            )
        # Provenance check: reject top-level held-out curves that are
        # byte-identical to any single fold's curve.  This is exactly the
        # failure mode that lets a fold-0 backfill slip past has_exact_k.
        if is_lofo:
            lofo_folds = curve.get("lofo", {})
            _fold_keys = sorted(lofo_folds.keys(), key=lambda x: int(x))
            for _hk in ("heldout_greedy_medae", "heldout_greedy_medape",
                        "heldout_random"):
                _top = curve.get(_hk)
                if _top is not None:
                    for _fk in _fold_keys:
                        _fc = lofo_folds[_fk].get(mode, {})
                        _fold_val = _fc.get(_hk)
                        if json.dumps(_top, sort_keys=True) == json.dumps(
                                _fold_val, sort_keys=True
                        ):
                            require(
                                False,
                                f"{mode}_{_hk}_is_byte_identical_to_lofo_fold{_fk}",
                            )
        require(rank.get("margin") == 5.0, f"{mode}_ranking_margin5")
        require(has_exact_k(rank.get("all_known_greedy", [])), f"{mode}_ranking_all_known_exact_k1_10")
        if is_lofo:
            # LOFO: ranking heldout_greedy goes to full k=10 (all_known_greedy depth),
            # not limited by lofo_max_probes (which only limits score curves).
            if "heldout_greedy" in rank:
                require(
                    has_exact_k(rank["heldout_greedy"]),
                    f"{mode}_ranking_holdout_exact_k1_10",
                )
            elif "lofo_folds" in rank:
                fold_ok = all(
                    has_exact_k(f.get("heldout_greedy", []))
                    for f in rank["lofo_folds"]
                )
                require(fold_ok, f"{mode}_ranking_holdout_exact_k1_10")
            else:
                require(False, f"{mode}_ranking_holdout_missing")
        else:
            require(has_exact_k(rank.get("heldout_greedy", [])), f"{mode}_ranking_holdout_exact_k1_10")
        require(
            has_exact_random_grid(
                rank.get("all_known_random", []),
                min(expected, int(config.get("ranking_random_curve_limit", 10))),
            ),
            f"{mode}_ranking_random_exact_configured_grid",
        )
    require(
        set(ranking) == set(expected_candidates),
        "ranking_schema_must_contain_only_current_candidate_universes",
    )
    return failures


def validate_experiment_set(root: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate commands and inputs without executing experiments."""

    results = []
    names = set()
    for experiment in payload.get("experiments", []):
        name = experiment.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("experiment names must be unique non-empty strings")
        names.add(name)
        command = experiment.get("command", "")
        tokens = shlex.split(command)
        if len(tokens) < 2:
            raise ValueError(f"invalid experiment command: {name}")
        script_candidates = [token for token in tokens if token.endswith(".py")]
        scripts_present = all((root / token).is_file() for token in script_candidates)
        inputs = [root / _path_value(value) for value in experiment.get("inputs", [])]
        missing = [str(path.relative_to(root)) for path in inputs if not path.exists()]
        dependencies = [root / _path_value(value) for value in experiment.get("dependencies", [])]
        missing_dependencies = [
            str(path.relative_to(root)) for path in dependencies if not path.exists()
        ]
        artifacts = [root / _path_value(value) for value in experiment.get("artifacts", [])]
        missing_artifacts = [
            str(path.relative_to(root)) for path in artifacts if not path.exists()
        ]
        external_calls = bool(experiment.get("external_calls", False))
        results.append(
            {
                "name": name,
                "status": "ready" if scripts_present and not missing and not missing_dependencies and not missing_artifacts else "blocked",
                "scripts_present": scripts_present,
                "missing_inputs": missing,
                "missing_dependencies": missing_dependencies,
                "missing_artifacts": missing_artifacts,
                "declared_inputs": len(inputs),
                "declared_dependencies": len(dependencies),
                "declared_artifacts": len(artifacts),
                "external_calls": external_calls,
                "dry_run_only": True,
            }
        )
    return results
