"""Hash-bound artifact freshness and experiment-set dry-run utilities."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any, Iterable


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
