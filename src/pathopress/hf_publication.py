"""Fail-closed local validation and explicitly authorized HF publication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .public_data import load_public_export, parquet_available


def validate_hf_export(root: str | Path) -> dict[str, Any]:
    """Validate hashes, schemas, counts, and optional Parquet mirrors."""

    release = load_public_export(root)
    manifest = release.manifest
    parquet_tables = [
        item for item in manifest["files"] if str(item["path"]).endswith(".parquet")
    ]
    if manifest.get("parquet_written"):
        if not parquet_available():
            raise RuntimeError(
                "Validating a Parquet publication requires pyarrow; install `pathopress[hf]`."
            )
        import pyarrow.parquet as pq

        for item in parquet_tables:
            table = pq.read_table(release.root / item["path"])
            if table.num_rows != item.get("rows"):
                raise ValueError(f"Parquet row count mismatch: {item['path']}")
            created_by = pq.read_metadata(release.root / item["path"]).created_by or ""
            expected_version = str(manifest["exporter"]["pyarrow_version"])
            if f"version {expected_version}" not in created_by:
                raise ValueError(
                    f"Parquet backend version mismatch: {item['path']}"
                )
    return {
        "status": "validated_local_export",
        "root": str(release.root),
        "dataset_id": manifest["dataset_id"],
        "paper_matrix": manifest["paper_filter"],
        "files": len(manifest["files"]),
        "parquet_files": len(parquet_tables),
    }


def publish_hf_export(
    root: str | Path,
    *,
    repo_id: str,
    upload: bool = False,
    authorized: bool = False,
    token: str | None = None,
    commit_message: str = "Update PathoPress score matrix export",
) -> dict[str, Any]:
    """Return a dry-run plan unless upload and authorization are both explicit."""

    validation = validate_hf_export(root)
    plan = {
        **validation,
        "repo_id": repo_id,
        "repo_type": "dataset",
        "commit_message": commit_message,
        "upload_requested": bool(upload),
    }
    if not upload:
        return {**plan, "status": "dry_run_no_network"}
    if not authorized:
        raise RuntimeError(
            "Upload refused: pass --authorize-upload only after the user explicitly authorizes publication."
        )
    if not token:
        raise RuntimeError("Upload refused: HF_TOKEN is missing or empty.")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Upload requires huggingface_hub; install `pathopress[hf]`."
        ) from exc
    HfApi(token=token).upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(Path(root)),
        path_in_repo=".",
        commit_message=commit_message,
    )
    return {**plan, "status": "uploaded"}
