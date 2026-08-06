#!/usr/bin/env python3
"""Build PathoPress's citation-backed benchmark registry from pinned clones.

This script deliberately uses only the Python standard library.  The source
repositories are expected to exist below ``--sources``; their exact commits
are recorded in ``provenance.json`` so every generated row can be audited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

try:
    from evidence.eva_scores import (
        merge_scores as merge_eva_scores,
        parse_midnight_scores,
        parse_repository_scores as parse_eva_repository_scores,
        required_additional_protocols,
    )
    from evidence.h0mini_uni2h_scores import (
        build_protocols as build_h0mini_uni2h_protocols,
        build_scores as build_h0mini_uni2h_scores,
        load_evidence as load_h0mini_uni2h_evidence,
    )
    from evidence.group_c_official_scores import (
        build_protocols as build_group_c_protocols,
        build_scores as build_group_c_scores,
        load_group_c_official_scores,
    )
    from evidence.wave_d_hoptimus_report import (
        build_protocols as build_wave_d_hoptimus_protocols,
        build_scores as build_wave_d_hoptimus_scores,
    )
    from evidence.wave_d_virchow2_paper import (
        build_protocols as build_wave_d_virchow2_protocols,
        build_scores as build_wave_d_virchow2_scores,
    )
    from evidence.wave_d_virchow_paper import (
        build_protocols as build_wave_d_virchow_protocols,
        build_scores as build_wave_d_virchow_scores,
    )
    from evidence.wave_d_uni_paper import (
        build_protocols as build_wave_d_uni_protocols,
        build_scores as build_wave_d_uni_scores,
    )
    from evidence.group_b_official_scores import (
        build_protocols as build_group_b_protocols,
        build_scores as build_group_b_scores,
    )
    from evidence.wave_e_official_scores import (
        build_protocols as build_wave_e_protocols,
        build_scores as build_wave_e_scores,
    )
    from evidence.wave_f_official_scores import (
        build_protocols as build_wave_f_protocols,
        build_scores as build_wave_f_scores,
    )
except ModuleNotFoundError:  # imported as scripts.build_registry in tests/tools
    from scripts.evidence.eva_scores import (
        merge_scores as merge_eva_scores,
        parse_midnight_scores,
        parse_repository_scores as parse_eva_repository_scores,
        required_additional_protocols,
    )
    from scripts.evidence.h0mini_uni2h_scores import (
        build_protocols as build_h0mini_uni2h_protocols,
        build_scores as build_h0mini_uni2h_scores,
        load_evidence as load_h0mini_uni2h_evidence,
    )
    from scripts.evidence.group_c_official_scores import (
        build_protocols as build_group_c_protocols,
        build_scores as build_group_c_scores,
        load_group_c_official_scores,
    )
    from scripts.evidence.wave_d_hoptimus_report import (
        build_protocols as build_wave_d_hoptimus_protocols,
        build_scores as build_wave_d_hoptimus_scores,
    )
    from scripts.evidence.wave_d_virchow2_paper import (
        build_protocols as build_wave_d_virchow2_protocols,
        build_scores as build_wave_d_virchow2_scores,
    )
    from scripts.evidence.wave_d_virchow_paper import (
        build_protocols as build_wave_d_virchow_protocols,
        build_scores as build_wave_d_virchow_scores,
    )
    from scripts.evidence.wave_d_uni_paper import (
        build_protocols as build_wave_d_uni_protocols,
        build_scores as build_wave_d_uni_scores,
    )
    from scripts.evidence.group_b_official_scores import (
        build_protocols as build_group_b_protocols,
        build_scores as build_group_b_scores,
    )
    from scripts.evidence.wave_e_official_scores import (
        build_protocols as build_wave_e_protocols,
        build_scores as build_wave_e_scores,
    )
    from scripts.evidence.wave_f_official_scores import (
        build_protocols as build_wave_f_protocols,
        build_scores as build_wave_f_scores,
    )


REPOSITORIES = {
    "benchpress": "https://github.com/microsoft/benchpress",
    "pathobench": "https://github.com/mahmoodlab/Patho-Bench",
    "pathobench_hf": "https://huggingface.co/datasets/MahmoodLab/Patho-Bench",
    "eva": "https://github.com/kaiko-ai/eva",
    "eva_midnight": "https://huggingface.co/kaiko-ai/midnight",
    "thunder": "https://github.com/MICS-Lab/thunder",
    "hest": "https://github.com/mahmoodlab/HEST",
    "pathorob": "https://github.com/bifold-pathomics/PathoROB",
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def portable_artifact_path(path: Path) -> str:
    """Record a repository-relative path without leaking a builder location."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.name

SCORE_FIELDS = [
    "model_id",
    "reported_model_alias",
    "model_revision",
    "evaluation_id",
    "value",
    "normalized_score",
    "suite_id",
    "metric",
    "reference_url",
    "source_locator",
    "extraction_date",
    "review_status",
    "uncertainty",
    "lineage",
    "audit_status",
]

SUITE_FIELDS = [
    "suite_id",
    "name",
    "scope",
    "task_count",
    "reference_url",
    "protocol",
    "audit_notes",
]

TASK_FIELDS = [
    "evaluation_id",
    "protocol_id",
    "task_identity_id",
    "dataset_artifact_id",
    "suite_id",
    "dataset_id",
    "task_name",
    "task_family",
    "target",
    "sample_unit",
    "task_type",
    "num_samples",
    "endpoint",
    "metric",
    "direction",
    "protocol",
    "reference_url",
    "audit_status",
    "audit_notes",
]

ALIAS_FIELDS = [
    "alias",
    "model_id",
    "suite_id",
    "reference_url",
    "audit_notes",
]

DEDUP_FIELDS = [
    "group_id",
    "match_type",
    "task_identity_id",
    "canonical_evaluation_id",
    "member_evaluation_id",
    "decision",
    "rationale",
]

EVA_CONFLICT_FIELDS = [
    "model_id",
    "evaluation_id",
    "selected_value",
    "selected_reference_url",
    "alternate_value",
    "alternate_reference_url",
    "absolute_difference",
    "decision",
]


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def blob_url(repo_key: str, commit: str, relative_path: str) -> str:
    base = REPOSITORIES[repo_key]
    marker = "blob"
    return f"{base}/{marker}/{commit}/{relative_path}"


def slug(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", "", value)).strip().lower()
    value = value.replace("&minus;", "-")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


MODEL_IDS = {
    "h-optimus-1": "h-optimus-1",
    "h-opt-1": "h-optimus-1",
    "hoptimus1": "h-optimus-1",
    "h-optimus-0": "h-optimus-0",
    "h-opt-0": "h-optimus-0",
    "hoptimus0": "h-optimus-0",
    "h0-mini": "h0-mini",
    "genbio-pathfm": "genbio-pathfm",
    "genbio-pfm": "genbio-pathfm",
    "uni2-h": "uni2-h",
    "uni2h": "uni2-h",
    "virchow2": "virchow-2",
    "virchow-2": "virchow-2",
    "midnight-12k": "midnight",
    "o-midnight": "openmidnight",
    "open-midnight": "openmidnight",
    "gigapath": "prov-gigapath",
    "provgigapath": "prov-gigapath",
    "prov-gigapath": "prov-gigapath",
    "conchv1-5": "conch-1.5",
    "conch-1-5": "conch-1.5",
    "conch-v1-5": "conch-1.5",
    "conch-v1": "conch",
    "conch-1": "conch",
    "phikon2": "phikon-v2",
    "phikon-2": "phikon-v2",
    "phikon-v2": "phikon-v2",
    "hibou-b": "hibou-b",
    "hibou-l": "hibou-l",
    "kaiko-s": "kaiko-vit-s-unspecified-patch",
    "kaiko-b": "kaiko-vit-b-unspecified-patch",
    "kaiko-vit-b-8": "kaiko-vit-b-8",
    "kaiko-vit-b-16": "kaiko-vit-b-16",
    "kaiko-vit-l-14": "kaiko-vit-l-14",
    "kaiko-vit-s-8": "kaiko-vit-s-8",
    "kaiko-vit-s-16": "kaiko-vit-s-16",
    "kaiko-s-8": "kaiko-vit-s-8",
    "kaiko-s-16": "kaiko-vit-s-16",
    "kaiko-b-8": "kaiko-vit-b-8",
    "kaiko-b-16": "kaiko-vit-b-16",
    "prism": "prism",
    "chief": "chief",
    "titan": "titan",
    "exaone-path-2-5": "exaone-path-2.5",
}

EXAONE_MODEL_IDS = {
    "CHIEF": "chief-slide",
    "GigaPath": "prov-gigapath-slide",
    "PRISM": "prism-slide",
    "TITAN": "titan-slide",
    "H-optimus-0": "h-optimus-0",
    "UNI2-h": "uni2-h",
    "EXAONE Path 2.5": "exaone-path-2.5-slide",
}

THREADS_MODEL_IDS = {
    "Virchow Mean Pooling": "virchow",
    "GigaPath Mean Pooling": "prov-gigapath",
    "Chief Mean Pooling": "chief-patch-mean",
    "CONCHv1.5 Mean Pooling": "conch-1.5",
    "PRISM": "prism-slide",
    "GigaPath": "prov-gigapath-slide",
    "CHIEF": "chief-slide",
    "Threads": "threads-slide",
}


def canonical_model(alias: str) -> str:
    key = slug(alias).replace("conch-1-5", "conch-v1-5")
    return MODEL_IDS.get(key, key)


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_simple_splits(path: Path) -> list[tuple[str, str]]:
    """Parse Patho-Bench's deliberately simple mapping-of-lists YAML."""
    pairs: list[tuple[str, str]] = []
    dataset = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw and not raw.startswith((" ", "-")) and raw.endswith(":"):
            dataset = raw[:-1]
        elif raw.startswith("- ") and dataset:
            pairs.append((dataset, raw[2:].strip()))
    return pairs


def pathobench_family(dataset: str, task: str) -> str:
    text = f"{dataset} {task}".lower()
    task_key = task.lower()
    if any(x in text for x in ("mutation", "mutant", "bap1", "pbrm1", "setd2", "egfr", "kras", "tp53", "pik3ca", "smad4", "vhl", "pten", "ctnnb1", "acvr2a", "apc", "arid1a", "keap1", "setd1b")):
        return "mutation_prediction"
    if any(x in text for x in ("response", "recist", "residual_cancer", "progression_regression")):
        return "treatment_response"
    if task_key in {"os", "pfs", "os_valentino", "pfs_valentino", "os_treatment_rdc"} or any(
        x in task_key for x in ("relapse", "died_within")
    ):
        return "survival_prediction"
    if any(x in text for x in ("grade", "grading", "isup")):
        return "tumor_grading"
    if task_key in {"er", "pr", "her2", "er_status", "her2_status"} or any(
        x in task_key for x in ("molecular_subtype", "msi", "mmr")
    ):
        return "molecular_subtyping"
    if any(x in text for x in ("immune", "invasion", "tils", "metastasis", "necrosis")):
        return "tme_characterization"
    return "morphological_subtyping"


def build_pathobench(source: Path, commit: str) -> list[dict[str, object]]:
    rel = "available_splits.yaml"
    rows = []
    for dataset, task in parse_simple_splits(source / rel):
        config_rel = f"{dataset}/{task}/config.yaml"
        config_path = source / config_rel
        if not config_path.is_file():
            raise ValueError(f"missing Patho-Bench task config: {config_rel}")
        config = config_path.read_text(encoding="utf-8")
        sample_col = yaml_scalar(config, "sample_col")
        task_type = yaml_scalar(config, "task_type")
        num_samples = int(yaml_scalar(config, "num_samples"))
        metric_match = re.search(r"(?m)^metrics:\s*\n\s*-\s*([^\n#]+)", config)
        if not metric_match:
            raise ValueError(f"missing Patho-Bench metric: {config_rel}")
        metric = metric_match.group(1).strip()
        sample_unit = "case" if sample_col == "case_id" else "slide"
        family = pathobench_family(dataset, task)
        rows.append(
            {
                "evaluation_id": f"pathobench.{dataset}.{slug(task)}",
                "suite_id": "pathobench",
                "dataset_id": dataset,
                "task_name": task,
                "task_family": family,
                "target": task,
                "sample_unit": sample_unit,
                "task_type": task_type,
                "num_samples": num_samples,
                "endpoint": f"{sample_unit}_level_{task_type}",
                "metric": metric,
                "direction": "higher",
                "protocol": "Public Patho-Bench split and task metadata; downstream protocol is task-specific in Patho-Bench.",
                "reference_url": blob_url("pathobench_hf", commit, config_rel),
                "audit_status": "parsed_primary_source",
                "audit_notes": f"Task identity is listed in available_splits.yaml; sample_col={sample_col}, task type, sample count, and metric are parsed from this task's config.yaml; family is deterministically inferred from the report taxonomy and endpoint label.",
            }
        )
    if len(rows) != 95:
        raise ValueError(f"expected 95 Patho-Bench tasks, found {len(rows)}")
    audits = {
        "sample_unit": {key: sum(row["sample_unit"] == key for row in rows) for key in ("case", "slide")},
        "task_type": {key: sum(row["task_type"] == key for row in rows) for key in ("classification", "survival")},
        "metric": {key: sum(row["metric"] == key for row in rows) for key in ("macro-ovr-auc", "bacc", "weighted_kappa", "cindex")},
    }
    expected_audits = {
        "sample_unit": {"case": 83, "slide": 12},
        "task_type": {"classification": 85, "survival": 10},
        "metric": {"macro-ovr-auc": 57, "bacc": 19, "weighted_kappa": 9, "cindex": 10},
    }
    if audits != expected_audits:
        raise ValueError(f"Patho-Bench config audit mismatch: {audits}")
    return rows


def build_exaone_pathobench_protocols(
    snapshot: Path, tasks: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Materialize EXAONE's reported recipe as distinct runnable protocols."""
    by_id = {str(row["evaluation_id"]): row for row in tasks}
    with snapshot.open(newline="", encoding="utf-8") as handle:
        evidence = list(csv.DictReader(handle))
    unique: dict[str, dict[str, str]] = {}
    for row in evidence:
        evaluation_id = row["evaluation_id"]
        previous = unique.setdefault(evaluation_id, row)
        if any(previous[key] != row[key] for key in ("base_evaluation_id", "metric", "source_task")):
            raise ValueError(f"inconsistent EXAONE protocol metadata for {evaluation_id}")
    rows: list[dict[str, object]] = []
    for evaluation_id, evidence_row in sorted(unique.items()):
        base_id = evidence_row["base_evaluation_id"]
        if base_id not in by_id:
            raise ValueError(f"EXAONE protocol references missing base task: {base_id}")
        base = by_id[base_id]
        row = dict(base)
        row.update(
            {
                "evaluation_id": evaluation_id,
                "protocol_id": evaluation_id,
                "endpoint": "exaone2025_cox_probe" if base["task_type"] == "survival" else "exaone2025_linear_probe",
                "metric": evidence_row["metric"],
                "protocol": "EXAONE Path 2.5 Table 4 protocol: official Patho-Bench predefined 5-fold/50-fold splits; fixed THREADS-style logistic-regression linear probe for classification and Cox probe for survival; all classification tasks use macro one-vs-rest AUROC and survival uses C-index.",
                "reference_url": evidence_row["reference_url"],
                "audit_status": "parsed_primary_source",
                "audit_notes": f"Protocol variant of {base_id}; shares task_identity_id and dataset_artifact_id but is kept separate because EXAONE reports macro-OvR AUROC for every classification task, including tasks whose current HF config uses bacc or weighted_kappa.",
            }
        )
        rows.append(row)
    if len(rows) != 80:
        raise ValueError(f"expected 80 EXAONE Patho-Bench protocol rows, found {len(rows)}")
    return rows


def build_threads_pathobench_protocols(
    snapshot: Path, tasks: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Materialize the public THREADS paper block as versioned protocols."""
    by_id = {str(row["evaluation_id"]): row for row in tasks}
    with snapshot.open(newline="", encoding="utf-8") as handle:
        evidence = list(csv.DictReader(handle))
    unique: dict[str, dict[str, str]] = {}
    for row in evidence:
        evaluation_id = row["evaluation_id"]
        previous = unique.setdefault(evaluation_id, row)
        if any(previous[key] != row[key] for key in ("base_evaluation_id", "metric", "source_table")):
            raise ValueError(f"inconsistent THREADS protocol metadata for {evaluation_id}")
    rows: list[dict[str, object]] = []
    for evaluation_id, evidence_row in sorted(unique.items()):
        base_id = evidence_row["base_evaluation_id"]
        if base_id not in by_id:
            raise ValueError(f"THREADS protocol references missing base task: {base_id}")
        base = by_id[base_id]
        row = dict(base)
        row.update(
            {
                "evaluation_id": evaluation_id,
                "protocol_id": evaluation_id,
                "endpoint": "threads2025_coxnet" if base["task_type"] == "survival" else "threads2025_balanced_linear_probe",
                "metric": evidence_row["metric"],
                "protocol": "THREADS Extended Data protocol: frozen representation, balanced linear probe with fixed cost 0.5 for classification or paper-reported CoxNet for survival, evaluated on the Patho-Bench-v1 public split/fold recipe.",
                "reference_url": evidence_row["reference_url"],
                "audit_status": "parsed_primary_source",
                "audit_notes": f"Public protocol variant of {base_id}, extracted from THREADS Extended Data Table {evidence_row['source_table']}; internal MGB tasks and supervised/fine-tuned rows are excluded.",
            }
        )
        rows.append(row)
    if len(rows) != 42:
        raise ValueError(f"expected 42 THREADS public protocol rows, found {len(rows)}")
    return rows


def build_eva(source: Path, commit: str) -> list[dict[str, object]]:
    root = source / "configs/vision/pathology/offline"
    excluded = {"camelyon16_small.yaml", "panda_small.yaml", "patch_camelyon_10shot.yaml"}
    rows = []
    for path in sorted(root.glob("*/*.yaml")):
        if path.name in excluded:
            continue
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(source).as_posix()
        dataset = path.stem
        endpoint = path.parent.name
        if endpoint == "segmentation":
            metric = "dice"
        elif "BinaryBalancedAccuracy" in text:
            metric = "balanced_accuracy"
        else:
            metric = "accuracy"
        runs = re.search(r"n_runs:.*?\$\{oc\.env:N_RUNS,\s*(\d+)\}", text)
        rows.append(
            {
                "evaluation_id": f"eva.{dataset}",
                "suite_id": "eva",
                "dataset_id": dataset,
                "task_name": f"{dataset} {endpoint}",
                "task_family": endpoint,
                "target": f"{dataset} {endpoint} target",
                "sample_unit": "slide" if dataset in {"camelyon16", "panda"} else "image",
                "task_type": endpoint,
                "num_samples": "not_reported",
                "endpoint": f"offline_{endpoint}",
                "metric": metric,
                "direction": "higher",
                "protocol": f"EVA offline config; frozen encoder evaluation; {runs.group(1) if runs else 'configured'} runs; monitored {metric}.",
                "reference_url": blob_url("eva", commit, relative),
                "audit_status": "parsed_primary_source",
                "audit_notes": "Canonical endpoint excludes explicit small-data and 10-shot variants to avoid counting protocol variants as independent tasks.",
            }
        )
    if len(rows) != 13:
        raise ValueError(f"expected 13 canonical EVA tasks, found {len(rows)}")
    return rows


def build_eva_leaderboard_protocols(
    source: Path, commit: str, tasks: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Add each reported EVA leaderboard column as a distinct protocol."""
    by_id = {str(row["evaluation_id"]): row for row in tasks}
    rows: list[dict[str, object]] = []
    for spec in required_additional_protocols():
        evaluation_id = spec["evaluation_id"]
        config_rel = f"configs/vision/pathology/offline/{spec['config']}"
        config_path = source / config_rel
        if not config_path.is_file():
            raise ValueError(f"missing EVA leaderboard config: {config_rel}")
        dataset = Path(spec["config"]).stem
        identity_dataset = (
            "patch_camelyon" if dataset == "patch_camelyon_10shot" else dataset
        )
        base = by_id.get(f"eva.{identity_dataset}")
        if base is not None:
            row = dict(base)
        else:
            endpoint = Path(spec["config"]).parent.name
            row = {
                "suite_id": "eva",
                "dataset_id": dataset,
                "task_name": f"{dataset} {endpoint}",
                "task_family": endpoint,
                "target": f"{dataset} {endpoint} target",
                "sample_unit": "slide" if dataset in {"camelyon16_small", "panda_small"} else "image",
                "task_type": endpoint,
                "num_samples": "not_reported",
                "dataset_artifact_id": f"artifact.eva.{dataset}",
                "task_identity_id": f"task.eva.{dataset}.{endpoint}",
            }
        row.update(
            {
                "evaluation_id": evaluation_id,
                "protocol_id": evaluation_id,
                "endpoint": f"leaderboard_{spec['split']}_{row['task_type']}",
                "metric": spec["metric"],
                "direction": "higher",
                "protocol": (
                    f"EVA pathology leaderboard; reported split={spec['split']}; "
                    f"reported metric={spec['metric']}; mean over {spec['runs']} runs. "
                    "Config monitor metrics are checkpoint-selection metadata and are not substituted for the documented leaderboard metric."
                ),
                "reference_url": blob_url("eva", commit, config_rel),
                "audit_status": "parsed_primary_source",
                "audit_notes": (
                    f"Protocol-specific score column backed by {spec['config']}; kept distinct from the generic EVA config row and from other report splits."
                ),
            }
        )
        rows.append(row)
    if len(rows) != 15 or len({str(row["evaluation_id"]) for row in rows}) != 15:
        raise ValueError("expected 15 distinct EVA leaderboard protocols")
    return rows


def yaml_scalar(text: str, name: str) -> str:
    match = re.search(rf"(?m)^{re.escape(name)}:\s*([^\n#]+)", text)
    if not match:
        raise ValueError(f"missing {name}")
    return match.group(1).strip().strip("'\"")


def build_thunder(source: Path, commit: str) -> list[dict[str, object]]:
    root = source / "src/thunder/config/dataset"
    rows = []
    for path in sorted(root.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        dataset = yaml_scalar(text, "dataset_name")
        endpoint = "segmentation" if '"segmentation"' in text else "linear_probing"
        metric = "dice" if endpoint == "segmentation" else "f1"
        sample_counts = []
        for key in ("nb_train_samples", "nb_val_samples", "nb_test_samples"):
            match = re.search(rf"(?m)^{key}:\s*([0-9_]+)", text)
            if match:
                sample_counts.append(int(match.group(1).replace("_", "")))
        num_samples: int | str = sum(sample_counts) if len(sample_counts) == 3 else "not_reported"
        rows.append(
            {
                "evaluation_id": f"thunder.{dataset}.{endpoint}",
                "suite_id": "thunder",
                "dataset_id": dataset,
                "task_name": f"{dataset} {endpoint}",
                "task_family": "segmentation" if endpoint == "segmentation" else "classification",
                "target": f"{dataset} {'segmentation mask' if endpoint == 'segmentation' else 'class label'}",
                "sample_unit": "image",
                "task_type": "segmentation" if endpoint == "segmentation" else "classification",
                "num_samples": num_samples,
                "endpoint": endpoint,
                "metric": metric,
                "direction": "higher",
                "protocol": "THUNDER dataset config with its code-declared compatible primary downstream endpoint; frozen linear probe for classification or segmentation decoder.",
                "reference_url": blob_url("thunder", commit, path.relative_to(source).as_posix()),
                "audit_status": "parsed_primary_source",
                "audit_notes": "Code-backed dataset endpoint. The current per-dataset score table covers 16 of the 17 classification configs; STARC9 has no column in that table.",
            }
        )
    if len(rows) != 21:
        raise ValueError(f"expected 21 THUNDER endpoints, found {len(rows)}")
    return rows


def build_hest(commit: str) -> list[dict[str, object]]:
    ref = blob_url("hest", commit, "README.md")
    datasets = ["IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC", "LUNG", "LYMPH_IDC"]
    return [
        {
            "evaluation_id": f"hest.{dataset.lower()}.gene_expression",
            "suite_id": "hest",
            "dataset_id": dataset.lower(),
            "task_name": f"{dataset} morphology-to-gene-expression",
            "task_family": "spatial_transcriptomics",
            "target": "expression of 50 highly variable genes",
            "sample_unit": "spatial_transcriptomics_spot",
            "task_type": "regression",
            "num_samples": "not_reported",
            "endpoint": "ridge_gene_expression_prediction",
            "metric": "pearson_r",
            "direction": "higher",
            "protocol": "Predict 50 highly variable genes from 112x112 um morphology regions at 0.5 um/px using ridge regression after PCA to 256 factors.",
            "reference_url": ref,
            "audit_status": "parsed_primary_source",
            "audit_notes": "One endpoint per HEST-Benchmark organ/cancer task; macro-average is not treated as an independent task.",
        }
        for dataset in datasets
    ]


def build_pathorob(commit: str) -> list[dict[str, object]]:
    ref = blob_url("pathorob", commit, "README.md")
    datasets = ["tcga_2x2", "tcga_4x4", "camelyon", "tolkach_esca"]
    metrics = [
        ("robustness_index", "representation_robustness"),
        ("average_performance_drop", "shortcut_generalization"),
        ("clustering_score", "clustering_robustness"),
    ]
    rows = []
    for metric, family in metrics:
        for dataset in datasets:
            is_apd = metric == "average_performance_drop"
            rows.append(
                {
                    "evaluation_id": f"pathorob.{dataset}.{metric}",
                    "suite_id": "pathorob",
                    "dataset_id": dataset,
                    "task_name": f"{dataset} {metric}",
                    "task_family": family,
                    "target": metric,
                    "sample_unit": "patch",
                    "task_type": "robustness_analysis",
                    "num_samples": "not_reported",
                    "endpoint": metric,
                    "metric": metric,
                    "direction": "higher",
                    "protocol": (
                        "PathoROB endpoint catalog across TCGA 2x2, TCGA 4x4, Camelyon, and Tolkach ESCA. Signed APD is higher/closer to zero when better; APD-ID and APD-OOD require separate score-bearing protocols."
                        if is_apd
                        else "PathoROB center-robustness protocol across TCGA 2x2, TCGA 4x4, Camelyon, and Tolkach ESCA benchmark datasets."
                    ),
                    "reference_url": ref,
                    "audit_status": "parsed_primary_source",
                    "audit_notes": (
                        "TCGA has distinct 2x2 and 4x4 robustness endpoints. Generic APD rows are catalog placeholders only: the published APD-ID and APD-OOD means are represented by separate Nature-2026 protocols and never merged into this ambiguous endpoint."
                        if is_apd
                        else "TCGA has distinct 2x2 and 4x4 robustness endpoints; current RI leaderboard reports TCGA 2x2, Camelyon, and Tolkach ESCA only, so TCGA 4x4 remains an unscored catalog endpoint."
                    ),
                }
            )
    if len(rows) != 12:
        raise AssertionError("PathoROB endpoint construction drifted")
    return rows


def build_pathorob_nature_protocols(
    snapshot: Path, tasks: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Materialize the five score-bearing protocols in the Nature Source Data."""
    with snapshot.open(newline="", encoding="utf-8") as handle:
        evidence = [row for row in csv.DictReader(handle) if row["source_scope"] == "published_paper"]
    if len(evidence) != 100:
        raise ValueError(f"expected 100 published PathoROB rows, found {len(evidence)}")
    by_id = {str(row["evaluation_id"]): row for row in tasks}
    unique = {row["evaluation_id"]: row for row in evidence}
    expected = {
        "pathorob.nature2026.all_datasets.apd_id",
        "pathorob.nature2026.all_datasets.apd_ood",
        "pathorob.nature2026.camelyon.clustering_score",
        "pathorob.nature2026.tcga_4x4.clustering_score",
        "pathorob.nature2026.tolkach_esca.clustering_score",
    }
    if set(unique) != expected:
        raise ValueError(f"unexpected PathoROB Nature protocol set: {sorted(unique)}")
    output: list[dict[str, object]] = []
    for evaluation_id in sorted(unique):
        source = unique[evaluation_id]
        endpoint = source["endpoint"]
        dataset = source["dataset_scope"]
        if endpoint == "clustering_score":
            base_id = f"pathorob.{dataset}.clustering_score"
            if base_id not in by_id:
                raise ValueError(f"missing PathoROB clustering base protocol: {base_id}")
            row = dict(by_id[base_id])
            row.update(
                {
                    "evaluation_id": evaluation_id,
                    "protocol_id": evaluation_id,
                    "metric": "clustering_score",
                    "direction": "higher",
                    "protocol": source["protocol"],
                    "reference_url": source["reference_url"],
                    "audit_status": "parsed_primary_source",
                    "audit_notes": "Canonical published mean from Nature Source Data Fig. 6b; approximately [-1,1], normalized as (score+1)*50. Shares task identity and dataset artifact with the generic PathoROB clustering endpoint.",
                }
            )
        else:
            row = {
                "evaluation_id": evaluation_id,
                "protocol_id": evaluation_id,
                "task_identity_id": f"task.pathorob.all_datasets.{endpoint}",
                "dataset_artifact_id": "artifact.pathorob.camelyon+tcga_4x4+tolkach_esca",
                "suite_id": "pathorob",
                "dataset_id": "all_datasets",
                "task_name": f"all-dataset {endpoint}",
                "task_family": "shortcut_generalization",
                "target": "signed relative generalization-performance change under spurious medical-center correlation",
                "sample_unit": "patch",
                "task_type": "robustness_analysis",
                "num_samples": "n=60 model-level observations (20 repetitions x 3 datasets)",
                "endpoint": endpoint,
                "metric": "average_performance_drop_percent",
                "direction": "higher_closer_to_zero",
                "protocol": source["protocol"],
                "reference_url": source["reference_url"],
                "audit_status": "parsed_primary_source",
                "audit_notes": "Canonical published signed APD mean from Nature Source Data Fig. 3d. Registry-eligible but factor-analysis-ineligible: the source defines no bounded common-scale normalization, so normalized_score is intentionally blank and its score audit_status excludes it from matrix loading.",
            }
        output.append(row)
    return output


EXACT_TASK_IDENTITIES = {
    "task.bach.four_class_classification": (
        {"eva.bach", "thunder.bach.linear_probing"},
        "BACH four-class histology label",
    ),
    "task.bracs.seven_class_classification": (
        {"eva.bracs", "thunder.bracs.linear_probing"},
        "BRACS seven-class lesion label",
    ),
    "task.breakhis.selected_four_class_classification": (
        {"eva.breakhis", "thunder.break_his.linear_probing"},
        "BreakHis selected four-class subtype label",
    ),
    "task.crc100k.nine_class_classification": (
        {"eva.crc", "thunder.crc.linear_probing"},
        "CRC-100K nine-class tissue label",
    ),
    "task.mhist.hp_vs_ssa_classification": (
        {"eva.mhist", "thunder.mhist.linear_probing"},
        "MHIST hyperplastic-polyp versus sessile-serrated-adenoma label",
    ),
    "task.patchcamelyon.binary_metastasis_classification": (
        {"eva.patch_camelyon", "thunder.patch_camelyon.linear_probing"},
        "PatchCamelyon binary metastatic-tissue label",
    ),
    "task.panda.isup_grade": (
        {"pathobench.panda.isup-grade", "eva.panda"},
        "PANDA ISUP grade",
    ),
}


def materialize_task_contracts(tasks: list[dict[str, object]]) -> None:
    """Attach explicit artifact, biological-task, and runnable-protocol IDs."""
    by_evaluation = {str(row["evaluation_id"]): row for row in tasks}
    if len(by_evaluation) != len(tasks):
        raise ValueError("duplicate evaluation_id before contract materialization")
    for row in tasks:
        evaluation_id = str(row["evaluation_id"])
        row["protocol_id"] = evaluation_id
        row["task_identity_id"] = f"task.{row['suite_id']}.{row['dataset_id']}.{slug(str(row['task_name']))}"
        row["dataset_artifact_id"] = f"artifact.{row['suite_id']}.{row['dataset_id']}"
    for identity, (members, target) in EXACT_TASK_IDENTITIES.items():
        missing = members - by_evaluation.keys()
        if missing:
            raise ValueError(f"exact task identity {identity} references missing rows: {sorted(missing)}")
        for evaluation_id in members:
            by_evaluation[evaluation_id]["task_identity_id"] = identity
            by_evaluation[evaluation_id]["target"] = target
    required = (
        "dataset_artifact_id",
        "task_identity_id",
        "protocol_id",
        "target",
        "sample_unit",
        "task_type",
        "num_samples",
    )
    for row in tasks:
        for field in required:
            if row.get(field) in (None, ""):
                row[field] = "not_reported"


def markdown_rows(text: str, header_prefix: str) -> list[list[str]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header_prefix))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def linked_label(cell: str) -> tuple[str, str]:
    match = re.search(r"\[([^]]+)\]\(([^)]+)\)", cell)
    if match:
        return match.group(1), match.group(2)
    return html.unescape(re.sub(r"<[^>]+>", "", cell)).strip(), ""


def score_row(
    reported_alias: str,
    model_id: str,
    evaluation_id: str,
    value: float,
    normalized: float | None,
    suite: str,
    metric: str,
    ref: str,
    source_locator: str,
    lineage: str,
    audit: str = "parsed_primary_source",
    uncertainty: str = "not_reported",
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "reported_model_alias": reported_alias,
        "model_revision": "not_reported",
        "evaluation_id": evaluation_id,
        "value": f"{value:.6g}",
        "normalized_score": "" if normalized is None else f"{normalized:.6g}",
        "suite_id": suite,
        "metric": metric,
        "reference_url": ref,
        "source_locator": source_locator,
        "extraction_date": "2026-08-05",
        "review_status": "machine_parsed_single_source",
        "uncertainty": uncertainty,
        "lineage": lineage,
        "audit_status": audit,
    }


def parse_exaone_pathobench_scores(
    snapshot: Path, tasks: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load the 80-task, seven-model Patho-Bench table from EXAONE Path 2.5."""
    task_contracts = {
        str(row["evaluation_id"]): row for row in tasks if row["suite_id"] == "pathobench"
    }
    expected_models = {
        "CHIEF", "GigaPath", "PRISM", "TITAN", "H-optimus-0", "UNI2-h",
        "EXAONE Path 2.5",
    }
    scores: list[dict[str, object]] = []
    aliases: dict[str, dict[str, object]] = {}
    seen_cells: set[tuple[str, str]] = set()
    seen_tasks: set[str] = set()
    with snapshot.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "source_task", "base_evaluation_id", "evaluation_id", "metric", "model_alias", "value",
        "reference_url", "source_archive_sha256",
    }
    if not rows or set(rows[0]) != required:
        raise ValueError(f"unexpected EXAONE Patho-Bench snapshot schema: {set(rows[0]) if rows else set()}")
    for row in rows:
        evaluation_id = row["evaluation_id"]
        if evaluation_id not in task_contracts:
            raise ValueError(f"EXAONE Table 4 references unknown task: {evaluation_id}")
        contract = task_contracts[evaluation_id]
        if row["metric"] != contract["metric"]:
            raise ValueError(
                f"metric mismatch for {evaluation_id}: {row['metric']} != {contract['metric']}"
            )
        alias = row["model_alias"]
        if alias not in expected_models:
            raise ValueError(f"unexpected EXAONE Table 4 model: {alias}")
        model_id = EXAONE_MODEL_IDS[alias]
        cell = (model_id, evaluation_id)
        if cell in seen_cells:
            raise ValueError(f"duplicate EXAONE Table 4 cell: {cell}")
        seen_cells.add(cell)
        seen_tasks.add(evaluation_id)
        value = float(row["value"])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"out-of-range EXAONE Table 4 score: {row}")
        ref = row["reference_url"]
        aliases[alias] = alias_row(
            alias,
            model_id,
            "pathobench",
            ref,
            "Model column in EXAONE Path 2.5 Table 4; all columns use the paper's shared linear-probing settings. H-optimus-0 and UNI2-h use mean pooling as explicitly reported.",
        )
        scores.append(
            score_row(
                alias,
                model_id,
                evaluation_id,
                value,
                value * 100.0,
                "pathobench",
                row["metric"],
                ref,
                f"paper=EXAONE Path 2.5|table=4|source_task={row['source_task']}|model_column={alias}",
                f"arxiv:2512.14019v1 source@sha256:{row['source_archive_sha256']}:tabs/pathobench_result.tex -> scripts/extract_exaone_pathobench.py -> {snapshot.name} -> build_registry.py -> scores.csv",
            )
        )
    if len(scores) != 560 or len(seen_tasks) != 80 or {row["model_alias"] for row in rows} != expected_models:
        raise ValueError(
            f"EXAONE Table 4 audit failed: scores={len(scores)}, tasks={len(seen_tasks)}, "
            f"models={sorted({row['model_alias'] for row in rows})}"
        )
    return scores, list(aliases.values())


def parse_threads_pathobench_scores(
    snapshot: Path, tasks: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load the 42-task, eight-representation public THREADS result block."""
    contracts = {
        str(row["evaluation_id"]): row for row in tasks if row["suite_id"] == "pathobench"
    }
    with snapshot.open(newline="", encoding="utf-8") as handle:
        evidence = list(csv.DictReader(handle))
    required = {
        "source_table", "base_evaluation_id", "evaluation_id", "metric", "model_alias",
        "value", "uncertainty", "reference_url", "source_archive_sha256",
        "source_html_sha256",
    }
    if not evidence or set(evidence[0]) != required:
        raise ValueError("unexpected THREADS snapshot schema")
    scores: list[dict[str, object]] = []
    aliases: dict[str, dict[str, object]] = {}
    seen: set[tuple[str, str]] = set()
    for row in evidence:
        evaluation_id = row["evaluation_id"]
        contract = contracts.get(evaluation_id)
        if contract is None or contract["metric"] != row["metric"]:
            raise ValueError(f"THREADS task/metric contract mismatch: {evaluation_id}")
        alias = row["model_alias"]
        if alias not in THREADS_MODEL_IDS:
            raise ValueError(f"unexpected THREADS representation: {alias}")
        model_id = THREADS_MODEL_IDS[alias]
        cell = (model_id, evaluation_id)
        if cell in seen:
            raise ValueError(f"duplicate THREADS score cell: {cell}")
        seen.add(cell)
        value = float(row["value"])
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"out-of-range THREADS score: {row}")
        normalized = (value + 1.0) * 50.0 if row["metric"] == "weighted_kappa" else value * 100.0
        ref = row["reference_url"]
        aliases[alias] = alias_row(
            alias,
            model_id,
            "pathobench",
            ref,
            "Frozen representation row in THREADS Extended Data. Patch-encoder mean-pooling and slide-encoder rows have deliberately distinct model IDs.",
        )
        scores.append(
            score_row(
                alias,
                model_id,
                evaluation_id,
                value,
                normalized,
                "pathobench",
                row["metric"],
                ref,
                f"paper=THREADS|extended_data_table={row['source_table']}|model_row={alias}|base_task={row['base_evaluation_id']}",
                f"arxiv:2501.16652v1 source@sha256:{row['source_archive_sha256']} -> scripts/extract_threads_pathobench.py -> {snapshot.name} -> build_registry.py -> scores.csv",
                uncertainty=row["uncertainty"],
            )
        )
    if len(scores) != 336 or len(seen) != 336:
        raise ValueError(f"THREADS public score audit failed: {len(scores)}")
    return scores, list(aliases.values())


def parse_hest_scores(source: Path, commit: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ref = blob_url("hest", commit, "README.md")
    text = (source / "README.md").read_text(encoding="utf-8")
    rows = markdown_rows(text, "| Model | Average | IDC |")
    tasks = ["idc", "prad", "paad", "skcm", "coad", "read", "ccrcc", "lung", "lymph_idc"]
    scores, aliases = [], []
    for cells in rows:
        alias, model_ref = linked_label(cells[0])
        model_id = canonical_model(alias)
        aliases.append(alias_row(alias, model_id, "hest", model_ref or ref, "Alias and optional checkpoint URL parsed from the current HEST results table."))
        for task, raw in zip(tasks, cells[2:11]):
            value = float(raw)
            scores.append(
                score_row(
                    alias,
                    model_id,
                    f"hest.{task}.gene_expression",
                    value,
                    (value + 1.0) * 50.0,
                    "hest",
                    "pearson_r",
                    ref,
                    f"table=HEST-Benchmark results|table_date=2026-04-03|model_row={alias}|task_column={task.upper()}",
                    f"hest@{commit}:README.md -> parse_hest_scores -> scores.csv",
                )
            )
    if not rows or len(scores) != len(rows) * 9:
        raise ValueError("HEST results table parse failed")
    return scores, aliases


def html_cells(row: str) -> list[str]:
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.I | re.S)
    return [html.unescape(re.sub(r"<[^>]+>", " ", cell)).replace("\xa0", " ").strip() for cell in cells]


def parse_thunder_scores(source: Path, commit: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    relative = "docs/leaderboards.md"
    ref = blob_url("thunder", commit, relative)
    text = (source / relative).read_text(encoding="utf-8")
    table = re.search(r'<table id="perDatasetLinearTable".*?</table>', text, re.S)
    if not table:
        raise ValueError("THUNDER per-dataset linear table not found")
    datasets = [
        "bach", "bracs", "break_his", "ccrcc", "crc", "esca", "mhist", "patch_camelyon",
        "tcga_crc_msi", "tcga_tils", "tcga_uniform", "wilds", "spider_breast",
        "spider_colorectal", "spider_skin", "spider_thorax",
    ]
    scores, aliases = [], []
    for raw_row in re.findall(r"<tr>(.*?)</tr>", table.group(0), flags=re.S):
        cells = html_cells(raw_row)
        if len(cells) < 20:
            continue
        alias = cells[0]
        model_id = canonical_model(alias)
        aliases.append(alias_row(alias, model_id, "thunder", ref, f"Domain={cells[1]}; type={cells[2]}; alias parsed from current per-dataset linear-probe table."))
        for dataset, cell in zip(datasets, cells[3:19]):
            match = re.match(r"\s*(-?\d+(?:\.\d+)?)", cell)
            if not match:
                raise ValueError(f"bad THUNDER score cell: {cell!r}")
            value = float(match.group(1))
            scores.append(
                score_row(
                    alias,
                    model_id,
                    f"thunder.{dataset}.linear_probing",
                    value,
                    value,
                    "thunder",
                    "f1",
                    ref,
                    f"table=perDatasetLinearTable|table_update=2026-04-08|model_row={alias}|task_column={dataset}",
                    f"thunder@{commit}:docs/leaderboards.md -> parse_thunder_scores -> scores.csv",
                )
            )
    if not scores or len(scores) % 16:
        raise ValueError("THUNDER table did not produce 16 scores per model")
    return scores, aliases


def parse_pathorob_scores(source: Path, commit: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ref = blob_url("pathorob", commit, "README.md")
    text = (source / "README.md").read_text(encoding="utf-8")
    rows = markdown_rows(text, "| Rank | Foundation Model")
    datasets = ["tcga_2x2", "camelyon", "tolkach_esca"]
    scores, aliases = [], []
    for cells in rows:
        alias, external_ref = linked_label(cells[1])
        # linked_label sees the citation label for external rows; strip it from the model name.
        clean_alias = re.sub(r"\s*\[\[\d+\]\].*$", "", alias).strip()
        if clean_alias.isdigit() or not clean_alias:
            clean_alias = html.unescape(re.sub(r"<sup>.*?</sup>", "", cells[1])).strip()
        model_id = canonical_model(clean_alias)
        external = "<sup>" in cells[1]
        if external:
            citation = re.search(r"\[\[\d+\]\]\(([^)]+)\)", cells[1])
            external_ref = citation.group(1) if citation else external_ref
        aliases.append(alias_row(clean_alias, model_id, "pathorob", external_ref or ref, "External-publication value is unvalidated by PathoROB authors." if external else "Alias parsed from the PathoROB-computed RI table."))
        for dataset, raw in zip(datasets, cells[2:5]):
            value = float(raw)
            scores.append(
                score_row(
                    clean_alias,
                    model_id,
                    f"pathorob.{dataset}.robustness_index",
                    value,
                    value * 100.0,
                    "pathorob",
                    "robustness_index",
                    ref,
                    f"table=Leaderboard: Robustness Index|source_snapshot_date=2026-04-02|model_row={clean_alias}|task_column={dataset}",
                    f"pathorob@{commit}:README.md -> parse_pathorob_scores -> scores.csv",
                    "reported_external" if external else "parsed_primary_source",
                )
            )
    if not rows or len(scores) != len(rows) * 3:
        raise ValueError("PathoROB RI table parse failed")
    return scores, aliases


def parse_pathorob_nature_scores(
    snapshot: Path, tasks: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load the 100 canonical means while excluding version-different examples."""
    with snapshot.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    required = {
        "source_scope", "source_table", "source_row", "model_alias", "model_id",
        "dataset_scope", "endpoint", "evaluation_id", "metric", "value", "value_unit",
        "uncertainty", "direction", "protocol", "reference_url", "source_locator",
        "source_revision", "source_sha256", "inclusion_status", "inclusion_reason",
    }
    if not all_rows or set(all_rows[0]) != required:
        raise ValueError("unexpected PathoROB Nature score snapshot schema")
    paper = [row for row in all_rows if row["source_scope"] == "published_paper"]
    examples = [row for row in all_rows if row["source_scope"] == "repository_example"]
    if len(paper) != 100 or len(examples) != 22:
        raise ValueError(
            f"PathoROB source-scope audit failed: paper={len(paper)}, examples={len(examples)}"
        )
    contracts = {str(row["evaluation_id"]): row for row in tasks if row["suite_id"] == "pathorob"}
    scores: list[dict[str, object]] = []
    aliases: dict[str, dict[str, object]] = {}
    seen: set[tuple[str, str]] = set()
    eligible = 0
    for row in paper:
        evaluation_id = row["evaluation_id"]
        contract = contracts.get(evaluation_id)
        if contract is None or contract["metric"] != row["metric"]:
            raise ValueError(f"PathoROB Nature task/metric mismatch: {evaluation_id}")
        model_id = row["model_id"]
        alias = row["model_alias"]
        cell = (model_id, evaluation_id)
        if cell in seen:
            raise ValueError(f"duplicate PathoROB Nature score cell: {cell}")
        seen.add(cell)
        value = float(row["value"])
        if row["endpoint"] == "clustering_score":
            if row["inclusion_status"] != "canonical_analysis_eligible" or not -1.0 <= value <= 1.0:
                raise ValueError(f"invalid canonical clustering row: {row}")
            normalized: float | None = (value + 1.0) * 50.0
            audit = "parsed_primary_source"
            eligible += 1
        else:
            if row["inclusion_status"] != "canonical_analysis_ineligible":
                raise ValueError(f"invalid canonical APD row: {row}")
            normalized = None
            audit = "parsed_primary_source_analysis_ineligible"
        score = score_row(
            alias,
            model_id,
            evaluation_id,
            value,
            normalized,
            "pathorob",
            row["metric"],
            row["reference_url"],
            row["source_locator"],
            f"{row['source_revision']}@sha256:{row['source_sha256']} -> scripts/extract_pathorob_scores.py -> {snapshot.name} -> build_registry.py -> scores.csv",
            audit,
            uncertainty=row["uncertainty"],
        )
        # Preserve the exact published decimal rather than score_row's compact formatting.
        score["value"] = row["value"]
        score["extraction_date"] = "2026-08-06"
        scores.append(score)
        aliases[alias] = alias_row(
            alias,
            model_id,
            "pathorob",
            row["reference_url"],
            "Alias in the canonical PathoROB Nature Source Data workbook.",
        )
    if len(scores) != 100 or eligible != 60 or len(seen) != 100:
        raise ValueError(
            f"PathoROB Nature score audit failed: scores={len(scores)}, eligible={eligible}"
        )
    return scores, list(aliases.values())


def alias_row(alias: str, model_id: str, suite: str, ref: str, notes: str) -> dict[str, object]:
    return {"alias": alias, "model_id": model_id, "suite_id": suite, "reference_url": ref, "audit_notes": notes}


def build_dedup(tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    ids = {str(row["evaluation_id"]) for row in tasks}
    identity_members: dict[str, list[str]] = {}
    for task in tasks:
        identity_members.setdefault(str(task["task_identity_id"]), []).append(str(task["evaluation_id"]))
    duplicates = {identity: sorted(members) for identity, members in identity_members.items() if len(members) > 1}
    protocol_variant_identities = {
        identity
        for identity, members in duplicates.items()
        if any(
            member.startswith((
                "pathobench.exaone2025.",
                "pathobench.threads2025.",
                "eva.leaderboard.",
                "pathorob.nature2026.",
                "eva.h0mini2025.",
                "hest.h0mini2025.",
                "plism.repo2025.",
                "virchow2g2024.",
                "titan2025.",
                "hest.hoptimus1report2025.",
                "hest.virchow2paper2024.",
                "virchow2paper2024.",
                "virchowpaper2023.",
                "unipaper2023.",
                "thunder.genbio2026.",
                "pathorob.genbio2026.",
                "eva.miccai2025.",
                "eva.openmidnight_tr2025.",
                "conch.natmed2024.",
                "conch15.titan2025.",
                "phikon.",
                "phikon-v2.",
                "hibou-b.hibou2024.",
                "hibou-l.hibou2024.",
                "musk.nature2025.",
                "gpfm.nbe2025.",
            ))
            for member in members
        )
    }
    expected_duplicate_identities = set(EXACT_TASK_IDENTITIES) | protocol_variant_identities
    pathobench_variant_identities = {
        identity for identity, members in duplicates.items()
        if any(member.startswith("pathobench.exaone2025.") for member in members)
    }
    eva_variant_identities = {
        identity for identity, members in duplicates.items()
        if any(member.startswith("eva.leaderboard.") for member in members)
    }
    pathorob_variant_identities = {
        identity for identity, members in duplicates.items()
        if any(member.startswith("pathorob.nature2026.") for member in members)
    }
    h0mini_variant_identities = {
        identity for identity, members in duplicates.items()
        if any(member.startswith(("eva.h0mini2025.", "hest.h0mini2025.", "plism.repo2025.")) for member in members)
    }
    group_c_variant_identities = {
        identity for identity, members in duplicates.items()
        if any(member.startswith(("virchow2g2024.", "titan2025.")) for member in members)
    }
    wave_d_variant_identities = {
        identity for identity, members in duplicates.items()
        if any(member.startswith(("hest.hoptimus1report2025.", "hest.virchow2paper2024.", "virchow2paper2024.", "virchowpaper2023.", "unipaper2023.")) for member in members)
    }
    group_b_variant_identities = {
        identity for identity, members in duplicates.items()
        if any(member.startswith(("thunder.genbio2026.", "pathorob.genbio2026.", "eva.miccai2025.", "eva.openmidnight_tr2025.")) for member in members)
    }
    wave_e_variant_identities = {
        identity for identity, members in duplicates.items()
        if any(member.startswith(("conch.natmed2024.", "conch15.titan2025.", "phikon.", "phikon-v2.")) for member in members)
    }
    wave_f_variant_identities = {
        identity for identity, members in duplicates.items()
        if any(member.startswith(("hibou-b.hibou2024.", "hibou-l.hibou2024.", "musk.nature2025.", "gpfm.nbe2025.")) for member in members)
    }
    if (
        set(duplicates) != expected_duplicate_identities
        or len(pathobench_variant_identities) != 80
        or len(eva_variant_identities) != 11
        or len(pathorob_variant_identities) != 3
        or len(h0mini_variant_identities) != 20
        or len(group_c_variant_identities) != 64
        or len(wave_d_variant_identities) != 45
        or len(group_b_variant_identities) != 20
        or len(wave_e_variant_identities) != 75
        or len(wave_f_variant_identities) != 47
    ):
        raise ValueError(
            "exact duplicate task identity audit failed: "
            f"pathobench_variants={len(pathobench_variant_identities)}, "
            f"eva_variants={len(eva_variant_identities)}, "
            f"wave_f_variants={len(wave_f_variant_identities)}, duplicates={duplicates}"
        )

    rows = []
    for identity, members in sorted(duplicates.items()):
        canonical = next(
            (
                member for member in members
                if not member.startswith(("pathobench.exaone2025.", "pathobench.threads2025.", "eva.leaderboard.", "pathorob.nature2026.", "eva.h0mini2025.", "hest.h0mini2025.", "plism.repo2025.", "virchow2g2024.", "titan2025.", "hest.hoptimus1report2025.", "hest.virchow2paper2024.", "virchow2paper2024.", "virchowpaper2023.", "unipaper2023.", "thunder.genbio2026.", "pathorob.genbio2026.", "eva.miccai2025.", "eva.openmidnight_tr2025.", "conch.natmed2024.", "conch15.titan2025.", "phikon.", "phikon-v2.", "hibou-b.hibou2024.", "hibou-l.hibou2024.", "musk.nature2025.", "gpfm.nbe2025."))
            ),
            members[0],
        )
        for member in members:
            rows.append(
                {
                    "group_id": "exact." + identity.removeprefix("task."),
                    "match_type": "exact",
                    "task_identity_id": identity,
                    "canonical_evaluation_id": canonical,
                    "member_evaluation_id": member,
                    "decision": "link_only",
                    "rationale": "Shared dataset artifact and prediction target, but potentially different runnable protocol and metric; retain distinct protocol_id values and never overwrite protocol-specific scores.",
                }
            )

    semantic_groups: list[tuple[str, str, list[str], str]] = [
        ("semantic.bracs-granularity", "pathobench.bracs.slidelevel-fine", ["pathobench.bracs.slidelevel-fine", "eva.bracs", "thunder.bracs.linear_probing"], "Shared BRACS source and related seven-class labels, but Patho-Bench is slide-level while EVA/THUNDER use their own image-level protocols."),
        ("semantic.camelyon", "eva.camelyon16", ["eva.camelyon16", "thunder.patch_camelyon.linear_probing", "thunder.wilds.linear_probing", "pathorob.camelyon.robustness_index", "pathorob.camelyon.average_performance_drop", "pathorob.camelyon.clustering_score"], "Camelyon-derived data recur, but endpoints span WSI classification, patch classification, center robustness, shortcut generalization, and clustering."),
    ]
    for group_id, canonical, members, rationale in semantic_groups:
        missing = [member for member in members if member not in ids]
        if missing:
            raise ValueError(f"dedup group {group_id} references missing tasks: {missing}")
        for member in members:
            rows.append(
                {
                    "group_id": group_id,
                    "match_type": "semantic",
                    "task_identity_id": "not_applicable_semantic_group",
                    "canonical_evaluation_id": canonical,
                    "member_evaluation_id": member,
                    "decision": "keep_separate",
                    "rationale": rationale,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("/tmp/pathopress_sources"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument(
        "--exaone-pathobench-snapshot",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "source_data/exaone_path_2_5_pathobench_2512.14019v1.csv",
    )
    parser.add_argument(
        "--threads-pathobench-snapshot",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "source_data/threads_pathobench_2501.16652v1.csv",
    )
    parser.add_argument(
        "--pathorob-nature-snapshot",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "source_data/pathorob_nature2026_and_repo_examples.csv",
    )
    parser.add_argument(
        "--h0mini-uni2h-snapshot",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "source_data/h0mini_uni2h_official_scores_2025.csv",
    )
    parser.add_argument(
        "--group-c-official-snapshot",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "source_data/virchow2g_gigapath_titan_official_scores_2024_2025.csv",
    )
    parser.add_argument(
        "--wave-d-hoptimus-snapshot", type=Path,
        default=Path(__file__).resolve().parents[1] / "source_data/wave_d_hoptimus1_official_report_2025.csv",
    )
    parser.add_argument(
        "--wave-d-virchow2-snapshot", type=Path,
        default=Path(__file__).resolve().parents[1] / "source_data/wave_d_virchow2_paper_2408.00738.csv",
    )
    parser.add_argument(
        "--wave-d-virchow-snapshot", type=Path,
        default=Path(__file__).resolve().parents[1] / "source_data/wave_d_virchow_paper_2309.07778.csv",
    )
    parser.add_argument(
        "--wave-d-uni-snapshot", type=Path,
        default=Path(__file__).resolve().parents[1] / "source_data/wave_d_uni_paper_2308.15474.csv",
    )
    parser.add_argument(
        "--group-b-snapshot-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "source_data",
    )
    parser.add_argument(
        "--wave-e-snapshot-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "source_data",
    )
    parser.add_argument(
        "--wave-f-snapshot-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "source_data",
    )
    args = parser.parse_args()

    commits = {name: git(args.sources / name, "rev-parse", "HEAD") for name in REPOSITORIES}
    tasks = [
        *build_pathobench(args.sources / "pathobench_hf", commits["pathobench_hf"]),
        *build_eva(args.sources / "eva", commits["eva"]),
        *build_thunder(args.sources / "thunder", commits["thunder"]),
        *build_hest(commits["hest"]),
        *build_pathorob(commits["pathorob"]),
    ]
    materialize_task_contracts(tasks)
    tasks.extend(build_pathorob_nature_protocols(args.pathorob_nature_snapshot, tasks))
    tasks.extend(build_exaone_pathobench_protocols(args.exaone_pathobench_snapshot, tasks))
    tasks.extend(build_threads_pathobench_protocols(args.threads_pathobench_snapshot, tasks))
    tasks.extend(build_eva_leaderboard_protocols(args.sources / "eva", commits["eva"], tasks))
    tasks.extend(build_h0mini_uni2h_protocols(args.h0mini_uni2h_snapshot, tasks))
    tasks.extend(build_group_c_protocols(args.group_c_official_snapshot))
    tasks.extend(build_wave_d_hoptimus_protocols(args.wave_d_hoptimus_snapshot, tasks))
    tasks.extend(build_wave_d_virchow2_protocols(args.wave_d_virchow2_snapshot, tasks))
    tasks.extend(build_wave_d_virchow_protocols(args.wave_d_virchow_snapshot))
    tasks.extend(build_wave_d_uni_protocols(args.wave_d_uni_snapshot, tasks))
    group_b_paths = [args.group_b_snapshot_dir / name for name in (
        "genbio_pathfm_official_2026.csv", "midnight_miccai2025_official_scores.csv",
        "openmidnight_technical_report_2025.csv",
    )]
    wave_e_paths = [args.wave_e_snapshot_dir / name for name in (
        "conch_official_scores_2024.csv", "conch15_titan_official_scores_2025.csv",
        "phikon_family_official_scores_2023_2024.csv", "ctranspath_official_evidence_2022_2024.csv",
    )]
    wave_f_paths = [args.wave_f_snapshot_dir / name for name in (
        "hibou_official_scores_2024.csv", "musk_official_scores_2025.csv",
        "gpfm_official_scores_2025.csv",
    )]
    tasks.extend(build_group_b_protocols(group_b_paths, tasks))
    tasks.extend(build_wave_e_protocols(wave_e_paths, tasks, args.group_c_official_snapshot))
    tasks.extend(build_wave_f_protocols(wave_f_paths, tasks))
    expected = {
        "pathobench": 217, "eva": 60, "thunder": 33, "hest": 45,
        "pathorob": 26, "plism": 10, "uni2_benchmark": 6,
        "virchow2g_paper": 36, "titan_paper": 333,
        "hoptimus1_report": 10, "virchow2_paper": 18, "virchow_paper": 15,
        "uni_paper": 227,
        "conch_primary": 77, "titan_patch_encoder": 297,
        "phikon_primary": 25, "phikon_v2_external": 18,
        "hibou_primary": 18, "musk_primary": 68, "gpfm_primary": 99,
    }
    actual = {suite: sum(row["suite_id"] == suite for row in tasks) for suite in expected}
    if actual != expected:
        raise ValueError(f"task inventory mismatch: {actual}")

    suite_protocols = {
        "pathobench": "95 public WSI task identities plus 80 EXAONE-2025 and 42 THREADS-2025 explicitly versioned result protocols.",
        "eva": "Canonical offline/leaderboard protocols plus exact Midnight CLS+Mean and OpenMidnight CLS-only report variants.",
        "thunder": "Code-backed dataset configs plus exact GenBio-PathFM frozen kNN/F1 report variants.",
        "hest": "PCA-256 plus ridge regression morphology-to-gene-expression benchmark; Pearson r.",
        "pathorob": "Generic/Nature protocols plus GenBio-PathFM dataset-specific balanced-accuracy and raw-only APD-ID/APD-OOD variants.",
        "plism": "Six H0-mini-paper context-specific protocols plus four refreshed official-repository leaderboard protocols; versions remain distinct.",
        "uni2_benchmark": "Six public leaf benchmark cells from the official UNI repository; raw-registry-only because the source does not name endpoint-specific metrics.",
        "virchow2g_paper": "Official Virchow2-family Tables 2-3; Virchow2G, Virchow2 and Virchow comparator cells preserve CLS/CLS+Mean linear probes and random-forest HEST.",
        "titan_paper": "Official TITAN/TITAN-V public morphology, grading, molecular, survival, full/few/zero-shot, slide-retrieval, and cross-modal retrieval protocols remain separate.",
        "hoptimus1_report": "First-party Bioptimus frozen-feature report; six clearly public slide cohorts and four public tile tasks, with nonpublic/access-ambiguous cohorts quarantined.",
        "virchow2_paper": "Virchow2 primary-paper public tile linear probes and HCC spatial-transcriptomics protocols; CLS-only and CLS+Mean remain distinct.",
        "virchow_paper": "Virchow primary-paper public frozen CLS+Mean tile linear probes; accuracy, balanced accuracy, and weighted F1 remain distinct protocols.",
        "uni_paper": "Exhaustive UNI primary-paper public frozen-feature leaves: patch linear/KNN/retrieval, slide ABMIL, and MI-SimpleShot; resolution, stain-normalization, cohort, K, and metric variants remain distinct.",
        "conch_primary": "CONCH v1 public primary-paper zero-shot, frozen supervised, retrieval, and segmentation leaves; prompt/probe variants remain distinct.",
        "titan_patch_encoder": "CONCHv1.5 public comparator leaves from TITAN tables; logistic, SimpleShot, kNN, ABMIL, few-shot, and retrieval protocols remain distinct.",
        "phikon_primary": "Phikon public primary-paper frozen tile and ABMIL leaves.",
        "phikon_v2_external": "Phikon and Phikon-v2 public external-cohort comparison leaves; component versions remain distinct.",
        "hibou_primary": "Hibou-B and Hibou-L public frozen tile linear-probe and slide attention-pooling leaves; checkpoint families remain distinct.",
        "musk_primary": "MUSK public zero-shot retrieval/classification, 10-shot, linear-probe, biomarker, and prognosis leaves; task-fine-tuned, controlled, and aggregate rows are excluded.",
        "gpfm_primary": "GPFM public frozen-feature WSI classification, survival, ROI classification, and nearest-neighbour retrieval leaves; metrics and protocols remain distinct.",
    }
    suite_names = {"pathobench": "Patho-Bench", "eva": "EVA", "thunder": "THUNDER", "hest": "HEST-Benchmark", "pathorob": "PathoROB", "plism": "PLISM robustness benchmark", "uni2_benchmark": "Official UNI 2 benchmark report", "virchow2g_paper": "Virchow2G primary-paper benchmarks", "titan_paper": "TITAN primary-paper benchmarks", "hoptimus1_report": "H-optimus-1 official benchmark report", "virchow2_paper": "Virchow2 primary-paper benchmarks", "virchow_paper": "Virchow primary-paper benchmarks", "uni_paper": "UNI primary-paper benchmarks", "conch_primary": "CONCH primary-paper benchmarks", "titan_patch_encoder": "CONCHv1.5 TITAN-paper benchmarks", "phikon_primary": "Phikon primary-paper benchmarks", "phikon_v2_external": "Phikon-v2 external-cohort benchmarks", "hibou_primary": "Hibou primary-paper benchmarks", "musk_primary": "MUSK primary-paper benchmarks", "gpfm_primary": "GPFM primary-paper benchmarks"}
    suite_references = {
        "pathobench": REPOSITORIES["pathobench_hf"] + f"/tree/{commits['pathobench_hf']}",
        "eva": REPOSITORIES["eva"] + f"/tree/{commits['eva']}",
        "thunder": REPOSITORIES["thunder"] + f"/tree/{commits['thunder']}",
        "hest": REPOSITORIES["hest"] + f"/tree/{commits['hest']}",
        "pathorob": REPOSITORIES["pathorob"] + f"/tree/{commits['pathorob']}",
        "plism": "https://github.com/owkin/plism-benchmark/tree/5ec9511893af993f6faa099f093d1924b291aed2",
        "uni2_benchmark": "https://github.com/mahmoodlab/UNI/tree/42715efc11722a496e0a67f3369505a8f277206c",
        "virchow2g_paper": "https://arxiv.org/abs/2408.00738",
        "titan_paper": "https://www.nature.com/articles/s41591-025-03982-3",
        "hoptimus1_report": "https://www.bioptimus.com/news/bioptimus-launches-h-optimus-1",
        "virchow2_paper": "https://arxiv.org/abs/2408.00738",
        "virchow_paper": "https://arxiv.org/abs/2309.07778",
        "uni_paper": "https://arxiv.org/abs/2308.15474",
        "conch_primary": "https://www.nature.com/articles/s41591-024-02856-4",
        "titan_patch_encoder": "https://www.nature.com/articles/s41591-025-03982-3",
        "phikon_primary": "https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v3",
        "phikon_v2_external": "https://arxiv.org/abs/2409.09173v1",
        "hibou_primary": "https://arxiv.org/abs/2406.05074",
        "musk_primary": "https://www.nature.com/articles/s41586-024-08378-w",
        "gpfm_primary": "https://www.nature.com/articles/s41551-025-01488-4",
    }
    suites = []
    for suite, count in expected.items():
        suites.append(
            {
                "suite_id": suite,
                "name": suite_names[suite],
                "scope": "pathology foundation-model evaluation",
                "task_count": count,
                "reference_url": suite_references[suite],
                "protocol": suite_protocols[suite],
                "audit_notes": "Inventory generated from a pinned upstream source snapshot; see provenance.json.",
            }
        )

    hest_scores, hest_aliases = parse_hest_scores(args.sources / "hest", commits["hest"])
    thunder_scores, thunder_aliases = parse_thunder_scores(args.sources / "thunder", commits["thunder"])
    pathorob_scores, pathorob_aliases = parse_pathorob_scores(args.sources / "pathorob", commits["pathorob"])
    pathorob_nature_scores, pathorob_nature_aliases = parse_pathorob_nature_scores(
        args.pathorob_nature_snapshot, tasks
    )
    pathobench_scores, pathobench_aliases = parse_exaone_pathobench_scores(
        args.exaone_pathobench_snapshot, tasks
    )
    threads_scores, threads_aliases = parse_threads_pathobench_scores(
        args.threads_pathobench_snapshot, tasks
    )
    eva_repository_scores = parse_eva_repository_scores(
        args.sources / "eva", commits["eva"]
    )
    eva_midnight_scores = parse_midnight_scores(
        args.sources / "eva_midnight", commits["eva_midnight"]
    )
    eva_selected, eva_conflicts = merge_eva_scores(
        eva_repository_scores, eva_midnight_scores
    )
    eva_scores = [row.registry_row("2026-08-05") for row in eva_selected]
    h0mini_uni2h_scores, h0mini_uni2h_aliases = build_h0mini_uni2h_scores(
        args.h0mini_uni2h_snapshot
    )
    group_c_scores, group_c_aliases = build_group_c_scores(args.group_c_official_snapshot)
    wave_d_hoptimus_scores, wave_d_hoptimus_aliases = build_wave_d_hoptimus_scores(args.wave_d_hoptimus_snapshot)
    wave_d_virchow2_scores, wave_d_virchow2_aliases = build_wave_d_virchow2_scores(args.wave_d_virchow2_snapshot)
    wave_d_virchow_scores, wave_d_virchow_aliases = build_wave_d_virchow_scores(args.wave_d_virchow_snapshot)
    wave_d_uni_scores, wave_d_uni_aliases = build_wave_d_uni_scores(args.wave_d_uni_snapshot)
    group_b_scores, group_b_aliases = build_group_b_scores(group_b_paths)
    wave_e_scores, wave_e_aliases = build_wave_e_scores(wave_e_paths)
    wave_f_scores, wave_f_aliases = build_wave_f_scores(wave_f_paths)
    scores = (
        pathobench_scores + threads_scores + eva_scores
        + hest_scores + thunder_scores + pathorob_scores + pathorob_nature_scores
        + h0mini_uni2h_scores
        + group_c_scores
        + wave_d_hoptimus_scores + wave_d_virchow2_scores + wave_d_virchow_scores
        + wave_d_uni_scores
        + group_b_scores + wave_e_scores + wave_f_scores
    )
    for row in scores:
        allowed_blank = (
            {"normalized_score"}
            if row["audit_status"] == "parsed_primary_source_analysis_ineligible"
            else set()
        )
        missing = [
            field for field in SCORE_FIELDS
            if row.get(field) in (None, "") and field not in allowed_blank
        ]
        if missing:
            raise ValueError(f"score evidence row has blank contract fields {missing}: {row}")
        if row["audit_status"] == "parsed_primary_source_analysis_ineligible" and row["normalized_score"] != "":
            raise ValueError("analysis-ineligible score must not carry an invented normalization")
    eva_aliases = [
        alias_row(
            row.reported_model_alias,
            row.model_id,
            "eva",
            row.reference_url,
            "Reported EVA leaderboard alias mapped by the source-specific exact alias table.",
        )
        for row in eva_selected
    ]
    aliases = (
        pathobench_aliases + threads_aliases + eva_aliases
        + hest_aliases + thunder_aliases + pathorob_aliases + pathorob_nature_aliases
        + h0mini_uni2h_aliases
        + group_c_aliases
        + wave_d_hoptimus_aliases + wave_d_virchow2_aliases + wave_d_virchow_aliases
        + wave_d_uni_aliases
        + group_b_aliases + wave_e_aliases + wave_f_aliases
    )
    aliases = sorted(
        {(row["suite_id"], row["alias"]): row for row in aliases}.values(),
        key=lambda row: (str(row["model_id"]), str(row["suite_id"]), str(row["alias"])),
    )
    dedup = build_dedup(tasks)

    write_csv(args.output / "suites.csv", SUITE_FIELDS, suites)
    write_csv(args.output / "tasks.csv", TASK_FIELDS, tasks)
    write_csv(args.output / "model_aliases.csv", ALIAS_FIELDS, aliases)
    write_csv(args.output / "scores.csv", SCORE_FIELDS, scores)
    write_csv(args.output / "deduplication.csv", DEDUP_FIELDS, dedup)
    write_csv(
        args.output / "eva_source_conflicts.csv",
        EVA_CONFLICT_FIELDS,
        eva_conflicts,
    )

    provenance = {
        "schema_version": 2,
        "generator": "scripts/build_registry.py",
        "normalization": {
            "macro-ovr-auc/bacc/cindex/balanced_accuracy/dice": "Raw metrics in [0,1] multiplied by 100.",
            "weighted_kappa": "(kappa + 1) * 50, preserving the metric's mathematical [-1,1] domain on a 0-100 scale.",
            "f1": "Scores already reported on 0-100 scale; preserved.",
            "robustness_index": "Raw RI in [0,1] multiplied by 100.",
            "pearson_r": "(r + 1) * 50; logit(normalized/100) equals 2 * atanh(r), a scaled Fisher-z.",
            "clustering_score": "PathoROB declares an approximately [-1,1] domain; canonical Nature-2026 means use (score + 1) * 50.",
            "average_performance_drop_percent": "No common-scale normalization. Raw signed percent is preserved exactly with normalized_score blank and audit_status=parsed_primary_source_analysis_ineligible. Zero is the no-drop reference; the paper states that increasingly negative values are worse and describes higher/closer-to-zero as better, but defines no bounded domain.",
            "cosine_similarity/top10_accuracy": "Raw PLISM metrics in [0,1] multiplied by 100.",
            "accuracy/balanced_accuracy/weighted_f1/auroc/top-k/KNN/retrieval accuracy": "Raw metrics in [0,1] multiplied by 100.",
            "*_percent/f1_score": "Source-reported 0-100 percentages are preserved; the metric suffix records the source unit.",
            "cohen_kappa/quadratic_weighted_kappa": "Primary-paper fractions multiplied by 100, matching the TITAN/CONCH reporting contract; legacy weighted_kappa retains its separate affine contract.",
            "c_index/recall/precision/mean_recall/recall_at_k": "Raw metrics in [0,1] multiplied by 100.",
            "reported_performance_score": "No normalization: the official UNI repository does not name endpoint-specific metrics. Public leaf values remain raw-registry-only with audit_status=parsed_primary_source_analysis_ineligible.",
        },
        "repositories": {
            name: {
                "url": url,
                "commit": commits[name],
            }
            for name, url in REPOSITORIES.items()
        },
        "source_reports": {
            "exaone_path_2_5_pathobench": {
                "url": "https://arxiv.org/abs/2512.14019v1",
                "pdf_url": "https://arxiv.org/pdf/2512.14019v1",
                "source_archive_url": "https://export.arxiv.org/e-print/2512.14019v1",
                "source_archive_sha256": "0c479164dfab7ac48a1e1876649ef73efe9f457e064c3ab00ee960856d35a268",
                "source_member": "tabs/pathobench_result.tex",
                "snapshot_path": portable_artifact_path(args.exaone_pathobench_snapshot),
                "snapshot_sha256": hashlib.sha256(
                    args.exaone_pathobench_snapshot.read_bytes()
                ).hexdigest(),
                "table": 4,
                "reported_tasks": 80,
                "reported_models": 7,
                "score_cells": 560,
            },
            "threads_pathobench_public": {
                "url": "https://arxiv.org/abs/2501.16652v1",
                "html_url": "https://arxiv.org/html/2501.16652v1",
                "html_sha256": "a6c7af63c1f527eba692f83b362651e0e1d96d07e303520f90cd08f34b00c92f",
                "source_archive_url": "https://export.arxiv.org/e-print/2501.16652v1",
                "source_archive_sha256": "3d8b3f6779b9b0eae21be12e8917bd6f0bab26e3c7943470e378383d20a1de4f",
                "snapshot_path": portable_artifact_path(args.threads_pathobench_snapshot),
                "snapshot_sha256": hashlib.sha256(args.threads_pathobench_snapshot.read_bytes()).hexdigest(),
                "reported_public_tasks": 42,
                "reported_frozen_representations": 8,
                "score_cells": 336,
                "excluded_internal_tasks": 12,
                "excluded_internal_cells": 96,
            },
            "eva_pathology_leaderboards": {
                "repository_score_cells": len(eva_repository_scores),
                "midnight_model_card_score_cells": len(eva_midnight_scores),
                "selected_unique_cells": len(eva_scores),
                "alternate_source_conflicts": len(eva_conflicts),
                "conflict_audit_path": portable_artifact_path(
                    args.output / "eva_source_conflicts.csv"
                ),
            },
            "pathorob_nature2026": {
                "url": "https://www.nature.com/articles/s41467-026-73923-2",
                "pmcid": "PMC13260997",
                "source_data_url": "https://pmc.ncbi.nlm.nih.gov/articles/instance/13260997/bin/41467_2026_73923_MOESM4_ESM.xlsx",
                "source_data_sha256": "07456f3ffc5270ea1d8d48a8f82c08a5be396c88f99cc0227968dad721943047",
                "snapshot_path": portable_artifact_path(args.pathorob_nature_snapshot),
                "snapshot_sha256": hashlib.sha256(args.pathorob_nature_snapshot.read_bytes()).hexdigest(),
                "published_models": 20,
                "published_apd_endpoints": 2,
                "published_apd_cells": 40,
                "published_clustering_protocols": 3,
                "published_clustering_cells": 60,
                "repository_example_cells_quarantined": 22,
            },
            "h0mini_uni2h_official": {
                "paper_url": "https://arxiv.org/abs/2501.16239v3",
                "paper_source_archive_url": "https://export.arxiv.org/e-print/2501.16239v3",
                "paper_source_archive_sha256": "222798059c15b554528d61f8caa04de8fcc2d5cc23997607dc25d851282a6f08",
                "plism_repository_url": "https://github.com/owkin/plism-benchmark/tree/5ec9511893af993f6faa099f093d1924b291aed2",
                "uni_repository_url": "https://github.com/mahmoodlab/UNI/tree/42715efc11722a496e0a67f3369505a8f277206c",
                "uni2h_model_card_revision": "d517a8dd47902dd7c308b3c36f63bce47e7b9a43",
                "h0mini_model_card_revision": "5b5cc0505d19ae558270045eb0df8c34df4d9609",
                "snapshot_path": portable_artifact_path(args.h0mini_uni2h_snapshot),
                "snapshot_sha256": hashlib.sha256(args.h0mini_uni2h_snapshot.read_bytes()).hexdigest(),
                "snapshot_cells": len(load_h0mini_uni2h_evidence(args.h0mini_uni2h_snapshot)),
                "public_leaf_cells": len(h0mini_uni2h_scores),
                "derived_aggregate_cells_excluded": 7,
                "private_or_internal_cells_quarantined": 19,
            },
            "wave_d_hoptimus1_official": {
                "url": "https://www.bioptimus.com/news/bioptimus-launches-h-optimus-1",
                "html_sha256": "a46ad75f2194ac2700eedfde6cef0edde5b56ff59438fa724cf53e4adc997bf3",
                "hoptimus1_model_card_revision": "43e14486f058782912d63d22ce27e2f1252eaf9e55ae8e2fd8ed416444dbc45e",
                "hoptimus0_model_card_revision": "a9110f5bb0de4a0d82796e0baee3658834245819a3e8651c38abf53c49d4077c",
                "snapshot_path": portable_artifact_path(args.wave_d_hoptimus_snapshot),
                "snapshot_sha256": hashlib.sha256(args.wave_d_hoptimus_snapshot.read_bytes()).hexdigest(),
                "snapshot_cells": 170,
                "public_leaf_cells": 95,
                "private_internal_or_access_ambiguous_cells_quarantined": 60,
                "derived_aggregate_cells_excluded": 15,
            },
            "wave_d_virchow2_official": {
                "url": "https://arxiv.org/abs/2408.00738",
                "source_archive_url": "https://export.arxiv.org/e-print/2408.00738",
                "source_archive_sha256": "1ee7317aa683b5c948f0c9b6e3fc588fd3a95c7f0a3db39d71511268db1f3fda",
                "snapshot_path": portable_artifact_path(args.wave_d_virchow2_snapshot),
                "snapshot_sha256": hashlib.sha256(args.wave_d_virchow2_snapshot.read_bytes()).hexdigest(),
                "snapshot_cells": 192,
                "public_leaf_cells": 144,
                "internal_cells_quarantined": 24,
                "derived_aggregate_cells_excluded": 24,
            },
            "wave_d_virchow_official": {
                "url": "https://arxiv.org/abs/2309.07778",
                "source_archive_url": "https://export.arxiv.org/e-print/2309.07778",
                "source_archive_sha256": "b2c31918a05d22534f38e35c5fc2cf3704439f7eb30af637dfd1e698373ba6b0",
                "snapshot_path": portable_artifact_path(args.wave_d_virchow_snapshot),
                "snapshot_sha256": hashlib.sha256(args.wave_d_virchow_snapshot.read_bytes()).hexdigest(),
                "snapshot_cells": 21,
                "public_leaf_cells": 15,
                "internal_or_nonpublic_cells_quarantined": 6,
            },
            "wave_d_uni_official": {
                "url": "https://arxiv.org/abs/2308.15474",
                "source_archive_url": "https://export.arxiv.org/e-print/2308.15474",
                "source_archive_sha256": "26da25ced22b205570f480c3a562af31ed74bdaf679455f30cbcaa5dfdad4e60",
                "snapshot_path": portable_artifact_path(args.wave_d_uni_snapshot),
                "snapshot_sha256": hashlib.sha256(args.wave_d_uni_snapshot.read_bytes()).hexdigest(),
                "active_uni_result_rows": 83,
                "expanded_metric_leaf_cells": 277,
                "public_frozen_feature_leaf_cells": 227,
                "finetuned_cells_quarantined": 24,
                "internal_cells_quarantined": 23,
                "derived_aggregate_cells_excluded": 3,
            },
            "group_b_official": {
                "snapshot_paths": [portable_artifact_path(path) for path in group_b_paths],
                "snapshot_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in group_b_paths},
                "snapshot_cells": 107,
                "public_analysis_eligible_cells": 51,
                "public_analysis_ineligible_apd_cells": 6,
                "duplicate_alternate_cells_excluded": 12,
                "fine_tuned_aggregate_or_conflict_cells_excluded": 38,
            },
            "wave_e_official": {
                "snapshot_paths": [portable_artifact_path(path) for path in wave_e_paths],
                "snapshot_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in wave_e_paths},
                "public_primary_leaf_cells": 417,
                "private_internal_cells_excluded": 144,
                "fine_tuned_cells_excluded": 23,
                "secondary_only_ctranspath_cells_excluded": 50,
            },
            "wave_f_official": {
                "snapshot_paths": [portable_artifact_path(path) for path in wave_f_paths],
                "snapshot_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in wave_f_paths},
                "snapshot_cells": 397,
                "public_primary_leaf_cells": 185,
                "private_or_controlled_cells_excluded": 79,
                "fine_tuned_cells_excluded": 72,
                "aggregate_or_ablation_cells_excluded": 61,
            },
        },
        "counts": {
            "suites": len(suites),
            "tasks": len(tasks),
            "tasks_by_suite": actual,
            "dataset_artifacts": len({row["dataset_artifact_id"] for row in tasks}),
            "task_identities": len({row["task_identity_id"] for row in tasks}),
            "protocols": len({row["protocol_id"] for row in tasks}),
            "pathobench_sample_units": {unit: sum(row["suite_id"] == "pathobench" and row["sample_unit"] == unit for row in tasks) for unit in ("case", "slide")},
            "pathobench_task_types": {kind: sum(row["suite_id"] == "pathobench" and row["task_type"] == kind for row in tasks) for kind in ("classification", "survival")},
            "pathobench_metrics": {metric: sum(row["suite_id"] == "pathobench" and row["metric"] == metric for row in tasks) for metric in ("macro-ovr-auc", "bacc", "weighted_kappa", "cindex")},
            "pathobench_task_identities": len({row["task_identity_id"] for row in tasks if row["suite_id"] == "pathobench"}),
            "pathobench_exaone_protocol_variants": sum(str(row["evaluation_id"]).startswith("pathobench.exaone2025.") for row in tasks),
            "pathobench_threads_protocol_variants": sum(str(row["evaluation_id"]).startswith("pathobench.threads2025.") for row in tasks),
            "eva_leaderboard_protocol_variants": sum(str(row["evaluation_id"]).startswith("eva.leaderboard.") for row in tasks),
            "model_aliases": len(aliases),
            "scores": len(scores),
            "scores_by_suite": {suite: sum(row["suite_id"] == suite for row in scores) for suite in expected},
            "scores_by_audit_status": {
                status: sum(row["audit_status"] == status for row in scores)
                for status in sorted({str(row["audit_status"]) for row in scores})
            },
            "scores_by_review_status": {"machine_parsed_single_source": sum(row["review_status"] == "machine_parsed_single_source" for row in scores)},
            "deduplication_memberships": len(dedup),
            "deduplication_groups": len({row["group_id"] for row in dedup}),
            "exact_duplicate_task_identities": len({row["task_identity_id"] for row in dedup if row["match_type"] == "exact"}),
            "semantic_adjudication_groups": len({row["group_id"] for row in dedup if row["match_type"] == "semantic"}),
        },
        "audit_notes": [
            "No network calls are made by the generator.",
            "EXAONE Path 2.5 Table 4 supplies 560 point estimates: seven models across 80 exact official Patho-Bench task names under the paper's shared linear-probing settings. The reported AVERAGE row is excluded because it is derived, not an independent evaluation cell.",
            "Fifteen current Patho-Bench HF tasks are absent from EXAONE Table 4 and remain unscored; the parser requires exact task-name matches and does not impute or fuzzy-map them.",
            "THREADS supplies 336 public Patho-Bench-v1 cells across 42 tasks and eight frozen representation pipelines. Twelve internal tasks (96 cells), supervised/fine-tuned rows, and aggregate summaries are excluded.",
            "EVA contributes 265 selected unique leaderboard cells. The current repository wins over the older Midnight model-card report for 110 overlapping cells; every alternate value remains in eva_source_conflicts.csv and is never averaged.",
            "HEST and THUNDER average columns are not independent evaluation cells and are excluded from scores.csv.",
            "Score rows are machine-parsed from one reporting source each. audit_status=parsed_primary_source is prototype evidence, not independent or dual-source verification.",
            "PathoROB external-publication rows are retained with audit_status=reported_external; PathoROB explicitly says those values were not validated by its authors.",
            "PathoROB Nature Source Data contributes 100 canonical primary-source means across 20 models: 40 aggregate APD-ID/APD-OOD cells and 60 clustering cells over three explicit dataset protocols. The 22 pinned-repository example results are preserved only in the evidence snapshot because their two-model values differ slightly from the paper.",
            "Canonical PathoROB clustering scores enter the factor matrix after the source-declared approximately [-1,1] affine normalization. Canonical APD values remain in the raw registry with exact signed percent values but no normalized_score; they are analysis-ineligible because neither the source nor project policy defines a non-arbitrary bounded normalization.",
            "H0-mini arXiv v3 contributes 46 public leaf cells (23 each for H0-mini and UNI2-h): 16 EVA, 18 HEST, and 12 PLISM. Seven paper/repository aggregate cells are excluded and 19 private/internal cells are quarantined in the evidence snapshot.",
            "The pinned PLISM repository contributes eight refreshed public leaf cells under four repository-version protocols. They are never merged with the earlier H0-mini paper values; three top-10 protocols share task identities only.",
            "The official UNI repository contributes six public UNI2-h leaf cells. They remain raw-registry-only because its benchmark table does not state the endpoint-specific metric; no metric or bounded normalization is inferred.",
            "Exact dedup groups are derived from duplicate task_identity_id values. Two broader semantic-overlap groups remain manually adjudicated and keep separate task identities.",
            "evaluation_id is retained as a compatibility alias equal to protocol_id; protocol-specific scores are never overwritten during deduplication.",
        ],
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance["counts"], indent=2))


if __name__ == "__main__":
    main()
