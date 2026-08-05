#!/usr/bin/env python3
"""Build PathoPress's citation-backed benchmark registry from pinned clones.

This script deliberately uses only the Python standard library.  The source
repositories are expected to exist below ``--sources``; their exact commits
are recorded in ``provenance.json`` so every generated row can be audited.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable


REPOSITORIES = {
    "benchpress": "https://github.com/microsoft/benchpress",
    "pathobench": "https://github.com/mahmoodlab/Patho-Bench",
    "pathobench_hf": "https://huggingface.co/datasets/MahmoodLab/Patho-Bench",
    "eva": "https://github.com/kaiko-ai/eva",
    "thunder": "https://github.com/MICS-Lab/thunder",
    "hest": "https://github.com/mahmoodlab/HEST",
    "pathorob": "https://github.com/bifold-pathomics/PathoROB",
}

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
}


def canonical_model(alias: str) -> str:
    key = slug(alias).replace("conch-1-5", "conch-v1-5")
    return MODEL_IDS.get(key, key)


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
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
                    "direction": "higher" if metric != "average_performance_drop" else "lower",
                    "protocol": "PathoROB center-robustness protocol across TCGA 2x2, TCGA 4x4, Camelyon, and Tolkach ESCA benchmark datasets.",
                    "reference_url": ref,
                    "audit_status": "parsed_primary_source",
                    "audit_notes": "TCGA has distinct 2x2 and 4x4 robustness endpoints; current RI leaderboard reports TCGA 2x2, Camelyon, and Tolkach ESCA only, so TCGA 4x4 remains an unscored catalog endpoint.",
                }
            )
    if len(rows) != 12:
        raise AssertionError("PathoROB endpoint construction drifted")
    return rows


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
    normalized: float,
    suite: str,
    metric: str,
    ref: str,
    source_locator: str,
    lineage: str,
    audit: str = "parsed_primary_source",
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "reported_model_alias": reported_alias,
        "model_revision": "not_reported",
        "evaluation_id": evaluation_id,
        "value": f"{value:.6g}",
        "normalized_score": f"{normalized:.6g}",
        "suite_id": suite,
        "metric": metric,
        "reference_url": ref,
        "source_locator": source_locator,
        "extraction_date": "2026-08-05",
        "review_status": "machine_parsed_single_source",
        "uncertainty": "not_reported",
        "lineage": lineage,
        "audit_status": audit,
    }


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


def alias_row(alias: str, model_id: str, suite: str, ref: str, notes: str) -> dict[str, object]:
    return {"alias": alias, "model_id": model_id, "suite_id": suite, "reference_url": ref, "audit_notes": notes}


def build_dedup(tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    ids = {str(row["evaluation_id"]) for row in tasks}
    identity_members: dict[str, list[str]] = {}
    for task in tasks:
        identity_members.setdefault(str(task["task_identity_id"]), []).append(str(task["evaluation_id"]))
    duplicates = {identity: sorted(members) for identity, members in identity_members.items() if len(members) > 1}
    if set(duplicates) != set(EXACT_TASK_IDENTITIES) or len(duplicates) != 7:
        raise ValueError(f"expected seven exact duplicate task identities, found {duplicates}")

    rows = []
    for identity, members in sorted(duplicates.items()):
        canonical = members[0]
        for member in members:
            rows.append(
                {
                    "group_id": "exact." + identity.removeprefix("task."),
                    "match_type": "exact",
                    "task_identity_id": identity,
                    "canonical_evaluation_id": canonical,
                    "member_evaluation_id": member,
                    "decision": "link_only",
                    "rationale": "Duplicate task_identity_id derived from the materialized task contract; retain distinct protocol_id values and do not overwrite protocol-specific scores.",
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
    expected = {"pathobench": 95, "eva": 13, "thunder": 21, "hest": 9, "pathorob": 12}
    actual = {suite: sum(row["suite_id"] == suite for row in tasks) for suite in expected}
    if actual != expected:
        raise ValueError(f"task inventory mismatch: {actual}")

    suite_protocols = {
        "pathobench": "Public WSI task splits spanning seven clinical task families; task-specific downstream heads.",
        "eva": "Canonical offline pathology configs only; explicit reduced-data protocol variants excluded.",
        "thunder": "All code-backed dataset configs represented by a primary linear-probe or segmentation endpoint.",
        "hest": "PCA-256 plus ridge regression morphology-to-gene-expression benchmark; Pearson r.",
        "pathorob": "Four dataset endpoints (TCGA 2x2, TCGA 4x4, Camelyon, Tolkach ESCA) crossed with three robustness metrics.",
    }
    suite_names = {"pathobench": "Patho-Bench", "eva": "EVA", "thunder": "THUNDER", "hest": "HEST-Benchmark", "pathorob": "PathoROB"}
    suite_repo = {"pathobench": "pathobench_hf", "eva": "eva", "thunder": "thunder", "hest": "hest", "pathorob": "pathorob"}
    suites = []
    for suite, count in expected.items():
        repo_key = suite_repo[suite]
        suites.append(
            {
                "suite_id": suite,
                "name": suite_names[suite],
                "scope": "pathology foundation-model evaluation",
                "task_count": count,
                "reference_url": REPOSITORIES[repo_key] + (f"/tree/{commits[repo_key]}"),
                "protocol": suite_protocols[suite],
                "audit_notes": "Inventory generated from a pinned upstream source snapshot; see provenance.json.",
            }
        )

    hest_scores, hest_aliases = parse_hest_scores(args.sources / "hest", commits["hest"])
    thunder_scores, thunder_aliases = parse_thunder_scores(args.sources / "thunder", commits["thunder"])
    pathorob_scores, pathorob_aliases = parse_pathorob_scores(args.sources / "pathorob", commits["pathorob"])
    scores = hest_scores + thunder_scores + pathorob_scores
    for row in scores:
        missing = [field for field in SCORE_FIELDS if row.get(field) in (None, "")]
        if missing:
            raise ValueError(f"score evidence row has blank contract fields {missing}: {row}")
    aliases = hest_aliases + thunder_aliases + pathorob_aliases
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

    provenance = {
        "schema_version": 2,
        "generator": "scripts/build_registry.py",
        "normalization": {
            "f1": "Scores already reported on 0-100 scale; preserved.",
            "robustness_index": "Raw RI in [0,1] multiplied by 100.",
            "pearson_r": "(r + 1) * 50; logit(normalized/100) equals 2 * atanh(r), a scaled Fisher-z.",
        },
        "repositories": {
            name: {
                "url": url,
                "commit": commits[name],
                "local_path": str((args.sources / name).resolve()),
            }
            for name, url in REPOSITORIES.items()
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
            "model_aliases": len(aliases),
            "scores": len(scores),
            "scores_by_suite": {suite: sum(row["suite_id"] == suite for row in scores) for suite in ("hest", "thunder", "pathorob")},
            "scores_by_audit_status": {status: sum(row["audit_status"] == status for row in scores) for status in ("parsed_primary_source", "reported_external")},
            "scores_by_review_status": {"machine_parsed_single_source": sum(row["review_status"] == "machine_parsed_single_source" for row in scores)},
            "deduplication_memberships": len(dedup),
            "deduplication_groups": len({row["group_id"] for row in dedup}),
            "exact_duplicate_task_identities": len({row["task_identity_id"] for row in dedup if row["match_type"] == "exact"}),
            "semantic_adjudication_groups": len({row["group_id"] for row in dedup if row["match_type"] == "semantic"}),
        },
        "audit_notes": [
            "No network calls are made by the generator.",
            "HEST and THUNDER average columns are not independent evaluation cells and are excluded from scores.csv.",
            "Score rows are machine-parsed from one reporting source each. audit_status=parsed_primary_source is prototype evidence, not independent or dual-source verification.",
            "PathoROB external-publication rows are retained with audit_status=reported_external; PathoROB explicitly says those values were not validated by its authors.",
            "Exact dedup groups are derived from duplicate task_identity_id values. Two broader semantic-overlap groups remain manually adjudicated and keep separate task identities.",
            "evaluation_id is retained as a compatibility alias equal to protocol_id; protocol-specific scores are never overwritten during deduplication.",
        ],
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance["counts"], indent=2))


if __name__ == "__main__":
    main()
