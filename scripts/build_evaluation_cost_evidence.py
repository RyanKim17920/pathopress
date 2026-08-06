#!/usr/bin/env python3
"""Build a source-backed pathology evaluation cost-evidence registry.

The registry follows the evidence discipline documented by BenchPress at
commit 0a684b63ee0e4a401cb907a3827a82ea997d74c4: raw source numbers are
preserved, configuration limits are not mislabeled as observed use, and
qualitative difficulty is never converted to numeric cost.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "0a684b63ee0e4a401cb907a3827a82ea997d74c4"
RETAINED_STATUSES = {"verified", "parsed_primary_source"}

COMMITS = {
    "eva": "e43e74a99b75660b0014f790f25a33dd9f11e121",
    "thunder": "3d1cc9513fb2cfd8c4afb0d7bb9f5c4f6b69117f",
    "hest": "3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3",
    "pathorob": "6583cf0b0d902c8cc032308262fa3a3befdc0687",
    "pathobench": "660e77044640e3d7d2f1150cc6721e97454993bf",
}


def github_url(repo: str, commit: str, path: str, anchor: str = "") -> str:
    return f"https://github.com/{repo}/blob/{commit}/{path}{anchor}"


SUITE_SOURCES = [
    {
        "source_id": "pathobench_readme_660e770",
        "source_name": "Patho-Bench README",
        "suite_id": "pathobench",
        "evidence_type": "official_repository_documentation",
        "source_url": github_url("mahmoodlab/Patho-Bench", COMMITS["pathobench"], "README.md"),
        "locators": ["README.md:16", "README.md:31-34", "README.md:62-85"],
        "raw_evidence": {
            "canonical_tasks": 95,
            "public_datasets": 33,
            "split_and_config_access": "automatically downloaded from the official Hugging Face repository",
            "input_requirement": "Trident patch embeddings are required before Patho-Bench; pooled features may be supplied",
            "execution_note": "large runs support automatic GPU load balancing",
        },
        "scope_note": "Operational benchmark-family context, not an observed run for every retained paper result.",
    },
    {
        "source_id": "pathobench_license_660e770",
        "source_name": "Patho-Bench repository license",
        "suite_id": "pathobench",
        "evidence_type": "official_repository_license",
        "source_url": github_url("mahmoodlab/Patho-Bench", COMMITS["pathobench"], "LICENSE"),
        "locators": ["LICENSE:58-64"],
        "raw_evidence": {"repository_license": "CC-BY-NC-4.0"},
        "scope_note": "Repository license only; it does not establish every underlying dataset license.",
    },
    {
        "source_id": "eva_config_e43e74a",
        "source_name": "eva pinned offline pathology configurations",
        "suite_id": "eva",
        "evidence_type": "official_executable_configuration",
        "source_url": github_url(
            "kaiko-ai/eva",
            COMMITS["eva"],
            "configs/vision/pathology/offline",
        ),
        "locators": ["trainer.init_args", "data.init_args.dataloaders"],
        "raw_evidence": {
            "execution_mode": "offline; embeddings are computed once and reused",
            "accelerator": "auto",
            "devices": 1,
            "predict_batch_size": 64,
        },
        "scope_note": "Run counts, step/epoch caps, and batch sizes are budgets/defaults, not realized runtime.",
    },
    {
        "source_id": "eva_license_e43e74a",
        "source_name": "eva repository license",
        "suite_id": "eva",
        "evidence_type": "official_repository_license",
        "source_url": github_url("kaiko-ai/eva", COMMITS["eva"], "LICENSE"),
        "locators": ["LICENSE:1-3"],
        "raw_evidence": {"repository_license": "Apache-2.0"},
        "scope_note": "Repository license only; dataset licenses are recorded separately when the task config states them.",
    },
    {
        "source_id": "thunder_configs_3d1cc95",
        "source_name": "THUNDER pinned dataset configurations",
        "suite_id": "thunder",
        "evidence_type": "official_executable_configuration",
        "source_url": github_url(
            "MICS-Lab/thunder",
            COMMITS["thunder"],
            "src/thunder/config/dataset",
        ),
        "locators": ["nb_train_samples", "nb_val_samples", "nb_test_samples", "mpp", "image_sizes"],
        "raw_evidence": {"configuration_fields_preserved_per_evaluation": True},
        "scope_note": "Split counts and mpp are raw config values; mpp is not converted to nominal magnification.",
    },
    {
        "source_id": "thunder_runtime_context_3d1cc95",
        "source_name": "THUNDER execution documentation",
        "suite_id": "thunder",
        "evidence_type": "official_repository_documentation_and_code",
        "source_url": github_url("MICS-Lab/thunder", COMMITS["thunder"], "README.md"),
        "locators": ["README.md:110-141", "src/thunder/benchmark.py:137-143"],
        "raw_evidence": {
            "leaderboard_orchestration": "generic SLURM array template",
            "device_selection": "CUDA when available, otherwise CPU",
            "segmentation_minimum_vram_gb": 32,
        },
        "scope_note": "The 32 GB requirement applies to segmentation datasets; retained THUNDER rows are linear probes, so it is not assigned to them.",
    },
    {
        "source_id": "thunder_license_3d1cc95",
        "source_name": "THUNDER repository license",
        "suite_id": "thunder",
        "evidence_type": "official_repository_license",
        "source_url": github_url("MICS-Lab/thunder", COMMITS["thunder"], "LICENSE"),
        "locators": ["LICENSE:57-63"],
        "raw_evidence": {"repository_license": "CC-BY-4.0"},
        "scope_note": "Repository license only; underlying dataset terms are not inferred.",
    },
    {
        "source_id": "hest_readme_3ddb5ea",
        "source_name": "HEST pinned README",
        "suite_id": "hest",
        "evidence_type": "official_repository_documentation",
        "source_url": github_url("mahmoodlab/HEST", COMMITS["hest"], "README.md"),
        "locators": ["README.md:14-18", "README.md:36-40", "README.md:64-65"],
        "raw_evidence": {
            "paired_spatial_transcriptomics_samples": 1276,
            "stain": "H&E",
            "whole_corpus_size_lower_bound_tb": 2,
            "access": "free access; subsets can be queried",
            "tested_gpu_software": "cucim-cu12==24.4.0 with CUDA 12.1",
        },
        "scope_note": "Corpus-level evidence. It is not a per-organ spot count or benchmark runtime.",
    },
    {
        "source_id": "hest_benchmark_config_3ddb5ea",
        "source_name": "HEST benchmark default configuration",
        "suite_id": "hest",
        "evidence_type": "official_executable_configuration",
        "source_url": github_url("mahmoodlab/HEST", COMMITS["hest"], "bench_config/bench_config.yaml"),
        "locators": ["bench_config.yaml:1-13", "src/hest/bench/benchmark.py:196-207"],
        "raw_evidence": {
            "embedding_batch_size": 128,
            "embedding_workers": 1,
            "embedding_execution": "torch inference with CUDA autocast",
        },
        "scope_note": "Defaults/configuration, not observed resource usage.",
    },
    {
        "source_id": "hest_license_3ddb5ea",
        "source_name": "HEST benchmark and data license statement",
        "suite_id": "hest",
        "evidence_type": "official_repository_license_statement",
        "source_url": github_url("mahmoodlab/HEST", COMMITS["hest"], "README.md"),
        "locators": ["README.md:18", "LICENSE.md"],
        "raw_evidence": {"benchmark_and_data_license": "CC-BY-NC-SA-4.0"},
        "scope_note": "Applies to HEST-1k, HEST-Library, and HEST-Benchmark per the source statement.",
    },
    {
        "source_id": "pathorob_readme_6583cf0",
        "source_name": "PathoROB pinned README",
        "suite_id": "pathorob",
        "evidence_type": "official_repository_documentation",
        "source_url": github_url("bifold-pathomics/PathoROB", COMMITS["pathorob"], "README.md"),
        "locators": ["README.md:64-77", "README.md:193-206"],
        "raw_evidence": {
            "feature_extraction_download_images_approx": 100000,
            "feature_extraction_download_size_approx_gb": 2,
            "access": "downloaded from the official Hugging Face collection",
        },
        "scope_note": "Approximate full-suite download evidence, not observed transfer size or runtime.",
    },
    {
        "source_id": "pathorob_configs_6583cf0",
        "source_name": "PathoROB pinned metadata and feature extraction code",
        "suite_id": "pathorob",
        "evidence_type": "official_executable_configuration_and_metadata",
        "source_url": github_url("bifold-pathomics/PathoROB", COMMITS["pathorob"], "pathorob/features/extract_features.py"),
        "locators": ["extract_features.py:35-45", "robustness_index.py:238-260", "data/metadata/*.csv"],
        "raw_evidence": {
            "feature_batch_size": 32,
            "feature_workers": 0,
            "device": "CUDA when available, otherwise CPU",
        },
        "scope_note": "Defaults and exact metadata row counts, not observed runtime.",
    },
    {
        "source_id": "pathorob_license_6583cf0",
        "source_name": "PathoROB repository license",
        "suite_id": "pathorob",
        "evidence_type": "official_repository_license",
        "source_url": github_url("bifold-pathomics/PathoROB", COMMITS["pathorob"], "LICENSE"),
        "locators": ["LICENSE:1"],
        "raw_evidence": {"repository_license": "BSD-3-Clause"},
        "scope_note": "Repository license; per-dataset redistribution licenses are recorded on each evaluation.",
    },
]


THUNDER = {
    "bach": ([218, 50, 132], 0.42, [[2048, 1536]]),
    "bracs": ([3657, 312, 570], 0.25, None),
    "break_his": ([936, 196, 339], 0.25, [[700, 460]]),
    "ccrcc": ([16797, 4298, 6074], 0.25, [[256, 256], [300, 300]]),
    "crc": ([80000, 20000, 7180], 0.5, [[224, 224]]),
    "esca": ([154438, 34704, 178187], 0.78, [[256, 256]]),
    "mhist": ([1743, 432, 977], 2.0, [[224, 224]]),
    "patch_camelyon": ([262144, 32768, 32768], 1.0, [[96, 96]]),
    "spider_breast": ([67581, 13277, 12034], 0.5, [[224, 224]]),
    "spider_colorectal": ([46892, 17097, 13193], 0.5, [[224, 224]]),
    "spider_skin": ([102815, 28349, 28690], 0.5, [[224, 224]]),
    "spider_thorax": ([49562, 13757, 14988], 0.5, [[224, 224]]),
    "tcga_crc_msi": ([14762, 4795, 32361], 0.5, [[512, 512]]),
    "tcga_tils": ([209221, 38601, 56275], 0.5, [[100, 100]]),
    "tcga_uniform": ([170840, 44670, 56200], 0.5, [[256, 256]]),
    "wilds": ([302436, 34904, 85054], 1.0, [[96, 96]]),
}

PATHOROB = {
    "camelyon": {
        "raw_rows": 22402,
        "excluded_ood_rows": 2002,
        "evaluated_rows": 20400,
        "metadata_file": "camelyon.csv",
        "dataset_license": "CC0-1.0",
    },
    "tcga_2x2": {
        "raw_rows": 112800,
        "excluded_ood_rows": 0,
        "evaluated_rows": 112800,
        "metadata_file": "tcga_2x2.csv",
        "dataset_license": "CC-BY-NC-SA-4.0",
    },
    "tolkach_esca": {
        "raw_rows": 9000,
        "excluded_ood_rows": 0,
        "evaluated_rows": 9000,
        "metadata_file": "tolkach_esca_reduced.csv",
        "dataset_license": "CC-BY-SA-4.0",
    },
}

EVA = {
    "bach": (5, "max_steps", 12500, 256),
    "bracs": (5, "max_steps", 12500, 256),
    "breakhis": (5, "max_steps", 12500, 256),
    "camelyon16_small": (20, "max_epochs", 100, 32),
    "consep": (5, "max_steps", 2000, 64),
    "crc": (5, "max_steps", 12500, 256),
    "gleason_arvaniti": (5, "max_steps", 12500, 256),
    "mhist": (5, "max_steps", 12500, 256),
    "monusac": (5, "max_steps", 2000, 64),
    "panda_small": (20, "max_epochs", 100, 32),
    "patch_camelyon": (5, "max_steps", 12500, 256),
    "patch_camelyon_10shot": (50, "max_steps", 12500, 256),
}

EVA_DATASET_LICENSE = {
    "bach": "CC-BY-NC-ND-4.0",
    "monusac": "CC-BY-NC-SA-4.0",
    "patch_camelyon": "CC0-1.0",
    "patch_camelyon_10shot": "CC0-1.0",
}

SOFTWARE_LICENSE = {
    "pathobench": ("CC-BY-NC-4.0", "pathobench_license_660e770"),
    "eva": ("Apache-2.0", "eva_license_e43e74a"),
    "thunder": ("CC-BY-4.0", "thunder_license_3d1cc95"),
    "hest": ("CC-BY-NC-SA-4.0", "hest_license_3ddb5ea"),
    "pathorob": ("BSD-3-Clause", "pathorob_license_6583cf0"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(
    value: Any,
    *,
    source_id: str,
    source_url: str,
    locator: str,
    evidence_type: str = "direct_primary_source",
    confidence: str = "high",
    unit: str | None = None,
    scope: str = "evaluation",
    notes: str | None = None,
) -> dict[str, Any]:
    result = {
        "status": "reported",
        "value": value,
        "source_id": source_id,
        "source_url": source_url,
        "locator": locator,
        "evidence_type": evidence_type,
        "confidence": confidence,
        "scope": scope,
    }
    if unit is not None:
        result["unit"] = unit
    if notes is not None:
        result["notes"] = notes
    return result


def missing(reason: str, searched_source_ids: list[str], *, unit: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "not_reported",
        "value": None,
        "reason": reason,
        "searched_source_ids": searched_source_ids,
    }
    if unit is not None:
        result["unit"] = unit
    return result


def suite_source_ids(suite: str) -> list[str]:
    return [source["source_id"] for source in SUITE_SOURCES if source["suite_id"] == suite]


def task_value(task: dict[str, str], field: str, value: Any, *, unit: str | None = None) -> dict[str, Any]:
    return evidence(
        value,
        source_id=f"task_source:{task['evaluation_id']}",
        source_url=task["reference_url"],
        locator=field,
        evidence_type="primary_task_config_or_report_table",
        confidence="high" if task["audit_status"] == "verified" else "medium",
        unit=unit,
        notes=task["audit_notes"],
    )


def label_artifact(task: dict[str, str]) -> dict[str, Any]:
    kinds = {
        "classification": "existing categorical or ordinal target label",
        "survival": "existing event and time-to-event outcome",
        "segmentation": "existing spatial mask or instance annotation",
        "regression": "existing continuous target measurement",
        "retrieval": "existing identity or grouping annotation",
        "robustness_analysis": "existing biological-class and acquisition-site metadata",
    }
    value = kinds.get(task["task_type"], "existing protocol target metadata")
    fact = task_value(task, "task_type + target", value)
    fact["evidence_type"] = "deterministic_protocol_metadata_classification"
    fact["confidence"] = "medium"
    fact["notes"] = "Describes the supplied label artifact only; it is not measured annotation labor."
    return fact


def access_fact(task: dict[str, str]) -> dict[str, Any]:
    suite = task["suite_id"]
    if suite == "pathobench":
        return evidence(
            "task split/config automatically downloadable; raw images or embeddings external",
            source_id="pathobench_readme_660e770",
            source_url=next(s["source_url"] for s in SUITE_SOURCES if s["source_id"] == "pathobench_readme_660e770"),
            locator="README.md:31-34,62-85",
            scope="benchmark_family",
            notes="Does not establish friction or license terms for each underlying image collection.",
        )
    if suite == "eva":
        return evidence(
            "download disabled by default in the pinned task configuration",
            source_id="eva_config_e43e74a",
            source_url=task["reference_url"],
            locator="data.*.download default",
            scope="evaluation",
            notes="A harness default is not evidence that the underlying dataset is open or gated.",
        )
    if suite == "thunder":
        if task["dataset_id"] == "mhist":
            return evidence(
                "manual access form followed by emailed download links",
                source_id="thunder_configs_3d1cc95",
                source_url=github_url(
                    "MICS-Lab/thunder", COMMITS["thunder"], "src/thunder/datasets/dataset/mhist.py"
                ),
                locator="mhist.py:4-18",
            )
        return evidence(
            "harness download implementation present",
            source_id="thunder_configs_3d1cc95",
            source_url=github_url(
                "MICS-Lab/thunder", COMMITS["thunder"], "src/thunder/datasets/download.py"
            ),
            locator=f"download_dataset branch for {task['dataset_id']}",
            notes="Implementation presence is not a blanket statement about dataset licensing or credentials.",
        )
    if suite == "hest":
        return evidence(
            "free Hugging Face access with subset querying",
            source_id="hest_readme_3ddb5ea",
            source_url=next(s["source_url"] for s in SUITE_SOURCES if s["source_id"] == "hest_readme_3ddb5ea"),
            locator="README.md:36-40",
            scope="benchmark_family",
        )
    return evidence(
        "official Hugging Face collection download",
        source_id="pathorob_readme_6583cf0",
        source_url=next(s["source_url"] for s in SUITE_SOURCES if s["source_id"] == "pathorob_readme_6583cf0"),
        locator="README.md:64-77",
        scope="benchmark_family",
    )


def compute_configuration(task: dict[str, str]) -> dict[str, Any]:
    suite = task["suite_id"]
    if suite == "eva":
        key = "patch_camelyon_10shot" if "10shot" in task["evaluation_id"] else task["dataset_id"]
        runs, limit_name, limit_value, train_batch = EVA[key]
        return evidence(
            {
                "offline_embedding_mode": True,
                "runs": runs,
                limit_name: limit_value,
                "train_batch_size": train_batch,
                "predict_batch_size": 64,
                "devices": 1,
                "accelerator": "auto",
            },
            source_id="eva_config_e43e74a",
            source_url=task["reference_url"],
            locator="trainer.init_args + data.init_args.dataloaders",
            evidence_type="declared_budget_not_observed_usage",
            notes="Step/epoch limits and batches are configuration ceilings/defaults, not elapsed time.",
        )
    if suite == "thunder":
        return evidence(
            {"device_selection": "CUDA when available, otherwise CPU", "orchestration": "SLURM template available"},
            source_id="thunder_runtime_context_3d1cc95",
            source_url=next(s["source_url"] for s in SUITE_SOURCES if s["source_id"] == "thunder_runtime_context_3d1cc95"),
            locator="README.md:110-141; src/thunder/benchmark.py:137-143",
            evidence_type="declared_execution_configuration",
            scope="benchmark_family",
        )
    if suite == "hest":
        return evidence(
            {"embedding_batch_size": 128, "workers": 1, "execution": "CUDA autocast"},
            source_id="hest_benchmark_config_3ddb5ea",
            source_url=next(s["source_url"] for s in SUITE_SOURCES if s["source_id"] == "hest_benchmark_config_3ddb5ea"),
            locator="bench_config.yaml:1-13; benchmark.py:196-207",
            evidence_type="declared_budget_not_observed_usage",
            scope="benchmark_default",
        )
    if suite == "pathorob":
        return evidence(
            {"feature_batch_size": 32, "workers": 0, "device": "CUDA when available, otherwise CPU"},
            source_id="pathorob_configs_6583cf0",
            source_url=next(s["source_url"] for s in SUITE_SOURCES if s["source_id"] == "pathorob_configs_6583cf0"),
            locator="extract_features.py:35-45",
            evidence_type="declared_budget_not_observed_usage",
            scope="benchmark_default",
        )
    return evidence(
        {
            "input_requirement": "precomputed patch or pooled embeddings",
            "pooling_device": "GPU with CPU fallback",
        },
        source_id="pathobench_readme_660e770",
        source_url=next(s["source_url"] for s in SUITE_SOURCES if s["source_id"] == "pathobench_readme_660e770"),
        locator="README.md:31,62-85; patho_bench/Pooler.py:59-74",
        evidence_type="benchmark_family_execution_context",
        scope="benchmark_family",
        notes="No retained result-specific runtime or hardware model is reported.",
    )


def sample_count(task: dict[str, str]) -> dict[str, Any]:
    if task["suite_id"] == "thunder":
        splits, _, _ = THUNDER[task["dataset_id"]]
        return evidence(
            {"train": splits[0], "validation": splits[1], "test": splits[2], "total": sum(splits)},
            source_id="thunder_configs_3d1cc95",
            source_url=task["reference_url"],
            locator="nb_train_samples + nb_val_samples + nb_test_samples",
            unit="images",
        )
    if task["suite_id"] == "pathorob":
        row = PATHOROB[task["dataset_id"]]
        url = github_url(
            "bifold-pathomics/PathoROB",
            COMMITS["pathorob"],
            f"data/metadata/{row['metadata_file']}",
        )
        return evidence(
            {
                "raw_metadata_rows": row["raw_rows"],
                "excluded_ood_rows": row["excluded_ood_rows"],
                "evaluated_rows": row["evaluated_rows"],
            },
            source_id="pathorob_configs_6583cf0",
            source_url=url,
            locator="CSV row count and robustness_index.get_meta OOD exclusion",
            unit="patch records",
        )
    value = task["num_samples"].strip()
    if value and value != "not_reported":
        return task_value(task, "num_samples", int(value), unit=task["sample_unit"])
    return missing(
        "No evaluation-specific count was reported in the retained task source.",
        suite_source_ids(task["suite_id"]) + [f"task_source:{task['evaluation_id']}"],
        unit=task["sample_unit"],
    )


def acquisition_scale(task: dict[str, str]) -> dict[str, Any]:
    if task["suite_id"] == "thunder":
        _, mpp, dimensions = THUNDER[task["dataset_id"]]
        return evidence(
            {"microns_per_pixel": mpp, "image_dimensions_pixels": dimensions},
            source_id="thunder_configs_3d1cc95",
            source_url=task["reference_url"],
            locator="mpp + image_sizes",
            notes="mpp is preserved verbatim and is not converted to nominal magnification.",
        )
    return missing(
        "No evaluation-specific magnification or mpp value was extracted from the audited source.",
        suite_source_ids(task["suite_id"]) + [f"task_source:{task['evaluation_id']}"],
    )


def stain(task: dict[str, str]) -> dict[str, Any]:
    if task["suite_id"] == "hest":
        return evidence(
            "H&E",
            source_id="hest_readme_3ddb5ea",
            source_url=next(s["source_url"] for s in SUITE_SOURCES if s["source_id"] == "hest_readme_3ddb5ea"),
            locator="README.md:14",
            scope="benchmark_family",
        )
    return missing(
        "Stain was not encoded as an evaluation-level field in the audited source.",
        suite_source_ids(task["suite_id"]) + [f"task_source:{task['evaluation_id']}"],
    )


def dataset_license(task: dict[str, str]) -> dict[str, Any]:
    suite = task["suite_id"]
    value = None
    source_id = None
    if suite == "hest":
        value, source_id = "CC-BY-NC-SA-4.0", "hest_license_3ddb5ea"
    elif suite == "pathorob":
        value, source_id = PATHOROB[task["dataset_id"]]["dataset_license"], "pathorob_readme_6583cf0"
    elif suite == "eva":
        key = "patch_camelyon_10shot" if "10shot" in task["evaluation_id"] else task["dataset_id"]
        value, source_id = EVA_DATASET_LICENSE.get(key), "eva_config_e43e74a"
    if value is not None and source_id is not None:
        return evidence(
            value,
            source_id=source_id,
            source_url=(task["reference_url"] if suite == "eva" else next(
                s["source_url"] for s in SUITE_SOURCES if s["source_id"] == source_id
            )),
            locator=("dataset license comments" if suite == "eva" else "license statement"),
        )
    return missing(
        "Underlying dataset license was not stated in the audited benchmark source; repository license is not substituted.",
        suite_source_ids(suite) + [f"task_source:{task['evaluation_id']}"],
    )


def feasibility_tier(task: dict[str, str], count_fact: dict[str, Any]) -> dict[str, Any]:
    unit = task["sample_unit"]
    task_type = task["task_type"]
    total = None
    if count_fact["status"] == "reported":
        value = count_fact["value"]
        total = value.get("total", value.get("evaluated_rows")) if isinstance(value, dict) else value
    if unit in {"image", "patch"} and task_type == "classification" and total is not None and total <= 10000:
        tier = "tier_1_direct_small_labeled"
        basis = "image_or_patch + classification + directly reported total <= 10000"
    elif unit in {"image", "patch"} and task_type == "classification":
        tier = "tier_2_direct_labeled"
        basis = "image_or_patch + classification"
    elif unit in {"slide", "case"} and task_type in {"classification", "survival", "retrieval"}:
        tier = "tier_3_aggregated_or_wsi"
        basis = "slide_or_case aggregation with supplied labels/outcomes"
    else:
        tier = "tier_4_specialized_protocol"
        basis = "segmentation, spatial regression, robustness, or other specialized protocol"
    return {
        "tier": tier,
        "basis": basis,
        "derivation_timing": "protocol metadata only; defined before prediction errors",
        "claim_boundary": "feasibility stratum, not measured money, labor, compute, or runtime",
    }


def build_record(task: dict[str, str]) -> dict[str, Any]:
    sources = suite_source_ids(task["suite_id"])
    count = sample_count(task)
    software_value, software_source = SOFTWARE_LICENSE[task["suite_id"]]
    facts = {
        "sample_count": count,
        "sample_unit": task_value(task, "sample_unit", task["sample_unit"]),
        "access": access_fact(task),
        "label_artifact": label_artifact(task),
        "annotation_hours": missing("No measured annotation labor was reported.", sources, unit="hours"),
        "acquisition_scale": acquisition_scale(task),
        "stain": stain(task),
        "compute_configuration": compute_configuration(task),
        "hardware_model": missing("No evaluation-specific hardware make/model was reported.", sources),
        "observed_runtime": missing("No observed per-evaluation elapsed time was reported.", sources, unit="seconds"),
        "dollar_cost": missing("No numeric evaluation dollar cost was reported.", sources, unit="USD"),
        "software_license": evidence(
            software_value,
            source_id=software_source,
            source_url=next(s["source_url"] for s in SUITE_SOURCES if s["source_id"] == software_source),
            locator="repository license",
            scope="benchmark_software",
            notes="Not a substitute for the underlying dataset license.",
        ),
        "dataset_license": dataset_license(task),
    }
    missing_fields = sorted(name for name, fact in facts.items() if fact["status"] != "reported")
    return {
        "evaluation_id": task["evaluation_id"],
        "suite_id": task["suite_id"],
        "dataset_id": task["dataset_id"],
        "protocol_id": task["protocol_id"],
        "task_source_url": task["reference_url"],
        "task_audit_status": task["audit_status"],
        "facts": facts,
        "pre_error_feasibility": feasibility_tier(task, count),
        "missing_fields": missing_fields,
        "numeric_cost_curve_eligible": False,
        "numeric_cost_curve_exclusion": "observed_runtime and dollar_cost are not both directly reported",
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = list(records[0]["facts"])
    coverage = {
        field: sum(record["facts"][field]["status"] == "reported" for record in records)
        for field in fields
    }
    direct_coverage = {
        field: sum(
            record["facts"][field]["status"] == "reported"
            and record["facts"][field].get("scope") == "evaluation"
            for record in records
        )
        for field in fields
    }
    by_suite: dict[str, dict[str, int]] = defaultdict(dict)
    for suite in sorted({record["suite_id"] for record in records}):
        suite_records = [record for record in records if record["suite_id"] == suite]
        by_suite[suite] = {
            field: sum(record["facts"][field]["status"] == "reported" for record in suite_records)
            for field in fields
        }
        by_suite[suite]["n_evaluations"] = len(suite_records)
    tiers = Counter(record["pre_error_feasibility"]["tier"] for record in records)
    return {
        "n_evaluations": len(records),
        "coverage_definition": (
            "field_coverage_count includes sourced benchmark-family/default context; "
            "field_direct_evaluation_coverage_count requires scope=evaluation"
        ),
        "field_coverage_count": coverage,
        "field_coverage_fraction": {field: count / len(records) for field, count in coverage.items()},
        "field_direct_evaluation_coverage_count": direct_coverage,
        "field_direct_evaluation_coverage_fraction": {
            field: count / len(records) for field, count in direct_coverage.items()
        },
        "field_coverage_by_suite": by_suite,
        "pre_error_feasibility_tier_counts": dict(sorted(tiers.items())),
        "numeric_cost_curve": {
            "supported": False,
            "eligible_evaluations": 0,
            "reason": (
                "No retained protocol has both directly reported observed evaluation runtime and dollar cost; "
                "configuration limits, sample counts, and qualitative access evidence are not converted to money."
            ),
        },
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "evaluation_id", "suite_id", "dataset_id", "sample_count_status", "sample_count_value",
        "sample_unit", "access", "label_artifact", "acquisition_scale_status", "acquisition_scale_value",
        "stain_status", "stain", "compute_configuration", "hardware_model_status",
        "observed_runtime_status", "annotation_hours_status", "dollar_cost_status",
        "software_license", "dataset_license_status", "dataset_license", "pre_error_feasibility_tier",
        "pre_error_feasibility_basis", "numeric_cost_curve_eligible", "task_source_url",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            facts = record["facts"]
            writer.writerow({
                "evaluation_id": record["evaluation_id"],
                "suite_id": record["suite_id"],
                "dataset_id": record["dataset_id"],
                "sample_count_status": facts["sample_count"]["status"],
                "sample_count_value": json.dumps(facts["sample_count"]["value"], sort_keys=True),
                "sample_unit": facts["sample_unit"]["value"],
                "access": facts["access"]["value"],
                "label_artifact": facts["label_artifact"]["value"],
                "acquisition_scale_status": facts["acquisition_scale"]["status"],
                "acquisition_scale_value": json.dumps(facts["acquisition_scale"]["value"], sort_keys=True),
                "stain_status": facts["stain"]["status"],
                "stain": facts["stain"]["value"],
                "compute_configuration": json.dumps(facts["compute_configuration"]["value"], sort_keys=True),
                "hardware_model_status": facts["hardware_model"]["status"],
                "observed_runtime_status": facts["observed_runtime"]["status"],
                "annotation_hours_status": facts["annotation_hours"]["status"],
                "dollar_cost_status": facts["dollar_cost"]["status"],
                "software_license": facts["software_license"]["value"],
                "dataset_license_status": facts["dataset_license"]["status"],
                "dataset_license": facts["dataset_license"]["value"],
                "pre_error_feasibility_tier": record["pre_error_feasibility"]["tier"],
                "pre_error_feasibility_basis": record["pre_error_feasibility"]["basis"],
                "numeric_cost_curve_eligible": "false",
                "task_source_url": record["task_source_url"],
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/tasks.csv")
    parser.add_argument("--scores", type=Path, default=ROOT / "data/scores.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "data/evaluation_cost_evidence.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "data/evaluation_cost_evidence.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = {row["evaluation_id"]: row for row in read_csv(args.tasks)}
    retained_ids = sorted({
        row["evaluation_id"]
        for row in read_csv(args.scores)
        if row["audit_status"] in RETAINED_STATUSES
    })
    missing_tasks = [evaluation_id for evaluation_id in retained_ids if evaluation_id not in tasks]
    if missing_tasks:
        raise ValueError(f"Retained evaluations without task metadata: {missing_tasks}")
    records = [build_record(tasks[evaluation_id]) for evaluation_id in retained_ids]
    task_sources = [
        {
            "source_id": f"task_source:{task['evaluation_id']}",
            "source_name": f"Retained protocol source for {task['evaluation_id']}",
            "suite_id": task["suite_id"],
            "evidence_type": "primary_task_config_or_report_table",
            "source_url": task["reference_url"],
            "locators": ["task row fields: num_samples, sample_unit, task_type, target, protocol"],
            "raw_evidence": {
                "protocol_id": task["protocol_id"],
                "dataset_id": task["dataset_id"],
                "num_samples": task["num_samples"],
                "sample_unit": task["sample_unit"],
                "task_type": task["task_type"],
                "target": task["target"],
                "audit_status": task["audit_status"],
            },
            "scope_note": task["audit_notes"],
        }
        for task in (tasks[evaluation_id] for evaluation_id in retained_ids)
    ]
    payload = {
        "schema_version": 1,
        "registry_id": "pathopress_evaluation_cost_evidence_v1",
        "upstream_cost_policy_reference": {
            "repository": "microsoft/benchpress",
            "commit": UPSTREAM_COMMIT,
            "path": "benchpress/data/benchmark_cost_evidence.README.md",
        },
        "policy": {
            "primary_sources_only": True,
            "raw_numbers_preserved": True,
            "configuration_budgets_are_not_observed_usage": True,
            "qualitative_claims_are_not_numeric_cost": True,
            "repository_license_is_not_dataset_license": True,
            "no_numeric_values_imputed": True,
        },
        "inputs": {
            "tasks_path": str(args.tasks.relative_to(ROOT)),
            "tasks_sha256": sha256(args.tasks),
            "scores_path": str(args.scores.relative_to(ROOT)),
            "scores_sha256": sha256(args.scores),
        },
        "sources": SUITE_SOURCES + task_sources,
        "summary": summarize(records),
        "evaluations": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, records)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
