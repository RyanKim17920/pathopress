#!/usr/bin/env python3
"""Extract the focused Virchow2G, Prov-GigaPath, and TITAN paper backlog.

Inputs are layout-preserving ``pdftotext -layout`` outputs from the official
paper/supplement PDFs.  The checked-in CSVs are deterministic extracts.  This
script intentionally keeps public frozen-feature rows separate from private,
fine-tuned, aggregate, and unreleased-protocol rows.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FIELDS = (
    "source_id", "suite_id", "table", "dataset_id", "task_name", "task_family", "target",
    "sample_unit", "task_type", "num_samples", "endpoint", "metric", "protocol",
    "model_alias", "model_id", "model_revision", "value", "uncertainty",
    "evaluation_id", "protocol_id", "task_identity_id", "dataset_artifact_id",
    "reference_url", "source_locator", "source_sha256", "review_status",
    "audit_status", "audit_notes",
)
QUARANTINE_FIELDS = (
    "source_id", "table", "dataset_id", "task_name", "metric", "protocol",
    "model_alias", "value", "uncertainty", "reference_url", "source_locator",
    "source_sha256", "quarantine_reason", "notes",
)

VIRCHOW_URL = "https://arxiv.org/pdf/2408.00738"
VIRCHOW_SHA = "41054dcfa720f5da2c933cb2a711c9d4618689a990513236b256652865418125"
GIGAPATH_URL = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-07441-w/MediaObjects/41586_2024_7441_MOESM1_ESM.pdf"
GIGAPATH_SHA = "b8279191eb635a60edd51621eab931f04d8d75b7af291794d8b91b58f73b866b"
TITAN_URL = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41591-025-03982-3/MediaObjects/41591_2025_3982_MOESM1_ESM.pdf"
TITAN_SHA = "26321e4018bec7b80f2fe7ea7cc497139c83b44fb60df5128417623ad1f71a70"
TITAN_REVISION = "github:mahmoodlab/TITAN@9e34c66ff66445c6c590da0dbf7acc103d39a40b"
GIGAPATH_REVISION = "github:prov-gigapath/prov-gigapath@92d8d20430f9645675db01c5f55e95ad58538526"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def table_block(text: str, number: int, *, supplementary: bool = True) -> str:
    prefix = "Supplementary Table" if supplementary else "Table"
    end = text.index(f"{prefix} {number}:")
    prior = text.rfind(f"{prefix} {number - 1}:", 0, end)
    if prior < 0:
        prior = 0
    return text[prior:end]


def numeric_cells(line: str) -> list[tuple[str, str]]:
    return re.findall(r"(0\.\d+)(?:\s*±\s*(0\.\d+))?", line)


def add_included(
    rows: list[dict[str, str]], *, source_id: str, table: str, dataset_id: str,
    task_name: str, task_family: str, target: str, sample_unit: str,
    task_type: str, num_samples: str, endpoint: str, metric: str,
    protocol: str, model_alias: str, model_id: str, model_revision: str,
    value: str, uncertainty: str, reference_url: str, source_locator: str,
    source_sha256: str, evaluation_id: str | None = None,
) -> None:
    task_slug = slug(task_name)
    protocol_slug = slug(protocol)
    evaluation_id = evaluation_id or f"{source_id}.{dataset_id}.{task_slug}.{protocol_slug}.{slug(metric)}"
    rows.append({
        "source_id": source_id,
        "suite_id": "virchow2g_paper" if source_id == "virchow2g2024" else "titan_paper",
        "table": table,
        "dataset_id": dataset_id,
        "task_name": task_name,
        "task_family": task_family,
        "target": target,
        "sample_unit": sample_unit,
        "task_type": task_type,
        "num_samples": num_samples,
        "endpoint": endpoint,
        "metric": metric,
        "protocol": protocol,
        "model_alias": model_alias,
        "model_id": model_id,
        "model_revision": model_revision,
        "value": value,
        "uncertainty": uncertainty,
        "evaluation_id": evaluation_id,
        "protocol_id": evaluation_id,
        "task_identity_id": f"task.{dataset_id}.{task_slug}",
        "dataset_artifact_id": f"artifact.{dataset_id}",
        "reference_url": reference_url,
        "source_locator": source_locator,
        "source_sha256": source_sha256,
        "review_status": "machine_parsed_single_primary_source",
        "audit_status": "parsed_primary_source",
        "audit_notes": "Exact leaf result; protocol variants remain distinct and are not deduplicated by dataset name alone.",
    })


def add_quarantine(
    rows: list[dict[str, str]], *, source_id: str, table: str,
    dataset_id: str, task_name: str, metric: str, protocol: str,
    model_alias: str, value: str, uncertainty: str, reference_url: str,
    source_locator: str, source_sha256: str, reason: str, notes: str,
) -> None:
    rows.append({
        "source_id": source_id, "table": table, "dataset_id": dataset_id,
        "task_name": task_name, "metric": metric, "protocol": protocol,
        "model_alias": model_alias, "value": value, "uncertainty": uncertainty,
        "reference_url": reference_url, "source_locator": source_locator,
        "source_sha256": source_sha256, "quarantine_reason": reason, "notes": notes,
    })


def extract_virchow(text: str, included: list[dict[str, str]], quarantine: list[dict[str, str]]) -> None:
    if "Virchow2G       0.559   0.385" not in text or "Virchow2G             94.7" not in text:
        raise ValueError("Virchow2G table anchors missing")
    ood_names = ("PCam", "CRC", "CRC No-Norm", "WILDS", "TILS", "MHIST", "DLBCL", "MIDOG")
    virchow_models = {
        "Virchow2G": ("virchow-2g", "paper_only_unreleased_checkpoint"),
        "Virchow2": ("virchow-2", "paper_reported_checkpoint"),
        "Virchow": ("virchow", "paper_reported_checkpoint"),
    }
    ood_values = {
        ("Virchow2G", "CLS+Mean"): ("0.947", "0.973", "0.970", "0.988", "0.948", "0.864", "0.629", "0.836"),
        ("Virchow2G", "CLS-Only"): ("0.944", "0.974", "0.970", "0.983", "0.947", "0.852", "0.624", "0.805"),
        ("Virchow2", "CLS+Mean"): ("0.935", "0.974", "0.969", "0.987", "0.948", "0.859", "0.606", "0.804"),
        ("Virchow2", "CLS-Only"): ("0.935", "0.976", "0.971", "0.985", "0.947", "0.860", "0.615", "0.800"),
        ("Virchow", "CLS+Mean"): ("0.933", "0.973", "0.968", "0.971", "0.949", "0.836", "0.602", "0.787"),
        ("Virchow", "CLS-Only"): ("0.934", "0.970", "0.932", "0.966", "0.948", "0.836", "0.591", "0.760"),
    }
    sizes = {"PCam": "327680", "CRC": "100000", "CRC No-Norm": "100000", "WILDS": "455954", "TILS": "304097", "MHIST": "3152", "DLBCL": "209", "MIDOG": "21806"}
    for (model, embedding), values in ood_values.items():
        model_id, revision = virchow_models[model]
        for name, value in zip(ood_names, values):
            dataset_id = slug(name)
            protocol = f"virchow2_linear_probe_{slug(embedding)}"
            add_included(
                included, source_id="virchow2g2024", table="Table 2", dataset_id=dataset_id,
                task_name=name, task_family="tile_classification", target=name,
                sample_unit="tile" if name != "DLBCL" else "patient", task_type="classification",
                num_samples=sizes[name], endpoint="tile_level_classification", metric="weighted_f1",
                protocol=protocol, model_alias=f"{model} {embedding}", model_id=model_id,
                model_revision=revision, value=value, uncertainty="not_reported",
                reference_url=VIRCHOW_URL, source_locator=f"paper=Virchow2|table=2|model={model}|embedding={embedding}|task={name}",
                source_sha256=VIRCHOW_SHA,
            )
    hest_names = ("IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC", "HCC", "LUAD", "LIDC")
    hest_values = {
        ("Virchow2G", "CLS+Mean"): ("0.559", "0.385", "0.458", "0.632", "0.139", "0.175", "0.222", "0.062", "0.588", "0.274"),
        ("Virchow2G", "CLS-Only"): ("0.547", "0.375", "0.420", "0.638", "0.140", "0.146", "0.213", "0.056", "0.590", "0.271"),
        ("Virchow2", "CLS+Mean"): ("0.539", "0.382", "0.425", "0.617", "0.127", "0.168", "0.226", "0.056", "0.586", "0.273"),
        ("Virchow2", "CLS-Only"): ("0.563", "0.379", "0.402", "0.631", "0.143", "0.169", "0.224", "0.060", "0.591", "0.275"),
        ("Virchow", "CLS+Mean"): ("0.545", "0.372", "0.465", "0.624", "0.159", "0.133", "0.211", "0.065", "0.601", "0.269"),
        ("Virchow", "CLS-Only"): ("0.529", "0.364", "0.431", "0.612", "0.150", "0.120", "0.205", "0.065", "0.584", "0.274"),
    }
    for (model, embedding), values in hest_values.items():
        model_id, revision = virchow_models[model]
        for name, value in zip(hest_names, values):
            protocol = f"virchow2_hest_random_forest_{slug(embedding)}"
            add_included(
                included, source_id="virchow2g2024", table="Table 3", dataset_id=f"hest-{name.lower()}",
                task_name=f"HEST {name}", task_family="spatial_transcriptomics", target="top_50_variable_genes",
                sample_unit="spatial_transcriptomics_spot", task_type="regression", num_samples="not_reported_by_task",
                endpoint="gene_expression_prediction", metric="pearson_r", protocol=protocol,
                model_alias=f"{model} {embedding}", model_id=model_id,
                model_revision=revision, value=value, uncertainty="not_reported",
                reference_url=VIRCHOW_URL, source_locator=f"paper=Virchow2|table=3|model={model}|embedding={embedding}|task={name}",
                source_sha256=VIRCHOW_SHA,
            )
    pan_values = {
        ("Virchow2G", "CLS+Mean"): ("0.966", "0.971", "0.975"), ("Virchow2G", "CLS-Only"): ("0.966", "0.972", "0.975"),
        ("Virchow2", "CLS+Mean"): ("0.964", "0.966", "0.967"), ("Virchow2", "CLS-Only"): ("0.964", "0.967", "0.968"),
        ("Virchow", "CLS+Mean"): ("0.950", "0.948", "0.933"), ("Virchow", "CLS-Only"): ("0.950", "0.949", "0.924"),
    }
    for (model, embedding), values in pan_values.items():
        for magnification, value in zip(("20x", "10x", "5x"), values):
            add_quarantine(
                quarantine, source_id="virchow2g2024", table="Table 1", dataset_id="panmsk",
                task_name=f"PanMSK {magnification}", metric="weighted_f1",
                protocol=f"virchow2_linear_probe_{slug(embedding)}", model_alias=f"{model} {embedding}",
                value=value, uncertainty="not_reported", reference_url=VIRCHOW_URL,
                source_locator=f"paper=Virchow2|table=1|model={model}|embedding={embedding}|task=PanMSK {magnification}",
                source_sha256=VIRCHOW_SHA, reason="internal_private_cohort",
                notes="PanMSK is sourced from MSKCC and is not a downloadable public benchmark artifact.",
            )
    aggregate_values = {
        ("Virchow2G", "CLS+Mean"): ("0.970", "0.894", "0.349"), ("Virchow2G", "CLS-Only"): ("0.971", "0.887", "0.339"),
        ("Virchow2", "CLS+Mean"): ("0.966", "0.885", "0.340"), ("Virchow2", "CLS-Only"): ("0.966", "0.886", "0.344"),
        ("Virchow", "CLS+Mean"): ("0.944", "0.877", "0.344"), ("Virchow", "CLS-Only"): ("0.941", "0.867", "0.334"),
    }
    for (model, embedding), values in aggregate_values.items():
        for table, label, value in zip(("Table 1", "Table 2", "Table 3"), ("PanMSK average", "OOD average", "HEST average"), values):
            add_quarantine(
                quarantine, source_id="virchow2g2024", table=table, dataset_id="aggregate",
                task_name=label, metric="aggregate", protocol=f"virchow2g_{slug(embedding)}",
                model_alias=f"{model} {embedding}", value=value, uncertainty="not_reported",
                reference_url=VIRCHOW_URL, source_locator=f"paper=Virchow2|{table}|model={model}|embedding={embedding}|aggregate={label}",
                source_sha256=VIRCHOW_SHA, reason="aggregate_not_leaf_task",
                notes="Reported mean is retained for audit only and excluded from the model-by-task matrix.",
            )


def extract_gigapath(text: str, quarantine: list[dict[str, str]]) -> None:
    block = table_block(text, 2)
    expected = (
        "NSCLC Typing", "BRCA Typing", "RCC Typing", "COADREAD Typing", "HB Typing", "DIFG Typing",
        "OVT Typing", "CNS Typing", "EGC Typing", "Pan EGFR", "Pan FAT1", "Pan KRAS", "Pan LRP1B",
        "Pan TP53", "LUAD EGFR", "LUAD FAT1", "LUAD KRAS", "LUAD LRP1B", "LUAD TP53",
        "LUAD EGFR (TCGA)", "LUAD FAT1 (TCGA)", "LUAD KRAS (TCGA)", "LUAD LRP1B (TCGA)",
        "LUAD TP53 (TCGA)", "Pan 18-biomarkers", "Pan TMB",
    )
    found = {}
    for line in block.splitlines():
        for task in expected:
            if re.match(rf"^\s*{re.escape(task)}\s+0\.", line):
                cells = numeric_cells(line)
                if cells:
                    found[task] = cells[0]
    if set(found) != set(expected):
        raise ValueError(f"GigaPath table 2 task mismatch: missing {set(expected) - set(found)}")
    for task in expected:
        value, error = found[task]
        public_tcga = task.endswith("(TCGA)")
        add_quarantine(
            quarantine, source_id="provgigapath2024", table="Supplementary Table 2",
            dataset_id="tcga-luad" if public_tcga else "prov-path", task_name=task, metric="auroc",
            protocol="task_specific_slide_encoder_finetuning", model_alias="Prov-GigaPath slide encoder",
            value=value, uncertainty=f"standard_error={error}", reference_url=GIGAPATH_URL,
            source_locator=f"paper=Prov-GigaPath|supplementary_table=2|task={task}|model=Prov-GigaPath",
            source_sha256=GIGAPATH_SHA,
            reason="task_specific_finetuning" if public_tcga else "internal_private_cohort_and_finetuning",
            notes=("Public TCGA cohort, but the paper fine-tunes the slide encoder for each downstream task."
                   if public_tcga else "Providence/Prov-Path cohort is not a downloadable benchmark and uses task-specific slide-encoder fine-tuning."),
        )


TITAN_TASKS = {
    22: ("tcga-ut-8k", "TCGA-UT-8K tumor subtype", "tumor_subtype", "7784", "32"),
    23: ("tcga-ot", "TCGA-OT OncoTree code", "oncotree_code", "9149", "46"),
    24: ("ot108", "OT108 OncoTree code", "oncotree_code", "5510", "108"),
    25: ("ebrains", "EBRAINS tumor type", "diagnosis", "2147", "30"),
}


def titan_rows(block: str, *, finetuned: bool | None = None) -> dict[str, list[tuple[str, str]]]:
    result = {}
    for raw in block.splitlines():
        line = raw.strip()
        for alias in ("TITANV", "TITAN"):
            marker = f"{alias} (finetuned)"
            if finetuned is True and not line.startswith(marker):
                continue
            if finetuned is False and not (line.startswith(alias) and not line.startswith(marker)):
                continue
            if finetuned is None and not (line.startswith(alias) and "finetuned" not in line):
                continue
            cells = numeric_cells(line)
            if cells:
                result[alias] = cells
    return result


def titan_model(alias: str) -> tuple[str, str]:
    return ("TITAN-V vision-only", "titan-v-slide") if alias == "TITANV" else ("TITAN vision-language", "titan-slide")


TITAN_BINARY_METRICS = (
    ("auroc", "logistic_regression"), ("balanced_accuracy", "logistic_regression"),
    ("balanced_accuracy", "simpleshot"), ("weighted_f1", "simpleshot"),
    ("balanced_accuracy", "20_nearest_neighbors"), ("weighted_f1", "20_nearest_neighbors"),
)
TITAN_MULTICLASS_METRICS = (
    ("balanced_accuracy", "logistic_regression"), ("weighted_f1", "logistic_regression"),
    ("balanced_accuracy", "simpleshot"), ("weighted_f1", "simpleshot"),
    ("balanced_accuracy", "20_nearest_neighbors"), ("weighted_f1", "20_nearest_neighbors"),
)
TITAN_GRADING_METRICS = (
    ("quadratic_weighted_kappa", "logistic_regression"), ("balanced_accuracy", "logistic_regression"),
    ("quadratic_weighted_kappa", "simpleshot"), ("balanced_accuracy", "simpleshot"),
    ("quadratic_weighted_kappa", "20_nearest_neighbors"), ("balanced_accuracy", "20_nearest_neighbors"),
)


def titan_labeled_rows(block: str) -> list[tuple[str, str, list[tuple[str, str]]]]:
    result = []
    for raw in block.splitlines():
        line = raw.strip()
        match = re.match(r"^(TITANV|TITAN)\s+(.*)$", line)
        if not match or "finetuned" in line:
            continue
        cells = numeric_cells(line)
        if cells:
            prefix = match.group(2).split(cells[0][0], 1)[0].strip()
            result.append((match.group(1), prefix, cells))
    return result


def extract_titan_broad_tasks(text: str, included: list[dict[str, str]], quarantine: list[dict[str, str]]) -> None:
    # Exact leaves from Tables 26--63.  The downloadable/public determination is
    # anchored to Supplementary Table 17; MGB/MGH/CRANE/renal cohorts stay quarantined.
    single_public = {
        26: ("tcga-brca", "TCGA-BRCA breast subtype", "subtype", "morphological_subtyping", "984", "slide", TITAN_BINARY_METRICS),
        29: ("bracs", "BRACS fine subtype", "subtype_7_class", "morphological_subtyping", "189", "slide", TITAN_MULTICLASS_METRICS),
        30: ("bracs", "BRACS coarse subtype", "subtype_3_class", "morphological_subtyping", "189", "slide", TITAN_MULTICLASS_METRICS),
        35: ("imp", "IMP dysplasia grading", "dysplasia_grade", "grading", "5333", "slide", TITAN_GRADING_METRICS),
        36: ("panda", "PANDA Gleason grading", "gleason_grade", "grading", "9555", "slide", TITAN_GRADING_METRICS),
        37: ("mut-het-rcc", "MUT-HET-RCC BAP1 mutation", "BAP1", "molecular_biomarker", "1292", "patient", TITAN_BINARY_METRICS),
        38: ("mut-het-rcc", "MUT-HET-RCC PBRM1 mutation", "PBRM1", "molecular_biomarker", "1292", "patient", TITAN_BINARY_METRICS),
        39: ("mut-het-rcc", "MUT-HET-RCC SETD2 mutation", "SETD2", "molecular_biomarker", "1292", "patient", TITAN_BINARY_METRICS),
        40: ("bcnb", "BCNB ER status", "ER", "molecular_biomarker", "1058", "patient", TITAN_BINARY_METRICS),
        41: ("bcnb", "BCNB PR status", "PR", "molecular_biomarker", "1058", "patient", TITAN_BINARY_METRICS),
        42: ("bcnb", "BCNB HER2 status", "HER2", "molecular_biomarker", "1058", "patient", TITAN_BINARY_METRICS),
        55: ("pdl1-luad", "PD-L1 expression level", "PD-L1_level", "molecular_biomarker", "217", "patient", TITAN_GRADING_METRICS),
    }
    cross_public = {
        27: {
            "TCGA": ("tcga-nsclc", "TCGA-NSCLC subtype", "subtype", "946"),
            "CPTAC": ("cptac-nsclc", "CPTAC-NSCLC subtype", "subtype", "422"),
        },
        28: {
            "TCGA": ("tcga-rcc", "TCGA-RCC OncoTree code", "oncotree_code", "895"),
            "CPTAC-DHMC": ("cptac-dhmc-rcc", "CPTAC-DHMC RCC OncoTree code", "oncotree_code", "673"),
        },
        43: {
            "TCGA": ("tcga-gbmlgg", "TCGA-GBMLGG IDH mutation", "IDH", "558"),
            "EBRAINS": ("ebrains-idh", "EBRAINS IDH mutation", "IDH", "795"),
        },
    }
    brca_targets = {44: ("ER", "937", "102"), 45: ("PR", "934", "97"), 46: ("HER2", "647", "103"), 47: ("PIK3CA", "970", "103")}
    for table, (target, tcga_n, cptac_n) in brca_targets.items():
        cross_public[table] = {
            "TCGA": ("tcga-brca", f"TCGA-BRCA {target} status", target, tcga_n),
            "CPTAC": ("cptac-brca", f"CPTAC-BRCA {target} status", target, cptac_n),
        }
    lung_targets = {48: "EGFR", 49: "TP53"}
    for table, target in lung_targets.items():
        cross_public[table] = {
            "TCGA": ("tcga-nsclc", f"TCGA-NSCLC {target} mutation", target, "462"),
            "CPTAC": ("cptac-luad", f"CPTAC-LUAD {target} mutation", target, "108"),
        }
    crc_targets = {50: "MSI", 51: "BRAF", 52: "KRAS"}
    for table, target in crc_targets.items():
        cross_public[table] = {
            "TCGA": ("tcga-crc", f"TCGA-CRC {target} status", target, "414" if target == "MSI" else ("487" if target == "BRAF" else "not_reported")),
            "CPTAC": ("cptac-coad", f"CPTAC-COAD {target} status", target, "103"),
        }

    for table, spec in single_public.items():
        dataset_id, task_name, target, family, count, unit, metrics = spec
        rows = titan_labeled_rows(table_block(text, table))
        if len(rows) != 2 or any(label for _, label, _ in rows):
            raise ValueError(f"unexpected TITAN broad public table {table}")
        for alias, _, cells in rows:
            model_alias, model_id = titan_model(alias)
            for (metric, evaluator), (value, error) in zip(metrics, cells):
                add_included(
                    included, source_id="titan2025", table=f"Supplementary Table {table}", dataset_id=dataset_id,
                    task_name=task_name, task_family=family, target=target, sample_unit=unit,
                    task_type="classification", num_samples=count, endpoint="slide_level_classification",
                    metric=metric, protocol=f"titan2025_frozen_{evaluator}", model_alias=model_alias,
                    model_id=model_id, model_revision=TITAN_REVISION, value=value,
                    uncertainty=f"reported_error={error}", reference_url=TITAN_URL,
                    source_locator=f"paper=TITAN|supplementary_table={table}|model={alias}|evaluator={evaluator}|metric={metric}",
                    source_sha256=TITAN_SHA,
                )

    for table, cohorts in cross_public.items():
        metrics = TITAN_MULTICLASS_METRICS if table == 28 else TITAN_BINARY_METRICS
        family = "morphological_subtyping" if table in (27, 28) else "molecular_biomarker"
        rows = titan_labeled_rows(table_block(text, table))
        if len(rows) != 4:
            raise ValueError(f"unexpected TITAN cross-cohort table {table}")
        for alias, label, cells in rows:
            dataset_id, task_name, target, count = cohorts[label]
            model_alias, model_id = titan_model(alias)
            for (metric, evaluator), (value, error) in zip(metrics, cells):
                add_included(
                    included, source_id="titan2025", table=f"Supplementary Table {table}", dataset_id=dataset_id,
                    task_name=task_name, task_family=family, target=target, sample_unit="slide" if table in (27, 28, 43) else "patient",
                    task_type="classification", num_samples=count, endpoint="slide_level_classification",
                    metric=metric, protocol=f"titan2025_frozen_{evaluator}", model_alias=model_alias,
                    model_id=model_id, model_revision=TITAN_REVISION, value=value,
                    uncertainty=f"reported_error={error}", reference_url=TITAN_URL,
                    source_locator=f"paper=TITAN|supplementary_table={table}|cohort={label}|model={alias}|evaluator={evaluator}|metric={metric}",
                    source_sha256=TITAN_SHA,
                )

    internal = {
        31: ("crane", "CRANE cellular rejection", TITAN_BINARY_METRICS),
        32: ("renal-allograft", "Renal antibody-mediated rejection", TITAN_BINARY_METRICS),
        33: ("renal-allograft", "Renal cellular rejection", TITAN_BINARY_METRICS),
        34: ("renal-allograft", "Renal IFTA grade", TITAN_GRADING_METRICS),
        53: ("mgh-brca", "MGH-BRCA ER expression level", TITAN_GRADING_METRICS),
        54: ("mgh-brca", "MGH-BRCA PR expression level", TITAN_GRADING_METRICS),
        56: ("mgb-brca", "MGB-BRCA ER status", TITAN_BINARY_METRICS),
        57: ("mgb-brca", "MGB-BRCA PR status", TITAN_BINARY_METRICS),
        58: ("mgb-brca", "MGB-BRCA HER2 status", TITAN_BINARY_METRICS),
        59: ("mgb-luad", "MGB-LUAD CDX-2 expression", TITAN_BINARY_METRICS),
        60: ("mgb-luad", "MGB-LUAD CK-5&6 expression", TITAN_BINARY_METRICS),
        61: ("mgb-luad", "MGB-LUAD Napsin A expression", TITAN_BINARY_METRICS),
        62: ("mgb-luad", "MGB-LUAD P40 expression", TITAN_BINARY_METRICS),
        63: ("mgb-luad", "MGB-LUAD P63 expression", TITAN_BINARY_METRICS),
    }
    for table, (dataset_id, task_name, metrics) in internal.items():
        rows = titan_labeled_rows(table_block(text, table))
        if len(rows) != 2:
            raise ValueError(f"unexpected TITAN internal table {table}")
        for alias, _, cells in rows:
            for (metric, evaluator), (value, error) in zip(metrics, cells):
                add_quarantine(
                    quarantine, source_id="titan2025", table=f"Supplementary Table {table}", dataset_id=dataset_id,
                    task_name=task_name, metric=metric, protocol=f"titan2025_frozen_{evaluator}",
                    model_alias=titan_model(alias)[0], value=value, uncertainty=f"reported_error={error}",
                    reference_url=TITAN_URL,
                    source_locator=f"paper=TITAN|supplementary_table={table}|model={alias}|evaluator={evaluator}|metric={metric}",
                    source_sha256=TITAN_SHA, reason="internal_private_cohort",
                    notes="Cohort is institutional/private and is excluded from the public benchmark matrix.",
                )

    # Disease-specific survival is a public TCGA task, with the paper's five-fold
    # site-preserving protocol and the final two columns corresponding to TITAN-V/TITAN.
    survival_n = {"BLCA": "360", "BRCA": "1022", "CRC": "545", "KIRC": "502", "NSCLC": "844", "UCEC": "504"}
    block = table_block(text, 64)
    for cohort, count in survival_n.items():
        line = next(line for line in block.splitlines() if line.strip().startswith(cohort))
        cells = numeric_cells(line)
        if len(cells) != 6:
            raise ValueError(f"unexpected TITAN survival row {cohort}")
        for alias, (value, error) in zip(("TITANV", "TITAN"), cells[-2:]):
            model_alias, model_id = titan_model(alias)
            add_included(
                included, source_id="titan2025", table="Supplementary Table 64", dataset_id=f"tcga-{cohort.lower()}",
                task_name=f"TCGA-{cohort} disease-specific survival", task_family="survival", target="disease_specific_survival",
                sample_unit="patient", task_type="survival", num_samples=count, endpoint="survival_prediction",
                metric="concordance_index", protocol="titan2025_frozen_five_fold_site_preserving",
                model_alias=model_alias, model_id=model_id, model_revision=TITAN_REVISION, value=value,
                uncertainty=f"reported_error={error}", reference_url=TITAN_URL,
                source_locator=f"paper=TITAN|supplementary_table=64|cohort={cohort}|model={alias}|metric=concordance_index",
                source_sha256=TITAN_SHA,
            )
    avg = next(line for line in block.splitlines() if line.strip().startswith("Avg."))
    for alias, (value, _) in zip(("TITANV", "TITAN"), numeric_cells(avg)[-2:]):
        add_quarantine(
            quarantine, source_id="titan2025", table="Supplementary Table 64", dataset_id="aggregate",
            task_name="Six-cohort survival average", metric="concordance_index", protocol="aggregate_survival",
            model_alias=titan_model(alias)[0], value=value, uncertainty="not_reported", reference_url=TITAN_URL,
            source_locator=f"paper=TITAN|supplementary_table=64|model={alias}|aggregate=average",
            source_sha256=TITAN_SHA, reason="aggregate_not_leaf_task",
            notes="Reported mean is retained for audit only and excluded from the model-by-task matrix.",
        )


def extract_titan(text: str, included: list[dict[str, str]], quarantine: list[dict[str, str]]) -> None:
    extract_titan_broad_tasks(text, included, quarantine)
    # Full-data frozen evaluation: keep every explicitly reported evaluator/metric cell.
    full_metrics = (
        ("balanced_accuracy", "logistic_regression"), ("weighted_f1", "logistic_regression"),
        ("balanced_accuracy", "simpleshot"), ("weighted_f1", "simpleshot"),
        ("balanced_accuracy", "20_nearest_neighbors"), ("weighted_f1", "20_nearest_neighbors"),
    )
    for table in (22, 23, 25):
        dataset_id, task_name, target, num_samples, _ = TITAN_TASKS[table]
        rows = titan_rows(table_block(text, table))
        if set(rows) != {"TITANV", "TITAN"} or any(len(v) != 6 for v in rows.values()):
            raise ValueError(f"unexpected TITAN table {table} rows")
        for alias, cells in rows.items():
            model_alias, model_id = titan_model(alias)
            for (metric, evaluator), (value, error) in zip(full_metrics, cells):
                protocol = f"titan2025_frozen_{evaluator}"
                add_included(
                    included, source_id="titan2025", table=f"Supplementary Table {table}", dataset_id=dataset_id,
                    task_name=task_name, task_family="morphological_subtyping", target=target, sample_unit="slide",
                    task_type="classification", num_samples=num_samples, endpoint="slide_level_classification",
                    metric=metric, protocol=protocol, model_alias=model_alias, model_id=model_id,
                    model_revision=TITAN_REVISION, value=value, uncertainty=f"reported_error={error}",
                    reference_url=TITAN_URL, source_locator=f"paper=TITAN|supplementary_table={table}|model={alias}|evaluator={evaluator}|metric={metric}",
                    source_sha256=TITAN_SHA,
                )
    # Few-shot evaluations: K is part of the protocol identity.
    for table, base_table, evaluator in ((82, 22, "linear_probe"), (83, 23, "linear_probe"), (85, 25, "linear_probe"),
                                         (86, 22, "simpleshot"), (87, 23, "simpleshot"), (89, 25, "simpleshot")):
        dataset_id, task_name, target, num_samples, _ = TITAN_TASKS[base_table]
        rows = titan_rows(table_block(text, table))
        if set(rows) != {"TITANV", "TITAN"} or any(len(v) != 6 for v in rows.values()):
            raise ValueError(f"unexpected TITAN few-shot table {table} rows")
        for alias, cells in rows.items():
            model_alias, model_id = titan_model(alias)
            for k, (value, error) in zip((1, 2, 4, 8, 16, 32), cells):
                protocol = f"titan2025_{k}shot_{evaluator}"
                add_included(
                    included, source_id="titan2025", table=f"Supplementary Table {table}", dataset_id=dataset_id,
                    task_name=task_name, task_family="morphological_subtyping", target=target, sample_unit="slide",
                    task_type="classification", num_samples=num_samples, endpoint="slide_level_classification",
                    metric="balanced_accuracy", protocol=protocol, model_alias=model_alias, model_id=model_id,
                    model_revision=TITAN_REVISION, value=value, uncertainty=f"reported_error={error}",
                    reference_url=TITAN_URL, source_locator=f"paper=TITAN|supplementary_table={table}|model={alias}|shots={k}|evaluator={evaluator}",
                    source_sha256=TITAN_SHA,
                )
    # Zero-shot exists only for the multimodal TITAN model.
    for table, base_table in ((97, 22), (98, 23), (100, 25)):
        dataset_id, task_name, target, num_samples, _ = TITAN_TASKS[base_table]
        rows = titan_rows(table_block(text, table))
        cells = rows.get("TITAN", [])
        if len(cells) != 3:
            raise ValueError(f"unexpected TITAN zero-shot table {table} row")
        for metric, (value, error) in zip(("balanced_accuracy", "weighted_f1", "auroc"), cells):
            add_included(
                included, source_id="titan2025", table=f"Supplementary Table {table}", dataset_id=dataset_id,
                task_name=task_name, task_family="morphological_subtyping", target=target, sample_unit="slide",
                task_type="classification", num_samples=num_samples, endpoint="slide_level_classification",
                metric=metric, protocol="titan2025_zero_shot_prompt_ensemble", model_alias="TITAN vision-language",
                model_id="titan-slide", model_revision=TITAN_REVISION, value=value,
                uncertainty=f"reported_error={error}", reference_url=TITAN_URL,
                source_locator=f"paper=TITAN|supplementary_table={table}|model=TITAN|protocol=zero-shot|metric={metric}",
                source_sha256=TITAN_SHA,
            )
    # Additional downloadable/public zero-shot cohorts.
    zero_public = {
        101: ("tcga-rcc", "TCGA-RCC OncoTree code", "oncotree_code", "895", "morphological_subtyping"),
        102: ("cptac-dhmc-rcc", "CPTAC-DHMC RCC OncoTree code", "oncotree_code", "673", "morphological_subtyping"),
        103: ("dhmc-luad", "DHMC-LUAD histological pattern", "histological_pattern", "not_reported", "morphological_subtyping"),
        104: ("tcga-nsclc", "TCGA-NSCLC subtype", "subtype", "946", "morphological_subtyping"),
        105: ("cptac-nsclc", "CPTAC-NSCLC subtype", "subtype", "422", "morphological_subtyping"),
        106: ("tcga-brca", "TCGA-BRCA breast subtype", "subtype", "984", "morphological_subtyping"),
    }
    for table, (dataset_id, task_name, target, count, family) in zero_public.items():
        cells = titan_rows(table_block(text, table)).get("TITAN", [])
        if len(cells) != 3:
            raise ValueError(f"unexpected TITAN public zero-shot table {table}")
        for metric, (value, error) in zip(("balanced_accuracy", "weighted_f1", "auroc"), cells):
            add_included(
                included, source_id="titan2025", table=f"Supplementary Table {table}", dataset_id=dataset_id,
                task_name=task_name, task_family=family, target=target, sample_unit="slide",
                task_type="classification", num_samples=count, endpoint="slide_level_classification",
                metric=metric, protocol="titan2025_zero_shot_prompt_ensemble", model_alias="TITAN vision-language",
                model_id="titan-slide", model_revision=TITAN_REVISION, value=value,
                uncertainty=f"reported_error={error}", reference_url=TITAN_URL,
                source_locator=f"paper=TITAN|supplementary_table={table}|model=TITAN|protocol=zero-shot|metric={metric}",
                source_sha256=TITAN_SHA,
            )
    for table, dataset_id, task_name in (
        (107, "crane", "CRANE cellular rejection"),
        (108, "renal-allograft", "Renal cellular rejection"),
        (109, "renal-allograft", "Renal antibody-mediated rejection"),
    ):
        cells = titan_rows(table_block(text, table)).get("TITAN", [])
        if len(cells) != 3:
            raise ValueError(f"unexpected TITAN private zero-shot table {table}")
        for metric, (value, error) in zip(("balanced_accuracy", "weighted_f1", "auroc"), cells):
            add_quarantine(
                quarantine, source_id="titan2025", table=f"Supplementary Table {table}", dataset_id=dataset_id,
                task_name=task_name, metric=metric, protocol="titan2025_zero_shot_prompt_ensemble",
                model_alias="TITAN vision-language", value=value, uncertainty=f"reported_error={error}",
                reference_url=TITAN_URL,
                source_locator=f"paper=TITAN|supplementary_table={table}|model=TITAN|protocol=zero-shot|metric={metric}",
                source_sha256=TITAN_SHA, reason="internal_private_cohort",
                notes="Institutional cohort is excluded from the public benchmark matrix.",
            )
    # Retrieval is a separate endpoint, never collapsed with classification.
    retrieval_metrics = ("top1_accuracy", "top3_accuracy", "majority_vote_at_3_accuracy", "top5_accuracy", "majority_vote_at_5_accuracy")
    for table, base_table in ((120, 22), (121, 23), (123, 25)):
        dataset_id, task_name, target, num_samples, _ = TITAN_TASKS[base_table]
        rows = titan_rows(table_block(text, table))
        if set(rows) != {"TITANV", "TITAN"} or any(len(v) != 5 for v in rows.values()):
            raise ValueError(f"unexpected TITAN retrieval table {table} rows")
        for alias, cells in rows.items():
            model_alias, model_id = titan_model(alias)
            for metric, (value, error) in zip(retrieval_metrics, cells):
                add_included(
                    included, source_id="titan2025", table=f"Supplementary Table {table}", dataset_id=dataset_id,
                    task_name=f"{task_name} slide retrieval", task_family="slide_retrieval", target=target,
                    sample_unit="slide", task_type="retrieval", num_samples=num_samples, endpoint="slide_retrieval",
                    metric=metric, protocol="titan2025_l2_slide_retrieval", model_alias=model_alias, model_id=model_id,
                    model_revision=TITAN_REVISION, value=value, uncertainty=f"reported_error={error}",
                    reference_url=TITAN_URL, source_locator=f"paper=TITAN|supplementary_table={table}|model={alias}|metric={metric}",
                    source_sha256=TITAN_SHA,
                )
    # Rare-Cancers-Public is the downloadable subset; the in-house and external
    # Kanagawa cohorts are separately retained in quarantine.
    rows = titan_rows(table_block(text, 119))
    for alias, cells in rows.items():
        model_alias, model_id = titan_model(alias)
        for metric, (value, error) in zip(retrieval_metrics, cells):
            add_included(
                included, source_id="titan2025", table="Supplementary Table 119", dataset_id="rare-cancers-public",
                task_name="Rare-Cancers-Public slide retrieval", task_family="slide_retrieval", target="oncotree_code_29_class",
                sample_unit="slide", task_type="retrieval", num_samples="not_reported_total", endpoint="slide_retrieval",
                metric=metric, protocol="titan2025_l2_slide_retrieval", model_alias=model_alias, model_id=model_id,
                model_revision=TITAN_REVISION, value=value, uncertainty=f"reported_error={error}", reference_url=TITAN_URL,
                source_locator=f"paper=TITAN|supplementary_table=119|model={alias}|metric={metric}", source_sha256=TITAN_SHA,
            )
    private_retrieval = {
        117: ("rare-cancers", "Rare-Cancers slide retrieval", retrieval_metrics),
        118: ("rare-cancers-external", "Rare-Cancers-External slide retrieval", retrieval_metrics + ("top10_accuracy", "majority_vote_at_10_accuracy")),
        124: ("renal-allograft", "Renal AMR slide retrieval", retrieval_metrics),
    }
    for table, (dataset_id, task_name, metrics) in private_retrieval.items():
        rows = titan_rows(table_block(text, table))
        for alias, cells in rows.items():
            for metric, (value, error) in zip(metrics, cells):
                add_quarantine(
                    quarantine, source_id="titan2025", table=f"Supplementary Table {table}", dataset_id=dataset_id,
                    task_name=task_name, metric=metric, protocol="titan2025_l2_slide_retrieval",
                    model_alias=titan_model(alias)[0], value=value, uncertainty=f"reported_error={error}",
                    reference_url=TITAN_URL, source_locator=f"paper=TITAN|supplementary_table={table}|model={alias}|metric={metric}",
                    source_sha256=TITAN_SHA, reason="internal_private_cohort",
                    notes="Non-downloadable institutional cohort is excluded from the public benchmark matrix.",
                )
    cross_metrics = ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mean_recall")
    for table, direction in ((126, "report_to_slide"), (127, "slide_to_report")):
        cells = titan_rows(table_block(text, table)).get("TITAN", [])
        if len(cells) != 5:
            raise ValueError(f"unexpected TITAN cross-modal retrieval table {table}")
        for metric, (value, error) in zip(cross_metrics, cells):
            add_included(
                included, source_id="titan2025", table=f"Supplementary Table {table}", dataset_id="tcga-slide-reports",
                task_name=f"TCGA Slide-Reports {direction.replace('_', ' ')}", task_family="cross_modal_retrieval",
                target=direction, sample_unit="slide_report_pair", task_type="retrieval", num_samples="not_reported_total",
                endpoint="cross_modal_retrieval", metric=metric, protocol=f"titan2025_{direction}_retrieval",
                model_alias="TITAN vision-language", model_id="titan-slide", model_revision=TITAN_REVISION,
                value=value, uncertainty=f"reported_error={error}", reference_url=TITAN_URL,
                source_locator=f"paper=TITAN|supplementary_table={table}|model=TITAN|metric={metric}", source_sha256=TITAN_SHA,
            )
    # OT108 is internal; preserve all matched headline protocols in quarantine.
    for table, metrics, protocol in (
        (24, full_metrics, "full_data_frozen"),
        (84, tuple(("balanced_accuracy", f"{k}shot_linear_probe") for k in (1, 2, 4, 8, 16, 32)), "few_shot"),
        (88, tuple(("balanced_accuracy", f"{k}shot_simpleshot") for k in (1, 2, 4, 8, 16, 32)), "few_shot"),
        (122, tuple((m, "l2_slide_retrieval") for m in retrieval_metrics), "retrieval"),
    ):
        rows = titan_rows(table_block(text, table))
        for alias, cells in rows.items():
            for (metric, evaluator), (value, error) in zip(metrics, cells):
                add_quarantine(
                    quarantine, source_id="titan2025", table=f"Supplementary Table {table}", dataset_id="ot108",
                    task_name="OT108 OncoTree code", metric=metric, protocol=f"titan2025_{evaluator}",
                    model_alias=titan_model(alias)[0], value=value, uncertainty=f"reported_error={error}",
                    reference_url=TITAN_URL, source_locator=f"paper=TITAN|supplementary_table={table}|model={alias}|protocol={protocol}|metric={metric}",
                    source_sha256=TITAN_SHA, reason="internal_private_cohort",
                    notes="OT108 is an internal MGB OncoTree benchmark, distinct from public TCGA-OT.",
                )
    zero_internal = titan_rows(table_block(text, 99)).get("TITAN", [])
    for metric, (value, error) in zip(("balanced_accuracy", "weighted_f1", "auroc"), zero_internal):
        add_quarantine(
            quarantine, source_id="titan2025", table="Supplementary Table 99", dataset_id="ot108",
            task_name="OT108 OncoTree code", metric=metric, protocol="titan2025_zero_shot_prompt_ensemble",
            model_alias="TITAN vision-language", value=value, uncertainty=f"reported_error={error}",
            reference_url=TITAN_URL, source_locator=f"paper=TITAN|supplementary_table=99|model=TITAN|metric={metric}",
            source_sha256=TITAN_SHA, reason="internal_private_cohort",
            notes="OT108 is an internal MGB OncoTree benchmark, distinct from public TCGA-OT.",
        )
    # Fine-tuned rows from public headline datasets are explicitly excluded.
    for table, base_table in ((78, 22), (79, 23), (81, 25)):
        dataset_id, task_name, _, _, _ = TITAN_TASKS[base_table]
        rows = titan_rows(table_block(text, table), finetuned=True)
        if set(rows) != {"TITANV", "TITAN"} or any(len(v) != 3 for v in rows.values()):
            raise ValueError(f"unexpected TITAN fine-tuned table {table} rows")
        for alias, cells in rows.items():
            for metric, (value, error) in zip(("balanced_accuracy", "weighted_f1", "auroc"), cells):
                add_quarantine(
                    quarantine, source_id="titan2025", table=f"Supplementary Table {table}", dataset_id=dataset_id,
                    task_name=task_name, metric=metric, protocol="task_specific_finetuning",
                    model_alias=f"{titan_model(alias)[0]} finetuned", value=value,
                    uncertainty=f"reported_error={error}", reference_url=TITAN_URL,
                    source_locator=f"paper=TITAN|supplementary_table={table}|model={alias} finetuned|metric={metric}",
                    source_sha256=TITAN_SHA, reason="task_specific_finetuning",
                    notes="Task-specific end-to-end fine-tuning is not mixed into the frozen-representation matrix.",
                )
    for alias, value, entropy in (("TITANV", "0.008", "0.90"), ("TITAN", "0.008", "0.82")):
        for metric, cell in (("ece", value), ("entropy", entropy)):
            add_quarantine(
                quarantine, source_id="titan2025", table="Supplementary Table 65", dataset_id="aggregate",
                task_name="Four-cohort calibration aggregate", metric=metric, protocol="aggregate_calibration",
                model_alias=titan_model(alias)[0], value=cell, uncertainty="not_reported", reference_url=TITAN_URL,
                source_locator=f"paper=TITAN|supplementary_table=65|model={alias}|metric={metric}", source_sha256=TITAN_SHA,
                reason="aggregate_not_leaf_task_and_mixes_internal_cohort",
                notes="Aggregate mixes public TCGA-UT-8K, TCGA-OT, EBRAINS with internal OT108.",
            )


def write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--virchow-text", type=Path, required=True)
    parser.add_argument("--gigapath-text", type=Path, required=True)
    parser.add_argument("--titan-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine-output", type=Path, required=True)
    args = parser.parse_args()
    included: list[dict[str, str]] = []
    quarantine: list[dict[str, str]] = []
    extract_virchow(args.virchow_text.read_text(encoding="utf-8"), included, quarantine)
    extract_gigapath(args.gigapath_text.read_text(encoding="utf-8"), quarantine)
    extract_titan(args.titan_text.read_text(encoding="utf-8"), included, quarantine)
    if len(included) != 737:
        raise ValueError(f"expected 737 included cells, found {len(included)}")
    if len(quarantine) != 346:
        raise ValueError(f"expected 346 quarantine cells, found {len(quarantine)}")
    write_csv(args.output, included, FIELDS)
    write_csv(args.quarantine_output, quarantine, QUARANTINE_FIELDS)
    print(f"wrote {len(included)} included and {len(quarantine)} quarantine rows")


if __name__ == "__main__":
    main()
