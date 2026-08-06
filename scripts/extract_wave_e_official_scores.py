#!/usr/bin/env python3
"""Extract protocol-preserving Wave E evidence for CONCH/Phikon/CTransPath.

This evidence-layer script intentionally does not rebuild the shared registry.  It
pins first-party papers/repositories, retains uncertainty, and quarantines private,
fine-tuned, aggregate, ambiguous-best-of-many, and secondary-only observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
from pathlib import Path


CONCH_SHA = "4940cf23c1f341791fd16db84e6f22a83d90d1c67657c26cb118ad8fcbbef457"
TITAN_SHA = "26321e4018bec7b80f2fe7ea7cc497139c83b44fb60df5128417623ad1f71a70"
PHIKON_V2_SHA = "52e26d832d27a6d51ce2b68da462f6a6ad851236fc1efafb9b1258359fc6b633"
PHIKON_SUPP_SHA = "dd73798e6b34e317a205bec036318388d60c33da3af4326690f51ca357b96faa"

CONCH_URL = "https://www.nature.com/articles/s41591-024-02856-4"
TITAN_URL = "https://www.nature.com/articles/s41591-025-03982-3"
PHIKON_URL = "https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v3"
PHIKON_V2_URL = "https://arxiv.org/abs/2409.09173v1"
CTRANSPATH_URL = "https://doi.org/10.1016/j.media.2022.102559"

FIELDS = (
    "source_scope", "source_table", "source_row", "model_alias", "model_id",
    "model_revision", "suite_id", "task_label", "evaluation_id", "dedup_key",
    "metric", "value", "value_unit", "uncertainty", "level", "magnification",
    "embedding_recipe", "downstream_protocol", "cohort_access", "reference_url",
    "source_locator", "source_revision", "source_sha256", "inclusion_status",
    "inclusion_reason",
)


def _text(path: Path, digest: str) -> str:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(f"unexpected SHA-256 for {path}: {actual}")
    return subprocess.run(
        ["pdftotext", "-layout", str(path), "-"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(**kwargs: str) -> dict[str, str]:
    missing = set(FIELDS) - set(kwargs)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    return {key: kwargs[key] for key in FIELDS}


CELL = re.compile(r"(0?\.\d+|1\.0+)\s*\((0?\.\d+|1\.0+),\s*(0?\.\d+|1\.0+)\)")
PM_CELL = re.compile(r"(0?\.\d+|1\.0+)±(0?\.\d+)")


def _preceding_blocks(text: str, caption_pattern: str) -> dict[int, tuple[str, str]]:
    matches = list(re.finditer(caption_pattern, text))
    out: dict[int, tuple[str, str]] = {}
    previous_end = 0
    for match in matches:
        number = int(match.group(1))
        caption_tail = re.sub(r"\s+", " ", text[match.start():match.start() + 420])
        caption = caption_tail.split(". The best", 1)[0].split(". Best performing", 1)[0]
        out[number] = (text[previous_end:match.start()], caption)
        previous_end = match.end()
    return out


def _metrics_from_header(block: str, count: int) -> list[str]:
    if count == 4:
        return ["recall_at_1", "recall_at_5", "recall_at_10", "mean_recall"]
    if count == 2:
        return ["meteor", "rouge"]
    if "Dice" in block[-2000:]:
        return ["dice", "precision", "recall"]
    tail = block[-2200:]
    if "Quadratic weighted" in tail:
        return ["quadratic_weighted_kappa", "balanced_accuracy", "weighted_f1"]
    if "Cohen’s κ" in tail or "Cohen's κ" in tail:
        return ["cohen_kappa", "balanced_accuracy", "weighted_f1"]
    return ["balanced_accuracy", "weighted_f1", "auroc"]


def conch_rows(pdf: Path) -> list[dict[str, str]]:
    text = _text(pdf, CONCH_SHA)
    blocks = _preceding_blocks(text, r"Supplementary Data Table (\d+):")
    if set(range(1, 32)) - set(blocks):
        raise ValueError("CONCH supplement table inventory changed")
    rows: list[dict[str, str]] = []
    for table in range(1, 32):
        block, caption = blocks[table]
        model_lines = [line for line in block.splitlines() if re.match(r"\s*CONCH\s+", line)]
        if table == 31:
            labels = ("100pct", "10pct", "1pct")
        else:
            labels = tuple(str(i) for i in range(len(model_lines)))
        for label, line in zip(labels, model_lines, strict=True):
            cells = CELL.findall(line)
            metrics = _metrics_from_header(block, len(cells))
            if table <= 21:
                status, access, reason = "canonical_candidate", "public", "Public leaf result from the model's primary paper."
            elif table <= 25:
                status, access, reason = "private_internal_excluded", "private", "Source A/B are unreleased author cohorts."
            elif table <= 29:
                status, access, reason = "canonical_candidate", "public", "Public leaf result from the model's primary paper."
            else:
                status, access, reason = "fine_tuned_excluded", "private" if table == 30 else "public", "Task-specific end-to-end fine-tuning is not a frozen-foundation-model observation."
            if table <= 14:
                protocol = "zero-shot classification; prompt ensemble" if table <= 7 else "zero-shot classification; single prompt"
            elif table <= 20:
                protocol = "frozen CONCH features; supervised linear evaluation"
            elif table == 21:
                protocol = "zero-shot classification"
            elif table <= 27:
                protocol = "zero-shot cross-modal retrieval"
            elif table <= 29:
                protocol = "zero-shot segmentation"
            elif table == 30:
                protocol = "captioning; task-specific fine-tuning"
            else:
                protocol = f"end-to-end fine-tuning; {label.replace('pct', '%')} training labels"
            for metric, (value, low, high) in zip(metrics, cells, strict=True):
                eval_id = f"conch.natmed2024.t{table}.{label}.{protocol.split(';')[0].replace(' ', '_')}.{metric}"
                rows.append(_row(
                    source_scope="official_primary_paper", source_table=f"Supplementary Data Table {table}",
                    source_row=f"CONCH/{label}/{metric}", model_alias="CONCH", model_id="conch",
                    model_revision="CONCH v1 ViT-B/16 CoCa", suite_id="conch_primary",
                    task_label=caption, evaluation_id=eval_id,
                    dedup_key=f"{caption.lower().replace(' ', '_')}.{metric}", metric=metric,
                    value=value, value_unit="fraction", uncertainty=f"95% CI [{low}, {high}]",
                    level="tile" if table not in {15, 16, 17, 20, 21, 26, 27, 31} else "slide",
                    magnification="source-defined", embedding_recipe="CONCH image/text embedding",
                    downstream_protocol=protocol, cohort_access=access, reference_url=CONCH_URL,
                    source_locator=f"Supplementary Information|Table {table}|CONCH row",
                    source_revision="Nature Medicine 2024 supplementary PDF", source_sha256=CONCH_SHA,
                    inclusion_status=status, inclusion_reason=reason,
                ))
    if len(rows) != 104:
        raise ValueError(f"expected 104 CONCH cells, got {len(rows)}")
    return rows


def _titan_metrics(block: str) -> list[str]:
    position = block.find("Mean pool (CONCHv1.5)")
    header_start = block.rfind("Encoder", 0, position)
    header = block[header_start:position] if header_start >= 0 else block[max(0, position - 600):position]
    if "Cohen’s κ" in header or "Cohen's κ" in header:
        return ["quadratic_weighted_kappa", "balanced_accuracy"] * 3
    if "AUROC" in header:
        return ["auroc", "balanced_accuracy", "balanced_accuracy", "weighted_f1", "balanced_accuracy", "weighted_f1"]
    return ["balanced_accuracy", "weighted_f1"] * 3


def conch15_rows(pdf: Path) -> list[dict[str, str]]:
    text = _text(pdf, TITAN_SHA)
    blocks = _preceding_blocks(text, r"Supplementary Table (\d+):")
    rows: list[dict[str, str]] = []
    public_tables = {22, 23, 25, 26, 27, 28, 29, 30, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                     44, 45, 46, 47, 48, 49, 50, 51, 52}
    for table in range(22, 64):
        block, caption = blocks[table]
        lines = [line for line in block.splitlines() if "Mean pool (CONCHv1.5)" in line]
        for row_index, line in enumerate(lines):
            cells = PM_CELL.findall(line)
            if len(cells) != 6:
                raise ValueError(f"TITAN table {table} expected six CONCHv1.5 cells: {line}")
            cohort = re.sub(r".*Mean pool \(CONCHv1\.5\)\s*", "", line)
            cohort = re.split(r"\s+0?\.\d+±", cohort, maxsplit=1)[0].strip() or "caption-defined"
            public = table in public_tables
            metrics = _titan_metrics(block)
            for column, (metric, (value, sd)) in enumerate(zip(metrics, cells, strict=True)):
                probe = ("logistic_regression", "logistic_regression", "simpleshot", "simpleshot", "knn", "knn")[column]
                rows.append(_row(
                    source_scope="official_primary_paper", source_table=f"Supplementary Table {table}",
                    source_row=f"Mean pool (CONCHv1.5)/{cohort}/{probe}/{metric}",
                    model_alias="CONCHv1.5", model_id="conch-1.5",
                    model_revision="CONCHv1.5 ViT-L/16 packaged with TITAN",
                    suite_id="titan_patch_encoder", task_label=caption,
                    evaluation_id=f"conch15.titan2025.t{table}.{row_index}.{probe}.{metric}",
                    dedup_key=f"{caption.lower().replace(' ', '_')}.{cohort}.{metric}", metric=metric,
                    value=value, value_unit="fraction", uncertainty=f"reported ± {sd}", level="slide",
                    magnification="20x patches; slide mean pooling", embedding_recipe="mean-pooled frozen CONCHv1.5 patch embeddings",
                    downstream_protocol=probe, cohort_access="public" if public else "private_internal",
                    reference_url=TITAN_URL, source_locator=f"Supplementary Information|Table {table}|Mean pool (CONCHv1.5)",
                    source_revision="Nature Medicine 2025 supplementary PDF", source_sha256=TITAN_SHA,
                    inclusion_status="canonical_candidate" if public else "private_internal_excluded",
                    inclusion_reason="Public leaf protocol cell." if public else "Internal/private cohort is not eligible for the public matrix.",
                ))

    for table in range(78, 82):
        block, caption = blocks[table]
        line = next(line for line in block.splitlines() if "ABMIL (CONCHv1.5) (finetuned)" in line)
        for metric, (value, sd) in zip(("balanced_accuracy", "weighted_f1", "auroc"), PM_CELL.findall(line), strict=True):
            rows.append(_row(
                source_scope="official_primary_paper", source_table=f"Supplementary Table {table}",
                source_row=f"ABMIL (CONCHv1.5) finetuned/{metric}", model_alias="CONCHv1.5 finetuned",
                model_id="conch-1.5", model_revision="CONCHv1.5 task-finetuned ablation",
                suite_id="titan_patch_encoder", task_label=caption,
                evaluation_id=f"conch15.titan2025.t{table}.finetuned.{metric}",
                dedup_key=f"{caption.lower().replace(' ', '_')}.{metric}", metric=metric, value=value,
                value_unit="fraction", uncertainty=f"reported ± {sd}", level="slide", magnification="20x patches",
                embedding_recipe="task-finetuned CONCHv1.5 with ABMIL", downstream_protocol="end-to-end task fine-tuning",
                cohort_access="private_internal" if table == 80 else "public", reference_url=TITAN_URL,
                source_locator=f"Supplementary Information|Table {table}|ABMIL (CONCHv1.5) (finetuned)",
                source_revision="Nature Medicine 2025 supplementary PDF", source_sha256=TITAN_SHA,
                inclusion_status="fine_tuned_excluded", inclusion_reason="Task-specific end-to-end fine-tuning changes the foundation checkpoint.",
            ))

    for table in range(82, 90):
        block, caption = blocks[table]
        lines = [line for line in block.splitlines() if "(CONCHv1.5)" in line]
        public = table in {82, 83, 85, 86, 87, 89}
        for line in lines:
            recipe = "abmil" if "ABMIL" in line else ("simpleshot" if table >= 86 else "linear_probe")
            for shot, (value, sd) in zip((1, 2, 4, 8, 16, 32), PM_CELL.findall(line), strict=True):
                rows.append(_row(
                    source_scope="official_primary_paper", source_table=f"Supplementary Table {table}",
                    source_row=f"{recipe}/k={shot}", model_alias="CONCHv1.5", model_id="conch-1.5",
                    model_revision="CONCHv1.5 ViT-L/16 packaged with TITAN", suite_id="titan_patch_encoder",
                    task_label=caption, evaluation_id=f"conch15.titan2025.t{table}.{recipe}.k{shot}.balanced_accuracy",
                    dedup_key=f"{caption.lower().replace(' ', '_')}.k{shot}.balanced_accuracy", metric="balanced_accuracy",
                    value=value, value_unit="fraction", uncertainty=f"reported ± {sd}", level="slide",
                    magnification="20x patches", embedding_recipe="frozen CONCHv1.5 patch embeddings",
                    downstream_protocol=f"{recipe}; {shot}-shot per class", cohort_access="public" if public else "private_internal",
                    reference_url=TITAN_URL, source_locator=f"Supplementary Information|Table {table}|CONCHv1.5 row",
                    source_revision="Nature Medicine 2025 supplementary PDF", source_sha256=TITAN_SHA,
                    inclusion_status="canonical_candidate" if public else "private_internal_excluded",
                    inclusion_reason="Public few-shot leaf cell." if public else "OT108 is explicitly internal.",
                ))

    for table in range(119, 125):
        block, caption = blocks[table]
        lines = [line for line in block.splitlines() if "Mean pool (CONCHv1.5)" in line]
        if not lines:
            continue
        public = table in {119, 120, 121, 123}
        for metric, (value, sd) in zip(("top1_accuracy", "top3_accuracy", "majority_vote_at3_accuracy", "top5_accuracy", "majority_vote_at5_accuracy"), PM_CELL.findall(lines[0]), strict=True):
            rows.append(_row(
                source_scope="official_primary_paper", source_table=f"Supplementary Table {table}",
                source_row=f"Mean pool (CONCHv1.5)/{metric}", model_alias="CONCHv1.5", model_id="conch-1.5",
                model_revision="CONCHv1.5 ViT-L/16 packaged with TITAN", suite_id="titan_patch_encoder",
                task_label=caption, evaluation_id=f"conch15.titan2025.t{table}.retrieval.{metric}",
                dedup_key=f"{caption.lower().replace(' ', '_')}.{metric}", metric=metric, value=value,
                value_unit="fraction", uncertainty=f"reported ± {sd}", level="slide", magnification="20x patches",
                embedding_recipe="mean-pooled frozen CONCHv1.5 patch embeddings", downstream_protocol="cosine slide retrieval",
                cohort_access="public" if public else "private_internal", reference_url=TITAN_URL,
                source_locator=f"Supplementary Information|Table {table}|Mean pool (CONCHv1.5)",
                source_revision="Nature Medicine 2025 supplementary PDF", source_sha256=TITAN_SHA,
                inclusion_status="canonical_candidate" if public else "private_internal_excluded",
                inclusion_reason="Public retrieval leaf cell." if public else "OT108/renal allograft is internal.",
            ))
    if len(rows) != 433:
        raise ValueError(f"expected 433 CONCHv1.5 cells, got {len(rows)}")
    return rows


PHIKON_TASKS = (
    ("ER", "BCNB"), ("PR", "BCNB"), ("HER2", "BCNB"), ("HER2", "HEROHE"),
    ("IDH1", "EBRAINS"), ("ISUP", "PANDA"), ("metastasis", "Camelyon16"),
    ("MSI", "Cy1"), ("MSI", "PAIP"), ("RCC subtype", "DHMC"),
)


def phikon_family_rows(v2_pdf: Path, supplement_pdf: Path) -> list[dict[str, str]]:
    v2 = _text(v2_pdf, PHIKON_V2_SHA)
    supp = _text(supplement_pdf, PHIKON_SUPP_SHA)
    for anchor in ("Phikon-v2", "Table 2", "Table 4"):
        if anchor not in v2:
            raise ValueError(f"missing Phikon-v2 anchor: {anchor}")
    for anchor in ("Table F1", "Table F2", "Table G3", "Table G4"):
        if anchor not in supp:
            raise ValueError(f"missing Phikon supplement anchor: {anchor}")
    rows: list[dict[str, str]] = []

    v2_values = (0.856, 0.804, 0.669, 0.770, 0.842, 0.936, 0.997, 0.882, 0.991, 0.989)
    v1_values = (0.803, 0.780, 0.699, 0.685, 0.851, 0.938, 1.000, 0.830, 0.977, 0.986)
    for model_id, alias, revision, values in (
        ("phikon-v2", "Phikon-v2", "Phikon-v2 arXiv v1 / official checkpoint", v2_values),
        ("phikon", "Phikon", "Phikon ViT-B/16 iBOT official checkpoint", v1_values),
    ):
        for (task, cohort), value in zip(PHIKON_TASKS, values, strict=True):
            public = cohort != "Cy1"
            rows.append(_row(
                source_scope="official_primary_paper", source_table="Phikon-v2 Table 2",
                source_row=f"{alias}/{task}/{cohort}", model_alias=alias, model_id=model_id,
                model_revision=revision, suite_id="phikon_v2_external", task_label=f"{task} on {cohort}",
                evaluation_id=f"{model_id}.phikonv2.t2.{task.lower().replace(' ', '_')}.{cohort.lower()}.auroc",
                dedup_key=f"{task.lower().replace(' ', '_')}.{cohort.lower()}.auroc", metric="auroc",
                value=f"{value:.3f}", value_unit="fraction", uncertainty="median over 10,000 bootstrap repetitions",
                level="slide", magnification="0.5 µm/px except CTransPath comparator at 1.0 µm/px",
                embedding_recipe="frozen tile features with ABMIL", downstream_protocol="external-cohort ABMIL evaluation",
                cohort_access="public" if public else "private", reference_url=PHIKON_V2_URL,
                source_locator=f"arXiv 2409.09173v1|Table 2|{alias}", source_revision="arXiv v1",
                source_sha256=PHIKON_V2_SHA, inclusion_status="canonical_candidate" if public else "private_internal_excluded",
                inclusion_reason="Primary author-reported public external-cohort leaf cell." if public else "Cy1 is explicitly described as a private French cohort.",
            ))
        rows.append(_row(
            source_scope="official_primary_paper", source_table="Phikon-v2 Table 4",
            source_row=f"{alias}/MSI/NGX1", model_alias=alias, model_id=model_id, model_revision=revision,
            suite_id="phikon_v2_external", task_label="MSI on NGX1",
            evaluation_id=f"{model_id}.phikonv2.t4.msi.ngx1.auroc", dedup_key="msi.ngx1.auroc",
            metric="auroc", value="0.919" if model_id == "phikon-v2" else "0.921", value_unit="fraction",
            uncertainty="median over 10,000 bootstrap repetitions", level="slide", magnification="0.5 µm/px",
            embedding_recipe="frozen tile features with ABMIL", downstream_protocol="external-cohort ABMIL evaluation",
            cohort_access="private", reference_url=PHIKON_V2_URL, source_locator=f"arXiv 2409.09173v1|Table 4|{alias}",
            source_revision="arXiv v1", source_sha256=PHIKON_V2_SHA, inclusion_status="private_internal_excluded",
            inclusion_reason="NGX1 is one of the two private MSI validation cohorts; Table 4 Cy1 and PAIP repeats are not duplicated.",
        ))

    slide_tasks = ("Camelyon16 Meta", "TCGA-BRCA Hist", "TCGA-BRCA HRD", "TCGA-BRCA Mol", "TCGA-BRCA OS",
                   "TCGA-CRC MSI", "TCGA-COAD OS", "TCGA-NSCLC CType", "TCGA-LUAD OS", "TCGA-LUSC OS",
                   "TCGA-OV HRD", "TCGA-RCC CType", "TCGA-STAD MSI", "TCGA-PAAD OS")
    slide_values = ((.929,.033),(.962,.015),(.793,.024),(.817,.022),(.647,.057),(.910,.022),(.628,.127),
                    (.977,.013),(.538,.045),(.622,.029),(.742,.086),(.995,.002),(.899,.039),(.553,.044))
    for task, (value, sd) in zip(slide_tasks, slide_values, strict=True):
        metric = "c_index" if task.endswith(" OS") else "auroc"
        rows.append(_row(
            source_scope="official_primary_paper", source_table="Supplementary Table F1", source_row=f"iBOT[ViT-B] PanCancer/{task}",
            model_alias="Phikon", model_id="phikon", model_revision="Phikon ViT-B/16 iBOT official checkpoint",
            suite_id="phikon_primary", task_label=task, evaluation_id=f"phikon.medrxiv2024.f1.{task.lower().replace(' ', '_')}.{metric}",
            dedup_key=f"{task.lower().replace(' ', '_')}.{metric}", metric=metric, value=f"{value:.3f}",
            value_unit="fraction", uncertainty=f"5-fold nested CV SD {sd:.3f}", level="slide", magnification="20x; 0.5 µm/px",
            embedding_recipe="frozen Phikon tile features", downstream_protocol="ABMIL; 5x5 nested CV",
            cohort_access="public", reference_url=PHIKON_URL, source_locator="MedRxiv supplement|Table F1|iBOT[ViT-B] PanCancer",
            source_revision="MedRxiv v3 paper; official supplement asset", source_sha256=PHIKON_SUPP_SHA,
            inclusion_status="canonical_candidate", inclusion_reason="Protocol-specific ABMIL public leaf cell.",
        ))
    rows.append(_row(
        source_scope="official_primary_paper", source_table="Supplementary Table F2", source_row="iBOT[ViT-B] PanCancer/TCGA-CRC to PAIP",
        model_alias="Phikon", model_id="phikon", model_revision="Phikon ViT-B/16 iBOT official checkpoint",
        suite_id="phikon_primary", task_label="MSI: TCGA-CRC train to PAIP external", evaluation_id="phikon.medrxiv2024.f2.tcga_crc_to_paip.abmil.auroc",
        dedup_key="msi.paip.auroc", metric="auroc", value="0.947", value_unit="fraction", uncertainty="95% CI [0.894, 1.000]",
        level="slide", magnification="20x; 0.5 µm/px", embedding_recipe="frozen Phikon tile features",
        downstream_protocol="ABMIL trained on TCGA-CRC; PAIP external validation", cohort_access="public",
        reference_url=PHIKON_URL, source_locator="MedRxiv supplement|Table F2|iBOT[ViT-B] PanCancer",
        source_revision="MedRxiv v3 paper; official supplement asset", source_sha256=PHIKON_SUPP_SHA,
        inclusion_status="canonical_candidate", inclusion_reason="Protocol-specific external validation leaf cell.",
    ))
    patch_names = ("Adipose", "Debris", "Lymphocytes", "Mucus", "Muscle", "Normal", "Stroma", "Tumor", "All", "Camelyon17 metastases")
    patch_values = (1.000,1.000,.998,1.000,.958,1.000,.981,1.000,.992,.995)
    for task, value in zip(patch_names, patch_values, strict=True):
        rows.append(_row(
            source_scope="official_primary_paper", source_table="Supplementary Table G3", source_row=f"iBOT[ViT-B] PanCancer/{task}",
            model_alias="Phikon", model_id="phikon", model_revision="Phikon ViT-B/16 iBOT official checkpoint",
            suite_id="phikon_primary", task_label=task, evaluation_id=f"phikon.medrxiv2024.g3.{task.lower().replace(' ', '_')}.auroc",
            dedup_key=f"{task.lower().replace(' ', '_')}.auroc", metric="auroc", value=f"{value:.3f}", value_unit="fraction",
            uncertainty="95% bootstrap CI reported for aggregate endpoints only" if task not in {"All", "Camelyon17 metastases"} else ("95% CI [0.991, 0.993]" if task == "All" else "95% CI [0.994, 0.996]"),
            level="tile", magnification="20x; 0.5 µm/px", embedding_recipe="frozen Phikon tile features",
            downstream_protocol="ensemble of 30 linear classifiers; full training set", cohort_access="public",
            reference_url=PHIKON_URL, source_locator="MedRxiv supplement|Table G3|iBOT[ViT-B] PanCancer",
            source_revision="MedRxiv v3 paper; official supplement asset", source_sha256=PHIKON_SUPP_SHA,
            inclusion_status="canonical_candidate", inclusion_reason="Public patch-level linear-evaluation leaf cell.",
        ))
    return rows


def ctranspath_rows(phikon_v2_pdf: Path, supplement_pdf: Path, conch_pdf: Path) -> list[dict[str, str]]:
    _text(phikon_v2_pdf, PHIKON_V2_SHA)
    _text(supplement_pdf, PHIKON_SUPP_SHA)
    conch = _text(conch_pdf, CONCH_SHA)
    rows: list[dict[str, str]] = []
    v2_values = (.800,.788,.678,.723,.895,.923,.896,.838,.977,.996)
    for (task, cohort), value in zip(PHIKON_TASKS, v2_values, strict=True):
        rows.append(_row(
            source_scope="official_secondary_comparator", source_table="Phikon-v2 Table 2", source_row=f"CTransPath/{task}/{cohort}",
            model_alias="CTransPath", model_id="ctranspath", model_revision="official GitHub checkpoint; exact file revision not reported",
            suite_id="phikon_v2_external", task_label=f"{task} on {cohort}", evaluation_id=f"ctranspath.phikonv2.t2.{task.lower().replace(' ', '_')}.{cohort.lower()}.auroc",
            dedup_key=f"{task.lower().replace(' ', '_')}.{cohort.lower()}.auroc", metric="auroc", value=f"{value:.3f}", value_unit="fraction",
            uncertainty="median over 10,000 bootstrap repetitions", level="slide", magnification="1.0 µm/px",
            embedding_recipe="CTransPath mean-token features with ABMIL", downstream_protocol="external-cohort ABMIL evaluation",
            cohort_access="public", reference_url=PHIKON_V2_URL, source_locator="arXiv 2409.09173v1|Table 2|CTransPath",
            source_revision="arXiv v1", source_sha256=PHIKON_V2_SHA, inclusion_status="secondary_only_excluded",
            inclusion_reason="Comparator value was not reported by the CTransPath authors; PAIP is additionally flagged for pretraining overlap.",
        ))
    slide_tasks = ("Camelyon16 Meta", "TCGA-BRCA Hist", "TCGA-BRCA HRD", "TCGA-BRCA Mol", "TCGA-BRCA OS", "TCGA-CRC MSI", "TCGA-COAD OS", "TCGA-NSCLC CType", "TCGA-LUAD OS", "TCGA-LUSC OS", "TCGA-OV HRD", "TCGA-RCC CType", "TCGA-STAD MSI", "TCGA-PAAD OS")
    slide_values = (.939,.954,.768,.808,.650,.881,.601,.973,.581,.605,.685,.987,.832,.570)
    for task, value in zip(slide_tasks, slide_values, strict=True):
        rows.append(_row(
            source_scope="official_secondary_comparator", source_table="Phikon Supplementary Table F1", source_row=f"CTransPath/{task}",
            model_alias="CTransPath", model_id="ctranspath", model_revision="official GitHub checkpoint; exact file revision not reported",
            suite_id="phikon_primary", task_label=task, evaluation_id=f"ctranspath.phikon2024.f1.{task.lower().replace(' ', '_')}",
            dedup_key=task.lower().replace(' ', '_'), metric="c_index" if task.endswith(" OS") else "auroc", value=f"{value:.3f}", value_unit="fraction",
            uncertainty="5-fold nested CV dispersion retained in source snapshot", level="slide", magnification="20x / source checkpoint recipe",
            embedding_recipe="CTransPath frozen features", downstream_protocol="ABMIL; 5x5 nested CV", cohort_access="public",
            reference_url=PHIKON_URL, source_locator="MedRxiv supplement|Table F1|CTransPath", source_revision="official supplement asset",
            source_sha256=PHIKON_SUPP_SHA, inclusion_status="secondary_only_excluded", inclusion_reason="Secondary comparator evidence only.",
        ))
    patch_names = ("Adipose", "Debris", "Lymphocytes", "Mucus", "Muscle", "Normal", "Stroma", "Tumor", "All", "Camelyon17 metastases")
    patch_values = (1.000,.971,1.000,.999,.969,.999,.939,.997,.984,.983)
    for task, value in zip(patch_names, patch_values, strict=True):
        rows.append(_row(
            source_scope="official_secondary_comparator", source_table="Phikon Supplementary Table G4", source_row=f"CTransPath/{task}",
            model_alias="CTransPath", model_id="ctranspath", model_revision="official GitHub checkpoint; exact file revision not reported",
            suite_id="phikon_primary", task_label=task, evaluation_id=f"ctranspath.phikon2024.g4.{task.lower().replace(' ', '_')}.auroc",
            dedup_key=f"{task.lower().replace(' ', '_')}.auroc", metric="auroc", value=f"{value:.3f}", value_unit="fraction",
            uncertainty="95% bootstrap CI reported for aggregate endpoints only", level="tile", magnification="20x / 1.0 µm/px extraction",
            embedding_recipe="CTransPath frozen features", downstream_protocol="ensemble of 30 linear classifiers; full training set",
            cohort_access="public", reference_url=PHIKON_URL, source_locator="MedRxiv supplement|Table G4|CTransPath",
            source_revision="official supplement asset", source_sha256=PHIKON_SUPP_SHA, inclusion_status="secondary_only_excluded",
            inclusion_reason="Secondary comparator evidence only; original publisher tables were not accessible as a first-party machine-readable artifact.",
        ))
    rows.append(_row(
        source_scope="official_secondary_comparator", source_table="Phikon Supplementary Table F2",
        source_row="CTransPath/TCGA-CRC to PAIP", model_alias="CTransPath", model_id="ctranspath",
        model_revision="official GitHub checkpoint; exact file revision not reported", suite_id="phikon_primary",
        task_label="MSI: TCGA-CRC train to PAIP external", evaluation_id="ctranspath.phikon2024.f2.tcga_crc_to_paip.abmil.auroc",
        dedup_key="msi.paip.auroc", metric="auroc", value="0.884", value_unit="fraction",
        uncertainty="95% CI [0.782, 1.000]", level="slide", magnification="20x / 1.0 µm/px extraction",
        embedding_recipe="CTransPath frozen features", downstream_protocol="ABMIL trained on TCGA-CRC; PAIP external validation",
        cohort_access="public", reference_url=PHIKON_URL, source_locator="MedRxiv supplement|Table F2|CTransPath",
        source_revision="official supplement asset", source_sha256=PHIKON_SUPP_SHA,
        inclusion_status="secondary_only_excluded", inclusion_reason="Secondary comparator evidence only.",
    ))
    conch_blocks = _preceding_blocks(conch, r"Supplementary Data Table (\d+):")
    for table in (18, 19, 31):
        block, caption = conch_blocks[table]
        lines = [line for line in block.splitlines() if re.match(r"\s*CTransPath\s+", line)]
        labels = ("100pct", "10pct", "1pct") if table == 31 else tuple(str(i) for i in range(len(lines)))
        for label, line in zip(labels, lines, strict=True):
            cells = CELL.findall(line)
            metrics = _metrics_from_header(block, len(cells))
            for metric, (value, low, high) in zip(metrics, cells, strict=True):
                rows.append(_row(
                    source_scope="official_secondary_comparator", source_table=f"CONCH Supplementary Data Table {table}",
                    source_row=f"CTransPath/{label}/{metric}", model_alias="CTransPath", model_id="ctranspath",
                    model_revision="CTransPath comparator checkpoint; exact file revision not reported", suite_id="conch_primary",
                    task_label=caption, evaluation_id=f"ctranspath.conch2024.t{table}.{label}.{metric}",
                    dedup_key=f"{caption.lower().replace(' ', '_')}.{metric}", metric=metric, value=value,
                    value_unit="fraction", uncertainty=f"95% CI [{low}, {high}]", level="tile" if table in {18, 19} else "slide",
                    magnification="source-defined", embedding_recipe="CTransPath frozen features" if table != 31 else "end-to-end task-finetuned CTransPath",
                    downstream_protocol="supervised frozen-feature evaluation" if table != 31 else f"end-to-end fine-tuning; {label.replace('pct', '%')} labels",
                    cohort_access="public", reference_url=CONCH_URL,
                    source_locator=f"CONCH Supplementary Information|Table {table}|CTransPath",
                    source_revision="Nature Medicine 2024 supplementary PDF", source_sha256=CONCH_SHA,
                    inclusion_status="secondary_only_excluded",
                    inclusion_reason="Secondary comparator evidence only; Table 31 is additionally task-finetuned." if table == 31 else "Secondary comparator evidence only.",
                ))
    if len(rows) != 50:
        raise ValueError(f"expected 50 quarantined CTransPath comparator cells, got {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conch-pdf", type=Path, required=True)
    parser.add_argument("--titan-pdf", type=Path, required=True)
    parser.add_argument("--phikon-v2-pdf", type=Path, required=True)
    parser.add_argument("--phikon-supp-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("source_data"))
    args = parser.parse_args()
    _write(args.output_dir / "conch_official_scores_2024.csv", conch_rows(args.conch_pdf))
    _write(args.output_dir / "conch15_titan_official_scores_2025.csv", conch15_rows(args.titan_pdf))
    _write(args.output_dir / "phikon_family_official_scores_2023_2024.csv", phikon_family_rows(args.phikon_v2_pdf, args.phikon_supp_pdf))
    _write(args.output_dir / "ctranspath_official_evidence_2022_2024.csv", ctranspath_rows(args.phikon_v2_pdf, args.phikon_supp_pdf, args.conch_pdf))


if __name__ == "__main__":
    main()
