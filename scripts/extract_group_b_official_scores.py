#!/usr/bin/env python3
"""Extract official GenBio-PathFM, Midnight, and OpenMidnight score evidence.

This is deliberately an evidence-layer extractor.  It does not mutate the shared
registry: the paper/report protocols differ from the current suite leaderboards,
so their rows must first be reviewed and mapped into versioned task contracts.
Aggregates, the HEST-joint GenBio variant, the high-resolution post-trained
Midnight checkpoint, and a contradictory OpenMidnight prose sentence are retained
as quarantined evidence rather than silently selected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path


GENBIO_PDF_SHA256 = "814ac10253ae55d737e02e1aad549e1995f05d90fd96c1fc2bc28a669996d9a1"
MIDNIGHT_PDF_SHA256 = "29dce4004f4f4c2e8f75cb5fc2c03c934d97e20099329bb9682901c647bc07cb"
OPENMIDNIGHT_HTML_SHA256 = "bf31b3c716dd59da9e230e75abd913b4809bd864ff89e38b752bd9b89f4c402b"
OPENMIDNIGHT_REPO_COMMIT = "4c3e4a83802010f47dc68bb2d25629f2b6f58eea"

GENBIO_REFERENCE = "https://genbio.ai/papers/genbio-pathfm.pdf"
MIDNIGHT_REFERENCE = "https://papers.miccai.org/miccai-2025/paper/4651_paper.pdf"
OPENMIDNIGHT_REFERENCE = "https://sophont.med/blog/openmidnight"
OPENMIDNIGHT_REPOSITORY = "https://github.com/MedARC-AI/OpenMidnight"

FIELDS = (
    "source_scope",
    "source_table",
    "source_row",
    "model_alias",
    "model_id",
    "model_revision",
    "suite_id",
    "task_label",
    "evaluation_id",
    "dedup_key",
    "metric",
    "value",
    "value_unit",
    "embedding_recipe",
    "protocol_variant",
    "reference_url",
    "source_locator",
    "source_revision",
    "source_sha256",
    "inclusion_status",
    "inclusion_reason",
)

EVA_TASKS = (
    ("PCam 10 shots", "patch_camelyon_10shot.test", "balanced_accuracy"),
    ("BACH", "bach.validation", "balanced_accuracy"),
    ("BRACS", "bracs.validation", "balanced_accuracy"),
    ("BreakHis", "breakhis.validation", "balanced_accuracy"),
    ("CRC-100K", "crc.validation", "balanced_accuracy"),
    ("Gleason", "gleason_arvaniti.validation", "balanced_accuracy"),
    ("MHIST", "mhist.test", "balanced_accuracy"),
    ("PCam", "patch_camelyon.test", "balanced_accuracy"),
    ("Cam16 (small)", "camelyon16_small.test", "balanced_accuracy"),
    ("Panda (small)", "panda_small.test", "balanced_accuracy"),
    ("CoNSeP", "consep.validation", "dice"),
    ("MoNuSAC", "monusac.test", "dice"),
)

HEST_TASKS = (
    ("IDC", "hest.idc.gene_expression"),
    ("PRAD", "hest.prad.gene_expression"),
    ("PAAD", "hest.paad.gene_expression"),
    ("SKCM", "hest.skcm.gene_expression"),
    ("COAD", "hest.coad.gene_expression"),
    ("READ", "hest.read.gene_expression"),
    ("CCRCC", "hest.ccrcc.gene_expression"),
    ("LUNG", "hest.lung.gene_expression"),
    ("LYMPH", "hest.lymph_idc.gene_expression"),
)

THUNDER_TASKS = (
    "BACH",
    "BRACS",
    "BreakHis",
    "CCRCC",
    "CRC-100K",
    "ESCA",
    "MHIST",
    "PCAM",
    "TCGA-CRC",
    "TCGA-TILS",
    "TCGA-Unif",
    "WILDS",
)

OPENMIDNIGHT_CONFIGS = {
    "PCam 10 shots": "pcam_10.yaml",
    "BACH": "bach.yaml",
    "BRACS": "bracs.yaml",
    "BreakHis": "breakhist.yaml",
    "CRC-100K": "crc.yaml",
    "Gleason": "gleason.yaml",
    "MHIST": "mhist.yaml",
    "PCam": "pcam.yaml",
    "Cam16 (small)": "cam16_small.yaml",
    "Panda (small)": "panda_small.yaml",
    "CoNSeP": "consep.yaml",
    "MoNuSAC": "monusac.yaml",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_digest(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"unexpected {label} SHA-256: {actual}")


def _pdf_text(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def _row(**values: str) -> dict[str, str]:
    missing = set(FIELDS) - set(values)
    if missing:
        raise ValueError(f"missing evidence fields: {sorted(missing)}")
    return {field: values[field] for field in FIELDS}


def _genbio_row(
    *,
    table: str,
    source_row: str,
    model_alias: str,
    model_id: str,
    revision: str,
    suite: str,
    task: str,
    evaluation_id: str,
    dedup_key: str,
    metric: str,
    value: float,
    unit: str,
    embedding: str,
    protocol: str,
    status: str,
    reason: str,
) -> dict[str, str]:
    return _row(
        source_scope="official_primary_paper",
        source_table=table,
        source_row=source_row,
        model_alias=model_alias,
        model_id=model_id,
        model_revision=revision,
        suite_id=suite,
        task_label=task,
        evaluation_id=evaluation_id,
        dedup_key=dedup_key,
        metric=metric,
        value=f"{value:g}",
        value_unit=unit,
        embedding_recipe=embedding,
        protocol_variant=protocol,
        reference_url=GENBIO_REFERENCE,
        source_locator=f"GenBio-PathFM PDF|{table}|{source_row}",
        source_revision="PDF created 2026-03-17",
        source_sha256=GENBIO_PDF_SHA256,
        inclusion_status=status,
        inclusion_reason=reason,
    )


def genbio_rows(pdf_path: Path) -> list[dict[str, str]]:
    _check_digest(pdf_path, GENBIO_PDF_SHA256, "GenBio-PathFM PDF")
    text = _pdf_text(pdf_path)
    for anchor in (
        "Table A1: Detailed results on the HEST benchmark",
        "Table A2: Comparison of GenBio-PathFM",
        "Table A3: Detailed results on the THUNDER benchmark",
        "Table A4: Results on the PathoROB benchmark",
        "Table A5: Performance drop under in-distribution",
        "We focused on the k-nearest neighbor (kNN) task",
    ):
        if anchor not in text:
            raise ValueError(f"missing GenBio source anchor: {anchor}")

    rows: list[dict[str, str]] = []
    standard_revision = "GenBio-PathFM official 1.1B JEDI checkpoint"
    hest_values = (0.587, 0.391, 0.496, 0.672, 0.328, 0.179, 0.262, 0.579, 0.284)
    for (task, evaluation_id), value in zip(HEST_TASKS, hest_values, strict=True):
        rows.append(_genbio_row(
            table="Table A1", source_row=f"GenBio-PathFM/{task}",
            model_alias="GenBio-PathFM", model_id="genbio-pathfm", revision=standard_revision,
            suite="hest", task=task, evaluation_id=evaluation_id, dedup_key=evaluation_id,
            metric="pearson_r", value=value, unit="fraction", embedding="CLS token",
            protocol="HEST single-task ridge regression on embedding PCs; top 50 variable genes",
            status="duplicate_alternate_evidence",
            reason="The pinned HEST leaderboard supplies the same cell at higher precision; retain this rounded primary-paper value as alternate evidence.",
        ))
    rows.append(_genbio_row(
        table="Table A1", source_row="GenBio-PathFM/Avg.", model_alias="GenBio-PathFM",
        model_id="genbio-pathfm", revision=standard_revision, suite="hest", task="Average",
        evaluation_id="hest.genbio2026.average", dedup_key="hest.aggregate.average",
        metric="mean_pearson_r", value=0.420, unit="fraction", embedding="CLS token",
        protocol="Arithmetic summary across nine HEST tasks", status="aggregate_excluded",
        reason="Aggregate is derived from task cells and must not be treated as an independent benchmark observation.",
    ))

    joint_values = (0.826, 0.654, 0.739, 0.873, 0.706, 0.403, 0.567, 0.778, 0.476)
    for (task, canonical_id), value in zip(HEST_TASKS, joint_values, strict=True):
        rows.append(_genbio_row(
            table="Table A2", source_row=f"GenBio-PathFM (Joint)/{task}",
            model_alias="GenBio-PathFM (Joint)", model_id="genbio-pathfm-hest-joint",
            revision="GenBio-PathFM trained jointly on all HEST-bench training sets",
            suite="hest", task=task, evaluation_id=f"hest.genbio2026_joint.{task.lower()}",
            dedup_key=canonical_id, metric="pearson_r", value=value, unit="fraction",
            embedding="CLS token", protocol="Joint downstream training across all HEST training sets",
            status="fine_tuned_excluded",
            reason="Task-specific joint training changes the foundation checkpoint and is quarantined from frozen-model compression analysis.",
        ))
    rows.append(_genbio_row(
        table="Table A2", source_row="GenBio-PathFM (Joint)/Average",
        model_alias="GenBio-PathFM (Joint)", model_id="genbio-pathfm-hest-joint",
        revision="GenBio-PathFM trained jointly on all HEST-bench training sets", suite="hest",
        task="Average", evaluation_id="hest.genbio2026_joint.average",
        dedup_key="hest.aggregate.average", metric="mean_pearson_r", value=0.669,
        unit="fraction", embedding="CLS token",
        protocol="Joint downstream training across all HEST training sets",
        status="fine_tuned_excluded",
        reason="Both fine-tuned and aggregate; retained only as quarantined evidence.",
    ))

    thunder_values = (81.8, 56.3, 86.0, 92.7, 94.8, 83.6, 69.9, 90.2, 64.4, 89.0, 76.8, 95.8)
    for task, value in zip(THUNDER_TASKS, thunder_values, strict=True):
        slug = task.lower().replace("-", "_")
        rows.append(_genbio_row(
            table="Table A3", source_row=f"GenBio-PathFM/{task}",
            model_alias="GenBio-PathFM", model_id="genbio-pathfm", revision=standard_revision,
            suite="thunder", task=task, evaluation_id=f"thunder.genbio2026.{slug}.knn",
            dedup_key=f"thunder.dataset.{slug}", metric="f1_score", value=value, unit="percent",
            embedding="CLS token", protocol="Official THUNDER kNN protocol; frozen encoder",
            status="canonical_candidate",
            reason="Exact public-suite task cell under the paper's explicitly reported kNN protocol; distinct from current THUNDER linear-probe rows.",
        ))
    rows.append(_genbio_row(
        table="Table A3", source_row="GenBio-PathFM/Avg.", model_alias="GenBio-PathFM",
        model_id="genbio-pathfm", revision=standard_revision, suite="thunder", task="Average",
        evaluation_id="thunder.genbio2026.average.knn", dedup_key="thunder.aggregate.average",
        metric="mean_f1_score", value=81.8, unit="percent", embedding="CLS token",
        protocol="Mean across 12 THUNDER kNN tasks", status="aggregate_excluded",
        reason="Aggregate is derived from task cells and must not be treated as an independent observation.",
    ))

    pathorob_values = {
        "camelyon": (0.865, 98.5),
        "tcga": (0.838, 94.1),
        "tolkach_esca": (0.960, 98.0),
    }
    for dataset, (ri, bacc) in pathorob_values.items():
        canonical_ri = "pathorob.tcga_2x2.robustness_index" if dataset == "tcga" else f"pathorob.{dataset}.robustness_index"
        for endpoint, metric, value, unit, status, reason in (
            ("RI", "robustness_index", ri, "fraction", "duplicate_alternate_evidence", "The pinned PathoROB leaderboard supplies the same external-publication RI cell; retain the paper table as alternate primary evidence."),
            ("BACC", "balanced_accuracy", bacc, "percent", "canonical_candidate", "Exact public PathoROB balanced-accuracy task cell not represented by the current RI-only leaderboard extraction."),
        ):
            evaluation_id = canonical_ri if endpoint == "RI" else f"pathorob.genbio2026.{dataset}.balanced_accuracy"
            rows.append(_genbio_row(
                table="Table A4", source_row=f"GenBio-PathFM/{dataset}/{endpoint}",
                model_alias="GenBio-PathFM", model_id="genbio-pathfm", revision=standard_revision,
                suite="pathorob", task=f"{dataset}/{endpoint}", evaluation_id=evaluation_id,
                dedup_key=f"pathorob.{dataset}.{metric}", metric=metric, value=value, unit=unit,
                embedding="CLS + mean pooled patch tokens", protocol="Original PathoROB protocol",
                status=status, reason=reason,
            ))
    for endpoint, metric, value, unit in (
        ("RI", "mean_robustness_index", 0.888, "fraction"),
        ("BACC", "mean_balanced_accuracy", 96.9, "percent"),
    ):
        rows.append(_genbio_row(
            table="Table A4", source_row=f"GenBio-PathFM/Average/{endpoint}",
            model_alias="GenBio-PathFM", model_id="genbio-pathfm", revision=standard_revision,
            suite="pathorob", task=f"Average/{endpoint}",
            evaluation_id=f"pathorob.genbio2026.average.{endpoint.lower()}",
            dedup_key=f"pathorob.aggregate.{metric}", metric=metric, value=value, unit=unit,
            embedding="CLS + mean pooled patch tokens", protocol="Mean across three PathoROB datasets",
            status="aggregate_excluded", reason="Aggregate is derived from dataset cells.",
        ))

    drops = {
        "camelyon": (-2.3, -0.6),
        "tcga": (-0.9, 0.8),
        "tolkach_esca": (0.2, -0.1),
    }
    for dataset, (id_drop, ood_drop) in drops.items():
        for endpoint, value in (("id", id_drop), ("ood", ood_drop)):
            rows.append(_genbio_row(
                table="Table A5", source_row=f"GenBio-PathFM/{dataset}/{endpoint.upper()}",
                model_alias="GenBio-PathFM", model_id="genbio-pathfm", revision=standard_revision,
                suite="pathorob", task=f"{dataset}/{endpoint.upper()} performance drop",
                evaluation_id=f"pathorob.genbio2026.{dataset}.apd_{endpoint}",
                dedup_key=f"pathorob.{dataset}.apd_{endpoint}",
                metric="average_performance_drop_percent", value=value, unit="percent_relative_change",
                embedding="CLS + mean pooled patch tokens", protocol="Original PathoROB APD protocol",
                status="canonical_candidate_analysis_ineligible",
                reason="Exact task endpoint is retained, but the signed unbounded metric has no source-defined common-scale normalization.",
            ))
    for endpoint, value in (("id", -1.0), ("ood", 0.04)):
        rows.append(_genbio_row(
            table="Table A5", source_row=f"GenBio-PathFM/Average/{endpoint.upper()}",
            model_alias="GenBio-PathFM", model_id="genbio-pathfm", revision=standard_revision,
            suite="pathorob", task=f"Average/{endpoint.upper()} performance drop",
            evaluation_id=f"pathorob.genbio2026.average.apd_{endpoint}",
            dedup_key=f"pathorob.aggregate.apd_{endpoint}", metric="average_performance_drop_percent",
            value=value, unit="percent_relative_change", embedding="CLS + mean pooled patch tokens",
            protocol="Mean across three PathoROB datasets", status="aggregate_excluded",
            reason="Aggregate is derived from dataset endpoints.",
        ))
    if len(rows) != 49:
        raise AssertionError(f"expected 49 GenBio rows, found {len(rows)}")
    return rows


def midnight_rows(pdf_path: Path) -> list[dict[str, str]]:
    _check_digest(pdf_path, MIDNIGHT_PDF_SHA256, "Midnight MICCAI PDF")
    text = _pdf_text(pdf_path)
    for anchor in (
        "Table 2. Performance metrics for all evaluated FMs",
        "disabled early-stopping in eva’s protocol",
        "For every model, we evaluated the CLS+Mean",
        "Midnight-12k         12k .803 .907 .639 .840 .967 .790 .815 .931 .869 .656 .625 .664 .412 .763",
    ):
        if anchor not in text:
            raise ValueError(f"missing Midnight source anchor: {anchor}")

    model_rows = (
        ("Midnight-92k/392", "midnight-92k-392", "Midnight-92k high-resolution post-training for 120k iterations; 392px", "clsmean_392", (0.900, 0.904, 0.646, 0.802, 0.966, 0.807, 0.828, 0.951, 0.868, 0.651, 0.662, 0.708), 0.415, 0.778, True),
        ("Midnight-92k", "midnight-92k", "Midnight trained on TCGA + NKI-80k; 224px", "clsmean_224", (0.882, 0.889, 0.615, 0.793, 0.967, 0.823, 0.831, 0.948, 0.872, 0.643, 0.629, 0.656), 0.425, 0.767, False),
        ("Midnight-12k", "midnight", "Released Midnight-12k TCGA checkpoint; 224px", "clsmean_224", (0.803, 0.907, 0.639, 0.840, 0.967, 0.790, 0.815, 0.931, 0.869, 0.656, 0.625, 0.664), 0.412, 0.763, False),
    )
    rows: list[dict[str, str]] = []
    for alias, model_id, revision, protocol_slug, values, hest_average, overall_average, fine_tuned in model_rows:
        for (task, canonical_suffix, metric), value in zip(EVA_TASKS, values, strict=True):
            status = "fine_tuned_excluded" if fine_tuned else "canonical_candidate"
            reason = (
                "The paper explicitly describes this checkpoint as high-resolution fine-tuning/post-training; quarantine from frozen base-checkpoint analysis."
                if fine_tuned else
                "Exact public EVA task cell under the MICCAI paper's versioned CLS+Mean protocol; preserve separately from later leaderboard/model-card reruns."
            )
            rows.append(_row(
                source_scope="official_primary_paper", source_table="Table 2",
                source_row=f"{alias}/{task}", model_alias=alias, model_id=model_id,
                model_revision=revision, suite_id="eva", task_label=task,
                evaluation_id=f"eva.miccai2025.{protocol_slug}.{canonical_suffix}",
                dedup_key=f"eva.leaderboard.{canonical_suffix}", metric=metric, value=f"{value:g}",
                value_unit="fraction", embedding_recipe="CLS + mean pooled patch tokens",
                protocol_variant=(
                    "MICCAI 2025 EVA fork; early stopping disabled except Camelyon16/PANDA; "
                    + ("392x392 input" if protocol_slug == "clsmean_392" else "224x224 input")
                ),
                reference_url=MIDNIGHT_REFERENCE,
                source_locator=f"4651_paper.pdf|Table 2|model={alias}|column={task}",
                source_revision="MICCAI 2025 proceedings PDF; modified 2025-08-25",
                source_sha256=MIDNIGHT_PDF_SHA256, inclusion_status=status, inclusion_reason=reason,
            ))
        for task, evaluation_id, metric, value in (
            ("HEST", f"hest.miccai2025.{protocol_slug}.average", "mean_pearson_r", hest_average),
            ("Average", f"eva_hest.miccai2025.{protocol_slug}.average", "cross_benchmark_mean", overall_average),
        ):
            rows.append(_row(
                source_scope="official_primary_paper", source_table="Table 2",
                source_row=f"{alias}/{task}", model_alias=alias, model_id=model_id,
                model_revision=revision, suite_id="hest" if task == "HEST" else "cross_suite",
                task_label=task, evaluation_id=evaluation_id,
                dedup_key="hest.aggregate.average" if task == "HEST" else "cross_suite.aggregate.average",
                metric=metric, value=f"{value:g}", value_unit="fraction",
                embedding_recipe="CLS + mean pooled patch tokens",
                protocol_variant="Aggregate reported in MICCAI Table 2",
                reference_url=MIDNIGHT_REFERENCE,
                source_locator=f"4651_paper.pdf|Table 2|model={alias}|column={task}",
                source_revision="MICCAI 2025 proceedings PDF; modified 2025-08-25",
                source_sha256=MIDNIGHT_PDF_SHA256, inclusion_status="aggregate_excluded",
                inclusion_reason="Aggregate is derived from underlying tasks and is not an independent benchmark observation.",
            ))
    if len(rows) != 42:
        raise AssertionError(f"expected 42 Midnight rows, found {len(rows)}")
    return rows


class _PerformanceTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_cell = False
        self.cell_parts: list[str] = []
        self.current: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "performance-table":
            self.in_table = True
        elif self.in_table and tag in {"th", "td"}:
            self.in_cell = True
            self.cell_parts = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.in_table and tag in {"th", "td"} and self.in_cell:
            self.current.append(" ".join("".join(self.cell_parts).split()))
            self.in_cell = False
        elif self.in_table and tag == "tr":
            if self.current:
                self.rows.append(self.current)
            self.current = []
        elif self.in_table and tag == "table":
            self.in_table = False


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def openmidnight_rows(html_path: Path, repository: Path) -> list[dict[str, str]]:
    _check_digest(html_path, OPENMIDNIGHT_HTML_SHA256, "OpenMidnight report HTML")
    commit = _git(repository, "rev-parse", "HEAD")
    if commit != OPENMIDNIGHT_REPO_COMMIT:
        raise ValueError(f"unexpected OpenMidnight repository revision: {commit}")
    actual_configs = {path.name for path in (repository / "eval_configs").glob("*.yaml")}
    if actual_configs != set(OPENMIDNIGHT_CONFIGS.values()):
        raise ValueError(f"unexpected OpenMidnight EVA config set: {sorted(actual_configs)}")

    text = html_path.read_text(encoding="utf-8")
    for anchor in (
        '"datePublished": "2025-11-14"',
        '"dateModified": "2026-06-16"',
        '"value": "10.5281/zenodo.20711012"',
        "Only performance with [CLS] token is reported.",
        "Scores for existing models are taken from Karasikov et al. (2025).",
    ):
        if anchor not in text:
            raise ValueError(f"missing OpenMidnight source anchor: {anchor}")

    parser = _PerformanceTableParser()
    parser.feed(text)
    header = parser.rows[0]
    own = next(row for row in parser.rows[1:] if row[0] == "OpenMidnight (Ours)")
    columns = dict(zip(header, own, strict=True))
    rows: list[dict[str, str]] = []
    for task, canonical_suffix, metric in EVA_TASKS:
        config = OPENMIDNIGHT_CONFIGS[task]
        config_digest = sha256(repository / "eval_configs" / config)
        report_column = "PCam (10 shots)" if task == "PCam 10 shots" else task
        value = float(columns[report_column])
        rows.append(_row(
            source_scope="official_primary_technical_report",
            source_table="Performance comparison table", source_row=f"OpenMidnight (Ours)/{task}",
            model_alias="OpenMidnight (Ours)", model_id="openmidnight",
            model_revision="SophontAI/OpenMidnight/teacher_checkpoint_load.pt; immutable HF revision unavailable anonymously",
            suite_id="eva", task_label=task,
            evaluation_id=f"eva.openmidnight_tr2025.cls.{canonical_suffix}",
            dedup_key=f"eva.leaderboard.{canonical_suffix}", metric=metric,
            value=f"{value:g}", value_unit="fraction", embedding_recipe="CLS token only",
            protocol_variant=(
                "OpenMidnight EVA fork; supporting config snapshot "
                f"{OPENMIDNIGHT_REPO_COMMIT}:eval_configs/{config} sha256:{config_digest}"
            ),
            reference_url=OPENMIDNIGHT_REFERENCE,
            source_locator=f"HTML table id=performance-table|model=OpenMidnight (Ours)|column={report_column};support={OPENMIDNIGHT_REPOSITORY}/blob/{OPENMIDNIGHT_REPO_COMMIT}/eval_configs/{config}",
            source_revision="SOPHONT-TR-2025-001; published 2025-11-14; modified 2026-06-16",
            source_sha256=OPENMIDNIGHT_HTML_SHA256, inclusion_status="canonical_candidate",
            inclusion_reason="Own-model public EVA task cell with exact CLS-only identity; prior-model rows in the report are deliberately not imported.",
        ))
    for task, metric in (("HEST", "mean_pearson_r"), ("Average", "cross_benchmark_mean")):
        rows.append(_row(
            source_scope="official_primary_technical_report",
            source_table="Performance comparison table", source_row=f"OpenMidnight (Ours)/{task}",
            model_alias="OpenMidnight (Ours)", model_id="openmidnight",
            model_revision="SophontAI/OpenMidnight/teacher_checkpoint_load.pt; immutable HF revision unavailable anonymously",
            suite_id="hest" if task == "HEST" else "cross_suite", task_label=task,
            evaluation_id="hest.openmidnight_tr2025.average" if task == "HEST" else "eva_hest.openmidnight_tr2025.average",
            dedup_key="hest.aggregate.average" if task == "HEST" else "cross_suite.aggregate.average",
            metric=metric, value=f"{float(columns[task]):g}", value_unit="fraction",
            embedding_recipe="CLS token only", protocol_variant="Aggregate reported by technical report",
            reference_url=OPENMIDNIGHT_REFERENCE,
            source_locator=f"HTML table id=performance-table|model=OpenMidnight (Ours)|column={task}",
            source_revision="SOPHONT-TR-2025-001; published 2025-11-14; modified 2026-06-16",
            source_sha256=OPENMIDNIGHT_HTML_SHA256, inclusion_status="aggregate_excluded",
            inclusion_reason="Aggregate is derived from task cells and must not be treated as an independent benchmark observation.",
        ))

    prose = re.search(
        r"BreakHis and Cam16\(small\) scoring 0\.946 and 0\.873 balanced accuracy",
        text,
    )
    if prose is None:
        raise ValueError("OpenMidnight prose/table contradiction anchor is missing")
    for task, canonical_suffix, metric, value in (
        ("BreakHis", "breakhis.validation", "balanced_accuracy", 0.946),
        ("Cam16 (small)", "camelyon16_small.test", "balanced_accuracy", 0.873),
    ):
        rows.append(_row(
            source_scope="official_primary_technical_report_narrative",
            source_table="Results prose", source_row=f"sentence/{task}",
            model_alias="OpenMidnight", model_id="openmidnight",
            model_revision="SophontAI/OpenMidnight/teacher_checkpoint_load.pt; immutable HF revision unavailable anonymously",
            suite_id="eva", task_label=task,
            evaluation_id=f"eva.openmidnight_tr2025.cls.{canonical_suffix}",
            dedup_key=f"eva.leaderboard.{canonical_suffix}", metric=metric,
            value=f"{value:g}", value_unit="fraction", embedding_recipe="CLS token only",
            protocol_variant="Narrative sentence, contradicted by labeled table columns",
            reference_url=OPENMIDNIGHT_REFERENCE,
            source_locator="Results paragraph beginning 'These results show'",
            source_revision="SOPHONT-TR-2025-001; published 2025-11-14; modified 2026-06-16",
            source_sha256=OPENMIDNIGHT_HTML_SHA256,
            inclusion_status="narrative_conflict_excluded",
            inclusion_reason="The prose swaps the labeled BreakHis and Cam16-small table values; the unambiguous table cells are selected instead.",
        ))
    if len(rows) != 16:
        raise AssertionError(f"expected 16 OpenMidnight rows, found {len(rows)}")
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genbio-pdf", type=Path, default=Path("/tmp/genbio-pathfm.pdf"))
    parser.add_argument("--midnight-pdf", type=Path, default=Path("/tmp/midnight-4651.pdf"))
    parser.add_argument("--openmidnight-html", type=Path, default=Path("/tmp/openmidnight_report.html"))
    parser.add_argument("--openmidnight-repo", type=Path, default=Path("/tmp/pathopress_sources/eva_openmidnight"))
    parser.add_argument("--output-dir", type=Path, default=Path("source_data"))
    args = parser.parse_args()
    write_rows(args.output_dir / "genbio_pathfm_official_2026.csv", genbio_rows(args.genbio_pdf))
    write_rows(args.output_dir / "midnight_miccai2025_official_scores.csv", midnight_rows(args.midnight_pdf))
    write_rows(args.output_dir / "openmidnight_technical_report_2025.csv", openmidnight_rows(args.openmidnight_html, args.openmidnight_repo))


if __name__ == "__main__":
    main()
