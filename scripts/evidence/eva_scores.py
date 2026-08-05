"""Extract and reconcile EVA pathology leaderboard scores.

Two first-party sources are intentionally treated as evidence snapshots:

* ``kaiko-ai/eva`` publishes the current machine-readable leaderboard, including
  separate validation and test columns for three datasets.
* the ``kaiko-ai/midnight`` Hugging Face model card publishes two additional
  EVA protocols (BRACS and PCam 10-shot) and four models absent from the current
  EVA CSV.

The sources overlap and some rerun/rounding values differ.  ``merge_scores``
keeps the current EVA repository value for an identical model/protocol cell,
adds non-overlapping Hugging Face cells, and returns every duplicate as an
explicit audit record.  It never averages or silently overwrites scores.
"""

from __future__ import annotations

import csv
import html
import re
from dataclasses import dataclass
from pathlib import Path


EVA_REPOSITORY_URL = "https://github.com/kaiko-ai/eva"
MIDNIGHT_REPOSITORY_URL = "https://huggingface.co/kaiko-ai/midnight"

# Protocol IDs distinguish validation from test and reduced-data settings.
# EVA's leaderboard documentation defines every binary/multiclass displayed
# value as balanced accuracy.  Some configs monitor ``MulticlassAccuracy`` for
# checkpoint selection; that is training metadata, not the leaderboard's
# declared reporting label, so it must not replace the documented metric here.
REPOSITORY_COLUMNS = {
    "bach": ("eva.leaderboard.bach.validation", "balanced_accuracy", 5),
    "breakhis": ("eva.leaderboard.breakhis.validation", "balanced_accuracy", 5),
    "crc": ("eva.leaderboard.crc.validation", "balanced_accuracy", 5),
    "gleason_arvaniti": ("eva.leaderboard.gleason_arvaniti.validation", "balanced_accuracy", 5),
    "mhist": ("eva.leaderboard.mhist.test", "balanced_accuracy", 5),
    "patch_camelyon": ("eva.leaderboard.patch_camelyon.validation", "balanced_accuracy", 5),
    "patch_camelyon/test": ("eva.leaderboard.patch_camelyon.test", "balanced_accuracy", 5),
    "camelyon16_small": ("eva.leaderboard.camelyon16_small.validation", "balanced_accuracy", 20),
    "camelyon16_small/test": ("eva.leaderboard.camelyon16_small.test", "balanced_accuracy", 20),
    "panda_small": ("eva.leaderboard.panda_small.validation", "balanced_accuracy", 20),
    "panda_small/test": ("eva.leaderboard.panda_small.test", "balanced_accuracy", 20),
    "consep": ("eva.leaderboard.consep.validation", "dice", 5),
    "monusac": ("eva.leaderboard.monusac.test", "dice", 5),
}

MIDNIGHT_COLUMNS = {
    "PCam 10 shots": ("eva.leaderboard.patch_camelyon_10shot.test", "balanced_accuracy", 5),
    "BACH": ("eva.leaderboard.bach.validation", "balanced_accuracy", 5),
    "BRACS": ("eva.leaderboard.bracs.validation", "balanced_accuracy", 5),
    "BreaKHis": ("eva.leaderboard.breakhis.validation", "balanced_accuracy", 5),
    "CRC": ("eva.leaderboard.crc.validation", "balanced_accuracy", 5),
    "Gleason": ("eva.leaderboard.gleason_arvaniti.validation", "balanced_accuracy", 5),
    "MHIST": ("eva.leaderboard.mhist.test", "balanced_accuracy", 5),
    "PCam": ("eva.leaderboard.patch_camelyon.test", "balanced_accuracy", 5),
    "Cam16 (small)": ("eva.leaderboard.camelyon16_small.test", "balanced_accuracy", 20),
    "Panda (small)": ("eva.leaderboard.panda_small.test", "balanced_accuracy", 20),
    "CoNSeP": ("eva.leaderboard.consep.validation", "dice", 5),
    "MoNuSAC": ("eva.leaderboard.monusac.test", "dice", 5),
}

PROTOCOL_CONFIGS = {
    "eva.leaderboard.bach.validation": "classification/bach.yaml",
    "eva.leaderboard.bracs.validation": "classification/bracs.yaml",
    "eva.leaderboard.breakhis.validation": "classification/breakhis.yaml",
    "eva.leaderboard.crc.validation": "classification/crc.yaml",
    "eva.leaderboard.gleason_arvaniti.validation": "classification/gleason_arvaniti.yaml",
    "eva.leaderboard.mhist.test": "classification/mhist.yaml",
    "eva.leaderboard.patch_camelyon.validation": "classification/patch_camelyon.yaml",
    "eva.leaderboard.patch_camelyon.test": "classification/patch_camelyon.yaml",
    "eva.leaderboard.patch_camelyon_10shot.test": "classification/patch_camelyon_10shot.yaml",
    "eva.leaderboard.camelyon16_small.validation": "classification/camelyon16_small.yaml",
    "eva.leaderboard.camelyon16_small.test": "classification/camelyon16_small.yaml",
    "eva.leaderboard.panda_small.validation": "classification/panda_small.yaml",
    "eva.leaderboard.panda_small.test": "classification/panda_small.yaml",
    "eva.leaderboard.consep.validation": "segmentation/consep.yaml",
    "eva.leaderboard.monusac.test": "segmentation/monusac.yaml",
}

# First-party identifiers are resolved here rather than with fuzzy matching.
MODEL_IDS = {
    "paige_virchow2": "virchow-2",
    "Virchow2": "virchow-2",
    "mahmood_uni2_h": "uni2-h",
    "UNI-2": "uni2-h",
    "kaiko_midnight_12k": "midnight",
    "Midnight-12k": "midnight",
    "Midnight-92k": "midnight-92k",
    "Midnight-92k/392": "midnight-92k-392",
    "bioptimus_h_optimus_0": "h-optimus-0",
    "H-Optimus-0": "h-optimus-0",
    "kaiko_vitb8": "kaiko-vit-b-8",
    "Kaiko-B8": "kaiko-vit-b-8",
    "prov_gigapath": "prov-gigapath",
    "Prov_GigaPath": "prov-gigapath",
    "mahmood_uni": "uni",
    "UNI": "uni",
    "histai_hibou_l": "hibou-l",
    "Hibou-L": "hibou-l",
    "kaiko_vitl14": "kaiko-vit-l-14",
    "kaiko_vitb16": "kaiko-vit-b-16",
    "kaiko_vits8": "kaiko-vit-s-8",
    "kaiko_vits16": "kaiko-vit-s-16",
    "owkin_phikon": "phikon",
    "Phikon": "phikon",
    "owkin_phikon_v2": "phikon-v2",
    "Phikon-v2": "phikon-v2",
    "lunit_vits16": "lunit-vit-s-16",
    "Lunit": "lunit-vit-s-16",
    "vitg14 (nat. img.)": "dinov2-vit-g-14-natural-images",
    "vitg14 (initial)": "dinov2-vit-g-14-initial",
}


@dataclass(frozen=True)
class EvaScore:
    model_id: str
    reported_model_alias: str
    evaluation_id: str
    value: float
    normalized_score: float
    metric: str
    reference_url: str
    source_locator: str
    uncertainty: str
    lineage: str

    def registry_row(self, extraction_date: str) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "reported_model_alias": self.reported_model_alias,
            "model_revision": "not_reported",
            "evaluation_id": self.evaluation_id,
            "value": f"{self.value:.6g}",
            "normalized_score": f"{self.normalized_score:.6g}",
            "suite_id": "eva",
            "metric": self.metric,
            "reference_url": self.reference_url,
            "source_locator": self.source_locator,
            "extraction_date": extraction_date,
            "review_status": "machine_parsed_single_source",
            "uncertainty": self.uncertainty,
            "lineage": self.lineage,
            "audit_status": "parsed_primary_source",
        }


def canonical_model(alias: str) -> str:
    try:
        return MODEL_IDS[alias]
    except KeyError as exc:
        raise ValueError(f"unmapped EVA model alias: {alias!r}") from exc


def _blob(base: str, commit: str, path: str) -> str:
    return f"{base}/blob/{commit}/{path}"


def parse_repository_scores(source: Path, commit: str, *, strict: bool = True) -> list[EvaScore]:
    relative = "tools/data/leaderboards/pathology.csv"
    reference = _blob(EVA_REPOSITORY_URL, commit, relative)
    with (source / relative).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if strict and (len(rows) != 15 or set(rows[0]) != {"model", *REPOSITORY_COLUMNS}):
        raise ValueError("unexpected EVA pathology leaderboard shape")

    scores: list[EvaScore] = []
    for row_number, row in enumerate(rows, start=2):
        alias = row["model"]
        model_id = canonical_model(alias)
        for column, (evaluation_id, metric, runs) in REPOSITORY_COLUMNS.items():
            raw = row.get(column, "").strip()
            if not raw:
                continue
            value = float(raw)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"EVA score outside [0,1] at row {row_number}, column {column}")
            scores.append(
                EvaScore(
                    model_id=model_id,
                    reported_model_alias=alias,
                    evaluation_id=evaluation_id,
                    value=value,
                    normalized_score=value * 100.0,
                    metric=metric,
                    reference_url=reference,
                    source_locator=f"csv_row={row_number}|model={alias}|column={column}",
                    uncertainty=f"mean_over_{runs}_runs_dispersion_not_reported",
                    lineage=f"eva@{commit}:{relative} -> parse_repository_scores -> scores.csv",
                )
            )
    if strict and len(scores) != 15 * 13:
        raise ValueError(f"expected 195 EVA repository scores, found {len(scores)}")
    return scores


def _plain_markdown(cell: str) -> str:
    cell = cell.replace("**", "").replace("__", "")
    cell = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", cell)
    # The model card spells "PCam 10 shots" with a non-breaking space.
    return html.unescape(cell).replace("\xa0", " ").strip()


def _markdown_table(text: str, header_start: str) -> tuple[list[str], list[tuple[int, list[str]]]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header_start))
    header = [_plain_markdown(cell) for cell in lines[start].strip().strip("|").split("|")]
    rows: list[tuple[int, list[str]]] = []
    for index, line in enumerate(lines[start + 2 :], start=start + 3):
        if not line.startswith("|"):
            break
        cells = [_plain_markdown(cell) for cell in line.strip().strip("|").split("|")]
        rows.append((index, cells))
    return header, rows


def parse_midnight_scores(source: Path, commit: str, *, strict: bool = True) -> list[EvaScore]:
    relative = "README.md"
    reference = _blob(MIDNIGHT_REPOSITORY_URL, commit, relative)
    header, rows = _markdown_table((source / relative).read_text(encoding="utf-8"), "| Model | AVG.")
    positions = {name: header.index(name) for name in MIDNIGHT_COLUMNS}
    scores: list[EvaScore] = []
    for line_number, cells in rows:
        alias = cells[0]
        model_id = canonical_model(alias)
        for column, (evaluation_id, metric, runs) in MIDNIGHT_COLUMNS.items():
            value = float(cells[positions[column]])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Midnight EVA score outside [0,1] at line {line_number}, column {column}")
            scores.append(
                EvaScore(
                    model_id=model_id,
                    reported_model_alias=alias,
                    evaluation_id=evaluation_id,
                    value=value,
                    normalized_score=value * 100.0,
                    metric=metric,
                    reference_url=reference,
                    source_locator=f"table=Results Summary|line={line_number}|model={alias}|column={column}",
                    uncertainty=f"mean_over_{runs}_runs_dispersion_not_reported",
                    lineage=f"eva_midnight@{commit}:{relative} -> parse_midnight_scores -> scores.csv",
                )
            )
    if strict and len(scores) != 15 * 12:
        raise ValueError(f"expected 180 Midnight EVA scores, found {len(scores)}")
    return scores


def merge_scores(
    repository_scores: list[EvaScore], midnight_scores: list[EvaScore]
) -> tuple[list[EvaScore], list[dict[str, object]]]:
    """Prefer the current EVA CSV and preserve duplicate evidence for audit."""
    selected = {(row.model_id, row.evaluation_id): row for row in repository_scores}
    if len(selected) != len(repository_scores):
        raise ValueError("duplicate model/protocol cell inside EVA repository source")
    duplicates: list[dict[str, object]] = []
    for row in midnight_scores:
        key = (row.model_id, row.evaluation_id)
        current = selected.get(key)
        if current is None:
            selected[key] = row
            continue
        duplicates.append(
            {
                "model_id": row.model_id,
                "evaluation_id": row.evaluation_id,
                "selected_value": current.value,
                "selected_reference_url": current.reference_url,
                "alternate_value": row.value,
                "alternate_reference_url": row.reference_url,
                "absolute_difference": abs(current.value - row.value),
                "decision": "prefer_current_eva_repository_snapshot",
            }
        )
    return sorted(selected.values(), key=lambda row: (row.model_id, row.evaluation_id)), duplicates


def required_additional_protocols() -> tuple[dict[str, str], ...]:
    """Task rows the main registry must add before integrating extracted scores."""
    reported = {spec[0] for spec in (*REPOSITORY_COLUMNS.values(), *MIDNIGHT_COLUMNS.values())}
    if reported != set(PROTOCOL_CONFIGS):
        raise ValueError("EVA reported protocol/config mapping drifted")
    metadata: dict[str, tuple[str, int]] = {}
    for evaluation_id, metric, runs in (
        *REPOSITORY_COLUMNS.values(), *MIDNIGHT_COLUMNS.values()
    ):
        previous = metadata.setdefault(evaluation_id, (metric, runs))
        if previous != (metric, runs):
            raise ValueError(f"inconsistent EVA protocol metadata: {evaluation_id}")
    return tuple(
        {
            "evaluation_id": evaluation_id,
            "config": PROTOCOL_CONFIGS[evaluation_id],
            "split": evaluation_id.rsplit(".", 1)[-1],
            "metric": metadata[evaluation_id][0],
            "runs": str(metadata[evaluation_id][1]),
        }
        for evaluation_id in sorted(reported)
    )
