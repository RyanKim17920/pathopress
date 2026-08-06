"""Auditable model metadata used by the Section 6 model-side hypotheses."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence


# Counts are nominal encoder parameter counts supported by the cited primary
# architecture/model source. They are deliberately absent for slide systems
# and checkpoints whose public source does not establish a comparable count.
PARAMETERS = {
    "ciga": 25_600_000,
    "clip-b": 86_000_000,
    "clip-l": 304_000_000,
    "conch": 86_000_000,
    "conch-1.5": 304_000_000,
    "ctranspath": 28_000_000,
    "dinov2-b": 86_000_000,
    "dinov2-l": 304_000_000,
    "dinov2-vit-g-14-initial": 1_100_000_000,
    "dinov2-vit-g-14-natural-images": 1_100_000_000,
    "dinov3-b": 86_000_000,
    "dinov3-l": 304_000_000,
    "dinov3-s": 21_000_000,
    "genbio-pathfm": 1_100_000_000,
    "gpfm": 304_000_000,
    "h-optimus-0": 1_100_000_000,
    "h-optimus-1": 1_100_000_000,
    "h0-mini": 86_000_000,
    "hibou-b": 86_000_000,
    "hibou-l": 304_000_000,
    "hipt": 21_700_000,
    "kaiko-vit-b-16": 86_000_000,
    "kaiko-vit-b-8": 86_000_000,
    "kaiko-vit-b-unspecified-patch": 86_000_000,
    "kaiko-vit-l-14": 304_000_000,
    "kaiko-vit-s-16": 21_700_000,
    "kaiko-vit-s-8": 21_700_000,
    "kaiko-vit-s-unspecified-patch": 21_700_000,
    "lunit-vit-s-16": 21_700_000,
    "lunit-vit-s-8": 21_700_000,
    "phikon": 86_000_000,
    "phikon-v2": 304_000_000,
    "plip": 86_000_000,
    "prov-gigapath": 1_100_000_000,
    "resnet50": 25_600_000,
    "retccl": 25_600_000,
    "uni": 304_000_000,
    "uni2-h": 632_000_000,
    "virchow": 632_000_000,
    "virchow-2": 632_000_000,
    "vit-b": 86_000_000,
    "vit-l": 304_000_000,
}


FAMILY_PROVIDER = {
    "atlas": ("Atlas", "Aignostics"),
    "chief-patch-mean": ("CHIEF", "Harvard Medical School"),
    "chief-slide": ("CHIEF", "Harvard Medical School"),
    "ciga": ("Ciga", "Ciga et al."),
    "clip-b": ("CLIP", "OpenAI"),
    "clip-l": ("CLIP", "OpenAI"),
    "dinov2-b": ("DINOv2", "Meta AI"),
    "dinov2-l": ("DINOv2", "Meta AI"),
    "dinov2-vit-g-14-initial": ("DINOv2", "Meta AI"),
    "dinov2-vit-g-14-natural-images": ("DINOv2", "Meta AI"),
    "dinov3-b": ("DINOv3", "Meta AI"),
    "dinov3-l": ("DINOv3", "Meta AI"),
    "dinov3-s": ("DINOv3", "Meta AI"),
    "exaone-path-2.5-slide": ("EXAONE Path", "LG AI Research"),
    "hipt": ("HIPT", "MahmoodLab"),
    "kaiko-vit-b-16": ("Kaiko", "Kaiko AI"),
    "kaiko-vit-b-8": ("Kaiko", "Kaiko AI"),
    "kaiko-vit-b-unspecified-patch": ("Kaiko", "Kaiko AI"),
    "kaiko-vit-l-14": ("Kaiko", "Kaiko AI"),
    "kaiko-vit-s-16": ("Kaiko", "Kaiko AI"),
    "kaiko-vit-s-8": ("Kaiko", "Kaiko AI"),
    "kaiko-vit-s-unspecified-patch": ("Kaiko", "Kaiko AI"),
    "kang-dino": ("Kang-DINO", "Kang et al."),
    "lunit-vit-s-16": ("Lunit-DINO", "Lunit"),
    "lunit-vit-s-8": ("Lunit-DINO", "Lunit"),
    "midnight-92k": ("Midnight", "Kaiko AI"),
    "midnight-92k-392": ("Midnight", "Kaiko AI"),
    "plip": ("PLIP", "PLIP authors"),
    "prism-slide": ("PRISM", "Paige"),
    "prov-gigapath": ("Prov-GigaPath", "Microsoft Research"),
    "quiltnet": ("QuiltNet", "Quilt AI"),
    "resnet50": ("ResNet", "Meta AI"),
    "retccl": ("RetCCL", "RetCCL authors"),
    "threads-slide": ("THREADS", "MahmoodLab"),
    "titan-slide": ("TITAN", "MahmoodLab"),
    "vit-b": ("ViT", "Google Research"),
    "vit-l": ("ViT", "Google Research"),
}


SOURCE_ALIASES = {
    "prov-gigapath": "prov-gigapath-tile",
    "titan-slide": "titan",
}
SLIDE_MODELS = {
    "chief-slide", "exaone-path-2.5-slide", "prism-slide",
    "prov-gigapath-slide", "threads-slide", "titan-slide",
}
VISION_LANGUAGE = {
    "clip-b", "clip-l", "conch", "conch-1.5", "musk", "plip",
    "quiltnet", "titan-slide",
}


def build_model_metadata(
    model_ids: Sequence[str],
    *,
    model_sources_path: str | Path,
    release_dates_path: str | Path,
) -> list[dict[str, object]]:
    """Merge citation ledgers without guessing unavailable parameter counts."""

    with Path(model_sources_path).open(newline="", encoding="utf-8") as handle:
        sources = {row["model_id"]: row for row in csv.DictReader(handle)}
    with Path(release_dates_path).open(newline="", encoding="utf-8") as handle:
        releases = {row["model_id"]: row for row in csv.DictReader(handle)}

    rows = []
    for model_id in model_ids:
        source = sources.get(SOURCE_ALIASES.get(model_id, model_id), {})
        release = releases.get(model_id, {})
        family = source.get("canonical_family", "").strip()
        provider = ""
        if model_id in FAMILY_PROVIDER:
            fallback_family, provider = FAMILY_PROVIDER[model_id]
            family = family or fallback_family
        elif family:
            provider = family
        parameter_count = PARAMETERS.get(model_id)
        primary_url = (
            source.get("primary_paper_url", "").strip()
            or release.get("primary_source_url", "").strip()
        )
        source_title = (
            source.get("primary_paper_title", "").strip()
            or release.get("source_title", "").strip()
        )
        rows.append({
            "model_id": model_id,
            "provider": provider,
            "family": family,
            "parameter_count": "" if parameter_count is None else parameter_count,
            "parameter_basis": "missing" if parameter_count is None else "nominal_encoder_count",
            "model_type": "slide_encoder" if model_id in SLIDE_MODELS else "tile_encoder",
            "modality": "vision_language" if model_id in VISION_LANGUAGE else "vision",
            "primary_source_url": primary_url,
            "source_title": source_title,
            "release_date": release.get("release_date", "").strip(),
            "verification_status": release.get("verification_status", "").strip(),
            "audit_notes": (
                "Parameter count intentionally unavailable; excluded from H1 denominator."
                if parameter_count is None
                else "Nominal encoder parameter count; slide aggregators and task heads are excluded."
            ),
        })
    return rows


def write_model_metadata(path: str | Path, rows: Sequence[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["model_id"]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_model_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["model_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate model_id in model metadata")
    return result
