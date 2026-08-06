#!/usr/bin/env python3
"""Extract canonical PathoROB paper scores and versioned repository examples.

The Nature Source Data workbook contains complete 20-model APD and clustering
tables.  The pinned PathoROB repository contains only two-model example result
artifacts.  They are deliberately emitted under different protocol IDs because
their values are not byte-for-byte identical to the published paper results.

This extractor uses only the Python standard library, including a small XLSX
reader for the two sheets needed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath


PAPER_REFERENCE = "https://www.nature.com/articles/s41467-026-73923-2"
PMC_SOURCE_DATA = (
    "https://pmc.ncbi.nlm.nih.gov/articles/instance/13260997/bin/"
    "41467_2026_73923_MOESM4_ESM.xlsx"
)
PAPER_WORKBOOK_SHA256 = "07456f3ffc5270ea1d8d48a8f82c08a5be396c88f99cc0227968dad721943047"
PATHOROB_COMMIT = "6583cf0b0d902c8cc032308262fa3a3befdc0687"
T_975_DF59 = 2.0009953780882674

MODEL_IDS = {
    "Atlas": "atlas",
    "Ciga": "ciga",
    "CONCH": "conch",
    "CONCHv1.5": "conch-1.5",
    "Conch": "conch",
    "Conch1.5": "conch-1.5",
    "CTransPath": "ctranspath",
    "H-optimus-0": "h-optimus-0",
    "H0-mini": "h0-mini",
    "HIPT": "hipt",
    "Kaiko ViT-B/8": "kaiko-vit-b-8",
    "KaikoViT-B/8": "kaiko-vit-b-8",
    "Kaiko": "kaiko-vit-b-8",
    "Kang-DINO": "kang-dino",
    "MUSK": "musk",
    "Musk": "musk",
    "Phikon": "phikon",
    "Phikon-v2": "phikon-v2",
    "Phikon2": "phikon-v2",
    "Prov-GigaPath": "prov-gigapath",
    "ProvGigapath": "prov-gigapath",
    "RetCCL": "retccl",
    "RudolfV": "rudolfv",
    "UNI": "uni",
    "UNI-2h": "uni2-h",
    "UNI2-h": "uni2-h",
    "UNI2": "uni2-h",
    "Virchow": "virchow",
    "Virchow2": "virchow-2",
}

REPO_MODELS = {
    "uni2h_clsmean": ("UNI2-h", "uni2-h"),
    "phikonv2_clsmean": ("Phikon-v2", "phikon-v2"),
}

DATASETS = {
    "Camelyon16_2x2": "camelyon",
    "TCGA_4x4": "tcga_4x4",
    "Tolkach_2x2": "tolkach_esca",
    "camelyon": "camelyon",
    "tcga": "tcga_4x4",
    "tolkach_esca": "tolkach_esca",
}

FIELDS = (
    "source_scope",
    "source_table",
    "source_row",
    "model_alias",
    "model_id",
    "dataset_scope",
    "endpoint",
    "evaluation_id",
    "metric",
    "value",
    "value_unit",
    "uncertainty",
    "direction",
    "protocol",
    "reference_url",
    "source_locator",
    "source_revision",
    "source_sha256",
    "inclusion_status",
    "inclusion_reason",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _column_index(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    value = 0
    for character in letters:
        value = value * 26 + ord(character.upper()) - ord("A") + 1
    return value - 1


def read_xlsx(path: Path, wanted: set[str]) -> dict[str, list[list[object | None]]]:
    """Read selected worksheet values without importing an XLSX dependency."""
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        package_relationships = {
            item.attrib["Id"]: item.attrib["Target"] for item in relationships
        }
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.text or "" for node in item.iter() if node.tag.endswith("}t")) for item in strings]
        sheets: dict[str, list[list[object | None]]] = {}
        for sheet in workbook.iter():
            if not sheet.tag.endswith("}sheet") or sheet.attrib.get("name") not in wanted:
                continue
            name = sheet.attrib["name"]
            relationship_id = next(value for key, value in sheet.attrib.items() if key.endswith("}id"))
            target = package_relationships[relationship_id]
            member = str(PurePosixPath("xl") / target) if not target.startswith("/") else target.lstrip("/")
            member = str(PurePosixPath(member))
            root = ET.fromstring(archive.read(member))
            rows: list[list[object | None]] = []
            for row in (node for node in root.iter() if node.tag.endswith("}row")):
                values: dict[int, object | None] = {}
                for cell in (node for node in row if node.tag.endswith("}c")):
                    index = _column_index(cell.attrib["r"])
                    kind = cell.attrib.get("t")
                    value_node = next((node for node in cell if node.tag.endswith("}v")), None)
                    if kind == "inlineStr":
                        value: object | None = "".join(
                            node.text or "" for node in cell.iter() if node.tag.endswith("}t")
                        )
                    elif value_node is None or value_node.text is None:
                        value = None
                    elif kind == "s":
                        value = shared[int(value_node.text)]
                    elif kind in {"str", "e"}:
                        value = value_node.text
                    else:
                        value = float(value_node.text)
                    values[index] = value
                if values:
                    rows.append([values.get(index) for index in range(max(values) + 1)])
            sheets[name] = rows
    missing = wanted - set(sheets)
    if missing:
        raise ValueError(f"missing workbook sheets: {sorted(missing)}")
    return sheets


def records(rows: list[list[object | None]]) -> list[dict[str, object | None]]:
    header = [str(value) for value in rows[0]]
    return [dict(zip(header, row)) for row in rows[1:]]


def paper_rows(workbook_path: Path) -> list[dict[str, str]]:
    digest = sha256(workbook_path)
    if digest != PAPER_WORKBOOK_SHA256:
        raise ValueError(f"unexpected PathoROB Source Data SHA-256: {digest}")
    names = {
        "Fig-3c-apds",
        "Fig-3d-correlation_apds_x_ri",
        "Fig-6b-clustering",
    }
    sheets = read_xlsx(workbook_path, names)
    apd_observations = records(sheets["Fig-3c-apds"])
    apd_summary = records(sheets["Fig-3d-correlation_apds_x_ri"])
    clustering = records(sheets["Fig-6b-clustering"])
    output: list[dict[str, str]] = []

    by_model: dict[str, list[dict[str, object | None]]] = {}
    for row in apd_observations:
        raw_alias = str(row["foundation_model"])
        by_model.setdefault(MODEL_IDS[raw_alias], []).append(row)
    if len(apd_summary) != 20 or any(len(rows) != 60 for rows in by_model.values()):
        raise ValueError("PathoROB APD source table is not 20 models x 60 observations")
    for source_row, row in enumerate(apd_summary, start=2):
        alias = str(row["foundation_model"])
        model_id = MODEL_IDS[alias]
        observations = by_model[model_id]
        for endpoint, source_column in (
            ("apd_id", "id_score_after_correction"),
            ("apd_ood", "ood_score_after_correction"),
        ):
            values = [float(item[source_column]) for item in observations]
            value = float(row[f"{endpoint} [%]"])
            if not math.isclose(statistics.mean(values), value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"APD summary/raw mismatch for {alias}/{endpoint}")
            half_width = T_975_DF59 * statistics.stdev(values) / math.sqrt(len(values))
            output.append(
                {
                    "source_scope": "published_paper",
                    "source_table": "Fig-3d-correlation_apds_x_ri",
                    "source_row": str(source_row),
                    "model_alias": alias,
                    "model_id": model_id,
                    "dataset_scope": "camelyon+tcga_4x4+tolkach_esca",
                    "endpoint": endpoint,
                    "evaluation_id": f"pathorob.nature2026.all_datasets.{endpoint}",
                    "metric": "average_performance_drop_percent",
                    "value": repr(value),
                    "value_unit": "percent_relative_change",
                    "uncertainty": (
                        f"95%_ci_half_width={half_width:.15g};n=60;df=59;"
                        "dataset_variance_corrected=true"
                    ),
                    "direction": "higher_closer_to_zero",
                    "protocol": (
                        "Mean signed relative accuracy change from the balanced split across "
                        "all nonbaseline spurious-correlation splits, 20 repetitions, and three "
                        "datasets; ID and OOD are distinct endpoints."
                    ),
                    "reference_url": PAPER_REFERENCE,
                    "source_locator": (
                        "41467_2026_73923_MOESM4_ESM.xlsx|sheet="
                        f"Fig-3d-correlation_apds_x_ri|row={source_row};ci_source=Fig-3c-apds"
                    ),
                    "source_revision": "PMC13260997.1",
                    "source_sha256": digest,
                    "inclusion_status": "canonical_analysis_ineligible",
                    "inclusion_reason": (
                        "Primary published mean is registry-eligible, but signed APD has no "
                        "source-defined bounded common-scale normalization; arbitrary clipping "
                        "or rescaling is forbidden."
                    ),
                }
            )

    if len(clustering) != 60:
        raise ValueError("PathoROB clustering source table is not 20 models x 3 datasets")
    counts: dict[str, int] = {}
    for source_row, row in enumerate(clustering, start=2):
        alias = str(row["foundation_model"])
        model_id = MODEL_IDS[alias]
        dataset = DATASETS[str(row["dataset"])]
        counts[dataset] = counts.get(dataset, 0) + 1
        value = float(row["clustering_score_mean"])
        std = float(row["clustering_score_std"])
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"clustering score outside paper domain: {value}")
        output.append(
            {
                "source_scope": "published_paper",
                "source_table": "Fig-6b-clustering",
                "source_row": str(source_row),
                "model_alias": alias,
                "model_id": model_id,
                "dataset_scope": dataset,
                "endpoint": "clustering_score",
                "evaluation_id": f"pathorob.nature2026.{dataset}.clustering_score",
                "metric": "clustering_score",
                "value": repr(value),
                "value_unit": "ari_biology_minus_ari_medical_center",
                "uncertainty": f"standard_deviation={std:.15g};n=50_random_initializations",
                "direction": "higher",
                "protocol": (
                    "K-means with cosine distance; K selected by maximum silhouette score; "
                    "clustering score is ARI(biology)-ARI(medical center), averaged over "
                    "balanced 2x2 combinations and 50 random initializations."
                ),
                "reference_url": PAPER_REFERENCE,
                "source_locator": (
                    "41467_2026_73923_MOESM4_ESM.xlsx|sheet=Fig-6b-clustering|"
                    f"row={source_row}"
                ),
                "source_revision": "PMC13260997.1",
                "source_sha256": digest,
                "inclusion_status": "canonical_analysis_eligible",
                "inclusion_reason": (
                    "Primary published mean with an explicit approximately [-1,1] domain, "
                    "higher-is-better direction, and auditable affine normalization."
                ),
            }
        )
    if counts != {"camelyon": 20, "tcga_4x4": 20, "tolkach_esca": 20}:
        raise ValueError(f"unexpected clustering dataset coverage: {counts}")
    if len(output) != 100:
        raise AssertionError(f"published score extraction drifted: {len(output)}")
    return output


def repo_reference(relative: Path) -> str:
    return (
        "https://github.com/bifold-pathomics/PathoROB/blob/"
        f"{PATHOROB_COMMIT}/{relative.as_posix()}"
    )


def repository_example_rows(repo: Path) -> list[dict[str, str]]:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot verify PathoROB source revision: {repo}") from exc
    if revision != PATHOROB_COMMIT:
        raise ValueError(f"unexpected PathoROB revision: {revision}")
    output: list[dict[str, str]] = []
    reason = (
        "Official repository example only; two-model values differ slightly from the "
        "published Source Data and are kept as a version-specific audit block."
    )
    for directory, (alias, model_id) in REPO_MODELS.items():
        apd_root = repo / "results" / "apd" / directory
        for dataset_name in ("camelyon", "tcga", "tolkach_esca"):
            dataset = DATASETS[dataset_name]
            relative = (apd_root / f"{dataset_name}_summary.json").relative_to(repo)
            payload = json.loads((repo / relative).read_text(encoding="utf-8"))
            for endpoint in ("apd_id", "apd_ood"):
                output.append(
                    {
                        "source_scope": "repository_example",
                        "source_table": "results/apd/*/*_summary.json",
                        "source_row": endpoint,
                        "model_alias": alias,
                        "model_id": model_id,
                        "dataset_scope": dataset,
                        "endpoint": endpoint,
                        "evaluation_id": f"pathorob.repoexample2026.{dataset}.{endpoint}",
                        "metric": "average_performance_drop_fraction",
                        "value": repr(float(payload[endpoint])),
                        "value_unit": "fractional_relative_change",
                        "uncertainty": "not_reported",
                        "direction": "higher_closer_to_zero",
                        "protocol": "Pinned repository example: per-dataset signed APD mean over repetitions.",
                        "reference_url": repo_reference(relative),
                        "source_locator": relative.as_posix(),
                        "source_revision": revision,
                        "source_sha256": sha256(repo / relative),
                        "inclusion_status": "quarantined_repository_example",
                        "inclusion_reason": reason,
                    }
                )
        relative = (apd_root / "aggregated_summary.json").relative_to(repo)
        payload = json.loads((repo / relative).read_text(encoding="utf-8"))
        for endpoint, ci in (("apd_id", "ci_id"), ("apd_ood", "ci_ood")):
            output.append(
                {
                    "source_scope": "repository_example",
                    "source_table": "results/apd/*/aggregated_summary.json",
                    "source_row": endpoint,
                    "model_alias": alias,
                    "model_id": model_id,
                    "dataset_scope": "+".join(payload["aggregation_datasets"]),
                    "endpoint": endpoint,
                    "evaluation_id": f"pathorob.repoexample2026.all_datasets.{endpoint}",
                    "metric": "average_performance_drop_fraction",
                    "value": repr(float(payload[endpoint])),
                    "value_unit": "fractional_relative_change",
                    "uncertainty": f"95%_ci_half_width={float(payload[ci]):.15g}",
                    "direction": "higher_closer_to_zero",
                    "protocol": "Pinned repository example: variance-corrected aggregate over three datasets.",
                    "reference_url": repo_reference(relative),
                    "source_locator": relative.as_posix(),
                    "source_revision": revision,
                    "source_sha256": sha256(repo / relative),
                    "inclusion_status": "quarantined_repository_example",
                    "inclusion_reason": reason,
                }
            )
        clustering_root = repo / "results" / "clustering_score" / directory
        for dataset_name in ("camelyon", "tcga", "tolkach_esca"):
            dataset = DATASETS[dataset_name]
            relative = (clustering_root / dataset_name / "results_summary.json").relative_to(repo)
            payload = json.loads((repo / relative).read_text(encoding="utf-8"))
            output.append(
                {
                    "source_scope": "repository_example",
                    "source_table": "results/clustering_score/*/*/results_summary.json",
                    "source_row": "clustering_score_mean",
                    "model_alias": alias,
                    "model_id": model_id,
                    "dataset_scope": dataset,
                    "endpoint": "clustering_score",
                    "evaluation_id": f"pathorob.repoexample2026.{dataset}.clustering_score",
                    "metric": "clustering_score",
                    "value": repr(float(payload["clustering_score_mean"])),
                    "value_unit": "ari_biology_minus_ari_medical_center",
                    "uncertainty": (
                        "mean_per_combination_standard_deviation="
                        f"{float(payload['clustering_score_mean_std']):.15g};"
                        f"n_trials={int(payload['num_trials'])}"
                    ),
                    "direction": "higher",
                    "protocol": "Pinned repository example clustering result with silhouette-selected K.",
                    "reference_url": repo_reference(relative),
                    "source_locator": relative.as_posix(),
                    "source_revision": revision,
                    "source_sha256": sha256(repo / relative),
                    "inclusion_status": "quarantined_repository_example",
                    "inclusion_reason": reason,
                }
            )
    if len(output) != 22:
        raise AssertionError(f"repository-example extraction drifted: {len(output)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--pathorob", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = paper_rows(args.workbook) + repository_example_rows(args.pathorob)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} PathoROB score-evidence rows to {args.output}")


if __name__ == "__main__":
    main()
