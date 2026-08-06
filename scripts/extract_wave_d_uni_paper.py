#!/usr/bin/env python3
"""Expand every active UNI result row in arXiv:2308.15474 into leaf cells."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "26da25ced22b205570f480c3a562af31ed74bdaf679455f30cbcaa5dfdad4e60"
FIELDS = [
    "source_id", "source_revision", "source_sha256", "source_locator", "model_alias", "model_id",
    "evaluation_id", "dataset_id", "task_family", "metric", "value", "uncertainty",
    "embedding_recipe", "protocol", "reference_url", "disposition", "disposition_reason",
]

LIN = ("balanced_accuracy", "weighted_f1", "auroc")
KNN = ("one_nn_balanced_accuracy", "one_nn_weighted_f1", "twenty_nn_balanced_accuracy", "twenty_nn_weighted_f1")
RET = ("retrieval_accuracy_at_1", "retrieval_accuracy_at_3", "retrieval_accuracy_at_5", "retrieval_majority_vote_accuracy_at_5")
TOP = ("top1_accuracy", "top3_accuracy", "top5_accuracy", "weighted_f1", "auroc")


def spec(dataset: str, target: str, metrics: tuple[str, ...], context: int = 0,
         endpoint: str = "linear_probe", disposition: str = "accepted_public_leaf",
         reason: str = "Public dataset leaf under the exact frozen-feature protocol reported by the paper.") -> dict[str, object]:
    return dict(dataset=dataset, target=target, metrics=metrics, context=context, endpoint=endpoint,
                disposition=disposition, reason=reason)


SPECS: dict[str, dict[str, object]] = {
    "tab:patch-crc100k-lin": spec("crc100k", "crc_tissue_9class", LIN),
    "tab:patch-crc100k-knn": spec("crc100k", "crc_tissue_9class", KNN, endpoint="knn"),
    "tab:patch-ccrcc-lin": spec("tcga_hel_ccrcc", "ccrcc_tissue_3class", LIN),
    "tab:patch-ccrcc-knn": spec("tcga_hel_ccrcc", "ccrcc_tissue_3class", KNN, endpoint="knn"),
    "tab:patch-bach-lin": spec("bach", "brca_subtyping_4class", LIN, 1),
    "tab:patch-bach-knn": spec("bach", "brca_subtyping_4class", KNN, 1, "knn"),
    "tab:patch-hun-lin": spec("huncrc", "crc_tissue_9class", LIN),
    "tab:patch-huncrc-knn": spec("huncrc", "crc_tissue_9class", KNN, endpoint="knn"),
    "tab:patch-esca-lin": spec("ukk_wns_tcga_cha_esca", "esca_subtyping_11class", LIN),
    "tab:patch-esca-knn": spec("ukk_wns_tcga_cha_esca", "esca_subtyping_11class", KNN, endpoint="knn"),
    "tab:patch-unitopatho-lin": spec("unitopatho", "crc_polyp_6class", LIN, 1),
    "tab:patch-unitopatho-knn": spec("unitopatho", "crc_polyp_6class", KNN, 1, "knn"),
    "tab:patch-aggc-lin": spec("aggc", "prad_tissue_5class", ("balanced_accuracy", "weighted_kappa", "weighted_f1", "auroc")),
    "tab:patch-aggc-knn": spec("aggc", "prad_tissue_5class", KNN, endpoint="knn"),
    "tab:patch-msi-lin": spec("tcga_msi", "crc_msi_2class", LIN, 1),
    "tab:patch-msi-knn": spec("tcga_msi", "crc_msi_2class", KNN, 1, "knn"),
    "tab:patch-tcga-uniform-lin": spec("tcga_uniform", "pan_cancer_tissue_32class", LIN, 1),
    "tab:patch-tcga-uniform-knn": spec("tcga_uniform", "pan_cancer_tissue_32class", KNN, 1, "knn"),
    "tab:patch-tcga-tils-lin": spec("tcga_tils", "til_detection_2class", LIN, 1),
    "tab:patch-tcga-tils-knn": spec("tcga_tils", "til_detection_2class", KNN, 1, "knn"),
    "tab:aggc-retrieval": spec("aggc", "prad_tissue_5class", RET, endpoint="image_retrieval"),
    "tab:crc-retrieval": spec("crc100k", "crc_tissue_9class", RET, endpoint="image_retrieval"),
    "tab:esca-retrieval": spec("ukk_wns_tcga_cha_esca", "esca_subtyping_11class", RET, endpoint="image_retrieval"),
    "tab:huncrc-retrieval": spec("huncrc", "crc_tissue_9class", RET, endpoint="image_retrieval"),
    "tab:tcga-uniform-retrieval": spec("tcga_uniform", "pan_cancer_tissue_32class", RET, 1, "image_retrieval"),
    "tab:unitopatho-retrieval": spec("unitopatho", "crc_polyp_6class", RET, endpoint="image_retrieval"),
    "tab:patch-level-seg": spec("segpath", "cell_type_segmentation", ("dice", "precision", "recall"), 1, "mask2former_finetune", "quarantined_finetuned", "Paper explicitly fine-tunes every pretrained encoder with Mask2Former; retained outside the frozen-feature compression matrix."),
    "tab:slide-c16": spec("camelyon16", "breast_metastasis_2class", LIN, endpoint="abmil"),
    "tab:slide-nsclc": spec("tcga_cptac_nsclc", "nsclc_subtyping_2class", LIN, 1, "abmil"),
    "tab:slide-rcc": spec("tcga_cptac_dhmc_rcc", "rcc_subtyping_3class", LIN, 1, "abmil"),
    "tab:slide-dhmc": spec("dhmc_kidney", "rcc_subtyping_5class", LIN, endpoint="abmil"),
    "tab:slide-huncrc": spec("huncrc", "crc_screening_4class", LIN, endpoint="abmil"),
    "tab:slide-bracs-c": spec("bracs", "brca_coarse_3class", LIN, endpoint="abmil"),
    "tab:slide-bracs-f": spec("bracs", "brca_fine_7class", LIN, endpoint="abmil"),
    "tab:slide-idh": spec("tcga_ebrains", "idh1_mutation_2class", LIN, 1, "abmil"),
    "tab:slide-molsub": spec("tcga_ebrains", "gbmlgg_histomolecular_5class", LIN, 1, "abmil"),
    "tab:slide-ebrains-c": spec("ebrains", "brain_tumor_12class", LIN, endpoint="abmil"),
    "tab:slide-ebrains-f": spec("ebrains", "brain_tumor_30class", LIN, endpoint="abmil"),
    "tab:slide-panda": spec("panda", "isup_grading_6class", ("balanced_accuracy", "weighted_kappa", "weighted_f1", "auroc"), endpoint="abmil"),
    "tab:slide-emb": spec("bwh_endomyocardial", "allograft_rejection_2class", LIN, endpoint="abmil", disposition="quarantined_internal_cohort", reason="The paper explicitly identifies the BWH cohort as in-house."),
    "tab:proto-nsclc": spec("tcga_cptac_nsclc", "nsclc_subtyping_2class", ("balanced_accuracy", "weighted_f1"), 2, "mi_simpleshot"),
    "tab:proto-rcc": spec("tcga_cptac_dhmc_rcc", "rcc_subtyping_3class", ("balanced_accuracy", "weighted_f1"), 2, "mi_simpleshot"),
    "tab:ot-43-compare": spec("bwh_ot43", "cancer_type_43class", TOP, 1, "abmil", "quarantined_internal_cohort", "The paper explicitly identifies OT-43 as in-house BWH data."),
    "tab:ot-108-compare": spec("bwh_ot108", "oncotree_108class", TOP, 1, "abmil", "quarantined_internal_cohort", "The paper explicitly identifies OT-108 as in-house BWH data."),
}


def slug(value: str) -> str:
    value = value.replace("\\checkmark", "with").replace("\\xmark", "without")
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "default"


def clean(cell: str) -> str:
    return re.sub(r"\\(?:bfseries|rowcolor\{[^}]+\})", "", cell).replace("\\\\", "").strip()


def numeric(cell: str) -> tuple[str, str]:
    cleaned = clean(cell)
    match = re.search(r"(?<![\d.])([01]\.\d{3})(?!\d)", cleaned)
    if not match:
        raise ValueError(f"no point estimate in {cell!r}")
    rest = cleaned[match.end():].strip()
    uncertainty = "not_reported"
    if rest.startswith("(") and ")" in rest:
        uncertainty = rest[:rest.index(")") + 1].strip("()")
    return match.group(1), uncertainty


def table_blocks(text: str) -> list[tuple[int, str, list[tuple[int, str]]]]:
    blocks = []
    for match in re.finditer(r"\\begin\{table\}(.*?)\\end\{table\}", text, re.S):
        block = match.group(1)
        label = re.search(r"\\label\{([^}]+)\}", block)
        if not label:
            continue
        start_line = text[:match.start()].count("\n") + 1
        rows = []
        for offset, line in enumerate(block.splitlines()):
            if re.match(r"^\s*UNI\s*&", line):
                rows.append((start_line + offset, line))
        if rows:
            blocks.append((start_line, label.group(1), rows))
    return blocks


def extract(source_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    active_row_count = 0
    tables = source_root / "tables"
    for path in sorted(tables.glob("tables_*.tex")):
        text = path.read_text(encoding="utf-8")
        for _, label, uni_rows in table_blocks(text):
            if label not in SPECS:
                raise ValueError(f"unclassified active UNI table: {label}")
            meta = SPECS[label]
            for line_no, line in uni_rows:
                active_row_count += 1
                cells = [clean(cell) for cell in line.split("&")]
                context_count = int(meta["context"])
                contexts = cells[1:1 + context_count]
                metric_cells = cells[1 + context_count:]
                metrics = tuple(meta["metrics"])
                if len(metric_cells) != len(metrics):
                    raise ValueError(f"metric arity mismatch for {label}:{line_no}: {metric_cells}")
                context_slug = ".".join(slug(item) for item in contexts) if contexts else "default"
                dataset_id = str(meta["dataset"])
                if contexts and label in {"tab:slide-nsclc", "tab:slide-rcc", "tab:slide-idh", "tab:slide-molsub", "tab:proto-nsclc", "tab:proto-rcc"}:
                    dataset_id += "_" + slug(contexts[0])
                for metric, cell in zip(metrics, metric_cells):
                    value, uncertainty = numeric(cell)
                    disposition = str(meta["disposition"])
                    reason = str(meta["reason"])
                    if label == "tab:patch-level-seg" and contexts[0].lower() == "average":
                        disposition = "excluded_derived_aggregate"
                        reason = "Average over eight cell-type tasks is derived, not an independent evaluation leaf."
                    evaluation_id = f"unipaper2023.{slug(label.removeprefix('tab:'))}.{context_slug}.{metric}"
                    context_desc = ", ".join(contexts) if contexts else "default setting"
                    protocol = f"{meta['endpoint']} on {dataset_id}; paper table {label}; row setting: {context_desc}."
                    rows.append({
                        "source_id": "uni_arxiv", "source_revision": "arxiv:2308.15474",
                        "source_sha256": SOURCE_SHA,
                        "source_locator": f"tables/{path.name}:{line_no}|{label}|setting={context_desc}|metric={metric}",
                        "model_alias": "UNI", "model_id": "uni", "evaluation_id": evaluation_id,
                        "dataset_id": dataset_id, "task_family": str(meta["target"]), "metric": metric,
                        "value": value, "uncertainty": uncertainty, "embedding_recipe": "frozen_uni_features" if disposition != "quarantined_finetuned" else "finetuned_uni_encoder",
                        "protocol": protocol, "reference_url": "https://arxiv.org/abs/2308.15474",
                        "disposition": disposition, "disposition_reason": reason,
                    })
    if active_row_count != 83:
        raise ValueError(f"expected 83 active UNI result rows, found {active_row_count}")
    if len(rows) != 277:
        raise ValueError(f"expected 277 UNI metric leaves, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/tmp/wave_d_sources/2308.15474"))
    parser.add_argument("--output", type=Path, default=ROOT / "source_data/wave_d_uni_paper_2308.15474.csv")
    args = parser.parse_args()
    rows = extract(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    from collections import Counter
    print({"active_rows": 83, "leaf_cells": len(rows), "dispositions": dict(Counter(row["disposition"] for row in rows))})


if __name__ == "__main__":
    main()
