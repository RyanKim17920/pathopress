#!/usr/bin/env python3
"""Build the exhaustive Prov-GigaPath tile-encoder evidence disposition ledger."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = (
    "evidence_id", "source", "source_locator", "dataset_access", "model_component",
    "downstream_component", "protocol", "exact_numeric_result", "eligible_public_leaf",
    "exclusion_reason", "reference_url", "pinned_revision", "notes",
)
PAPER = "https://www.nature.com/articles/s41586-024-07441-w"
SUPP = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-07441-w/MediaObjects/41586_2024_7441_MOESM1_ESM.pdf"
HF_MODEL = "https://huggingface.co/prov-gigapath/prov-gigapath"
HF_DATA = "https://huggingface.co/datasets/prov-gigapath/prov-gigapath-tile-embeddings"
REPO = "https://github.com/prov-gigapath/prov-gigapath"
MAIN_SHA = "52eda6b291c49c85274f034462e3e0d168d8bca505d845bed648fabae4192b76"
SUPP_SHA = "b8279191eb635a60edd51621eab931f04d8d75b7af291794d8b91b58f73b866b"
REPO_REV = "92d8d20430f9645675db01c5f55e95ad58538526"
HF_MODEL_REV = "0dd9f5561e6b98d27c1d7a919c252d75c2eb66cb"
HF_DATA_REV = "0c5eebef6041f990eb04e1f6f9d835958b0b941f"


def row(
    evidence_id: str, source: str, locator: str, access: str, component: str,
    downstream: str, protocol: str, exact: str, reason: str, url: str,
    revision: str, notes: str,
) -> dict[str, str]:
    return {
        "evidence_id": evidence_id, "source": source, "source_locator": locator,
        "dataset_access": access, "model_component": component,
        "downstream_component": downstream, "protocol": protocol,
        "exact_numeric_result": exact, "eligible_public_leaf": "no",
        "exclusion_reason": reason, "reference_url": url,
        "pinned_revision": revision, "notes": notes,
    }


def audit(main_text: str, supp_text: str, repo: Path) -> list[dict[str, str]]:
    anchors = {
        main_text: (
            "we froze the tile encoder and", "directly applied one ABMIL layer",
            "Supplementary Fig. 4", "GigaPath pretrained on Prov-Path",
        ),
        supp_text: (
            "Comparison of different tile-level pretraining methods",
            "LUAD 5-gene mutation prediction in TCGA", "Supplementary Table 2",
            "Prov-GigaPath w. ABMIL",
        ),
    }
    for text, required in anchors.items():
        missing = [value for value in required if value not in text]
        if missing:
            raise ValueError(f"Prov-GigaPath evidence anchors missing: {missing}")
    readme = (repo / "README.md").read_text(encoding="utf-8")
    required_files = (
        repo / "linear_probe/main.py", repo / "scripts/run_pcam.sh",
        repo / "finetune/main.py", repo / "scripts/run_panda.sh",
    )
    if "Tile-Level Linear Probing Example Using PCam Dataset" not in readme:
        raise ValueError("official repository PCam example anchor missing")
    if any(not path.exists() for path in required_files):
        raise ValueError("official repository benchmark example files missing")

    return [
        row(
            "nature_supp_fig4", "Nature supplement", "Supplementary Figure 4",
            "public_tcga_luad", "Prov-GigaPath DINOv2 tile encoder",
            "LongNet slide encoder plus task head", "tile-pretraining-method ablation; ten-run WSI mutation prediction",
            "graphical_bars_without_exact_labels", "not_tile_encoder_only_and_no_exact_numeric_table", SUPP,
            f"sha256:{SUPP_SHA}",
            "Public TCGA evaluation, but the changed tile pretraining is assessed through the full task-trained WSI stack.",
        ),
        row(
            "nature_extended_fig6", "Nature main article", "Extended Data Figure 6",
            "public_tcga_luad", "GigaPath tile plus LongNet encoders pretrained on Prov-Path or TCGA",
            "task-trained LongNet/ABMIL/classifier", "pretraining-corpus ablation; ten-run WSI mutation prediction",
            "graphical_bars_without_exact_labels", "whole_slide_ablation_not_tile_only", PAPER,
            f"sha256:{MAIN_SHA}",
            "Cannot attribute the plotted result to the tile encoder independently of slide pretraining and downstream training.",
        ),
        row(
            "nature_extended_fig7", "Nature main article", "Extended Data Figure 7",
            "mixed_public_tcga_and_private_providence", "Prov-GigaPath tile encoder",
            "GigaPath LongNet versus HIPT hierarchy plus task heads", "whole-slide architecture comparison on mutation tasks",
            "graphical_bars_without_exact_labels", "whole_slide_architecture_comparison", PAPER,
            f"sha256:{MAIN_SHA}", "Architecture comparison is not a frozen tile-feature benchmark.",
        ),
        row(
            "nature_extended_fig8", "Nature main article", "Extended Data Figure 8",
            "private_providence", "Prov-GigaPath tile encoder",
            "GigaPath LongNet versus HIPT hierarchy plus task heads", "whole-slide cancer-subtyping comparison",
            "graphical_bars_without_exact_labels", "private_and_whole_slide_architecture_comparison", PAPER,
            f"sha256:{MAIN_SHA}", "All evaluated subtyping cohorts in this figure are Providence cohorts.",
        ),
        row(
            "nature_supp_fig5", "Nature supplement", "Supplementary Figure 5",
            "private_providence", "frozen Prov-GigaPath tile encoder",
            "pretrained/frozen/random LongNet or external ABMIL plus classifier", "slide-component ablation on cancer subtyping",
            "graphical_bars_without_exact_labels", "private_slide_component_ablation", SUPP,
            f"sha256:{SUPP_SHA}", "Ablates LongNet and ABMIL; it does not report an isolated tile-encoder score.",
        ),
        row(
            "nature_supp_table2_tcga", "Nature supplement", "Supplementary Table 2; five TCGA-LUAD rows",
            "public_tcga_luad", "frozen Prov-GigaPath tile encoder",
            "task-specifically fine-tuned LongNet slide encoder plus ABMIL/classifier", "ten-fold WSI mutation prediction",
            "five_exact_auroc_means_with_standard_errors", "task_specific_slide_encoder_finetuning", SUPP,
            f"sha256:{SUPP_SHA}", "Exact cells are already retained in the Group C quarantine ledger, not assigned to the tile encoder.",
        ),
        row(
            "repo_pcam_example", "Official GitHub repository", "README Fine-tuning / Tile-Level Linear Probing Example",
            "public_pcam", "frozen Prov-GigaPath tile encoder", "linear classifier",
            "runnable tile-level linear-probe example", "none_published", "runnable_example_without_reference_result", REPO,
            f"git:{REPO_REV}", "The repository provides code and embeddings but no checked-in result or expected score.",
        ),
        row(
            "repo_linear_probe", "Official GitHub repository", "linear_probe/main.py and scripts/run_pcam.sh",
            "public_pcam", "frozen Prov-GigaPath tile encoder", "linear classifier",
            "code computes weighted F1, macro precision/recall, AUROC and AUPRC", "none_checked_in",
            "result_producer_without_reported_result", REPO, f"git:{REPO_REV}",
            "Runtime results.txt is locally generated; no official output artifact is committed.",
        ),
        row(
            "hf_model_card_pcam", "Official Hugging Face model card", "Fine-tuning / Tile-Level Linear Probing Example",
            "public_pcam", "frozen Prov-GigaPath tile encoder", "linear classifier",
            "runnable PCam example", "none_published", "model_card_without_reference_result", HF_MODEL,
            f"hf:{HF_MODEL_REV}",
            "The current gated card points to the official repository and embedding dataset but supplies no score table.",
        ),
        row(
            "hf_embeddings_pcam", "Official Hugging Face dataset", "GigaPath_PCam_embeddings.zip",
            "public_pcam", "frozen Prov-GigaPath tile embeddings", "none",
            "pre-extracted feature artifact", "none_published", "features_without_labels_or_reference_result", HF_DATA,
            f"hf:{HF_DATA_REV}", "A 2.33 GB feature archive is provided; the dataset has no result card or score artifact.",
        ),
        row(
            "repo_panda_example", "Official GitHub repository", "README and scripts/run_panda.sh",
            "public_panda", "Prov-GigaPath tile encoder", "LongNet slide encoder plus task head",
            "runnable slide-level fine-tuning example", "none_published", "whole_slide_example_without_reference_result", REPO,
            f"git:{REPO_REV}", "PANDA is not a tile-only evaluation and no official reference score is checked in.",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-text", type=Path, required=True)
    parser.add_argument("--supp-text", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = audit(args.main_text.read_text(encoding="utf-8"), args.supp_text.read_text(encoding="utf-8"), args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} audited evidence dispositions; 0 eligible public tile-only numeric leaves")


if __name__ == "__main__":
    main()
