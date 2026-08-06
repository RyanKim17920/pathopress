from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from scripts.evidence.wave_d_hoptimus_report import build_protocols, build_scores, load_evidence


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "source_data/wave_d_hoptimus1_official_report_2025.csv"


def test_wave_d_report_dispositions_and_models() -> None:
    rows = load_evidence(SNAPSHOT)
    assert Counter(row["disposition"] for row in rows) == {
        "accepted_public_leaf": 95,
        "quarantined_nonpublic_or_access_ambiguous": 60,
        "excluded_derived_aggregate": 15,
    }
    assert Counter(row["model_id"] for row in rows) == {
        "h-optimus-1": 34, "h-optimus-0": 34, "uni2-h": 34,
        "virchow-2": 34, "uni": 34,
    }


def test_wave_d_public_registry_cells_are_exact_and_versioned() -> None:
    scores, _ = build_scores(SNAPSHOT)
    assert len(scores) == 95
    assert Counter(row["model_id"] for row in scores) == {
        "h-optimus-1": 19, "h-optimus-0": 19, "uni2-h": 19,
        "virchow-2": 19, "uni": 19,
    }
    by_key = {(row["model_id"], row["evaluation_id"]): row for row in scores}
    assert by_key[("h-optimus-1", "hest.hoptimus1report2025.idc.gene_expression")]["value"] == "0.602"
    assert by_key[("h-optimus-0", "hoptimus1report2025.meta_bc.camelyon16_test.abmil_auc")]["value"] == "0.998"
    assert by_key[("virchow-2", "hoptimus1report2025.mhist.linear_probe_top1")]["value"] == "0.851"


def test_wave_d_protocols_preserve_hest_identity_but_not_protocol() -> None:
    with (ROOT / "data/tasks.csv").open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle))
    protocols = build_protocols(SNAPSHOT, tasks)
    assert len(protocols) == 19
    base = next(row for row in tasks if row["evaluation_id"] == "hest.idc.gene_expression")
    report = next(row for row in protocols if row["evaluation_id"] == "hest.hoptimus1report2025.idc.gene_expression")
    assert report["task_identity_id"] == base["task_identity_id"]
    assert report["protocol_id"] != base["protocol_id"]
    tile_links = {
        "hoptimus1report2025.cam17_wilds.linear_probe_top1": "thunder.wilds.linear_probing",
        "hoptimus1report2025.crc_no_norm.linear_probe_top1": "thunder.crc.linear_probing",
        "hoptimus1report2025.mhist.linear_probe_top1": "thunder.mhist.linear_probing",
        "hoptimus1report2025.tcga_uniform.linear_probe_top1": "thunder.tcga_uniform.linear_probing",
    }
    by_evaluation = {row["evaluation_id"]: row for row in [*tasks, *protocols]}
    for report_id, base_id in tile_links.items():
        assert by_evaluation[report_id]["task_identity_id"] == by_evaluation[base_id][
            "task_identity_id"
        ]
