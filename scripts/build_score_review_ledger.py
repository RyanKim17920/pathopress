#!/usr/bin/env python3
"""Reparse every pinned score source and build an automated review ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pathopress.score_review import (  # noqa: E402
    LEDGER_FIELDS,
    REVIEWED_AT,
    REVIEWER_TYPE,
    duplicate_memberships,
    read_csv,
    row_key,
    sha256_path,
    validate_ledger,
    write_summary,
)
from scripts import build_registry as registry  # noqa: E402
from scripts.evidence.eva_scores import (  # noqa: E402
    merge_scores as merge_eva_scores,
    parse_midnight_scores,
    parse_repository_scores,
)


SOURCE_SPECS = {
    "eva@": ("repository", "eva/tools/data/leaderboards/pathology.csv", "eva"),
    "eva_midnight@": ("model_card", "eva_midnight/README.md", "eva_midnight"),
    "thunder@": ("repository", "thunder/docs/leaderboards.md", "thunder"),
    "hest@": ("repository", "hest/README.md", "hest"),
    "pathorob@": ("repository", "pathorob/README.md", "pathorob"),
    "arxiv:2512.14019v1": ("paper_snapshot", "source_data/exaone_path_2_5_pathobench_2512.14019v1.csv", "arxiv:2512.14019v1"),
    "arxiv:2501.16652v1": ("paper_snapshot", "source_data/threads_pathobench_2501.16652v1.csv", "arxiv:2501.16652v1"),
    "PMC13260997.1": ("paper_snapshot", "source_data/pathorob_nature2026_and_repo_examples.csv", "PMC13260997.1"),
    "arxiv:2501.16239v3": ("paper_snapshot", "source_data/h0mini_uni2h_official_scores_2025.csv", "arxiv:2501.16239v3"),
    "git:5ec9511893af993f6faa099f093d1924b291aed2": ("repository_snapshot", "source_data/h0mini_uni2h_official_scores_2025.csv", "git:5ec9511893af993f6faa099f093d1924b291aed2"),
    "git:42715efc11722a496e0a67f3369505a8f277206c": ("repository_snapshot", "source_data/h0mini_uni2h_official_scores_2025.csv", "git:42715efc11722a496e0a67f3369505a8f277206c"),
    "official PDF@": ("paper_snapshot", "source_data/virchow2g_gigapath_titan_official_scores_2024_2025.csv", "official-primary-pdfs"),
    "official Bioptimus report@": ("first_party_report_snapshot", "source_data/wave_d_hoptimus1_official_report_2025.csv", "bioptimus-report-2025"),
    "arxiv:2408.00738 source@": ("paper_snapshot", "source_data/wave_d_virchow2_paper_2408.00738.csv", "arxiv:2408.00738"),
    "arxiv:2309.07778 source@": ("paper_snapshot", "source_data/wave_d_virchow_paper_2309.07778.csv", "arxiv:2309.07778"),
    "arxiv:2308.15474 source@": ("paper_snapshot", "source_data/wave_d_uni_paper_2308.15474.csv", "arxiv:2308.15474"),
    "group-b:genbio_pathfm_official_2026.csv@": ("paper_snapshot", "source_data/genbio_pathfm_official_2026.csv", "genbio-pathfm-2026"),
    "group-b:midnight_miccai2025_official_scores.csv@": ("paper_snapshot", "source_data/midnight_miccai2025_official_scores.csv", "miccai-2025-4651"),
    "group-b:openmidnight_technical_report_2025.csv@": ("first_party_report_snapshot", "source_data/openmidnight_technical_report_2025.csv", "sophont-tr-2025-001"),
    "wave-e:conch_official_scores_2024.csv@": ("paper_snapshot", "source_data/conch_official_scores_2024.csv", "conch-nature-medicine-2024"),
    "wave-e:conch15_titan_official_scores_2025.csv@": ("paper_snapshot", "source_data/conch15_titan_official_scores_2025.csv", "titan-nature-medicine-2025"),
    "wave-e:phikon_family_official_scores_2023_2024.csv@": ("paper_snapshot", "source_data/phikon_family_official_scores_2023_2024.csv", "phikon-family-primary-papers"),
    "wave-f:hibou_official_scores_2024.csv@": ("paper_snapshot", "source_data/hibou_official_scores_2024.csv", "arxiv:2406.05074v1"),
    "wave-f:musk_official_scores_2025.csv@": ("paper_snapshot", "source_data/musk_official_scores_2025.csv", "nature-638-2025"),
    "wave-f:gpfm_official_scores_2025.csv@": ("paper_snapshot", "source_data/gpfm_official_scores_2025.csv", "nature-biomedical-engineering-2025"),
}

PINNED_SOURCE_DESTINATIONS = {
    "eva/tools/data/leaderboards/pathology.csv": "source_data/pinned/eva/pathology.csv",
    "eva_midnight/README.md": "source_data/pinned/eva_midnight/README.md",
    "thunder/docs/leaderboards.md": "source_data/pinned/thunder/leaderboards.md",
    "hest/README.md": "source_data/pinned/hest/README.md",
    "pathorob/README.md": "source_data/pinned/pathorob/README.md",
}


def source_spec(row: dict[str, str], sources: Path) -> tuple[str, Path, str]:
    lineage = row["lineage"]
    for prefix, (kind, relative, revision_key) in SOURCE_SPECS.items():
        if lineage.startswith(prefix):
            if kind in {"repository", "model_card"}:
                pinned = ROOT / PINNED_SOURCE_DESTINATIONS[relative]
                path = pinned if pinned.is_file() else sources / relative
            else:
                path = ROOT / relative
            if kind in {"repository", "model_card"}:
                revision = lineage.split(":", 1)[0].split("@", 1)[1]
            else:
                revision = revision_key
            return kind, path, revision
    raise ValueError(f"unknown score lineage: {lineage}")


def expected_rows(sources: Path, tasks: list[dict[str, str]], provenance: dict) -> list[dict[str, str]]:
    commits = {key: value["commit"] for key, value in provenance["repositories"].items()}
    exaone, _ = registry.parse_exaone_pathobench_scores(
        ROOT / "source_data/exaone_path_2_5_pathobench_2512.14019v1.csv", tasks
    )
    threads, _ = registry.parse_threads_pathobench_scores(
        ROOT / "source_data/threads_pathobench_2501.16652v1.csv", tasks
    )
    hest, _ = registry.parse_hest_scores(sources / "hest", commits["hest"])
    thunder, _ = registry.parse_thunder_scores(sources / "thunder", commits["thunder"])
    pathorob, _ = registry.parse_pathorob_scores(sources / "pathorob", commits["pathorob"])
    nature, _ = registry.parse_pathorob_nature_scores(
        ROOT / "source_data/pathorob_nature2026_and_repo_examples.csv", tasks
    )
    eva_repo = parse_repository_scores(sources / "eva", commits["eva"])
    eva_card = parse_midnight_scores(sources / "eva_midnight", commits["eva_midnight"])
    eva, _ = merge_eva_scores(eva_repo, eva_card)
    eva_rows = [row.registry_row("2026-08-05") for row in eva]
    h0mini_uni2h, _ = registry.build_h0mini_uni2h_scores(
        ROOT / "source_data/h0mini_uni2h_official_scores_2025.csv"
    )
    group_c, _ = registry.build_group_c_scores(
        ROOT / "source_data/virchow2g_gigapath_titan_official_scores_2024_2025.csv"
    )
    wave_d_hoptimus, _ = registry.build_wave_d_hoptimus_scores(
        ROOT / "source_data/wave_d_hoptimus1_official_report_2025.csv"
    )
    wave_d_virchow2, _ = registry.build_wave_d_virchow2_scores(
        ROOT / "source_data/wave_d_virchow2_paper_2408.00738.csv"
    )
    wave_d_virchow, _ = registry.build_wave_d_virchow_scores(
        ROOT / "source_data/wave_d_virchow_paper_2309.07778.csv"
    )
    wave_d_uni, _ = registry.build_wave_d_uni_scores(
        ROOT / "source_data/wave_d_uni_paper_2308.15474.csv"
    )
    group_b, _ = registry.build_group_b_scores([
        ROOT / "source_data/genbio_pathfm_official_2026.csv",
        ROOT / "source_data/midnight_miccai2025_official_scores.csv",
        ROOT / "source_data/openmidnight_technical_report_2025.csv",
    ])
    wave_e, _ = registry.build_wave_e_scores([
        ROOT / "source_data/conch_official_scores_2024.csv",
        ROOT / "source_data/conch15_titan_official_scores_2025.csv",
        ROOT / "source_data/phikon_family_official_scores_2023_2024.csv",
        ROOT / "source_data/ctranspath_official_evidence_2022_2024.csv",
    ])
    wave_f, _ = registry.build_wave_f_scores([
        ROOT / "source_data/hibou_official_scores_2024.csv",
        ROOT / "source_data/musk_official_scores_2025.csv",
        ROOT / "source_data/gpfm_official_scores_2025.csv",
    ])
    return [*exaone, *threads, *eva_rows, *hest, *thunder, *pathorob, *nature, *h0mini_uni2h, *group_c, *wave_d_hoptimus, *wave_d_virchow2, *wave_d_virchow, *wave_d_uni, *group_b, *wave_e, *wave_f]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("/tmp/pathopress_sources"))
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "data/score_review_ledger.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "data/score_review_summary.json")
    parser.add_argument(
        "--materialize-pinned-sources",
        action="store_true",
        help="Copy only the five score-bearing upstream files into source_data/pinned.",
    )
    args = parser.parse_args()

    scores = read_csv(args.data / "scores.csv")
    tasks = read_csv(args.data / "tasks.csv")
    dedup = read_csv(args.data / "deduplication.csv")
    provenance = json.loads((args.data / "provenance.json").read_text(encoding="utf-8"))
    regenerated = expected_rows(args.sources, tasks, provenance)
    if args.materialize_pinned_sources:
        for upstream_rel, destination_rel in PINNED_SOURCE_DESTINATIONS.items():
            upstream = args.sources / upstream_rel
            destination = ROOT / destination_rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(upstream.read_bytes())
    expected = {row_key(row): {key: str(value) for key, value in row.items()} for row in regenerated}
    actual = {row_key(row): row for row in scores}
    if set(expected) != set(actual):
        raise ValueError("reparsed sources do not exactly cover score cells")
    # The review_status is workflow metadata, not source evidence.
    compared_fields = [field for field in scores[0] if field != "review_status"]
    for key in sorted(actual):
        mismatches = [field for field in compared_fields if actual[key][field] != expected[key][field]]
        if mismatches:
            raise ValueError(f"source reparse mismatch for {key}: {mismatches}")

    memberships = duplicate_memberships(dedup)
    rows: list[dict[str, str]] = []
    for score in scores:
        key = row_key(score)
        groups = memberships.get(score["evaluation_id"], ())
        kind, evidence, revision = source_spec(score, args.sources)
        if not evidence.is_file():
            raise FileNotFoundError(evidence)
        audit = score["audit_status"]
        if audit == "reported_external":
            outcome, eligible = "reported_external", "false"
            canonical = "retain_reported_external_only"
            reason = "PathoROB table transcription and locator validated; upstream marks the value as external and not author-validated, so no status promotion."
        elif audit == "parsed_primary_source_analysis_ineligible":
            outcome = "source_locator_crosschecked" if groups else "source_locator_validated"
            eligible = "true"
            if score["metric"] == "average_performance_drop_percent":
                canonical = "retain_analysis_ineligible_apd"
                reason = "Pinned source value and protocol validated; APD remains outside the bounded normalized matrix by contract."
            else:
                canonical = "retain_analysis_ineligible_public_metric_unspecified"
                reason = "Pinned public leaf value and protocol validated; the source does not name an endpoint-specific metric, so no bounded normalization is inferred."
        else:
            outcome = "source_locator_crosschecked" if groups else "source_locator_validated"
            eligible = "true"
            canonical = "retain_protocol_specific_cell"
            reason = "Pinned source value, locator, alias, metric transform, protocol, and revision exactly reproduce the registry row."
        checks = [
            "metric_direction_scale", "model_alias", "protocol_setting", "source_locator",
            "source_value", "split_version",
        ]
        if groups:
            checks.append("dedup_group_contract")
        review_id = "review." + hashlib.sha256((key[0] + "\0" + key[1]).encode()).hexdigest()[:20]
        rows.append({
            "review_id": review_id,
            "model_id": score["model_id"],
            "evaluation_id": score["evaluation_id"],
            "suite_id": score["suite_id"],
            "reviewer_type": REVIEWER_TYPE,
            "reviewed_at": REVIEWED_AT,
            "evidence_kind": kind,
            "evidence_path": evidence.relative_to(args.sources).as_posix() if evidence.is_relative_to(args.sources) else evidence.relative_to(ROOT).as_posix(),
            "evidence_revision": revision,
            "evidence_sha256": sha256_path(evidence),
            "source_locator": score["source_locator"],
            "source_locator_reachable": "true",
            "checks_passed": ";".join(sorted(checks)),
            "duplicate_group_ids": ";".join(groups),
            "canonical_setting_decision": canonical,
            "audit_status_preserved": audit,
            "prior_review_status": score["review_status"],
            "review_outcome": outcome,
            "promotion_eligible": eligible,
            "decision_reason": reason,
        })
    rows.sort(key=lambda row: (row["model_id"], row["evaluation_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = validate_ledger(scores, tasks, dedup, rows)
    summary["source_files"] = {
        row["evidence_path"]: row["evidence_sha256"] for row in rows
    }
    write_summary(args.summary, summary, args.output.relative_to(ROOT))


if __name__ == "__main__":
    main()
