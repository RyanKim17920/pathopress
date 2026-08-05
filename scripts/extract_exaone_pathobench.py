#!/usr/bin/env python3
"""Extract EXAONE Path 2.5's official Patho-Bench Table 4 from TeX.

The input is ``tabs/pathobench_result.tex`` from the official arXiv v1
source archive for 2512.14019.  Task names are joined against a pinned
MahmoodLab/Patho-Bench Hugging Face checkout; no fuzzy matching is used.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ARXIV_URL = "https://arxiv.org/pdf/2512.14019v1"
SOURCE_ARCHIVE_SHA256 = "0c479164dfab7ac48a1e1876649ef73efe9f457e064c3ab00ee960856d35a268"
MODELS = (
    "CHIEF",
    "GigaPath",
    "PRISM",
    "TITAN",
    "H-optimus-0",
    "UNI2-h",
    "EXAONE Path 2.5",
)
FIELDS = (
    "source_task",
    "base_evaluation_id",
    "evaluation_id",
    "metric",
    "model_alias",
    "value",
    "reference_url",
    "source_archive_sha256",
)


def slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower())
    return value.strip("-")


def yaml_scalar(text: str, name: str) -> str:
    match = re.search(rf"(?m)^{re.escape(name)}:\s*([^\n#]+)", text)
    if not match:
        raise ValueError(f"missing {name}")
    return match.group(1).strip().strip("'\"")


def official_tasks(pathobench_hf: Path) -> dict[str, tuple[str, str, str]]:
    tasks: dict[str, tuple[str, str, str]] = {}
    for config_path in sorted(pathobench_hf.glob("*/*/config.yaml")):
        dataset = config_path.parent.parent.name
        task = config_path.parent.name
        source_task = f"{dataset}_{task}"
        config = config_path.read_text(encoding="utf-8")
        metric_match = re.search(r"(?m)^metrics:\s*\n\s*-\s*([^\n#]+)", config)
        if not metric_match:
            raise ValueError(f"missing metric in {config_path}")
        metric = metric_match.group(1).strip()
        base_evaluation_id = f"pathobench.{dataset}.{slug(task)}"
        task_type = yaml_scalar(config, "task_type")
        report_metric = "cindex" if task_type == "survival" else "macro-ovr-auc"
        tasks[source_task] = (
            base_evaluation_id,
            f"pathobench.exaone2025.{dataset}.{slug(task)}",
            report_metric,
        )
    if len(tasks) != 95:
        raise ValueError(f"expected 95 official Patho-Bench tasks, found {len(tasks)}")
    return tasks


def extract(tex_path: Path, pathobench_hf: Path) -> list[dict[str, str]]:
    text = tex_path.read_text(encoding="utf-8")
    tasks = official_tasks(pathobench_hf)
    rows: list[dict[str, str]] = []
    table_tasks: list[str] = []
    pattern = re.compile(
        r"^\s*([^%&]+?)\s*&\s*"
        r"(0\.\d+)\s*&\s*(0\.\d+)\s*&\s*(0\.\d+)\s*&\s*"
        r"(0\.\d+)\s*&\s*(0\.\d+)\s*&\s*(0\.\d+)\s*&\s*"
        r"(0\.\d+)\s*\\\\\s*$"
    )
    for raw_line in text.splitlines():
        match = pattern.match(raw_line)
        if not match:
            continue
        source_task = match.group(1).replace(r"\_", "_").strip()
        if "AVERAGE" in source_task:
            continue
        if source_task not in tasks:
            raise ValueError(f"Table 4 task is not in the official HF inventory: {source_task}")
        table_tasks.append(source_task)
        base_evaluation_id, evaluation_id, metric = tasks[source_task]
        for model, value in zip(MODELS, match.groups()[1:]):
            rows.append(
                {
                    "source_task": source_task,
                    "base_evaluation_id": base_evaluation_id,
                    "evaluation_id": evaluation_id,
                    "metric": metric,
                    "model_alias": model,
                    "value": value,
                    "reference_url": ARXIV_URL,
                    "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                }
            )
    if len(table_tasks) != 80 or len(set(table_tasks)) != 80:
        raise ValueError(f"expected 80 unique Table 4 tasks, found {len(table_tasks)}")
    if len(rows) != 560:
        raise ValueError(f"expected 560 score cells, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--pathobench-hf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = extract(args.tex, args.pathobench_hf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} scores to {args.output}")


if __name__ == "__main__":
    main()
