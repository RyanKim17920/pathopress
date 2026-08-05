#!/usr/bin/env python3
"""Extract the public Patho-Bench-compatible frozen rows from Threads.

Input is the official arXiv HTML for 2501.16652v1.  Extended Data Tables
11--26 and 28--37 contain 42 public Patho-Bench tasks.  Tables 9, 10, and
27 are internal cohorts and are deliberately quarantined; supervised and
fine-tuned rows are also excluded.
"""

from __future__ import annotations

import argparse
import csv
import re
from html.parser import HTMLParser
from pathlib import Path


SOURCE_ARCHIVE_SHA256 = "3d8b3f6779b9b0eae21be12e8917bd6f0bab26e3c7943470e378383d20a1de4f"
SOURCE_HTML_SHA256 = "a6c7af63c1f527eba692f83b362651e0e1d96d07e303520f90cd08f34b00c92f"
MODELS = (
    "Virchow Mean Pooling",
    "GigaPath Mean Pooling",
    "Chief Mean Pooling",
    "CONCHv1.5 Mean Pooling",
    "PRISM",
    "GigaPath",
    "CHIEF",
    "Threads",
)
TABLE_TASKS = {
    11: ("pathobench.bcnb.er", "pathobench.bcnb.pr", "pathobench.bcnb.her2"),
    12: ("pathobench.mut-het-rcc.bap1-mutation", "pathobench.mut-het-rcc.pbrm1-mutation", "pathobench.mut-het-rcc.setd2-mutation"),
    13: ("pathobench.imp.grade",),
    14: ("pathobench.panda.isup-grade",),
    15: ("pathobench.cptac_brca.pik3ca-mutation", "pathobench.cptac_brca.tp53-mutation"),
    16: ("pathobench.cptac_ccrcc.bap1-mutation", "pathobench.cptac_ccrcc.pbrm1-mutation"),
    17: ("pathobench.cptac_coad.kras-mutation", "pathobench.cptac_coad.tp53-mutation"),
    18: ("pathobench.cptac_gbm.egfr-mutation", "pathobench.cptac_gbm.tp53-mutation"),
    19: ("pathobench.cptac_hnsc.casp8-mutation",),
    20: ("pathobench.cptac_lscc.keap1-mutation", "pathobench.cptac_lscc.arid1a-mutation"),
    21: ("pathobench.cptac_luad.egfr-mutation", "pathobench.cptac_luad.stk11-mutation", "pathobench.cptac_luad.tp53-mutation"),
    22: ("pathobench.cptac_pda.smad4-mutation",),
    23: ("pathobench.bracs.slidelevel-fine", "pathobench.bracs.slidelevel-coarse"),
    24: ("pathobench.ebrains.diagnosis", "pathobench.ebrains.diagnosis-group"),
    25: ("pathobench.ovarian.response",),
    26: ("pathobench.nadt.response",),
    28: ("pathobench.natbrca.lymphovascular-invasion",),
    29: ("pathobench.sr386_.braf-mutant-binary", "pathobench.sr386_.ras-mutant-binary", "pathobench.sr386_.mmr-loss-binary", "pathobench.sr386_.died-within-5-years"),
    30: ("pathobench.mbc_.recist",),
    31: ("pathobench.cptac_pda.os",),
    32: ("pathobench.cptac_luad.os",),
    33: ("pathobench.cptac_ccrcc.os",),
    34: ("pathobench.cptac_hnsc.os",),
    35: ("pathobench.sr386_.os",),
    36: ("pathobench.mbc_.os",),
    37: ("pathobench.boehmk_.pfs",),
}
FIELDS = (
    "source_table",
    "base_evaluation_id",
    "evaluation_id",
    "metric",
    "model_alias",
    "value",
    "uncertainty",
    "reference_url",
    "source_html_sha256",
    "source_archive_sha256",
)


class ArxivTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.figures: list[dict[str, object]] = []
        self.figure: dict[str, object] | None = None
        self.in_caption = False
        self.in_row = False
        self.in_cell = False
        self.cell_parts: list[str] = []
        self.row: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "figure" and "ltx_table" in (attributes.get("class") or "").split():
            self.figure = {"caption": [], "rows": []}
        elif self.figure is not None and tag == "figcaption":
            self.in_caption = True
        elif self.figure is not None and tag == "tr":
            self.in_row = True
            self.row = []
        elif self.in_row and tag in {"td", "th"}:
            self.in_cell = True
            self.cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        if self.figure is not None and tag == "figcaption":
            self.in_caption = False
        elif self.in_cell and tag in {"td", "th"}:
            self.row.append(" ".join("".join(self.cell_parts).split()))
            self.in_cell = False
        elif self.figure is not None and tag == "tr":
            if self.row:
                self.figure["rows"].append(self.row)  # type: ignore[union-attr]
            self.in_row = False
        elif self.figure is not None and tag == "figure":
            self.figure["caption"] = " ".join(self.figure["caption"]).strip()  # type: ignore[arg-type]
            self.figures.append(self.figure)
            self.figure = None

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_parts.append(data)
        if self.figure is not None and self.in_caption:
            self.figure["caption"].append(data)  # type: ignore[union-attr]


def source_metrics(pathobench_hf: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for config_path in pathobench_hf.glob("*/*/config.yaml"):
        dataset, task = config_path.parent.parent.name, config_path.parent.name
        evaluation_id = "pathobench." + dataset + "." + re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")
        match = re.search(r"(?m)^metrics:\s*\n\s*-\s*([^\n#]+)", config_path.read_text(encoding="utf-8"))
        if not match:
            raise ValueError(f"missing metric in {config_path}")
        metrics[evaluation_id] = match.group(1).strip()
    return metrics


def score_value(cell: str) -> tuple[str, str]:
    clean = re.sub(r"[^0-9.±()\-]", "", cell)
    match = re.search(r"(0\.\d+)", clean)
    if not match:
        raise ValueError(f"missing score value: {cell!r}")
    value = match.group(1)
    se = re.search(r"±(0\.\d+)", clean)
    interval = re.search(r"\((0\.\d+)-(0\.\d+)\)", clean)
    if se:
        uncertainty = f"standard_error={se.group(1)}"
    elif interval:
        uncertainty = f"95%_ci=[{interval.group(1)},{interval.group(2)}]"
    else:
        uncertainty = "not_reported"
    return value, uncertainty


def extract(html_path: Path, pathobench_hf: Path) -> list[dict[str, str]]:
    parser = ArxivTableParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    figures: dict[int, dict[str, object]] = {}
    for figure in parser.figures:
        match = re.search(r"Extended Data Table\s+(\d+)", str(figure["caption"]))
        if match:
            figures[int(match.group(1))] = figure
    metrics = source_metrics(pathobench_hf)
    output: list[dict[str, str]] = []
    for table_number, evaluations in TABLE_TASKS.items():
        if table_number not in figures:
            raise ValueError(f"missing Extended Data Table {table_number}")
        rows = figures[table_number]["rows"]  # type: ignore[assignment]
        start = next(i for i, row in enumerate(rows) if any(x in " ".join(row) for x in ("Linear Probe", "CoxNet")))
        model_rows = rows[start : start + 8]
        if len(model_rows) != 8:
            raise ValueError(f"Table {table_number} has {len(model_rows)} frozen rows")
        for index, (alias, row) in enumerate(zip(MODELS, model_rows)):
            score_cells = row[-len(evaluations) :]
            if len(score_cells) != len(evaluations):
                raise ValueError(f"Table {table_number} row {index} score width mismatch")
            for base_evaluation_id, cell in zip(evaluations, score_cells):
                if base_evaluation_id not in metrics:
                    raise ValueError(f"unknown HF task crosswalk: {base_evaluation_id}")
                value, uncertainty = score_value(cell)
                suffix = base_evaluation_id.removeprefix("pathobench.")
                output.append(
                    {
                        "source_table": str(table_number),
                        "base_evaluation_id": base_evaluation_id,
                        "evaluation_id": "pathobench.threads2025." + suffix,
                        "metric": metrics[base_evaluation_id],
                        "model_alias": alias,
                        "value": value,
                        "uncertainty": uncertainty,
                        "reference_url": f"https://arxiv.org/html/2501.16652v1#S0.T{table_number}",
                        "source_html_sha256": SOURCE_HTML_SHA256,
                        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                    }
                )
    if len(output) != 336 or len({row["evaluation_id"] for row in output}) != 42:
        raise ValueError(f"Threads public-cell audit failed: rows={len(output)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--pathobench-hf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = extract(args.html, args.pathobench_hf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} scores to {args.output}")


if __name__ == "__main__":
    main()
