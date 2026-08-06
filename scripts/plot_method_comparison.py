#!/usr/bin/env python3
"""Plot BenchPress-style transform-by-method heatmaps from prediction shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRANSFORMS = ("identity", "log", "logit", "asinh", "sqrt", "probit", "quantile")
TRANSFORM_LABELS = ("Identity", "Log", "Logit", "Arcsinh", "Square root", "Probit", "Quantile")
METHODS = (
    "Benchmark Mean", "Model Mean", "Bench-KNN", "Model-KNN", "BenchReg", "ModelReg",
    "Soft-Impute", "Bias ALS", "NMF", "PMF", "Nuclear Norm", "MLP",
)


def _grid(results: dict, field: str) -> np.ndarray:
    output = np.full((len(METHODS), len(TRANSFORMS)), np.nan)
    for column, transform in enumerate(TRANSFORMS):
        for row, method in enumerate(METHODS):
            value = results.get(transform, {}).get(method, {}).get(field)
            if value is not None:
                output[row, column] = float(value)
    return output


def _display_grid(results: dict, field: str) -> np.ndarray:
    """Return the units displayed in cells and on the matching colorbar."""
    values = _grid(results, field)
    return values * 100.0 if field == "coverage" else values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "experiments" / "method_comparison" / "results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "method_comparison_grid")
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    manifest_path = args.results.with_name("manifest.json")
    completion_note = ""
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts = manifest.get("counts", {})
        total = sum(int(counts.get(key, 0)) for key in ("completed", "unsupported", "missing"))
        completion_note = f" — {counts.get('completed', 0)}/{total} shards complete"
    definitions = (
        ("medape_median", "Median fold MedAPE (%)", "viridis_r", lambda value: f"{value:.1f}"),
        ("medae_median", "Median fold MedAE", "viridis_r", lambda value: f"{value:.1f}"),
        ("coverage", "Held-out coverage (%)", "viridis", lambda value: f"{value:.0f}"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
    for panel, (field, title, cmap, formatter) in zip(axes, definitions):
        values = _display_grid(results, field)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError(f"results contain no finite {field} values")
        image = panel.imshow(values, aspect="auto", cmap=cmap, vmin=float(finite.min()), vmax=float(finite.max()))
        panel.set_title(title, fontweight="bold")
        panel.set_xticks(range(len(TRANSFORMS)), TRANSFORM_LABELS, rotation=45, ha="right")
        panel.set_yticks(range(len(METHODS)), METHODS if panel is axes[0] else [""] * len(METHODS))
        midpoint = float(np.median(finite))
        for row in range(len(METHODS)):
            for column in range(len(TRANSFORMS)):
                value = values[row, column]
                label = formatter(value) if np.isfinite(value) else "—"
                color = "white" if np.isfinite(value) and value > midpoint else "black"
                panel.text(column, row, label, ha="center", va="center", fontsize=7, color=color)
        fig.colorbar(image, ax=panel, shrink=0.75, pad=0.02)
    fig.suptitle(
        "PathoPress transform × completion-method comparison" + completion_note,
        fontsize=15,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
