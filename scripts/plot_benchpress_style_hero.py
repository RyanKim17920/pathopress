#!/usr/bin/env python3
"""Render the faithful upstream-shaped PathoPress hero and ranking panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.completion import complete  # noqa: E402
from pathopress.prediction import load_prediction_dataset  # noqa: E402
from pathopress.probes import evaluate_column_median_baseline  # noqa: E402


MAGENTA, BLUE, GRAY, CHARCOAL = "#d33682", "#268bd2", "#93a1a1", "#073642"
EXAMPLE_COLORS = (MAGENTA, BLUE, "#6c71c4", "#2aa198")


def _cell_job(job: tuple[np.ndarray, int, int, tuple[int, ...]]) -> float:
    matrix, row, column, probes = job
    actual = float(matrix[row, column])
    if column in probes:
        return actual
    probe_mask = np.zeros(matrix.shape[1], dtype=bool)
    probe_mask[list(probes)] = True
    training = matrix.copy()
    training[row, ~probe_mask] = np.nan
    return float(complete(training, rank=1, regularization=.1, allow_empty_rows=True)[row, column])


def _display_evaluation(identifier: str, width: int = 28) -> str:
    """Return a compact suite-aware label without collapsing protocol identities."""

    parts = identifier.split(".")
    suite = parts[0].upper()
    if identifier.startswith("thunder.") and len(parts) >= 3:
        label = f"THU {parts[1].replace('_', ' ').upper()}"
    elif identifier.startswith("hest.") and len(parts) >= 3:
        label = f"HEST {parts[1].upper()}"
    elif identifier.startswith("eva.leaderboard.") and len(parts) >= 4:
        dataset = parts[2].replace("patch_camelyon", "PCam")
        dataset = dataset.replace("camelyon16_small", "CAM16-S")
        split = {"validation": "val"}.get(parts[-1], parts[-1])
        label = f"EVA {dataset.replace('_', ' ')} {split}"
    elif identifier.startswith("pathobench.") and len(parts) >= 4:
        source = {"threads2025": "THR", "exaone2025": "EXA"}.get(
            parts[1], parts[1].upper()
        )
        dataset = parts[2].replace("cptac_", "").replace("_", " ").upper()
        task = parts[-1].replace("slidelevel-", "").replace("-mutation", "")
        task = task.replace("-", " ").upper()
        label = f"{source} {dataset} {task}"
    else:
        label = f"{suite} {identifier.replace('.', ' ').replace('_', ' ')}"
    return label if len(label) <= width else label[: width - 1].rstrip() + "…"


def _trajectory_labels(rows: list[dict[str, object]], width: int = 28) -> list[str]:
    labels = [_display_evaluation(str(row["added_evaluation_id"]), width) for row in rows]
    if len(set(labels)) != len(labels):
        raise ValueError(f"suite-aware hero labels collide: {labels}")
    return labels


def _random_band(rows: list[dict[str, object]], field: str):
    by_k = {k: [float(row["metrics"][field]) for row in rows if int(row["k"]) == k] for k in range(1, 11)}
    return (
        np.arange(1, 11),
        np.asarray([np.median(by_k[k]) for k in range(1, 11)]),
        np.asarray([np.quantile(by_k[k], .25) for k in range(1, 11)]),
        np.asarray([np.quantile(by_k[k], .75) for k in range(1, 11)]),
    )


def _select_examples(matrix: np.ndarray, models: list[str], evaluations: list[str], suites: dict[str, str]):
    medians = np.nanmedian(matrix, axis=0)
    row_support = np.isfinite(matrix).sum(axis=1)
    preferred = ("eva", "hest", "pathobench", "thunder")
    output = []
    used_models: set[int] = set()
    for suite in preferred:
        candidates = []
        for i, j in np.argwhere(np.isfinite(matrix)):
            i, j = int(i), int(j)
            if suites[evaluations[j]] != suite or row_support[i] < 10 or i in used_models:
                continue
            candidates.append((abs(float(matrix[i, j] - medians[j])), models[i], evaluations[j], i, j))
        if not candidates:
            continue
        _, _, _, i, j = max(candidates)
        used_models.add(i)
        output.append((i, j))
    if len(output) != 4:
        raise ValueError("could not select four distinct supported pathology examples")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "data/scores.csv")
    parser.add_argument("--compression", type=Path, default=ROOT / "experiments/probe_compression_rank1.json")
    parser.add_argument("--exhaustive", type=Path, default=ROOT / "experiments/probe_exhaustive_rank1.json")
    parser.add_argument(
        "--omit-stale-exhaustive",
        action="store_true",
        help=(
            "Render current greedy/random panels without exact-search stars. "
            "Use only when no scalar-certified exhaustive result exists for the current scores."
        ),
    )
    parser.add_argument("--hero-output", type=Path, default=ROOT / "figures/pathopress_benchpress_hero_rank1")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "experiments/benchpress_style_hero_summary.json")
    parser.add_argument(
        "--workers",
        type=int,
        choices=range(1, 5),
        default=max(1, min(4, os.cpu_count() or 1)),
        metavar="{1,2,3,4}",
        help="process workers (hard-capped at four; each worker uses one BLAS thread)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_prediction_dataset(args.scores)
    matrix, models, evaluations = dataset.matrix, dataset.models, dataset.evaluations
    suites = {score.evaluation_id: score.suite_id for score in dataset.scores}
    payload = json.loads(args.compression.read_text(encoding="utf-8"))
    config = payload["configuration"]
    scores_hash = hashlib.sha256(args.scores.read_bytes()).hexdigest()
    exact_cheap = None
    exact_pruned = None
    exhaustive_hash = None
    if not args.omit_stale_exhaustive:
        exhaustive = json.loads(args.exhaustive.read_text(encoding="utf-8"))
        if (
            exhaustive.get("status") != "executed_complete_scalar_certified"
            or exhaustive.get("scores_sha256") != scores_hash
        ):
            raise ValueError("hero requires scalar-certified exact searches on current scores")
        exact_cheap = exhaustive["spaces"]["pre_error_proxy_25_choose_5"]["best"]
        exact_pruned = exhaustive["spaces"]["error_informed_pruned_30_choose_5"]["best"]
        exhaustive_hash = hashlib.sha256(args.exhaustive.read_bytes()).hexdigest()
    if float(config.get("ranking_margin", -1)) != 5:
        raise ValueError("hero requires the regenerated margin-5 compression artifact")
    if config.get("scores_sha256") != scores_hash:
        raise ValueError("hero requires a compression artifact generated from current scores")
    expected_candidate_ids = {
        "any_candidate": evaluations,
        "pre_error_low_friction_allowlist": payload["allowlist"]["evaluation_ids"],
    }
    for mode, expected_ids in expected_candidate_ids.items():
        rank = payload["ranking_aware"].get(mode, {})
        if payload["curves"][mode]["candidate_ids"] != expected_ids:
            raise ValueError(f"hero candidate identities do not match current inputs for {mode}")
        if len(rank.get("all_known_greedy", [])) != 10 or len(rank.get("all_known_random", [])) != 100:
            raise ValueError(f"hero requires complete k<=10 margin-5 ranking curves for {mode}")

    examples = _select_examples(matrix, models, evaluations, suites)
    random_rows = payload["curves"]["any_candidate"]["all_known_random"]
    prefixes = [
        (int(row["k"]), tuple(int(value) for value in row["probe_indices"]))
        for row in random_rows
        if int(row["k"]) <= 10
    ]
    jobs = [(matrix, i, j, probes) for i, j in examples for _, probes in prefixes]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        values = list(executor.map(_cell_job, jobs, chunksize=1))
    cursor = 0
    example_summaries = []
    for i, j in examples:
        actual = float(matrix[i, j])
        baseline_prediction = float(np.nanmedian(matrix[:, j]))
        by_k = {k: [] for k in range(1, 11)}
        for k, _ in prefixes:
            by_k[k].append(abs(values[cursor] - actual))
            cursor += 1
        example_summaries.append({
            "model_id": models[i], "evaluation_id": evaluations[j], "suite_id": suites[evaluations[j]],
            "actual": actual, "baseline_prediction": baseline_prediction,
            "baseline_absolute_error": abs(baseline_prediction - actual),
            "random": [
                {"k": k, "median": float(np.median(by_k[k])), "q1": float(np.quantile(by_k[k], .25)), "q3": float(np.quantile(by_k[k], .75))}
                for k in range(1, 11)
            ],
        })

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(15.8, 7.2))
    grid = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1.18, 1.18), wspace=.42, hspace=.45)
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(2)]
    ymax = max(6.0, max(item["baseline_absolute_error"] for item in example_summaries) * 1.35)
    for index, (ax, item) in enumerate(zip(axes, example_summaries)):
        x = np.arange(0, 11)
        median = np.asarray([item["baseline_absolute_error"], *[row["median"] for row in item["random"]]])
        q1 = np.asarray([item["baseline_absolute_error"], *[row["q1"] for row in item["random"]]])
        q3 = np.asarray([item["baseline_absolute_error"], *[row["q3"] for row in item["random"]]])
        color = EXAMPLE_COLORS[index]
        ax.fill_between(x, q1, q3, color=color, alpha=.15, lw=0)
        ax.plot(x, median, "o-", color=color, lw=2, ms=4)
        ax.plot([0], [item["baseline_absolute_error"]], marker="D", mfc="white", mec=CHARCOAL, ms=6, zorder=5)
        ax.annotate("Evaluation median", (0, item["baseline_absolute_error"]), xytext=(7, 5), textcoords="offset points", fontsize=8)
        ax.axhline(5, color=CHARCOAL, ls="--", lw=1.1)
        ax.axhline(2, color=GRAY, ls=":", lw=1.2)
        ax.set(xlim=(-.4, 10.4), ylim=(0, ymax), xticks=range(0, 11))
        ax.grid(axis="y", alpha=.18)
        ax.set_title(
            f"{item['model_id']}\n{_display_evaluation(item['evaluation_id'], 34)}",
            fontsize=10,
            fontweight="bold",
        )
        if index in (0, 2): ax.set_ylabel("Absolute error")
        if index in (2, 3): ax.set_xlabel("# Known pathology evaluations")

    ax = fig.add_subplot(grid[:, 2:])
    baseline = evaluate_column_median_baseline(matrix).parity.median_absolute_error
    random_k, random_med, random_q1, random_q3 = _random_band(random_rows, "medae")
    ax.fill_between(np.r_[0, random_k], np.r_[baseline, random_q1], np.r_[baseline, random_q3], color=GRAY, alpha=.15, lw=0)
    ax.plot(np.r_[0, random_k], np.r_[baseline, random_med], "o--", color=GRAY, lw=2.4, label="Random evaluation set")
    tracks = (
        ("any_candidate", MAGENTA, "Most predictive evaluations", "o"),
        ("pre_error_low_friction_allowlist", BLUE, "25-task feasibility proxy", "s"),
    )
    for mode, color, label, marker in tracks:
        rows = payload["curves"][mode]["all_known_greedy_medae"]
        display_labels = _trajectory_labels(rows)
        xs = np.r_[0, [int(row["k"]) for row in rows]]
        ys = np.r_[baseline, [float(row["selection_metrics"]["medae"]) for row in rows]]
        ax.plot(xs, ys, marker=marker, color=color, lw=2.5, label=label)
        for position, display_label in enumerate(display_labels, 1):
            above = mode == "pre_error_low_friction_allowlist"
            offset = (3, 8 + 5 * (position % 2)) if above else (-3, -10 - 5 * (position % 2))
            ax.annotate(
                display_label,
                (position, ys[position]),
                xytext=offset,
                textcoords="offset points",
                rotation=29,
                ha="left" if above else "right",
                va="bottom" if above else "top",
                fontsize=7.0,
                color=color,
            )
    if exact_cheap is not None and exact_pruned is not None:
        ax.plot(
            [5], [float(exact_cheap["medae"])], marker="*", ms=14,
            color=BLUE, mec=CHARCOAL, mew=.6, linestyle="none",
            label=f"Exact 25-task: {float(exact_cheap['medae']):.3f} (53,130 sets)", zorder=8,
        )
        ax.plot(
            [5], [float(exact_pruned["medae"])], marker="*", ms=14,
            color="#2aa198", mec=CHARCOAL, mew=.6, linestyle="none",
            label=f"Exact pruned-30: {float(exact_pruned['medae']):.3f} (142,506 sets)", zorder=8,
        )
    ax.plot([0], [baseline], marker="D", mfc="white", mec=CHARCOAL, ms=7, zorder=6)
    ax.annotate("Evaluation-column median", (0, baseline), xytext=(8, 2), textcoords="offset points", fontsize=9)
    ax.set(
        xlim=(-.4, 10.45),
        ylim=(1.05, None),
        xticks=range(0, 11),
        xlabel="# Top pathology evaluations",
        ylabel="Median absolute error (normalized points)",
    )
    ax.set_title("Overall score prediction", fontweight="bold")
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle("PathoPress reconstruction of the BenchPress hero", fontsize=16, fontweight="bold", y=.985)
    exhaustive_note = (
        "Stars: complete scalar-certified MedAE searches over C(25,5)=53,130 "
        "and error-informed C(30,5)=142,506"
        if exact_cheap is not None
        else "Current-matrix exhaustive MedAE searches are not shown or claimed"
    )
    fig.text(.5, .01, f"Exact masking · Pathology-adapted selected rank 1 · {exhaustive_note} · 25-task track is a feasibility proxy, not measured cost", ha="center", fontsize=8.2)
    # Reserve a real title band: the two-line example labels otherwise collide
    # with the figure title on compact renderers.
    fig.subplots_adjust(bottom=.09, top=.84)

    args.hero_output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.hero_output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "schema_version": 1,
        "source_shape": list(matrix.shape),
        "inputs": {"scores_sha256": scores_hash, "compression_sha256": hashlib.sha256(args.compression.read_bytes()).hexdigest(), "exhaustive_sha256": exhaustive_hash},
        "example_selection": "largest evaluation-column-median absolute error within EVA, HEST, PathoBench, and THUNDER; distinct models with at least ten retained scores",
        "examples": example_summaries,
        "overall_tracks": [
            "random_any_candidate",
            "greedy_any_candidate",
            "greedy_25_task_feasibility_proxy",
            *(
                ["exact_25_task_k5_medae", "exact_error_informed_30_task_k5_medae"]
                if exact_cheap is not None
                else []
            ),
        ],
        "ranking_tracks": ["random_any_candidate_margin5", "greedy_any_candidate_margin5", "greedy_25_task_feasibility_proxy_margin5"],
        "exact_results": (
            {"pre_error_proxy_25_choose_5": exact_cheap, "error_informed_pruned_30_choose_5": exact_pruned}
            if exact_cheap is not None
            else {}
        ),
        "contract_status": {
            "masking_and_k_budget": "exact",
            "rank_and_domain": "pathology_adapted",
            "exhaustive_25C5_30C5": (
                "executed_complete_scalar_certified"
                if exact_cheap is not None
                else "not_run_for_current_scores"
            ),
        },
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.hero_output}.{{png,pdf}} and {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
