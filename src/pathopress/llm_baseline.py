"""Provider-neutral BenchPress-style LLM score-completion scaffolding.

This module builds prompts and validates request/response caches.  It performs
no network calls.  The deterministic mock backend exists only to exercise the
artifact contract and is never headline-eligible.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = 1
CONDITIONS = (
    "matrix_named",
    "matrix_blind",
    "five_shot_named",
    "five_shot_blind",
)
REAL_BACKEND_KINDS = {"openai_compatible", "anthropic_compatible", "local_model"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def make_config(
    *,
    scores_sha256: str,
    folds_sha256: str,
    models: Sequence[str],
    evaluations: Sequence[str],
    fold_ids: Sequence[int],
    cell_limit: int | None,
    n_shots: int = 5,
    min_shared: int = 5,
    max_target_known: int = 12,
    max_peer_shared: int = 4,
) -> dict[str, Any]:
    config = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "pathopress_llm_baseline_s10_f3_bs42",
        "upstream_semantics": {
            "matrix": "sparse score matrix; named/informed versus anonymized/blind",
            "five_shot": "five highest-Pearson peers with target score observed and at least five shared visible scores",
        },
        "pathology_adaptation": {
            "score_scale": "normalized 0-100 for every retained evaluation",
            "selected_completion_rank": 1,
            "reasoning_flag": "omitted because pathology encoders do not have the upstream LLM reasoning-mode attribute",
        },
        "scores_sha256": scores_sha256,
        "folds_sha256": folds_sha256,
        "matrix_shape": [len(models), len(evaluations)],
        "models": list(models),
        "evaluations": list(evaluations),
        "fold_ids": [int(value) for value in fold_ids],
        "cell_limit": cell_limit,
        "n_shots": int(n_shots),
        "min_shared": int(min_shared),
        "max_target_known": int(max_target_known),
        "max_peer_shared": int(max_peer_shared),
        "conditions": list(CONDITIONS),
    }
    config["config_sha256"] = object_sha256(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    supplied = config.get("config_sha256")
    unsigned = {key: value for key, value in config.items() if key != "config_sha256"}
    expected = object_sha256(unsigned)
    if supplied != expected:
        raise ValueError("LLM baseline configuration hash mismatch")
    if tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError("LLM baseline conditions do not match the pinned condition set")
    if config.get("n_shots") != 5 or config.get("min_shared") != 5:
        raise ValueError("five-shot peer semantics drifted from the pinned protocol")


def format_matrix_csv(
    matrix: np.ndarray,
    models: Sequence[str],
    evaluations: Sequence[str],
    *,
    blind: bool,
) -> str:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (len(models), len(evaluations)):
        raise ValueError("matrix identifiers do not match matrix shape")
    columns = [f"B{j}" for j in range(len(evaluations))] if blind else list(evaluations)
    lines = ["model," + ",".join(columns)]
    for i, model in enumerate(models):
        label = f"M{i}" if blind else model
        cells = ["?" if not np.isfinite(value) else f"{value:.3f}" for value in values[i]]
        lines.append(label + "," + ",".join(cells))
    return "\n".join(lines)


def _target_records(
    cells: Sequence[tuple[int, int]], models: Sequence[str], evaluations: Sequence[str]
) -> list[dict[str, Any]]:
    return [
        {
            "query_id": f"q{offset}",
            "model_index": int(i),
            "evaluation_index": int(j),
            "model_id": models[i],
            "evaluation_id": evaluations[j],
        }
        for offset, (i, j) in enumerate(cells)
    ]


def build_matrix_messages(
    train: np.ndarray,
    cells: Sequence[tuple[int, int]],
    models: Sequence[str],
    evaluations: Sequence[str],
    task_metadata: dict[str, dict[str, str]],
    *,
    blind: bool,
) -> list[dict[str, str]]:
    matrix_csv = format_matrix_csv(train, models, evaluations, blind=blind)
    if blind:
        definitions = "Model and evaluation identities and metadata are anonymized."
    else:
        target_evaluations = sorted({evaluations[j] for _, j in cells})
        definitions = "Target evaluation definitions:\n" + "\n".join(
            f"- {evaluation}: family={task_metadata[evaluation].get('task_family', 'unknown')}; "
            f"sample_unit={task_metadata[evaluation].get('sample_unit', 'unknown')}; "
            f"metric={task_metadata[evaluation].get('metric', 'unknown')}"
            for evaluation in target_evaluations
        )
    system = (
        "You predict missing normalized pathology foundation-model evaluation scores. "
        "Scores use a common 0-100 orientation where larger is better. The retained "
        "matrix has strong low-dimensional structure; pathology validation selected a "
        "rank-1 completion model. Return JSON only, mapping each query_id to one number.\n\n"
        + definitions
        + "\n\nObserved matrix ('?' means hidden):\n"
        + matrix_csv
    )
    requests = []
    for offset, (i, j) in enumerate(cells):
        model = f"M{i}" if blind else models[i]
        evaluation = f"B{j}" if blind else evaluations[j]
        requests.append(f"q{offset}: predict {model} on {evaluation}")
    user = "Predict these hidden cells. Return only JSON like {\"q0\": 72.5}.\n" + "\n".join(requests)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _shared_columns(train: np.ndarray, target: int, peer: int, excluded: int) -> np.ndarray:
    mask = np.isfinite(train[target]) & np.isfinite(train[peer])
    mask[excluded] = False
    return np.flatnonzero(mask)


def select_peer_examples(
    train: np.ndarray,
    target: int,
    evaluation: int,
    *,
    n_shots: int = 5,
    min_shared: int = 5,
) -> list[dict[str, Any]]:
    """Select nearest peers using visible training cells only."""

    candidates = []
    for peer in range(train.shape[0]):
        if peer == target or not np.isfinite(train[peer, evaluation]):
            continue
        shared = _shared_columns(train, target, peer, evaluation)
        if len(shared) < min_shared:
            continue
        x, y = train[target, shared], train[peer, shared]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        correlation = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(correlation):
            candidates.append(
                {"peer_index": int(peer), "correlation": correlation, "shared_indices": [int(value) for value in shared]}
            )
    return sorted(candidates, key=lambda row: (-row["correlation"], row["peer_index"]))[:n_shots]


def build_five_shot_queries(
    train: np.ndarray,
    cells: Sequence[tuple[int, int]],
    *,
    n_shots: int = 5,
    min_shared: int = 5,
    max_target_known: int = 12,
    max_peer_shared: int = 4,
) -> list[dict[str, Any]]:
    queries = []
    for offset, (i, j) in enumerate(cells):
        target_known = [int(value) for value in np.flatnonzero(np.isfinite(train[i])) if int(value) != j][
            :max_target_known
        ]
        peers = select_peer_examples(train, i, j, n_shots=n_shots, min_shared=min_shared)
        for peer in peers:
            peer["shared_indices"] = peer["shared_indices"][:max_peer_shared]
        queries.append(
            {
                "query_id": f"q{offset}",
                "model_index": int(i),
                "evaluation_index": int(j),
                "target_known_indices": target_known,
                "examples": peers,
            }
        )
    return queries


def _score_list(
    train: np.ndarray,
    row: int,
    indices: Sequence[int],
    evaluations: Sequence[str],
    labels: dict[int, str] | None,
) -> str:
    return ", ".join(
        f"{labels[j] if labels is not None else evaluations[j]}={train[row, j]:.3f}" for j in indices
    ) or "none"


def render_five_shot_messages(
    train: np.ndarray,
    queries: Sequence[dict[str, Any]],
    models: Sequence[str],
    evaluations: Sequence[str],
    task_metadata: dict[str, dict[str, str]],
    *,
    blind: bool,
) -> list[dict[str, str]]:
    lines = [
        "Estimate normalized pathology benchmark scores from five nearest peer-model examples.",
        "Return only JSON mapping each query_id to a 0-100 numeric score; do not explain.",
    ]
    for query in queries:
        i, j = query["model_index"], query["evaluation_index"]
        used = [j, *query["target_known_indices"]]
        for example in query["examples"]:
            used.extend(example["shared_indices"])
        unique = list(dict.fromkeys(used))
        labels = {value: f"Benchmark {offset + 1}" for offset, value in enumerate(unique)} if blind else None
        target_model = f"Target {query['query_id']}" if blind else models[i]
        target_evaluation = labels[j] if labels is not None else evaluations[j]
        if not blind:
            metadata = task_metadata[evaluations[j]]
            target_evaluation += (
                f" [family={metadata.get('task_family', 'unknown')}; "
                f"unit={metadata.get('sample_unit', 'unknown')}; metric={metadata.get('metric', 'unknown')}]"
            )
        lines.extend(
            [
                "",
                f"Query {query['query_id']}",
                f"Target model: {target_model}",
                f"Known target scores: {_score_list(train, i, query['target_known_indices'], evaluations, labels)}",
                f"Predict: {target_evaluation}",
                "Nearest peer examples:",
            ]
        )
        if not query["examples"]:
            lines.append("- No eligible peers.")
        for number, example in enumerate(query["examples"], 1):
            peer = example["peer_index"]
            peer_name = f"Peer {query['query_id']}-{number}" if blind else models[peer]
            lines.append(
                f"- {peer_name}; shared={_score_list(train, peer, example['shared_indices'], evaluations, labels)}; "
                f"{target_evaluation}={train[peer, j]:.3f}"
            )
    return [{"role": "user", "content": "\n".join(lines)}]


def build_request(
    *,
    config: dict[str, Any],
    condition: str,
    fold_id: int,
    seed: int,
    fold: int,
    batch_index: int,
    train: np.ndarray,
    cells: Sequence[tuple[int, int]],
    models: Sequence[str],
    evaluations: Sequence[str],
    task_metadata: dict[str, dict[str, str]],
) -> dict[str, Any]:
    validate_config(config)
    if condition not in CONDITIONS:
        raise ValueError(f"unknown LLM baseline condition: {condition}")
    blind = condition.endswith("blind")
    if condition.startswith("matrix"):
        messages = build_matrix_messages(train, cells, models, evaluations, task_metadata, blind=blind)
        query_meta: list[dict[str, Any]] = []
    else:
        query_meta = build_five_shot_queries(
            train,
            cells,
            n_shots=config["n_shots"],
            min_shared=config["min_shared"],
            max_target_known=config["max_target_known"],
            max_peer_shared=config["max_peer_shared"],
        )
        messages = render_five_shot_messages(train, query_meta, models, evaluations, task_metadata, blind=blind)
    identity = {
        "config_sha256": config["config_sha256"],
        "condition": condition,
        "fold_id": int(fold_id),
        "batch_index": int(batch_index),
        "targets": [[int(i), int(j)] for i, j in cells],
    }
    request = {
        "schema_version": SCHEMA_VERSION,
        "request_id": object_sha256(identity)[:24],
        **identity,
        "seed": int(seed),
        "fold": int(fold),
        "targets": _target_records(cells, models, evaluations),
        "query_meta": query_meta,
        "messages": messages,
        "response_contract": {"type": "json_object", "values": "normalized_score_0_100"},
        "execution_status": "prepared_unrun",
    }
    request["request_sha256"] = object_sha256(request)
    return request


def validate_request(request: dict[str, Any], config: dict[str, Any]) -> None:
    validate_config(config)
    supplied = request.get("request_sha256")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if supplied != object_sha256(unsigned):
        raise ValueError("request hash mismatch")
    if request.get("config_sha256") != config["config_sha256"]:
        raise ValueError("request configuration mismatch")
    if request.get("condition") not in CONDITIONS:
        raise ValueError("request has an unknown condition")


def parse_prediction_payload(text: str, query_ids: set[str]) -> dict[str, float]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict) and isinstance(data.get("predictions"), dict):
        data = data["predictions"]
    if not isinstance(data, dict):
        return {}
    parsed = {}
    for key, value in data.items():
        if key not in query_ids:
            continue
        if isinstance(value, dict):
            value = value.get("score", value.get("prediction"))
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            parsed[key] = float(np.clip(numeric, 0.0, 100.0))
    return parsed


def deterministic_mock_response(
    request: dict[str, Any], train: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    """Return a deterministic contract test response, never a real LLM result."""

    validate_request(request, config)
    predictions = {}
    meta_by_id = {row["query_id"]: row for row in request.get("query_meta", [])}
    for target in request["targets"]:
        qid, j = target["query_id"], target["evaluation_index"]
        meta = meta_by_id.get(qid)
        if meta and meta.get("examples"):
            peer_values = [train[example["peer_index"], j] for example in meta["examples"]]
            prediction = float(np.mean(peer_values))
        else:
            prediction = float(np.nanmedian(train[:, j]))
        predictions[qid] = round(float(np.clip(prediction, 0.0, 100.0)), 6)
    content = canonical_json(predictions)
    response = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "config_sha256": config["config_sha256"],
        "backend_kind": "deterministic_mock",
        "provider": "none",
        "model": "deterministic-column-or-peer-mean",
        "status": "complete_mock_only",
        "headline_eligible": False,
        "response_text": content,
        "parsed_predictions": predictions,
        "usage": {"input_tokens": None, "output_tokens": None, "status": "not_applicable"},
    }
    response["response_sha256"] = object_sha256(response)
    return response


def validate_response(
    response: dict[str, Any], request: dict[str, Any], config: dict[str, Any]
) -> None:
    validate_request(request, config)
    supplied = response.get("response_sha256")
    unsigned = {key: value for key, value in response.items() if key != "response_sha256"}
    if supplied != object_sha256(unsigned):
        raise ValueError("response hash mismatch")
    for key in ("request_id", "request_sha256", "config_sha256"):
        expected = request["request_id"] if key == "request_id" else request[key]
        if response.get(key) != expected:
            raise ValueError(f"response {key} mismatch")
    if response.get("backend_kind") == "deterministic_mock" and response.get("headline_eligible") is not False:
        raise ValueError("mock responses must never be headline eligible")


def evaluate_cached_responses(
    requests: Sequence[dict[str, Any]],
    responses: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    by_request = {row["request_id"]: row for row in responses}
    if len(by_request) != len(responses):
        raise ValueError("duplicate cached response request_id")
    raw = []
    backend_kinds = set()
    for request in requests:
        response = by_request.get(request["request_id"])
        if response is None:
            continue
        validate_response(response, request, config)
        backend_kinds.add(response["backend_kind"])
        parsed = parse_prediction_payload(
            response["response_text"], {target["query_id"] for target in request["targets"]}
        )
        for target in request["targets"]:
            qid, i, j = target["query_id"], target["model_index"], target["evaluation_index"]
            if qid not in parsed:
                continue
            raw.append(
                {
                    "request_id": request["request_id"],
                    "condition": request["condition"],
                    "fold_id": request["fold_id"],
                    "seed": request["seed"],
                    "fold": request["fold"],
                    "model_id": target["model_id"],
                    "evaluation_id": target["evaluation_id"],
                    "actual": float(matrix[i, j]),
                    "predicted": float(parsed[qid]),
                    "backend_kind": response["backend_kind"],
                    "headline_eligible": bool(response["headline_eligible"]),
                }
            )
    summaries = []
    for condition in CONDITIONS:
        rows = [row for row in raw if row["condition"] == condition]
        errors = np.asarray([abs(row["predicted"] - row["actual"]) for row in rows], dtype=float)
        ape = np.asarray(
            [100 * abs(row["predicted"] - row["actual"]) / abs(row["actual"]) for row in rows if abs(row["actual"]) > 1e-12],
            dtype=float,
        )
        requested = sum(len(request["targets"]) for request in requests if request["condition"] == condition)
        summaries.append(
            {
                "condition": condition,
                "n": len(rows),
                "n_requested": requested,
                "coverage": len(rows) / requested if requested else 0.0,
                "medae": float(np.median(errors)) if len(errors) else None,
                "medape": float(np.median(ape)) if len(ape) else None,
            }
        )
    headline = bool(backend_kinds) and backend_kinds.issubset(REAL_BACKEND_KINDS) and all(
        row["headline_eligible"] for row in raw
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config["config_sha256"],
        "backend_kinds": sorted(backend_kinds),
        "headline_eligible": headline,
        "result_status": "real_cached_results" if headline else ("mock_contract_validation_only" if raw else "unrun"),
        "summary": summaries,
        "raw_predictions": raw,
    }
