"""BenchPress transform-by-method grid, adapted to arbitrary pathology matrices.

The algorithms and grids are ports of Microsoft BenchPress commit
0a684b63ee0e4a401cb907a3827a82ea997d74c4.  Unlike upstream, this module does
not depend on package-global matrix dimensions.  Inputs are already normalized
pathology scores on a 0--100 scale. See THIRD_PARTY_NOTICES.md for the upstream
files and preserved MIT notice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .completion import _bias_als

TRANSFORMS = ("identity", "log", "logit", "asinh", "sqrt", "probit", "quantile")
METHODS = (
    "Benchmark Mean",
    "Model Mean",
    "Bench-KNN",
    "Model-KNN",
    "BenchReg",
    "ModelReg",
    "Soft-Impute",
    "Bias ALS",
    "NMF",
    "PMF",
    "Nuclear Norm",
    "MLP",
)

# Pathology adaptation: upstream fixes Soft-Impute and Bias ALS at rank 2.
# Pathology CV selected rank 1.  We use rank 1 as the primary fixed rank and
# retain direct rank-2 sensitivity configurations.
HP_GRIDS: dict[str, list[dict[str, Any]]] = {
    "Benchmark Mean": [{}],
    "Model Mean": [{}],
    "Bench-KNN": [{"k": k} for k in (3, 5, 7, 10)],
    "Model-KNN": [{"k": k} for k in (3, 5, 7, 10)],
    "BenchReg": [
        {"top_k": k, "min_r2": r2}
        for k in (3, 5, 7)
        for r2 in (0.1, 0.2, 0.3)
    ],
    "ModelReg": [
        {"top_k": k, "min_r2": r2}
        for k in (3, 5, 7)
        for r2 in (0.1, 0.2, 0.3)
    ],
    "Soft-Impute": [{"rank": 1}, {"rank": 2, "sensitivity": True}],
    "Bias ALS": [
        {"rank": 1, "lam": lam} for lam in (0.01, 0.1, 1.0)
    ] + [{"rank": 2, "lam": 0.1, "sensitivity": True}],
    "NMF": [{"rank": rank} for rank in (1, 2, 3, 5)],
    "PMF": [{"rank": rank} for rank in (1, 2, 3, 5)],
    "Nuclear Norm": [{"lam": lam} for lam in (0.1, 0.5, 1.0, 5.0)],
    "MLP": [{"lr": lr} for lr in (1e-4, 1e-3, 1e-2)],
}


class UnsupportedMethodError(RuntimeError):
    """A method cannot run because an optional dependency is unavailable."""


@dataclass(frozen=True)
class TransformState:
    observed: np.ndarray
    is_percentage: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    quantiles: tuple[np.ndarray | None, ...]


def _column_normalize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    stds[~np.isfinite(stds) | (stds < 1e-8)] = 1.0
    return (matrix - means) / stds, means, stds


def _average_ranks(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise UnsupportedMethodError("quantile transform requires scipy") from exc
    return np.asarray(rankdata(values), dtype=float)


def _forward(values: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray(values, dtype=float)
    if name == "identity":
        return values, None
    if name == "log":
        return np.log1p(np.maximum(values, 0.0)), None
    if name == "logit":
        probability = np.clip(values, 0.5, 99.5) / 100.0
        return np.log(probability / (1.0 - probability)), None
    if name == "asinh":
        return np.arcsinh(values / 50.0), None
    if name == "sqrt":
        return np.sqrt(np.maximum(values, 0.0)), None
    if name == "probit":
        try:
            from scipy.stats import norm
        except ImportError as exc:  # pragma: no cover
            raise UnsupportedMethodError("probit transform requires scipy") from exc
        return norm.ppf(np.clip(values, 0.5, 99.5) / 100.0), None
    if name == "quantile":
        n = len(values)
        if n <= 1:
            return values, np.sort(values)
        return _average_ranks(values) / (n + 1), np.sort(values)
    raise ValueError(f"unknown transform: {name}")


def _inverse(values: np.ndarray, name: str, quantiles: np.ndarray | None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if name == "identity":
        return values
    if name == "log":
        return np.expm1(values)
    if name == "logit":
        return 100.0 / (1.0 + np.exp(-values))
    if name == "asinh":
        return np.sinh(values) * 50.0
    if name == "sqrt":
        return values**2
    if name == "probit":
        try:
            from scipy.stats import norm
        except ImportError as exc:  # pragma: no cover
            raise UnsupportedMethodError("probit transform requires scipy") from exc
        return norm.cdf(values) * 100.0
    if name == "quantile":
        if quantiles is None or len(quantiles) == 0:
            return values * 100.0
        n = len(quantiles)
        index = np.clip(values * (n + 1) - 1, 0, n - 1)
        low = np.floor(index).astype(int)
        high = np.minimum(low + 1, n - 1)
        fraction = index - low
        return quantiles[low] * (1.0 - fraction) + quantiles[high] * fraction
    raise ValueError(f"unknown transform: {name}")


def apply_transform(matrix: np.ndarray, name: str) -> tuple[np.ndarray, TransformState]:
    """Apply BenchPress's feature transform followed by column z-scoring."""
    if name not in TRANSFORMS:
        raise ValueError(f"unknown transform: {name}")
    matrix = np.asarray(matrix, dtype=float)
    observed = np.isfinite(matrix)
    is_percentage = np.asarray(
        [
            bool(observed[:, j].any())
            and float(np.nanmin(matrix[:, j])) >= -1.0
            and float(np.nanmax(matrix[:, j])) <= 101.0
            for j in range(matrix.shape[1])
        ],
        dtype=bool,
    )
    percentage_only = name in {"logit", "asinh", "sqrt", "probit"}
    transformed = matrix.copy()
    quantiles: list[np.ndarray | None] = []
    for j in range(matrix.shape[1]):
        valid = observed[:, j]
        if valid.any() and (not percentage_only or is_percentage[j]):
            values, state = _forward(matrix[valid, j], name)
            transformed[valid, j] = values
            quantiles.append(state)
        else:
            quantiles.append(None)
    normalized, means, stds = _column_normalize(transformed)
    return normalized, TransformState(observed, is_percentage, means, stds, tuple(quantiles))


def invert_transform(
    predicted_z: np.ndarray,
    training: np.ndarray,
    name: str,
    state: TransformState,
) -> np.ndarray:
    """Invert the pipeline only at missing cells and preserve training cells."""
    output = np.asarray(training, dtype=float).copy()
    percentage_only = name in {"logit", "asinh", "sqrt", "probit"}
    for j in range(training.shape[1]):
        missing = ~state.observed[:, j]
        finite = missing & np.isfinite(predicted_z[:, j])
        if not finite.any():
            continue
        values = predicted_z[finite, j] * state.stds[j] + state.means[j]
        if not percentage_only or state.is_percentage[j]:
            values = _inverse(values, name, state.quantiles[j])
        if state.is_percentage[j]:
            values = np.clip(values, 0.0, 100.0)
        output[finite, j] = values
    return output


def complete_benchmark_mean(matrix: np.ndarray) -> np.ndarray:
    output = matrix.copy()
    means = np.nanmean(matrix, axis=0)
    missing = ~np.isfinite(output)
    output[missing] = np.broadcast_to(means, output.shape)[missing]
    return output


def complete_model_mean(matrix: np.ndarray) -> np.ndarray:
    output = matrix.copy()
    observed = np.isfinite(matrix)
    row_count = observed.sum(axis=1)
    row_mean = np.divide(
        np.nansum(matrix, axis=1), row_count,
        out=np.full(matrix.shape[0], np.nan), where=row_count > 0,
    )
    column_count = observed.sum(axis=0)
    column_mean = np.divide(
        np.nansum(matrix, axis=0), column_count,
        out=np.full(matrix.shape[1], np.nan), where=column_count > 0,
    )
    for i, j in np.argwhere(~observed):
        output[i, j] = row_mean[i] if np.isfinite(row_mean[i]) else column_mean[j]
    return output


def complete_model_knn(matrix: np.ndarray, *, k: int = 5) -> np.ndarray:
    observed = np.isfinite(matrix)
    output = matrix.copy()
    for i in range(matrix.shape[0]):
        distances = []
        for other in range(matrix.shape[0]):
            if i == other:
                continue
            shared = observed[i] & observed[other]
            if shared.sum() >= 3:
                distance = float(np.sqrt(np.mean((matrix[i, shared] - matrix[other, shared]) ** 2)))
                distances.append((other, distance))
        neighbors = [other for other, _ in sorted(distances, key=lambda item: item[1])[:k]]
        for j in np.flatnonzero(~observed[i]):
            values = [matrix[other, j] for other in neighbors if observed[other, j]]
            output[i, j] = float(np.mean(values)) if values else float(np.nanmean(matrix[:, j]))
    return output


def complete_bench_knn(matrix: np.ndarray, *, k: int = 5) -> np.ndarray:
    observed = np.isfinite(matrix)
    output = matrix.copy()
    _, means, stds = _column_normalize(matrix)
    for j in range(matrix.shape[1]):
        correlations = []
        for other in range(matrix.shape[1]):
            if j == other:
                continue
            shared = observed[:, j] & observed[:, other]
            if shared.sum() < 5:
                continue
            correlation = float(np.corrcoef(matrix[shared, j], matrix[shared, other])[0, 1])
            if np.isfinite(correlation):
                correlations.append((other, correlation))
        top = sorted(correlations, key=lambda item: -item[1])[:k]
        for i in np.flatnonzero(~observed[:, j]):
            available = [(other, corr) for other, corr in top if observed[i, other]]
            if available:
                weights = np.asarray([max(corr, 0.01) for _, corr in available], dtype=float)
                weights /= weights.sum()
                output[i, j] = means[j] + stds[j] * sum(
                    weight * (matrix[i, other] - means[other]) / stds[other]
                    for weight, (other, _) in zip(weights, available)
                )
            else:
                output[i, j] = means[j]
    return output


def _regression_candidates(matrix: np.ndarray, target: int, *, axis: int) -> list[tuple[int, float]]:
    values = matrix.T if axis == 0 else matrix
    observed = np.isfinite(values)
    candidates = []
    for other in range(values.shape[0]):
        if other == target:
            continue
        shared = observed[target] & observed[other]
        if shared.sum() < 5:
            candidates.append((other, -1.0))
            continue
        x, y = values[other, shared], values[target, shared]
        total = float(np.sum((y - y.mean()) ** 2))
        variance_x = float(np.sum((x - x.mean()) ** 2))
        if total < 1e-10 or variance_x < 1e-10:
            candidates.append((other, -1.0))
            continue
        slope = float(np.sum((x - x.mean()) * (y - y.mean())) / variance_x)
        residual = float(np.sum((y - (y.mean() - slope * x.mean() + slope * x)) ** 2))
        candidates.append((other, 1.0 - residual / total))
    return sorted(candidates, key=lambda item: -item[1])


def _complete_regression(
    matrix: np.ndarray, *, axis: int, top_k: int = 5, min_r2: float = 0.2
) -> np.ndarray:
    values = matrix.T.copy() if axis == 0 else matrix.copy()
    observed = np.isfinite(values)
    output = values.copy()
    for target in range(values.shape[0]):
        if observed[target].sum() < 5:
            continue
        best = [item for item in _regression_candidates(matrix, target, axis=axis)[:top_k] if item[1] >= min_r2]
        for position in np.flatnonzero(~observed[target]):
            predictions, weights = [], []
            for other, r2 in best:
                if not observed[other, position]:
                    continue
                shared = observed[target] & observed[other]
                x, y = values[other, shared], values[target, shared]
                variance_x = float(np.sum((x - x.mean()) ** 2))
                if variance_x < 1e-10:
                    continue
                slope = float(np.sum((x - x.mean()) * (y - y.mean())) / variance_x)
                intercept = float(y.mean() - slope * x.mean())
                predictions.append(slope * values[other, position] + intercept)
                weights.append(r2)
            if predictions:
                output[target, position] = float(np.average(predictions, weights=weights))
    return output.T if axis == 0 else output


def complete_benchreg(matrix: np.ndarray, *, top_k: int = 5, min_r2: float = 0.2) -> np.ndarray:
    return _complete_regression(matrix, axis=0, top_k=top_k, min_r2=min_r2)


def complete_modelreg(matrix: np.ndarray, *, top_k: int = 5, min_r2: float = 0.2) -> np.ndarray:
    return _complete_regression(matrix, axis=1, top_k=top_k, min_r2=min_r2)


def complete_soft_impute(matrix: np.ndarray, *, rank: int, **_: Any) -> np.ndarray:
    observed = np.isfinite(matrix)
    imputed = matrix.copy()
    missing = ~observed
    means = np.nanmean(matrix, axis=0)
    imputed[missing] = np.broadcast_to(means, matrix.shape)[missing]
    for _iteration in range(100):
        previous = imputed.copy()
        left, singular, right = np.linalg.svd(imputed, full_matrices=False)
        keep = min(rank, len(singular))
        approximation = left[:, :keep] @ np.diag(singular[:keep]) @ right[:keep, :]
        imputed = np.where(observed, matrix, approximation)
        relative = np.sqrt(np.mean((imputed - previous) ** 2)) / (np.sqrt(np.mean(previous**2)) + 1e-12)
        if relative < 1e-4:
            break
    imputed[observed] = matrix[observed]
    return imputed


def complete_bias_als(matrix: np.ndarray, *, rank: int, lam: float, **_: Any) -> np.ndarray:
    """Upstream bias ALS with algebraically identical batched sufficient statistics.

    BenchPress solves one small ridge system at a time.  Batching those systems
    is important for the 30-fold pathology sweep, especially for rank 2.
    """
    if rank <= 1:
        # PathoPress's primary rank has an algebraically exact optimized path.
        return _bias_als(matrix, rank=rank, regularization=lam)

    observed = np.isfinite(matrix)
    observed_float = observed.astype(float)
    cells = np.argwhere(observed)
    values = matrix[observed]
    n_rows, n_columns = matrix.shape
    row_supported = observed.any(axis=1)
    column_supported = observed.any(axis=0)
    ridge = np.eye(rank + 1) * lam

    def run_one(seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        mean = float(np.mean(values))
        row_bias = np.zeros(n_rows)
        column_bias = np.zeros(n_columns)
        row_factors = rng.normal(0.0, 0.01, size=(n_rows, rank))
        column_factors = rng.normal(0.0, 0.01, size=(n_columns, rank))
        for _iteration in range(40):
            column_design = np.column_stack([np.ones(n_columns), column_factors])
            column_outer = np.einsum("ja,jb->jab", column_design, column_design)
            row_system = np.einsum("ij,jab->iab", observed_float, column_outer) + ridge
            row_target = np.where(observed, matrix - mean - column_bias[None, :], 0.0)
            row_rhs = row_target @ column_design
            row_solution = np.linalg.solve(row_system, row_rhs[..., None])[..., 0]
            row_bias[row_supported] = row_solution[row_supported, 0]
            row_factors[row_supported] = row_solution[row_supported, 1:]

            row_design = np.column_stack([np.ones(n_rows), row_factors])
            row_outer = np.einsum("ia,ib->iab", row_design, row_design)
            column_system = np.einsum("ij,iab->jab", observed_float, row_outer) + ridge
            column_target = np.where(observed, matrix - mean - row_bias[:, None], 0.0)
            column_rhs = column_target.T @ row_design
            column_solution = np.linalg.solve(column_system, column_rhs[..., None])[..., 0]
            column_bias[column_supported] = column_solution[column_supported, 0]
            column_factors[column_supported] = column_solution[column_supported, 1:]

            interactions = np.einsum(
                "ij,ij->i", row_factors[cells[:, 0]], column_factors[cells[:, 1]]
            )
            mean = float(
                np.mean(
                    values
                    - row_bias[cells[:, 0]]
                    - column_bias[cells[:, 1]]
                    - interactions
                )
            )
        return mean + row_bias[:, None] + column_bias[None, :] + row_factors @ column_factors.T

    output = sum(run_one(42 + offset) for offset in range(10)) / 10
    output[observed] = matrix[observed]
    return output


def complete_nuclear_norm(matrix: np.ndarray, *, lam: float, **_: Any) -> np.ndarray:
    observed = np.isfinite(matrix)
    means = np.nanmean(matrix, axis=0)
    imputed = matrix.copy()
    missing = ~observed
    imputed[missing] = np.broadcast_to(means, matrix.shape)[missing]
    for _iteration in range(200):
        gradient = np.zeros_like(imputed)
        gradient[observed] = imputed[observed] - matrix[observed]
        left, singular, right = np.linalg.svd(imputed - 0.1 * gradient, full_matrices=False)
        thresholded = np.maximum(singular - lam * 0.1, 0.0)
        imputed = left @ np.diag(thresholded) @ right
    imputed[observed] = matrix[observed]
    return imputed


def complete_nmf(matrix: np.ndarray, *, rank: int, **_: Any) -> np.ndarray:
    observed = np.isfinite(matrix)
    column_min = np.nanmin(matrix, axis=0)
    shift = np.where(column_min < 0, -column_min + 0.1, 0.0)
    shifted = matrix.copy()
    for j in range(matrix.shape[1]):
        shifted[observed[:, j], j] += shift[j]
    shifted[~observed] = 0.0
    rng = np.random.RandomState(42)
    scale = np.sqrt(np.nanmean(shifted[observed]) / rank + 0.01)
    left = np.abs(rng.randn(matrix.shape[0], rank)) * scale + 0.1
    right = np.abs(rng.randn(rank, matrix.shape[1])) * scale + 0.1
    for _iteration in range(500):
        error = np.zeros_like(shifted)
        approximation = left @ right
        error[observed] = approximation[observed] - shifted[observed]
        left = np.maximum(left - 0.0005 * (error @ right.T + 0.01 * left), 1e-10)
        right = np.maximum(right - 0.0005 * (left.T @ error + 0.01 * right), 1e-10)
    output = left @ right - shift[None, :]
    output[observed] = matrix[observed]
    return output


def complete_pmf(matrix: np.ndarray, *, rank: int, **_: Any) -> np.ndarray:
    observed = np.isfinite(matrix)
    working = np.where(observed, matrix, 0.0)
    rng = np.random.RandomState(42)
    left = rng.randn(matrix.shape[0], rank) * 0.1
    right = rng.randn(matrix.shape[1], rank) * 0.1
    for _iteration in range(300):
        error = np.zeros_like(working)
        approximation = left @ right.T
        error[observed] = approximation[observed] - working[observed]
        left -= 0.001 * (error @ right + 0.1 * left)
        right -= 0.001 * (error.T @ left + 0.1 * right)
    output = left @ right.T
    output[observed] = matrix[observed]
    return output


def complete_mlp(
    matrix: np.ndarray, *, lr: float, hidden: int = 32, epochs: int = 500, n_seeds: int = 3
) -> np.ndarray:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise UnsupportedMethodError("MLP requires torch") from exc
    observed = np.isfinite(matrix)
    normalized, means, stds = _column_normalize(matrix)
    inputs = torch.tensor(np.where(observed, normalized, 0.0), dtype=torch.float32)
    mask = torch.tensor(observed, dtype=torch.float32)
    predictions = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        network = nn.Sequential(
            nn.Linear(matrix.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, matrix.shape[1])
        )
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        for _epoch in range(epochs):
            output = network(inputs)
            loss = (((output - inputs) ** 2) * mask).sum() / mask.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predictions.append(network(inputs).numpy() * stds + means)
    output = np.mean(predictions, axis=0)
    output[observed] = matrix[observed]
    return output


COMPLETERS: dict[str, Callable[..., np.ndarray]] = {
    "Benchmark Mean": complete_benchmark_mean,
    "Model Mean": complete_model_mean,
    "Bench-KNN": complete_bench_knn,
    "Model-KNN": complete_model_knn,
    "BenchReg": complete_benchreg,
    "ModelReg": complete_modelreg,
    "Soft-Impute": complete_soft_impute,
    "Bias ALS": complete_bias_als,
    "NMF": complete_nmf,
    "PMF": complete_pmf,
    "Nuclear Norm": complete_nuclear_norm,
    "MLP": complete_mlp,
}


def predict_scores(
    training: np.ndarray, transform: str, method: str, hyperparameters: dict[str, Any]
) -> np.ndarray:
    """Run one exact transform/completer pipeline and return raw-scale scores."""
    normalized, state = apply_transform(training, transform)
    # Within-model folds can remove every observation from an exceptionally
    # sparse benchmark column.  Such a column is not identifiable in that
    # fold: exclude it from the numerical solver and restore it as NaN so the
    # evaluation layer records honest non-coverage instead of contaminating
    # SVD/MLP inputs with an all-NaN feature.
    supported = np.isfinite(normalized).any(axis=0)
    kwargs = {key: value for key, value in hyperparameters.items() if key != "sensitivity"}
    completed = np.full_like(normalized, np.nan)
    if supported.any():
        completed[:, supported] = COMPLETERS[method](normalized[:, supported], **kwargs)
    return invert_transform(completed, training, transform, state)
