"""Local covariance spectrum metrics for MTEB unsupervised evaluation."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

LOCAL_SUMMARY_SUFFIXES = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "q25",
    "q75",
    "q10",
    "q90",
    "q05",
    "q95",
)
LOCAL_COV_BRANCHES = ("cov", "cov2")
LOCAL_COV_BASE_TRANSFORMS = ("rankme", "ne_sum", "participation_ratio")
LOCAL_COV_TRANSFORM_GROUPS = (*LOCAL_COV_BASE_TRANSFORMS, "invariants")
DEFAULT_LOCAL_COV_TRANSFORMS = LOCAL_COV_TRANSFORM_GROUPS
DEFAULT_LOCAL_COV_N_NEIGHBORS = (50,)
DEFAULT_LOCAL_COV_INVARIANT_MAX_ORDER = 8
DEFAULT_LOCAL_COV_DEVICE = "cpu"
DEFAULT_LOCAL_COV_TOP_EIGEN_ITERS = 12


def normalize_local_cov_n_neighbors(n_neighbors: int | Sequence[int] | str) -> Tuple[int, ...]:
    if isinstance(n_neighbors, str):
        raw = [part for part in n_neighbors.replace(",", " ").split() if part]
        values = [int(part) for part in raw]
    elif isinstance(n_neighbors, (int, np.integer)):
        values = [int(n_neighbors)]
    else:
        values = [int(k) for k in n_neighbors]
    values = sorted(dict.fromkeys(values))
    if not values:
        values = [50]
    if any(k < 2 for k in values):
        raise ValueError("local_cov_n_neighbors values must be at least 2")
    return tuple(values)


def normalize_local_cov_transforms(transforms: Sequence[str] | str | None) -> Tuple[str, ...]:
    if transforms is None:
        raw = list(DEFAULT_LOCAL_COV_TRANSFORMS)
    elif isinstance(transforms, str):
        raw = [part for part in transforms.replace(",", " ").split() if part]
    else:
        raw = [str(part) for part in transforms]

    normalized = []
    for value in raw:
        name = value.strip().lower().replace("-", "_")
        if not name:
            continue
        if name == "all":
            normalized.extend(DEFAULT_LOCAL_COV_TRANSFORMS)
        elif name in {"invariant", "invariants"}:
            normalized.append("invariants")
        elif name in LOCAL_COV_BASE_TRANSFORMS:
            normalized.append(name)
        else:
            valid = ", ".join((*LOCAL_COV_TRANSFORM_GROUPS, "all"))
            raise ValueError(f"Unknown local covariance transform '{value}'. Valid values: {valid}")

    deduped = tuple(dict.fromkeys(normalized))
    if not deduped:
        raise ValueError("At least one local covariance transform must be selected")
    return deduped


def local_cov_spectrum_output_names(
    invariant_max_order: int,
    n_neighbors: int | Sequence[int] | str = DEFAULT_LOCAL_COV_N_NEIGHBORS,
    transforms: Sequence[str] | str | None = DEFAULT_LOCAL_COV_TRANSFORMS,
) -> Tuple[str, ...]:
    max_order = max(0, int(invariant_max_order))
    k_values = normalize_local_cov_n_neighbors(n_neighbors)
    selected = normalize_local_cov_transforms(transforms)
    transform_names = [name for name in LOCAL_COV_BASE_TRANSFORMS if name in selected]
    if "invariants" in selected:
        transform_names.extend(f"invariant_l{order}" for order in range(1, max_order + 1))
    return tuple(
        f"local_cov_spectrum_k{k}_{branch}_{transform}_{suffix}"
        for k in k_values
        for branch in LOCAL_COV_BRANCHES
        for transform in transform_names
        for suffix in LOCAL_SUMMARY_SUFFIXES
    )


def local_mle_cov_mst_output_names(
    invariant_max_order: int,
    n_neighbors: int | Sequence[int] | str = DEFAULT_LOCAL_COV_N_NEIGHBORS,
) -> Tuple[str, ...]:
    max_order = max(0, int(invariant_max_order))
    k_values = normalize_local_cov_n_neighbors(n_neighbors)
    transform_names = ["volume_over_trace", "trace_over_volume"]
    transform_names.extend(
        f"invariant_l{order}_volume_over_trace"
        for order in range(1, max_order + 1)
    )
    transform_names.extend(
        f"invariant_l{order}_trace_over_volume"
        for order in range(1, max_order + 1)
    )
    return tuple(
        f"local_mle_cov_mst_k{k}_{transform}_{suffix}"
        for k in k_values
        for transform in transform_names
        for suffix in LOCAL_SUMMARY_SUFFIXES
    )


def _finite_point_summary(values: np.ndarray, prefix: str) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_{suffix}": float("nan") for suffix in LOCAL_SUMMARY_SUFFIXES}
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q25": float(np.percentile(arr, 25)),
        f"{prefix}_q75": float(np.percentile(arr, 75)),
        f"{prefix}_q10": float(np.percentile(arr, 10)),
        f"{prefix}_q90": float(np.percentile(arr, 90)),
        f"{prefix}_q05": float(np.percentile(arr, 5)),
        f"{prefix}_q95": float(np.percentile(arr, 95)),
    }


def _rankme_from_spectrum(spectrum: np.ndarray, epsilon: float = 1e-12) -> float:
    vals = np.asarray(spectrum, dtype=np.float64)
    total = float(vals.sum())
    if not np.isfinite(total) or total <= epsilon:
        return float("nan")
    p_ks = vals / (total + epsilon) + epsilon
    return float(np.exp(-np.sum(p_ks * np.log(p_ks))))


def _rankme_per_point(spectra: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    vals = np.asarray(spectra, dtype=np.float64)
    totals = vals.sum(axis=1, keepdims=True)
    valid = np.isfinite(totals[:, 0]) & (totals[:, 0] > epsilon)
    out = np.full(vals.shape[0], np.nan, dtype=np.float64)
    if not valid.any():
        return out
    p_ks = vals[valid] / (totals[valid] + epsilon) + epsilon
    out[valid] = np.exp(-np.sum(p_ks * np.log(p_ks), axis=1))
    return out


def _ne_sum_from_spectrum(spectrum: np.ndarray, epsilon: float = 1e-12) -> float:
    vals = np.asarray(spectrum, dtype=np.float64)
    max_val = float(vals.max(initial=0.0))
    if not np.isfinite(max_val) or max_val <= epsilon:
        return float("nan")
    return float((vals / (max_val + epsilon)).sum())


def _ne_sum_per_point(spectra: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    vals = np.asarray(spectra, dtype=np.float64)
    trace = vals.sum(axis=1)
    lambda1 = vals[:, 0] if vals.shape[1] else np.zeros(vals.shape[0], dtype=np.float64)
    valid = np.isfinite(trace) & np.isfinite(lambda1) & (lambda1 > epsilon)
    out = np.full(vals.shape[0], np.nan, dtype=np.float64)
    out[valid] = trace[valid] / (lambda1[valid] + epsilon)
    return out


def _participation_ratio_from_spectrum(spectrum: np.ndarray, epsilon: float = 1e-12) -> float:
    vals = np.asarray(spectrum, dtype=np.float64)
    total = float(vals.sum())
    sq_total = float(np.square(vals).sum())
    if not np.isfinite(total) or not np.isfinite(sq_total) or sq_total <= epsilon:
        return float("nan")
    return float(total * total / (sq_total + epsilon))


def _participation_ratio_per_point(spectra: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    vals = np.asarray(spectra, dtype=np.float64)
    trace = vals.sum(axis=1)
    fro_sq = np.square(vals).sum(axis=1)
    valid = np.isfinite(trace) & np.isfinite(fro_sq) & (fro_sq > epsilon)
    out = np.full(vals.shape[0], np.nan, dtype=np.float64)
    out[valid] = trace[valid] * trace[valid] / (fro_sq[valid] + epsilon)
    return out


def _elementary_symmetric_invariants(
    spectrum: np.ndarray,
    max_order: int,
    epsilon: float = 1e-12,
) -> np.ndarray:
    vals = np.asarray(spectrum, dtype=np.float64)
    max_val = float(vals.max(initial=0.0))
    out = np.full(max(0, int(max_order)), np.nan, dtype=np.float64)
    if out.size == 0 or not np.isfinite(max_val) or max_val <= epsilon:
        return out

    positive = vals[vals > epsilon]
    usable_rank = int(positive.size)
    if usable_rank == 0:
        return out

    e = np.zeros(out.size + 1, dtype=np.float64)
    e[0] = 1.0
    for idx, val in enumerate(positive, start=1):
        upto = min(idx, out.size)
        for order in range(upto, 0, -1):
            e[order] += e[order - 1] * val

    for order in range(1, min(usable_rank, out.size) + 1):
        out[order - 1] = float(np.power(e[order], 1.0 / order) / (max_val + epsilon))
    return out


def _elementary_symmetric_invariants_per_point(
    spectra: np.ndarray,
    max_order: int,
    epsilon: float = 1e-12,
) -> np.ndarray:
    vals = np.asarray(spectra, dtype=np.float64)
    n_points = vals.shape[0]
    out = np.full((n_points, max(0, int(max_order))), np.nan, dtype=np.float64)
    if out.shape[1] == 0:
        return out

    max_vals = vals[:, 0] if vals.shape[1] else np.zeros(n_points, dtype=np.float64)
    positive = np.where(vals > epsilon, vals, 0.0)
    usable_rank = (positive > 0.0).sum(axis=1)
    valid = np.isfinite(max_vals) & (max_vals > epsilon)
    if not valid.any():
        return out

    e = np.zeros((n_points, out.shape[1] + 1), dtype=np.float64)
    e[:, 0] = 1.0
    for val in positive.T:
        for order in range(out.shape[1], 0, -1):
            e[:, order] += e[:, order - 1] * val

    for order in range(1, out.shape[1] + 1):
        order_valid = valid & (usable_rank >= order) & np.isfinite(e[:, order]) & (e[:, order] > 0.0)
        out[order_valid, order - 1] = (
            np.power(e[order_valid, order], 1.0 / order)
            / (max_vals[order_valid] + epsilon)
        )
    return out


def _elementary_symmetric_invariants_from_power_sums(
    power_sums: np.ndarray,
    max_order: int,
    lambda_max: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    pows = np.asarray(power_sums, dtype=np.float64)
    max_vals = np.asarray(lambda_max, dtype=np.float64)
    n_points = pows.shape[0]
    out = np.full((n_points, max(0, int(max_order))), np.nan, dtype=np.float64)
    if out.shape[1] == 0:
        return out

    e = np.zeros((n_points, out.shape[1] + 1), dtype=np.float64)
    e[:, 0] = 1.0
    for order in range(1, out.shape[1] + 1):
        acc = np.zeros(n_points, dtype=np.float64)
        for i in range(1, order + 1):
            sign = 1.0 if i % 2 == 1 else -1.0
            acc += sign * e[:, order - i] * pows[:, i]
        e[:, order] = acc / order

    valid_base = np.isfinite(max_vals) & (max_vals > epsilon)
    for order in range(1, out.shape[1] + 1):
        order_valid = valid_base & np.isfinite(e[:, order]) & (e[:, order] > 0.0)
        out[order_valid, order - 1] = (
            np.power(e[order_valid, order], 1.0 / order)
            / (max_vals[order_valid] + epsilon)
        )
    return out


def _elementary_symmetric_roots_from_power_sums(
    power_sums: np.ndarray,
    max_order: int,
) -> np.ndarray:
    pows = np.asarray(power_sums, dtype=np.float64)
    n_points = pows.shape[0]
    out = np.full((n_points, max(0, int(max_order))), np.nan, dtype=np.float64)
    if out.shape[1] == 0:
        return out

    e = np.zeros((n_points, out.shape[1] + 1), dtype=np.float64)
    e[:, 0] = 1.0
    for order in range(1, out.shape[1] + 1):
        acc = np.zeros(n_points, dtype=np.float64)
        for i in range(1, order + 1):
            sign = 1.0 if i % 2 == 1 else -1.0
            acc += sign * e[:, order - i] * pows[:, i]
        e[:, order] = acc / order
        valid = np.isfinite(e[:, order]) & (e[:, order] > 0.0)
        out[valid, order - 1] = np.power(e[valid, order], 1.0 / order)
    return out


def _local_mle_dimensions(
    embeddings: np.ndarray,
    n_neighbors: int,
    n_jobs: int = -1,
) -> np.ndarray:
    import skdim.id

    estimator = skdim.id.MLE()
    dims = estimator.fit_transform_pw(
        np.asarray(embeddings, dtype=np.float32),
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
    )
    return np.asarray(dims, dtype=np.float64)


def _local_covariance_eigvals(
    embeddings: np.ndarray,
    n_neighbors: int,
    n_jobs: int = -1,
    device: str = DEFAULT_LOCAL_COV_DEVICE,
) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    n_points, n_features = arr.shape
    if n_points < 3 or n_features < 1:
        return np.empty((0, n_features), dtype=np.float64)

    k = max(2, min(int(n_neighbors), n_points - 1))
    device_name = str(device or "cpu").lower()
    if device_name.startswith("cuda"):
        try:
            return _local_covariance_eigvals_torch_cuda(arr, k, device_name)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    indices = _nearest_neighbor_indices_cpu(arr, k, n_jobs=n_jobs)
    return _local_covariance_eigvals_from_indices(arr, indices)


def _local_covariance_moment_stats(
    embeddings: np.ndarray,
    n_neighbors: int,
    max_power: int,
    *,
    estimate_top: bool,
    n_jobs: int = -1,
    device: str = DEFAULT_LOCAL_COV_DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(embeddings, dtype=np.float32)
    n_points, n_features = arr.shape
    if n_points < 3 or n_features < 1:
        return (
            np.empty((0, max(0, int(max_power)) + 1), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    k = max(2, min(int(n_neighbors), n_points - 1))
    device_name = str(device or "cpu").lower()
    if device_name.startswith("cuda"):
        try:
            return _local_covariance_moment_stats_torch_cuda(
                arr,
                k,
                max_power,
                estimate_top=estimate_top,
                device=device_name,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    indices = _nearest_neighbor_indices_cpu(arr, k, n_jobs=n_jobs)
    gram = _local_covariance_gram_from_indices(arr, indices)
    return _moment_stats_from_gram(gram, max_power, estimate_top=estimate_top)


def _nearest_neighbor_indices_cpu(arr: np.ndarray, k: int, n_jobs: int = -1) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs)
    knn.fit(arr)
    return knn.kneighbors(return_distance=False)


def _local_covariance_gram_from_indices(
    arr: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    neighborhoods = arr[indices].astype(np.float32, copy=False)
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    denom = max(centered.shape[1] - 1, 1)
    return np.matmul(centered, np.swapaxes(centered, 1, 2)) / denom


def _local_covariance_eigvals_from_indices(
    arr: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    # Nonzero eigenvalues of X^T X and X X^T match; k x k is much cheaper when k << d.
    gram = _local_covariance_gram_from_indices(arr, indices)
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = np.clip(eigvals, 0.0, None)[:, ::-1]
    return eigvals.astype(np.float64, copy=False)


def _moment_stats_from_gram(
    gram: np.ndarray,
    max_power: int,
    *,
    estimate_top: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    max_power = max(0, int(max_power))
    n_points = gram.shape[0]
    power_sums = np.zeros((n_points, max_power + 1), dtype=np.float64)
    if max_power > 0:
        current = gram.astype(np.float64, copy=False)
        for power in range(1, max_power + 1):
            power_sums[:, power] = np.trace(current, axis1=1, axis2=2)
            if power != max_power:
                current = np.matmul(current, gram)

    if estimate_top:
        lambda1 = _top_eigenvalue_power_from_gram(gram)
    else:
        lambda1 = np.zeros(n_points, dtype=np.float64)
    return power_sums, lambda1


def _top_eigenvalue_power_from_gram(
    gram: np.ndarray,
    n_iter: int = DEFAULT_LOCAL_COV_TOP_EIGEN_ITERS,
    epsilon: float = 1e-12,
) -> np.ndarray:
    if gram.size == 0:
        return np.empty(0, dtype=np.float64)
    n_points, k, _ = gram.shape
    v = np.full((n_points, k, 1), 1.0 / np.sqrt(k), dtype=np.float64)
    gram64 = gram.astype(np.float64, copy=False)
    for _ in range(max(1, int(n_iter))):
        v = np.matmul(gram64, v)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.divide(v, np.maximum(norms, epsilon), out=np.zeros_like(v), where=norms > epsilon)
    av = np.matmul(gram64, v)
    return np.clip(np.sum(v * av, axis=(1, 2)), 0.0, None)


def _local_covariance_eigvals_torch_cuda(
    arr: np.ndarray,
    k: int,
    device: str,
) -> np.ndarray:
    import torch

    if not torch.cuda.is_available():
        return _local_covariance_eigvals_from_indices(arr, _nearest_neighbor_indices_cpu(arr, k))

    dev = torch.device(device)
    with torch.inference_mode():
        x = torch.as_tensor(arr, dtype=torch.float32, device=dev)
        dists = torch.cdist(x, x, p=2)
        indices = torch.topk(dists, k=k, largest=False).indices
        neighborhoods = x[indices]
        centered = neighborhoods - neighborhoods.mean(dim=1, keepdim=True)
        # Nonzero eigenvalues of X^T X and X X^T match; k x k is much cheaper when k << d.
        gram = torch.matmul(centered, centered.transpose(1, 2)) / max(k - 1, 1)
        eigvals = torch.linalg.eigvalsh(gram).clamp_min_(0.0).flip(dims=(-1,))
        out = eigvals.detach().cpu().numpy().astype(np.float64, copy=False)
    return out


def _local_covariance_moment_stats_torch_cuda(
    arr: np.ndarray,
    k: int,
    max_power: int,
    *,
    estimate_top: bool,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    if not torch.cuda.is_available():
        indices = _nearest_neighbor_indices_cpu(arr, k)
        gram = _local_covariance_gram_from_indices(arr, indices)
        return _moment_stats_from_gram(gram, max_power, estimate_top=estimate_top)

    dev = torch.device(device)
    max_power = max(0, int(max_power))
    with torch.inference_mode():
        x = torch.as_tensor(arr, dtype=torch.float32, device=dev)
        dists = torch.cdist(x, x, p=2)
        indices = torch.topk(dists, k=k, largest=False).indices
        neighborhoods = x[indices]
        centered = neighborhoods - neighborhoods.mean(dim=1, keepdim=True)
        gram = torch.matmul(centered, centered.transpose(1, 2)) / max(k - 1, 1)

        power_sums = torch.zeros(
            (gram.shape[0], max_power + 1),
            dtype=torch.float64,
            device=dev,
        )
        if max_power > 0:
            current = gram.to(torch.float64)
            gram64 = current
            for power in range(1, max_power + 1):
                power_sums[:, power] = current.diagonal(dim1=1, dim2=2).sum(dim=1)
                if power != max_power:
                    current = torch.matmul(current, gram64)

        if estimate_top:
            lambda1 = _top_eigenvalue_power_from_gram_torch(gram)
        else:
            lambda1 = torch.zeros(gram.shape[0], dtype=torch.float64, device=dev)

        return (
            power_sums.detach().cpu().numpy().astype(np.float64, copy=False),
            lambda1.detach().cpu().numpy().astype(np.float64, copy=False),
        )


def _top_eigenvalue_power_from_gram_torch(
    gram,
    n_iter: int = DEFAULT_LOCAL_COV_TOP_EIGEN_ITERS,
    epsilon: float = 1e-12,
):
    import torch

    if gram.numel() == 0:
        return torch.empty(0, dtype=torch.float64, device=gram.device)
    n_points, k, _ = gram.shape
    gram64 = gram.to(torch.float64)
    v = torch.full(
        (n_points, k, 1),
        1.0 / np.sqrt(k),
        dtype=torch.float64,
        device=gram.device,
    )
    eps = torch.as_tensor(epsilon, dtype=torch.float64, device=gram.device)
    for _ in range(max(1, int(n_iter))):
        v = torch.matmul(gram64, v)
        norms = torch.linalg.vector_norm(v, dim=1, keepdim=True)
        v = torch.where(norms > eps, v / torch.clamp_min(norms, epsilon), torch.zeros_like(v))
    av = torch.matmul(gram64, v)
    return torch.sum(v * av, dim=(1, 2)).clamp_min_(0.0)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    valid = np.isfinite(num) & np.isfinite(den) & (den > epsilon)
    out = np.full(num.shape, np.nan, dtype=np.float64)
    out[valid] = num[valid] / (den[valid] + epsilon)
    return out


def _local_cov_moment_per_point(
    power_sums: np.ndarray,
    lambda1: np.ndarray,
    *,
    k: int,
    max_order: int,
    selected_transforms: Tuple[str, ...],
) -> Dict[str, np.ndarray]:
    per_point: Dict[str, np.ndarray] = {}
    p1 = power_sums[:, 1] if power_sums.shape[1] > 1 else np.zeros(power_sums.shape[0])
    p2 = power_sums[:, 2] if power_sums.shape[1] > 2 else np.zeros(power_sums.shape[0])
    p4 = power_sums[:, 4] if power_sums.shape[1] > 4 else np.zeros(power_sums.shape[0])

    lambda1_sq = np.square(lambda1)
    branch_inputs = {
        "cov": (p1, p2, lambda1, power_sums),
        "cov2": (
            p2,
            p4,
            lambda1_sq,
            _squared_spectrum_power_sums(power_sums, max_order),
        ),
    }
    for branch, (trace, trace_sq, top, branch_power_sums) in branch_inputs.items():
        if "ne_sum" in selected_transforms:
            per_point[f"local_cov_spectrum_k{k}_{branch}_ne_sum"] = _safe_ratio(trace, top)
        if "participation_ratio" in selected_transforms:
            per_point[f"local_cov_spectrum_k{k}_{branch}_participation_ratio"] = _safe_ratio(
                np.square(trace),
                trace_sq,
            )
        if "invariants" in selected_transforms:
            invariants = _elementary_symmetric_invariants_from_power_sums(
                branch_power_sums,
                max_order,
                top,
            )
            for order in range(1, max_order + 1):
                per_point[f"local_cov_spectrum_k{k}_{branch}_invariant_l{order}"] = invariants[:, order - 1]
    return per_point


def _squared_spectrum_power_sums(power_sums: np.ndarray, max_order: int) -> np.ndarray:
    out = np.zeros((power_sums.shape[0], max(0, int(max_order)) + 1), dtype=np.float64)
    for order in range(1, out.shape[1]):
        source = 2 * order
        if source < power_sums.shape[1]:
            out[:, order] = power_sums[:, source]
    return out


def _local_mle_cov_mst_per_point(
    power_sums: np.ndarray,
    mle_dims: np.ndarray,
    *,
    k: int,
    max_order: int,
    epsilon: float = 1e-12,
) -> Dict[str, np.ndarray]:
    dims = np.asarray(mle_dims, dtype=np.float64)
    trace = power_sums[:, 1] if power_sums.shape[1] > 1 else np.zeros(power_sums.shape[0])
    roots = _elementary_symmetric_roots_from_power_sums(power_sums, max_order)
    per_point: Dict[str, np.ndarray] = {}
    n_points = power_sums.shape[0]
    valid_dim = np.isfinite(dims) & (dims > epsilon)
    valid_trace = np.isfinite(trace) & (trace > epsilon)
    valid = valid_dim & valid_trace
    if not valid.any() or max_order < 1:
        per_point["volume_over_trace"] = np.full(n_points, np.nan, dtype=np.float64)
        per_point["trace_over_volume"] = np.full(n_points, np.nan, dtype=np.float64)
        for order in range(1, max_order + 1):
            per_point[f"invariant_l{order}_volume_over_trace"] = np.full(n_points, np.nan, dtype=np.float64)
            per_point[f"invariant_l{order}_trace_over_volume"] = np.full(n_points, np.nan, dtype=np.float64)
        return per_point

    orders = np.clip(np.rint(dims).astype(np.int64), 1, max_order)
    row_idx = np.arange(power_sums.shape[0])
    alpha = (dims - 1.0) / dims
    scale = np.full(n_points, np.nan, dtype=np.float64)
    scale_valid = valid & np.isfinite(alpha)
    scale[scale_valid] = np.power(float(k), alpha[scale_valid])

    adaptive_volume = roots[row_idx, orders - 1]
    per_point["volume_over_trace"] = _scaled_ratio(scale, adaptive_volume, trace, epsilon=epsilon)
    per_point["trace_over_volume"] = _scaled_ratio(scale, trace, adaptive_volume, epsilon=epsilon)
    for order in range(1, max_order + 1):
        volume = roots[:, order - 1]
        per_point[f"invariant_l{order}_volume_over_trace"] = _scaled_ratio(
            scale,
            volume,
            trace,
            epsilon=epsilon,
        )
        per_point[f"invariant_l{order}_trace_over_volume"] = _scaled_ratio(
            scale,
            trace,
            volume,
            epsilon=epsilon,
        )
    return per_point


def _scaled_ratio(
    scale: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    sc = np.asarray(scale, dtype=np.float64)
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    valid = np.isfinite(sc) & np.isfinite(num) & np.isfinite(den) & (den > epsilon)
    out = np.full(num.shape, np.nan, dtype=np.float64)
    out[valid] = sc[valid] * num[valid] / (den[valid] + epsilon)
    return out


def local_mle_cov_mst_metric(
    sample: np.ndarray,
    u=None,
    s=None,
    *,
    n_neighbors: int | Sequence[int] | str = DEFAULT_LOCAL_COV_N_NEIGHBORS,
    invariant_max_order: int = DEFAULT_LOCAL_COV_INVARIANT_MAX_ORDER,
    device: str = DEFAULT_LOCAL_COV_DEVICE,
) -> Dict[str, float]:
    max_order = max(0, int(invariant_max_order))
    k_values = normalize_local_cov_n_neighbors(n_neighbors)
    output_names = local_mle_cov_mst_output_names(max_order, k_values)
    out: Dict[str, float] = {}

    for k in k_values:
        if max_order < 1:
            out.update({
                name: float("nan")
                for name in local_mle_cov_mst_output_names(max_order, (k,))
            })
            continue

        power_sums, _lambda1 = _local_covariance_moment_stats(
            sample,
            n_neighbors=k,
            max_power=max_order,
            estimate_top=False,
            n_jobs=-1,
            device=device,
        )
        if power_sums.size == 0:
            out.update({
                name: float("nan")
                for name in local_mle_cov_mst_output_names(max_order, (k,))
            })
            continue

        mle_dims = _local_mle_dimensions(sample, k, n_jobs=-1)
        per_point = _local_mle_cov_mst_per_point(
            power_sums,
            mle_dims,
            k=k,
            max_order=max_order,
        )
        for name, values in per_point.items():
            out.update(_finite_point_summary(values, f"local_mle_cov_mst_k{k}_{name}"))

    return {name: out.get(name, float("nan")) for name in output_names}


def local_cov_spectrum_metric(
    sample: np.ndarray,
    u=None,
    s=None,
    *,
    n_neighbors: int | Sequence[int] | str = DEFAULT_LOCAL_COV_N_NEIGHBORS,
    invariant_max_order: int = DEFAULT_LOCAL_COV_INVARIANT_MAX_ORDER,
    transforms: Sequence[str] | str | None = DEFAULT_LOCAL_COV_TRANSFORMS,
    device: str = DEFAULT_LOCAL_COV_DEVICE,
) -> Dict[str, float]:
    max_order = max(0, int(invariant_max_order))
    k_values = normalize_local_cov_n_neighbors(n_neighbors)
    selected_transforms = normalize_local_cov_transforms(transforms)
    output_names = local_cov_spectrum_output_names(max_order, k_values, selected_transforms)
    out: Dict[str, float] = {}

    for k in k_values:
        if "rankme" not in selected_transforms:
            max_power = 0
            if "ne_sum" in selected_transforms:
                max_power = max(max_power, 2)
            if "participation_ratio" in selected_transforms:
                max_power = max(max_power, 4)
            if "invariants" in selected_transforms:
                max_power = max(max_power, 2 * max_order)
            power_sums, lambda1 = _local_covariance_moment_stats(
                sample,
                n_neighbors=k,
                max_power=max_power,
                estimate_top=(
                    "ne_sum" in selected_transforms
                    or ("invariants" in selected_transforms and max_order > 0)
                ),
                n_jobs=-1,
                device=device,
            )
            if power_sums.size == 0:
                out.update({
                    name: float("nan")
                    for name in local_cov_spectrum_output_names(max_order, (k,), selected_transforms)
                })
                continue
            per_point = _local_cov_moment_per_point(
                power_sums,
                lambda1,
                k=k,
                max_order=max_order,
                selected_transforms=selected_transforms,
            )
            for name, values in per_point.items():
                out.update(_finite_point_summary(np.asarray(values, dtype=np.float64), name))
            continue

        eigvals = _local_covariance_eigvals(
            sample,
            n_neighbors=k,
            n_jobs=-1,
            device=device,
        )
        if eigvals.size == 0:
            out.update({
                name: float("nan")
                for name in local_cov_spectrum_output_names(max_order, (k,), selected_transforms)
            })
            continue

        per_point: Dict[str, np.ndarray] = {
            f"local_cov_spectrum_k{k}_{branch}_{transform}": []
            for branch in LOCAL_COV_BRANCHES
            for transform in (
                *(name for name in LOCAL_COV_BASE_TRANSFORMS if name in selected_transforms),
                *(
                    f"invariant_l{order}"
                    for order in range(1, max_order + 1)
                    if "invariants" in selected_transforms
                ),
            )
        }

        branch_spectra = {
            "cov": eigvals,
            "cov2": np.square(eigvals),
        }
        for branch, spectra in branch_spectra.items():
            # RankMe is spectral entropy. For PSD matrices, singular values equal eigenvalues.
            if "rankme" in selected_transforms:
                per_point[f"local_cov_spectrum_k{k}_{branch}_rankme"] = _rankme_per_point(spectra)
            if "ne_sum" in selected_transforms:
                per_point[f"local_cov_spectrum_k{k}_{branch}_ne_sum"] = _ne_sum_per_point(spectra)
            if "participation_ratio" in selected_transforms:
                per_point[f"local_cov_spectrum_k{k}_{branch}_participation_ratio"] = (
                    _participation_ratio_per_point(spectra)
                )
            if "invariants" in selected_transforms:
                invariants = _elementary_symmetric_invariants_per_point(spectra, max_order)
                for order in range(1, max_order + 1):
                    per_point[f"local_cov_spectrum_k{k}_{branch}_invariant_l{order}"] = invariants[:, order - 1]

        for name, values in per_point.items():
            out.update(_finite_point_summary(np.asarray(values, dtype=np.float64), name))
    return {name: out.get(name, float("nan")) for name in output_names}
