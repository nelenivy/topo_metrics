"""
unsup_metrics.py
Thin wrapper around run_metrics.compute_metrics from the topology article.

Key adaptations vs original run_metrics.py:
  - Dropped: eval_downstream, evaluate_one_emb (MTEB API is our downstream)
  - Added:   adaptive sample-size floor (max(fraction*N, min_samples))
  - Added:   compute_metrics_retrieval() for 3 retrieval variants
  - Added:   ph_dim support behind --include-ph-dim flag (passed as kwarg)
  - Metrics dict keys: metric_<name> (same prefix as original, ready for CSV)
"""

from __future__ import annotations

import gc
import math
import logging
from time import perf_counter
from collections import defaultdict
from typing import Callable, List, Optional, Dict, Tuple

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import resample

logger = logging.getLogger(__name__)


def _profile_line(label: str, elapsed_s: float, /, **fields) -> None:
    extra = " | ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    if extra:
        logger.info("[profile] %s | %.3fs | %s", label, elapsed_s, extra)
    else:
        logger.info("[profile] %s | %.3fs", label, elapsed_s)


def _persistence_diagrams_point_cloud(
    embeddings: np.ndarray,
    maxdim: int = 1,
) -> List[List[Tuple[float, float]]]:
    """Point-cloud persistence diagrams (H0, …, H_maxdim) as finite (birth, death) pairs.

    ``maxdim=0`` computes **H0 only** (faster; no H1). ``maxdim=1`` is the usual H0+H1.

    Tries **ripserplusplus** first (same as original topology code); falls back to
    **ripser** (`pip install ripser`) if the fast binding is missing or errors.
    """
    try:
        import ripserplusplus as rpp

        diagrams = rpp.run(f"--format point-cloud --dim {maxdim}", embeddings)
        out: List[List[Tuple[float, float]]] = []
        # ripser++ may return list (per dim) or dict keyed by dimension
        if isinstance(diagrams, dict):
            dim_keys = sorted(diagrams.keys())
            seq = [diagrams[k] for k in dim_keys]
        else:
            seq = diagrams
        for dim in seq:
            pairs: List[Tuple[float, float]] = []
            for item in dim:
                row = np.asarray(item, dtype=np.float64).ravel()
                if row.size < 2:
                    continue
                b, d = float(row[0]), float(row[1])
                if np.isfinite(d) and d > b:
                    pairs.append((b, d))
            out.append(pairs)
        return out
    except Exception as e_pp:
        try:
            from ripser import ripser as ripser_fn

            res = ripser_fn(embeddings, maxdim=maxdim)
            dgms = res["dgms"]
        except Exception as e_py:
            raise ImportError(
                "Neither ripserplusplus nor ripser could compute persistence. "
                "Install one of: pip install ripserplusplus   OR   pip install ripser"
            ) from e_py

        out = []
        for dgm in dgms:
            if dgm is None or len(dgm) == 0:
                out.append([])
                continue
            pairs = []
            for row in np.asarray(dgm, dtype=np.float64):
                b, d = float(row[0]), float(row[1])
                if np.isfinite(d) and d > b:
                    pairs.append((b, d))
            out.append(pairs)
        logger.debug("Using ripser (Python) for persistence; ripserplusplus unavailable: %s", e_pp)
        return out


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _condensed_start_index(row: int, n_points: int) -> int:
    """Start offset for row ``row`` in SciPy/PyTorch condensed distance layout."""
    return n_points * row - (row * (row + 1)) // 2


def _auto_pairwise_batch_rows(n_points: int, x: torch.Tensor) -> int:
    """Pick a conservative row batch for GPU cdist fallback."""
    if x.device.type != "cuda":
        return max(1, min(n_points - 1, 1024))
    try:
        idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
        free_b, _ = torch.cuda.mem_get_info(idx)
    except Exception:
        return max(1, min(n_points - 1, 1024))

    # Keep the temporary cdist block well below available memory.
    elem_bytes = x.element_size()
    conservative_budget = int(free_b * 0.20)
    if conservative_budget <= 0:
        return 1
    row_batch = conservative_budget // max(n_points * elem_bytes, 1)
    return max(1, min(n_points - 1, row_batch, 2048))


def _batched_condensed_pairwise_distances_gpu(
    x: torch.Tensor, batch_rows: Optional[int] = None
) -> torch.Tensor:
    """
    Exact condensed pairwise Euclidean distances using batched GPU ``cdist``.

    This is a fallback for cases where ``torch.pdist`` OOMs.
    The result is returned as a CPU tensor so later selection / linkage can
    proceed without holding the temporary GPU workspace.
    """
    n_points = int(x.shape[0])
    if n_points < 2:
        return torch.empty(0, dtype=x.dtype, device="cpu")

    if batch_rows is None:
        batch_rows = _auto_pairwise_batch_rows(n_points, x)

    out = torch.empty(n_points * (n_points - 1) // 2, dtype=x.dtype, device="cpu")
    for start in range(0, n_points - 1, batch_rows):
        stop = min(n_points - 1, start + batch_rows)
        batch = x[start:stop]
        rest = x[start + 1 :]
        with torch.inference_mode():
            block = torch.cdist(batch, rest, p=2).cpu()

        for local_row, row in enumerate(range(start, stop)):
            row_len = n_points - row - 1
            offset = _condensed_start_index(row, n_points)
            out[offset : offset + row_len] = block[local_row, local_row : local_row + row_len]

    return out


def _condensed_pairwise_distances(
    embeddings: np.ndarray,
    *,
    prefer_gpu: bool = True,
    gpu_min_points: int = 1024,
    batch_rows: Optional[int] = None,
) -> torch.Tensor:
    """
    Return condensed pairwise Euclidean distances for ``embeddings``.

    Uses GPU ``torch.pdist`` when available and worthwhile; falls back to a
    batched GPU ``cdist`` path if ``pdist`` runs out of memory; otherwise uses
    SciPy ``pdist`` on CPU.
    """
    n_points = int(embeddings.shape[0])
    if n_points < 2:
        return torch.empty(0, dtype=torch.float32, device="cpu")

    if prefer_gpu and torch.cuda.is_available() and n_points >= gpu_min_points:
        device = torch.device("cuda")
        x = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
        try:
            with torch.inference_mode():
                return torch.pdist(x, p=2)
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            logger.warning("torch.pdist OOM on GPU; falling back to batched cdist")
            torch.cuda.empty_cache()
            return _batched_condensed_pairwise_distances_gpu(x, batch_rows=batch_rows)

    return torch.as_tensor(pdist(embeddings), dtype=torch.float32, device="cpu")


def _full_flat_quantile_from_condensed(
    condensed: torch.Tensor, n_points: int, quantile: float
) -> float:
    """
    Exact quantile of ``distance_matrix.ravel()`` without materializing squareform.

    The flattened symmetric matrix contains ``n_points`` zeros on the diagonal and
    every off-diagonal condensed distance twice.
    """
    n_points = int(n_points)
    if n_points <= 1:
        return 0.0
    if condensed.numel() == 0:
        return 0.0

    total = n_points * n_points
    q = float(min(1.0, max(0.0, quantile)))
    pos = q * (total - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    def _value_at_rank(rank: int) -> float:
        if rank < n_points:
            return 0.0
        condensed_rank = (rank - n_points) // 2
        condensed_rank = max(0, min(condensed_rank, int(condensed.numel()) - 1))
        return float(torch.kthvalue(condensed.reshape(-1), condensed_rank + 1).values.item())

    lo_val = _value_at_rank(lo)
    if hi == lo:
        return lo_val
    hi_val = _value_at_rank(hi)
    return lo_val + (pos - lo) * (hi_val - lo_val)


def _mst_total_weight_from_condensed(condensed: torch.Tensor | np.ndarray) -> float:
    """
    Exact H0 lifetime sum for a metric point cloud.

    For Vietoris-Rips H0, this equals the total weight of the minimum spanning
    tree. We compute it via single-linkage clustering on the condensed distances.
    """
    if isinstance(condensed, torch.Tensor):
        if condensed.numel() == 0:
            return 0.0
        condensed_np = np.asarray(condensed.detach().cpu().numpy(), dtype=np.float64)
    else:
        condensed_np = np.asarray(condensed, dtype=np.float64)
    if condensed_np.size == 0:
        return 0.0
    z = linkage(condensed_np, method="single", optimal_ordering=False)
    return float(np.asarray(z[:, 2], dtype=np.float64).sum())


def _import_graph_metrics():
    """Import graph / spectral metrics (Google Research `metrics.py` API).

    If ``GRAPH_METRICS_PATH`` is set, load ``metrics`` from that directory;
    otherwise use the vendored ``src.metrics`` shipped with this repo.
    """
    import os
    import sys

    extra_path = os.environ.get("GRAPH_METRICS_PATH", "").strip()
    if extra_path:
        if extra_path not in sys.path:
            sys.path.insert(0, extra_path)
        from metrics import (
            rankme, coherence, pseudo_condition_number, alpha_req,
            stable_rank, ne_sum, self_clustering,
        )
    else:
        from src.metrics import (
            rankme, coherence, pseudo_condition_number, alpha_req,
            stable_rank, ne_sum, self_clustering,
        )
    return rankme, coherence, pseudo_condition_number, alpha_req, stable_rank, ne_sum, self_clustering


def _available_metric_functions(
    include_ph_dim: bool = False,
    ripser_maxdim: int = 1,
) -> Dict[str, Callable]:
    """Return the ordered metric registry used by ``compute_metrics``."""
    (
        rankme,
        coherence,
        pseudo_condition_number,
        alpha_req,
        stable_rank,
        ne_sum,
        self_clustering,
    ) = _import_graph_metrics()

    def _ripser_wrapped(sample, u=None, s=None):
        return ripser_metric(sample, u=u, s=s, maxdim=ripser_maxdim)

    metrics: Dict[str, Callable] = {
        "rankme": rankme,
        "coherence": coherence,
        "pseudo_condition_number": pseudo_condition_number,
        "alpha_req": alpha_req,
        "stable_rank": stable_rank,
        "ne_sum": ne_sum,
        "self_clustering": self_clustering,
        "ripser": _ripser_wrapped,
    }
    if include_ph_dim:
        metrics["ph_dim"] = _ph_dim_metric
    return metrics


def _ripser_output_names(maxdim: int) -> List[str]:
    """Flattened metric keys produced by ``ripser_metric`` for one ``maxdim``."""
    if maxdim == 0:
        return [
            "ripser_sum_H0",
            "ripser_q90_flat",
            "ripser_sum_H0_norm0.9",
        ]

    quants = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    quants_labels = quants + ["mean_10", "mean_last_10"]
    out: List[str] = []
    for k in range(maxdim + 1):
        out.extend(
            [
                f"ripser_sum_H{k}",
                f"ripser_log_sum{k}",
                f"ripser_norm_sum{k}",
                f"ripser_log_sum_norm{k}",
                f"ripser_sq_sum_H{k}",
            ]
        )
        for q in quants_labels:
            out.extend(
                [
                    f"ripser_sum_H{k}_norm{q}",
                    f"ripser_sq_sum_H{k}_norm{q}",
                    f"ripser_log_sum{k}_norm{q}",
                ]
            )
    return out


def metric_output_map(
    selected_metrics: Optional[List[str]] = None,
    include_ph_dim: bool = False,
    ripser_maxdim: int = 1,
) -> Dict[str, List[str]]:
    """Map each selected metric name to the flattened output keys it produces.

    This mirrors the metric columns that end up in ``master_results.csv`` and is
    used by the evaluation script to decide whether a cached row is stale.
    """
    available = _available_metric_functions(
        include_ph_dim=include_ph_dim, ripser_maxdim=ripser_maxdim
    )
    chosen = (
        list(available.keys())
        if selected_metrics is None
        else [m for m in selected_metrics if m in available]
    )
    out: Dict[str, List[str]] = {}
    for metric_name in chosen:
        if metric_name == "ripser":
            out[metric_name] = _ripser_output_names(ripser_maxdim)
        else:
            out[metric_name] = [metric_name]
    return out


def expected_metric_columns(
    selected_metrics: Optional[List[str]] = None,
    include_ph_dim: bool = False,
    ripser_maxdim: int = 1,
    retrieval: bool = False,
) -> List[str]:
    """Return the CSV columns that ``compute_metrics`` would emit."""
    cols: List[str] = []
    for output_names in metric_output_map(
        selected_metrics=selected_metrics,
        include_ph_dim=include_ph_dim,
        ripser_maxdim=ripser_maxdim,
    ).values():
        cols.extend([f"metric_{name}" for name in output_names])
        cols.extend([f"std_{name}" for name in output_names])
    if not retrieval:
        return cols
    suffixed: List[str] = []
    for variant in ("corpus", "queries", "combined"):
        suffixed.extend(f"{col}_{variant}" for col in cols)
    return suffixed


def ripser_metric(
    embeddings: np.ndarray, u=None, s=None, maxdim: int = 1
) -> Dict[str, float]:
    """Compute topology-flavored metrics for a sampled embedding matrix.

    ``maxdim == 0`` uses a fast exact H0 path: GPU/CPU condensed pairwise
    distances, one full-flatten 0.9 quantile, and an exact MST weight via
    single-linkage clustering.
    """
    if maxdim == 0:
        # Fast H0-only path: exact MST weight + a single 0.9 quantile over the
        # flattened symmetric distance matrix, without materializing squareform.
        condensed = _condensed_pairwise_distances(embeddings)
        n_points = int(embeddings.shape[0])
        q90 = _full_flat_quantile_from_condensed(condensed, n_points, 0.9)
        condensed_np = condensed.detach().cpu().numpy() if isinstance(condensed, torch.Tensor) else condensed
        del condensed
        mst_total = _mst_total_weight_from_condensed(condensed_np)
        return {
            "ripser_sum_H0": mst_total,
            "ripser_q90_flat": q90,
            "ripser_sum_H0_norm0.9": mst_total / (q90 + 1e-10),
        }

    diagrams = _persistence_diagrams_point_cloud(embeddings, maxdim=maxdim)

    distances = pdist(embeddings)
    distance_matrix = squareform(distances)
    sorted_rows = np.sort(distance_matrix, axis=1)
    mean_nearest_dist = sorted_rows[:, min(10, sorted_rows.shape[1] - 1)].mean()
    mean_largest_dist = sorted_rows[:, -min(10, sorted_rows.shape[1])].mean()
    distances_arr = distance_matrix.ravel()
    quants = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    norms = list(np.quantile(distances_arr, quants)) + [mean_nearest_dist, mean_largest_dist]
    quants_labels = quants + ['mean_10', 'mean_last_10']

    persistence: Dict[str, float] = {}
    for k in range(len(diagrams)):
        pers_lens = [death - birth for birth, death in diagrams[k] if death > birth]
        persistence[f"ripser_sum_H{k}"] = sum(pers_lens)
        persistence_sq_sum = sum(l ** 2 for l in pers_lens)
        persistence[f"ripser_log_sum{k}"] = sum(math.log(1.0 + l) for l in pers_lens)
        persistence[f"ripser_norm_sum{k}"] = sum(
            (death - birth) / (death + birth)
            for birth, death in diagrams[k] if death > birth
        )
        persistence[f"ripser_log_sum_norm{k}"] = sum(
            math.log(1.0 + (death - birth) / (death + birth))
            for birth, death in diagrams[k] if death > birth
        )
        persistence[f"ripser_sq_sum_H{k}"] = math.sqrt(persistence_sq_sum)
        for q, v in zip(quants_labels, norms):
            persistence[f"ripser_sum_H{k}_norm{q}"] = persistence[f"ripser_sum_H{k}"] / (v + 1e-10)
            persistence[f"ripser_sq_sum_H{k}_norm{q}"] = persistence[f"ripser_sq_sum_H{k}"] / (v + 1e-10)
            persistence[f"ripser_log_sum{k}_norm{q}"] = persistence[f"ripser_log_sum{k}"] / (math.log(1.0 + v + 1e-10))

    return persistence


def _ph_dim_metric(embeddings: np.ndarray, u=None, s=None) -> float:
    from topology import calculate_ph_dim
    return calculate_ph_dim(embeddings)


def compute_metrics(
    embeddings_np: np.ndarray,
    selected_metrics: Optional[List[str]] = None,
    n_samples: int = 1,
    sample_fraction: float = 1 / 20,
    min_sample_size: int = 100,
    include_ph_dim: bool = False,
    ripser_maxdim: int = 1,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute unsupervised quality metrics on a 2-D embedding matrix.

    Parameters
    ----------
    embeddings_np : (N, d) float32 array
    selected_metrics : list of metric names to compute (None = all)
    n_samples : number of random subsamples to average (default 1 for speed)
    sample_fraction : fraction of N to use per subsample
    min_sample_size : lower bound on subsample size (adaptive floor)
    include_ph_dim : also compute persistent-homology dimension (slow)
    ripser_maxdim : max homology dimension for Ripser (0 = H0 only, faster; 1 = H0+H1)
    verbose : print per-metric values

    Returns
    -------
    dict with keys "metric_<name>" (mean) and "std_<name>" (relative std)
    """
    N = embeddings_np.shape[0]
    sample_size = max(int(sample_fraction * N), min_sample_size)
    sample_size = min(sample_size, N)

    available_metrics = _available_metric_functions(
        include_ph_dim=include_ph_dim, ripser_maxdim=ripser_maxdim
    )

    if selected_metrics is None:
        selected_metrics = list(available_metrics.keys())

    started_total = perf_counter()
    metrics: Dict = {}
    metric_timings: Dict[str, float] = defaultdict(float)
    metric_call_count: Dict[str, int] = defaultdict(int)
    metric_output_names: Dict[str, List[str]] = {}
    for i in range(n_samples):
        started_sample = perf_counter()
        sample = resample(embeddings_np, n_samples=sample_size,
                          replace=False, random_state=42 + i)
        u, s, _ = np.linalg.svd(sample, compute_uv=True, full_matrices=False)

        for metric_name in selected_metrics:
            if metric_name not in available_metrics:
                continue
            try:
                started_metric = perf_counter()
                result = available_metrics[metric_name](sample, u=u, s=s)
                metric_elapsed = perf_counter() - started_metric
                metric_timings[metric_name] += metric_elapsed
                metric_call_count[metric_name] += 1
                if isinstance(result, dict):
                    metric_output_names.setdefault(metric_name, list(result.keys()))
                    for subname, val in result.items():
                        metrics.setdefault(subname, []).append(float(val))
                else:
                    metric_output_names.setdefault(metric_name, [metric_name])
                    metrics.setdefault(metric_name, []).append(float(result))
            except Exception as e:
                logger.warning(f"Failed to compute {metric_name} on sample {i}: {e}")

        gc.collect()
        _profile_line(
            "unsup_compute_sample",
            perf_counter() - started_sample,
            sample=i,
            sample_size=sample_size,
            metrics=len([m for m in selected_metrics if m in available_metrics]),
        )

    averaged = {f"metric_{k}": float(np.mean(v)) for k, v in metrics.items()}
    std_rel  = {f"std_{k}":    float(np.std(v) / (np.mean(v) + 1e-10))
                for k, v in metrics.items()}

    for metric_name in selected_metrics:
        if metric_name not in available_metrics:
            continue
        outs = metric_output_names.get(metric_name, [metric_name])
        _profile_line(
            "unsup_metric",
            metric_timings.get(metric_name, 0.0),
            metric=metric_name,
            calls=metric_call_count.get(metric_name, 0),
            outputs=len(outs),
            output_names=",".join(outs[:12]) + ("..." if len(outs) > 12 else ""),
        )

    _profile_line(
        "unsup_compute_total",
        perf_counter() - started_total,
        samples=n_samples,
        sample_size=sample_size,
        metrics=len([m for m in selected_metrics if m in available_metrics]),
    )

    if verbose:
        for k, v in averaged.items():
            logger.info(f"  {k:45s} = {v:.4f}")

    return {**averaged, **std_rel}


def compute_metrics_retrieval(
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    max_combined_size: int = 5000,
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    """
    For retrieval tasks produce three metric dicts:
      "corpus"   – metrics on corpus embeddings only
      "queries"  – metrics on query embeddings only
      "combined" – metrics on queries + corpus sample (up to max_combined_size)

    Returns dict of {variant_name: metrics_dict}.
    The caller suffixes column names with _{variant_name}.
    """
    started_total = perf_counter()
    results = {}
    started = perf_counter()
    results["corpus"] = compute_metrics(corpus_embs, **kwargs)
    _profile_line("unsup_retrieval_variant", perf_counter() - started, variant="corpus", rows=len(corpus_embs))
    started = perf_counter()
    results["queries"] = compute_metrics(query_embs, **kwargs)
    _profile_line("unsup_retrieval_variant", perf_counter() - started, variant="queries", rows=len(query_embs))

    n_q = len(query_embs)
    n_c = len(corpus_embs)
    budget = max_combined_size - n_q
    if budget > 0 and n_c > 0:
        if n_c <= budget:
            corpus_sample = corpus_embs
        else:
            rng = np.random.default_rng(0)
            idx = rng.choice(n_c, size=budget, replace=False)
            corpus_sample = corpus_embs[idx]
        combined = np.concatenate([query_embs, corpus_sample], axis=0)
    else:
        combined = query_embs

    started = perf_counter()
    results["combined"] = compute_metrics(combined, **kwargs)
    _profile_line("unsup_retrieval_variant", perf_counter() - started, variant="combined", rows=len(combined))
    _profile_line(
        "unsup_retrieval_total",
        perf_counter() - started_total,
        corpus_rows=len(corpus_embs),
        query_rows=len(query_embs),
        combined_rows=len(combined),
    )
    return results


def suffix_metrics(metrics: Dict[str, float], suffix: str) -> Dict[str, float]:
    """Rename 'metric_X' -> 'metric_X_<suffix>' (same for std_)."""
    out = {}
    for k, v in metrics.items():
        if k.startswith("metric_") or k.startswith("std_"):
            prefix, rest = k.split("_", 1)
            out[f"{prefix}_{rest}_{suffix}"] = v
        else:
            out[f"{k}_{suffix}"] = v
    return out
