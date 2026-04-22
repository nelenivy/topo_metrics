"""
Oracle-gap pairwise metrics (paper: soft empirical fibers, Algorithms 2–3).

Pure NumPy/SciPy. Intended for row-aligned embeddings U, V from two encoders
on the same texts. Consensus graph modes (Algorithm 4) live here as well.

Changes vs original:
- Per-row adaptive bandwidth selected by per-row LOO-CV with per-row geometric grid
  (bounds: (10% quantile of local 1-NN distances) / 3 … 0.9-quantile of that row's kNN distances).
- Primary kernel: unnormalized Gaussian exp(-(dist/sigma)^2 / 2) with per-row sigma,
  hard-zeroed for dist > cutoff_i = max(knn_k-th neighbor dist, 3 * sigma_i).
  This guarantees all k kNN entries are nonzero while suppressing very distant ghosts.
- Guards against pathological per-row CV cases: flat curves, single-neighbor rows,
  noisy isolated selections (median-smoothed over kNN graph).
- ``build_sparse_T_adaptive`` replaces ``build_sparse_T_from_knn_csr`` for the
  main pipeline; optional **directed** density rescale
  ``W_ij <- W_ij / (q_out[i]^α q_in[j]^α)`` (default ``α=1``) before row-normalization,
  with ``q_out`` / ``q_in`` row- and column-marginals of the nonnegative kernel.
- The legacy scalar-eps ``build_sparse_T_from_knn_csr`` remains for backward compatibility.
- diagnostics updated: eps_hat_mean/median/std replace single eps_hat scalar;
  eps_T_over_eps_hat removed (no longer meaningful); new fields added.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_og_log = logging.getLogger(__name__)

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import (
        ArpackError,
        ArpackNoConvergence,
        LinearOperator,
        eigsh,
        lobpcg,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError("oracle_gap_pairwise requires scipy") from e


_BLAS_THREAD_ENV_KEYS: Tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _default_principal_blas_threads() -> int:
    """Reasonable cap for eigsh matvec so many parallel workers do not oversubscribe."""
    c = os.cpu_count() or 8
    return int(min(32, max(1, c)))


@contextmanager
def _blas_thread_env(n_threads: Optional[int]) -> Iterator[None]:
    """
    Temporarily pin BLAS/OpenMP thread counts for the SciPy ARPACK matvec path.

    ``n_threads is None`` uses ``_default_principal_blas_threads()``.
    ``n_threads <= 0`` skips overriding the environment (caller-managed).
    """
    if n_threads is not None and n_threads <= 0:
        yield
        return
    nt = int(n_threads) if n_threads is not None else _default_principal_blas_threads()
    nt = max(1, nt)
    s = str(nt)
    saved: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in _BLAS_THREAD_ENV_KEYS}
    try:
        for k in _BLAS_THREAD_ENV_KEYS:
            os.environ[k] = s
        yield
    finally:
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def _principal_cuda_device_index(principal_device: Optional[str]) -> Optional[int]:
    """
    Map ``principal_device`` to a CuPy device index, or None for CPU SciPy path.

    Accepts None / "" / "cpu" → None; "cuda" / "cuda:0" → 0; "cuda:1" → 1; case-insensitive.
    """
    if principal_device is None:
        return None
    s = str(principal_device).strip().lower()
    if not s or s == "cpu":
        return None
    if s == "cuda":
        return 0
    if s.startswith("cuda:"):
        tail = s.split(":", 1)[1].strip()
        if not tail:
            return 0
        try:
            return int(tail)
        except ValueError:
            _og_log.warning("Ignoring invalid principal_device %r; using CPU", principal_device)
            return None
    _og_log.warning("Unknown principal_device %r; using CPU SciPy path", principal_device)
    return None


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def epanechnikov(u: np.ndarray) -> np.ndarray:
    """k(u) = (1-u^2)_+ compact on [0,1], max=1."""
    u = np.asarray(u, dtype=np.float64)
    return np.where(u <= 1.0, np.maximum(0.0, 1.0 - u * u), 0.0)


def _gaussian_weights(dist: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Unnormalized Gaussian exp(-(dist/sigma)^2 / 2).

    ``dist`` and ``sigma`` must be broadcastable. Exponent capped at 750 to
    avoid exp(-inf) underflow silently zeroing entire rows.
    """
    dist = np.asarray(dist, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-12)
    x = dist / sigma
    quad = 0.5 * np.minimum(x * x, 750.0)
    return np.exp(-quad)


def _fiber_neighbor_weights(dist: np.ndarray, eps: float, *, kernel: str) -> np.ndarray:
    """
    Legacy scalar-eps entry point kept for backward compatibility.
    ``kernel='gaussian'``: exp(-(dist/eps)^2/2).
    ``kernel='epanechnikov'``: (1-(dist/eps)^2)_+.
    """
    dist = np.asarray(dist, dtype=np.float64)
    sigma = max(float(eps), 1e-12)
    if kernel == "gaussian":
        return _gaussian_weights(dist, np.full_like(dist, sigma))
    if kernel == "epanechnikov":
        return epanechnikov(dist / sigma)
    raise ValueError(f"unknown fiber kernel {kernel!r} (use 'gaussian' or 'epanechnikov')")


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------

def pairwise_sq_euclidean(X: np.ndarray) -> np.ndarray:
    """(n,n) squared Euclidean distances for rows of X (float64)."""
    X = np.asarray(X, dtype=np.float64)
    g = X @ X.T
    d = np.diag(g)
    return np.maximum(0.0, d[:, None] + d[None, :] - 2.0 * g)


def knn_from_sq_dist(sq_dist: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (knn_sq, knn_idx): per row, squared distances and column indices of the k
    nearest neighbors, excluding self, sorted ascending by distance.
    """
    n = int(sq_dist.shape[0])
    k = min(max(1, k), n - 1)
    d = np.asarray(sq_dist, dtype=np.float64).copy()
    np.fill_diagonal(d, np.inf)
    idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    knn_sq = d[rows, idx]
    order = np.argsort(knn_sq, axis=1)
    r = np.arange(n)[:, None]
    knn_sq = knn_sq[r, order]
    idx = idx[r, order]
    return knn_sq, idx


# ---------------------------------------------------------------------------
# Per-row adaptive bandwidth selection
# ---------------------------------------------------------------------------


def bandwidth_cv_grid_adaptive(
    U: np.ndarray,
    knn_idx: np.ndarray,
    knn_sq: np.ndarray,
    *,
    M: int = 24,
    sigma_clip: float = 3.0,
    smooth_bw: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-row LOO-CV bandwidth selection with per-row geometric grid.

    For each point i a log-spaced grid of M bandwidths is built between
    eps_min_i = (10% quantile of {1-NN distance at i and at each of i's k nearest neighbors}) / 3
    (1-NN = first column of sorted kNN distances), and
    eps_max_i = the 0.9 quantile of that row's kNN distances.

    Kernel: unnormalized Gaussian hard-zeroed at cutoff = max(knn_k-th dist, sigma_clip * sigma).
    This guarantees that for each candidate sigma all k retained kNN entries are nonzero
    and entries beyond 3-sigma are suppressed.

    Pathological-case guards
    ------------------------
    1. Flat CV curve (all eps give same loss within rtol=1e-4): use eps_max_i.
    2. Single usable neighbor after LOO zero-out: floor to eps_min_i (no selection possible).
    3. Noisy isolated selection: if smooth_bw=True, replace each eps_hat_i with the
       median of eps_hat values over the row's kNN (including itself).  This is the
       Abramson/Sain-Scott neighborhood-smoothing trick.

    Returns
    -------
    eps_hat_rows : (n,) per-point selected bandwidths (possibly smoothed)
    eps_grids    : (n, M) per-point candidate grids
    cv_scores    : (n, M) per-point LOO losses
    """
    n, k = knn_idx.shape
    U = np.asarray(U, dtype=np.float64)
    knn_dist = np.sqrt(np.maximum(0.0, knn_sq.astype(np.float64)))  # (n, k)

    M = max(2, int(M))

    # Per-row grid lower bound: (10% quantile of 1-NN distances over {i} ∪ kNN(i)) / 3.
    d1 = knn_dist[:, 0]  # (n,) nearest-neighbor distance (knn sorted ascending)
    d1_stack = np.concatenate([d1[:, None], d1[knn_idx]], axis=1)  # (n, k + 1)
    eps_min_rows = np.maximum(np.quantile(d1_stack, 0.1, axis=1) / 3.0, 1e-8)  # (n,)
    q90 = np.quantile(knn_dist, 0.9, axis=1).astype(np.float64)           # (n,)
    eps_max_rows = np.maximum(q90, eps_min_rows * 1.0001)

    # Build (n, M) log-spaced grids
    t = np.linspace(0.0, 1.0, M)                       # (M,)
    log_grids = (
        np.log(eps_min_rows)[:, None]
        + t[None, :] * (np.log(eps_max_rows) - np.log(eps_min_rows))[:, None]
    )                                                   # (n, M)
    eps_grids = np.exp(log_grids)                      # (n, M)

    # kth-NN distance per row (distance to the farthest retained neighbor)
    kth_dist = knn_dist[:, k - 1]                     # (n,)

    cv_scores = np.full((n, M), np.inf, dtype=np.float64)

    for mi in range(M):
        sigma_i = eps_grids[:, mi]                     # (n,) per-row bandwidth

        # Hard cutoff: zero weights beyond max(kth_dist_i, sigma_clip * sigma_i)
        cutoff_i = np.maximum(kth_dist, sigma_clip * sigma_i)  # (n,)

        # Gaussian weights (n, k), vectorised
        # w[i,j] = exp(-(knn_dist[i,j] / sigma_i[i])^2 / 2)  if dist <= cutoff_i[i]
        w = _gaussian_weights(knn_dist, sigma_i[:, None])   # (n, k)
        beyond_cutoff = knn_dist > cutoff_i[:, None]
        w[beyond_cutoff] = 0.0

        # LOO: zero out self-edges (knn_idx[i,:] == i should not occur since knn
        # excludes self, but guard anyway)
        self_mask = knn_idx == np.arange(n)[:, None]
        w[self_mask] = 0.0

        row_sums = w.sum(axis=1, keepdims=True)         # (n, 1)

        # Guard 2: rows with no usable neighbor → mark inf, will be handled below
        valid = row_sums.ravel() > 1e-12
        row_sums_safe = np.where(row_sums > 1e-12, row_sums, 1.0)
        T_row = w / row_sums_safe                       # (n, k) row-stochastic where valid

        # LOO prediction: pred[i] = sum_j T_row[i,j] * U[knn_idx[i,j]]
        pred = np.einsum("ij,ijk->ik", T_row, U[knn_idx])  # (n, d_U)

        diff_sq = np.sum((U - pred) ** 2, axis=1)      # (n,)
        diff_sq[~valid] = np.inf                        # invalid rows stay inf
        cv_scores[:, mi] = diff_sq

    # --------------- select best eps per row ---------------
    # Guard 2: rows where ALL grid entries are inf → use eps_max_rows
    all_inf = np.all(~np.isfinite(cv_scores), axis=1)

    best_idx = np.argmin(cv_scores, axis=1)            # (n,) argmin ignores inf naturally
    eps_hat_rows = eps_grids[np.arange(n), best_idx]  # (n,)

    # Guard 1: flat CV curve (relative range < rtol) → use eps_max_rows
    cv_finite = np.where(np.isfinite(cv_scores), cv_scores, np.nan)
    cv_row_min = np.nanmin(cv_finite, axis=1)
    cv_row_max = np.nanmax(cv_finite, axis=1)
    flat = (cv_row_max - cv_row_min) < 1e-4 * (np.abs(cv_row_min) + 1e-12)
    eps_hat_rows = np.where(flat | all_inf, eps_max_rows, eps_hat_rows)

    # Guard 3: smooth selected bandwidths over kNN graph (Abramson / Sain-Scott)
    if smooth_bw:
        # For each row i, take median of eps_hat over {i} union knn_idx[i, :]
        smoothed = np.empty(n, dtype=np.float64)
        for i in range(n):
            nbrs = knn_idx[i]                          # (k,)
            vals = np.concatenate([[eps_hat_rows[i]], eps_hat_rows[nbrs]])
            smoothed[i] = float(np.median(vals))
        eps_hat_rows = smoothed

    return eps_hat_rows, eps_grids, cv_scores


# ---------------------------------------------------------------------------
# Sparse T construction (adaptive per-row eps)
# ---------------------------------------------------------------------------

def _csr_directed_density_rescale(mat: csr_matrix, *, alpha: float, flo: float = 1e-12) -> csr_matrix:
    """
    Directed analogue of α-rescaling: W_ij <- W_ij / (q_out[i]^α · q_in[j]^α),
    with q_out = row sums and q_in = column sums of the current nonnegative matrix.
    """
    if abs(float(alpha)) < 1e-15:
        return mat
    mat = mat.tocsr()
    n = int(mat.shape[0])
    q_out = np.asarray(mat.sum(axis=1), dtype=np.float64).ravel()
    q_in = np.asarray(mat.sum(axis=0), dtype=np.float64).ravel()
    q_out = np.maximum(q_out, flo)
    q_in = np.maximum(q_in, flo)
    coo = mat.tocoo()
    scale = 1.0 / (q_out[coo.row] ** float(alpha) * q_in[coo.col] ** float(alpha))
    out = csr_matrix((coo.data * scale, (coo.row, coo.col)), shape=(n, n), dtype=np.float64)
    out.eliminate_zeros()
    return out


def build_sparse_T_adaptive(
    knn_idx: np.ndarray,
    knn_sq: np.ndarray,
    eps_rows: np.ndarray,
    *,
    loo: bool,
    sigma_clip: float = 3.0,
    density_normalize: bool = True,
    density_alpha: float = 1.0,
) -> csr_matrix:
    """
    Row-stochastic sparse T with per-row Gaussian kernel and hard cutoff.

    Kernel for row i, neighbor j:
        w_ij = exp(-(dist_ij / sigma_i)^2 / 2)   if dist_ij <= max(kth_dist_i, sigma_clip * sigma_i)
             = 0                                   otherwise

    After optional LOO zero-out of the diagonal, optionally applies directed
    density rescale (see ``_csr_directed_density_rescale``), then rows are
    normalized to sum 1.

    For non-LOO mode a self-loop (weight = 1.0 = k(0)) is added before the
    rescale / row-normalization, consistent with Algorithm 1 / paper Section 5.

    Parameters
    ----------
    eps_rows : (n,) per-row bandwidth (sigma).
    loo      : if True zero out self-edges and renormalize (for CV).
    density_normalize : if True (default), divide each nonzero by
        ``q_out[i]**density_alpha * q_in[j]**density_alpha`` before final row-normalization.
    density_alpha : exponent α (default 1.0); 0 disables rescaling even if
        ``density_normalize`` is True.
    """
    n, k = knn_idx.shape
    knn_dist = np.sqrt(np.maximum(0.0, knn_sq.astype(np.float64)))  # (n, k)
    eps_rows = np.asarray(eps_rows, dtype=np.float64).ravel()
    assert eps_rows.shape[0] == n

    kth_dist = knn_dist[:, k - 1]                         # (n,)
    cutoff = np.maximum(kth_dist, sigma_clip * eps_rows)   # (n,)

    w = _gaussian_weights(knn_dist, eps_rows[:, None])     # (n, k)
    beyond = knn_dist > cutoff[:, None]
    w[beyond] = 0.0

    if loo:
        self_mask = knn_idx == np.arange(n)[:, None]
        w[self_mask] = 0.0

    rows_rep = np.repeat(np.arange(n, dtype=np.int32), k)
    cols_rep = knn_idx.astype(np.int32).ravel()
    mat = csr_matrix((w.ravel(), (rows_rep, cols_rep)), shape=(n, n))
    mat.eliminate_zeros()
    mat = mat.tocsr()

    if not loo:
        # Self-loop weight = k(0) = 1.0 (cancels in row-normalization but keeps
        # the self-prediction contribution consistent with the paper)
        mat = mat + csr_matrix(
            (np.ones(n, dtype=np.float64), (np.arange(n), np.arange(n))),
            shape=(n, n),
        )

    if density_normalize:
        mat = _csr_directed_density_rescale(mat, alpha=float(density_alpha))

    sums = np.array(mat.sum(axis=1)).ravel()
    sums = np.maximum(sums, 1e-12)
    mat = mat.multiply((1.0 / sums)[:, None]).tocsr()
    mat.eliminate_zeros()
    return mat


# ---------------------------------------------------------------------------
# Legacy scalar-eps T builder (kept for backward compatibility)
# ---------------------------------------------------------------------------

def build_sparse_T_from_knn_csr(
    knn_idx: np.ndarray,
    knn_sq: np.ndarray,
    eps: float,
    d_V: int,
    *,
    loo: bool,
    kernel: str = "gaussian",
) -> csr_matrix:
    """
    Original scalar-bandwidth T builder (kept for backward compatibility).
    ``d_V`` is retained for API compatibility only; it does not affect weights.
    """
    n, k = knn_idx.shape
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = knn_idx.astype(np.int32).ravel()
    dist = np.sqrt(np.maximum(0.0, knn_sq.astype(np.float64)))
    w = _fiber_neighbor_weights(dist, float(eps), kernel=kernel)
    if loo:
        mask = (knn_idx != np.arange(n)[:, None]).ravel()
        w = np.where(mask, w.ravel(), 0.0)
    else:
        w = w.ravel()
    mat = csr_matrix((w, (rows, cols)), shape=(n, n))
    mat.eliminate_zeros()
    mat = mat.tocsr()
    if not loo:
        mat = mat + csr_matrix(
            (np.ones(n, dtype=np.float64), (np.arange(n), np.arange(n))),
            shape=(n, n),
        )
    sums = np.array(mat.sum(axis=1)).ravel()
    sums = np.maximum(sums, 1e-12)
    mat = mat.multiply((1.0 / sums)[:, None]).tocsr()
    mat.eliminate_zeros()
    return mat


# ---------------------------------------------------------------------------
# T diagnostics
# ---------------------------------------------------------------------------

def t_matrix_row_nnz_stats(T: csr_matrix) -> Dict[str, float]:
    """Nonzeros per row of T (after construction)."""
    if not isinstance(T, csr_matrix):
        T = T.tocsr()
    n = int(T.shape[0])
    if n == 0:
        return {"T_row_nnz_mean": 0.0, "T_row_nnz_min": 0.0, "T_row_nnz_max": 0.0}
    nnz = np.diff(T.indptr).astype(np.int64)
    return {
        "T_row_nnz_mean": float(nnz.mean()),
        "T_row_nnz_min": float(nnz.min()),
        "T_row_nnz_max": float(nnz.max()),
    }


# ---------------------------------------------------------------------------
# Consensus graph
# ---------------------------------------------------------------------------

def consensus_affinity_matrix(embeddings_list: Sequence[np.ndarray]) -> np.ndarray:
    """
    W_ij = exp(-(1/L) * sum_l (dist_ij^(l) / median_l)^2).
    Diagonal set to 0.
    """
    if len(embeddings_list) == 0:
        raise ValueError("embeddings_list is empty")
    n = int(embeddings_list[0].shape[0])
    acc = np.zeros((n, n), dtype=np.float64)
    for X in embeddings_list:
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] != n:
            raise ValueError("all embeddings must have the same number of rows")
        sq = pairwise_sq_euclidean(X)
        dist = np.sqrt(np.maximum(0.0, sq))
        tri = dist[np.triu_indices(n, k=1)]
        med = float(np.median(tri)) if tri.size else 1.0
        med = max(med, 1e-12)
        acc += (dist / med) ** 2
    L = len(embeddings_list)
    W = np.exp(-(1.0 / L) * acc)
    np.fill_diagonal(W, 0.0)
    return W


def consensus_summary_from_W(W: np.ndarray) -> Dict[str, float]:
    """Scalar diagnostics from a consensus affinity matrix W."""
    deg = W.sum(axis=1).astype(np.float64)
    return {
        "consensus_W_sum": float(W.sum()),
        "consensus_deg_min": float(deg.min()),
        "consensus_deg_max": float(deg.max()),
        "consensus_deg_mean": float(deg.mean()),
        "consensus_n_edges_pos": float(np.mean(deg > 0)),
    }


def consensus_graph_summary(embeddings_list: Sequence[np.ndarray]) -> Dict[str, float]:
    """Scalar diagnostics (builds W once)."""
    return consensus_summary_from_W(consensus_affinity_matrix(embeddings_list))


def consensus_graph_modes_from_W(W: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nontrivial low-frequency modes of L_sym = I - D^{-1/2} W D^{-1/2}.
    Equivalent to the generalized eigenproblem L_W psi = lambda D psi.
    Returns (psi, lambda) for modes 1..r (skipping the trivial zero mode).
    Sign convention: each mode flipped so its sum is >= 0.
    """
    n = int(W.shape[0])
    deg = W.sum(axis=1).astype(np.float64)
    deg = np.maximum(deg, 1e-10)
    inv_sqrt = 1.0 / np.sqrt(deg)
    L_sym = np.eye(n, dtype=np.float64)
    L_sym -= (inv_sqrt[:, None] * W) * inv_sqrt[None, :]
    r = int(min(max(1, r), n - 1))
    evals, evecs = np.linalg.eigh(L_sym)
    y = evecs[:, 1 : r + 1].astype(np.float64)
    lam = evals[1 : r + 1].astype(np.float64)
    psi = y * inv_sqrt[:, None]
    for j in range(psi.shape[1]):
        if psi[:, j].sum() < 0.0:
            psi[:, j] *= -1.0
    return psi, lam


def consensus_graph_modes(
    embeddings_list: Sequence[np.ndarray], r: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Same as consensus_graph_modes_from_W but builds W from embeddings."""
    W = consensus_affinity_matrix(embeddings_list)
    return consensus_graph_modes_from_W(W, r)


# ---------------------------------------------------------------------------
# Oracle gap estimators (Algorithms 2 & 3)
# ---------------------------------------------------------------------------

def oracle_gap_rank_one(
    T: csr_matrix, phi: np.ndarray, *, a: float = 1.0
) -> Tuple[float, np.ndarray]:
    """
    Algorithm 2: global Q_hat and local ell_i for one mode phi (length n).

    ell_i = (|a|/2) * (T|phi|_i  -  |T phi_i|)
    Q_hat = mean(ell)
    """
    phi = np.asarray(phi, dtype=np.float64).ravel()
    Tphi = T @ phi
    Tabs_phi = T @ np.abs(phi)
    ell = 0.5 * float(np.abs(a)) * (Tabs_phi - np.abs(Tphi))
    Q = float(np.mean(ell))
    return Q, ell


def _L_matvec(T: csr_matrix, x: np.ndarray) -> np.ndarray:
    """Matrix-vector product for L = I - T^T T (or batched columns)."""
    if x.ndim == 1:
        return x - (T.T @ (T @ x))
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        col = x[:, j]
        out[:, j] = col - (T.T @ (T @ col))
    return out


def _topk_eigen_symmetric_lm(
    Lop: LinearOperator,
    n: int,
    k: int,
    *,
    maxiter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Largest-magnitude eigenpairs of symmetric PSD Lop.
    ARPACK (eigsh) with progressive retries, LOBPCG fallback.
    """
    if k <= 0 or k >= n:
        raise ValueError(f"need 0 < k < n, got k={k}, n={n}")
    ncv_base = int(max(4 * k + 8, k + 40, min(n - 1, 2 * k + 64)))
    ncv_base = max(k + 1, min(ncv_base, n - 1))

    attempts: List[Tuple[int, float, Optional[int], int]] = [
        (maxiter,                     0.0,  ncv_base,                             0),
        (max(maxiter, 12_000),        1e-7, min(n - 1, max(ncv_base, 3*k + 20)), 1),
        (max(maxiter, 40_000),        1e-5, min(n - 1, max(ncv_base, 6*k + 32)), 2),
    ]
    for mi, tol, ncv_try, seed_off in attempts:
        ncv_use = int(min(max(ncv_try, k + 1), n - 1)) if ncv_try is not None else None
        rng_i = np.random.default_rng(8023 + int(seed_off) * 17)
        v0 = rng_i.standard_normal(n)
        v0 /= max(float(np.linalg.norm(v0)), 1e-12)
        kwargs: Dict[str, Any] = {
            "k": k, "which": "LM", "maxiter": int(mi), "tol": float(tol), "v0": v0,
        }
        if ncv_use is not None:
            kwargs["ncv"] = ncv_use
        try:
            return eigsh(Lop, **kwargs)
        except (ArpackNoConvergence, ArpackError):
            continue

    # LOBPCG fallback
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((n, k))
    x0, _ = np.linalg.qr(x0, mode="reduced")
    out = lobpcg(
        Lop, x0, largest=True,
        maxiter=max(400, min(4000, maxiter * 2)),
        tol=1e-4, verbosityLevel=0,
    )
    evals, evecs = out[0], out[1]
    return evals.astype(np.float64), evecs.astype(np.float64)


def _topk_eigen_symmetric_lm_cupy(
    T: csr_matrix,
    n: int,
    k: int,
    *,
    maxiter: int,
    device_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same operator L = I - T^T T as ``_topk_eigen_symmetric_lm``, but matvec on GPU via CuPy CSR.
    Requires ``cupy`` (e.g. ``pip install cupy-cuda12x`` matching your CUDA build).
    """
    import cupy as cp  # type: ignore[import-not-found]
    import cupyx.scipy.sparse as cps  # type: ignore[import-not-found]
    import cupyx.scipy.sparse.linalg as cpsl  # type: ignore[import-not-found]

    if k <= 0 or k >= n:
        raise ValueError(f"need 0 < k < n, got k={k}, n={n}")

    with cp.cuda.Device(device_id):
        T_c = cps.csr_matrix(
            (
                cp.asarray(T.data, dtype=cp.float64),
                cp.asarray(T.indices),
                cp.asarray(T.indptr),
            ),
            shape=T.shape,
        )
        Tt = T_c.T

        def mv(x: Any) -> Any:
            xv = cp.asarray(x, dtype=cp.float64).ravel()
            return xv - (Tt @ (T_c @ xv))

        Lop = cpsl.LinearOperator((n, n), matvec=mv, dtype=np.float64)

        ncv_base = int(max(4 * k + 8, k + 40, min(n - 1, 2 * k + 64)))
        ncv_base = max(k + 1, min(ncv_base, n - 1))

        attempts: List[Tuple[int, float, Optional[int], int]] = [
            (maxiter,                     0.0,  ncv_base,                             0),
            (max(maxiter, 12_000),        1e-7, min(n - 1, max(ncv_base, 3*k + 20)), 1),
            (max(maxiter, 40_000),        1e-5, min(n - 1, max(ncv_base, 6*k + 32)), 2),
        ]
        last_err: Optional[BaseException] = None
        for mi, tol, ncv_try, seed_off in attempts:
            ncv_use = int(min(max(ncv_try, k + 1), n - 1)) if ncv_try is not None else None
            rng_i = np.random.default_rng(8023 + int(seed_off) * 17)
            v0 = rng_i.standard_normal(n)
            v0 /= max(float(np.linalg.norm(v0)), 1e-12)
            kwargs: Dict[str, Any] = {
                "k": k,
                "which": "LM",
                "maxiter": int(mi),
                "tol": float(tol),
                "v0": cp.asarray(v0, dtype=cp.float64),
            }
            if ncv_use is not None:
                kwargs["ncv"] = ncv_use
            try:
                evals, evecs = cpsl.eigsh(Lop, **kwargs)
                return cp.asnumpy(evals).astype(np.float64), cp.asnumpy(evecs).astype(np.float64)
            except Exception as e:  # noqa: BLE001 — mirror SciPy retry loop
                last_err = e
                continue

        if last_err is not None:
            raise RuntimeError(f"CuPy eigsh did not converge after retries: {last_err}") from last_err
        raise RuntimeError("CuPy eigsh failed without a stored error")


def oracle_gap_principal_modes(
    T: csr_matrix,
    r: int,
    *,
    pi: Optional[np.ndarray] = None,
    a: Optional[np.ndarray] = None,
    maxiter: int = 2000,
    principal_device: Optional[str] = None,
    principal_blas_threads: Optional[int] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Algorithm 3: top-r eigenvectors of L = I - T^T T, then weighted oracle gap.

    Returns (Q_hat_r, ell_pointwise, lambdas, eigenvectors) where eigenvectors is (n, r).
    Default weights: uniform pi_k = 1/r, amplitudes a_k = 1.

    principal_device : optional CuPy device string (``\"cuda\"``, ``\"cuda:1\"``). When set
        and CuPy is installed, the ARPACK matvec runs on GPU CSR (same math as SciPy).
        Otherwise falls back to SciPy on CPU.

    principal_blas_threads : for the CPU SciPy path, temporarily set OMP/MKL/OpenBLAS
        thread env vars. ``None`` uses a conservative default; ``0`` leaves env unchanged.
    """
    n = T.shape[0]
    r = int(min(max(1, r), n - 1))
    if pi is None:
        pi = np.ones(r, dtype=np.float64) / r
    else:
        pi = np.asarray(pi, dtype=np.float64).ravel()
        pi = pi[:r] / max(float(pi[:r].sum()), 1e-12)
    if a is None:
        a = np.ones(r, dtype=np.float64)
    else:
        a = np.asarray(a, dtype=np.float64).ravel()[:r]

    k_req = min(r + 2, n - 1)
    cuda_idx = _principal_cuda_device_index(principal_device)
    vals: np.ndarray
    vecs: np.ndarray
    if cuda_idx is not None:
        try:
            vals, vecs = _topk_eigen_symmetric_lm_cupy(
                T, n, k_req, maxiter=maxiter, device_id=cuda_idx,
            )
        except ImportError:
            _og_log.warning(
                "principal_device=%r requested but CuPy is not installed; "
                "using CPU SciPy (install cupy-cuda12x or matching wheel).",
                principal_device,
            )
            cuda_idx = None
        except Exception as e:
            _og_log.warning(
                "CuPy principal eigen solve failed (%s); falling back to CPU SciPy.",
                e,
            )
            cuda_idx = None

    if cuda_idx is None:

        def mv(x: np.ndarray) -> np.ndarray:
            return _L_matvec(T, x)

        Lop = LinearOperator(dtype=np.float64, shape=(n, n), matvec=mv, rmatvec=mv)
        with _blas_thread_env(principal_blas_threads):
            vals, vecs = _topk_eigen_symmetric_lm(Lop, n, k_req, maxiter=maxiter)
    order = np.argsort(-vals)
    vals = vals[order][:r]
    vecs = vecs[:, order][:, :r]

    ell = np.zeros(n, dtype=np.float64)
    for j in range(r):
        _, ellj = oracle_gap_rank_one(T, vecs[:, j], a=float(a[j]))
        ell += float(pi[j]) * ellj
    Q = float(np.mean(ell))
    return Q, ell, vals.astype(np.float64), vecs.astype(np.float64)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def local_score_stats(ell: np.ndarray, *, name: str = "local") -> Dict[str, float]:
    """Summary statistics for per-point local scores."""
    x = np.asarray(ell, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{name}_mean": float("nan")}
    qs = np.quantile(x, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        f"{name}_mean":        float(x.mean()),
        f"{name}_std":         float(x.std()),
        f"{name}_min":         float(qs[0]),
        f"{name}_q05":         float(qs[1]),
        f"{name}_q25":         float(qs[2]),
        f"{name}_median":      float(qs[3]),
        f"{name}_q75":         float(qs[4]),
        f"{name}_q95":         float(qs[5]),
        f"{name}_max":         float(qs[6]),
        f"{name}_frac_gt_1e-8": float(np.mean(x > 1e-8)),
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PairwiseOracleGapResult:
    """
    Algorithm 2 = consensus rank-one modes applied via fixed prescribed modes.
    Algorithm 3 = principal lost modes from I - T^T T.
    Bandwidth is now per-row adaptive (``eps_hat_rows`` replaces a single global ``eps_hat``).
    Use ``eps_grids`` / ``cv_scores`` as (n, M); the ``eps_grid`` property is an (M,) column
    mean for coarse diagnostics only.
    """

    eps_hat_rows: np.ndarray          # (n,) per-point selected bandwidths
    eps_hat_mean: float               # mean of eps_hat_rows (for CSV logging)
    eps_hat_median: float             # median of eps_hat_rows
    eps_hat_std: float                # std of eps_hat_rows
    eps_grids: np.ndarray             # (n, M) per-point candidate grids
    cv_scores: np.ndarray             # (n, M) per-point LOO losses
    # Algorithm 2
    alg2_Q_per_mode: List[float]
    alg2_Q_mean: float
    local_stats_alg2: List[Dict[str, float]]
    lambdas_consensus: np.ndarray
    # Algorithm 3
    alg3_Q_rank_r: float
    local_stats_alg3: Dict[str, float]
    lambdas_principal: np.ndarray
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ---------- backward-compatible aliases ----------
    @property
    def eps_hat(self) -> float:
        """Backward-compatible scalar: returns mean of per-row bandwidths."""
        return self.eps_hat_mean

    @property
    def eps_grid(self) -> np.ndarray:
        """(M,) bandwidth axis for coarse diagnostics: column-wise nanmean of ``eps_grids``."""
        g = self.eps_grids
        if g.ndim == 2:
            return np.nanmean(np.asarray(g, dtype=np.float64), axis=0)
        return np.asarray(g, dtype=np.float64).ravel()

    @property
    def Q_rank1_modes(self) -> List[float]:
        return self.alg2_Q_per_mode

    @property
    def Q_principal(self) -> float:
        return self.alg3_Q_rank_r

    @property
    def local_stats_rank1(self) -> List[Dict[str, float]]:
        return self.local_stats_alg2

    @property
    def local_stats_principal(self) -> Dict[str, float]:
        return self.local_stats_alg3


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_pairwise_adaptive_fiber_operator(
    U: np.ndarray,
    V: np.ndarray,
    *,
    knn_k: int = 24,
    bandwidth_grid_M: int = 24,
    sigma_clip: float = 3.0,
    smooth_bw: bool = True,
    density_normalize: bool = True,
    density_alpha: float = 1.0,
) -> Tuple[csr_matrix, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the same row-stochastic fiber operator ``T`` as in ``compute_pairwise_oracle_gap``
    (kNN on ``V``, per-row adaptive bandwidth from ``U``, Gaussian + cutoff).

    Also returns ``eps_hat_rows``, ``eps_grids``, and ``cv_scores`` for
    ``PairwiseOracleGapResult``.  Does not build the consensus graph; ``diag``
    has no ``consensus_*`` keys.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n = int(U.shape[0])
    d_V = int(V.shape[1])
    if int(V.shape[0]) != n:
        raise ValueError("U and V must have the same number of rows")

    sq_V = pairwise_sq_euclidean(V)
    knn_sq, knn_idx = knn_from_sq_dist(sq_V, knn_k)
    eps_hat_rows, eps_grids, cv_scores = bandwidth_cv_grid_adaptive(
        U,
        knn_idx,
        knn_sq,
        M=bandwidth_grid_M,
        sigma_clip=sigma_clip,
        smooth_bw=smooth_bw,
    )
    T = build_sparse_T_adaptive(
        knn_idx,
        knn_sq,
        eps_hat_rows,
        loo=False,
        sigma_clip=sigma_clip,
        density_normalize=density_normalize,
        density_alpha=density_alpha,
    )
    t_stats = t_matrix_row_nnz_stats(T)
    cv_arr = np.asarray(cv_scores, dtype=np.float64)
    diag: Dict[str, Any] = {
        **t_stats,
        "cv_loss_mean":          float(np.nanmean(cv_arr[np.isfinite(cv_arr)])),
        "eps_hat_mean":          float(eps_hat_rows.mean()),
        "eps_hat_median":        float(np.median(eps_hat_rows)),
        "eps_hat_std":           float(eps_hat_rows.std()),
        "eps_hat_min":           float(eps_hat_rows.min()),
        "eps_hat_max":           float(eps_hat_rows.max()),
        "bandwidth_grid_M":      int(bandwidth_grid_M),
        "knn_k":                 int(knn_k),
        "d_V":                   int(d_V),
        "sigma_clip":            float(sigma_clip),
        "smooth_bw":             int(smooth_bw),
        "fiber_kernel":          "gaussian_adaptive",
        "density_normalize":     int(bool(density_normalize)),
        "density_alpha":         float(density_alpha),
    }
    return T, diag, eps_hat_rows, eps_grids, cv_scores


def compute_pairwise_oracle_gap(
    U: np.ndarray,
    V: np.ndarray,
    consensus_embeddings: Sequence[np.ndarray],
    *,
    r_consensus: int = 8,
    r_principal: int = 8,
    knn_k: int = 24,
    bandwidth_grid_M: int = 24,
    principal_maxiter: int = 2000,
    principal_device: Optional[str] = None,
    principal_blas_threads: Optional[int] = None,
    run_alg2: bool = True,
    run_alg3: bool = True,
    fiber_kernel: str = "gaussian",   # kept for API compat; adaptive always uses Gaussian
    sigma_clip: float = 3.0,
    smooth_bw: bool = True,
    density_normalize: bool = True,
    density_alpha: float = 1.0,
) -> PairwiseOracleGapResult:
    """
    Full pipeline for one ordered pair (U is fine, V is coarse for fiber construction).

    Changes vs original
    -------------------
    * Bandwidth: per-row adaptive LOO-CV (bandwidth_cv_grid_adaptive) replaces
      global CV + aggressive max-knn guard.
    * Kernel: unnormalized Gaussian with hard cutoff at max(kth-NN dist, sigma_clip * sigma_i)
      per row; ``fiber_kernel`` argument is retained for API compatibility but ignored
      (adaptive path always uses Gaussian with cutoff).
    * Diagnostics: eps_hat_mean / median / std replace scalar eps_hat;
      eps_T_over_eps_hat removed.

    consensus_embeddings : all L model embeddings (aligned rows), used only to build
                           consensus graph modes (Algorithm 4 / Algorithm 2 modes).

    principal_device / principal_blas_threads : passed to ``oracle_gap_principal_modes``
    (optional CuPy GPU matvec; CPU path pins BLAS thread env during eigsh).

    density_normalize / density_alpha : passed to ``build_sparse_T_adaptive`` (directed
    marginal rescale before row-stochastic normalization).
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n = U.shape[0]
    d_V = int(V.shape[1])
    if V.shape[0] != n:
        raise ValueError("U and V must have the same number of rows")

    d_u = int(U.shape[1])
    _og_log.info(
        "[oracle_gap_pairwise] step=knn_on_V — n=%d d_U=%d d_V=%d knn_k=%d",
        n,
        d_u,
        d_V,
        knn_k,
    )
    _og_log.info(
        "[oracle_gap_pairwise] step=bandwidth_cv — M=%d sigma_clip=%g smooth_bw=%s",
        bandwidth_grid_M,
        sigma_clip,
        smooth_bw,
    )
    T, fiber_diag, eps_hat_rows, eps_grids, cv_scores = build_pairwise_adaptive_fiber_operator(
        U,
        V,
        knn_k=knn_k,
        bandwidth_grid_M=bandwidth_grid_M,
        sigma_clip=sigma_clip,
        smooth_bw=smooth_bw,
        density_normalize=density_normalize,
        density_alpha=density_alpha,
    )
    _og_log.info("[oracle_gap_pairwise] step=build_T — adaptive Gaussian + cutoff")

    cv_arr = np.asarray(cv_scores, dtype=np.float64)

    _og_log.info("[oracle_gap_pairwise] step=consensus_W — L=%d", len(consensus_embeddings))
    W_cons = consensus_affinity_matrix(consensus_embeddings)

    diag: Dict[str, Any] = {
        **fiber_diag,
        "principal_device":      principal_device,
        "principal_blas_threads": principal_blas_threads,
        **consensus_summary_from_W(W_cons),
    }

    # 5. Algorithm 2: consensus modes → rank-one oracle gap
    Q_modes: List[float] = []
    stats_modes: List[Dict[str, float]] = []
    lam_c = np.array([], dtype=np.float64)
    if run_alg2:
        _og_log.info("[oracle_gap_pairwise] step=alg2 — consensus modes r=%d", r_consensus)
        psi, lam_c = consensus_graph_modes_from_W(W_cons, r_consensus)
        for j in range(psi.shape[1]):
            Qj, ellj = oracle_gap_rank_one(T, psi[:, j], a=1.0)
            Q_modes.append(Qj)
            stats_modes.append(local_score_stats(ellj, name=f"alg2_mode{j + 1}_local"))
    alg2_mean = float(np.mean(Q_modes)) if Q_modes else float("nan")

    # 6. Algorithm 3: principal lost modes
    Qp = float("nan")
    ellp = np.zeros(n, dtype=np.float64)
    lamp = np.array([], dtype=np.float64)
    stats_p: Dict[str, float] = {}
    if run_alg3:
        r_p = min(r_principal, n - 1)
        _og_log.info(
            "[oracle_gap_pairwise] step=alg3 — principal modes r=%d maxiter=%d",
            r_p,
            principal_maxiter,
        )
        Qp, ellp, lamp, _ = oracle_gap_principal_modes(
            T,
            r_p,
            pi=None,
            a=None,
            maxiter=principal_maxiter,
            principal_device=principal_device,
            principal_blas_threads=principal_blas_threads,
        )
        stats_p = local_score_stats(ellp, name="alg3_rank_r_local")

    _og_log.info("[oracle_gap_pairwise] step=done — alg2_Q_mean=%.6g alg3_Q=%.6g", alg2_mean, float(Qp))

    return PairwiseOracleGapResult(
        eps_hat_rows=eps_hat_rows,
        eps_hat_mean=float(eps_hat_rows.mean()),
        eps_hat_median=float(np.median(eps_hat_rows)),
        eps_hat_std=float(eps_hat_rows.std()),
        eps_grids=eps_grids,
        cv_scores=cv_arr,
        alg2_Q_per_mode=Q_modes,
        alg2_Q_mean=alg2_mean,
        local_stats_alg2=stats_modes,
        lambdas_consensus=lam_c,
        alg3_Q_rank_r=float(Qp),
        local_stats_alg3=stats_p,
        lambdas_principal=lamp,
        diagnostics=diag,
    )