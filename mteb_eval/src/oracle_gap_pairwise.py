"""
Oracle-gap pairwise metrics (paper: soft empirical fibers, Algorithms 2–3).

Pure NumPy/SciPy. Intended for row-aligned embeddings U, V from two encoders
on the same texts. Consensus graph modes (Algorithm 4) live here as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import LinearOperator, eigsh
except ImportError as e:  # pragma: no cover
    raise ImportError("oracle_gap_pairwise requires scipy") from e


def epanechnikov(u: np.ndarray) -> np.ndarray:
    """k(u) = (1-u^2)_+ scaled so max is 1 (compact on [0,1])."""
    u = np.asarray(u, dtype=np.float64)
    out = np.maximum(0.0, 1.0 - u * u)
    return np.where(u <= 1.0, out, 0.0)


def _row_normalize_nonneg(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = W.sum(axis=1, keepdims=True)
    s = np.maximum(s, eps)
    return W / s


def pairwise_sq_euclidean(X: np.ndarray) -> np.ndarray:
    """(n,n) squared Euclidean distances for rows of X (float64)."""
    X = np.asarray(X, dtype=np.float64)
    g = X @ X.T
    d = np.diag(g)
    return np.maximum(0.0, d[:, None] + d[None, :] - 2.0 * g)


def knn_from_sq_dist(
    sq_dist: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (knn_sq, knn_idx): per row, squared distances and indices of the k
    nearest neighbors excluding self.
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


def consensus_affinity_matrix(
    embeddings_list: Sequence[np.ndarray],
) -> np.ndarray:
    """
    W_ij = exp(-(1/L) * sum_l (d_ij^(l) / c_l)^2) with c_l = median pairwise distance
    in embedding l (excluding zeros on diagonal). Diagonal set to 0.
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
    """Scalar diagnostics from a consensus affinity matrix ``W``."""
    deg = W.sum(axis=1).astype(np.float64)
    return {
        "consensus_W_sum": float(W.sum()),
        "consensus_deg_min": float(deg.min()),
        "consensus_deg_max": float(deg.max()),
        "consensus_deg_mean": float(deg.mean()),
        "consensus_n_edges_pos": float(np.mean(deg > 0)),
    }


def consensus_graph_summary(embeddings_list: Sequence[np.ndarray]) -> Dict[str, float]:
    """Scalar diagnostics (builds ``W`` once)."""
    return consensus_summary_from_W(consensus_affinity_matrix(embeddings_list))


def consensus_graph_modes_from_W(W: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smallest nontrivial modes of ``L_sym = I - D^{-1/2} W D^{-1/2}`` given ``W``.
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
        s = psi[:, j].sum()
        if s < 0.0:
            psi[:, j] *= -1.0
    return psi, lam


def consensus_graph_modes(
    embeddings_list: Sequence[np.ndarray],
    r: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same as ``consensus_graph_modes_from_W`` but builds ``W`` from embeddings."""
    W = consensus_affinity_matrix(embeddings_list)
    return consensus_graph_modes_from_W(W, r)


def build_sparse_T_from_knn_csr(
    knn_idx: np.ndarray,
    knn_sq: np.ndarray,
    eps: float,
    d_V: int,
    *,
    loo: bool,
) -> csr_matrix:
    """Same as build_sparse_T_from_knn but without full (n,n) sq matrix."""
    n, k = knn_idx.shape
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = knn_idx.astype(np.int32).ravel()
    dist = np.sqrt(np.maximum(0.0, knn_sq.astype(np.float64)))
    u = dist / max(eps, 1e-12)
    w = epanechnikov(u) * (max(eps, 1e-12) ** (-float(d_V)))
    if loo:
        mask = (knn_idx != np.arange(n)[:, None]).ravel()
        w = np.where(mask, w.ravel(), 0.0)
    else:
        w = w.ravel()
    mat = csr_matrix((w, (rows, cols)), shape=(n, n))
    mat.eliminate_zeros()
    mat = mat.tocsr()
    if not loo:
        # self-inclusion (Algorithm 1 final step): k(0)=1 on the diagonal
        scale0 = max(eps, 1e-12) ** (-float(d_V))
        mat = mat + csr_matrix(
            (np.full(n, scale0, dtype=np.float64), (np.arange(n), np.arange(n))),
            shape=(n, n),
        )
    sums = np.array(mat.sum(axis=1)).ravel()
    sums = np.maximum(sums, 1e-12)
    mat = mat.multiply((1.0 / sums)[:, None])
    return mat


def t_matrix_row_nnz_stats(T: csr_matrix) -> Dict[str, float]:
    """Nonzeros per row of T (after construction)."""
    n = int(T.shape[0])
    if n == 0:
        return {"T_row_nnz_mean": 0.0, "T_row_nnz_min": 0.0, "T_row_nnz_max": 0.0}
    nnz = np.diff(T.indptr).astype(np.int64)
    return {
        "T_row_nnz_mean": float(nnz.mean()),
        "T_row_nnz_min": float(nnz.min()),
        "T_row_nnz_max": float(nnz.max()),
    }


def bandwidth_cv_grid(
    U: np.ndarray,
    V: np.ndarray,
    knn_idx: np.ndarray,
    knn_sq: np.ndarray,
    d_V: int,
    *,
    M: int = 24,
    loo: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Leave-one-out CV (Eq. CV in paper) over a geometric bandwidth grid.

    Returns (eps_hat, eps_grid, cv_scores).
    """
    n = V.shape[0]
    if n < 3:
        raise ValueError("need at least 3 points for bandwidth CV")

    # 1-NN distance on V (closest neighbor per row, excluding self)
    d1 = np.sqrt(np.maximum(0.0, knn_sq[:, 0]))
    m0 = int(np.ceil(np.sqrt(n)))
    m0 = min(max(2, m0), knn_sq.shape[1])
    dm = np.sqrt(np.maximum(0.0, knn_sq[:, m0 - 1]))
    eps_min = float(np.median(d1))
    eps_max = float(np.median(dm))
    eps_min = max(eps_min, 1e-8)
    eps_max = max(eps_max, eps_min * 1.0001)
    M = max(2, int(M))
    log_grid = np.linspace(np.log(eps_min), np.log(eps_max), M)
    eps_grid = np.exp(log_grid)

    U = np.asarray(U, dtype=np.float64)
    cv_scores = np.empty(M, dtype=np.float64)
    for mi, eps in enumerate(eps_grid):
        T = build_sparse_T_from_knn_csr(knn_idx, knn_sq, float(eps), d_V, loo=loo)
        pred = T @ U
        if loo:
            # rows still sum to 1 over j!=i; pred is LOO conditional mean
            diff = U - pred
        else:
            diff = U - pred
        cv_scores[mi] = float(np.mean(np.sum(diff * diff, axis=1)))
    best = int(np.argmin(cv_scores))
    return float(eps_grid[best]), eps_grid, cv_scores


def oracle_gap_rank_one(
    T: csr_matrix, phi: np.ndarray, *, a: float = 1.0
) -> Tuple[float, np.ndarray]:
    """
    Algorithm 2: global Q_hat and local ell_i for one mode phi (length n).
    """
    phi = np.asarray(phi, dtype=np.float64).ravel()
    Tphi = T @ phi
    Tabs_phi = T @ np.abs(phi)
    ell = 0.5 * float(np.abs(a)) * (Tabs_phi - np.abs(Tphi))
    Q = float(np.mean(ell))
    return Q, ell


def _L_matvec(T: csr_matrix, x: np.ndarray) -> np.ndarray:
    Tx = T @ x
    if x.ndim == 1:
        return x - (T.T @ Tx)
    # batched columns
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        col = x[:, j]
        out[:, j] = col - (T.T @ (T @ col))
    return out


def oracle_gap_principal_modes(
    T: csr_matrix,
    r: int,
    *,
    pi: Optional[np.ndarray] = None,
    a: Optional[np.ndarray] = None,
    maxiter: int = 2000,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Algorithm 3: top-r eigenvectors of L = I - T^T T, then weighted oracle gap.

    Returns (Q_hat_r, ell_pointwise, lambdas, eigenvectors) where eigenvectors is (n,r).
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

    def mv(x: np.ndarray) -> np.ndarray:
        return _L_matvec(T, x)

    Lop = LinearOperator(
        dtype=np.float64,
        shape=(n, n),
        matvec=mv,
        rmatvec=mv,
    )
    # largest eigenvalues of symmetric PSD L
    k = min(r + 2, n - 1, n)
    vals, vecs = eigsh(Lop, k=k, which="LM", maxiter=maxiter)
    order = np.argsort(-vals)
    vals = vals[order][:r]
    vecs = vecs[:, order][:, :r]

    ell = np.zeros(n, dtype=np.float64)
    for j in range(r):
        _, ellj = oracle_gap_rank_one(T, vecs[:, j], a=float(a[j]))
        ell += float(pi[j]) * ellj
    Q = float(np.mean(ell))
    return Q, ell, vals.astype(np.float64), vecs.astype(np.float64)


def local_score_stats(ell: np.ndarray, *, name: str = "local") -> Dict[str, float]:
    """Summary statistics for per-point local scores (no raw vectors saved)."""
    x = np.asarray(ell, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{name}_mean": float("nan")}
    qs = np.quantile(x, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        f"{name}_mean": float(x.mean()),
        f"{name}_std": float(x.std()),
        f"{name}_min": float(qs[0]),
        f"{name}_q05": float(qs[1]),
        f"{name}_q25": float(qs[2]),
        f"{name}_median": float(qs[3]),
        f"{name}_q75": float(qs[4]),
        f"{name}_q95": float(qs[5]),
        f"{name}_max": float(qs[6]),
        f"{name}_frac_gt_1e-8": float(np.mean(x > 1e-8)),
    }


@dataclass
class PairwiseOracleGapResult:
    """Algorithm 2 = consensus rank-one modes; Algorithm 3 = principal lost modes."""

    eps_hat: float
    eps_grid: np.ndarray
    cv_scores: np.ndarray
    # Algorithm 2 (prescribed consensus modes ψ_k)
    alg2_Q_per_mode: List[float]
    alg2_Q_mean: float
    local_stats_alg2: List[Dict[str, float]]
    lambdas_consensus: np.ndarray
    # Algorithm 3 (eigenvectors of I - T^T T)
    alg3_Q_rank_r: float
    local_stats_alg3: Dict[str, float]
    lambdas_principal: np.ndarray
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def Q_rank1_modes(self) -> List[float]:  # backward-compatible alias
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


def compute_pairwise_oracle_gap(
    U: np.ndarray,
    V: np.ndarray,
    consensus_embeddings: Sequence[np.ndarray],
    *,
    r_consensus: int = 8,
    r_principal: int = 8,
    knn_k: int = 128,
    bandwidth_grid_M: int = 24,
    principal_maxiter: int = 2000,
    run_alg2: bool = True,
    run_alg3: bool = True,
) -> PairwiseOracleGapResult:
    """
    Full pipeline for one ordered pair (U is fine, V is coarse for fiber construction).

    consensus_embeddings: all L model embeddings (aligned rows), used only to build modes.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n = U.shape[0]
    d_V = int(V.shape[1])
    if V.shape[0] != n:
        raise ValueError("U and V must have the same number of rows")

    sq_V = pairwise_sq_euclidean(V)
    knn_sq, knn_idx = knn_from_sq_dist(sq_V, knn_k)

    eps_hat, eps_grid, cv_scores = bandwidth_cv_grid(
        U,
        V,
        knn_idx,
        knn_sq,
        d_V,
        M=bandwidth_grid_M,
        loo=True,
    )
    T = build_sparse_T_from_knn_csr(knn_idx, knn_sq, eps_hat, d_V, loo=False)
    t_stats = t_matrix_row_nnz_stats(T)
    cv_arr = np.asarray(cv_scores, dtype=np.float64)
    best_i = int(np.argmin(cv_arr))
    W_cons = consensus_affinity_matrix(consensus_embeddings)
    diag: Dict[str, Any] = {
        **t_stats,
        "cv_loss_min": float(cv_arr.min()),
        "cv_loss_at_eps_hat": float(cv_arr[best_i]),
        "eps_min_grid": float(eps_grid[0]),
        "eps_max_grid": float(eps_grid[-1]),
        "bandwidth_grid_M": int(bandwidth_grid_M),
        "knn_k": int(knn_k),
        "d_V": int(d_V),
        **consensus_summary_from_W(W_cons),
    }

    Q_modes: List[float] = []
    stats_modes: List[Dict[str, float]] = []
    lam_c = np.array([], dtype=np.float64)
    if run_alg2:
        psi, lam_c = consensus_graph_modes_from_W(W_cons, r_consensus)
        for j in range(psi.shape[1]):
            Qj, ellj = oracle_gap_rank_one(T, psi[:, j], a=1.0)
            Q_modes.append(Qj)
            stats_modes.append(local_score_stats(ellj, name=f"alg2_mode{j + 1}_local"))
    alg2_mean = float(np.mean(Q_modes)) if Q_modes else float("nan")

    Qp = float("nan")
    ellp = np.zeros(n, dtype=np.float64)
    lamp = np.array([], dtype=np.float64)
    stats_p: Dict[str, float] = {}
    if run_alg3:
        r_p = min(r_principal, n - 1)
        Qp, ellp, lamp, _ = oracle_gap_principal_modes(
            T,
            r_p,
            pi=None,
            a=None,
            maxiter=principal_maxiter,
        )
        stats_p = local_score_stats(ellp, name="alg3_rank_r_local")

    return PairwiseOracleGapResult(
        eps_hat=eps_hat,
        eps_grid=eps_grid,
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
