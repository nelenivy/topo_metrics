"""
oracle_gap_unary.py
===================
Unary (single-embedding) oracle-gap metric: metric(U), not metric(U, V).

Geometric interpretation
------------------------
The pairwise oracle gap measures how much the consensus manifold geometry is
destroyed when going U -> V.  In the unitary case there is no separate V:
U itself plays the role of the fiber base space.  The fiber operator T is
built on U, and the test functions psi_k come from the consensus graph of the
full ensemble.  The score measures:

    "How much of the consensus manifold geometry oscillates within the
     local neighborhoods imposed by U?"

Low score = U's neighborhoods coincide with consensus neighborhoods (good).
High score = U mixes consensus-distant points in its fibers (bad).

Robust consensus graph construction
-------------------------------------
Uses mutual-kNN intersection + inverse-Fiedler weighting instead of the
plain exponential-of-sum used in oracle_gap_pairwise.  This guards against:
  - AND-collapse with large L (many models)
  - Bad-model majority (bad models get near-zero weight via Fiedler criterion)

Inverse-Fiedler weighting
--------------------------
For each model l, weight_l = 1 / (lambda_1(L_sym^(l)) + eps).
A well-clustered / structured model has a small Fiedler value (tight clusters)
and therefore high weight.  A random / uninformative model has Fiedler near 1
and weight near 1.  Weights are then L1-normalized.

Self-fiber bandwidth
---------------------
Per-row LOO-CV selects the scale at which U is locally self-consistent.
The candidate grid spans (10% quantile of local 1-NN distances) / 3 … 0.9 quantile of each row's
kNN distances (same as ``oracle_gap_pairwise.bandwidth_cv_grid_adaptive``).
Flat-curve guard uses eps_min (tightest scale) because a flat LOO curve for
self-prediction indicates U is already very smooth at all tested scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.sparse import csr_matrix, diags as sp_diags
    from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh
except ImportError as e:
    raise ImportError("oracle_gap_unary requires scipy") from e

from src.oracle_gap_pairwise import build_sparse_T_adaptive, oracle_gap_principal_modes


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _gaussian_weights(dist: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    x = np.asarray(dist, dtype=np.float64) / sigma
    return np.exp(-0.5 * np.minimum(x * x, 750.0))


def pairwise_sq_euclidean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    g = X @ X.T
    d = np.diag(g)
    return np.maximum(0.0, d[:, None] + d[None, :] - 2.0 * g)


def knn_from_sq_dist(sq_dist: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
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
# Model quality weight: inverse Fiedler value
# ---------------------------------------------------------------------------

def _inverse_fiedler_weight(X: np.ndarray, k: int, *, fiedler_floor: float = 1e-3) -> float:
    """
    1 / (fiedler_value + fiedler_floor) for embedding X.

    The Fiedler value is lambda_1 (smallest nontrivial eigenvalue) of the
    symmetrised normalised Laplacian L_sym built from the kNN graph of X.

    Small Fiedler = tight cluster structure = informative model -> high weight.
    Large Fiedler = diffuse / random geometry = uninformative model -> low weight.
    """
    n = X.shape[0]
    sq = pairwise_sq_euclidean(X)
    knn_sq, knn_idx = knn_from_sq_dist(sq, k)
    dist = np.sqrt(np.maximum(0.0, knn_sq.astype(np.float64)))
    med = float(np.median(dist[dist > 0])) if np.any(dist > 0) else 1.0
    med = max(med, 1e-12)

    w = _gaussian_weights(dist, np.full_like(dist, med))
    sums = w.sum(axis=1, keepdims=True)
    w /= np.maximum(sums, 1e-12)

    rows_r = np.repeat(np.arange(n, dtype=np.int32), k)
    cols_r = knn_idx.ravel().astype(np.int32)
    W_sp = csr_matrix((w.ravel(), (rows_r, cols_r)), shape=(n, n))
    W_sym = 0.5 * (W_sp + W_sp.T)

    deg = np.array(W_sym.sum(axis=1)).ravel()
    deg = np.maximum(deg, 1e-10)
    inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sp_diags(inv_sqrt)
    L_sym = (csr_matrix(np.eye(n, dtype=np.float64)) - D_inv_sqrt @ W_sym @ D_inv_sqrt).tocsr()

    try:
        vals, _ = eigsh(L_sym, k=2, which="SM", tol=1e-4, maxiter=5000)
        fiedler = float(sorted(np.maximum(0.0, vals.real))[1])
    except (ArpackNoConvergence, ArpackError, Exception):
        fiedler = 0.5  # neutral fallback

    return 1.0 / max(fiedler, fiedler_floor)


# ---------------------------------------------------------------------------
# Robust consensus affinity: mutual-kNN + inverse-Fiedler weights
# ---------------------------------------------------------------------------

def mutual_knn_consensus_affinity(
    embeddings_list: Sequence[np.ndarray],
    knn_k: int = 15,
    *,
    use_fiedler_weights: bool = True,
    fiedler_floor: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a robust consensus affinity matrix.

    For each directed edge (i->j), only models that include j in kNN(i)
    contribute.  This prevents AND-collapse: one dissenting model cannot
    zero a consensus edge.  Weights are inverse-Fiedler so that structured
    models dominate.

    Returns
    -------
    W       : (n, n) symmetric affinity, diagonal = 0
    weights : (L,) per-model weights (sums to 1)
    """
    if not embeddings_list:
        raise ValueError("embeddings_list is empty")

    L = len(embeddings_list)
    n = int(embeddings_list[0].shape[0])
    k = min(knn_k, n - 1)

    # Per-model kNN
    all_knn_idx  = []
    all_knn_dist = []   # normalised by per-model median
    all_knn_sq   = []

    for X in embeddings_list:
        X = np.asarray(X, dtype=np.float64)
        sq = pairwise_sq_euclidean(X)
        knn_sq, knn_idx = knn_from_sq_dist(sq, k)
        dist = np.sqrt(np.maximum(0.0, knn_sq))
        med = float(np.median(dist[dist > 0])) if np.any(dist > 0) else 1.0
        med = max(med, 1e-12)
        all_knn_idx.append(knn_idx)
        all_knn_dist.append(dist / med)
        all_knn_sq.append(knn_sq)

    # Per-model weights
    if use_fiedler_weights:
        raw_w = np.array([
            _inverse_fiedler_weight(
                np.asarray(embeddings_list[l], dtype=np.float64), k,
                fiedler_floor=fiedler_floor,
            )
            for l in range(L)
        ], dtype=np.float64)
        raw_w = np.maximum(raw_w, 1e-6)
        weights = raw_w / raw_w.sum()
    else:
        weights = np.ones(L, dtype=np.float64) / L

    # Accumulate weighted mutual-kNN counts and weighted squared distances
    W_cnt   = np.zeros((n, n), dtype=np.float64)
    W_dist2 = np.zeros((n, n), dtype=np.float64)

    for l in range(L):
        w_l = float(weights[l])
        knn_idx_l  = all_knn_idx[l]
        knn_dist_l = all_knn_dist[l]

        rows_rep = np.repeat(np.arange(n, dtype=np.int32), k)
        cols_rep = knn_idx_l.ravel().astype(np.int32)
        dist_rep = knn_dist_l.ravel()

        np.add.at(W_cnt,   (rows_rep, cols_rep), w_l)
        np.add.at(W_dist2, (rows_rep, cols_rep), w_l * dist_rep ** 2)

    # Symmetrise
    W_cnt   = np.maximum(W_cnt, W_cnt.T)
    W_dist2 = 0.5 * (W_dist2 + W_dist2.T)

    # Affinity: consensus_fraction * exp(-weighted_mean_dist^2)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_d2 = np.where(W_cnt > 1e-12,
                           W_dist2 / np.maximum(W_cnt, 1e-12), 0.0)

    W = W_cnt * np.exp(-mean_d2)
    np.fill_diagonal(W, 0.0)
    return W, weights


def _consensus_summary_from_W(W: np.ndarray) -> Dict[str, float]:
    """Same scalar diagnostics as ``oracle_gap_pairwise.consensus_summary_from_W``."""
    deg = W.sum(axis=1).astype(np.float64)
    return {
        "consensus_W_sum": float(W.sum()),
        "consensus_deg_min": float(deg.min()),
        "consensus_deg_max": float(deg.max()),
        "consensus_deg_mean": float(deg.mean()),
        "consensus_n_edges_pos": float(np.mean(deg > 0)),
    }


# ---------------------------------------------------------------------------
# Consensus graph modes
# ---------------------------------------------------------------------------

def consensus_graph_modes_from_W(
    W: np.ndarray, r: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(W.shape[0])
    deg = np.maximum(W.sum(axis=1).astype(np.float64), 1e-10)
    inv_sqrt = 1.0 / np.sqrt(deg)
    L_sym = np.eye(n, dtype=np.float64) - (inv_sqrt[:, None] * W) * inv_sqrt[None, :]
    r = int(min(max(1, r), n - 1))
    evals, evecs = np.linalg.eigh(L_sym)
    psi = (evecs[:, 1 : r + 1] * inv_sqrt[:, None]).astype(np.float64)
    lam = evals[1 : r + 1].astype(np.float64)
    for j in range(psi.shape[1]):
        if psi[:, j].sum() < 0.0:
            psi[:, j] *= -1.0
    return psi, lam


# ---------------------------------------------------------------------------
# Adaptive per-row bandwidth (self-fiber)
# ---------------------------------------------------------------------------

def _bandwidth_cv_self(
    U: np.ndarray,
    knn_idx: np.ndarray,
    knn_sq: np.ndarray,
    *,
    M: int = 24,
    sigma_clip: float = 3.0,
    smooth_bw: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, k = knn_idx.shape
    U = np.asarray(U, dtype=np.float64)
    knn_dist = np.sqrt(np.maximum(0.0, knn_sq.astype(np.float64)))
    M = max(2, int(M))

    d1 = knn_dist[:, 0]
    d1_stack = np.concatenate([d1[:, None], d1[knn_idx]], axis=1)
    eps_min_rows = np.maximum(np.quantile(d1_stack, 0.1, axis=1) / 3.0, 1e-8)
    q90 = np.quantile(knn_dist, 0.9, axis=1).astype(np.float64)
    eps_max_rows = np.maximum(q90, eps_min_rows * 1.0001)
    t = np.linspace(0.0, 1.0, M)
    eps_grids = np.exp(
        np.log(eps_min_rows)[:, None]
        + t[None, :] * (np.log(eps_max_rows) - np.log(eps_min_rows))[:, None]
    )
    kth_dist = knn_dist[:, k - 1]
    cv_scores = np.full((n, M), np.inf, dtype=np.float64)

    for mi in range(M):
        sigma_i = eps_grids[:, mi]
        cutoff_i = np.maximum(kth_dist, sigma_clip * sigma_i)
        w = _gaussian_weights(knn_dist, sigma_i[:, None])
        w[knn_dist > cutoff_i[:, None]] = 0.0
        w[knn_idx == np.arange(n)[:, None]] = 0.0
        row_sums = w.sum(axis=1, keepdims=True)
        valid = row_sums.ravel() > 1e-12
        w /= np.where(row_sums > 1e-12, row_sums, 1.0)
        pred = np.einsum("ij,ijk->ik", w, U[knn_idx])
        diff_sq = np.sum((U - pred) ** 2, axis=1)
        diff_sq[~valid] = np.inf
        cv_scores[:, mi] = diff_sq

    all_inf = np.all(~np.isfinite(cv_scores), axis=1)
    best_idx = np.argmin(cv_scores, axis=1)
    eps_hat_rows = eps_grids[np.arange(n), best_idx]

    cv_finite = np.where(np.isfinite(cv_scores), cv_scores, np.nan)
    cv_row_min = np.nanmin(cv_finite, axis=1)
    cv_row_max = np.nanmax(cv_finite, axis=1)
    flat = (cv_row_max - cv_row_min) < 1e-4 * (np.abs(cv_row_min) + 1e-12)
    # flat => use eps_min: tightest meaningful scale for self-consistent U
    eps_hat_rows = np.where(flat | all_inf, eps_min_rows, eps_hat_rows)

    if smooth_bw:
        smoothed = np.empty(n, dtype=np.float64)
        for i in range(n):
            vals = np.concatenate([[eps_hat_rows[i]], eps_hat_rows[knn_idx[i]]])
            smoothed[i] = float(np.median(vals))
        eps_hat_rows = smoothed

    return eps_hat_rows, eps_grids, cv_scores


# ---------------------------------------------------------------------------
# Build self-fiber T
# ---------------------------------------------------------------------------

def _build_self_fiber_T(
    knn_idx: np.ndarray,
    knn_sq: np.ndarray,
    eps_rows: np.ndarray,
    *,
    sigma_clip: float = 3.0,
    density_normalize: bool = True,
    density_alpha: float = 1.0,
) -> csr_matrix:
    """Same adaptive Gaussian + cutoff + optional density rescale as pairwise ``T``."""
    return build_sparse_T_adaptive(
        knn_idx,
        knn_sq,
        eps_rows,
        loo=False,
        sigma_clip=sigma_clip,
        density_normalize=density_normalize,
        density_alpha=density_alpha,
    )


# ---------------------------------------------------------------------------
# Oracle gap estimators
# ---------------------------------------------------------------------------

def _oracle_gap_rank_one(T, phi, *, a=1.0):
    phi = np.asarray(phi, dtype=np.float64).ravel()
    Tphi = T @ phi
    Tabs = T @ np.abs(phi)
    ell = 0.5 * abs(a) * (Tabs - np.abs(Tphi))
    return float(np.mean(ell)), ell


def local_score_stats(ell: np.ndarray, *, name: str = "local") -> Dict[str, float]:
    x = np.asarray(ell, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{name}_mean": float("nan")}
    qs = np.quantile(x, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        f"{name}_mean":         float(x.mean()),
        f"{name}_std":          float(x.std()),
        f"{name}_min":          float(qs[0]),
        f"{name}_q05":          float(qs[1]),
        f"{name}_q25":          float(qs[2]),
        f"{name}_median":       float(qs[3]),
        f"{name}_q75":          float(qs[4]),
        f"{name}_q95":          float(qs[5]),
        f"{name}_max":          float(qs[6]),
        f"{name}_frac_gt_1e-8": float(np.mean(x > 1e-8)),
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class UnaryOracleGapResult:
    """
    Unary oracle-gap result for a single embedding U (same shape contract as
    ``PairwiseOracleGapResult`` for shared tooling: ``eps_hat_rows``, ``eps_grids``,
    ``cv_scores``, alg2 / alg3 fields, ``diagnostics``).

    Extra field ``model_weights`` documents inverse-Fiedler ensemble weights.

    Score interpretation (both alg2 and alg3)
    ------------------------------------------
    Lower is better.  Score = 0 means U's fibers perfectly respect the
    consensus geometry.  Score > 0 means U mixes consensus-distinct points
    within its local neighborhoods.

    alg2_Q_mean       : mean over consensus modes -- consensus alignment score.
    alg3_Q_rank_r     : self-inconsistency score -- how much U collapses
                        directions that its own fiber operator should preserve.
    lambdas_principal : eigenvalues of I - T^T T; the spectral mass in the
                        lost subspace.
    model_weights     : (L,) inverse-Fiedler weights; inspect to see which
                        models the consensus considers most structured.
    """

    eps_hat_rows: np.ndarray
    eps_hat_mean: float
    eps_hat_median: float
    eps_hat_std: float
    eps_grids: np.ndarray
    cv_scores: np.ndarray
    alg2_Q_per_mode: List[float]
    alg2_Q_mean: float
    local_stats_alg2: List[Dict[str, float]]
    lambdas_consensus: np.ndarray
    alg3_Q_rank_r: float
    local_stats_alg3: Dict[str, float]
    lambdas_principal: np.ndarray
    model_weights: np.ndarray
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def eps_hat(self) -> float:
        """Scalar bandwidth summary (matches pairwise: mean of per-row values)."""
        return self.eps_hat_mean

    @property
    def eps_grid(self) -> np.ndarray:
        """(M,) coarse bandwidth axis: column-wise nanmean of ``eps_grids`` (pairwise-compatible)."""
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


# Back-compat name used in early drafts of this file.
UnitaryOracleGapResult = UnaryOracleGapResult


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_unary_oracle_gap(
    U: np.ndarray,
    consensus_embeddings: Sequence[np.ndarray],
    *,
    r_consensus: int = 8,
    r_principal: int = 8,
    knn_k: int = 128,
    consensus_knn_k: int = 15,
    bandwidth_grid_M: int = 24,
    principal_maxiter: int = 2000,
    principal_device: Optional[str] = None,
    principal_blas_threads: Optional[int] = None,
    run_alg2: bool = True,
    run_alg3: bool = True,
    fiber_kernel: str = "gaussian",
    sigma_clip: float = 3.0,
    smooth_bw: bool = True,
    use_fiedler_weights: bool = True,
    fiedler_floor: float = 1e-3,
    density_normalize: bool = True,
    density_alpha: float = 1.0,
) -> UnaryOracleGapResult:
    """
    Unary oracle-gap metric for a single embedding U.

    Call shape matches ``compute_pairwise_oracle_gap`` where possible (same keyword
    names for shared hyperparameters). ``fiber_kernel`` is accepted for API
    compatibility and ignored (self-fiber always uses Gaussian + cutoff).

    Extra kwargs vs pairwise: ``consensus_knn_k``, ``use_fiedler_weights``,
    ``fiedler_floor`` for the mutual-kNN consensus graph.
    ``principal_device`` / ``principal_blas_threads`` match ``compute_pairwise_oracle_gap``.

    Parameters
    ----------
    U : (n, d)  embedding to evaluate.
    consensus_embeddings : list of L embeddings forming the ensemble.
        U may or may not be included.  Excluding U gives a pure peer-reference
        score; including U gives a self-vs-peers consistency score.
    r_consensus : number of consensus graph modes for Algorithm 2.
    r_principal : rank for I - T^T T  in Algorithm 3.
    knn_k : neighbourhood size for the self-fiber T (built on U).
    consensus_knn_k : neighbourhood size for the consensus affinity.
    use_fiedler_weights : weight models by inverse Fiedler value of their
        kNN graph Laplacian.  Structured models get higher weight.
    fiedler_floor : minimum Fiedler value to avoid division by zero.
    """
    U = np.asarray(U, dtype=np.float64)
    n = int(U.shape[0])
    d_u = int(U.shape[1])

    sq_U = pairwise_sq_euclidean(U)
    knn_sq_U, knn_idx_U = knn_from_sq_dist(sq_U, knn_k)

    eps_hat_rows, eps_grids, cv_scores = _bandwidth_cv_self(
        U, knn_idx_U, knn_sq_U,
        M=bandwidth_grid_M, sigma_clip=sigma_clip, smooth_bw=smooth_bw,
    )
    cv_arr = np.asarray(cv_scores, dtype=np.float64)

    T = _build_self_fiber_T(
        knn_idx_U,
        knn_sq_U,
        eps_hat_rows,
        sigma_clip=sigma_clip,
        density_normalize=density_normalize,
        density_alpha=density_alpha,
    )

    W_cons, model_weights = mutual_knn_consensus_affinity(
        consensus_embeddings,
        knn_k=consensus_knn_k,
        use_fiedler_weights=use_fiedler_weights,
        fiedler_floor=fiedler_floor,
    )

    nnz = np.diff(T.tocsr().indptr).astype(np.int64)
    diag: Dict[str, Any] = {
        "T_row_nnz_mean": float(nnz.mean()),
        "T_row_nnz_min": float(nnz.min()),
        "T_row_nnz_max": float(nnz.max()),
        "cv_loss_mean": float(np.nanmean(cv_arr[np.isfinite(cv_arr)])),
        "eps_hat_mean": float(eps_hat_rows.mean()),
        "eps_hat_median": float(np.median(eps_hat_rows)),
        "eps_hat_std": float(eps_hat_rows.std()),
        "eps_hat_min": float(eps_hat_rows.min()),
        "eps_hat_max": float(eps_hat_rows.max()),
        "bandwidth_grid_M": int(bandwidth_grid_M),
        "knn_k": int(knn_k),
        "consensus_knn_k": int(consensus_knn_k),
        "d_U": int(d_u),
        "sigma_clip": float(sigma_clip),
        "smooth_bw": int(smooth_bw),
        "fiber_kernel": "gaussian_adaptive",
        "fiber_kernel_arg": str(fiber_kernel),
        "n_models": int(len(consensus_embeddings)),
        "model_weight_max": float(model_weights.max()),
        "model_weight_min": float(model_weights.min()),
        "principal_device": principal_device,
        "principal_blas_threads": principal_blas_threads,
        "density_normalize": int(bool(density_normalize)),
        "density_alpha": float(density_alpha),
        **_consensus_summary_from_W(W_cons),
    }

    Q_modes: List[float] = []
    stats_modes: List[Dict[str, float]] = []
    lam_c = np.array([], dtype=np.float64)
    if run_alg2:
        psi, lam_c = consensus_graph_modes_from_W(W_cons, r_consensus)
        for j in range(psi.shape[1]):
            Qj, ellj = _oracle_gap_rank_one(T, psi[:, j], a=1.0)
            Q_modes.append(Qj)
            stats_modes.append(local_score_stats(ellj, name=f"alg2_mode{j + 1}_local"))
    alg2_mean = float(np.mean(Q_modes)) if Q_modes else float("nan")

    Qp = float("nan")
    lamp = np.array([], dtype=np.float64)
    stats_p: Dict[str, float] = {}
    if run_alg3:
        Qp, ellp, lamp, _ = oracle_gap_principal_modes(
            T,
            min(r_principal, n - 1),
            maxiter=principal_maxiter,
            principal_device=principal_device,
            principal_blas_threads=principal_blas_threads,
        )
        stats_p = local_score_stats(ellp, name="alg3_rank_r_local")

    return UnaryOracleGapResult(
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
        model_weights=model_weights,
        diagnostics=diag,
    )


# Alias: original function name from the first draft of this module.
compute_oracle_gap_unitary = compute_unary_oracle_gap