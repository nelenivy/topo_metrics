"""
Local fiber-covariance priors for oracle-gap diagnostics (pairwise / fiber operator T).

Builds on ``oracle_gap_pairwise`` (Algorithms 2–3 machinery): reuses eigenvectors of
``L = I - T^T T`` as test functions, but scales per-point contributions using top
eigenvalues of each point's soft fiber covariance ``Sigma_i``.

Use ``oracle_gap_pairwise.build_pairwise_adaptive_fiber_operator`` to obtain the same
``T`` as in ``compute_pairwise_oracle_gap`` (kNN on ``V``, adaptive bandwidth on ``U``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    from scipy.sparse import csr_matrix
except ImportError as e:  # pragma: no cover
    raise ImportError("oracle_gap_fiber_cov requires scipy") from e

from src.oracle_gap_pairwise import (
    local_score_stats,
    oracle_gap_principal_modes,
    oracle_gap_rank_one,
)

FiberCovSchedule = Literal["uniform", "whitened", "frobenius"]


def fiber_covariance_eigenvalues(
    T: csr_matrix,
    U: np.ndarray,
    r: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-point soft fiber covariance spectrum (top-r eigenvalues only).

    For row ``i`` of row-stochastic ``T``, soft mean ``mu_i = (T U)_i`` and
    ``Sigma_i = sum_j T_ij (u_j - mu_i)(u_j - mu_i)^T``.  Top eigenvalues are
    obtained from an SVD of the weighted centred neighbor matrix (sparse support
    of ``T`` only).

    Returns
    -------
    lam_all : (n, r)  nonnegative, descending per row (padded with zeros)
    trace_all : (n,)  ``tr(Sigma_i)`` (exact from bias–variance identity)
    erank_all : (n,)   effective rank ``exp(-sum p_k log p_k)`` on normalized lam
    """
    T_csr = T.tocsr()
    U = np.asarray(U, dtype=np.float64)
    n, d = U.shape
    r = int(min(max(1, r), d))

    mu = np.asarray(T_csr @ U, dtype=np.float64)
    u_sq_row = np.sum(U * U, axis=1)
    trace_all = np.asarray(T_csr @ u_sq_row, dtype=np.float64) - np.sum(mu * mu, axis=1)
    trace_all = np.maximum(trace_all, 0.0)

    lam_all = np.zeros((n, r), dtype=np.float64)

    for i in range(n):
        start, end = int(T_csr.indptr[i]), int(T_csr.indptr[i + 1])
        if end <= start:
            continue
        j_idx = T_csr.indices[start:end]
        w_ij = T_csr.data[start:end]
        delta = U[j_idx] - mu[i]
        sw = np.sqrt(np.maximum(w_ij, 0.0))
        a_mat = delta * sw[:, None]
        k_nbr, d_loc = a_mat.shape
        rank_svd = min(r, k_nbr, d_loc)
        if rank_svd <= 0:
            continue
        try:
            if k_nbr <= d_loc:
                s = np.linalg.svd(a_mat, compute_uv=False, full_matrices=False)
            else:
                gram = a_mat @ a_mat.T
                s2 = np.linalg.eigvalsh(gram)
                s2 = np.sort(s2)[::-1]
                s = np.sqrt(np.maximum(s2, 0.0))
        except np.linalg.LinAlgError:
            continue
        lam_all[i, :rank_svd] = (s[:rank_svd] ** 2)

    eps_t = 1e-30
    erank_all = np.zeros(n, dtype=np.float64)
    for i in range(n):
        t_tr = float(trace_all[i])
        if t_tr < eps_t:
            erank_all[i] = 1.0
            continue
        p = lam_all[i] / t_tr
        p = p[p > 1e-15]
        if p.size == 0:
            erank_all[i] = 1.0
            continue
        erank_all[i] = float(np.exp(-np.sum(p * np.log(p))))

    return lam_all, trace_all, erank_all


def fiber_cov_summary(
    lam_all: np.ndarray,
    trace_all: np.ndarray,
    erank_all: np.ndarray,
) -> Dict[str, float]:
    """Aggregate scalars from ``fiber_covariance_eigenvalues`` (prefix ``fc_``)."""
    lam_all = np.asarray(lam_all, dtype=np.float64)
    lam1 = lam_all[:, 0] if lam_all.shape[1] > 0 else np.zeros(lam_all.shape[0])
    lam_r = lam_all[:, -1] if lam_all.shape[1] > 0 else np.zeros(lam_all.shape[0])
    mass_r = np.sum(lam_all, axis=1)
    tr_safe = np.maximum(trace_all, 1e-30)
    frac_top_r = mass_r / tr_safe
    return {
        "fc_trace_mean": float(trace_all.mean()),
        "fc_trace_median": float(np.median(trace_all)),
        "fc_trace_std": float(trace_all.std()),
        "fc_trace_q05": float(np.quantile(trace_all, 0.05)),
        "fc_trace_q95": float(np.quantile(trace_all, 0.95)),
        "fc_lambda1_mean": float(lam1.mean()),
        "fc_lambda1_median": float(np.median(lam1)),
        "fc_lambda_r_mean": float(lam_r.mean()),
        "fc_lambda_r_median": float(np.median(lam_r)),
        "fc_erank_mean": float(erank_all.mean()),
        "fc_erank_median": float(np.median(erank_all)),
        "fc_erank_std": float(erank_all.std()),
        "fc_frobenius_mean": float(np.sqrt(np.sum(lam_all**2, axis=1)).mean()),
        "fc_spectral_mass_top_r_mean": float(np.mean(frac_top_r)),
        "fc_cond_proxy_median": float(np.nanmedian(np.where(lam_r > 1e-20, lam1 / lam_r, np.nan))),
    }


def _fiber_cov_run_diagnostics(
    *,
    n: int,
    d_u: int,
    r_eff: int,
    schedules: Tuple[FiberCovSchedule, ...],
    alg3_Q: float,
    lam_principal: np.ndarray,
    lam_all: np.ndarray,
    trace_all: np.ndarray,
) -> Dict[str, Any]:
    """Extra JSON/CSV-friendly diagnostics for runners (not redundant with ``fc_summary`` floats)."""
    lam_p = np.asarray(lam_principal, dtype=np.float64).ravel()
    out: Dict[str, Any] = {
        "fiber_n": int(n),
        "fiber_d_U": int(d_u),
        "fiber_r": int(r_eff),
        "fiber_schedules": ",".join(schedules),
        "alg3_Q_rank_r": float(alg3_Q),
        "lambda_principal_max": float(lam_p.max()) if lam_p.size else float("nan"),
        "lambda_principal_min": float(lam_p.min()) if lam_p.size else float("nan"),
        "fc_lam_col_mean_max": float(np.max(np.nanmean(lam_all, axis=0))) if lam_all.size else float("nan"),
        "fc_trace_min": float(np.min(trace_all)) if trace_all.size else float("nan"),
        "fc_trace_max": float(np.max(trace_all)) if trace_all.size else float("nan"),
    }
    for j in range(min(8, lam_p.size)):
        out[f"lambda_principal_{j + 1}"] = float(lam_p[j])
    return out


def _amplitude_schedule(
    lam: np.ndarray,
    schedule: FiberCovSchedule,
    *,
    reg: float = 1e-8,
) -> np.ndarray:
    lam = np.asarray(lam, dtype=np.float64)
    lam_safe = np.maximum(lam, reg)
    if schedule == "uniform":
        return np.ones_like(lam)
    if schedule == "whitened":
        return 1.0 / np.sqrt(lam_safe)
    if schedule == "frobenius":
        return np.sqrt(lam_safe)
    raise ValueError(f"unknown amplitude schedule {schedule!r}")


def oracle_gap_fiber_cov_prior(
    T: csr_matrix,
    U: np.ndarray,
    r: int,
    schedule: FiberCovSchedule,
    *,
    pi: Optional[np.ndarray] = None,
    maxiter: int = 2000,
    principal_device: Optional[str] = None,
    principal_blas_threads: Optional[int] = None,
    lam_all: Optional[np.ndarray] = None,
    vecs: Optional[np.ndarray] = None,
    lam_principal: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Oracle gap with fiber-covariance-adaptive amplitudes on fixed principal modes."""
    n = int(T.shape[0])
    r = int(min(max(1, r), n - 1))

    if vecs is None or lam_principal is None:
        _, _, lam_principal, vecs = oracle_gap_principal_modes(
            T,
            r,
            pi=None,
            a=None,
            maxiter=maxiter,
            principal_device=principal_device,
            principal_blas_threads=principal_blas_threads,
        )

    vecs = np.asarray(vecs, dtype=np.float64)[:, :r]
    lam_principal = np.asarray(lam_principal, dtype=np.float64)[:r]

    if lam_all is None:
        lam_all, _, _ = fiber_covariance_eigenvalues(T, U, r)
    lam_all = np.asarray(lam_all, dtype=np.float64)[:, :r]

    if pi is None:
        pi_arr = np.ones(r, dtype=np.float64) / r
    else:
        pi_arr = np.asarray(pi, dtype=np.float64).ravel()[:r]
        pi_arr = pi_arr / max(float(pi_arr.sum()), 1e-12)

    ell = np.zeros(n, dtype=np.float64)
    for j in range(r):
        phi_j = vecs[:, j]
        _, ell_j_base = oracle_gap_rank_one(T, phi_j, a=1.0)
        a_ij = _amplitude_schedule(lam_all[:, j], schedule)
        ell += pi_arr[j] * a_ij * ell_j_base

    return float(np.mean(ell)), ell


@dataclass
class FiberCovPriorResult:
    """Scalar and local summaries for fiber-covariance prior schedules."""

    Q_uniform: float
    Q_whitened: float
    Q_frobenius: float
    local_stats_uniform: Dict[str, float]
    local_stats_whitened: Dict[str, float]
    local_stats_frobenius: Dict[str, float]
    fc_summary: Dict[str, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def compute_fiber_cov_prior_metrics(
    T: csr_matrix,
    U: np.ndarray,
    r: int,
    *,
    principal_maxiter: int = 2000,
    principal_device: Optional[str] = None,
    principal_blas_threads: Optional[int] = None,
    vecs: Optional[np.ndarray] = None,
    lam_principal: Optional[np.ndarray] = None,
    schedules: Tuple[FiberCovSchedule, ...] = ("uniform", "whitened", "frobenius"),
) -> FiberCovPriorResult:
    """
    Run fiber-covariance prior scores (single principal solve unless eigenpairs passed).

    ``diagnostics`` merges spectrum/run metadata with ``fc_summary`` (scalar) entries
    duplicated under ``diag_fc_*`` for flat CSV export alongside ``fiber_*`` keys.
    """
    n = int(T.shape[0])
    d_u = int(np.asarray(U).shape[1])
    r_eff = int(min(max(1, r), n - 1))

    if vecs is None or lam_principal is None:
        alg3_Q, _, lam_principal, vecs = oracle_gap_principal_modes(
            T,
            r_eff,
            pi=None,
            a=None,
            maxiter=principal_maxiter,
            principal_device=principal_device,
            principal_blas_threads=principal_blas_threads,
        )
    else:
        vecs = np.asarray(vecs, dtype=np.float64)[:, :r_eff]
        lam_principal = np.asarray(lam_principal, dtype=np.float64)[:r_eff]
        ell_acc = np.zeros(n, dtype=np.float64)
        for j in range(r_eff):
            _, ej = oracle_gap_rank_one(T, vecs[:, j], a=1.0)
            ell_acc += ej / r_eff
        alg3_Q = float(np.mean(ell_acc))

    vecs = np.asarray(vecs, dtype=np.float64)[:, :r_eff]
    lam_principal = np.asarray(lam_principal, dtype=np.float64)[:r_eff]

    lam_all, trace_all, erank_all = fiber_covariance_eigenvalues(T, U, r_eff)
    fc_sum = fiber_cov_summary(lam_all, trace_all, erank_all)

    qs: Dict[str, Tuple[float, np.ndarray]] = {}
    for sch in schedules:
        q_i, ell_i = oracle_gap_fiber_cov_prior(
            T,
            U,
            r_eff,
            sch,
            lam_all=lam_all,
            vecs=vecs,
            lam_principal=lam_principal,
            maxiter=principal_maxiter,
            principal_device=principal_device,
            principal_blas_threads=principal_blas_threads,
        )
        qs[sch] = (q_i, ell_i)

    qu, ell_u = qs["uniform"] if "uniform" in qs else (float("nan"), np.full(n, np.nan))
    qw, ell_w = qs["whitened"] if "whitened" in qs else (float("nan"), np.full(n, np.nan))
    qf, ell_f = qs["frobenius"] if "frobenius" in qs else (float("nan"), np.full(n, np.nan))

    run_diag = _fiber_cov_run_diagnostics(
        n=n,
        d_u=d_u,
        r_eff=r_eff,
        schedules=schedules,
        alg3_Q=float(alg3_Q),
        lam_principal=lam_principal,
        lam_all=lam_all,
        trace_all=trace_all,
    )
    diag_flat: Dict[str, Any] = {**run_diag}
    for k, v in fc_sum.items():
        diag_flat[f"diag_{k}"] = v

    return FiberCovPriorResult(
        Q_uniform=float(qu),
        Q_whitened=float(qw),
        Q_frobenius=float(qf),
        local_stats_uniform=local_score_stats(ell_u, name="fc_uniform_local"),
        local_stats_whitened=local_score_stats(ell_w, name="fc_whitened_local"),
        local_stats_frobenius=local_score_stats(ell_f, name="fc_frobenius_local"),
        fc_summary=fc_sum,
        diagnostics=diag_flat,
    )


__all__: List[str] = [
    "FiberCovPriorResult",
    "FiberCovSchedule",
    "compute_fiber_cov_prior_metrics",
    "fiber_covariance_eigenvalues",
    "fiber_cov_summary",
    "oracle_gap_fiber_cov_prior",
]
