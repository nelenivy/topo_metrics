"""
Shared helpers for correlation and selection-quality scripts:

- ``analyze_correlations.py`` — unary rows in ``master_results.csv``
- ``analyze_pairwise_correlations.py`` — ordered-pair tables + constructed
  ``metric(U)-metric(V)`` columns vs ``delta_mteb`` / selection-quality (script
  lives under ``scripts/``; reuses ``build_correlation_views_vs_target``,
  ``estimate_sign`` with ``group_cols=(task_name, model_U)``, and summaries here).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def metric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("metric_") and c != "metric_error"]


def ensure_task_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "task_type" not in out.columns:
        out["task_type"] = "unknown"
    else:
        out["task_type"] = out["task_type"].fillna("unknown").astype(str)
    return out


def safe_corr(x, y) -> Tuple[float, float, float, float]:
    """Returns (spearman_r, spearman_p, pearson_r, pearson_p) safely."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    try:
        s_r, s_p = spearmanr(x, y, nan_policy="omit")
        p_r, p_p = pearsonr(x, y)
        return float(s_r), float(s_p), float(p_r), float(p_p)
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")


def fisher_z_mean(corrs: List[float]) -> float:
    clipped = np.clip(corrs, -0.9999, 0.9999)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def estimate_sign(
    df: pd.DataFrame,
    metric_col: str,
    target_col: str,
    scope: str = "per_task",
    group_cols: Tuple[str, ...] = ("task_name", "model_name"),
) -> pd.Series:
    """
    Direction of ``metric_col`` w.r.t. ``target_col``.

    scope='global'  — single sign from global Spearman correlation
    scope='per_task' — sign within each ``group_cols`` group (e.g. task+model or task+model_U)
    """
    sub = df.dropna(subset=[metric_col, target_col]).copy()
    if len(sub) == 0:
        return pd.Series(1, index=df.index)

    if scope == "global":
        r, *_ = safe_corr(sub[metric_col].values, sub[target_col].values)
        sign = 1 if (np.isnan(r) or r >= 0) else -1
        return pd.Series(sign, index=sub.index)

    def _group_sign(group: pd.DataFrame) -> float:
        r, *_ = safe_corr(group[metric_col].values, group[target_col].values)
        return 1.0 if (np.isnan(r) or r >= 0) else -1.0

    rows_sign: List[dict] = []
    for _, grp in sub.groupby(list(group_cols), sort=False):
        row = {k: grp[k].iloc[0] for k in group_cols}
        row["sign"] = _group_sign(grp)
        rows_sign.append(row)
    sign_df = pd.DataFrame(rows_sign)
    merged = sub.merge(sign_df, on=list(group_cols), how="left")
    return merged["sign"]


def build_correlation_views_vs_target(
    df: pd.DataFrame,
    metric_cols: List[str],
    target_col: str,
    *,
    per_model_col: str = "model_name",
) -> Dict[str, pd.DataFrame]:
    """
    Correlate each column in ``metric_cols`` with ``target_col`` at several granularities.

    Expects ``df`` to already contain ``task_type`` and numeric ``target_col``.
    """
    df_valid = ensure_task_type(df.dropna(subset=[target_col]).copy())
    df_valid[target_col] = df_valid[target_col].astype(float)

    rows_all: List[dict] = []
    rows_per_task: List[dict] = []
    rows_per_model: List[dict] = []
    rows_per_dataset: List[dict] = []
    rows_per_tt_pooled: List[dict] = []

    for col in metric_cols:
        sub = df_valid.dropna(subset=[col])
        if len(sub) == 0:
            continue

        s_r, s_p, p_r, p_p = safe_corr(sub[col].values, sub[target_col].values)
        rows_all.append(
            dict(
                metric=col,
                spearman_r=s_r,
                spearman_p=s_p,
                pearson_r=p_r,
                pearson_p=p_p,
                n=len(sub),
            )
        )

        task_corrs: List[float] = []
        for _, grp in sub.groupby("task_name"):
            r, *_ = safe_corr(grp[col].values, grp[target_col].values)
            if not np.isnan(r):
                task_corrs.append(r)
        mean_r = fisher_z_mean(task_corrs) if task_corrs else float("nan")
        rows_per_task.append(
            dict(
                metric=col,
                spearman_r=mean_r,
                spearman_p=float("nan"),
                pearson_r=float("nan"),
                pearson_p=float("nan"),
                n_datasets=len(task_corrs),
            )
        )

        if per_model_col in sub.columns:
            model_corrs: List[float] = []
            for _, grp in sub.groupby(per_model_col):
                r, *_ = safe_corr(grp[col].values, grp[target_col].values)
                if not np.isnan(r):
                    model_corrs.append(r)
            mean_rm = fisher_z_mean(model_corrs) if model_corrs else float("nan")
            rows_per_model.append(
                dict(
                    metric=col,
                    spearman_r=mean_rm,
                    spearman_p=float("nan"),
                    pearson_r=float("nan"),
                    pearson_p=float("nan"),
                    n_models=len(model_corrs),
                )
            )

        for task_name, grp in sub.groupby("task_name"):
            tt = grp["task_type"].iloc[0]
            s_r2, s_p2, p_r2, p_p2 = safe_corr(grp[col].values, grp[target_col].values)
            rows_per_dataset.append(
                dict(
                    metric=col,
                    task_name=task_name,
                    dataset=task_name,
                    task_type=tt,
                    spearman_r=s_r2,
                    spearman_p=s_p2,
                    pearson_r=p_r2,
                    pearson_p=p_p2,
                    n=len(grp),
                )
            )

        for task_type, grp in sub.groupby("task_type"):
            s_r3, s_p3, p_r3, p_p3 = safe_corr(grp[col].values, grp[target_col].values)
            rows_per_tt_pooled.append(
                dict(
                    metric=col,
                    task_type=task_type,
                    spearman_r=s_r3,
                    spearman_p=s_p3,
                    pearson_r=p_r3,
                    pearson_p=p_p3,
                    n=len(grp),
                )
            )

    per_ds_df = pd.DataFrame(rows_per_dataset)
    rows_tt_fisher: List[dict] = []
    if len(per_ds_df):
        for (metric, task_type), grp in per_ds_df.groupby(["metric", "task_type"]):
            corrs = [float(x) for x in grp["spearman_r"].values if not np.isnan(x)]
            mean_r = fisher_z_mean(corrs) if corrs else float("nan")
            rows_tt_fisher.append(
                dict(
                    metric=metric,
                    task_type=task_type,
                    spearman_r=mean_r,
                    n_datasets=len(corrs),
                )
            )

    out: Dict[str, pd.DataFrame] = {
        "all": pd.DataFrame(rows_all),
        "per_task": pd.DataFrame(rows_per_task),
        "per_dataset": per_ds_df,
        "per_task_type_pooled": pd.DataFrame(rows_per_tt_pooled),
        "per_task_type_fisher_mean": pd.DataFrame(rows_tt_fisher),
    }
    if per_model_col in df_valid.columns:
        out["per_model"] = pd.DataFrame(rows_per_model)
    else:
        out["per_model"] = pd.DataFrame()
    return out


def compute_summary(sq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, grp in sq.groupby("metric"):
        rows.append(
            dict(
                metric=metric,
                mean_gap=grp["gap"].mean(),
                median_gap=grp["gap"].median(),
                mean_selected_rank=grp["selected_rank"].mean(),
                top1_accuracy=float((grp["selected_rank"] == 1).mean()),
                top3_accuracy=float((grp["selected_rank"] <= 3).mean()),
                n_groups=len(grp),
            )
        )
    return pd.DataFrame(rows).sort_values("mean_gap")


def compute_summary_grouped(sq: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in sq.columns:
        return pd.DataFrame()
    rows = []
    for (metric, gval), grp in sq.groupby(["metric", group_col]):
        row = dict(
            metric=metric,
            mean_gap=grp["gap"].mean(),
            median_gap=grp["gap"].median(),
            mean_selected_rank=grp["selected_rank"].mean(),
            top1_accuracy=float((grp["selected_rank"] == 1).mean()),
            top3_accuracy=float((grp["selected_rank"] <= 3).mean()),
            n_groups=len(grp),
        )
        row[group_col] = gval
        rows.append(row)
    sort_keys = [group_col, "mean_gap"]
    return pd.DataFrame(rows).sort_values(sort_keys)
