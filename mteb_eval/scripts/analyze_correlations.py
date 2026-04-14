#!/usr/bin/env python3
"""
analyze_correlations.py
Correlation analysis between unsupervised embedding metrics and MTEB scores.

Outputs (default: ``<run_dir>/correlations/`` next to ``master_results.csv``)
----------------------------------------------------------------------------
Directory layout::

    correlations/
      correlation/
        all.csv                      — pooled over all rows
        per_task.csv                 — Fisher–z mean of within-task correlations
        per_model.csv                — Fisher–z mean of within-model correlations
        per_dataset.csv              — one Spearman/Pearson per (metric, task_name)
        per_task_type_pooled.csv     — correlation within each MTEB task_type slice
        per_task_type_fisher_mean.csv — Fisher–z mean of per-dataset r's per type
      selection_quality.csv          — full detail (includes task_type)
      summary/
        overall.csv
        by_task_type.csv             — mean_gap etc. averaged per task_type
        by_dataset.csv               — mean_gap etc. averaged per task (dataset)

Usage
-----
python scripts/analyze_correlations.py \
    --results-csv ./results/my_run/master_results.csv \
    --sign-estimation per_task

Omit ``--results-csv`` to use the **newest** ``master_results.csv`` found under
``--search-root`` (default ``./results``, recursive).

Omit ``--output-dir`` to write under ``<run_dir>/correlations/``.
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def find_latest_master_results(search_root: str) -> Path:
    """Newest ``master_results.csv`` under ``search_root`` (by file mtime)."""
    root = Path(search_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"--search-root does not exist: {root}")
    candidates = list(root.rglob("master_results.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No master_results.csv found under {root}. "
            "Use a separate --output-dir per run, or pass --results-csv explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ══════════════════════════════════════════════════════════════════════════ #
#  Helpers                                                                   #
# ══════════════════════════════════════════════════════════════════════════ #

def _metric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("metric_") and c != "metric_error"]


def _ensure_task_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "task_type" not in out.columns:
        out["task_type"] = "unknown"
    else:
        out["task_type"] = out["task_type"].fillna("unknown").astype(str)
    return out


def _safe_corr(x, y) -> Tuple[float, float, float, float]:
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


def _fisher_z_mean(corrs: List[float]) -> float:
    """Fisher-z-transform mean of a list of correlations."""
    clipped = np.clip(corrs, -0.9999, 0.9999)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def _estimate_sign(
    df: pd.DataFrame, metric_col: str, scope: str = "per_task"
) -> pd.Series:
    """
    Estimate the direction of metric_col w.r.t. mteb_score.

    scope='global'   — single sign from global Spearman correlation
    scope='per_task' — majority sign within each (task_name, model_name) group
    """
    sub = df.dropna(subset=[metric_col, "mteb_score"]).copy()

    if scope == "global":
        r, *_ = _safe_corr(sub[metric_col].values, sub["mteb_score"].values)
        sign = 1 if (np.isnan(r) or r >= 0) else -1
        return pd.Series(sign, index=sub.index)

    def _group_sign(group):
        r, *_ = _safe_corr(group[metric_col].values, group["mteb_score"].values)
        return 1 if (np.isnan(r) or r >= 0) else -1

    signs = sub.groupby(["task_name", "model_name"]).apply(_group_sign)
    merged = sub[["task_name", "model_name"]].join(
        signs.rename("sign"), on=["task_name", "model_name"]
    )
    return merged["sign"]


# ══════════════════════════════════════════════════════════════════════════ #
#  1. Correlation analysis                                                   #
# ══════════════════════════════════════════════════════════════════════════ #

def build_correlation_views(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Several correlation tables at different granularities.

    - all: pooled Spearman/Pearson over all rows
    - per_task: Fisher–z mean of Spearman r computed within each task_name (dataset)
    - per_model: Fisher–z mean of Spearman r within each model_name
    - per_dataset: one Spearman/Pearson per (metric, task_name)
    - per_task_type_pooled: Spearman/Pearson on all rows with a given task_type
    - per_task_type_fisher_mean: Fisher–z mean of per_dataset Spearman r's, per task_type
    """
    metric_cols = _metric_columns(df)
    df_valid = _ensure_task_type(df.dropna(subset=["mteb_score"]).copy())
    df_valid["mteb_score"] = df_valid["mteb_score"].astype(float)

    rows_all: List[dict] = []
    rows_per_task: List[dict] = []
    rows_per_model: List[dict] = []
    rows_per_dataset: List[dict] = []
    rows_per_tt_pooled: List[dict] = []

    for col in metric_cols:
        sub = df_valid.dropna(subset=[col])
        if len(sub) == 0:
            continue

        # all
        s_r, s_p, p_r, p_p = _safe_corr(sub[col].values, sub["mteb_score"].values)
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

        # per_task (Fisher across datasets)
        task_corrs: List[float] = []
        for _, grp in sub.groupby("task_name"):
            r, *_ = _safe_corr(grp[col].values, grp["mteb_score"].values)
            if not np.isnan(r):
                task_corrs.append(r)
        mean_r = _fisher_z_mean(task_corrs) if task_corrs else float("nan")
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

        # per_model
        model_corrs: List[float] = []
        for _, grp in sub.groupby("model_name"):
            r, *_ = _safe_corr(grp[col].values, grp["mteb_score"].values)
            if not np.isnan(r):
                model_corrs.append(r)
        mean_r = _fisher_z_mean(model_corrs) if model_corrs else float("nan")
        rows_per_model.append(
            dict(
                metric=col,
                spearman_r=mean_r,
                spearman_p=float("nan"),
                pearson_r=float("nan"),
                pearson_p=float("nan"),
                n_models=len(model_corrs),
            )
        )

        # per_dataset (one correlation per MTEB task / dataset name)
        for task_name, grp in sub.groupby("task_name"):
            tt = grp["task_type"].iloc[0]
            s_r, s_p, p_r, p_p = _safe_corr(grp[col].values, grp["mteb_score"].values)
            rows_per_dataset.append(
                dict(
                    metric=col,
                    task_name=task_name,
                    dataset=task_name,
                    task_type=tt,
                    spearman_r=s_r,
                    spearman_p=s_p,
                    pearson_r=p_r,
                    pearson_p=p_p,
                    n=len(grp),
                )
            )

        # per_task_type pooled (all configs in that type together)
        for task_type, grp in sub.groupby("task_type"):
            s_r, s_p, p_r, p_p = _safe_corr(grp[col].values, grp["mteb_score"].values)
            rows_per_tt_pooled.append(
                dict(
                    metric=col,
                    task_type=task_type,
                    spearman_r=s_r,
                    spearman_p=s_p,
                    pearson_r=p_r,
                    pearson_p=p_p,
                    n=len(grp),
                )
            )

    per_ds_df = pd.DataFrame(rows_per_dataset)
    rows_tt_fisher: List[dict] = []
    if len(per_ds_df):
        for (metric, task_type), grp in per_ds_df.groupby(["metric", "task_type"]):
            corrs = [float(x) for x in grp["spearman_r"].values if not np.isnan(x)]
            mean_r = _fisher_z_mean(corrs) if corrs else float("nan")
            rows_tt_fisher.append(
                dict(
                    metric=metric,
                    task_type=task_type,
                    spearman_r=mean_r,
                    n_datasets=len(corrs),
                )
            )

    return {
        "all": pd.DataFrame(rows_all),
        "per_task": pd.DataFrame(rows_per_task),
        "per_model": pd.DataFrame(rows_per_model),
        "per_dataset": per_ds_df,
        "per_task_type_pooled": pd.DataFrame(rows_per_tt_pooled),
        "per_task_type_fisher_mean": pd.DataFrame(rows_tt_fisher),
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  2. Selection quality                                                      #
# ══════════════════════════════════════════════════════════════════════════ #

def compute_selection_quality(
    df: pd.DataFrame,
    sign_estimation: str = "per_task",
    min_configs: int = 3,
) -> pd.DataFrame:
    """
    For each (metric_col, task_name, model_name, pooling) group:
      oracle_score, selected_score, gap, selected_rank, n_configs.

    Retrieval variants (_corpus, _queries, _combined) are separate metric columns.
    """
    metric_cols = _metric_columns(df)
    df_valid = _ensure_task_type(df.dropna(subset=["mteb_score"]).copy())
    df_valid["mteb_score"] = df_valid["mteb_score"].astype(float)

    rows = []
    for col in metric_cols:
        sub = df_valid.dropna(subset=[col]).copy()
        signs = _estimate_sign(sub, col, scope=sign_estimation)
        sub = sub.copy()
        sub["_sign"] = signs.values if len(signs) == len(sub) else 1

        for grp_key, grp in sub.groupby(["task_name", "model_name", "pooling"]):
            if len(grp) < min_configs:
                continue

            oracle_score = grp["mteb_score"].max()
            n_configs = len(grp)
            task_type = grp["task_type"].iloc[0]

            sign = int(np.sign(grp["_sign"].mean()))
            if sign == 0:
                sign = 1

            best_idx = (sign * grp[col]).idxmax()
            selected_score = float(grp.loc[best_idx, "mteb_score"])
            gap = float(oracle_score - selected_score)

            rank = int((grp["mteb_score"] > selected_score).sum()) + 1

            rows.append(
                dict(
                    metric=col,
                    task_name=grp_key[0],
                    task_type=task_type,
                    model_name=grp_key[1],
                    pooling=grp_key[2],
                    oracle_score=float(oracle_score),
                    selected_score=selected_score,
                    gap=gap,
                    selected_rank=rank,
                    n_configs=n_configs,
                )
            )

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════ #
#  3. Metric summary                                                         #
# ══════════════════════════════════════════════════════════════════════════ #

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
    """Aggregate selection metrics per (metric, group_col) e.g. task_type or task_name."""
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


def write_correlation_bundle(views: Dict[str, pd.DataFrame], corr_dir: Path) -> None:
    corr_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "all": "all.csv",
        "per_task": "per_task.csv",
        "per_model": "per_model.csv",
        "per_dataset": "per_dataset.csv",
        "per_task_type_pooled": "per_task_type_pooled.csv",
        "per_task_type_fisher_mean": "per_task_type_fisher_mean.csv",
    }
    for key, fname in mapping.items():
        path = corr_dir / fname
        views[key].to_csv(path, index=False)
        logger.info(f"  Saved correlation/{fname}")


# ══════════════════════════════════════════════════════════════════════════ #
#  CLI                                                                       #
# ══════════════════════════════════════════════════════════════════════════ #

def main():
    parser = argparse.ArgumentParser(
        description="Correlation analysis: unsupervised metrics vs MTEB"
    )
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Path to master_results.csv. If omitted, use newest under --search-root.",
    )
    parser.add_argument(
        "--search-root",
        default="results",
        help="When --results-csv is omitted: directory tree to search for master_results.csv (recursive).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Base directory for correlation/, selection_quality.csv, summary/. "
        "Default: <directory of master_results.csv>/correlations",
    )
    parser.add_argument(
        "--sign-estimation", default="per_task", choices=["global", "per_task"]
    )
    parser.add_argument(
        "--min-configs",
        type=int,
        default=3,
        help="Min configs per group for selection quality",
    )
    args = parser.parse_args()

    if args.results_csv:
        csv_path = Path(args.results_csv).resolve()
    else:
        csv_path = find_latest_master_results(args.search_root)
        logger.info(f"Using latest master_results.csv (by mtime): {csv_path}")

    if args.output_dir is None:
        output_dir = csv_path.parent / "correlations"
    else:
        output_dir = Path(args.output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    logger.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  {len(df)} rows, {len(df.columns)} columns")

    df["mteb_score"] = pd.to_numeric(df["mteb_score"], errors="coerce")
    metric_cols = _metric_columns(df)
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"  {len(metric_cols)} metric columns found")

    corr_dir = output_dir / "correlation"
    summary_dir = output_dir / "summary"

    logger.info("Computing correlations…")
    views = build_correlation_views(df)
    write_correlation_bundle(views, corr_dir)

    logger.info("Computing selection quality…")
    sq_df = compute_selection_quality(
        df, sign_estimation=args.sign_estimation, min_configs=args.min_configs
    )
    sq_path = output_dir / "selection_quality.csv"
    sq_df.to_csv(sq_path, index=False)
    logger.info(f"  Saved {sq_path} ({len(sq_df)} rows)")

    summary_dir.mkdir(parents=True, exist_ok=True)
    if len(sq_df):
        overall = compute_summary(sq_df)
        overall.to_csv(summary_dir / "overall.csv", index=False)
        logger.info(f"  Saved {summary_dir / 'overall.csv'}")

        by_tt = compute_summary_grouped(sq_df, "task_type")
        if len(by_tt):
            by_tt.to_csv(summary_dir / "by_task_type.csv", index=False)
            logger.info(f"  Saved {summary_dir / 'by_task_type.csv'}")

        by_ds = compute_summary_grouped(sq_df, "task_name")
        if len(by_ds):
            by_ds.to_csv(summary_dir / "by_dataset.csv", index=False)
            logger.info(f"  Saved {summary_dir / 'by_dataset.csv'}")

        logger.info("\n" + "═" * 60)
        logger.info("Top-10 metrics by mean_gap (overall, lower = better selection):")
        top_cols = [
            "metric",
            "mean_gap",
            "top1_accuracy",
            "top3_accuracy",
            "mean_selected_rank",
            "n_groups",
        ]
        with pd.option_context(
            "display.max_colwidth", 45, "display.float_format", "{:.4f}".format
        ):
            print(overall[top_cols].head(10).to_string(index=False))

    logger.info("\n" + "═" * 60)
    logger.info("Global Spearman (top-10 by |r|, pooled all):")
    all_df = views["all"]
    if len(all_df):
        global_corr = (
            all_df.assign(abs_r=lambda d: d["spearman_r"].abs())
            .sort_values("abs_r", ascending=False)
            .head(10)[["metric", "spearman_r", "pearson_r", "n"]]
        )
        with pd.option_context(
            "display.max_colwidth", 45, "display.float_format", "{:.4f}".format
        ):
            print(global_corr.to_string(index=False))


if __name__ == "__main__":
    main()
