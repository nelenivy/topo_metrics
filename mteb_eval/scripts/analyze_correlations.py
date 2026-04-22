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
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.corr_utils import (
    build_correlation_views_vs_target,
    compute_summary,
    compute_summary_grouped,
    ensure_task_type,
    estimate_sign,
    metric_columns,
)

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
    mcols = metric_columns(df)
    df_valid = ensure_task_type(df.dropna(subset=["mteb_score"]).copy())
    df_valid["mteb_score"] = pd.to_numeric(df_valid["mteb_score"], errors="coerce")
    return build_correlation_views_vs_target(
        df_valid, mcols, "mteb_score", per_model_col="model_name"
    )


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
    mcols = metric_columns(df)
    df_valid = ensure_task_type(df.dropna(subset=["mteb_score"]).copy())
    df_valid["mteb_score"] = df_valid["mteb_score"].astype(float)

    rows = []
    for col in mcols:
        sub = df_valid.dropna(subset=[col]).copy()
        signs = estimate_sign(
            sub,
            col,
            "mteb_score",
            scope=sign_estimation,
            group_cols=("task_name", "model_name"),
        )
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
    metric_cols = metric_columns(df)
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
