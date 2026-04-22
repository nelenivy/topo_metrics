#!/usr/bin/env python3
"""
Pairwise oracle-gap + fiber-cov headline scores vs MTEB (correlations + anchor-U selection).

Joins:
  - oracle_gap_fiber/oracle_gap_fiber_pairwise.csv (prefixed ``fiber__``)
  - oracle_gap_fiber/oracle_gap_pairwise.csv (prefixed ``ogpair__``)
with ``master_results.csv`` for ``mteb_score`` and unary ``metric_*`` columns.

**Metrics only** (no bandwidth, ``diag_*``, ``T_*``, fiber summaries, etc.):

- Fiber: ``Q_fc_uniform``, ``Q_fc_whitened``, ``Q_fc_frobenius``, ``alg3_Q_rank_r``
  (same as ``_emit_fiber_pair_row`` in ``run_oracle_gap_fiber_consensus.py``).
- Oracle-gap pair: ``alg2_Q_mean``, ``alg3_Q_rank_r``, ``alg2_Q_mode*``,
  ``lambda_consensus_*``, ``lambda_principal_*`` (same as ``_emit_pair`` in
  ``run_oracle_gap_consensus.py``; excludes ``eps_hat``, ``diag_*``, dimensions).
- Constructed: ``metric(U) - metric(V)`` only for ``metric_*`` columns from master
  (``metric_columns`` / ``corr_utils``).

Writes under ``<run_dir>/pairwise_correlations/`` (default):
  - correlation_vs_delta/  (vs ``delta_mteb = mteb_score_U - mteb_score_V``)
  - correlation_vs_max_mteb/ (vs ``max(mteb_score_U, mteb_score_V)``)
  - selection_quality.csv, summary/
  - run_info.json
"""

from __future__ import annotations

import argparse
import re
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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

MERGE_KEYS: Tuple[str, ...] = ("task_name", "model_U", "model_V", "pooling", "layer_spec")

# Headline scores written by run_oracle_gap_fiber_consensus._emit_fiber_pair_row (not T_*, fc_*, fiber_*).
_FIBER_METRIC_BASES: Tuple[str, ...] = (
    "Q_fc_uniform",
    "Q_fc_whitened",
    "Q_fc_frobenius",
    "alg3_Q_rank_r",
)

# Headline scores from run_oracle_gap_consensus._emit_pair (not eps_hat, diag_*, n, d_*).
_OGPAIR_METRIC_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"^alg2_Q_mean$"),
    re.compile(r"^alg3_Q_rank_r$"),
    re.compile(r"^alg2_Q_mode\d+$"),
    re.compile(r"^lambda_consensus_\d+$"),
    re.compile(r"^lambda_principal_\d+$"),
)


def _ogpair_base_is_metric(base: str) -> bool:
    return any(p.fullmatch(base) for p in _OGPAIR_METRIC_PATTERNS)


def _load_prefixed_csv(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {c: f"{prefix}{c}" for c in df.columns if c not in MERGE_KEYS}
    out = df.rename(columns=ren)
    for k in MERGE_KEYS:
        if k not in out.columns:
            raise ValueError(f"{path} missing required column {k!r}")
    return out


def _merge_pairwise_sources(
    fiber_path: Path,
    og_path: Path,
    *,
    require_both: bool,
) -> pd.DataFrame:
    fab = _load_prefixed_csv(fiber_path, "fiber__")
    og = _load_prefixed_csv(og_path, "ogpair__")
    how: str = "inner" if require_both else "outer"
    pair = fab.merge(og, on=list(MERGE_KEYS), how=how)
    logger.info(
        "Merged pairwise sources (%s): %d rows (fiber rows=%d og rows=%d)",
        how,
        len(pair),
        len(fab),
        len(og),
    )
    return pair


def _master_side(
    master: pd.DataFrame,
    *,
    model_col_out: str,
    score_suffix: str,
    metric_suffix: str,
) -> pd.DataFrame:
    """One row per (task_name, model, pooling, layer_spec) with mteb_score and metric_* renamed."""
    mcols = metric_columns(master)
    base = ["task_name", "task_type", "model_name", "pooling", "layer_spec", "mteb_score"]
    keep = [c for c in base if c in master.columns] + [c for c in mcols if c in master.columns]
    sub = master[keep].copy()
    sub["mteb_score"] = pd.to_numeric(sub["mteb_score"], errors="coerce")
    for c in mcols:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    ren: Dict[str, str] = {"model_name": model_col_out, "mteb_score": f"mteb_score{score_suffix}"}
    for c in mcols:
        if c in sub.columns:
            ren[c] = f"{c}{metric_suffix}"
    out = sub.rename(columns=ren)
    return out


def _join_master_scores_and_metrics(
    pair: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    m_u = _master_side(master, model_col_out="model_U", score_suffix="_U", metric_suffix="_U")
    m_v = _master_side(master, model_col_out="model_V", score_suffix="_V", metric_suffix="_V")
    m_v = m_v[[c for c in m_v.columns if c != "task_type"]]

    out = pair.merge(
        m_u,
        on=["task_name", "model_U", "pooling", "layer_spec"],
        how="left",
    )
    out = out.merge(
        m_v,
        on=["task_name", "model_V", "pooling", "layer_spec"],
        how="left",
    )
    return out


def _add_constructed_metric_diffs(df: pd.DataFrame, master_metric_cols: List[str]) -> pd.DataFrame:
    pieces: Dict[str, pd.Series] = {}
    for c in master_metric_cols:
        u, v = f"{c}_U", f"{c}_V"
        if u in df.columns and v in df.columns:
            pieces[f"constructed__{c}"] = df[u] - df[v]
    if not pieces:
        return df
    return pd.concat([df, pd.DataFrame(pieces, index=df.index)], axis=1)


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    su = pd.to_numeric(out["mteb_score_U"], errors="coerce")
    sv = pd.to_numeric(out["mteb_score_V"], errors="coerce")
    out["delta_mteb"] = su - sv
    out["max_mteb_uv"] = np.maximum(su, sv)
    return out


def _pairwise_metric_columns(
    df: pd.DataFrame,
    *,
    master_metric_names: Sequence[str],
) -> List[str]:
    """
    Only headline pairwise scores (fiber + oracle-gap) and ``constructed__metric_*``.
    Excludes diagnostics (``diag_*``, ``T_*``, ``fc_*``, ``fiber_*``, ``eps_hat``, etc.).
    """
    out: List[str] = []
    for base in _FIBER_METRIC_BASES:
        c = f"fiber__{base}"
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    for c in df.columns:
        if not c.startswith("ogpair__"):
            continue
        base = c[len("ogpair__") :]
        if not _ogpair_base_is_metric(base):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    for m in master_metric_names:
        c = f"constructed__{m}"
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return sorted(set(out))


def _write_correlation_bundle(
    views: Dict[str, pd.DataFrame],
    corr_dir: Path,
    *,
    per_model_filename: str = "per_anchor_U.csv",
) -> None:
    corr_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "all": "all.csv",
        "per_task": "per_task.csv",
        "per_model": per_model_filename,
        "per_dataset": "per_dataset.csv",
        "per_task_type_pooled": "per_task_type_pooled.csv",
        "per_task_type_fisher_mean": "per_task_type_fisher_mean.csv",
    }
    for key, fname in mapping.items():
        path = corr_dir / fname
        views[key].to_csv(path, index=False)
        logger.info("  Saved %s/%s", corr_dir.name, fname)


def compute_selection_quality_pairwise_anchor_u(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    *,
    sign_estimation: str,
    min_vs: int,
) -> pd.DataFrame:
    """
    Anchor-U selection: group (task_name, model_U, pooling, layer_spec).

    oracle_score = max over the group of max(mteb_score_U, mteb_score_V) per row.
    Sign from Spearman(metric, mteb_score_V) within (task_name, model_U).
    Pick partner row maximizing sign * metric; selected_score = mteb_score_V there.
    """
    group_anchor = ("task_name", "model_U", "pooling", "layer_spec")
    rows: List[Dict[str, Any]] = []

    for col in metric_cols:
        sub = df.dropna(subset=[col, "mteb_score_V", "mteb_score_U"]).copy()
        if len(sub) == 0:
            continue
        signs = estimate_sign(
            sub,
            col,
            "mteb_score_V",
            scope=sign_estimation,
            group_cols=("task_name", "model_U"),
        )
        sub = sub.copy()
        sig = pd.Series(signs).reindex(sub.index)
        sub["_sign"] = sig.fillna(1.0).astype(float)

        for grp_key, grp in sub.groupby(list(group_anchor), sort=False):
            if len(grp) < min_vs:
                continue
            oracle_score = float(
                np.nanmax(np.maximum(grp["mteb_score_U"].to_numpy(), grp["mteb_score_V"].to_numpy()))
            )
            sign = int(np.sign(grp["_sign"].mean()))
            if sign == 0:
                sign = 1
            best_idx = (sign * grp[col]).idxmax()
            selected_score = float(grp.loc[best_idx, "mteb_score_V"])
            gap = float(oracle_score - selected_score)
            rank = int((grp["mteb_score_V"] > selected_score).sum()) + 1
            task_type = str(grp["task_type"].iloc[0]) if "task_type" in grp.columns else "unknown"

            rows.append(
                dict(
                    metric=col,
                    task_name=grp_key[0],
                    task_type=task_type,
                    model_U=grp_key[1],
                    pooling=grp_key[2],
                    layer_spec=grp_key[3],
                    oracle_score=oracle_score,
                    selected_score=selected_score,
                    gap=gap,
                    selected_rank=rank,
                    n_configs=len(grp),
                )
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate pairwise oracle-gap / fiber metrics with MTEB (and constructed unary diffs).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory containing master_results.csv and oracle_gap_fiber/ (optional if paths explicit).",
    )
    parser.add_argument(
        "--master-results",
        type=str,
        default=None,
        help="Path to master_results.csv (default: RUN_DIR/master_results.csv).",
    )
    parser.add_argument(
        "--fiber-pairwise",
        type=str,
        default=None,
        help="oracle_gap_fiber_pairwise.csv path (default: RUN_DIR/oracle_gap_fiber/oracle_gap_fiber_pairwise.csv).",
    )
    parser.add_argument(
        "--og-pairwise",
        type=str,
        default=None,
        help="oracle_gap_pairwise.csv under fiber dir (default: RUN_DIR/oracle_gap_fiber/oracle_gap_pairwise.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root (default: RUN_DIR/pairwise_correlations).",
    )
    parser.add_argument(
        "--require-both-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inner-merge fiber and ogpair rows (default: true). If false, outer-merge.",
    )
    parser.add_argument(
        "--sign-estimation",
        default="per_task",
        choices=["global", "per_task"],
    )
    parser.add_argument(
        "--min-vs",
        type=int,
        default=2,
        help="Minimum number of partner rows (distinct V) per anchor group for selection_quality.",
    )
    args = parser.parse_args()

    if not args.run_dir and not (args.master_results and args.fiber_pairwise and args.og_pairwise):
        parser.error("Provide --run-dir or all of --master-results, --fiber-pairwise, --og-pairwise")

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    master_path = (
        Path(args.master_results).resolve()
        if args.master_results
        else (run_dir / "master_results.csv").resolve()  # type: ignore[union-attr]
    )
    base_for_defaults = run_dir if run_dir is not None else master_path.parent
    fiber_path = (
        Path(args.fiber_pairwise).resolve()
        if args.fiber_pairwise
        else (base_for_defaults / "oracle_gap_fiber" / "oracle_gap_fiber_pairwise.csv").resolve()
    )
    og_path = (
        Path(args.og_pairwise).resolve()
        if args.og_pairwise
        else (base_for_defaults / "oracle_gap_fiber" / "oracle_gap_pairwise.csv").resolve()
    )
    out_root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (base_for_defaults / "pairwise_correlations").resolve()
    )

    for p, label in ((master_path, "master_results"), (fiber_path, "fiber pairwise"), (og_path, "og pairwise")):
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    os.makedirs(out_root, exist_ok=True)
    logger.info("Output: %s", out_root)

    pair = _merge_pairwise_sources(fiber_path, og_path, require_both=bool(args.require_both_sources))
    master = pd.read_csv(master_path)
    logger.info("Loaded master_results %s (%d rows)", master_path, len(master))

    mcols_master = metric_columns(master)
    wide = _join_master_scores_and_metrics(pair, master)
    wide = _add_constructed_metric_diffs(wide, mcols_master)
    wide = _add_targets(wide)
    wide = ensure_task_type(wide)

    # Rows usable for correlation (both MTEB sides)
    corr_base = wide.dropna(subset=["mteb_score_U", "mteb_score_V", "delta_mteb"]).copy()
    metric_cols = _pairwise_metric_columns(corr_base, master_metric_names=mcols_master)
    logger.info("Pairwise + constructed metric columns: %d", len(metric_cols))

    n_join_u = wide["mteb_score_U"].notna().sum()
    n_join_v = wide["mteb_score_V"].notna().sum()
    n_both = wide["mteb_score_U"].notna() & wide["mteb_score_V"].notna()
    logger.info(
        "Join coverage: mteb_score_U non-null %d / %d, V %d / %d, both %d",
        int(n_join_u),
        len(wide),
        int(n_join_v),
        len(wide),
        int(n_both.sum()),
    )

    corr_delta = out_root / "correlation_vs_delta"
    corr_max = out_root / "correlation_vs_max_mteb"
    summary_dir = out_root / "summary"

    for col in ("delta_mteb", "max_mteb_uv"):
        corr_base[col] = pd.to_numeric(corr_base[col], errors="coerce")

    for c in metric_cols:
        corr_base[c] = pd.to_numeric(corr_base[c], errors="coerce")

    logger.info("Correlation vs delta_mteb …")
    views_d = build_correlation_views_vs_target(
        corr_base,
        metric_cols,
        "delta_mteb",
        per_model_col="model_U",
    )
    _write_correlation_bundle(views_d, corr_delta, per_model_filename="per_anchor_U.csv")

    logger.info("Correlation vs max_mteb_uv …")
    views_m = build_correlation_views_vs_target(
        corr_base.dropna(subset=["max_mteb_uv"]),
        metric_cols,
        "max_mteb_uv",
        per_model_col="model_U",
    )
    _write_correlation_bundle(views_m, corr_max, per_model_filename="per_anchor_U.csv")

    logger.info("Selection quality (anchor U) …")
    sq = compute_selection_quality_pairwise_anchor_u(
        corr_base,
        metric_cols,
        sign_estimation=args.sign_estimation,
        min_vs=int(args.min_vs),
    )
    sq_path = out_root / "selection_quality.csv"
    sq.to_csv(sq_path, index=False)
    logger.info("  Saved %s (%d rows)", sq_path, len(sq))

    summary_dir.mkdir(parents=True, exist_ok=True)
    if len(sq):
        overall = compute_summary(sq)
        overall.to_csv(summary_dir / "overall.csv", index=False)
        by_tt = compute_summary_grouped(sq, "task_type")
        if len(by_tt):
            by_tt.to_csv(summary_dir / "by_task_type.csv", index=False)
        by_ds = compute_summary_grouped(sq, "task_name")
        if len(by_ds):
            by_ds.to_csv(summary_dir / "by_dataset.csv", index=False)

    info = {
        "master_results": str(master_path),
        "fiber_pairwise": str(fiber_path),
        "og_pairwise": str(og_path),
        "output_dir": str(out_root),
        "require_both_sources": bool(args.require_both_sources),
        "sign_estimation": args.sign_estimation,
        "min_vs": int(args.min_vs),
        "n_pairwise_merged": int(len(pair)),
        "n_after_master_join": int(len(wide)),
        "n_corr_rows": int(len(corr_base)),
        "n_metric_cols": len(metric_cols),
        "argv": sys.argv,
    }
    (out_root / "run_info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Done.")


if __name__ == "__main__":
    main()
