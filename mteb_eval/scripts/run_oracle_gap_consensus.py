#!/usr/bin/env python3
"""
Consensus Laplacian prior + pairwise oracle-gap (Algorithms 2 & 3).

Defaults match ``run_unsup_eval.py`` layout:
  - ``--output-dir`` default ``./results/unsup_eval``
  - HDF5 cache default ``OUTPUT_DIR/embedding_cache`` (or ``REUSE_RUN_DIR/embedding_cache``)

Metrics and diagnostics are written under ``OUTPUT_DIR / METRICS_SUBDIR /``
(default ``.../oracle_gap/``), including ``oracle_gap_pairwise_agg.csv`` /
``oracle_gap_by_task.csv`` (mean, std, quantiles over tasks) and
``oracle_gap_local_stats_agg.csv`` / ``oracle_gap_local_by_task.csv`` for pooled
local score columns. This script does **not** read or write
``master_results.csv`` (that file is only produced by ``run_unsup_eval.py``).
HDF5 embedding cache defaults to ``OUTPUT_DIR/embedding_cache`` or, with
``--reuse-run-dir``, ``REUSE_RUN_DIR/embedding_cache``.

Logging: ``--log-level`` overrides ``-v`` counts; structured lines use
``[oracle_gap] step=…``. Optional tqdm bars: ``--progress`` / ``--no-progress``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import logging
import multiprocessing as mp
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

logger = logging.getLogger("oracle_gap")

_unsup_path = ROOT / "scripts" / "run_unsup_eval.py"
_spec = importlib.util.spec_from_file_location("run_unsup_eval_mod", _unsup_path)
_run_unsup = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_run_unsup)
extract_all_texts = _run_unsup.extract_all_texts
load_tasks = _run_unsup.load_tasks
_default_embedding_weight_dtype = _run_unsup._default_embedding_weight_dtype
_infer_embedding_batch_size = _run_unsup._infer_embedding_batch_size
_normalize_torch_dtype_str = _run_unsup._normalize_torch_dtype_str

from src.aggregated_encoder import LayerEncoder  # noqa: E402
from src.cache_manager import LayerEmbeddingStore  # noqa: E402
from src.embedding_extractor import extract_embedding_matrix  # noqa: E402
from src.layer_spec import build_layer_specs  # noqa: E402
from src.model_sets import MODEL_SET_MAP  # noqa: E402
from src.oracle_gap_pairwise import (  # noqa: E402
    PairwiseOracleGapResult,
    compute_pairwise_oracle_gap,
)
from src.pooling_rules import pooling_supported, skip_reason  # noqa: E402
from src.task_sets import TASK_SET_MAP  # noqa: E402

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise SystemExit("run_oracle_gap_consensus requires pandas") from e

try:
    from tqdm import tqdm as _tqdm_cls
except ImportError:  # pragma: no cover
    _tqdm_cls = None


def _pbar(
    iterable,
    *,
    enabled: bool,
    **kwargs: Any,
):
    """tqdm when enabled and installed; otherwise plain iteration."""
    if not enabled or _tqdm_cls is None:
        return iterable
    return _tqdm_cls(iterable, **kwargs)

# Fork-pool workers inherit these (set in main before Pool); do not pickle large arrays.
_OG_EMBS: Dict[str, np.ndarray] = {}
_OG_CL: List[np.ndarray] = []
_OG_KW: Dict[str, Any] = {}


def _worker_compute_pair(pair: Tuple[str, str]) -> Tuple[str, str, PairwiseOracleGapResult]:
    mu, mv = pair
    res = compute_pairwise_oracle_gap(_OG_EMBS[mu], _OG_EMBS[mv], _OG_CL, **_OG_KW)
    return mu, mv, res


def _pick_layer_spec(spec_name: str, n_layers: int):
    for s in build_layer_specs(n_layers):
        if s.name == spec_name:
            return s
    names = [s.name for s in build_layer_specs(n_layers)]
    raise ValueError(
        f"Unknown layer_spec {spec_name!r} for n_layers={n_layers}. "
        f"Available (sample): {names[:24]} …"
    )


def _aligned_texts(stores: Dict[str, Any], all_texts: List[str]) -> List[str]:
    common = None
    for _m, view in stores.items():
        keys = set(view._text_index.keys())
        common = keys if common is None else (common & keys)
    if common is None:
        return []
    return [t for t in all_texts if t in common]


def _load_store_matrix(
    *,
    model_name: str,
    pooling: str,
    texts: List[str],
    task_name: str,
    cache_dir: Path,
    batch_size: int,
    device: str,
    trust_remote_code: bool,
    torch_dtype: Optional[str],
    spec,
    n_layers: int,
) -> Tuple[Any, np.ndarray]:
    store = LayerEmbeddingStore(
        model_name=model_name,
        n_layers=n_layers,
        poolings=[pooling],
        batch_size=batch_size,
        device=device,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    store.precompute_or_load(texts=texts, dataset_name=task_name, split_name="test")
    view = store.as_pooling(pooling)
    mat = extract_embedding_matrix(view, texts, spec, n_layers)
    return store, np.asarray(mat, dtype=np.float32)


def _flatten_stats(prefix: str, stats: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in stats.items()}


def _quantile_agg(q: float):
    """Finite-sample quantile for pandas NamedAgg (avoids lambda closure bugs)."""

    def _inner(series: "pd.Series") -> float:
        a = np.asarray(series, dtype=np.float64)
        a = a[np.isfinite(a)]
        if not a.size:
            return float("nan")
        return float(np.quantile(a, q))

    return _inner


def _pandas_group_numeric_stats(
    df: "pd.DataFrame",
    keys: List[str],
    value_cols: List[str],
    *,
    over_suffix: str,
) -> "pd.DataFrame":
    """
    Per-group summaries of numeric columns: mean, std, min, q05…q95, max
    (same spirit as ``local_score_stats`` plus q10 / q90).

    Column names: ``{col}_mean{over_suffix}``, ``{col}_q05{over_suffix}``, …
    ``over_suffix`` is '' or e.g. ``_over_pairs`` for task-level exports.
    """
    if df.empty or not value_cols:
        return df.loc[:, list(keys)].drop_duplicates().reset_index(drop=True)
    spec: Dict[str, Any] = {}
    suf = over_suffix
    for c in value_cols:
        spec[f"{c}_mean{suf}"] = pd.NamedAgg(column=c, aggfunc="mean")
        spec[f"{c}_std{suf}"] = pd.NamedAgg(column=c, aggfunc="std")
        spec[f"{c}_min{suf}"] = pd.NamedAgg(column=c, aggfunc="min")
        spec[f"{c}_q05{suf}"] = pd.NamedAgg(column=c, aggfunc=_quantile_agg(0.05))
        spec[f"{c}_q10{suf}"] = pd.NamedAgg(column=c, aggfunc=_quantile_agg(0.10))
        spec[f"{c}_q25{suf}"] = pd.NamedAgg(column=c, aggfunc=_quantile_agg(0.25))
        spec[f"{c}_median{suf}"] = pd.NamedAgg(column=c, aggfunc=_quantile_agg(0.50))
        spec[f"{c}_q75{suf}"] = pd.NamedAgg(column=c, aggfunc=_quantile_agg(0.75))
        spec[f"{c}_q90{suf}"] = pd.NamedAgg(column=c, aggfunc=_quantile_agg(0.90))
        spec[f"{c}_q95{suf}"] = pd.NamedAgg(column=c, aggfunc=_quantile_agg(0.95))
        spec[f"{c}_max{suf}"] = pd.NamedAgg(column=c, aggfunc="max")
    return df.groupby(keys, as_index=False, dropna=False).agg(**spec)


def _bandwidth_cv_curve_for_export(res: PairwiseOracleGapResult) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-row adaptive CV stores ``cv_scores`` / ``eps_grids`` as (n, M).
    For a single diagnostic curve, aggregate over points (nanmean along axis 0).
    """
    cv = np.asarray(res.cv_scores, dtype=np.float64)
    eg = np.asarray(res.eps_grids, dtype=np.float64)
    if cv.ndim == 2:
        cv_m = np.nanmean(cv, axis=0)
    else:
        cv_m = cv.ravel()
    if eg.ndim == 2:
        eps_m = np.nanmean(eg, axis=0)
    else:
        eps_m = eg.ravel()
    m = min(int(cv_m.shape[0]), int(eps_m.shape[0]))
    return eps_m[:m], cv_m[:m]


def _write_run_info(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


class _IncrementalDictWriter:
    """Append dict rows to CSV; header written on first row (fieldnames grow rarely)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fieldnames: Optional[List[str]] = None

    def write_row(self, row: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        else:
            for k in row:
                if k not in self.fieldnames:
                    self.fieldnames.append(k)
        exists_nonempty = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
                restval="",
            )
            if not exists_nonempty:
                w.writeheader()
            out = {}
            for k in self.fieldnames:
                v = row.get(k, "")
                if v is None:
                    out[k] = ""
                elif isinstance(v, (np.floating, float)) and not np.isfinite(v):
                    out[k] = ""
                elif isinstance(v, np.floating):
                    out[k] = float(v)
                elif isinstance(v, np.integer):
                    out[k] = int(v)
                else:
                    out[k] = v
            w.writerow(out)


def _setup_logging(*, verbose: int, log_level: Optional[str]) -> None:
    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        force=True,
    )


def _stage(
    step: str,
    message: str,
    *,
    task: Optional[str] = None,
    pooling: Optional[str] = None,
) -> None:
    """Structured INFO line so logs are easy to grep (step=…)."""
    bits = [f"step={step}"]
    if task is not None:
        bits.append(f"task={task}")
    if pooling is not None:
        bits.append(f"pooling={pooling}")
    tail = " ".join(bits)
    logger.info("[oracle_gap] %s — %s", tail, message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise oracle-gap (Alg. 2 + 3) with consensus graph modes — "
        "mirrors run_unsup_eval paths and flags where applicable.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        metavar="MODEL",
        help="HF model ids. If omitted, use --model-set (default: core).",
    )
    parser.add_argument(
        "--model-set",
        default="core",
        choices=["core", "standard", "full"],
        help="Predefined model list when --models is omitted.",
    )
    parser.add_argument(
        "--task-set",
        default="core",
        choices=list(TASK_SET_MAP.keys()),
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=None,
        choices=[
            "Classification",
            "Clustering",
            "PairClassification",
            "Reranking",
            "Retrieval",
            "STS",
            "Summarization",
        ],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--poolings",
        nargs="+",
        default=["mean"],
        choices=["mean", "cls", "last_token"],
        help="Same keyword as run_unsup_eval; each pooling is a separate metrics run.",
    )
    parser.add_argument(
        "--layer-spec",
        default="last_1",
        help="LayerSpec name from build_layer_specs (default: last_1).",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/unsup_eval",
        help="Same root as run_unsup_eval (default: ./results/unsup_eval).",
    )
    parser.add_argument(
        "--metrics-subdir",
        default="oracle_gap",
        help="Subfolder under --output-dir for oracle-gap CSV + diagnostics.",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        default=None,
        help="HDF5 cache dir (default: OUTPUT_DIR/embedding_cache or REUSE_RUN_DIR/embedding_cache).",
    )
    parser.add_argument(
        "--reuse-run-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory from a prior run_unsup_eval / gpu_proxy job (contains embedding_cache/). "
            "If set and --embedding-cache-dir is omitted, embeddings are read/written under "
            "DIR/embedding_cache."
        ),
    )
    parser.add_argument(
        "--no-incremental-csv",
        action="store_true",
        default=False,
        help=(
            "Write oracle_gap_pairwise/local and diagnostics CSVs only once per stage "
            "(legacy). Default is to flush each pairwise result to disk immediately."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Default: cuda if torch.cuda.is_available() else cpu.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--auto-embedding-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Same heuristic as run_unsup_eval (CUDA + low precision).",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        default=False,
    )
    parser.add_argument("--torch-dtype", default=None, metavar="DTYPE")
    parser.add_argument("--r-consensus", type=int, default=8)
    parser.add_argument("--r-principal", type=int, default=8)
    parser.add_argument("--knn-k", type=int, default=24)
    parser.add_argument("--bandwidth-grid-m", type=int, default=24)
    parser.add_argument(
        "--sigma-clip",
        type=float,
        default=3.0,
        help=(
            "Adaptive per-row fiber: zero Gaussian weights beyond "
            "max(kNN_k distance, sigma_clip * sigma_i) (default: 3)."
        ),
    )
    parser.add_argument(
        "--density-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Directed fiber kernel W: divide each nonzero by "
            "q_out[i]^α · q_in[j]^α before row-normalizing to T (default: on). "
            "α is --density-alpha."
        ),
    )
    parser.add_argument(
        "--density-alpha",
        type=float,
        default=1.0,
        help="Exponent α for directed marginal rescale (default: 1). α≈0 skips rescale.",
    )
    parser.add_argument(
        "--smooth-bw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adaptive: median-smooth selected per-row bandwidth over kNN (default: on).",
    )
    parser.add_argument(
        "--fiber-kernel",
        choices=["gaussian", "epanechnikov"],
        default="gaussian",
        help=(
            "Ignored for the adaptive Gaussian+cutoff path (kept for CLI compatibility). "
            "Legacy scalar-eps code would use gaussian vs epanechnikov."
        ),
    )
    parser.add_argument(
        "--principal-maxiter",
        type=int,
        default=12000,
        help="ARPACK maxiter for Algorithm 3 (eigsh on I-T^T T); retries/LOBPCG also used.",
    )
    parser.add_argument(
        "--principal-device",
        type=str,
        default=None,
        metavar="DEV",
        help=(
            "Algorithm 3 matvec device: unset or cpu = SciPy CPU (default). "
            "cuda / cuda:0 / cuda:1 = CuPy CSR on that GPU if cupy is installed "
            "(pip install cupy-cuda12x matching CUDA); else falls back to CPU."
        ),
    )
    parser.add_argument(
        "--principal-blas-threads",
        type=int,
        default=None,
        metavar="N",
        help=(
            "For the CPU SciPy path only: temporarily set OMP/MKL/OpenBLAS thread "
            "env vars during eigsh. Default caps at min(32, cpu_count). Use 0 to leave env unchanged."
        ),
    )
    parser.add_argument("--min-n", type=int, default=40)
    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="Cap texts (same prefix for all models).",
    )
    parser.add_argument(
        "--pair-workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel ordered-pair jobs (Unix fork; shares embedding RAM). "
        "Not like run_unsup_eval layer-spec workers; safe with read-only arrays.",
    )
    parser.add_argument(
        "--skip-alg2",
        action="store_true",
        help="Skip Algorithm 2 (consensus rank-one modes).",
    )
    parser.add_argument(
        "--skip-alg3",
        action="store_true",
        help="Skip Algorithm 3 (principal lost modes).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Logging: -v INFO, -vv DEBUG (ignored if --log-level is set).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set logging level explicitly (overrides -v count).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="tqdm progress for tasks, per-model I/O, and pairwise jobs (default: on).",
    )
    args = parser.parse_args()
    _setup_logging(verbose=int(args.verbose), log_level=args.log_level)

    if args.device is None:
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    models = list(args.models) if args.models else list(MODEL_SET_MAP[args.model_set])
    if len(models) < 2:
        parser.error("Need at least two models for pairwise metrics.")

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_dir = (out_root / str(args.metrics_subdir)).resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = metrics_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    incremental = not bool(args.no_incremental_csv)
    reuse_run_dir = Path(args.reuse_run_dir).resolve() if args.reuse_run_dir else None

    if args.embedding_cache_dir:
        cache_dir = Path(args.embedding_cache_dir).resolve()
    elif reuse_run_dir is not None:
        cache_dir = (reuse_run_dir / "embedding_cache").resolve()
    else:
        cache_dir = (out_root / "embedding_cache").resolve()

    torch_dtype = _normalize_torch_dtype_str(args.torch_dtype)
    if torch_dtype is None:
        torch_dtype = _default_embedding_weight_dtype(args.device)
    bs = int(args.batch_size)
    if bool(args.auto_embedding_batch):
        bs = _infer_embedding_batch_size(
            device=args.device,
            torch_dtype=torch_dtype,
            base=int(args.batch_size),
        )
    trust_remote_code = not bool(args.no_trust_remote_code)

    run_alg2 = not bool(args.skip_alg2)
    run_alg3 = not bool(args.skip_alg3)
    if not run_alg2 and not run_alg3:
        parser.error("Cannot skip both --skip-alg2 and --skip-alg3.")

    _write_run_info(
        metrics_dir / "run_info.json",
        {
            "argv": sys.argv,
            "note": (
                "Oracle-gap: does not read or write master_results.csv. "
                "Default incremental CSV flush per pair."
            ),
            "output_dir": str(out_root),
            "metrics_dir": str(metrics_dir),
            "embedding_cache_dir": str(cache_dir),
            "reuse_run_dir": str(reuse_run_dir) if reuse_run_dir else None,
            "incremental_csv": incremental,
            "models": models,
            "poolings": list(args.poolings),
            "layer_spec": args.layer_spec,
            "r_consensus": args.r_consensus,
            "r_principal": args.r_principal,
            "knn_k": args.knn_k,
            "bandwidth_grid_m": args.bandwidth_grid_m,
            "sigma_clip": float(args.sigma_clip),
            "density_normalize": bool(args.density_normalize),
            "density_alpha": float(args.density_alpha),
            "smooth_bw": bool(args.smooth_bw),
            "fiber_kernel": args.fiber_kernel,
            "principal_maxiter": args.principal_maxiter,
            "principal_device": args.principal_device,
            "principal_blas_threads": args.principal_blas_threads,
            "pair_workers": int(args.pair_workers),
            "run_alg2": run_alg2,
            "run_alg3": run_alg3,
            "platform": platform.platform(),
            "log_level": logging.getLevelName(logger.getEffectiveLevel()),
            "progress_bars": bool(args.progress),
        },
    )
    use_progress = bool(args.progress)
    logger.info("Output root %s | metrics %s | cache %s", out_root, metrics_dir, cache_dir)
    _stage(
        "init",
        f"device={args.device} batch_size={bs} log_level={logging.getLevelName(logger.getEffectiveLevel())} "
        f"progress_bars={use_progress}",
    )

    _stage("load_task_list", f"resolving tasks (task_set={args.task_set!r})")
    tasks = load_tasks(
        task_set=args.task_set,
        tasks=args.tasks,
        task_types=args.task_types,
        max_samples=args.max_samples,
    )
    _stage("load_task_list", f"loaded {len(tasks)} task(s)")

    pair_csv = metrics_dir / "oracle_gap_pairwise.csv"
    local_csv = metrics_dir / "oracle_gap_local_stats.csv"
    agg_csv = metrics_dir / "oracle_gap_pairwise_agg.csv"
    by_task_csv = metrics_dir / "oracle_gap_by_task.csv"
    local_agg_csv = metrics_dir / "oracle_gap_local_stats_agg.csv"
    local_by_task_csv = metrics_dir / "oracle_gap_local_by_task.csv"
    cv_all: List[Dict[str, Any]] = []
    pre_all: List[Dict[str, Any]] = []

    pair_rows: List[Dict[str, Any]] = []
    local_rows: List[Dict[str, Any]] = []

    diag_cv_path = diag_dir / "bandwidth_cv_curves.csv"
    diag_pre_path = diag_dir / "pair_preprocess.csv"
    cv_writer: Optional[_IncrementalDictWriter] = None
    pre_writer: Optional[_IncrementalDictWriter] = None
    if incremental:
        for pth in (diag_cv_path, diag_pre_path):
            if pth.exists():
                pth.unlink()
        cv_writer = _IncrementalDictWriter(diag_cv_path)
        pre_writer = _IncrementalDictWriter(diag_pre_path)

    for pooling in args.poolings:
        logger.info("=== Pooling %s ===", pooling)
        _stage("pooling_pass", f"starting {len(tasks)} task(s) for this pooling", pooling=pooling)
        pair_rows.clear()
        local_rows.clear()

        suffix = f"_{pooling}" if len(args.poolings) > 1 else ""
        p_out = pair_csv.with_name(pair_csv.stem + suffix + pair_csv.suffix)
        l_out = local_csv.with_name(local_csv.stem + suffix + local_csv.suffix)
        a_out = agg_csv.with_name(agg_csv.stem + suffix + agg_csv.suffix)
        t_out = by_task_csv.with_name(by_task_csv.stem + suffix + by_task_csv.suffix)
        la_out = local_agg_csv.with_name(local_agg_csv.stem + suffix + local_agg_csv.suffix)
        lt_out = local_by_task_csv.with_name(local_by_task_csv.stem + suffix + local_by_task_csv.suffix)

        pair_writer: Optional[_IncrementalDictWriter] = None
        local_writer: Optional[_IncrementalDictWriter] = None
        if incremental:
            for pth in (p_out, l_out):
                if pth.exists():
                    pth.unlink()
            pair_writer = _IncrementalDictWriter(p_out)
            local_writer = _IncrementalDictWriter(l_out)

        task_iter = _pbar(
            tasks,
            enabled=use_progress,
            desc=f"tasks [{pooling}]",
            unit="task",
            smoothing=0.0,
        )
        for task in task_iter:
            task_name = task.metadata.name
            if use_progress and hasattr(task_iter, "set_postfix"):
                task_iter.set_postfix(task=task_name[:40], refresh=False)
            try:
                _stage("mteb_load_data", "calling task.load_data()", task=task_name, pooling=pooling)
                task.load_data()
            except Exception as e:
                logger.warning("Skip task %s (load_data): %s", task_name, e)
                continue

            try:
                _stage("mteb_texts", "extract_all_texts()", task=task_name, pooling=pooling)
                all_texts = extract_all_texts(task)
            except Exception as e:
                logger.warning("Skip task %s (texts): %s", task_name, e)
                continue

            if args.max_n is not None and len(all_texts) > int(args.max_n):
                all_texts = list(all_texts[: int(args.max_n)])

            n_layers_by_model: Dict[str, int] = {}
            spec_by_model: Dict[str, Any] = {}
            for model_name in _pbar(
                models,
                enabled=use_progress,
                desc=f"probe [{task_name}]",
                unit="model",
                leave=False,
            ):
                if not pooling_supported(model_name, pooling):
                    logger.info(
                        "Skip model %s: pooling %s — %s",
                        model_name,
                        pooling,
                        skip_reason(model_name, pooling),
                    )
                    continue
                try:
                    probe = LayerEncoder(
                        model_name=model_name,
                        pooling="mean",
                        batch_size=1,
                        device=args.device,
                        use_cache=False,
                        trust_remote_code=trust_remote_code,
                        torch_dtype=torch_dtype,
                    )
                    nl = int(probe.num_layers)
                    del probe
                    gc.collect()
                except Exception as e:
                    logger.warning("Probe failed %s: %s", model_name, e)
                    continue
                try:
                    spec_by_model[model_name] = _pick_layer_spec(args.layer_spec, nl)
                except ValueError as e:
                    logger.warning("%s: %s", model_name, e)
                    continue
                n_layers_by_model[model_name] = nl

            active = [m for m in models if m in n_layers_by_model]
            if len(active) < 2:
                logger.warning("Task %s: fewer than 2 usable models; skip", task_name)
                continue

            _stage(
                "encoders_ready",
                f"{len(active)} model(s) after probe | corpus texts={len(all_texts)}",
                task=task_name,
                pooling=pooling,
            )

            stores: Dict[str, Any] = {}
            try:
                for model_name in _pbar(
                    active,
                    enabled=use_progress,
                    desc=f"cache [{task_name}]",
                    unit="model",
                    leave=False,
                ):
                    nl = n_layers_by_model[model_name]
                    spec = spec_by_model[model_name]
                    st, _mat = _load_store_matrix(
                        model_name=model_name,
                        pooling=pooling,
                        texts=all_texts,
                        task_name=task_name,
                        cache_dir=cache_dir,
                        batch_size=bs,
                        device=args.device,
                        trust_remote_code=trust_remote_code,
                        torch_dtype=torch_dtype,
                        spec=spec,
                        n_layers=nl,
                    )
                    stores[model_name] = st

                texts_use = _aligned_texts(
                    {m: s.as_pooling(pooling) for m, s in stores.items()},
                    all_texts,
                )
                n = len(texts_use)
                if n < int(args.min_n):
                    logger.warning("Task %s: only n=%d aligned texts; skip", task_name, n)
                    continue

                _stage(
                    "aligned_corpus",
                    f"n={n} aligned texts across models",
                    task=task_name,
                    pooling=pooling,
                )

                embs: Dict[str, np.ndarray] = {}
                consensus_list: List[np.ndarray] = []
                for model_name in _pbar(
                    active,
                    enabled=use_progress,
                    desc=f"matrices [{task_name}]",
                    unit="model",
                    leave=False,
                ):
                    view = stores[model_name].as_pooling(pooling)
                    spec = spec_by_model[model_name]
                    nl = n_layers_by_model[model_name]
                    X = extract_embedding_matrix(view, texts_use, spec, nl)
                    X = np.asarray(X, dtype=np.float32)
                    embs[model_name] = X
                    consensus_list.append(X)

                pairs: List[Tuple[str, str]] = [(u, v) for u in active for v in active]
                kw = dict(
                    r_consensus=int(args.r_consensus),
                    r_principal=int(args.r_principal),
                    knn_k=int(args.knn_k),
                    bandwidth_grid_M=int(args.bandwidth_grid_m),
                    principal_maxiter=int(args.principal_maxiter),
                    principal_device=args.principal_device,
                    principal_blas_threads=args.principal_blas_threads,
                    run_alg2=run_alg2,
                    run_alg3=run_alg3,
                    fiber_kernel=str(args.fiber_kernel),
                    sigma_clip=float(args.sigma_clip),
                    smooth_bw=bool(args.smooth_bw),
                    density_normalize=bool(args.density_normalize),
                    density_alpha=float(args.density_alpha),
                )

                eff_pw = max(1, int(args.pair_workers))
                # Fork pool shares read-only embedding arrays (copy-on-write). Non-Linux
                # start methods are unsafe or slow to pickle large arrays → Linux only.
                use_fork_pool = eff_pw > 1 and platform.system() == "Linux"
                if eff_pw > 1 and not use_fork_pool:
                    logger.warning(
                        "--pair-workers>1 uses fork+shared RAM (Linux only). "
                        "Using 1 worker on this platform."
                    )
                    eff_pw = 1

                _stage(
                    "pairwise_oracle_gap",
                    f"{len(pairs)} ordered pairs | pair_workers={eff_pw}",
                    task=task_name,
                    pooling=pooling,
                )

                def _emit_pair(
                    model_u: str,
                    model_v: str,
                    res: PairwiseOracleGapResult,
                    elapsed: float,
                    *,
                    record_seconds: bool,
                ) -> None:
                    row: Dict[str, Any] = {
                        "task_name": task_name,
                        "model_U": model_u,
                        "model_V": model_v,
                        "pooling": pooling,
                        "layer_spec": args.layer_spec,
                        "n": n,
                        "d_U": int(embs[model_u].shape[1]),
                        "d_V": int(embs[model_v].shape[1]),
                        "eps_hat": res.eps_hat,
                        "alg2_Q_mean": res.alg2_Q_mean,
                        "alg3_Q_rank_r": res.alg3_Q_rank_r,
                        "run_alg2": int(run_alg2),
                        "run_alg3": int(run_alg3),
                    }
                    if record_seconds:
                        row["seconds"] = float(elapsed)
                    for j, q in enumerate(res.alg2_Q_per_mode):
                        row[f"alg2_Q_mode{j + 1}"] = q
                    for j, lam in enumerate(res.lambdas_consensus):
                        row[f"lambda_consensus_{j + 1}"] = float(lam)
                    for j, lam in enumerate(res.lambdas_principal):
                        row[f"lambda_principal_{j + 1}"] = float(lam)
                    for k, v in res.diagnostics.items():
                        row[f"diag_{k}"] = v
                    pair_rows.append(row)
                    if incremental and pair_writer is not None:
                        pair_writer.write_row(row)

                    eps_curve, cv_curve = _bandwidth_cv_curve_for_export(res)
                    cv_batch = [
                        {
                            "task_name": task_name,
                            "model_U": model_u,
                            "model_V": model_v,
                            "pooling": pooling,
                            "layer_spec": args.layer_spec,
                            "grid_index": gi,
                            "eps": float(eps_v),
                            "cv_loss": float(cv_curve[gi]),
                            "cv_curve_agg": "mean_over_points",
                        }
                        for gi, eps_v in enumerate(eps_curve)
                    ]
                    if incremental and cv_writer is not None:
                        for d in cv_batch:
                            cv_writer.write_row(d)
                    else:
                        cv_all.extend(cv_batch)

                    pre_row = {
                        "task_name": task_name,
                        "model_U": model_u,
                        "model_V": model_v,
                        "pooling": pooling,
                        "layer_spec": args.layer_spec,
                        **{f"diag_{k}": v for k, v in res.diagnostics.items()},
                    }
                    if incremental and pre_writer is not None:
                        pre_writer.write_row(pre_row)
                    else:
                        pre_all.append(pre_row)

                    base_local = {
                        "task_name": task_name,
                        "model_U": model_u,
                        "model_V": model_v,
                        "pooling": pooling,
                        "layer_spec": args.layer_spec,
                    }
                    local_batch: List[Dict[str, Any]] = []
                    for j, st in enumerate(res.local_stats_alg2):
                        local_batch.append(
                            {
                                **base_local,
                                "mode": f"alg2_consensus_mode{j + 1}",
                                **_flatten_stats("", st),
                            }
                        )
                    if run_alg3 and res.local_stats_alg3:
                        local_batch.append(
                            {
                                **base_local,
                                "mode": "alg3_principal_rank_r",
                                **_flatten_stats("", res.local_stats_alg3),
                            }
                        )
                    for lr in local_batch:
                        local_rows.append(lr)
                        if incremental and local_writer is not None:
                            local_writer.write_row(lr)

                if eff_pw <= 1:
                    pair_loop = _pbar(
                        pairs,
                        enabled=use_progress,
                        desc=f"pairs [{task_name}]",
                        unit="pair",
                        leave=False,
                    )
                    for model_u, model_v in pair_loop:
                        if use_progress and hasattr(pair_loop, "set_postfix"):
                            pair_loop.set_postfix(
                                U=model_u[:18],
                                V=model_v[:18],
                                refresh=False,
                            )
                        logger.info(
                            "Task %s | pair %s → %s | n=%d | pooling=%s",
                            task_name,
                            model_u,
                            model_v,
                            n,
                            pooling,
                        )
                        t0 = time.perf_counter()
                        res = compute_pairwise_oracle_gap(
                            embs[model_u], embs[model_v], consensus_list, **kw
                        )
                        dt = time.perf_counter() - t0
                        logger.info("  done in %.2fs | alg2_Q_mean=%.6g alg3_Q=%.6g", dt, res.alg2_Q_mean, res.alg3_Q_rank_r)
                        _emit_pair(model_u, model_v, res, dt, record_seconds=True)
                else:
                    global _OG_EMBS, _OG_CL, _OG_KW
                    _OG_EMBS = embs
                    _OG_CL = consensus_list
                    _OG_KW = kw
                    logger.info(
                        "Task %s | %d pairs | fork pool workers=%d",
                        task_name,
                        len(pairs),
                        eff_pw,
                    )
                    ctx = mp.get_context("fork")
                    chunksize = max(1, len(pairs) // max(1, eff_pw * 8))
                    with ctx.Pool(processes=eff_pw) as pool:
                        imap_it = pool.imap(_worker_compute_pair, pairs, chunksize=chunksize)
                        for (mu, mv, res) in _pbar(
                            imap_it,
                            enabled=use_progress,
                            total=len(pairs),
                            desc=f"pairs [{task_name}]",
                            unit="pair",
                            leave=False,
                        ):
                            _emit_pair(mu, mv, res, 0.0, record_seconds=False)

                _stage("task_complete", "finished pairwise block for task", task=task_name, pooling=pooling)

            finally:
                for st in stores.values():
                    try:
                        del st
                    except Exception:
                        pass
                gc.collect()

        if not incremental:
            if pair_rows:
                pd.DataFrame(pair_rows).to_csv(p_out, index=False)
                logger.info("Wrote %s (%d rows)", p_out, len(pair_rows))
            else:
                logger.warning("No rows for pooling=%s; skip %s", pooling, p_out)

            if local_rows:
                pd.DataFrame(local_rows).to_csv(l_out, index=False)
                logger.info("Wrote %s (%d rows)", l_out, len(local_rows))
        else:
            if pair_rows:
                logger.info("Incremental pairwise CSV %s (%d rows)", p_out, len(pair_rows))
            else:
                logger.warning("No rows for pooling=%s; skip %s", pooling, p_out)
            if local_rows:
                logger.info("Incremental local stats %s (%d rows)", l_out, len(local_rows))

        if pair_rows:
            df = pd.DataFrame(pair_rows)
            keys = ["model_U", "model_V", "pooling", "layer_spec"]
            skip = set(keys) | {"task_name", "seconds", "run_alg2", "run_alg3"}
            num_cols = [
                c
                for c in df.columns
                if c not in skip and pd.api.types.is_numeric_dtype(df[c])
            ]
            _pandas_group_numeric_stats(df, keys, num_cols, over_suffix="").to_csv(a_out, index=False)
            logger.info("Wrote %s", a_out)

            _pandas_group_numeric_stats(
                df, ["task_name"], num_cols, over_suffix="_over_pairs"
            ).to_csv(t_out, index=False)
            logger.info("Wrote %s", t_out)

        if local_rows:
            ldf = pd.DataFrame(local_rows)
            lkeys = ["model_U", "model_V", "pooling", "layer_spec", "mode"]
            lskip = set(lkeys) | {"task_name"}
            lnum = [
                c
                for c in ldf.columns
                if c not in lskip and pd.api.types.is_numeric_dtype(ldf[c])
            ]
            if lnum:
                _pandas_group_numeric_stats(ldf, lkeys, lnum, over_suffix="").to_csv(la_out, index=False)
                logger.info("Wrote %s", la_out)
                _pandas_group_numeric_stats(
                    ldf, ["task_name"], lnum, over_suffix="_over_task"
                ).to_csv(lt_out, index=False)
                logger.info("Wrote %s", lt_out)

    if not incremental and cv_all:
        pd.DataFrame(cv_all).to_csv(diag_cv_path, index=False)
        logger.info("Wrote %s (%d rows)", diag_cv_path, len(cv_all))
    if not incremental and pre_all:
        pd.DataFrame(pre_all).to_csv(diag_pre_path, index=False)
        logger.info("Wrote %s (%d rows)", diag_pre_path, len(pre_all))
    logger.info("Diagnostics directory: %s", diag_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
