#!/usr/bin/env python3
"""
Pairwise fiber-covariance prior metrics (separate from global oracle_gap / unary).

Uses ``build_pairwise_adaptive_fiber_operator`` — same ``T`` as ``compute_pairwise_oracle_gap``.
Writes under ``OUTPUT_DIR / METRICS_SUBDIR /`` (default ``oracle_gap_fiber/``) with CSV names
distinct from ``run_oracle_gap_consensus.py`` and unary runners.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

logger = logging.getLogger("oracle_gap_fiber")

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
from src.oracle_gap_fiber_cov import (  # noqa: E402
    FiberCovPriorResult,
    FiberCovSchedule,
    compute_fiber_cov_prior_metrics,
)
from src.oracle_gap_pairwise import build_pairwise_adaptive_fiber_operator  # noqa: E402
from src.pooling_rules import pooling_supported, skip_reason  # noqa: E402
from src.task_sets import TASK_SET_MAP  # noqa: E402

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise SystemExit("run_oracle_gap_fiber_consensus requires pandas") from e

try:
    from tqdm import tqdm as _tqdm_cls
except ImportError:  # pragma: no cover
    _tqdm_cls = None


def _pbar(iterable, *, enabled: bool, **kwargs: Any):
    if not enabled or _tqdm_cls is None:
        return iterable
    return _tqdm_cls(iterable, **kwargs)


def _pick_layer_spec(spec_name: str, n_layers: int):
    for s in build_layer_specs(n_layers):
        if s.name == spec_name:
            return s
    names = [s.name for s in build_layer_specs(n_layers)]
    raise ValueError(f"Unknown layer_spec {spec_name!r} for n_layers={n_layers}. Sample: {names[:24]} …")


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


def _write_run_info(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _parse_fiber_schedules(s: Optional[str]) -> Tuple[FiberCovSchedule, ...]:
    if not s or not str(s).strip():
        return ("uniform", "whitened", "frobenius")
    out: List[FiberCovSchedule] = []
    for part in str(s).split(","):
        p = part.strip().lower()
        if p in ("uniform", "whitened", "frobenius"):
            out.append(p)  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown fiber schedule {part!r} (use uniform,whitened,frobenius)")
    return tuple(out) if out else ("uniform", "whitened", "frobenius")  # type: ignore[return-value]


class _IncrementalDictWriter:
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
            out: Dict[str, Any] = {}
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
    )


def _emit_fiber_pair_row(
    *,
    task_name: str,
    model_u: str,
    model_v: str,
    pooling: str,
    layer_spec: str,
    n: int,
    d_u: int,
    d_v: int,
    diag_t: Dict[str, Any],
    fc: FiberCovPriorResult,
    elapsed: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "task_name": task_name,
        "model_U": model_u,
        "model_V": model_v,
        "pooling": pooling,
        "layer_spec": layer_spec,
        "n": n,
        "d_U": d_u,
        "d_V": d_v,
        "seconds": float(elapsed),
        "Q_fc_uniform": fc.Q_uniform,
        "Q_fc_whitened": fc.Q_whitened,
        "Q_fc_frobenius": fc.Q_frobenius,
        "alg3_Q_rank_r": fc.diagnostics.get("alg3_Q_rank_r", ""),
    }
    for k, v in diag_t.items():
        row[f"T_{k}"] = v
    for k, v in fc.fc_summary.items():
        row[k] = v
    for k, v in fc.diagnostics.items():
        row[f"fiber_{k}"] = v
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Fiber-covariance oracle metrics (pairwise, separate CSV tree).")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--model-set", default="core", choices=["core", "standard", "full"])
    parser.add_argument("--task-set", default="core", choices=list(TASK_SET_MAP.keys()))
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--task-types", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--poolings", nargs="+", default=["mean"], choices=["mean", "cls", "last_token"])
    parser.add_argument("--layer-spec", default="last_1")
    parser.add_argument("--output-dir", default="./results/unsup_eval")
    parser.add_argument(
        "--metrics-subdir",
        default="oracle_gap_fiber",
        help="Subfolder under --output-dir (default oracle_gap_fiber; not oracle_gap / unary).",
    )
    parser.add_argument("--embedding-cache-dir", default=None)
    parser.add_argument("--reuse-run-dir", default=None)
    parser.add_argument("--no-incremental-csv", action="store_true", default=False)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--auto-embedding-batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-trust-remote-code", action="store_true", default=False)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--r-principal", type=int, default=8)
    parser.add_argument("--knn-k", type=int, default=24)
    parser.add_argument("--bandwidth-grid-m", type=int, default=24)
    parser.add_argument("--sigma-clip", type=float, default=3.0)
    parser.add_argument(
        "--density-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Directed marginal rescale on W before row-stochastic T (same as pairwise oracle_gap).",
    )
    parser.add_argument(
        "--density-alpha",
        type=float,
        default=1.0,
        help="Exponent α for q_out^α · q_in^α denominator (default: 1).",
    )
    parser.add_argument("--smooth-bw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--principal-maxiter", type=int, default=12000)
    parser.add_argument("--principal-device", type=str, default=None)
    parser.add_argument("--principal-blas-threads", type=int, default=None)
    parser.add_argument(
        "--fiber-schedules",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated subset of uniform,whitened,frobenius (default: all three).",
    )
    parser.add_argument("--min-n", type=int, default=40)
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--pair-workers", type=int, default=1)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--log-level", default=None, choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
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
        parser.error("Need at least two models.")

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
    use_progress = bool(args.progress)
    sched_tuple = _parse_fiber_schedules(args.fiber_schedules)

    _write_run_info(
        metrics_dir / "run_info_fiber.json",
        {
            "argv": sys.argv,
            "output_dir": str(out_root),
            "metrics_dir": str(metrics_dir),
            "embedding_cache_dir": str(cache_dir),
            "models": models,
            "poolings": list(args.poolings),
            "layer_spec": args.layer_spec,
            "r_principal": int(args.r_principal),
            "knn_k": int(args.knn_k),
            "bandwidth_grid_m": int(args.bandwidth_grid_m),
            "sigma_clip": float(args.sigma_clip),
            "density_normalize": bool(args.density_normalize),
            "density_alpha": float(args.density_alpha),
            "smooth_bw": bool(args.smooth_bw),
            "principal_maxiter": int(args.principal_maxiter),
            "principal_device": args.principal_device,
            "principal_blas_threads": args.principal_blas_threads,
            "fiber_schedules": list(sched_tuple),
            "pair_workers": int(args.pair_workers),
            "platform": platform.platform(),
            "log_level": logging.getLevelName(logger.getEffectiveLevel()),
            "progress_bars": use_progress,
        },
    )

    pair_csv = metrics_dir / "oracle_gap_fiber_pairwise.csv"
    local_csv = metrics_dir / "oracle_gap_fiber_local_stats.csv"
    diag_csv = diag_dir / "oracle_gap_fiber_per_pair_diagnostics.csv"

    tasks = load_tasks(
        task_set=args.task_set,
        tasks=args.tasks,
        task_types=args.task_types,
        max_samples=args.max_samples,
    )

    build_kw = dict(
        knn_k=int(args.knn_k),
        bandwidth_grid_M=int(args.bandwidth_grid_m),
        sigma_clip=float(args.sigma_clip),
        smooth_bw=bool(args.smooth_bw),
        density_normalize=bool(args.density_normalize),
        density_alpha=float(args.density_alpha),
    )
    met_kw = dict(
        principal_maxiter=int(args.principal_maxiter),
        principal_device=args.principal_device,
        principal_blas_threads=args.principal_blas_threads,
        schedules=sched_tuple,
    )

    for pooling in args.poolings:
        pair_rows: List[Dict[str, Any]] = []
        local_rows: List[Dict[str, Any]] = []
        diag_rows: List[Dict[str, Any]] = []

        suffix = f"_{pooling}" if len(args.poolings) > 1 else ""
        p_out = pair_csv.with_name(pair_csv.stem + suffix + pair_csv.suffix)
        l_out = local_csv.with_name(local_csv.stem + suffix + local_csv.suffix)
        d_out = diag_csv.with_name(diag_csv.stem + suffix + diag_csv.suffix)

        pair_writer = _IncrementalDictWriter(p_out) if incremental else None
        local_writer = _IncrementalDictWriter(l_out) if incremental else None
        diag_writer = _IncrementalDictWriter(d_out) if incremental else None
        if incremental:
            for pth in (p_out, l_out, d_out):
                if pth.exists():
                    pth.unlink()

        for task in _pbar(tasks, enabled=use_progress, desc=f"tasks [{pooling}]", unit="task"):
            task_name = task.metadata.name
            try:
                task.load_data()
            except Exception as e:
                logger.warning("Skip task %s (load_data): %s", task_name, e)
                continue
            try:
                all_texts = extract_all_texts(task)
            except Exception as e:
                logger.warning("Skip task %s (texts): %s", task_name, e)
                continue
            if args.max_n is not None and len(all_texts) > int(args.max_n):
                all_texts = list(all_texts[: int(args.max_n)])

            n_layers_by_model: Dict[str, int] = {}
            spec_by_model: Dict[str, Any] = {}
            for model_name in models:
                if not pooling_supported(model_name, pooling):
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
                continue

            stores: Dict[str, Any] = {}
            try:
                for model_name in _pbar(active, enabled=use_progress, desc=f"cache [{task_name}]", unit="model", leave=False):
                    nl = n_layers_by_model[model_name]
                    spec = spec_by_model[model_name]
                    st, _ = _load_store_matrix(
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

                texts_use = _aligned_texts({m: s.as_pooling(pooling) for m, s in stores.items()}, all_texts)
                n = len(texts_use)
                if n < int(args.min_n):
                    continue

                embs: Dict[str, np.ndarray] = {}
                for model_name in active:
                    view = stores[model_name].as_pooling(pooling)
                    spec = spec_by_model[model_name]
                    nl = n_layers_by_model[model_name]
                    X = extract_embedding_matrix(view, texts_use, spec, nl)
                    embs[model_name] = np.asarray(X, dtype=np.float32)

                pairs = [(u, v) for u in active for v in active]
                if int(args.pair_workers) > 1:
                    logger.warning(
                        "Fiber runner ignores --pair-workers>1 (sequential pairs only); "
                        "each pair runs a full per-point fiber spectrum.",
                    )

                def _run_one(model_u: str, model_v: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
                    U = np.asarray(embs[model_u], dtype=np.float64)
                    V = np.asarray(embs[model_v], dtype=np.float64)
                    t0 = time.perf_counter()
                    T, diag_t, _, _, _ = build_pairwise_adaptive_fiber_operator(U, V, **build_kw)
                    fc = compute_fiber_cov_prior_metrics(T, U, int(args.r_principal), **met_kw)
                    dt = time.perf_counter() - t0
                    prow = _emit_fiber_pair_row(
                        task_name=task_name,
                        model_u=model_u,
                        model_v=model_v,
                        pooling=pooling,
                        layer_spec=args.layer_spec,
                        n=n,
                        d_u=int(U.shape[1]),
                        d_v=int(V.shape[1]),
                        diag_t=diag_t,
                        fc=fc,
                        elapsed=dt,
                    )
                    base_local = {
                        "task_name": task_name,
                        "model_U": model_u,
                        "model_V": model_v,
                        "pooling": pooling,
                        "layer_spec": args.layer_spec,
                    }
                    loc: List[Dict[str, Any]] = [
                        {**base_local, "mode": "fc_uniform", **_flatten_stats("", fc.local_stats_uniform)},
                        {**base_local, "mode": "fc_whitened", **_flatten_stats("", fc.local_stats_whitened)},
                        {**base_local, "mode": "fc_frobenius", **_flatten_stats("", fc.local_stats_frobenius)},
                    ]
                    drow = {
                        "task_name": task_name,
                        "model_U": model_u,
                        "model_V": model_v,
                        "pooling": pooling,
                        "layer_spec": args.layer_spec,
                        "seconds": float(dt),
                    }
                    for k, v in fc.diagnostics.items():
                        drow[f"fiber_{k}"] = v
                    return prow, loc, drow

                for model_u, model_v in _pbar(
                    pairs,
                    enabled=use_progress,
                    desc=f"pairs [{task_name}]",
                    unit="pair",
                    leave=False,
                ):
                    prow, loc, drow = _run_one(model_u, model_v)
                    pair_rows.append(prow)
                    local_rows.extend(loc)
                    diag_rows.append(drow)
                    if incremental and pair_writer:
                        pair_writer.write_row(prow)
                    if incremental and local_writer:
                        for x in loc:
                            local_writer.write_row(x)
                    if incremental and diag_writer:
                        diag_writer.write_row(drow)
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
            if local_rows:
                pd.DataFrame(local_rows).to_csv(l_out, index=False)
            if diag_rows:
                pd.DataFrame(diag_rows).to_csv(d_out, index=False)
        else:
            logger.info("Incremental fiber CSVs: %s, %s, %s", p_out, l_out, d_out)


if __name__ == "__main__":
    main()
