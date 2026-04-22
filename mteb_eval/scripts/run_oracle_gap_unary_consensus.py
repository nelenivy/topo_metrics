#!/usr/bin/env python3
"""
Unary oracle-gap (single embedding U vs ensemble consensus).

Same embedding cache / task layout as ``run_oracle_gap_consensus.py``, but:

* Calls ``compute_unary_oracle_gap`` per (task, model).
* Writes **separate** CSV / JSON names so pairwise outputs are not overwritten:

  - ``oracle_gap_unary.csv``
  - ``oracle_gap_unary_local_stats.csv``
  - ``oracle_gap_unary_agg.csv``
  - ``oracle_gap_unary_by_task.csv``
  - ``oracle_gap_unary_local_stats_agg.csv``
  - ``oracle_gap_unary_local_by_task.csv``
  - ``diagnostics/oracle_gap_unary_bandwidth_cv_curves.csv``
  - ``diagnostics/oracle_gap_unary_preprocess.csv``
  - ``run_info_unary.json``
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

logger = logging.getLogger("oracle_gap_unary")

# Load pairwise runner module for shared helpers (does not run ``main()``).
_pw_path = Path(__file__).resolve().parent / "run_oracle_gap_consensus.py"
_pw_spec = importlib.util.spec_from_file_location("_oracle_gap_pairwise_runner", _pw_path)
_pw = importlib.util.module_from_spec(_pw_spec)
assert _pw_spec.loader is not None
_pw_spec.loader.exec_module(_pw)

_flatten_stats = _pw._flatten_stats
_IncrementalDictWriter = _pw._IncrementalDictWriter
_write_run_info = _pw._write_run_info
_bandwidth_cv_curve_for_export = _pw._bandwidth_cv_curve_for_export
_pick_layer_spec = _pw._pick_layer_spec
_aligned_texts = _pw._aligned_texts
_load_store_matrix = _pw._load_store_matrix
_pbar = _pw._pbar
_setup_logging = _pw._setup_logging
_stage = _pw._stage
_pandas_group_numeric_stats = _pw._pandas_group_numeric_stats

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
from src.embedding_extractor import extract_embedding_matrix  # noqa: E402
from src.model_sets import MODEL_SET_MAP  # noqa: E402
from src.oracle_gap_unary import UnaryOracleGapResult, compute_unary_oracle_gap  # noqa: E402
from src.pooling_rules import pooling_supported  # noqa: E402
from src.task_sets import TASK_SET_MAP  # noqa: E402

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise SystemExit("run_oracle_gap_unary_consensus requires pandas") from e


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unary oracle-gap (Alg. 2 + 3) per model with consensus ensemble — "
        "writes oracle_gap_unary*.csv (does not touch oracle_gap_pairwise.csv).",
    )
    parser.add_argument("--models", nargs="*", default=None, metavar="MODEL")
    parser.add_argument(
        "--model-set",
        default="core",
        choices=["core", "standard", "full"],
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
    )
    parser.add_argument("--layer-spec", default="last_1")
    parser.add_argument("--output-dir", default="./results/unsup_eval")
    parser.add_argument(
        "--metrics-subdir",
        default="oracle_gap",
        help="Subfolder under --output-dir (same as pairwise; filenames differ).",
    )
    parser.add_argument("--embedding-cache-dir", default=None)
    parser.add_argument("--reuse-run-dir", default=None, metavar="DIR")
    parser.add_argument(
        "--no-incremental-csv",
        action="store_true",
        default=False,
        help="Write unary CSVs only once per pooling (legacy). Default: incremental flush.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--auto-embedding-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-trust-remote-code", action="store_true", default=False)
    parser.add_argument("--torch-dtype", default=None, metavar="DTYPE")
    parser.add_argument("--r-consensus", type=int, default=8)
    parser.add_argument("--r-principal", type=int, default=8)
    parser.add_argument("--knn-k", type=int, default=128)
    parser.add_argument(
        "--consensus-knn-k",
        type=int,
        default=15,
        help="kNN size inside mutual_knn_consensus_affinity (default: 15).",
    )
    parser.add_argument("--bandwidth-grid-m", type=int, default=24)
    parser.add_argument("--sigma-clip", type=float, default=3.0)
    parser.add_argument(
        "--density-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Same directed marginal rescale on self-fiber W as pairwise (default: on).",
    )
    parser.add_argument(
        "--density-alpha",
        type=float,
        default=1.0,
        help="Exponent α for q_out^α · q_in^α (default: 1).",
    )
    parser.add_argument(
        "--smooth-bw",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fiber-kernel",
        choices=["gaussian", "epanechnikov"],
        default="gaussian",
    )
    parser.add_argument(
        "--use-fiedler-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Weight ensemble models by inverse Fiedler (default: on).",
    )
    parser.add_argument("--fiedler-floor", type=float, default=1e-3)
    parser.add_argument("--principal-maxiter", type=int, default=12000)
    parser.add_argument(
        "--principal-device",
        type=str,
        default=None,
        metavar="DEV",
        help=(
            "Algorithm 3: cuda / cuda:0 for CuPy GPU matvec if installed; else CPU SciPy."
        ),
    )
    parser.add_argument(
        "--principal-blas-threads",
        type=int,
        default=None,
        metavar="N",
        help="CPU SciPy eigsh: pin BLAS/OpenMP threads (0 = do not change env).",
    )
    parser.add_argument("--min-n", type=int, default=40)
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--skip-alg2", action="store_true")
    parser.add_argument("--skip-alg3", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        parser.error("Need at least two models in the ensemble for unary consensus metrics.")

    run_alg2 = not bool(args.skip_alg2)
    run_alg3 = not bool(args.skip_alg3)
    if not run_alg2 and not run_alg3:
        parser.error("Cannot skip both --skip-alg2 and --skip-alg3.")

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

    _write_run_info(
        metrics_dir / "run_info_unary.json",
        {
            "argv": sys.argv,
            "note": (
                "Unary oracle-gap: separate CSV names from pairwise; "
                "does not read/write master_results.csv."
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
            "consensus_knn_k": int(args.consensus_knn_k),
            "bandwidth_grid_m": args.bandwidth_grid_m,
            "sigma_clip": float(args.sigma_clip),
            "density_normalize": bool(args.density_normalize),
            "density_alpha": float(args.density_alpha),
            "smooth_bw": bool(args.smooth_bw),
            "fiber_kernel": args.fiber_kernel,
            "use_fiedler_weights": bool(args.use_fiedler_weights),
            "fiedler_floor": float(args.fiedler_floor),
            "principal_maxiter": args.principal_maxiter,
            "principal_device": args.principal_device,
            "principal_blas_threads": args.principal_blas_threads,
            "run_alg2": run_alg2,
            "run_alg3": run_alg3,
            "platform": platform.platform(),
            "log_level": logging.getLevelName(logger.getEffectiveLevel()),
            "progress_bars": use_progress,
        },
    )

    logger.info("Unary oracle-gap | output root %s | metrics %s", out_root, metrics_dir)
    _stage("init", f"unary runner | device={args.device} batch_size={bs}", pooling=None)

    tasks = load_tasks(
        task_set=args.task_set,
        tasks=args.tasks,
        task_types=args.task_types,
        max_samples=args.max_samples,
    )

    unary_csv = metrics_dir / "oracle_gap_unary.csv"
    local_csv = metrics_dir / "oracle_gap_unary_local_stats.csv"
    agg_csv = metrics_dir / "oracle_gap_unary_agg.csv"
    by_task_csv = metrics_dir / "oracle_gap_unary_by_task.csv"
    local_agg_csv = metrics_dir / "oracle_gap_unary_local_stats_agg.csv"
    local_by_task_csv = metrics_dir / "oracle_gap_unary_local_by_task.csv"
    diag_cv_path = diag_dir / "oracle_gap_unary_bandwidth_cv_curves.csv"
    diag_pre_path = diag_dir / "oracle_gap_unary_preprocess.csv"

    cv_all: List[Dict[str, Any]] = []
    pre_all: List[Dict[str, Any]] = []
    unary_rows: List[Dict[str, Any]] = []
    local_rows: List[Dict[str, Any]] = []

    cv_writer: Optional[_IncrementalDictWriter] = None
    pre_writer: Optional[_IncrementalDictWriter] = None
    if incremental:
        for pth in (diag_cv_path, diag_pre_path):
            if pth.exists():
                pth.unlink()
        cv_writer = _IncrementalDictWriter(diag_cv_path)
        pre_writer = _IncrementalDictWriter(diag_pre_path)

    ukw = dict(
        r_consensus=int(args.r_consensus),
        r_principal=int(args.r_principal),
        knn_k=int(args.knn_k),
        consensus_knn_k=int(args.consensus_knn_k),
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
        use_fiedler_weights=bool(args.use_fiedler_weights),
        fiedler_floor=float(args.fiedler_floor),
    )

    for pooling in args.poolings:
        logger.info("=== Unary | pooling %s ===", pooling)
        unary_rows.clear()
        local_rows.clear()

        suffix = f"_{pooling}" if len(args.poolings) > 1 else ""
        u_out = unary_csv.with_name(unary_csv.stem + suffix + unary_csv.suffix)
        l_out = local_csv.with_name(local_csv.stem + suffix + local_csv.suffix)
        a_out = agg_csv.with_name(agg_csv.stem + suffix + agg_csv.suffix)
        t_out = by_task_csv.with_name(by_task_csv.stem + suffix + by_task_csv.suffix)
        la_out = local_agg_csv.with_name(local_agg_csv.stem + suffix + local_agg_csv.suffix)
        lt_out = local_by_task_csv.with_name(local_by_task_csv.stem + suffix + local_by_task_csv.suffix)

        unary_writer: Optional[_IncrementalDictWriter] = None
        local_writer: Optional[_IncrementalDictWriter] = None
        if incremental:
            for pth in (u_out, l_out):
                if pth.exists():
                    pth.unlink()
            unary_writer = _IncrementalDictWriter(u_out)
            local_writer = _IncrementalDictWriter(l_out)

        task_iter = _pbar(
            tasks,
            enabled=use_progress,
            desc=f"unary tasks [{pooling}]",
            unit="task",
            smoothing=0.0,
        )
        for task in task_iter:
            task_name = task.metadata.name
            if use_progress and hasattr(task_iter, "set_postfix"):
                task_iter.set_postfix(task=task_name[:40], refresh=False)
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
            for model_name in _pbar(
                models,
                enabled=use_progress,
                desc=f"probe [{task_name}]",
                unit="model",
                leave=False,
            ):
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
                logger.warning("Task %s: fewer than 2 usable models; skip", task_name)
                continue

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

                def _emit_unary(model: str, res: UnaryOracleGapResult, elapsed: float) -> None:
                    row: Dict[str, Any] = {
                        "task_name": task_name,
                        "model": model,
                        "pooling": pooling,
                        "layer_spec": args.layer_spec,
                        "n": n,
                        "d": int(embs[model].shape[1]),
                        "eps_hat": res.eps_hat,
                        "alg2_Q_mean": res.alg2_Q_mean,
                        "alg3_Q_rank_r": res.alg3_Q_rank_r,
                        "run_alg2": int(run_alg2),
                        "run_alg3": int(run_alg3),
                        "seconds": float(elapsed),
                    }
                    for j, q in enumerate(res.alg2_Q_per_mode):
                        row[f"alg2_Q_mode{j + 1}"] = q
                    for j, lam in enumerate(res.lambdas_consensus):
                        row[f"lambda_consensus_{j + 1}"] = float(lam)
                    for j, lam in enumerate(res.lambdas_principal):
                        row[f"lambda_principal_{j + 1}"] = float(lam)
                    for j, w in enumerate(np.asarray(res.model_weights, dtype=np.float64).ravel()):
                        row[f"ensemble_weight_{j + 1}"] = float(w)
                    for k, v in res.diagnostics.items():
                        row[f"diag_{k}"] = v
                    unary_rows.append(row)
                    if incremental and unary_writer is not None:
                        unary_writer.write_row(row)

                    eps_curve, cv_curve = _bandwidth_cv_curve_for_export(res)
                    cv_batch = [
                        {
                            "task_name": task_name,
                            "model": model,
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
                        "model": model,
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
                        "model": model,
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

                for model_name in _pbar(
                    active,
                    enabled=use_progress,
                    desc=f"unary [{task_name}]",
                    unit="model",
                    leave=False,
                ):
                    logger.info(
                        "Task %s | unary model %s | n=%d | pooling=%s",
                        task_name,
                        model_name,
                        n,
                        pooling,
                    )
                    t0 = time.perf_counter()
                    res = compute_unary_oracle_gap(
                        embs[model_name], consensus_list, **ukw
                    )
                    dt = time.perf_counter() - t0
                    logger.info(
                        "  done in %.2fs | alg2_Q_mean=%.6g alg3_Q=%.6g",
                        dt,
                        res.alg2_Q_mean,
                        res.alg3_Q_rank_r,
                    )
                    _emit_unary(model_name, res, dt)

            finally:
                for st in stores.values():
                    try:
                        del st
                    except Exception:
                        pass
                gc.collect()

        if not incremental:
            if unary_rows:
                pd.DataFrame(unary_rows).to_csv(u_out, index=False)
                logger.info("Wrote %s (%d rows)", u_out, len(unary_rows))
            else:
                logger.warning("No unary rows for pooling=%s; skip %s", pooling, u_out)
            if local_rows:
                pd.DataFrame(local_rows).to_csv(l_out, index=False)
                logger.info("Wrote %s (%d rows)", l_out, len(local_rows))
        else:
            if unary_rows:
                logger.info("Incremental unary CSV %s (%d rows)", u_out, len(unary_rows))
            if local_rows:
                logger.info("Incremental unary local stats %s (%d rows)", l_out, len(local_rows))

        if unary_rows:
            df = pd.DataFrame(unary_rows)
            keys = ["model", "pooling", "layer_spec"]
            skip = set(keys) | {"task_name", "seconds", "run_alg2", "run_alg3"}
            num_cols = [
                c
                for c in df.columns
                if c not in skip and pd.api.types.is_numeric_dtype(df[c])
            ]
            _pandas_group_numeric_stats(df, keys, num_cols, over_suffix="").to_csv(a_out, index=False)
            logger.info("Wrote %s", a_out)

            _pandas_group_numeric_stats(
                df, ["task_name"], num_cols, over_suffix="_over_models"
            ).to_csv(t_out, index=False)
            logger.info("Wrote %s", t_out)

        if local_rows:
            ldf = pd.DataFrame(local_rows)
            lkeys = ["model", "pooling", "layer_spec", "mode"]
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
    logger.info("Unary diagnostics directory: %s", diag_dir)
    logger.info("Unary run done.")


if __name__ == "__main__":
    main()
