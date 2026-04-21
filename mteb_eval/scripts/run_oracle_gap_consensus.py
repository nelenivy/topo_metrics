#!/usr/bin/env python3
"""
Consensus Laplacian prior + pairwise oracle-gap (Algorithms 2 & 3).

Defaults match ``run_unsup_eval.py`` layout:
  - ``--output-dir`` default ``./results/unsup_eval``
  - HDF5 cache default ``OUTPUT_DIR/embedding_cache``

Metrics and diagnostics are written under ``OUTPUT_DIR / METRICS_SUBDIR /``
(default ``.../oracle_gap/``). This script **never** opens or updates
``master_results.csv`` (that file belongs solely to ``run_unsup_eval.py``);
only the shared ``embedding_cache/`` under ``OUTPUT_DIR`` is reused.
"""

from __future__ import annotations

import argparse
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


def _write_run_info(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _setup_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        force=True,
    )


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
        help="HDF5 cache dir (default: OUTPUT_DIR/embedding_cache).",
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
    parser.add_argument("--knn-k", type=int, default=128)
    parser.add_argument("--bandwidth-grid-m", type=int, default=24)
    parser.add_argument("--principal-maxiter", type=int, default=2000)
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
        help="-v INFO, -vv DEBUG (progress + diagnostics).",
    )
    args = parser.parse_args()
    _setup_logging(int(args.verbose))

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

    cache_dir = (
        Path(args.embedding_cache_dir).resolve()
        if args.embedding_cache_dir
        else (out_root / "embedding_cache")
    )

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
                "Oracle-gap run: does not read or write master_results.csv "
                "(that file is only produced by run_unsup_eval.py). "
                "Same output_dir shares embedding_cache/ only."
            ),
            "output_dir": str(out_root),
            "metrics_dir": str(metrics_dir),
            "embedding_cache_dir": str(cache_dir),
            "models": models,
            "poolings": list(args.poolings),
            "layer_spec": args.layer_spec,
            "r_consensus": args.r_consensus,
            "r_principal": args.r_principal,
            "knn_k": args.knn_k,
            "bandwidth_grid_m": args.bandwidth_grid_m,
            "principal_maxiter": args.principal_maxiter,
            "pair_workers": int(args.pair_workers),
            "run_alg2": run_alg2,
            "run_alg3": run_alg3,
            "platform": platform.platform(),
        },
    )
    logger.info("Output root %s | metrics %s | cache %s", out_root, metrics_dir, cache_dir)

    tasks = load_tasks(
        task_set=args.task_set,
        tasks=args.tasks,
        task_types=args.task_types,
        max_samples=args.max_samples,
    )

    pair_csv = metrics_dir / "oracle_gap_pairwise.csv"
    local_csv = metrics_dir / "oracle_gap_local_stats.csv"
    agg_csv = metrics_dir / "oracle_gap_pairwise_agg.csv"
    by_task_csv = metrics_dir / "oracle_gap_by_task.csv"
    cv_all: List[Dict[str, Any]] = []
    pre_all: List[Dict[str, Any]] = []

    pair_rows: List[Dict[str, Any]] = []
    local_rows: List[Dict[str, Any]] = []

    for pooling in args.poolings:
        logger.info("=== Pooling %s ===", pooling)
        pair_rows.clear()
        local_rows.clear()

        for task in tasks:
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

            stores: Dict[str, Any] = {}
            try:
                for model_name in active:
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
                for model_name in active:
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
                    run_alg2=run_alg2,
                    run_alg3=run_alg3,
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

                results_list: List[Tuple[str, str, PairwiseOracleGapResult, float]] = []
                if eff_pw <= 1:
                    for model_u, model_v in pairs:
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
                        results_list.append((model_u, model_v, res, dt))
                        logger.info("  done in %.2fs | alg2_Q_mean=%.6g alg3_Q=%.6g", dt, res.alg2_Q_mean, res.alg3_Q_rank_r)
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
                    with ctx.Pool(processes=eff_pw) as pool:
                        raw = pool.map(_worker_compute_pair, pairs)
                    for (mu, mv, res) in raw:
                        results_list.append((mu, mv, res, 0.0))

                for model_u, model_v, res, elapsed in results_list:
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
                    if eff_pw <= 1:
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

                    for gi, eps_v in enumerate(np.asarray(res.eps_grid).ravel()):
                        cv_all.append(
                            {
                                "task_name": task_name,
                                "model_U": model_u,
                                "model_V": model_v,
                                "pooling": pooling,
                                "layer_spec": args.layer_spec,
                                "grid_index": gi,
                                "eps": float(eps_v),
                                "cv_loss": float(res.cv_scores[gi]),
                            }
                        )

                    pre_all.append(
                        {
                            "task_name": task_name,
                            "model_U": model_u,
                            "model_V": model_v,
                            "pooling": pooling,
                            "layer_spec": args.layer_spec,
                            **{f"diag_{k}": v for k, v in res.diagnostics.items()},
                        }
                    )

                    base_local = {
                        "task_name": task_name,
                        "model_U": model_u,
                        "model_V": model_v,
                        "pooling": pooling,
                        "layer_spec": args.layer_spec,
                    }
                    for j, st in enumerate(res.local_stats_alg2):
                        local_rows.append(
                            {
                                **base_local,
                                "mode": f"alg2_consensus_mode{j + 1}",
                                **_flatten_stats("", st),
                            }
                        )
                    if run_alg3 and res.local_stats_alg3:
                        local_rows.append(
                            {
                                **base_local,
                                "mode": "alg3_principal_rank_r",
                                **_flatten_stats("", res.local_stats_alg3),
                            }
                        )

            finally:
                for st in stores.values():
                    try:
                        del st
                    except Exception:
                        pass
                gc.collect()

        suffix = f"_{pooling}" if len(args.poolings) > 1 else ""
        p_out = pair_csv.with_name(pair_csv.stem + suffix + pair_csv.suffix)
        l_out = local_csv.with_name(local_csv.stem + suffix + local_csv.suffix)
        a_out = agg_csv.with_name(agg_csv.stem + suffix + agg_csv.suffix)
        t_out = by_task_csv.with_name(by_task_csv.stem + suffix + by_task_csv.suffix)

        if pair_rows:
            pd.DataFrame(pair_rows).to_csv(p_out, index=False)
            logger.info("Wrote %s (%d rows)", p_out, len(pair_rows))
        else:
            logger.warning("No rows for pooling=%s; skip %s", pooling, p_out)

        if local_rows:
            pd.DataFrame(local_rows).to_csv(l_out, index=False)
            logger.info("Wrote %s (%d rows)", l_out, len(local_rows))

        if pair_rows:
            df = pd.DataFrame(pair_rows)
            keys = ["model_U", "model_V", "pooling", "layer_spec"]
            skip = set(keys) | {"task_name", "seconds", "run_alg2", "run_alg3"}
            num_cols = [
                c
                for c in df.columns
                if c not in skip and pd.api.types.is_numeric_dtype(df[c])
            ]
            mean_df = df.groupby(keys, as_index=False)[num_cols].mean()
            std_df = df.groupby(keys, as_index=False)[num_cols].std()
            mean_df = mean_df.rename(columns={c: f"{c}_mean" for c in num_cols})
            std_df = std_df.rename(columns={c: f"{c}_std" for c in num_cols})
            mean_df.merge(std_df, on=keys).to_csv(a_out, index=False)
            logger.info("Wrote %s", a_out)

            t_mean = df.groupby("task_name", as_index=False)[num_cols].mean()
            t_std = df.groupby("task_name", as_index=False)[num_cols].std()
            t_mean = t_mean.rename(columns={c: f"{c}_mean_over_pairs" for c in num_cols})
            t_std = t_std.rename(columns={c: f"{c}_std_over_pairs" for c in num_cols})
            t_mean.merge(t_std, on="task_name").to_csv(t_out, index=False)
            logger.info("Wrote %s", t_out)

    if cv_all:
        pd.DataFrame(cv_all).to_csv(diag_dir / "bandwidth_cv_curves.csv", index=False)
        logger.info("Wrote %s (%d rows)", diag_dir / "bandwidth_cv_curves.csv", len(cv_all))
    if pre_all:
        pd.DataFrame(pre_all).to_csv(diag_dir / "pair_preprocess.csv", index=False)
        logger.info("Wrote %s (%d rows)", diag_dir / "pair_preprocess.csv", len(pre_all))
    logger.info("Diagnostics directory: %s", diag_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
