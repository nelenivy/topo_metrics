#!/usr/bin/env python3
"""
run_unsup_eval.py
Main evaluation script: unsupervised metrics + MTEB test-split scores.

Usage examples
--------------
# Core tasks + predefined model list, all poolings:
python scripts/run_unsup_eval.py \
    --model-set core \
    --task-set core \
    --poolings mean cls last_token \
    --output-dir ./results/unsup_eval

# Standard tasks, single model:
python scripts/run_unsup_eval.py \
    --models sentence-transformers/all-mpnet-base-v2 \
    --task-set standard \
    --poolings mean \
    --include-ph-dim \
    --output-dir ./results/unsup_eval

# Specific tasks only:
python scripts/run_unsup_eval.py \
    --models bert-base-uncased \
    --tasks STSBenchmark Banking77Classification NFCorpus \
    --poolings mean cls \
    --output-dir ./results/unsup_eval

Key design points
-----------------
* Per-(model, task) HDF5 embedding store: all valid poolings for that task are
  filled in one forward pass per batch; each pooling is sliced for metrics/MTEB.
  Single-pooling runs still use the legacy one-pooling HDF5 layout on disk.
* MTEB evaluation on the test split via ``task.filter_eval_splits(["test"])`` and
  ``mteb.evaluate``; ``StoreBackedAggregatedEncoder`` reads the HDF5 store only —
  **no** second transformer forward.
* Retrieval tasks: unsup metrics on corpus, queries, and combined separately.
* Incremental CSV output: appended after each (model, task, pooling) triple.
  Reruns reuse cached embeddings and refresh missing metric columns in place.
* --overwrite to redo specific configs.
* --model-set core|standard|full when ``--models`` is omitted (see ``src/model_sets.py``).
* Invalid (model, pooling) pairs (e.g. ``cls`` on causal LMs) are **skipped by name
  rules** in ``src/pooling_rules.py`` — not by catching exceptions.
* ``--layer-spec-workers N`` (Unix fork): after embeddings are in RAM, run MTEB + unsup
  metrics for several layer specs in parallel (separate ``mteb_raw`` subdirs per spec;
  workers force CPU for the store-backed encoder).
* ``--mteb-gpu-proxy``: for **Retrieval** and **Reranking** (full ``MTEB(eng, v2)`` and any
  other run), score with dense GPU (or CPU) similarity plus MTEB's own ``pytrec_eval`` metric
  stack. ``Classification`` tasks use a GPU logistic-regression proxy on cached vectors when
  CUDA is available; other task types still use ``mteb.evaluate`` with the HDF5-backed encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import gc
import json
import logging
import multiprocessing
import os
import re
import sys
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import mteb
from datasets import Dataset, DatasetDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mteb_hf_hub_file_cache import patch_mteb_hub_file_cache

patch_mteb_hub_file_cache()

from src.layer_spec import build_layer_specs, LayerSpec
from src.task_sets import TASK_SET_MAP, FULL_BENCHMARK_NAME
from src.model_sets import MODEL_SET_MAP
from src.pooling_rules import pooling_supported, skip_reason
from src.unsup_metrics import (
    compute_metrics,
    compute_metrics_retrieval,
    suffix_metrics,
    metric_output_map,
)
from src.embedding_extractor import extract_embedding_matrix, extract_retrieval_embeddings
from src.result_store import ResultStore

from src.aggregated_encoder import LayerEncoder, StoreBackedAggregatedEncoder
from src.cache_manager import LayerEmbeddingStore, PooledLayerEmbeddingView, layer_store_hdf5_path
from src.mteb_gpu_proxy import gpu_proxy_main_score
from src.mteb_text_align import iter_retrieval_corpus_passages

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _profile_fields_text(fields: Dict[str, Any]) -> str:
    parts = []
    for k, v in fields.items():
        if v is None:
            continue
        parts.append(f"{k}={v}")
    return " | ".join(parts)


def _profile_log(label: str, started_at: float, /, **fields: Any) -> None:
    elapsed_s = time.perf_counter() - started_at
    extra = _profile_fields_text(fields)
    if extra:
        logger.info("[profile] %s | %.3fs | %s", label, elapsed_s, extra)
    else:
        logger.info("[profile] %s | %.3fs", label, elapsed_s)


def _silence_noisy_loggers() -> None:
    """HF / MTEB default INFO is very chatty (HTTP, hub, dataset prep, eval banners)."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    for name in (
        "httpx",
        "httpcore",
        "urllib3",
        "huggingface_hub",
        "datasets",
        "filelock",
        "fsspec",
        "mteb",
        "mteb.models.model_meta",
        "sentence_transformers",
        "torch.distributed",
        "safetensors",
        "src.cache_manager",
        "sklearn",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        from transformers.utils import logging as tr_logging

        tr_logging.set_verbosity_error()
    except Exception:
        pass
    warnings.filterwarnings(
        "ignore",
        message=r"The input point cloud has more columns than rows.*",
        category=UserWarning,
        module="ripser.ripser",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*[Dd]ataset.*superseded.*",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    try:
        from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    except Exception:
        pass


def _normalize_torch_dtype_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return s


def _default_embedding_weight_dtype(device: str) -> Optional[str]:
    """
    Default HF weight dtype for embedding extraction.

    On CUDA with native bf16 support, default to bf16 weights (fast + lower VRAM).
    Otherwise keep None (HF default / fp32), which is safer on CPU and older GPUs.
    """
    if not str(device).lower().startswith("cuda"):
        return None
    if not torch.cuda.is_available():
        return None
    try:
        if torch.cuda.is_bf16_supported():
            return "bfloat16"
    except Exception:
        pass
    return None


def _infer_embedding_batch_size(
    *,
    device: str,
    torch_dtype: Optional[str],
    base: int,
) -> int:
    """
    Heuristic batch-size bump when using low-precision weights on GPU.

    This is intentionally conservative: users can always set ``--batch-size`` explicitly.
    """
    bs = int(base)
    if bs < 1:
        return 1
    if not str(device).lower().startswith("cuda") or not torch.cuda.is_available():
        return bs

    dt = (_normalize_torch_dtype_str(torch_dtype) or "").lower()
    mult = 1.0
    if dt in ("bfloat16", "bf16", "float16", "fp16"):
        mult = 2.0
    else:
        mult = 1.25

    out = int(round(bs * mult))
    return max(8, min(512, out))


def _parse_precompute_devices_arg(raw: Optional[str], *, fallback_device: str) -> Optional[List[str]]:
    """
    Parse ``--embedding-precompute-devices``.

    - omitted / empty / "auto": infer from ``fallback_device`` + visible CUDA devices
    - "cuda": all visible GPUs as cuda:0..cuda:N-1
    - comma-separated list: cuda:1,cuda:0,...
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("auto", "default"):
        return None

    if s.lower() == "cuda":
        if not torch.cuda.is_available():
            return [fallback_device]
        n = int(torch.cuda.device_count())
        if n <= 0:
            return [fallback_device]
        return [f"cuda:{i}" for i in range(n)]

    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts or None


def _embedding_precompute_devices(*, device: str, override: Optional[List[str]]) -> List[str]:
    if override:
        return list(override)

    d = str(device).strip()
    dl = d.lower()
    if dl.startswith("cuda") and torch.cuda.is_available():
        if dl == "cuda":
            n = int(torch.cuda.device_count())
            if n > 1:
                return [f"cuda:{i}" for i in range(n)]
            return ["cuda:0"] if n == 1 else [d]
        return [d]
    return [d]


_silence_noisy_loggers()


def _metric_subsample_rows(n_rows: int, metric_kwargs: Dict) -> int:
    """Rows used per unsup subsample (same rule as ``src.unsup_metrics.compute_metrics``)."""
    if n_rows <= 0:
        return 0
    frac = float(metric_kwargs.get("sample_fraction") or 0.05)
    floor = int(metric_kwargs.get("min_sample_size") or 100)
    sample_size = max(int(frac * n_rows), floor)
    return min(sample_size, n_rows)


def _unsup_metrics_hash(metric_kwargs: Dict[str, Any]) -> str:
    """Stable hash for the unsupervised-metric configuration."""
    payload = json.dumps(metric_kwargs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _csv_value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "nan", "none", "null"}
    try:
        if isinstance(value, (float, np.floating)):
            return not np.isnan(value)
    except Exception:
        pass
    return True


def _metric_layout(metric_kwargs: Dict[str, Any]) -> Dict[str, List[str]]:
    return metric_output_map(
        include_ph_dim=bool(metric_kwargs.get("include_ph_dim", False)),
        ripser_maxdim=int(metric_kwargs.get("ripser_maxdim", 1)),
    )


def _base_metric_columns(output_names: List[str], *, retrieval: bool) -> List[str]:
    cols = [f"metric_{name}" for name in output_names]
    cols.extend(f"std_{name}" for name in output_names)
    if not retrieval:
        return cols
    suffixed: List[str] = []
    for variant in ("corpus", "queries", "combined"):
        suffixed.extend(f"{col}_{variant}" for col in cols)
    return suffixed


def _missing_metric_bases(
    existing_row: Dict[str, Any],
    metric_layout: Dict[str, List[str]],
    *,
    retrieval: bool,
) -> List[str]:
    missing: List[str] = []
    for base_name, output_names in metric_layout.items():
        cols = _base_metric_columns(output_names, retrieval=retrieval)
        if any(not _csv_value_present(existing_row.get(col)) for col in cols):
            missing.append(base_name)
    return missing


def _plan_metric_refresh(
    existing_row: Dict[str, Any],
    metric_layout: Dict[str, List[str]],
    current_hash: str,
    *,
    retrieval: bool,
) -> tuple[Optional[List[str]], bool] | None:
    """
    Decide whether a cached row is up to date.

    Returns:
      - None if the row can be skipped entirely.
      - (selected_metrics, run_mteb) otherwise.

    ``selected_metrics`` is ``None`` for a full unsup recompute, ``[]`` for
    "MTEB only", or a list of base metric names to recompute.
    """
    has_mteb = _csv_value_present(existing_row.get("mteb_score"))
    has_error = _csv_value_present(existing_row.get("metric_error"))
    row_hash = str(existing_row.get("unsup_metrics_hash", "") or "").strip().lower()
    missing_bases = _missing_metric_bases(existing_row, metric_layout, retrieval=retrieval)

    # Legacy rows without a hash are treated as fresh if all expected metric
    # columns are already present.
    if not has_error and has_mteb and (not row_hash or row_hash == current_hash) and not missing_bases:
        return None

    if has_error or (row_hash and row_hash != current_hash):
        selected_metrics: Optional[List[str]] = None
    else:
        selected_metrics = missing_bases

    run_mteb = not has_mteb
    return selected_metrics, run_mteb


def _last_mteb_score_for_progress_row(row: Dict[str, Any]) -> str:
    ms = row.get("mteb_score", "")
    if ms in ("", None):
        return ""
    if isinstance(ms, (int, float)):
        return f"{float(ms):.4f}"
    try:
        return f"{float(ms):.4f}"
    except (TypeError, ValueError):
        return str(ms)


def _progress_message_body_from_dict(d: Dict[str, Any]) -> str:
    """Single ``[progress] …`` line (no timestamp / logger name)."""
    short_model = (d.get("model") or "").split("/")[-1][:48]
    nt = int(d.get("n_texts") or 0)
    nl = int(d.get("n_layers") or 0)
    msn = int(d.get("metric_subsample_n") or 0)
    nq = int(d.get("n_queries") or 0)
    nc = int(d.get("n_corpus") or 0)
    texts_part = f" | n_texts={nt}" if nt else ""
    layers_part = f" | n_layers={nl}" if nl else ""
    unsup_part = f" | unsup_sample={msn}" if msn else ""
    retr_part = f" | q={nq} c={nc}" if (nq or nc) else ""
    return (
        "[progress] pooled %s/%s | model=%s | task=%s | pool=%s | spec=%s %s/%s | %s | mteb=%s%s%s%s%s"
        % (
            d.get("pooled_steps_done", 0),
            d.get("pooled_steps_total", "?"),
            short_model or "—",
            d.get("task") or "—",
            d.get("pooling") or "—",
            d.get("layer_spec") or "—",
            d.get("layer_spec_index", 0),
            d.get("layer_spec_count", 0),
            d.get("phase") or "—",
            d.get("last_mteb_score", ""),
            texts_part,
            layers_part,
            unsup_part,
            retr_part,
        )
    )


def _progress_info_log_from_dict(d: Dict[str, Any]) -> None:
    """Same one-line ``[progress]`` INFO as ``ProgressReporter.flush`` (no file I/O)."""
    msg = _progress_message_body_from_dict(d)
    logger.info(msg)
    for h in logging.root.handlers:
        try:
            h.flush()
        except (OSError, ValueError, AttributeError):
            pass


def _progress_emit_from_pool_worker(d: Dict[str, Any]) -> None:
    """
    Emit the same line as ``logger.info`` would, but without the ``logging`` module.
    Forked pool workers can deadlock or lose output on inherited logging handlers;
    ``os.write`` to fd 1/2 is reliable for screen/tty.
    """
    body = _progress_message_body_from_dict(d)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} INFO {logger.name} — {body}\n"
    buf = line.encode("utf-8", errors="replace")
    try:
        os.write(1, buf)
    except OSError:
        pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (OSError, ValueError):
        pass
    # Screen / log capture often only shows stdio from ``print`` in forked workers.
    try:
        print(line, end="", file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


def _emit_progress_after_json_write(d: Dict[str, Any]) -> None:
    if multiprocessing.current_process().name != "MainProcess":
        _progress_emit_from_pool_worker(d)
    else:
        _progress_info_log_from_dict(d)


def _locked_merge_run_progress_json(
    output_dir: str, updates: Dict[str, Any], *, emit_log: bool
) -> None:
    """
    Read/modify/write ``run_progress.json`` under an exclusive lock so forked
    layer-spec workers can publish the same progress as the parent without races.
    """
    path = Path(output_dir) / "run_progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:
        d: Dict[str, Any] = {}
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                d = {}
        d.update(updates)
        d["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2), encoding="utf-8")
        tmp.replace(path)
        if emit_log:
            _emit_progress_after_json_write(d)
        return

    lock_path = path.parent / ".run_progress.json.lock"
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        d = {}
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                d = {}
        d.update(updates)
        d["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2), encoding="utf-8")
        tmp.replace(path)
        if emit_log:
            _emit_progress_after_json_write(d)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _worker_emit_layer_row_progress(ctx: Dict[str, Any], si: int, row: Dict[str, Any]) -> None:
    if not ctx.get("progress_enabled"):
        return
    base = dict(ctx.get("progress_base") or {})
    updates = {
        **base,
        "model": ctx["model_name"],
        "task": ctx["task_name"],
        "task_type": ctx["task_type"],
        "pooling": ctx["pooling"],
        "layer_spec": str(row.get("layer_spec", "")),
        "layer_spec_index": si,
        "layer_spec_count": int(ctx["n_specs"]),
        "phase": "layer_row",
        "last_mteb_score": _last_mteb_score_for_progress_row(row),
        "n_layers": int(ctx["n_layers"]),
        "n_texts": int(ctx["n_task_texts"]),
        "n_queries": int(ctx["n_queries"]),
        "n_corpus": int(ctx["n_corpus"]),
        "metric_subsample_n": int(ctx["metric_subsample_n"]),
    }
    _locked_merge_run_progress_json(str(ctx["output_dir"]), updates, emit_log=True)


def _worker_emit_layer_spec_started(
    ctx: Dict[str, Any], si: int, spec_name: str, *, emit_log: bool = True
) -> None:
    """Heartbeat: update ``run_progress.json`` when a worker begins a spec (long MTEB otherwise freezes mtime)."""
    if not ctx.get("progress_enabled"):
        return
    base = dict(ctx.get("progress_base") or {})
    updates = {
        **base,
        "model": ctx["model_name"],
        "task": ctx["task_name"],
        "task_type": ctx["task_type"],
        "pooling": ctx["pooling"],
        "layer_spec": spec_name,
        "layer_spec_index": si,
        "layer_spec_count": int(ctx["n_specs"]),
        "phase": "layer_spec_started",
        "last_mteb_score": "",
        "n_layers": int(ctx["n_layers"]),
        "n_texts": int(ctx["n_task_texts"]),
        "n_queries": int(ctx["n_queries"]),
        "n_corpus": int(ctx["n_corpus"]),
        "metric_subsample_n": int(ctx["metric_subsample_n"]),
    }
    _locked_merge_run_progress_json(str(ctx["output_dir"]), updates, emit_log=emit_log)


class ProgressReporter:
    """Writes ``run_progress.json`` and one-line INFO so progress survives tiny terminals."""

    def __init__(self, output_dir: str) -> None:
        self._path = Path(output_dir) / "run_progress.json"
        self._d: Dict[str, Any] = {
            "pooled_steps_done": 0,
            "pooled_steps_total": 0,
            "model": "",
            "task": "",
            "task_type": "",
            "pooling": "",
            "layer_spec": "",
            "layer_spec_index": 0,
            "layer_spec_count": 0,
            "phase": "starting",
            "last_mteb_score": "",
            "n_texts": 0,
            "n_layers": 0,
            "n_queries": 0,
            "n_corpus": 0,
            "metric_subsample_n": 0,
        }

    def set_totals(self, pooled_steps_total: int) -> None:
        self._d["pooled_steps_total"] = int(pooled_steps_total)
        self.flush(silent=True)

    def flush(self, *, silent: bool = False, **updates: Any) -> None:
        if "pooled_steps_done" in updates:
            psd = updates.pop("pooled_steps_done", None)
            if psd is not None:
                self._d["pooled_steps_done"] = int(psd)
        if updates:
            self._d.update(updates)
        self._d["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._d, indent=2), encoding="utf-8")
        tmp.replace(self._path)
        if silent:
            return
        _progress_info_log_from_dict(self._d)


def _count_progress_steps(models: List[str], tasks: List, poolings: List[str]) -> int:
    """(model, task, pooling) combinations where pooling is valid for model."""
    n = 0
    for model_name in models:
        for _task in tasks:
            for pooling in poolings:
                if pooling_supported(model_name, pooling):
                    n += 1
    return n


def _steps_skipped_for_task(model_name: str, poolings: List[str]) -> int:
    return sum(1 for p in poolings if pooling_supported(model_name, p))


def _make_progress_bar(total: int, enabled: bool):
    """Progress bar on stderr so INFO logs on stdout stay readable."""
    if not enabled or tqdm is None or total <= 0:
        return None
    return tqdm(
        total=total,
        desc="unsup_eval",
        unit="step",
        file=sys.stderr,
        mininterval=2.0,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )


def _pbar_update(pbar, n: int = 1) -> None:
    if pbar is not None:
        pbar.update(n)


def _pbar_sync_progress(progress: Optional[ProgressReporter], pbar) -> None:
    if progress is not None and pbar is not None:
        progress.flush(silent=True, pooled_steps_done=pbar.n)


def _pbar_close(pbar) -> None:
    if pbar is not None:
        pbar.close()

IMAGE_TASK_NAMES = {
    "BirdsnapZeroShot", "Caltech101ZeroShot", "CIFAR100ZeroShot",
    "Country211ZeroShot", "DTDZeroShot", "EuroSATZeroShot",
    "FER2013ZeroShot", "FGVCAircraftZeroShot", "Food101ZeroShot",
    "Flowers102ZeroShot", "GTSRBZeroShot", "Imagenet1kZeroShot",
    "OxfordPetsZeroShot", "PatchCamelyonZeroShot", "RESISC45ZeroShot",
    "StanfordCarsZeroShot", "STL10ZeroShot", "SUN397ZeroShot", "UCF101ZeroShot",
    "CIFAR10ZeroShot", "MNISTZeroShot", "CIFAR10Clustering", "CIFAR100Clustering",
    "ImageNetDog15Clustering", "TinyImageNetClustering",
    "BLINKIT2IRetrieval", "BLINKIT2TRetrieval", "COCO2017Retrieval",
    "Flickr30kRetrieval", "MIRACLRetrieval", "MIRACLVisionRetrieval",
    "VisualNewsRetrieval", "WebQARetrieval", "Wiki-SS-NQRetrieval",
    "ChartQA", "DocVQA", "InfographicVQA", "OCR-VQA", "TextVQA",
    "SUGARCREPEAddition", "SUGARCREPEReplacement", "SUGARCREPESwap",
    "VOC2007Classification", "ImageNet", "Places365",
}


# ══════════════════════════════════════════════════════════════════════════ #
#  Task loading & filtering                                                   #
# ══════════════════════════════════════════════════════════════════════════ #

def load_tasks(
    task_set: Optional[str],
    tasks: Optional[List[str]],
    task_types: Optional[List[str]],
    max_samples: Optional[int],
) -> List:
    if tasks:
        all_tasks = mteb.get_tasks(tasks=tasks, languages=["eng"])
    elif task_set == "full":
        bench = mteb.get_benchmark(FULL_BENCHMARK_NAME)
        all_tasks = list(bench.tasks)
    elif task_set in TASK_SET_MAP and TASK_SET_MAP[task_set]:
        all_tasks = mteb.get_tasks(
            tasks=list(TASK_SET_MAP[task_set]), languages=["eng"]
        )
    else:
        raise ValueError(f"Unknown task_set: {task_set!r}.")

    all_tasks = [
        t for t in all_tasks
        if t.metadata.name not in IMAGE_TASK_NAMES
        and (getattr(t.metadata, "modalities", None) in (None, ["text"], ("text",)))
    ]

    if task_types:
        all_tasks = [
            t for t in all_tasks
            if getattr(t.metadata, "type", None) in task_types
        ]

    if max_samples is not None:
        all_tasks = _filter_by_max_samples(all_tasks, max_samples)

    names = [t.metadata.name for t in all_tasks]
    if len(names) <= 8:
        logger.info("Running %d tasks: %s", len(names), ", ".join(names))
    else:
        logger.info(
            "Running %d tasks: %s … (+%d more)",
            len(names),
            ", ".join(names[:8]),
            len(names) - 8,
        )
    return all_tasks


def _filter_by_max_samples(tasks, max_samples: int) -> List:
    kept = []
    for task in tasks:
        try:
            task.load_data()
            count = 0
            if hasattr(task, "dataset") and task.dataset:
                for split_data in task.dataset.values():
                    if split_data is not None:
                        count += len(split_data)
            if count == 0 or count <= max_samples:
                kept.append(task)
        except Exception:
            kept.append(task)
    return kept


# ══════════════════════════════════════════════════════════════════════════ #
#  Text extraction helpers (nested splits)                                   #
# ══════════════════════════════════════════════════════════════════════════ #

def _resolve_dataset(task):
    """Unwrap nested dataset dicts to the innermost HuggingFace DatasetDict."""
    dataset = task.dataset
    for key in ("default", "en"):
        if isinstance(dataset, dict) and key in dataset:
            dataset = dataset[key]
    return dataset


def _unique_nonempty_stripped(strings: List[str]) -> List[str]:
    """
    Stable de-duplication with strip(), matching keys used in ``LayerEmbeddingStore``
    after ``extract_all_texts`` precompute (must stay consistent with retrieval lists).
    """
    seen = set()
    out: List[str] = []
    for t in strings:
        t = str(t).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _iter_leaf_datasets(dataset_like):
    """Yield HuggingFace leaf datasets from nested dict / DatasetDict containers."""
    if isinstance(dataset_like, Dataset):
        yield dataset_like
        return
    if isinstance(dataset_like, DatasetDict):
        for value in dataset_like.values():
            yield from _iter_leaf_datasets(value)
        return
    if isinstance(dataset_like, dict) or hasattr(dataset_like, "values"):
        try:
            values = list(dataset_like.values())
        except Exception:
            values = []
        if values:
            for value in values:
                yield from _iter_leaf_datasets(value)
        return


def extract_all_texts(task) -> List[str]:
    """Extract all unique texts needed by the task, including nested splits."""
    task_type = task.metadata.type
    raw: List[str] = []

    def _add(items):
        for item in items:
            if isinstance(item, list):
                raw.extend(str(s) for s in item if s)
            elif item:
                raw.append(str(item))

    if task_type in ("Retrieval", "Reranking"):
        dataset = _resolve_dataset(task)
        split_name = "test"
        if split_name not in dataset:
            split_name = list(dataset.keys())[0]

        val_data = dataset[split_name]
        data_dict = val_data.to_dict() if hasattr(val_data, "to_dict") else dict(val_data)

        queries = data_dict.get("queries", {})
        if isinstance(queries, dict):
            raw.extend(str(v) for v in queries.values() if v)
        elif hasattr(queries, "column_names"):
            for col in ("text", "query"):
                if col in queries.column_names:
                    raw.extend(str(t) for t in queries[col] if t)
                    break
        corpus = data_dict.get("corpus") or dataset.get("corpus", {})
        raw.extend(iter_retrieval_corpus_passages(corpus))
        return _unique_nonempty_stripped(raw)

    for leaf in _iter_leaf_datasets(task.dataset):
        if not hasattr(leaf, "to_dict"):
            continue
        data_dict = leaf.to_dict()

        if task_type in ("Classification", "MultilabelClassification"):
            for col in ("text", "texts", "sentence", "content"):
                if col in data_dict:
                    _add(data_dict[col])
                    break

        elif task_type in ("STS", "PairClassification", "BitextMining"):
            for col in ("sentence1", "sentence2"):
                if col in data_dict:
                    _add(data_dict[col])

        elif task_type == "Clustering":
            for col in ("sentences", "text", "texts"):
                if col in data_dict:
                    _add(data_dict[col])
                    break

        else:
            for col in ("text", "sentence1", "sentence2", "sentences"):
                if col in data_dict:
                    _add(data_dict[col])

    return _unique_nonempty_stripped(raw)


def extract_retrieval_texts(task) -> Dict[str, List[str]]:
    """Return {'queries': [...], 'corpus': [...]} for retrieval tasks."""
    dataset = _resolve_dataset(task)
    split_name = "test"
    if split_name not in dataset:
        split_name = list(dataset.keys())[0]

    val_data = dataset[split_name]
    data_dict = val_data.to_dict() if hasattr(val_data, "to_dict") else dict(val_data)

    queries_raw: List[str] = []
    queries = data_dict.get("queries", {})
    if isinstance(queries, dict):
        queries_raw = [str(v) for v in queries.values() if v]
    elif hasattr(queries, "column_names"):
        for col in ("text", "query"):
            if col in queries.column_names:
                queries_raw = [str(t) for t in queries[col] if t]
                break

    corpus = data_dict.get("corpus") or dataset.get("corpus", {})
    corpus_raw: List[str] = list(iter_retrieval_corpus_passages(corpus))

    return {
        "queries": _unique_nonempty_stripped(queries_raw),
        "corpus": _unique_nonempty_stripped(corpus_raw),
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  Multi-GPU embedding precompute (orchestrated here, not in LayerEmbeddingStore) #
# ══════════════════════════════════════════════════════════════════════════ #
#
# Uses the same high-level pattern as layer-spec parallel work: ProcessPoolExecutor +
# per-worker tasks. GPU shard workers use ``spawn`` (not ``fork``): the main script has
# usually already initialized CUDA (model probe), and fork+CUDA in children is unsafe.


def _embedding_precompute_shard_worker(payload: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        shard_texts,
        tmp_path,
        model_name,
        n_layers,
        poolings,
        pooling0,
        batch_size,
        device,
        trust_remote_code,
        torch_dtype,
        shard_id,
    ) = payload

    from src.aggregated_encoder import LayerEncoder

    started_total = time.perf_counter()
    started_load = time.perf_counter()
    layer_enc = LayerEncoder(
        model_name=model_name,
        pooling=pooling0,
        batch_size=batch_size,
        device=device,
        use_cache=False,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    if int(layer_enc.num_layers) != int(n_layers):
        raise RuntimeError(
            f"precompute shard {shard_id}: layer count mismatch "
            f"(expect {n_layers}, model returned {layer_enc.num_layers})"
        )
    load_s = time.perf_counter() - started_load

    started_encode = time.perf_counter()
    if len(poolings) == 1:
        p = poolings[0]
        accum: Dict[int, List[np.ndarray]] = {i: [] for i in range(n_layers)}
        for start in range(0, len(shard_texts), batch_size):
            batch = shard_texts[start : start + batch_size]
            all_layer_embs = layer_enc.encode_batch(batch, return_all_layers=True)
            for layer_idx, emb in enumerate(all_layer_embs):
                accum[layer_idx].append(np.asarray(emb, dtype=np.float32))
        layer_embs = {i: np.concatenate(accum[i], axis=0) for i in range(n_layers)}
        stub = LayerEmbeddingStore(
            model_name=model_name,
            n_layers=n_layers,
            pooling=p,
            poolings=[p],
            batch_size=batch_size,
            device=device,
            cache_dir=".",
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        stub._save_to_hdf5_legacy(tmp_path, shard_texts, layer_embs)
    else:
        accum_mp: Dict[str, Dict[int, List[np.ndarray]]] = {
            p: {i: [] for i in range(n_layers)} for p in poolings
        }
        for start in range(0, len(shard_texts), batch_size):
            batch = shard_texts[start : start + batch_size]
            multi = layer_enc.encode_batch_multi_poolings(batch, poolings)
            for p in poolings:
                for layer_idx, emb in enumerate(multi[p]):
                    accum_mp[p][layer_idx].append(np.asarray(emb, dtype=np.float32))
        pooled = {
            p: {i: np.concatenate(accum_mp[p][i], axis=0) for i in range(n_layers)}
            for p in poolings
        }
        stub = LayerEmbeddingStore(
            model_name=model_name,
            n_layers=n_layers,
            pooling=poolings[0],
            poolings=poolings,
            batch_size=batch_size,
            device=device,
            cache_dir=".",
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        stub._save_to_hdf5_multi(tmp_path, shard_texts, pooled)

    del layer_enc
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {
        "tmp_path": str(tmp_path),
        "device": device,
        "shard_id": shard_id,
        "n_texts": len(shard_texts),
        "load_s": load_s,
        "encode_save_s": time.perf_counter() - started_encode,
        "total_s": time.perf_counter() - started_total,
    }


def _run_embedding_precompute_sharded(
    *,
    texts: List[str],
    cache_file: Path,
    model_name: str,
    n_layers: int,
    poolings: List[str],
    batch_size: int,
    devices: List[str],
    trust_remote_code: bool,
    torch_dtype: Optional[str],
) -> None:
    """Round-robin shard texts across ``devices``; each worker loads one model on one GPU."""
    if len(devices) <= 1:
        raise RuntimeError("sharded precompute requires 2+ devices")
    devs = list(devices)
    shards: List[List[str]] = [[] for _ in devs]
    for i, t in enumerate(texts):
        shards[i % len(devs)].append(t)

    tmp_parts = [str(cache_file) + f".part{sid}.h5.tmp" for sid in range(len(devs))]
    payloads = []
    for shard_id, (dv, shard_texts, tmp) in enumerate(zip(devs, shards, tmp_parts)):
        payloads.append(
            (
                shard_texts,
                tmp,
                model_name,
                n_layers,
                poolings,
                poolings[0],
                batch_size,
                dv,
                trust_remote_code,
                torch_dtype,
                shard_id,
            )
        )

    mp_ctx = multiprocessing.get_context("spawn")
    out_tmp = str(cache_file) + ".tmp"
    try:
        started_total = time.perf_counter()
        logger.info(
            "Embedding precompute: ProcessPoolExecutor(%d, spawn) shard workers → %s",
            len(devs),
            cache_file.name,
        )
        with ProcessPoolExecutor(max_workers=len(devs), mp_context=mp_ctx) as ex:
            fut_to_payload = {
                ex.submit(_embedding_precompute_shard_worker, payload): payload
                for payload in payloads
            }
            for fut in as_completed(fut_to_payload):
                stats = fut.result()
                logger.info(
                    "[profile] embedding_shard_worker | %.3fs | model=%s | shard=%s | device=%s | texts=%s | "
                    "load_s=%.3f | encode_save_s=%.3f",
                    float(stats["total_s"]),
                    model_name,
                    stats["shard_id"],
                    stats["device"],
                    stats["n_texts"],
                    float(stats["load_s"]),
                    float(stats["encode_save_s"]),
                )

        started_merge = time.perf_counter()
        if os.path.exists(out_tmp):
            os.unlink(out_tmp)
        LayerEmbeddingStore.merge_shards_to_hdf5(
            tmp_parts,
            out_tmp,
            texts,
            n_layers=n_layers,
            poolings=poolings,
        )
        os.replace(out_tmp, str(cache_file))
        _profile_log(
            "embedding_precompute_sharded_merge",
            started_merge,
            model=model_name,
            cache_file=cache_file.name,
            texts=len(texts),
            devices=len(devs),
        )
        _profile_log(
            "embedding_precompute_sharded_total",
            started_total,
            model=model_name,
            cache_file=cache_file.name,
            texts=len(texts),
            devices=",".join(devs),
        )
    finally:
        for p in tmp_parts:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass
        try:
            if os.path.exists(out_tmp):
                os.unlink(out_tmp)
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════════════════ #
#  MTEB evaluation helpers                                                   #
# ══════════════════════════════════════════════════════════════════════════ #

def run_mteb_eval(
    encoder,
    task,
    output_dir: str,
    *,
    batch_size: int = 32,
    prediction_folder: Optional[Path] = None,
) -> Optional[float]:
    """Run MTEB on ``task`` using ``mteb.evaluate`` (test split only, no global cache)."""
    task_name = task.metadata.name
    started = time.perf_counter()
    if prediction_folder is None:
        raw_dir = Path(output_dir) / "mteb_raw"
    else:
        raw_dir = Path(prediction_folder)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prev_eval_splits = getattr(task, "_eval_splits", None)
    try:
        eval_names = list(task.eval_splits)
        if "test" in eval_names:
            task.filter_eval_splits(["test"])
        result = mteb.evaluate(
            encoder,
            task,
            co2_tracker=False,
            raise_error=False,
            encode_kwargs={
                "batch_size": batch_size,
                "show_progress_bar": False,
            },
            cache=None,
            overwrite_strategy="always",
            prediction_folder=raw_dir,
            show_progress_bar=False,
        )
        score = _main_score_from_model_result(result, task_name)
        _profile_log("mteb_evaluate", started, task=task_name, batch_size=batch_size, score=score)
        return score
    except Exception as e:
        logger.error("MTEB eval failed for %s: %s", task_name, e)
        return None
    finally:
        task._eval_splits = prev_eval_splits  # type: ignore[attr-defined]


def _main_score_from_model_result(result: Any, task_name: str) -> Optional[float]:
    if getattr(result, "exceptions", None):
        for ex in result.exceptions:
            logger.error("MTEB task error %s: %s", ex.task_name, ex.exception)
    if not result.task_results:
        return None
    tr = result.task_results[0]
    try:
        splits = list(tr.scores.keys())
        if "test" in splits:
            return float(tr.get_score(splits=["test"]))
        if splits:
            return float(tr.get_score(splits=[splits[0]]))
    except Exception as e:
        logger.debug("_main_score_from_model_result error for %s: %s", task_name, e)
    return None


# ══════════════════════════════════════════════════════════════════════════ #
#  Parallel layer-spec workers (fork + inherited store; Unix fork only)      #
# ══════════════════════════════════════════════════════════════════════════ #

_LAYER_SPEC_WORKER_CTX: Optional[Dict[str, Any]] = None
_WORKER_TASK_BY_NAME: Dict[str, Any] = {}


def _fork_layer_spec_parallel_available() -> bool:
    if sys.platform == "win32":
        return False
    try:
        multiprocessing.get_context("fork")
        return True
    except ValueError:
        return False


def _sanitize_spec_for_path(spec_name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", spec_name).strip("_")
    return (s or "spec")[:120]


def _effective_layer_spec_workers(requested: int, n_pending: int) -> int:
    if requested <= 1 or n_pending <= 1:
        return 1
    if not _fork_layer_spec_parallel_available():
        return 1
    cpu = os.cpu_count() or 4
    # Avoid spawning hundreds of processes on large machines.
    cap = max(2, min(32, cpu // 4))
    return min(int(requested), n_pending, cap)


def _worker_load_task(task_name: str):
    if task_name not in _WORKER_TASK_BY_NAME:
        t = mteb.get_tasks(tasks=[task_name], languages=["eng"])[0]
        t.load_data()
        _WORKER_TASK_BY_NAME[task_name] = t
    return _WORKER_TASK_BY_NAME[task_name]


def _spec_for_name(spec_name: str, n_layers: int) -> LayerSpec:
    for s in build_layer_specs(n_layers):
        if s.name == spec_name:
            return s
    raise KeyError(f"Unknown layer_spec {spec_name!r}")


def _make_fork_safe_store_data(store: PooledLayerEmbeddingView) -> Dict[str, Any]:
    """Create a plain dict snapshot of the store that is safe to pass across fork.
    Copies the critical numpy arrays so workers do not share mutable state with parent.
    """
    embeddings = {}
    for layer_idx, arr in store._embeddings.items():
        embeddings[layer_idx] = arr.copy()  # ensure independent memory

    overflow = {}
    for sent, layers in store._overflow.items():
        overflow[sent] = {k: v.copy() for k, v in layers.items()}

    return {
        "text_index": dict(store._text_index),  # copy dict
        "embeddings": embeddings,
        "overflow": overflow,
        "n_layers": store.n_layers,
        "pooling": store.pooling,
        "model_name": store.model_name,
    }


class _ForkSafeStoreView:
    """Minimal view that mimics PooledLayerEmbeddingView for workers.
    Uses the snapshot dict so no complex parent object is shared across fork.
    """
    def __init__(self, data: Dict[str, Any]):
        self._text_index = data["text_index"]
        self._embeddings = data["embeddings"]
        self._overflow = data["overflow"]
        self.n_layers = data["n_layers"]
        self.pooling = data["pooling"]
        self.model_name = data["model_name"]

    def get_aggregated(self, texts: List[str], weights: np.ndarray) -> np.ndarray:
        indices = [self._text_index[t] for t in texts]
        em = self._embeddings
        return sum(
            weights[i] * em[i][indices]
            for i in range(self.n_layers)
        )


def _init_layer_spec_worker_process() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Forked workers inherit ``logging`` locks/handlers from the parent; emitting
    # INFO from workers can deadlock or drop lines on stdout. Rebuild logging
    # in the child before any worker logs (see Python logging + fork issues).
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        force=True,
    )
    _silence_noisy_loggers()
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass
    try:
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


def _layer_spec_worker_error_row(
    model_name: str,
    task_name: str,
    task_type: str,
    pooling: str,
    spec_name: str,
    err: str,
) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "task_name": task_name,
        "task_type": task_type,
        "pooling": pooling,
        "layer_spec": spec_name,
        "mteb_score": "",
        "unsup_metrics_hash": "",
        "metric_error": err,
    }


def _layer_spec_worker(
    item: Tuple[int, str, Optional[Dict[str, Any]], Optional[List[str]], bool]
) -> Dict[str, Any]:
    si, spec_name, existing_row, selected_metrics, run_mteb = item
    ctx = _LAYER_SPEC_WORKER_CTX
    if ctx is None:
        print("WORKER ERROR: context missing for", spec_name, flush=True)
        return _layer_spec_worker_error_row(
            "", "", "", "", spec_name, "parallel worker context missing"
        )
    try:
        task = ctx["task"]  # pre-loaded in parent to avoid mteb.get_tasks in forked workers
        spec = _spec_for_name(spec_name, int(ctx["n_layers"]))
        pred = Path(ctx["output_dir"]) / "mteb_raw" / _sanitize_spec_for_path(spec_name)
        store_view = _ForkSafeStoreView(ctx["store_data"])
        _worker_emit_layer_spec_started(ctx, si, spec_name, emit_log=True)
        row = _evaluate_layer_spec_row(
            model_name=ctx["model_name"],
            task=task,
            pooling=ctx["pooling"],
            store=store_view,  # now a lightweight view
            spec=spec,
            n_layers=int(ctx["n_layers"]),
            task_name=ctx["task_name"],
            task_type=ctx["task_type"],
            is_retrieval=bool(ctx["is_retrieval"]),
            query_texts=ctx["query_texts"],
            corpus_texts=ctx["corpus_texts"],
            all_texts=ctx["all_texts"],
            metric_kwargs=ctx["metric_kwargs"],
            output_dir=ctx["output_dir"],
            batch_size=int(ctx["batch_size"]),
            encoder_device="cpu",
            mteb_prediction_folder=pred,
            mteb_gpu_proxy=bool(ctx.get("mteb_gpu_proxy")),
            mteb_proxy_device=str(ctx.get("mteb_proxy_device", "cpu")),
            mteb_proxy_mem_fraction=float(ctx.get("mteb_proxy_mem_fraction", 0.72)),
            mteb_proxy_query_batch=ctx.get("mteb_proxy_query_batch"),
            mteb_proxy_corpus_chunk=ctx.get("mteb_proxy_corpus_chunk"),
            existing_row=existing_row,
            selected_metrics=selected_metrics,
            run_mteb=run_mteb,
            unsup_metrics_hash=str(ctx.get("metric_config_hash", "")),
        )
        print(f"WORKER {os.getpid()} FINISHED spec={spec_name} mteb={row.get('mteb_score')}", flush=True)
        _worker_emit_layer_row_progress(ctx, si, row)
        return row
    except Exception as e:
        import traceback
        print("WORKER EXCEPTION for", spec_name, ":", e, flush=True)
        traceback.print_exc()
        row = _layer_spec_worker_error_row(
            str(ctx.get("model_name", "")),
            str(ctx.get("task_name", "")),
            str(ctx.get("task_type", "")),
            str(ctx.get("pooling", "")),
            spec_name,
            str(e),
        )
        _worker_emit_layer_row_progress(ctx, si, row)
        return row


def _evaluate_layer_spec_row(
    *,
    model_name: str,
    task,
    pooling: str,
    store: PooledLayerEmbeddingView,
    spec: LayerSpec,
    n_layers: int,
    task_name: str,
    task_type: str,
    is_retrieval: bool,
    query_texts: List[str],
    corpus_texts: List[str],
    all_texts: List[str],
    metric_kwargs: Dict,
    output_dir: str,
    batch_size: int,
    encoder_device: str,
    mteb_prediction_folder: Optional[Path] = None,
    mteb_gpu_proxy: bool = False,
    mteb_proxy_device: str = "cuda",
    mteb_proxy_mem_fraction: float = 0.72,
    mteb_proxy_query_batch: Optional[int] = None,
    mteb_proxy_corpus_chunk: Optional[int] = None,
    existing_row: Optional[Dict[str, Any]] = None,
    selected_metrics: Optional[List[str]] = None,
    run_mteb: bool = True,
    unsup_metrics_hash: str = "",
) -> Dict[str, Any]:
    pid = os.getpid()
    started_total = time.perf_counter()
    try:
        unsup_metrics: Dict[str, Any] = {}
        if selected_metrics is None or len(selected_metrics) > 0:
            started_unsup = time.perf_counter()
            metric_kwargs_local = dict(metric_kwargs)
            if selected_metrics is not None:
                metric_kwargs_local["selected_metrics"] = selected_metrics
            if is_retrieval:
                q_emb, c_emb = extract_retrieval_embeddings(
                    store, query_texts, corpus_texts, spec, n_layers
                )
                unsup_raw = compute_metrics_retrieval(
                    query_embs=q_emb, corpus_embs=c_emb, **metric_kwargs_local
                )
                for variant, mdict in unsup_raw.items():
                    unsup_metrics.update(suffix_metrics(mdict, variant))
            else:
                emb_matrix = extract_embedding_matrix(store, all_texts, spec, n_layers)
                unsup_metrics = compute_metrics(emb_matrix, **metric_kwargs_local)
            _profile_log(
                "layer_row_unsup_metrics",
                started_unsup,
                pid=pid,
                model=model_name,
                task=task_name,
                pooling=pooling,
                spec=spec.name,
                retrieval=is_retrieval,
                selected_metrics="all" if selected_metrics is None else len(selected_metrics),
            )
    except Exception as e:
        logger.error("Unsup metrics failed for %s: %s", spec.name, e)
        unsup_metrics = {"metric_error": str(e)}

    if run_mteb:
        started_encoder = time.perf_counter()
        hidden = int(store._embeddings[0].shape[1])
        encoder = StoreBackedAggregatedEncoder(
            model_name=model_name,
            pooling=pooling,
            store=store,
            spec=spec,
            n_layers=n_layers,
            hidden_size=hidden,
            batch_size=batch_size,
            device=encoder_device,
        )
        _profile_log(
            "layer_row_encoder_init",
            started_encoder,
            pid=pid,
            model=model_name,
            task=task_name,
            pooling=pooling,
            spec=spec.name,
            device=encoder_device,
        )
        if mteb_gpu_proxy:
            started_proxy = time.perf_counter()
            mteb_score = gpu_proxy_main_score(
                task,
                store,
                spec,
                n_layers,
                mteb_proxy_device,
                encoder,
                proxy_mem_fraction=mteb_proxy_mem_fraction,
                proxy_query_batch=mteb_proxy_query_batch,
                proxy_corpus_chunk=mteb_proxy_corpus_chunk,
            )
            _profile_log(
                "layer_row_gpu_proxy",
                started_proxy,
                pid=pid,
                model=model_name,
                task=task_name,
                pooling=pooling,
                spec=spec.name,
                device=mteb_proxy_device,
                score=mteb_score,
            )
            if mteb_score is None:
                mteb_score = run_mteb_eval(
                    encoder,
                    task,
                    output_dir,
                    batch_size=batch_size,
                    prediction_folder=mteb_prediction_folder,
                )
        else:
            started_mteb = time.perf_counter()
            mteb_score = run_mteb_eval(
                encoder,
                task,
                output_dir,
                batch_size=batch_size,
                prediction_folder=mteb_prediction_folder,
            )
            _profile_log(
                "layer_row_mteb_eval",
                started_mteb,
                pid=pid,
                model=model_name,
                task=task_name,
                pooling=pooling,
                spec=spec.name,
                score=mteb_score,
            )
    else:
        mteb_score = existing_row.get("mteb_score", "") if existing_row else ""

    row = dict(existing_row or {})
    row.update({
        "model_name": model_name,
        "task_name": task_name,
        "task_type": task_type,
        "pooling": pooling,
        "layer_spec": spec.name,
        "mteb_score": mteb_score if mteb_score is not None else "",
        "unsup_metrics_hash": unsup_metrics_hash,
        "metric_error": unsup_metrics.get("metric_error", ""),
        **unsup_metrics,
    })
    _profile_log(
        "layer_row_total",
        started_total,
        pid=pid,
        model=model_name,
        task=task_name,
        pooling=pooling,
        spec=spec.name,
        run_mteb=run_mteb,
        score=row.get("mteb_score", ""),
        metric_error=row.get("metric_error", ""),
    )
    return row


# ══════════════════════════════════════════════════════════════════════════ #
#  Core evaluation loop: one (model, task, pooling)                          #
# ══════════════════════════════════════════════════════════════════════════ #

def evaluate_model_task_pooling(
    model_name: str,
    task,
    pooling: str,
    layer_specs: List[LayerSpec],
    store: PooledLayerEmbeddingView,
    n_layers: int,
    result_store: ResultStore,
    output_dir: str,
    metric_kwargs: Dict,
    overwrite: bool = False,
    device: str = "cuda",
    trust_remote_code: bool = True,
    torch_dtype: Optional[str] = None,
    progress: Optional[ProgressReporter] = None,
    batch_size: int = 32,
    layer_spec_workers: int = 1,
    mteb_gpu_proxy: bool = False,
    mteb_proxy_mem_fraction: float = 0.72,
    mteb_proxy_query_batch: Optional[int] = None,
    mteb_proxy_corpus_chunk: Optional[int] = None,
) -> None:
    started_total = time.perf_counter()
    task_name = task.metadata.name
    task_type = task.metadata.type
    is_retrieval = task_type in ("Retrieval", "Reranking")

    if is_retrieval:
        retrieval_texts = extract_retrieval_texts(task)
        query_texts  = retrieval_texts["queries"]
        corpus_texts = retrieval_texts["corpus"]
        all_texts    = list(dict.fromkeys(query_texts + corpus_texts))
    else:
        all_texts  = extract_all_texts(task)
        query_texts = corpus_texts = []

    n_task_texts = len(all_texts)
    n_queries = len(query_texts) if is_retrieval else 0
    n_corpus = len(corpus_texts) if is_retrieval else 0
    if is_retrieval:
        metric_subsample_n = max(
            _metric_subsample_rows(len(query_texts), metric_kwargs),
            _metric_subsample_rows(len(corpus_texts), metric_kwargs),
        )
    else:
        metric_subsample_n = _metric_subsample_rows(n_task_texts, metric_kwargs)

    metric_layout = _metric_layout(metric_kwargs)
    current_metric_hash = _unsup_metrics_hash(metric_kwargs)
    existing_rows_lookup: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    try:
        existing_rows, _ = result_store._read_all()
    except Exception:
        existing_rows = []
    for row in existing_rows:
        key = (
            str(row.get("model_name", "")),
            str(row.get("task_name", "")),
            str(row.get("pooling", "")),
            str(row.get("layer_spec", "")),
        )
        existing_rows_lookup[key] = row

    n_specs = len(layer_specs)
    pending: List[tuple[int, str, Optional[Dict[str, Any]], Optional[List[str]], bool, bool]] = []
    for si, spec in enumerate(layer_specs, start=1):
        key = (model_name, task_name, pooling, spec.name)
        cached_row = existing_rows_lookup.get(key)
        row_exists = cached_row is not None
        if cached_row is not None and not overwrite:
            plan = _plan_metric_refresh(
                cached_row,
                metric_layout,
                current_metric_hash,
                retrieval=is_retrieval,
            )
            if plan is None:
                logger.debug("skip (cached): %s", spec.name)
                if progress is not None:
                    progress.flush(
                        silent=True,
                        model=model_name,
                        task=task_name,
                        task_type=task_type,
                        pooling=pooling,
                        layer_spec=spec.name,
                        layer_spec_index=si,
                        layer_spec_count=n_specs,
                        phase="cached_skip",
                        n_layers=n_layers,
                        n_texts=n_task_texts,
                        n_queries=n_queries,
                        n_corpus=n_corpus,
                        metric_subsample_n=metric_subsample_n,
                        last_mteb_score=_last_mteb_score_for_progress_row(cached_row),
                    )
                continue
            selected_metrics, run_mteb = plan
        else:
            selected_metrics = None
            run_mteb = True
        pending.append((si, spec.name, cached_row, selected_metrics, run_mteb, row_exists))

    if not pending:
        return

    nw = _effective_layer_spec_workers(layer_spec_workers, len(pending))
    if layer_spec_workers > 1:
        if len(pending) <= 1:
            logger.debug(
                "layer_spec_workers=%s ignored: only one layer spec pending",
                layer_spec_workers,
            )
        elif not _fork_layer_spec_parallel_available():
            logger.warning(
                "layer_spec_workers=%s requested but fork is unavailable on this platform; using serial.",
                layer_spec_workers,
            )
        elif nw < layer_spec_workers:
            if nw == len(pending):
                logger.info(
                    "layer_spec_workers=%s limited to %s (only %s layer spec(s) pending; rest cached)",
                    layer_spec_workers,
                    nw,
                    len(pending),
                )
            else:
                logger.info(
                    "layer_spec_workers capped from %s to %s (parallel layer-spec CPU cap)",
                    layer_spec_workers,
                    nw,
                )

    mteb_proxy_device = device
    if mteb_gpu_proxy and nw > 1 and str(device).lower().startswith("cuda"):
        logger.warning(
            "MTEB GPU proxy: CUDA disabled with --layer-spec-workers > 1; "
            "using CPU for dense retrieval scores (use --layer-spec-workers 1 for GPU).",
        )
        mteb_proxy_device = "cpu"

    def _flush_layer_row(
        si: int, row: Dict[str, Any], *, silent_progress: bool = False
    ) -> None:
        ms = row.get("mteb_score", "")
        if ms in ("", None):
            score_str = "None"
            last_score = ""
        elif isinstance(ms, (int, float)):
            score_str = f"{float(ms):.4f}"
            last_score = score_str
        else:
            try:
                fv = float(ms)
                score_str = f"{fv:.4f}"
                last_score = score_str
            except (TypeError, ValueError):
                score_str = str(ms)
                last_score = score_str
        logger.debug("layer_spec=%s mteb=%s", row.get("layer_spec"), score_str)
        if progress is not None:
            progress.flush(
                silent=silent_progress,
                model=model_name,
                task=task_name,
                task_type=task_type,
                pooling=pooling,
                layer_spec=str(row.get("layer_spec", "")),
                layer_spec_index=si,
                layer_spec_count=n_specs,
                phase="layer_row",
                n_layers=n_layers,
                n_texts=n_task_texts,
                n_queries=n_queries,
                n_corpus=n_corpus,
                metric_subsample_n=metric_subsample_n,
                last_mteb_score=last_score,
            )

    if nw <= 1:
        for si, spec_name, existing_row, selected_metrics, run_mteb, row_exists in pending:
            spec = _spec_for_name(spec_name, n_layers)
            row = _evaluate_layer_spec_row(
                model_name=model_name,
                task=task,
                pooling=pooling,
                store=store,
                spec=spec,
                n_layers=n_layers,
                task_name=task_name,
                task_type=task_type,
                is_retrieval=is_retrieval,
                query_texts=query_texts,
                corpus_texts=corpus_texts,
                all_texts=all_texts,
                metric_kwargs=metric_kwargs,
                output_dir=output_dir,
                batch_size=batch_size,
                encoder_device=device,
                mteb_prediction_folder=None,
                mteb_gpu_proxy=mteb_gpu_proxy,
                mteb_proxy_device=mteb_proxy_device,
                mteb_proxy_mem_fraction=mteb_proxy_mem_fraction,
                mteb_proxy_query_batch=mteb_proxy_query_batch,
                mteb_proxy_corpus_chunk=mteb_proxy_corpus_chunk,
                existing_row=existing_row,
                selected_metrics=selected_metrics,
                run_mteb=run_mteb,
                unsup_metrics_hash=current_metric_hash,
            )
            if row_exists:
                result_store.upsert(row)
            else:
                result_store.append(row)
            _flush_layer_row(si, row)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return

    global _LAYER_SPEC_WORKER_CTX
    logger.info(
        "Layer-spec parallel: %d workers (fork) for %d specs on %s / %s / %s",
        nw,
        len(pending),
        model_name,
        task_name,
        pooling,
    )
    # Pre-load the task in the parent so workers don't call mteb.get_tasks + load_data()
    # (this is a common source of fork-related hangs with HF datasets).
    task_for_workers = _worker_load_task(task_name)  # uses the shared _WORKER_TASK_BY_NAME
    store_data = _make_fork_safe_store_data(store)
    _LAYER_SPEC_WORKER_CTX = {
        "model_name": model_name,
        "task_name": task_name,
        "task_type": task_type,
        "pooling": pooling,
        "task": task_for_workers,      # pre-loaded task object
        "store_data": store_data,      # lightweight dict instead of full view
        "n_layers": n_layers,
        "is_retrieval": is_retrieval,
        "query_texts": query_texts,
        "corpus_texts": corpus_texts,
        "all_texts": all_texts,
        "metric_kwargs": metric_kwargs,
        "output_dir": output_dir,
        "batch_size": batch_size,
        "progress_enabled": progress is not None,
        "progress_base": dict(progress._d) if progress is not None else {},
        "n_specs": n_specs,
        "n_task_texts": n_task_texts,
        "n_queries": n_queries,
        "n_corpus": n_corpus,
        "metric_subsample_n": metric_subsample_n,
        "metric_config_hash": current_metric_hash,
        "mteb_gpu_proxy": mteb_gpu_proxy,
        "mteb_proxy_device": mteb_proxy_device,
        "mteb_proxy_mem_fraction": mteb_proxy_mem_fraction,
        "mteb_proxy_query_batch": mteb_proxy_query_batch,
        "mteb_proxy_corpus_chunk": mteb_proxy_corpus_chunk,
    }
    mp_ctx = multiprocessing.get_context("fork")
    stop_hb = threading.Event()
    rp = Path(output_dir).resolve() / "run_progress.json"
    short_m = model_name.split("/")[-1][:48]

    def _parallel_wait_heartbeat() -> None:
        n = 0
        while not stop_hb.wait(120.0):
            n += 1
            logger.info(
                "[parallel-layer-specs] still running: %d workers | %s / %s / %s | "
                "per-spec lines only when workers finish phases; json=%s (heartbeat #%d)",
                nw,
                short_m or model_name,
                task_name,
                pooling,
                rp,
                n,
            )

    hb = threading.Thread(target=_parallel_wait_heartbeat, daemon=True)
    hb.start()
    try:
        with ProcessPoolExecutor(
            max_workers=nw,
            mp_context=mp_ctx,
            initializer=_init_layer_spec_worker_process,
        ) as ex:
            rows = list(ex.map(_layer_spec_worker, pending))
    finally:
        stop_hb.set()
        _LAYER_SPEC_WORKER_CTX = None

    for (si, _sn, existing_row, _selected_metrics, _run_mteb, row_exists), row in sorted(
        zip(pending, rows), key=lambda z: z[0][0]
    ):
        if row_exists:
            result_store.upsert(row)
        else:
            result_store.append(row)
        _flush_layer_row(si, row, silent_progress=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _profile_log(
        "evaluate_model_task_pooling",
        started_total,
        model=model_name,
        task=task_name,
        pooling=pooling,
        pending_specs=len(pending),
        layer_spec_workers=nw,
    )


# ══════════════════════════════════════════════════════════════════════════ #
#  Top-level orchestration                                                   #
# ══════════════════════════════════════════════════════════════════════════ #

def run_evaluation(args) -> None:
    tasks = load_tasks(
        task_set=args.task_set,
        tasks=args.tasks,
        task_types=args.task_types,
        max_samples=args.max_samples,
    )

    result_store = ResultStore(os.path.join(args.output_dir, "master_results.csv"))

    metric_kwargs = dict(
        n_samples=args.n_samples,
        sample_fraction=args.sample_fraction,
        min_sample_size=args.min_sample_size,
        include_ph_dim=args.include_ph_dim,
        ripser_maxdim=args.ripser_maxdim,
    )

    eval_torch_dtype = _normalize_torch_dtype_str(getattr(args, "torch_dtype", None))
    if eval_torch_dtype is None:
        eval_torch_dtype = _default_embedding_weight_dtype(args.device)

    precompute_devs = _embedding_precompute_devices(
        device=args.device,
        override=_parse_precompute_devices_arg(
            getattr(args, "embedding_precompute_devices", None),
            fallback_device=args.device,
        ),
    )
    logger.info(
        "Embedding dtype (HF weights): %s | precompute devices: %s",
        eval_torch_dtype,
        ", ".join(precompute_devs),
    )

    embedding_batch_size = int(args.batch_size)
    if getattr(args, "auto_embedding_batch", False):
        embedding_batch_size = _infer_embedding_batch_size(
            device=args.device,
            torch_dtype=eval_torch_dtype,
            base=int(args.batch_size),
        )
        if embedding_batch_size != int(args.batch_size):
            logger.info(
                "Embedding batch size (auto): %s → %s (device=%s, torch_dtype=%s)",
                int(args.batch_size),
                embedding_batch_size,
                args.device,
                eval_torch_dtype,
            )

    total_steps = _count_progress_steps(args.models, tasks, args.poolings)
    want_pbar = getattr(args, "progress_bar", True)
    if want_pbar and tqdm is None:
        logger.warning("Install tqdm for a progress bar: pip install tqdm")
    pbar = _make_progress_bar(total_steps, want_pbar)

    progress = ProgressReporter(args.output_dir)
    progress.set_totals(total_steps)

    try:
        for model_name in args.models:
            progress.flush(
                model=model_name,
                task="—",
                task_type="",
                pooling="",
                layer_spec="",
                layer_spec_index=0,
                layer_spec_count=0,
                phase="model_load",
                last_mteb_score="",
            )
            if pbar is not None:
                pbar.set_postfix_str(model_name[:42] + ("…" if len(model_name) > 42 else ""), refresh=False)

            # Probe model once to get n_layers
            try:
                started_probe = time.perf_counter()
                probe = LayerEncoder(
                    model_name=model_name,
                    pooling="mean",
                    batch_size=1,
                    device=args.device,
                    use_cache=False,
                    trust_remote_code=args.trust_remote_code,
                    torch_dtype=eval_torch_dtype,
                )
                n_layers = probe.num_layers
                del probe
                gc.collect()
                _profile_log("model_probe", started_probe, model=model_name, n_layers=n_layers, device=args.device)
            except Exception as e:
                logger.error(f"    Cannot load model {model_name}: {e}")
                _pbar_update(pbar, _steps_skipped_for_task(model_name, args.poolings) * len(tasks))
                _pbar_sync_progress(progress, pbar)
                continue

            layer_specs = build_layer_specs(n_layers)
            logger.debug(
                "%s layer specs: %s … %s",
                len(layer_specs),
                [s.name for s in layer_specs[:5]],
                layer_specs[-1].name,
            )
            progress.flush(
                model=model_name,
                task="—",
                task_type="",
                pooling="",
                layer_spec="",
                layer_spec_index=0,
                layer_spec_count=len(layer_specs),
                phase="model_ready",
                n_layers=n_layers,
                last_mteb_score="",
            )

            for task in tasks:
                task_name = task.metadata.name
                progress.flush(
                    model=model_name,
                    task=task_name,
                    task_type=task.metadata.type,
                    pooling="",
                    layer_spec="",
                    layer_spec_index=0,
                    layer_spec_count=len(layer_specs),
                    phase="task_load",
                    n_layers=n_layers,
                    n_texts=0,
                    n_queries=0,
                    n_corpus=0,
                    metric_subsample_n=0,
                    last_mteb_score="",
                )

                try:
                    started_task_load = time.perf_counter()
                    task.load_data()
                    _profile_log("task_load_data", started_task_load, model=model_name, task=task_name)
                except Exception as e:
                    logger.error(f"    Cannot load {task_name}: {e}")
                    _pbar_update(pbar, _steps_skipped_for_task(model_name, args.poolings))
                    _pbar_sync_progress(progress, pbar)
                    continue

                valid_poolings = [
                    p for p in args.poolings if pooling_supported(model_name, p)
                ]
                for pooling in args.poolings:
                    if not pooling_supported(model_name, pooling):
                        logger.debug(
                            "skip (incompatible pooling): %r for %s — %s",
                            pooling,
                            model_name,
                            skip_reason(model_name, pooling),
                        )
                if not valid_poolings:
                    continue

                store = None
                try:
                    started_task_prepare = time.perf_counter()
                    all_texts = extract_all_texts(task)
                    _tt = task.metadata.type
                    if _tt in ("Retrieval", "Reranking"):
                        _rt = extract_retrieval_texts(task)
                        _nq, _nc = len(_rt["queries"]), len(_rt["corpus"])
                        _msn = max(
                            _metric_subsample_rows(_nq, metric_kwargs),
                            _metric_subsample_rows(_nc, metric_kwargs),
                        )
                    else:
                        _nq, _nc = 0, 0
                        _msn = _metric_subsample_rows(len(all_texts), metric_kwargs)
                    _profile_log(
                        "task_prepare_texts",
                        started_task_prepare,
                        model=model_name,
                        task=task_name,
                        task_type=_tt,
                        texts=len(all_texts),
                        queries=_nq,
                        corpus=_nc,
                    )

                    cache_emb_dir = Path(args.output_dir) / "embedding_cache"
                    cache_emb_dir.mkdir(parents=True, exist_ok=True)
                    cache_h5 = layer_store_hdf5_path(
                        cache_emb_dir,
                        model_name,
                        task_name,
                        "test",
                        valid_poolings,
                        n_layers,
                    )
                    if len(precompute_devs) > 1:
                        need_shards = (not cache_h5.exists()) or (
                            not LayerEmbeddingStore._hdf5_is_valid(str(cache_h5))
                        )
                        if need_shards:
                            if cache_h5.exists():
                                cache_h5.unlink()
                            started_sharded_precompute = time.perf_counter()
                            _run_embedding_precompute_sharded(
                                texts=all_texts,
                                cache_file=cache_h5,
                                model_name=model_name,
                                n_layers=n_layers,
                                poolings=valid_poolings,
                                batch_size=embedding_batch_size,
                                devices=precompute_devs,
                                trust_remote_code=args.trust_remote_code,
                                torch_dtype=eval_torch_dtype,
                            )
                            _profile_log(
                                "task_precompute_sharded_cache",
                                started_sharded_precompute,
                                model=model_name,
                                task=task_name,
                                cache_file=cache_h5.name,
                                texts=len(all_texts),
                                devices=",".join(precompute_devs),
                            )

                    store = LayerEmbeddingStore(
                        model_name=model_name,
                        n_layers=n_layers,
                        poolings=valid_poolings,
                        batch_size=embedding_batch_size,
                        device=args.device,
                        cache_dir=str(cache_emb_dir),
                        trust_remote_code=args.trust_remote_code,
                        torch_dtype=eval_torch_dtype,
                    )
                    started_store_ready = time.perf_counter()
                    progress.flush(
                        model=model_name,
                        task=task_name,
                        task_type=task.metadata.type,
                        pooling="+".join(valid_poolings),
                        layer_spec="",
                        layer_spec_index=0,
                        layer_spec_count=len(layer_specs),
                        phase="embedding_cache",
                        n_texts=len(all_texts),
                        n_layers=n_layers,
                        n_queries=_nq,
                        n_corpus=_nc,
                        metric_subsample_n=_msn,
                        last_mteb_score="",
                    )
                    store.precompute_or_load(
                        texts=all_texts,
                        dataset_name=task_name,
                        split_name="test",
                    )
                    _profile_log(
                        "task_embedding_cache_ready",
                        started_store_ready,
                        model=model_name,
                        task=task_name,
                        cache_file=cache_h5.name,
                        texts=len(all_texts),
                        poolings="+".join(valid_poolings),
                    )
                    progress.flush(
                        model=model_name,
                        task=task_name,
                        task_type=task.metadata.type,
                        pooling="+".join(valid_poolings),
                        layer_spec="",
                        layer_spec_index=0,
                        layer_spec_count=len(layer_specs),
                        phase="embeddings_ready",
                        n_texts=len(all_texts),
                        n_layers=n_layers,
                        n_queries=_nq,
                        n_corpus=_nc,
                        metric_subsample_n=_msn,
                        last_mteb_score="",
                    )
                except Exception as e:
                    logger.error(
                        f"      Failed to prepare or precompute "
                        f"({model_name} / {task_name} / poolings={valid_poolings}): {e}",
                        exc_info=True,
                    )
                    _pbar_update(pbar, len(valid_poolings))
                    _pbar_sync_progress(progress, pbar)
                    if store is not None:
                        del store
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    continue

                try:
                    for pooling in valid_poolings:
                        started_pooling = time.perf_counter()
                        progress.flush(
                            model=model_name,
                            task=task_name,
                            task_type=task.metadata.type,
                            pooling=pooling,
                            layer_spec="",
                            layer_spec_index=0,
                            layer_spec_count=len(layer_specs),
                            phase="pooling_start",
                            n_layers=n_layers,
                            n_texts=len(all_texts),
                            n_queries=_nq,
                            n_corpus=_nc,
                            metric_subsample_n=_msn,
                            last_mteb_score="",
                        )
                        store_view = store.as_pooling(pooling)
                        try:
                            evaluate_model_task_pooling(
                                model_name=model_name,
                                task=task,
                                pooling=pooling,
                                layer_specs=layer_specs,
                                store=store_view,
                                n_layers=n_layers,
                                result_store=result_store,
                                output_dir=args.output_dir,
                                metric_kwargs=metric_kwargs,
                                overwrite=args.overwrite,
                                device=args.device,
                                trust_remote_code=args.trust_remote_code,
                                torch_dtype=eval_torch_dtype,
                                progress=progress,
                                batch_size=embedding_batch_size,
                                layer_spec_workers=args.layer_spec_workers,
                                mteb_gpu_proxy=args.mteb_gpu_proxy,
                                mteb_proxy_mem_fraction=args.mteb_proxy_mem_fraction,
                                mteb_proxy_query_batch=args.mteb_proxy_query_batch,
                                mteb_proxy_corpus_chunk=args.mteb_proxy_corpus_chunk,
                            )
                        except Exception as e:
                            logger.error(
                                f"      Evaluation failed "
                                f"({model_name} / {task_name} / {pooling}): {e}",
                                exc_info=True,
                            )
                        finally:
                            _pbar_update(pbar, 1)
                            _pbar_sync_progress(progress, pbar)
                            _profile_log(
                                "task_pooling_eval",
                                started_pooling,
                                model=model_name,
                                task=task_name,
                                pooling=pooling,
                                n_layers=n_layers,
                            )
                finally:
                    del store
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        _pbar_close(pbar)

    progress.flush(
        phase="finished",
        task="—",
        pooling="",
        layer_spec="",
        layer_spec_index=0,
        layer_spec_count=0,
        last_mteb_score="",
    )
    logger.info("Done. Results: %s", result_store.path)


# ══════════════════════════════════════════════════════════════════════════ #
#  CLI                                                                        #
# ══════════════════════════════════════════════════════════════════════════ #

def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised embedding quality metrics + MTEB test evaluation"
    )

    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        metavar="MODEL",
        help="HuggingFace model ids. If omitted, use --model-set (default: core).",
    )
    parser.add_argument(
        "--model-set",
        default="core",
        choices=["core", "standard", "full"],
        help="Predefined model list when --models is omitted (see src/model_sets.py).",
    )
    parser.add_argument(
        "--task-set", default="core", choices=["core", "standard", "full"],
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Explicit MTEB task names (overrides --task-set)",
    )
    parser.add_argument(
        "--task-types", nargs="+", default=None,
        choices=["Classification", "Clustering", "PairClassification",
                 "Reranking", "Retrieval", "STS", "Summarization"],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--poolings", nargs="+", default=["mean"],
        choices=["mean", "cls", "last_token"],
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Bootstrap draws for unsupervised topology/spectral metrics (per variant); "
        "was 10 historically, default 1 is faster.",
    )
    parser.add_argument("--sample-fraction",  type=float, default=1/20)
    parser.add_argument("--min-sample-size",  type=int,   default=100)
    parser.add_argument("--include-ph-dim",   action="store_true")
    parser.add_argument("--batch-size",       type=int,   default=32)
    parser.add_argument(
        "--auto-embedding-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled (default), bump ``--batch-size`` heuristically on CUDA when using "
            "low-precision weights (bf16/fp16). Disable with ``--no-auto-embedding-batch`` "
            "for strict manual control."
        ),
    )
    parser.add_argument(
        "--embedding-precompute-devices",
        default=None,
        metavar="SPEC",
        help=(
            "When 2+ devices are selected, ``run_unsup_eval`` runs embedding precompute in "
            "parallel (ProcessPoolExecutor + spawn; one HF model per GPU), then merges shards "
            "into the usual HDF5 cache. "
            "Default: if ``--device cuda`` (no index) and multiple GPUs are visible, use "
            "``cuda:0..cuda:N-1``; if ``--device cuda:K``, only that GPU. "
            "Override: ``cuda`` (all visible GPUs) or ``cuda:0,cuda:1``."
        ),
    )
    parser.add_argument(
        "--layer-spec-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Evaluate multiple layer specs in parallel after embeddings are ready "
            "(Unix fork only; separate MTEB prediction dirs per spec; workers use CPU). "
            "Default 1 = serial. Ignored when fork is unavailable (e.g. Windows)."
        ),
    )
    parser.add_argument(
        "--mteb-gpu-proxy",
        action="store_true",
        default=False,
        help=(
            "Use precomputed embeddings instead of mteb.evaluate encoding: dense GPU "
            "similarity for Retrieval/Reranking; STS/PairClassification (GPU cosines + "
            "MTEB metrics); Classification tasks use a GPU logistic-regression proxy on cached "
            "vectors when CUDA is available. Multilabel classification still uses cached vectors. "
            "BitextMining, Summarization, Clustering, Zero-shot, etc. still call mteb.evaluate. "
            "With --layer-spec-workers > 1, retrieval proxy matmul uses CPU."
        ),
    )
    parser.add_argument(
        "--mteb-proxy-mem-fraction",
        type=float,
        default=0.72,
        metavar="F",
        help=(
            "With --mteb-gpu-proxy: fraction of free VRAM (after Q/C on device) to allow for "
            "batched similarity + top-k scratch. Lower if you see OOM; raise (e.g. 0.85) on "
            "idle A100s to grow query_batch × corpus_chunk."
        ),
    )
    parser.add_argument(
        "--mteb-proxy-query-batch",
        type=int,
        default=None,
        metavar="B",
        help="Override auto-tuned number of queries scored together on GPU (default: infer from VRAM).",
    )
    parser.add_argument(
        "--mteb-proxy-corpus-chunk",
        type=int,
        default=None,
        metavar="C",
        help="Override auto-tuned corpus slice size per matmul (default: infer from VRAM).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", default="./results/unsup_eval")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-evaluate already-computed configs",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        default=False,
        help="Pass trust_remote_code=False to HuggingFace (stricter; may break Mamba/custom models).",
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        metavar="DTYPE",
        help=(
            "HF ``from_pretrained(torch_dtype=...)`` for embedding extraction / probe. "
            "Examples: bfloat16, float16, float32. "
            "If omitted on CUDA with bf16 support, defaults to bfloat16 weights for speed/VRAM."
        ),
    )
    parser.add_argument(
        "--ripser-maxdim",
        type=int,
        default=1,
        metavar="D",
        help="Ripser max homology dimension: 0 = H0 only (faster), 1 = H0+H1 (default).",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        default=False,
        help="Disable tqdm progress bar (writes to stderr by default).",
    )

    args = parser.parse_args()
    if args.layer_spec_workers < 1:
        parser.error("--layer-spec-workers must be at least 1")
    args.trust_remote_code = not args.no_trust_remote_code
    args.progress_bar = not args.no_progress_bar
    if not args.models:
        args.models = list(MODEL_SET_MAP[args.model_set])
    logger.info("Models to run (%d): %s", len(args.models), args.models)
    run_evaluation(args)


if __name__ == "__main__":
    main()
