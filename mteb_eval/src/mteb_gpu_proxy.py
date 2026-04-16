"""
Dense GPU scoring for MTEB retrieval / reranking tasks using the same qrels,
k-values, and metric code as MTEB (``calculate_retrieval_scores`` +
``make_score_dict`` + ``task.task_specific_scores``).

Other task types are not accelerated here; the driver script keeps using
``mteb.evaluate`` with the store-backed encoder for those.

VRAM-shaped batching: ``infer_proxy_batch_sizes`` sizes ``query_batch`` and
``corpus_chunk`` from ``torch.cuda.mem_get_info`` after embeddings are on
device. With two A100s, point ``--device cuda:0`` or ``cuda:1`` (or run two
jobs with ``CUDA_VISIBLE_DEVICES``) so sizing uses the intended GPU's free memory.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from mteb.abstasks.retrieval import _filter_queries_without_positives
from mteb.models.model_meta import ScoringFunction
from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores, make_score_dict
from mteb.results.task_result import TaskResult

from src.embedding_extractor import extract_embedding_matrix
from src.layer_spec import LayerSpec
from src.mteb_text_align import retrieval_corpus_text_for_encode

logger = logging.getLogger(__name__)

# Peak scratch (sims + merged top-k workspace) as a multiple of B*chunk floats; keep conservative.
_SCRATCH_FLOAT_MULT = 3.0


def _finite_score_or_none(score: Any) -> Optional[float]:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _cuda_device_index(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    idx = device.index
    if idx is None:
        return torch.cuda.current_device()
    return int(idx)


def infer_proxy_batch_sizes(
    *,
    device: torch.device,
    n_queries: int,
    n_corpus: int,
    hidden_dim: int,
    mem_fraction: float = 0.72,
    min_query_batch: int = 1,
    min_corpus_chunk: int = 1024,
    max_corpus_chunk: int = 524_288,
    max_query_batch: int = 2048,
) -> Tuple[int, int]:
    """
    Choose (query_batch, corpus_chunk) so similarity scratch fits in free VRAM.

    Assumes query and corpus embeddings are already resident on ``device``; this
    budgets only extra activations: batched ``(B, d) @ (d, chunk)`` scores plus
    top-k merge buffers (see ``_SCRATCH_FLOAT_MULT``).

    For multi-GPU nodes (e.g. 2× A100), pass ``device=torch.device('cuda:0')`` or
    ``cuda:1`` so sizing uses that GPU's free memory.
    """
    n_queries = max(1, int(n_queries))
    n_corpus = max(1, int(n_corpus))
    d = max(1, int(hidden_dim))
    mf = float(min(0.95, max(0.05, mem_fraction)))

    idx = _cuda_device_index(device)
    if idx is None:
        qb = min(max_query_batch, n_queries)
        cc = min(max_corpus_chunk, max(min_corpus_chunk, n_corpus))
        logger.info(
            "mteb_gpu_proxy: non-CUDA device %s — using query_batch=%d corpus_chunk=%d",
            device,
            qb,
            cc,
        )
        return qb, cc

    try:
        torch.cuda.synchronize(device)
        free_b, total_b = torch.cuda.mem_get_info(idx)
    except Exception as e:
        logger.warning("mteb_gpu_proxy: mem_get_info failed (%s); using defaults", e)
        return min(max_query_batch, n_queries), min(
            max_corpus_chunk, max(min_corpus_chunk, n_corpus)
        )

    emb_bytes = (n_queries + n_corpus) * d * 4
    scratch_budget = max(0, int(free_b * mf) - 64 * 1024 * 1024)
    if scratch_budget <= 0:
        return min_query_batch, min_corpus_chunk

    # Try larger query batches first (better GPU occupancy), then widen corpus chunk.
    budget_elems = int(scratch_budget // max(int(4 * _SCRATCH_FLOAT_MULT), 1))
    best_b, best_c = min_query_batch, min_corpus_chunk
    best_score = 0
    candidates = {
        min_query_batch,
        min(n_queries, max_query_batch),
        min(n_queries, 512),
        min(n_queries, 128),
        min(n_queries, 32),
        min(n_queries, 8),
    }
    for qb in sorted(candidates, reverse=True):
        if qb < 1:
            continue
        cc = budget_elems // max(qb, 1)
        cc = max(1, min(n_corpus, max_corpus_chunk, max(min_corpus_chunk, cc)))
        prod = qb * cc
        if prod >= best_score:
            best_score = prod
            best_b, best_c = qb, cc

    logger.info(
        "mteb_gpu_proxy: VRAM free≈%.2f GiB total≈%.2f GiB emb≈%.2f MiB scratch_budget≈%.2f MiB → "
        "query_batch=%d corpus_chunk=%d (d=%d, n_q=%d, n_c=%d)",
        free_b / (1024**3),
        total_b / (1024**3),
        emb_bytes / (1024**2),
        scratch_budget / (1024**2),
        best_b,
        best_c,
        d,
        n_queries,
        n_corpus,
    )
    return best_b, best_c


def _retrieval_splits_branch(task: Any, hf_subset: str) -> Any:
    """Return the split→data mapping for a HuggingFace subset (mirrors MTEB ``AbsTask.evaluate``)."""
    ds = task.dataset
    if hf_subset not in ds and hf_subset == "default":
        return ds
    return ds[hf_subset]


def _pick_eval_split_name(split_branch: Any) -> str:
    if "test" in split_branch:
        return "test"
    return list(split_branch.keys())[0]


def _resolve_retrieval_split(task: Any, hf_subset: str, split_name: str) -> Any:
    branch = _retrieval_splits_branch(task, hf_subset)
    return branch[split_name]


def _query_id_text_rows(queries: Any) -> List[Tuple[str, str]]:
    if hasattr(queries, "column_names"):
        cols = queries.column_names
        id_col = "id" if "id" in cols else next(c for c in cols if "id" in c.lower())
        text_col = next(
            (c for c in ("text", "query", "sentence") if c in cols),
            cols[1],
        )
        out: List[Tuple[str, str]] = []
        for i in range(len(queries)):
            row = queries[i]
            qid = str(row[id_col])
            txt = str(row[text_col]).strip()
            if txt:
                out.append((qid, txt))
        return out
    if isinstance(queries, Mapping):
        return [(str(k), str(v).strip()) for k, v in queries.items() if str(v).strip()]
    raise TypeError(f"Unsupported queries type: {type(queries)}")


def _corpus_id_text_rows(corpus: Any) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    if isinstance(corpus, Mapping):
        for pid, doc in corpus.items():
            if isinstance(doc, dict):
                t = retrieval_corpus_text_for_encode(doc)
            else:
                t = str(doc).strip() or None
            if t:
                rows.append((str(pid), t))
        return rows
    if hasattr(corpus, "column_names"):
        id_col = "id" if "id" in corpus.column_names else corpus.column_names[0]
        for i in range(len(corpus)):
            row = corpus[i]
            doc = dict(row) if isinstance(row, dict) else {k: corpus[i][k] for k in corpus.column_names}
            t = retrieval_corpus_text_for_encode(doc)
            if t:
                rows.append((str(row[id_col]), t))
        return rows
    raise TypeError(f"Unsupported corpus type: {type(corpus)}")


def _prepare_q_c(
    Q: np.ndarray,
    C: np.ndarray,
    similarity: ScoringFunction | str | None,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = torch.from_numpy(Q).to(device=device, dtype=torch.float32)
    c = torch.from_numpy(C).to(device=device, dtype=torch.float32)
    if similarity in (ScoringFunction.EUCLIDEAN, "euclidean"):
        return q, c
    if similarity in (ScoringFunction.COSINE, "cosine", None):
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-8)
        c = torch.nn.functional.normalize(c, dim=-1, eps=1e-8)
    elif similarity in (ScoringFunction.DOT_PRODUCT, "dot"):
        pass
    else:
        logger.warning(
            "mteb_gpu_proxy: unknown similarity %r; defaulting to cosine-style dot product",
            similarity,
        )
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-8)
        c = torch.nn.functional.normalize(c, dim=-1, eps=1e-8)
    return q, c


def _scores_full_corpus(
    q_t: torch.Tensor,
    c_t: torch.Tensor,
    qids: List[str],
    corpus_ids: List[str],
    top_k: int,
    similarity: ScoringFunction | str | None,
    corpus_chunk: int,
    query_batch: int,
) -> Dict[str, Dict[str, float]]:
    """
    Dense retrieval: batched queries × chunked corpus, streaming top-``top_k``.

    ``query_batch`` and ``corpus_chunk`` are typically from ``infer_proxy_batch_sizes``
    after Q/C are on device.
    """
    results: Dict[str, Dict[str, float]] = {}
    n_c = c_t.shape[0]
    n_q = q_t.shape[0]
    qb = max(1, min(int(query_batch), n_q))
    chunk = max(1, min(int(corpus_chunk), n_c))

    for q_start in range(0, n_q, qb):
        q_end = min(n_q, q_start + qb)
        B = q_end - q_start
        q_batch = q_t[q_start:q_end]
        best_vals: Optional[torch.Tensor] = None
        best_ids: Optional[torch.Tensor] = None

        for c_start in range(0, n_c, chunk):
            c_end = min(n_c, c_start + chunk)
            part = c_t[c_start:c_end]
            if similarity in (ScoringFunction.EUCLIDEAN, "euclidean"):
                sims = -torch.cdist(q_batch, part, p=2.0)
            else:
                sims = q_batch @ part.T
            off = torch.arange(c_start, c_end, device=sims.device, dtype=torch.long)
            off_b = off.unsqueeze(0).expand(B, -1)

            if best_vals is None:
                k0 = min(top_k, sims.shape[1])
                best_vals, lix = torch.topk(sims, k0, dim=1)
                best_ids = torch.gather(off_b, 1, lix)
            else:
                cat_v = torch.cat([best_vals, sims], dim=1)
                cat_i = torch.cat([best_ids, off_b], dim=1)
                k2 = min(top_k, cat_v.shape[1])
                best_vals, tix = torch.topk(cat_v, k2, dim=1)
                best_ids = torch.gather(cat_i, 1, tix)

        if best_vals is None or best_vals.numel() == 0:
            for i in range(B):
                results[qids[q_start + i]] = {}
            continue

        k_eff = min(top_k, best_vals.shape[1])
        vals, tix = torch.topk(best_vals, k_eff, dim=1)
        row_ids = torch.gather(best_ids, 1, tix)
        for i in range(B):
            qi = q_start + i
            pids = [corpus_ids[int(j)] for j in row_ids[i].tolist()]
            results[qids[qi]] = {
                p: float(s) for p, s in zip(pids, vals[i].tolist())
            }
    return results


def _subset_retrieval_results(
    q: torch.Tensor,
    c: torch.Tensor,
    corpus_ids: List[str],
    query_ids: List[str],
    top_ranked: Mapping[str, Sequence[str]],
    corpus_pos: Dict[str, int],
    top_k: int,
    similarity: ScoringFunction | str | None,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for qi, qid in enumerate(query_ids):
        cand = [d for d in top_ranked.get(qid, ()) if d in corpus_pos]
        if not cand:
            results[qid] = {}
            continue
        idx = [corpus_pos[d] for d in cand]
        Csub = c[idx]
        qb = q[qi : qi + 1]
        if similarity in (ScoringFunction.EUCLIDEAN, "euclidean"):
            sims = -torch.cdist(qb, Csub, p=2.0).squeeze(0)
        else:
            sims = (qb @ Csub.T).squeeze(0)
        k_eff = min(top_k, sims.numel())
        vals, loc = torch.topk(sims, k_eff)
        pids = [cand[j] for j in loc.tolist()]
        results[qid] = {pid: float(s) for pid, s in zip(pids, vals.tolist())}
    return results


def _apply_ignore_identical_ids(
    results: Dict[str, Dict[str, float]], enabled: bool
) -> None:
    if not enabled:
        return
    for qid, doc_scores in list(results.items()):
        for pid in list(doc_scores):
            if qid == pid:
                doc_scores.pop(pid, None)


def retrieval_gpu_proxy_main_score(
    task: Any,
    store: Any,
    spec: LayerSpec,
    n_layers: int,
    device: str,
    encoder: Any,
    *,
    proxy_mem_fraction: float = 0.72,
    proxy_query_batch: Optional[int] = None,
    proxy_corpus_chunk: Optional[int] = None,
) -> Optional[float]:
    """
    Return the same aggregate main score MTEB would report for a retrieval /
    reranking task, using dense similarity on GPU (or ``device``) and MTEB's
    metric stack. Requires ``task.load_data()`` and filtered eval splits
    (e.g. test) to match the official eval script.

    Batching: after embeddings are moved to ``device``, free VRAM is read via
    ``torch.cuda.mem_get_info`` to pick ``query_batch`` and ``corpus_chunk`` so
    batched similarity scores (~``query_batch * corpus_chunk`` floats × a small
    multiplier) stay under ``proxy_mem_fraction`` of free memory. Override with
    ``proxy_query_batch`` / ``proxy_corpus_chunk`` when set. Use ``device='cuda:0'``
    or ``cuda:1`` to size for a specific A100.
    """
    from mteb.abstasks import AbsTaskRetrieval

    if not isinstance(task, AbsTaskRetrieval):
        logger.error("retrieval_gpu_proxy_main_score called on non-retrieval task")
        return None
    started_total = time.perf_counter()

    if getattr(task, "data_loaded", False) and hasattr(
        task, "convert_v1_dataset_format_to_v2"
    ):
        task.convert_v1_dataset_format_to_v2(num_proc=None)

    similarity = getattr(
        getattr(encoder, "mteb_model_meta", None),
        "similarity_fn_name",
        ScoringFunction.COSINE,
    )

    hf_subsets: Sequence[str]
    if task.hf_subsets is None:
        hf_subsets = list(task.dataset.keys())
    else:
        hf_subsets = list(task.hf_subsets)

    split_name = _pick_eval_split_name(_retrieval_splits_branch(task, hf_subsets[0]))
    nested_scores: Dict[str, Dict[str, Any]] = {split_name: {}}
    torch_device = torch.device(device)

    top_k = int(getattr(task, "_top_k", max(task.k_values)))

    for hf_subset in hf_subsets:
        started_subset = time.perf_counter()
        data_split = _resolve_retrieval_split(task, hf_subset, split_name)
        rel, queries = _filter_queries_without_positives(
            data_split["relevant_docs"], data_split["queries"]
        )
        corpus = data_split["corpus"]
        top_ranked = data_split.get("top_ranked")

        q_rows = _query_id_text_rows(queries)
        if not q_rows:
            logger.warning("GPU proxy: no queries after filtering for %s", hf_subset)
            continue
        c_rows = _corpus_id_text_rows(corpus)
        if not c_rows:
            logger.warning("GPU proxy: empty corpus for %s", hf_subset)
            continue

        qids = [qid for qid, _ in q_rows]
        q_texts = [t for _, t in q_rows]
        corpus_ids = [pid for pid, _ in c_rows]
        c_texts = [t for _, t in c_rows]

        started_extract = time.perf_counter()
        Q = extract_embedding_matrix(store, q_texts, spec, n_layers)
        C = extract_embedding_matrix(store, c_texts, spec, n_layers)
        extract_s = time.perf_counter() - started_extract
        if Q.shape[0] != len(qids) or C.shape[0] != len(corpus_ids):
            logger.error("GPU proxy: embedding shape mismatch")
            return None

        started_prepare = time.perf_counter()
        q_t, c_t = _prepare_q_c(Q, C, similarity, torch_device)
        prepare_s = time.perf_counter() - started_prepare
        corpus_pos = {pid: i for i, pid in enumerate(corpus_ids)}

        started_score = time.perf_counter()
        if top_ranked:
            results = _subset_retrieval_results(
                q_t,
                c_t,
                corpus_ids,
                qids,
                top_ranked,
                corpus_pos,
                top_k,
                similarity,
            )
        else:
            q_batch, corpus_chunk = infer_proxy_batch_sizes(
                device=torch_device,
                n_queries=q_t.shape[0],
                n_corpus=c_t.shape[0],
                hidden_dim=q_t.shape[1],
                mem_fraction=proxy_mem_fraction,
            )
            if proxy_query_batch is not None:
                q_batch = max(1, int(proxy_query_batch))
            if proxy_corpus_chunk is not None:
                corpus_chunk = max(1, int(proxy_corpus_chunk))
            results = _scores_full_corpus(
                q_t,
                c_t,
                qids,
                corpus_ids,
                top_k,
                similarity,
                corpus_chunk,
                q_batch,
            )
        score_s = time.perf_counter() - started_score

        _apply_ignore_identical_ids(results, bool(task.ignore_identical_ids))

        started_metrics = time.perf_counter()
        started_eval = time.perf_counter()
        ev = calculate_retrieval_scores(
            results,
            rel,
            list(task.k_values),
            skip_first_result=bool(getattr(task, "skip_first_result", False)),
        )
        eval_s = time.perf_counter() - started_eval
        started_task_specific = time.perf_counter()
        task_specific = task.task_specific_scores(
            ev.all_scores,
            rel,
            results,
            hf_split=split_name,
            hf_subset=hf_subset,
        )
        task_specific_s = time.perf_counter() - started_task_specific
        # MTEB ≥2.12: ``RetrievalEvaluationResult`` exposes ``cv_recall``; ``make_score_dict`` takes that
        # instead of ``hit_rate`` (older MTEB).
        cv_recall = getattr(ev, "cv_recall", getattr(ev, "hit_rate", {}))
        if not isinstance(cv_recall, dict):
            cv_recall = {}
        started_score_dict = time.perf_counter()
        subset_scores = make_score_dict(
            ndcg=ev.ndcg,
            _map=ev.map,
            recall=ev.recall,
            precision=ev.precision,
            mrr=ev.mrr,
            naucs=ev.naucs,
            naucs_mrr=ev.naucs_mrr,
            cv_recall=cv_recall,
            task_scores=task_specific,
            previous_results_model_meta=getattr(task, "_previous_results_model_meta", None),
        )
        score_dict_s = time.perf_counter() - started_score_dict
        started_add = time.perf_counter()
        task._add_main_score(subset_scores)
        add_s = time.perf_counter() - started_add
        nested_scores[split_name][hf_subset] = subset_scores
        logger.info(
            "[profile] retrieval_gpu_proxy_subset | %.3fs | task=%s | subset=%s | q=%s | c=%s | d=%s | "
            "extract_s=%.3f | prepare_s=%.3f | score_s=%.3f | eval_s=%.3f | task_specific_s=%.3f | "
            "score_dict_s=%.3f | add_s=%.3f | metric_s=%.3f | top_k=%s",
            time.perf_counter() - started_subset,
            getattr(getattr(task, "metadata", None), "name", type(task).__name__),
            hf_subset,
            len(qids),
            len(corpus_ids),
            int(q_t.shape[1]),
            extract_s,
            prepare_s,
            score_s,
            eval_s,
            task_specific_s,
            score_dict_s,
            add_s,
            time.perf_counter() - started_metrics,
            top_k,
        )

    if not nested_scores[split_name]:
        return None
    tr = TaskResult.from_task_results(task, nested_scores, evaluation_time=0.0)
    try:
        score = _finite_score_or_none(tr.get_score(splits=[split_name]))
        if score is None:
            logger.warning(
                "GPU proxy: non-finite retrieval score for %s split %s; falling back to mteb.evaluate",
                getattr(getattr(task, "metadata", None), "name", type(task).__name__),
                split_name,
            )
            return None
        logger.info(
            "[profile] retrieval_gpu_proxy_total | %.3fs | task=%s | split=%s | subsets=%s | score=%s",
            time.perf_counter() - started_total,
            getattr(getattr(task, "metadata", None), "name", type(task).__name__),
            split_name,
            len(hf_subsets),
            score,
        )
        return score
    except Exception as e:
        logger.error("GPU proxy: TaskResult.get_score failed: %s", e)
        return None


def gpu_proxy_main_score(
    task: Any,
    store: Any,
    spec: LayerSpec,
    n_layers: int,
    device: str,
    encoder: Any,
    *,
    proxy_mem_fraction: float = 0.72,
    proxy_query_batch: Optional[int] = None,
    proxy_corpus_chunk: Optional[int] = None,
) -> Optional[float]:
    """Dense GPU proxy for retrieval/reranking and supported non-retrieval tasks."""
    from mteb.abstasks import AbsTaskRetrieval

    if isinstance(task, AbsTaskRetrieval):
        return retrieval_gpu_proxy_main_score(
            task,
            store,
            spec,
            n_layers,
            device,
            encoder,
            proxy_mem_fraction=proxy_mem_fraction,
            proxy_query_batch=proxy_query_batch,
            proxy_corpus_chunk=proxy_corpus_chunk,
        )
    from src.mteb_gpu_proxy_nonretrieval import non_retrieval_gpu_proxy_main_score

    return non_retrieval_gpu_proxy_main_score(
        task, store, spec, n_layers, device, encoder
    )
