# utils.py

import logging
from typing import List, Optional

from src.mteb_text_align import iter_retrieval_corpus_passages

logger = logging.getLogger(__name__)


def extract_texts_from_task(
    source,                                   # ValidationSplitResolver  OR  mteb Task
    val_name:        Optional[str]  = None,   # override split (ignored for resolver)
    max_corpus_size: Optional[int]  = None,
) -> List[str]:
    """
    Extract all unique texts that MTEB will encode for a task.

    Accepts either:
      - ValidationSplitResolver  — uses resolver.dataset / .val_name / .task_type
      - mteb 2.0 Task object     — normalises nested structure, detects split

    In both cases mirrors MTEB retrieval corpus strings (title + space + body
    when title is present; see ``retrieval_corpus_text_for_encode``) so
    LayerEmbeddingStore keys match ``mteb.evaluate`` / ``encode_corpus``.

    For retrieval tasks with large corpora, set max_corpus_size to limit
    precomputation; docs beyond the limit are handled by LayerEmbeddingStore's
    overflow mechanism.

    Args:
        source:          ValidationSplitResolver or a loaded mteb Task.
        val_name:        Override the split to use. Ignored when source is a
                         resolver (resolver already owns that decision).
        max_corpus_size: Cap on retrieval corpus size.

    Returns:
        Deduplicated list of text strings in encounter order.
    """

    # ------------------------------------------------------------------ #
    # 1. Normalise: extract (dataset, val_name, task_type) from either    #
    # ------------------------------------------------------------------ #

    if hasattr(source, "val_name"):
        # ── ValidationSplitResolver ────────────────────────────────────
        dataset   = source.dataset
        val_name  = source.val_name          # resolver owns this decision
        task_type = source.task_type

    else:
        # ── MTEB 2.0 Task object ───────────────────────────────────────
        if not hasattr(source, "dataset") or source.dataset is None:
            raise ValueError(
                "Task has no loaded dataset. Call task.load_data() first."
            )

        # Strip default / language nesting (same logic as ValidationSplitResolver)
        dataset = source.dataset
        for key in ("default", "en", "default"):
            if isinstance(dataset, dict) and key in dataset:
                dataset = dataset[key]

        task_type = source.metadata.type

        # Determine split: explicit override → task._eval_splits → heuristic
        if val_name is None:
            eval_splits = (
                getattr(source, "_eval_splits", None)
                or getattr(source, "eval_splits", None)
                or []
            )
            if eval_splits:
                val_name = eval_splits[0]

        if val_name is None:
            for candidate in ("validation", "dev", "val", "train", "test"):
                if candidate in dataset:
                    val_name = candidate
                    logger.warning(
                        f"extract_texts_from_task: no split specified, "
                        f"falling back to '{val_name}'"
                    )
                    break

        if val_name is None:
            raise ValueError(
                f"Could not determine validation split. "
                f"Available splits: {list(dataset.keys())}. "
                f"Pass val_name= explicitly."
            )

    # ------------------------------------------------------------------ #
    # 2. Resolve val_data to a plain dict                                  #
    # ------------------------------------------------------------------ #

    val_data = dataset[val_name]

    if hasattr(val_data, "to_dict"):
        data_dict = val_data.to_dict()
    elif isinstance(val_data, dict):
        data_dict = val_data
    else:
        data_dict = {"text": val_data}

    # ------------------------------------------------------------------ #
    # 3. Task-type-specific extraction                                     #
    # ------------------------------------------------------------------ #

    raw: List[str] = []

    if task_type in ("Classification", "MultilabelClassification"):
        for col in ("text", "texts", "sentence", "content"):
            if col in data_dict:
                for item in data_dict[col]:
                    if isinstance(item, list):
                        raw.extend(str(s) for s in item if s)
                    elif item:
                        raw.append(str(item))
                break

    elif task_type in ("STS", "PairClassification", "BitextMining"):
        for col in ("sentence1", "sentence2"):
            if col in data_dict:
                for item in data_dict[col]:
                    if isinstance(item, list):
                        raw.extend(str(s) for s in item if s)
                    elif item:
                        raw.append(str(item))

    elif task_type == "Clustering":
        for col in ("sentences", "text", "texts"):
            if col in data_dict:
                for item in data_dict[col]:
                    if isinstance(item, list):       # list-of-lists in clustering
                        raw.extend(str(s) for s in item if s)
                    elif item:
                        raw.append(str(item))
                break

    elif task_type in ("Retrieval", "Reranking"):
        # Queries — plain dict {query_id: text}
        queries = data_dict.get("queries", {})
        if isinstance(queries, dict):
            raw.extend(str(v) for v in queries.values() if v)
        elif hasattr(queries, "column_names"):
            for col in ("text", "query", "sentence"):
                if col in queries.column_names:
                    raw.extend(str(t) for t in queries[col] if t)
                    break

        # Corpus — two possible locations:
        #   (a) inside the split dict   → data_dict["corpus"]
        #   (b) top-level shared corpus → dataset["corpus"]
        corpus = data_dict.get("corpus") or dataset.get("corpus", {})
        corpus_texts: List[str] = []

        corpus_texts.extend(iter_retrieval_corpus_passages(corpus))

        if max_corpus_size and len(corpus_texts) > max_corpus_size:
            logger.warning(
                f"extract_texts_from_task: corpus has {len(corpus_texts)} docs, "
                f"limiting to {max_corpus_size}. "
                f"Remaining handled by LayerEmbeddingStore overflow."
            )
            corpus_texts = corpus_texts[:max_corpus_size]

        raw.extend(corpus_texts)

    else:
        # Unknown task type — generic column scan as last resort
        logger.warning(
            f"extract_texts_from_task: unhandled task_type={task_type!r}, "
            f"running generic column scan"
        )
        for col in ("text", "sentence1", "sentence2", "sentences", "query", "passage"):
            if col in data_dict:
                for item in data_dict[col]:
                    if isinstance(item, list):
                        raw.extend(str(s) for s in item if s)
                    elif item:
                        raw.append(str(item))

    # ------------------------------------------------------------------ #
    # 4. Deduplicate, preserve order                                       #
    # ------------------------------------------------------------------ #

    seen:   set       = set()
    unique: List[str] = []
    for t in raw:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            unique.append(t)

    logger.info(
        f"extract_texts_from_task: {len(unique)} unique texts "
        f"[task_type={task_type}, split={val_name}]"
    )
    return unique
