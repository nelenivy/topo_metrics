
# utils.py

import logging
from typing import Any, List, Optional

from datasets import Dataset, DatasetDict

from src.mteb_text_align import iter_retrieval_corpus_passages

logger = logging.getLogger(__name__)


def _is_column_mapping(data: dict) -> bool:
    """Return True when a mapping looks like a leaf column dict."""
    if not data:
        return True
    for value in data.values():
        if isinstance(value, (Dataset, DatasetDict, dict)):
            return False
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            if isinstance(first, (Dataset, DatasetDict, dict)):
                return False
    return True


def _append_text_values(raw: List[str], values: Any) -> None:
    if values is None:
        return
    if isinstance(values, (list, tuple)):
        for value in values:
            _append_text_values(raw, value)
        return
    text = str(values).strip()
    if text:
        raw.append(text)


def _extract_texts_from_data_dict(data_dict: dict, task_type: str) -> List[str]:
    raw: List[str] = []

    def _add_first(candidates: tuple[str, ...]) -> bool:
        for col in candidates:
            if col in data_dict:
                _append_text_values(raw, data_dict[col])
                return True
        return False

    if task_type in ("Classification", "MultilabelClassification"):
        _add_first(("text", "texts", "sentence", "content"))
    elif task_type in ("STS", "PairClassification", "BitextMining"):
        _add_first(("sentence1", "sent1", "text1"))
        _add_first(("sentence2", "sent2", "text2"))
    elif task_type == "Clustering":
        _add_first(("sentences", "text", "texts"))
    elif task_type == "Summarization":
        for col in ("human_summaries", "machine_summaries"):
            if col in data_dict:
                _append_text_values(raw, data_dict[col])
    else:
        for col in (
            "text",
            "texts",
            "sentence",
            "sentence1",
            "sentence2",
            "sent1",
            "sent2",
            "sentences",
            "content",
            "human_summaries",
            "machine_summaries",
        ):
            if col in data_dict:
                _append_text_values(raw, data_dict[col])

    return raw


def extract_texts_from_dataset(dataset_like: Any, task_type: str) -> List[str]:
    """Extract and normalize texts from a nested dataset container."""
    raw: List[str] = []

    def _walk(obj: Any) -> None:
        if obj is None:
            return
        if hasattr(obj, "to_dict"):
            _walk(obj.to_dict())
            return
        if isinstance(obj, dict):
            if _is_column_mapping(obj):
                raw.extend(_extract_texts_from_data_dict(obj, task_type))
            else:
                for value in obj.values():
                    _walk(value)
            return
        if isinstance(obj, (list, tuple)):
            for value in obj:
                _walk(value)

    _walk(dataset_like)

    seen: set[str] = set()
    unique: List[str] = []
    for text in raw:
        text = str(text).strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


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

    # 1. Normalize: extract (dataset, val_name, task_type) from either source type.
    if hasattr(source, "val_name"):
        dataset = source.dataset
        val_name = source.val_name
        task_type = source.task_type
    else:
        if not hasattr(source, "dataset") or source.dataset is None:
            raise ValueError("Task has no loaded dataset. Call task.load_data() first.")

        dataset = source.dataset
        for key in ("default", "en", "default"):
            if isinstance(dataset, dict) and key in dataset:
                dataset = dataset[key]

        task_type = source.metadata.type

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
                        f"extract_texts_from_task: no split specified, falling back to '{val_name}'"
                    )
                    break

        if val_name is None:
            raise ValueError(
                f"Could not determine validation split. Available splits: {list(dataset.keys())}. Pass val_name= explicitly."
            )

    # 2. Resolve val_data to a plain dict.
    val_data = dataset[val_name]

    if hasattr(val_data, "to_dict"):
        data_dict = val_data.to_dict()
    elif isinstance(val_data, dict):
        data_dict = val_data
    else:
        data_dict = {"text": val_data}

    if task_type in ("Retrieval", "Reranking"):
        queries = data_dict.get("queries", {})
        raw: List[str] = []
        if isinstance(queries, dict):
            raw.extend(str(v) for v in queries.values() if v)
        elif hasattr(queries, "column_names"):
            for col in ("text", "query", "sentence"):
                if col in queries.column_names:
                    raw.extend(str(t) for t in queries[col] if t)
                    break

        corpus = data_dict.get("corpus") or dataset.get("corpus", {})
        corpus_texts = list(iter_retrieval_corpus_passages(corpus))
        if max_corpus_size and len(corpus_texts) > max_corpus_size:
            logger.warning(
                f"extract_texts_from_task: corpus has {len(corpus_texts)} docs, limiting to {max_corpus_size}. Remaining handled by LayerEmbeddingStore overflow."
            )
            corpus_texts = corpus_texts[:max_corpus_size]

        raw.extend(corpus_texts)
        unique: List[str] = []
        seen: set[str] = set()
        for text in raw:
            text = str(text).strip()
            if text and text not in seen:
                seen.add(text)
                unique.append(text)
    else:
        unique = extract_texts_from_dataset(val_data, task_type)

    logger.info(
        f"extract_texts_from_task: {len(unique)} unique texts [task_type={task_type}, split={val_name}]"
    )
    return unique
