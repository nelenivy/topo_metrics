"""Align retrieval corpus strings with MTEB's dataloader (see mteb._create_dataloaders._corpus_to_dict)."""

from __future__ import annotations

from typing import Any, Iterable


def retrieval_corpus_text_for_encode(doc: dict) -> str | None:
    """
    Return the same ``text`` field MTEB builds for retrieval corpus rows:
    ``(title + " " + text).strip()`` when ``title`` is present and non-empty,
    else ``text`` (optional ``content`` fallback) stripped.
    """
    body = doc.get("text")
    if body is None:
        body = doc.get("content")
    if body is None:
        body = ""
    title = doc.get("title") if "title" in doc else None
    if title is not None and len(str(title)) > 0:
        return (str(title) + " " + str(body)).strip() or None
    s = str(body).strip()
    return s or None


def iter_retrieval_corpus_passages(corpus: Any) -> Iterable[str]:
    """
    Yield one encoding string per corpus document, for dict-of-docs or HF ``Dataset``.

    MTEB maps each row with ``_corpus_to_dict``; this matches that for every row.
    """
    if isinstance(corpus, dict):
        for doc in corpus.values():
            if isinstance(doc, dict):
                s = retrieval_corpus_text_for_encode(doc)
                if s:
                    yield s
            elif isinstance(doc, str) and doc:
                yield doc
        return
    if hasattr(corpus, "column_names") and hasattr(corpus, "__len__"):
        for i in range(len(corpus)):
            row = corpus[i]
            doc = dict(row) if isinstance(row, dict) else {k: corpus[i][k] for k in corpus.column_names}
            s = retrieval_corpus_text_for_encode(doc)
            if s:
                yield s
