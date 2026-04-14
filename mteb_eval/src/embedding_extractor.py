"""
embedding_extractor.py

Extract raw embedding matrices from a LayerEmbeddingStore for a given
LayerSpec and list of texts.  Also handles the retrieval split:
queries and corpus are tracked separately so that unsup_metrics can be
applied to each subset independently.

No model inference here — all data comes from the pre-computed HDF5 store.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.layer_spec import LayerSpec

logger = logging.getLogger(__name__)


def extract_embedding_matrix(
    store,
    texts: List[str],
    spec: LayerSpec,
    n_layers: int,
) -> np.ndarray:
    """
    Build the (N, d) embedding matrix for *texts* according to *spec*.

    For single-layer specs: direct index into the HDF5 store (fastest).
    For weighted specs: calls store.get_aggregated().

    Parameters
    ----------
    store    : fully pre-computed LayerEmbeddingStore
    texts    : ordered list of text strings (must all be in store)
    spec     : LayerSpec describing which layers/weights to use
    n_layers : total number of layers in the model

    Returns
    -------
    (N, d) float32 ndarray
    """
    weights = spec.weights(n_layers)

    if spec.spec_type == "single":
        layer_idx = spec.layer_idx
        indices = [store._text_index[t] for t in texts]
        return store._embeddings[layer_idx][indices].astype(np.float32)
    else:
        return store.get_aggregated(texts, weights).astype(np.float32)


def extract_retrieval_embeddings(
    store,
    query_texts: List[str],
    corpus_texts: List[str],
    spec: LayerSpec,
    n_layers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (query_matrix, corpus_matrix) for a retrieval task.

    Handles overflow: texts that arrived via encode_queries / encode_corpus
    at MTEB eval time (i.e. not in precomputed bulk store) fall back to
    the store's overflow dict.  Missing texts are skipped with a warning.
    """
    weights = spec.weights(n_layers)

    def _get_matrix(texts: List[str]) -> np.ndarray:
        bulk_texts = [t for t in texts if t in store._text_index]
        over_texts = [t for t in texts
                      if t not in store._text_index and t in store._overflow]
        missing    = [t for t in texts
                      if t not in store._text_index and t not in store._overflow]

        if missing:
            logger.warning(
                f"extract_retrieval_embeddings: {len(missing)} texts "
                f"not found in store (skipping)"
            )

        order_map: Dict[str, np.ndarray] = {}

        if bulk_texts:
            bulk_emb = extract_embedding_matrix(store, bulk_texts, spec, n_layers)
            for t, row in zip(bulk_texts, bulk_emb):
                order_map[t] = row

        if over_texts:
            for t in over_texts:
                layer_dict = store._overflow[t]
                d = next(iter(layer_dict.values())).shape[0]
                row = np.zeros(d, dtype=np.float32)
                for i in range(n_layers):
                    if i in layer_dict and weights[i] != 0:
                        row += weights[i] * layer_dict[i].astype(np.float32)
                order_map[t] = row

        valid_texts = [t for t in texts if t in order_map]
        if not valid_texts:
            d = next(iter(store._embeddings.values())).shape[1]
            return np.zeros((0, d), dtype=np.float32)
        return np.stack([order_map[t] for t in valid_texts])

    return _get_matrix(query_texts), _get_matrix(corpus_texts)
