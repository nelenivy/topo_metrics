"""
Aggregated encoder that uses provided weights.
Inherits from MTEB's EncoderProtocol for full compatibility.
Includes pooler_output support for SimCSE models.
"""

import logging
import torch
import pickle
import numpy as np
from typing import Any, Optional, Dict, Union, List
from torch.utils.data import DataLoader
from mteb.models import EncoderProtocol
from mteb.models.model_meta import ModelMeta, ScoringFunction
from src.model_loading import load_encoder_model_for_layers
from src.cache_manager import LayerEmbeddingStore, PooledLayerEmbeddingView
from src.pooling_rules import pool_hidden_states
from src.strategies import normalize_weights
from src.layer_spec import LayerSpec
from src.mteb_text_align import retrieval_corpus_text_for_encode

_log = logging.getLogger(__name__)


def _build_aggregated_mteb_meta(
    model_name: str,
    pooling: str,
    aggregation_weights: Optional[np.ndarray],
    hidden_size: int,
) -> ModelMeta:
    """Full MTEB ModelMeta so deprecated MTEB.run() gets framework, modalities, model_name_as_path, etc."""
    safe = model_name.replace("/", "__").replace("\\", "_")
    if aggregation_weights is not None:
        import hashlib

        weights_str = ",".join(f"{w:.4f}" for w in aggregation_weights)
        weights_hash = hashlib.md5(weights_str.encode()).hexdigest()[:8]
        agg_id = f"{safe}_{pooling}_w{weights_hash}"
    else:
        agg_id = f"{safe}_{pooling}_uniform"
    custom_name = f"aggregated/{agg_id}"

    # fill_missing=False: avoid ModelCard license quirks (list vs enum) that break Pydantic.
    try:
        meta = ModelMeta._from_hub(model_name, revision="main", fill_missing=False)
    except Exception:
        meta = ModelMeta._from_hub(None, revision="main", fill_missing=False)

    ref = f"https://huggingface.co/{model_name}" if model_name and "/" in model_name else meta.reference
    meta = meta.model_copy(
        update={
            "name": custom_name,
            "revision": "main",
            "embed_dim": hidden_size,
            "reference": ref,
        }
    )
    if meta.similarity_fn_name is None:
        meta = meta.model_copy(update={"similarity_fn_name": ScoringFunction.COSINE})
    if not meta.modalities:
        meta = meta.model_copy(update={"modalities": ["text"]})
    return meta


class SimpleWeightedAggregation:
    """Simple weighted aggregation that uses provided weights."""

    def __init__(self, weights: np.ndarray):
        """Initialize with weights."""
        self.weights = normalize_weights(weights, threshold=0.001)
        self.weights_tensor = torch.from_numpy(self.weights).float()

    def aggregate(self, layer_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate using weights.

        Args:
            layer_embeddings: (num_layers, batch_size, hidden_dim)

        Returns:
            aggregated: (batch_size, hidden_dim)
        """
        weights = self.weights_tensor.to(layer_embeddings.device).view(-1, 1, 1)
        return (layer_embeddings * weights).sum(dim=0)

    def get_weights(self) -> np.ndarray:
        """Get weights."""
        return self.weights.copy()

    def set_weights(self, weights: np.ndarray):
        """Set new weights."""
        self.weights = normalize_weights(weights, threshold=0.001)
        self.weights_tensor = torch.from_numpy(self.weights).float()

class LayerEncoder:
    """
    Layer encoder that extracts all layers with LMDB caching support.
    """
    
    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",
        batch_size: int = 32,
        device: str = "cuda",
        use_pooler_output: bool = False,
        use_cache: bool = False,  
        cache_dir: str = "./embedding_cache",
        trust_remote_code: bool = True,
        torch_dtype: Optional[str] = None,
    ):
        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.device = device
        self.use_pooler_output = use_pooler_output
        self.use_cache = use_cache
        
        self.model, self.tokenizer, self.num_layers = load_encoder_model_for_layers(
            model_name,
            device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        
        self.max_length = 512
        
        # Initialize cache
        if use_cache:
            self.cache = LayerEmbeddingStore(
                cache_dir=cache_dir,
                model_name=model_name,
                batch_size=batch_size,
                n_layers=self.num_layers,
                pooling=pooling)#EmbeddingCache(cache_dir)
            _log.debug("cache enabled for LayerEncoder")
        else:
            self.cache = None
    
    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool hidden states. Supports: cls, mean, last_token."""
        return pool_hidden_states(hidden_states, attention_mask, self.pooling)

    def encode_batch_multi_poolings(
        self,
        sentences: List[str],
        poolings: List[str],
    ) -> Dict[str, List[np.ndarray]]:
        """
        One ``model`` forward per batch; apply every ``pooling`` to each layer in memory.

        Returns ``{pooling: [ (B, H) ndarray per layer ]}``. Does not use the
        per-sentence LMDB-style cache (precompute path uses ``use_cache=False``).
        """
        h = self.model.config.hidden_size
        empty = {p: [np.zeros((0, h)) for _ in range(self.num_layers)] for p in poolings}
        if not poolings:
            return {}
        if not sentences:
            return empty

        sentences = [str(s).strip() for s in sentences if s]
        if not sentences:
            return empty

        try:
            inputs = self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model(**inputs)
                hidden_states = outputs.hidden_states

            out: Dict[str, List[np.ndarray]] = {p: [] for p in poolings}
            mask = inputs["attention_mask"]
            for layer_idx in range(self.num_layers):
                layer_h = hidden_states[layer_idx]
                for p in poolings:
                    pooled = pool_hidden_states(layer_h, mask, p)
                    out[p].append(pooled.float().cpu().numpy())

            return out
        except Exception as e:
            _log.warning("encode_batch_multi_poolings: %s", e)
            return {p: [np.zeros((len(sentences), h)) for _ in range(self.num_layers)] for p in poolings}

    def encode_batch(
        self,
        sentences: List[str],
        return_all_layers: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode a batch of sentences.
        
        Args:
            sentences: List of sentences to encode
            return_all_layers: If True, return list of arrays (one per layer)
                             If False, return single array (last layer only)
        
        Returns:
            If return_all_layers: List[np.ndarray] of shape [(B, H)] * num_layers
            Otherwise: np.ndarray of shape (B, H)
        """
        if not sentences:
            return [] if return_all_layers else np.zeros((0, self.model.config.hidden_size))
        
        # Normalize sentences
        sentences = [str(s).strip() for s in sentences if s]
        if not sentences:
            return [] if return_all_layers else np.zeros((0, self.model.config.hidden_size))
        
        # Check cache for each layer
        if self.use_cache and self.cache:
            cached_layers = self._get_cached_batch(sentences)
            if cached_layers is not None:
                # All layers cached
                if return_all_layers:
                    return cached_layers
                else:
                    return cached_layers[-1]  # Return last layer
            
            # Partial cache hit - for now, recompute all
            # (You could optimize this to only compute missing layers)
        
        # Compute embeddings
        try:
            inputs = self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model(**inputs)
                hidden_states = outputs.hidden_states
                
                # Extract all layers
                all_layer_embeddings = []
                for layer_idx in range(self.num_layers):
                    layer_hidden = hidden_states[layer_idx]
                    pooled = self.pool(layer_hidden, inputs['attention_mask'])
                    all_layer_embeddings.append(pooled.float().cpu().numpy())
                
                # Add pooler_output if requested
                if self.use_pooler_output and outputs.pooler_output is not None:
                    all_layer_embeddings.append(outputs.pooler_output.float().cpu().numpy())
            
            # Cache all layers
            if self.use_cache and self.cache:
                self._cache_batch(sentences, all_layer_embeddings)
            
            if return_all_layers:
                return all_layer_embeddings
            else:
                return all_layer_embeddings[-1]
        
        except Exception as e:
            _log.warning("encode_batch: %s", e)
            if return_all_layers:
                return [np.zeros((len(sentences), self.model.config.hidden_size)) 
                       for _ in range(self.num_layers)]
            else:
                return np.zeros((len(sentences), self.model.config.hidden_size))
    
    def _get_cached_batch(self, sentences: List[str]) -> Optional[List[np.ndarray]]:
        """
        Try to get all layers for all sentences from cache.
        Returns None if any sentence/layer is missing.
        """
        # Check if all sentences have all layers cached
        all_cached = []
        
        for layer_idx in range(self.num_layers):
            layer_embeddings = []
            for sentence in sentences:
                emb = self.cache.get_sentence(
                    self.model_name, layer_idx, self.pooling, sentence
                )
                if emb is None:
                    return None  # Cache miss
                layer_embeddings.append(emb)
            
            all_cached.append(np.vstack(layer_embeddings))
        
        return all_cached
    
    def _cache_batch(self, sentences: List[str], all_layer_embeddings: List[np.ndarray]):
        """Cache all layers for all sentences"""
        for layer_idx, layer_embs in enumerate(all_layer_embeddings):
            for sentence, emb in zip(sentences, layer_embs):
                self.cache.set_sentence(
                    self.model_name, layer_idx, self.pooling, sentence, emb
                )
    
    def __del__(self):
        """Close cache on cleanup"""
        if hasattr(self, 'cache') and self.cache:
            self.cache.close()


class AggregatedEncoder(EncoderProtocol):
    """
    Aggregated encoder with caching support.
    """
    
    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",
        batch_size: int = 32,
        device: Optional[str] = None,
        aggregation_weights: Optional[np.ndarray] = None,
        normalize_weights: bool = False,
        use_pooler_output: bool = False,
        use_cache: bool = False,
        cache_dir: str = "./embedding_cache",
        trust_remote_code: bool = True,
        torch_dtype: Optional[str] = None,
    ):
        """Initialize encoder with caching support."""
        if not model_name:
            raise ValueError("model_name is required!")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.pooling = pooling
        self.batch_size = batch_size
        self.num_layers = aggregation_weights.shape[0]
        self.use_pooler_output = use_pooler_output
        
        # Create layer encoder WITH CACHING
        self.encoder = LayerEncoder(
            model_name=self.model_name,
            pooling=pooling,
            batch_size=batch_size,
            device=self.device,
            use_pooler_output=use_pooler_output,
            use_cache=use_cache,  # Pass through
            cache_dir=cache_dir,   # Pass through
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        if self.encoder.num_layers != self.num_layers:
            raise ValueError(
                f"aggregation_weights length ({self.num_layers}) != model hidden-state depth "
                f"({self.encoder.num_layers}) for {model_name!r}"
            )
        
        cfg = self.encoder.model.config
        self.hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "d_model", None)
            or getattr(cfg, "hidden_dim", None)
        )
        if self.hidden_size is None:
            raise ValueError(
                f"Unknown hidden size for {model_name!r}; config has no hidden_size / d_model / hidden_dim."
            )
        
        # Setup aggregation weights

        if aggregation_weights is not None:
            initial_weights = aggregation_weights
        else:
            initial_weights = np.ones(self.num_layers)

        self.aggregator = SimpleWeightedAggregation(initial_weights)

        self.mteb_model_meta = _build_aggregated_mteb_meta(
            model_name=model_name,
            pooling=pooling,
            aggregation_weights=aggregation_weights,
            hidden_size=self.hidden_size,
        )

    # In aggregated_encoder.py, find the property and add setter:

    @property
    def mteb_model_meta(self):
        return self._mteb_model_meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, value):
        self._mteb_model_meta = value

    
    def _encode_batch(self, sentences: List[str]) -> np.ndarray:
        """Encode a single batch using cached layer encoder."""
        if not sentences or len(sentences) == 0:
            return None
        
        sentences = [str(s).strip() for s in sentences if s is not None]
        sentences = [s for s in sentences if s]
        
        if not sentences:
            return None
        
        try:
            # Get all layer embeddings (uses cache internally)
            all_layer_embs = self.encoder.encode_batch(
                sentences, return_all_layers=True
            )
            
            if not all_layer_embs:
                return np.zeros((len(sentences), self.hidden_size), dtype=np.float32)
            
            # Stack: (num_layers, B, H)
            all_layer_embs = [
                torch.from_numpy(emb) if isinstance(emb, np.ndarray) else emb
                for emb in all_layer_embs
            ]
            all_layer_embs = torch.stack(all_layer_embs, dim=0)
            
            # Aggregate using weights
            result = self.aggregator.aggregate(all_layer_embs)
            
            if result is None:
                return np.zeros((len(sentences), self.hidden_size), dtype=np.float32)
            
            if isinstance(result, torch.Tensor):
                result = result.float().cpu().numpy()
            
            return result
        
        except Exception as e:
            _log.exception("_encode_batch: %s", e)
            return np.zeros((len(sentences), self.hidden_size), dtype=np.float32)


    # ========== MTEB Interface Methods ==========

    def encode(
        self,
        sentences: Union[List[str], str, DataLoader],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences (MTEB interface)."""
        return self._encode_impl(sentences, batch_size=batch_size, **kwargs)

    def encode_queries(
        self,
        queries: Union[List[str], str, DataLoader],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Encode queries (MTEB interface)."""
        return self._encode_impl(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(
        self,
        corpus: Union[List[str], List[Dict[str, str]], DataLoader],
        *,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Encode corpus (MTEB interface)."""
        # Handle dict format (MTEB corpus)
        if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], dict):
            sentences = []
            for doc in corpus:
                s = retrieval_corpus_text_for_encode(doc)
                if s:
                    sentences.append(s)
            return self._encode_impl(sentences, batch_size=batch_size, **kwargs)

        return self._encode_impl(corpus, batch_size=batch_size, **kwargs)

    def similarity(self, queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between queries and corpus."""
        queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
        return queries_norm @ corpus_norm.T

    def similarity_pairwise(self, sentences1: np.ndarray, sentences2: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity."""
        s1_norm = sentences1 / (np.linalg.norm(sentences1, axis=1, keepdims=True) + 1e-8)
        s2_norm = sentences2 / (np.linalg.norm(sentences2, axis=1, keepdims=True) + 1e-8)
        return np.sum(s1_norm * s2_norm, axis=1)

    # ========== Internal Implementation ==========

    def _encode_impl(
        self,
        sentences: Union[List[str], str, DataLoader],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Internal encoding implementation."""
        # Handle DataLoader
        if isinstance(sentences, DataLoader):
            all_embeddings = []
            for batch in sentences:
                batch_sentences = self._extract_sentences_from_batch(batch)
                if not batch_sentences:
                    continue
                batch_embs = self._encode_batch(batch_sentences)
                if batch_embs is not None and len(batch_embs) > 0:
                    all_embeddings.append(batch_embs)
            if not all_embeddings:
                return np.zeros((0, self.hidden_size), dtype=np.float32)
            return np.vstack(all_embeddings)

        if isinstance(sentences, str):
            sentences = [sentences]

        if not sentences or len(sentences) == 0:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        batch_size = batch_size or self.batch_size
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            if not batch:
                continue

            batch_embs = self._encode_batch(batch)
            if batch_embs is not None and len(batch_embs) > 0:
                all_embeddings.append(batch_embs)

        if not all_embeddings:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        return np.vstack(all_embeddings)

    def _extract_sentences_from_batch(self, batch):
        """Extract sentences from batch (for DataLoader compatibility)."""
        if isinstance(batch, dict):
            for key in ["text", "sentence", "sentences", "query", "passage", "title", "content"]:
                if key in batch and batch[key] is not None:
                    return batch[key]
            for v in batch.values():
                if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str):
                    return v
        elif isinstance(batch, (list, tuple)):
            return batch
        else:
            return [str(batch)]
        return []

    # ========== Weight Management ==========

    def get_aggregation_weights(self) -> np.ndarray:
        """Get current aggregation weights."""
        return self.aggregator.get_weights()

    def set_aggregation_weights(self, weights: np.ndarray):
        """Set new aggregation weights."""
        self.aggregator.set_weights(weights)

    def __repr__(self):
        return f"AggregatedEncoder(model={self.model_name}, pooling={self.pooling}, layers={self.num_layers})"


class StoreBackedAggregatedEncoder(EncoderProtocol):
    """
    MTEB-facing encoder that **only** reads precomputed pooled layers from
    ``PooledLayerEmbeddingView`` (HDF5 / in-memory store). No transformer forward.

    After ``precompute_or_load``, layer matrices already live in RAM (HDF5 is
    read once with ``[:]``). Here we additionally materialize one **(N, H)**
    matrix for this ``LayerSpec``'s aggregation so each MTEB batch is a single
    ``numpy`` fancy-index (no repeated weighted sum over layers).
    """

    def __init__(
        self,
        model_name: str,
        pooling: str,
        store: PooledLayerEmbeddingView,
        spec: LayerSpec,
        n_layers: int,
        hidden_size: int,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.pooling = pooling
        self._store = store
        self._spec = spec
        self._n_layers = int(n_layers)
        self.batch_size = batch_size
        self.device = device or "cpu"
        self.hidden_size = int(hidden_size)
        self.num_layers = self._n_layers
        self._text_index = store._text_index
        w = spec.weights(n_layers)
        self._mteb_model_meta = _build_aggregated_mteb_meta(
            model_name=model_name,
            pooling=pooling,
            aggregation_weights=w,
            hidden_size=self.hidden_size,
        )
        em = store._embeddings
        n_rows = len(self._text_index)
        if n_rows == 0:
            self._all_rows = np.zeros((0, self.hidden_size), dtype=np.float32)
        else:
            acc = np.zeros((n_rows, self.hidden_size), dtype=np.float32)
            w_vec = w.astype(np.float32)
            for i in range(self._n_layers):
                acc += w_vec[i] * em[i].astype(np.float32)
            self._all_rows = acc

    @property
    def mteb_model_meta(self):
        return self._mteb_model_meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, value):
        self._mteb_model_meta = value

    def _encode_batch(self, sentences: List[str]) -> Optional[np.ndarray]:
        if not sentences:
            return None
        sentences = [str(s).strip() for s in sentences if s is not None]
        sentences = [s for s in sentences if s]
        if not sentences:
            return None
        try:
            idx = np.array([self._text_index[t] for t in sentences], dtype=np.intp)
        except KeyError as e:
            _log.error("Text missing from HDF5 store (no model forward): %s", e)
            raise RuntimeError(
                "MTEB requested text not found in the precomputed embedding store. "
                "Ensure task texts match extract_all_texts / precompute keys exactly."
            ) from e
        return self._all_rows[idx].copy()

    def encode(
        self,
        sentences: Union[List[str], str, DataLoader],
        *,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        return self._encode_impl(sentences, batch_size=batch_size, **kwargs)

    def encode_queries(
        self,
        queries: Union[List[str], str, DataLoader],
        *,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        return self._encode_impl(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(
        self,
        corpus: Union[List[str], List[Dict[str, str]], DataLoader],
        *,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], dict):
            sentences = []
            for doc in corpus:
                s = retrieval_corpus_text_for_encode(doc)
                if s:
                    sentences.append(s)
            return self._encode_impl(sentences, batch_size=batch_size, **kwargs)
        return self._encode_impl(corpus, batch_size=batch_size, **kwargs)

    def similarity(self, queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
        return queries_norm @ corpus_norm.T

    def similarity_pairwise(self, sentences1: np.ndarray, sentences2: np.ndarray) -> np.ndarray:
        s1_norm = sentences1 / (np.linalg.norm(sentences1, axis=1, keepdims=True) + 1e-8)
        s2_norm = sentences2 / (np.linalg.norm(sentences2, axis=1, keepdims=True) + 1e-8)
        return np.sum(s1_norm * s2_norm, axis=1)

    def _encode_impl(
        self,
        sentences: Union[List[str], str, DataLoader],
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(sentences, DataLoader):
            all_embeddings = []
            for batch in sentences:
                batch_sentences = self._extract_sentences_from_batch(batch)
                if not batch_sentences:
                    continue
                batch_embs = self._encode_batch(batch_sentences)
                if batch_embs is not None and len(batch_embs) > 0:
                    all_embeddings.append(batch_embs)
            if not all_embeddings:
                return np.zeros((0, self.hidden_size), dtype=np.float32)
            return np.vstack(all_embeddings)

        if isinstance(sentences, str):
            sentences = [sentences]

        if not sentences or len(sentences) == 0:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        batch_size = batch_size or self.batch_size
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            if not batch:
                continue
            batch_embs = self._encode_batch(batch)
            if batch_embs is not None and len(batch_embs) > 0:
                all_embeddings.append(batch_embs)

        if not all_embeddings:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        return np.vstack(all_embeddings)

    def _extract_sentences_from_batch(self, batch):
        if isinstance(batch, dict):
            for key in ["text", "sentence", "sentences", "query", "passage", "title", "content"]:
                if key in batch and batch[key] is not None:
                    return batch[key]
            for v in batch.values():
                if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str):
                    return v
        elif isinstance(batch, (list, tuple)):
            return batch
        else:
            return [str(batch)]
        return []

    def __repr__(self):
        return (
            f"StoreBackedAggregatedEncoder(model={self.model_name}, pooling={self.pooling}, "
            f"spec={self._spec.name})"
        )

