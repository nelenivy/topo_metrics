# cache_manager.py
import os
import pickle
import hashlib
import gc
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List

# cache_manager.py - Sentence-level caching

# cache_manager.py - Robust LMDB implementation

import numpy as np
import pickle
import hashlib
import shutil
from pathlib import Path
from typing import Optional, List, Dict
import logging
import h5py

logger = logging.getLogger(__name__)


def layer_store_hdf5_path(
    cache_dir: os.PathLike | str,
    model_name: str,
    dataset_name: str,
    split_name: str,
    poolings: List[str],
    n_layers: int,
) -> Path:
    """
    Absolute path to the HDF5 file ``LayerEmbeddingStore`` uses for this key
    (same hashing as ``_cache_file_path``).
    """
    cache_dir = Path(cache_dir)
    if len(poolings) == 1:
        key = (
            f"{model_name}_{dataset_name}_{split_name}_"
            f"{poolings[0]}_{n_layers}"
        )
    else:
        pl = ",".join(poolings)
        key = f"{model_name}_{dataset_name}_{split_name}_mp_{pl}_{n_layers}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return cache_dir / f"layer_store_{h}.h5"


import io
import struct


class PooledLayerEmbeddingView:
    """
    View of one pooling inside a multi-pooling ``LayerEmbeddingStore``.

    Exposes the same subset of attributes used by ``embedding_extractor``:
    ``_text_index``, ``_embeddings``, ``_overflow``, ``get_aggregated``,
    ``get_sentence`` / ``set_sentence`` (delegating with a fixed pooling).
    """

    def __init__(self, parent: "LayerEmbeddingStore", pooling: str):
        self._parent = parent
        self._pooling = pooling
        self.model_name = parent.model_name
        self.n_layers = parent.n_layers
        self.pooling = pooling

    @property
    def _text_index(self) -> Dict[str, int]:
        return self._parent._text_index

    @property
    def _embeddings(self) -> Dict[int, np.ndarray]:
        return self._parent._pooled[self._pooling]

    @property
    def _overflow(self) -> Dict[str, Dict[int, np.ndarray]]:
        inner = self._parent._overflow.get(self._pooling)
        if not inner:
            return {}
        return {sent: dict(layers) for sent, layers in inner.items()}

    def get_aggregated(self, texts: List[str], weights: np.ndarray) -> np.ndarray:
        if not self._parent._is_precomputed:
            raise RuntimeError("Call precompute_or_load() first")
        indices = [self._parent._text_index[t] for t in texts]
        em = self._embeddings
        return sum(
            weights[i] * em[i][indices]
            for i in range(self._parent.n_layers)
        )

    def get_sentence(
        self, model_name: str, layer_idx: int, pooling: str, sentence: str
    ) -> Optional[np.ndarray]:
        return self._parent.get_sentence(model_name, layer_idx, self._pooling, sentence)

    def set_sentence(
        self,
        model_name: str,
        layer_idx: int,
        pooling: str,
        sentence: str,
        embedding: np.ndarray,
    ) -> None:
        self._parent.set_sentence(model_name, layer_idx, self._pooling, sentence, embedding)

    def close(self) -> None:
        pass


class LayerEmbeddingStore:
    """
    In-memory pre-computed embedding store with HDF5 persistence.

    PRIMARY ROLE: pre-compute ALL layer embeddings for a known dataset.
    With a **single** pooling, one forward pass per batch extracts all layers
    (``output_hidden_states=True``). With **multiple** poolings, the same
    forward still runs **once** per batch; each pooling is applied in memory
    to every layer, and only **pooled** arrays are persisted (never full
    hidden tensors on disk).

    UNIFIED CACHE INTERFACE: same get_sentence / set_sentence / close
    as EmbeddingCache (SQLite) and LMDBEmbeddingCache, so make_embedding_cache
    can return it transparently.

    OVERFLOW: ``pooling → sentence → layer_idx → vector`` for sentences not
    in the bulk precompute (e.g. retrieval queries encoded separately).
    """

    def __init__(
        self,
        model_name: str,
        n_layers: int,
        pooling: str = "mean",
        poolings: Optional[List[str]] = None,
        batch_size: int = 32,
        device: str = "cuda",
        cache_dir: str = ".embeddingcache",
        trust_remote_code: bool = True,
        torch_dtype: Optional[str] = None,
    ):
        self.model_name = model_name
        self.n_layers = n_layers
        if poolings is not None:
            self.poolings = list(dict.fromkeys(poolings))
        else:
            self.poolings = [pooling]
        if not self.poolings:
            raise ValueError("poolings must be non-empty (or pass pooling=...).")
        self.pooling = self.poolings[0]
        self.batch_size = batch_size
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype

        # Bulk: pooling → layer_idx → (N, d)
        self._pooled: Dict[str, Dict[int, np.ndarray]] = {}
        self._text_index: Dict[str, int] = {}
        # Overflow: pooling → sentence → layer_idx → vector
        self._overflow: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {}

        self._is_precomputed = False

    def as_pooling(self, pooling: str) -> PooledLayerEmbeddingView:
        """Narrow this store to one pooling for metric extraction / MTEB prep."""
        if pooling not in self.poolings:
            raise ValueError(
                f"pooling {pooling!r} not in this store ({self.poolings})."
            )
        if not self._is_precomputed or pooling not in self._pooled:
            raise RuntimeError("Call precompute_or_load() before as_pooling(...).")
        return PooledLayerEmbeddingView(self, pooling)

    # ------------------------------------------------------------------ #
    # Unified cache interface                                              #
    # ------------------------------------------------------------------ #

    def get_sentence(
        self, model_name: str, layer_idx: int, pooling: str, sentence: str
    ) -> Optional[np.ndarray]:
        row = self._text_index.get(sentence)
        if row is not None and pooling in self._pooled:
            layer_map = self._pooled[pooling]
            if layer_idx in layer_map:
                return layer_map[layer_idx][row]
        return self._overflow.get(pooling, {}).get(sentence, {}).get(layer_idx)

    def set_sentence(
        self,
        model_name: str,
        layer_idx: int,
        pooling: str,
        sentence: str,
        embedding: np.ndarray,
    ) -> None:
        if sentence in self._text_index:
            return
        self._overflow.setdefault(pooling, {}).setdefault(sentence, {})[layer_idx] = embedding

    def close(self) -> None:
        pass   # HDF5 written atomically during precompute_or_load; nothing to flush

    # ------------------------------------------------------------------ #
    # Additional: pre-computation                                          #
    # ------------------------------------------------------------------ #

    def precompute_or_load_from_task(
        self,
        source,                                  # ValidationSplitResolver OR mteb Task
        val_name:        Optional[str] = None,
        max_corpus_size: Optional[int] = None,
    ) -> None:
        """
        Extract texts from a resolver or MTEB task and precompute embeddings.
        See extract_texts_from_task() for full docs.
        """
        from src.utils import extract_texts_from_task

        # Determine dataset_name for HDF5 cache filename
        if hasattr(source, "task_name"):
            dataset_name = source.task_name           # ValidationSplitResolver
        else:
            dataset_name = getattr(source.metadata, "name", str(source))

        texts = extract_texts_from_task(
            source,
            val_name=val_name,
            max_corpus_size=max_corpus_size,
        )

        if not texts:
            logger.warning(
                f"LayerEmbeddingStore: no texts extracted for {dataset_name!r}. "
                f"All embeddings will be computed on-the-fly."
            )
            return

        self.precompute_or_load(texts, dataset_name=dataset_name, split_name=val_name)


    def precompute_or_load(self, texts: List[str], dataset_name: str, split_name: str) -> None:
        """
        Pre-compute or load all layer embeddings for ``texts``.

        - If a valid HDF5 file exists: load into RAM (no inference).
        - Else: one forward per batch; with multiple poolings, all are applied
          in memory after each forward; only pooled arrays are written.

        Args:
            texts:        All sentences for this dataset split.
            dataset_name: Used to build a stable HDF5 cache filename.
        """
        cache_file = self._cache_file_path(dataset_name, split_name)

        if cache_file.exists() and self._hdf5_is_valid(str(cache_file)):
            started = time.perf_counter()
            logger.info(f"LayerEmbeddingStore: loading from {cache_file}")
            self._load_from_hdf5(str(cache_file), texts)
            logger.info(
                "[profile] layer_store_hdf5_load | %.3fs | model=%s | dataset=%s | split=%s | texts=%s | poolings=%s",
                time.perf_counter() - started,
                self.model_name,
                dataset_name,
                split_name,
                len(texts),
                ",".join(self.poolings),
            )
        else:
            if cache_file.exists():
                logger.warning("LayerEmbeddingStore: corrupt/incomplete HDF5, recomputing")
                cache_file.unlink()
            extra = (
                f" × {len(self.poolings)} poolings (one forward/batch)"
                if len(self.poolings) > 1
                else ""
            )
            logger.info(
                f"LayerEmbeddingStore: pre-computing "
                f"{self.n_layers} layers × {len(texts)} texts{extra}"
            )
            started = time.perf_counter()
            self._compute_and_save(texts, str(cache_file))
            logger.info(
                "[profile] layer_store_compute_and_save | %.3fs | model=%s | dataset=%s | split=%s | texts=%s | poolings=%s",
                time.perf_counter() - started,
                self.model_name,
                dataset_name,
                split_name,
                len(texts),
                ",".join(self.poolings),
            )

        self._is_precomputed = True

    def get_aggregated(self, texts: List[str], weights: np.ndarray) -> np.ndarray:
        """
        Weighted sum of pre-computed layer embeddings for the **first** pooling
        only, or the sole pooling. For multi-pooling stores use ``.as_pooling(p)``.
        """
        if len(self.poolings) != 1:
            raise RuntimeError(
                "Multi-pooling store: use store.as_pooling(pooling).get_aggregated(...)"
            )
        if not self._is_precomputed:
            raise RuntimeError("Call precompute_or_load() first")
        indices = [self._text_index[t] for t in texts]
        em = self._pooled[self.poolings[0]]
        return sum(weights[i] * em[i][indices] for i in range(self.n_layers))

    def is_ready(self) -> bool:
        return self._is_precomputed

    # ------------------------------------------------------------------ #
    # Internal: compute via LayerEncoder (single forward pass per batch)  #
    # ------------------------------------------------------------------ #

    def _compute_and_save(self, texts: List[str], cache_file: str) -> None:
        self._text_index = {t: i for i, t in enumerate(texts)}

        tmp = cache_file + ".tmp"
        if tmp.exists():
            tmp.unlink()

        from src.aggregated_encoder import LayerEncoder  # local import avoids circular

        layer_enc = LayerEncoder(
            model_name=self.model_name,
            pooling=self.pooling,
            batch_size=self.batch_size,
            device=self.device,
            use_cache=False,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.torch_dtype,
        )

        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        if len(self.poolings) == 1:
            p = self.poolings[0]
            accum: Dict[int, List[np.ndarray]] = {i: [] for i in range(self.n_layers)}
            for b, start in enumerate(range(0, len(texts), self.batch_size)):
                batch = texts[start : start + self.batch_size]
                logger.debug("  batch %s/%s (%s texts)", b + 1, n_batches, len(batch))
                all_layer_embs = layer_enc.encode_batch(batch, return_all_layers=True)
                for layer_idx, emb in enumerate(all_layer_embs):
                    accum[layer_idx].append(np.asarray(emb, dtype=np.float32))
            layer_embs = {i: np.concatenate(accum[i], axis=0) for i in range(self.n_layers)}
            self._pooled = {p: layer_embs}
            self._save_to_hdf5_legacy(tmp, texts, layer_embs)
        else:
            accum_mp: Dict[str, Dict[int, List[np.ndarray]]] = {
                p: {i: [] for i in range(self.n_layers)} for p in self.poolings
            }
            for b, start in enumerate(range(0, len(texts), self.batch_size)):
                batch = texts[start : start + self.batch_size]
                logger.debug(
                    "  batch %s/%s (%s texts, poolings=%s)",
                    b + 1,
                    n_batches,
                    len(batch),
                    self.poolings,
                )
                multi = layer_enc.encode_batch_multi_poolings(batch, self.poolings)
                for p in self.poolings:
                    for layer_idx, emb in enumerate(multi[p]):
                        accum_mp[p][layer_idx].append(np.asarray(emb, dtype=np.float32))
            self._pooled = {
                p: {i: np.concatenate(accum_mp[p][i], axis=0) for i in range(self.n_layers)}
                for p in self.poolings
            }
            self._save_to_hdf5_multi(tmp, texts, self._pooled)

        del layer_enc
        gc.collect()

        os.replace(tmp, cache_file)
        logger.info(f"LayerEmbeddingStore: saved → {cache_file}")

    @classmethod
    def merge_shards_to_hdf5(
        cls,
        shard_paths: List[str],
        out_tmp: str,
        full_texts: List[str],
        *,
        n_layers: int,
        poolings: List[str],
    ) -> None:
        """
        Merge per-shard HDF5 files (same on-disk layout as ``_compute_and_save`` shards)
        into one HDF5 at ``out_tmp``. Used by ``run_unsup_eval`` multi-GPU orchestration.

        Shards must be a round-robin split of ``full_texts`` in device list order.
        """
        pls = list(dict.fromkeys(poolings))
        if not pls:
            raise ValueError("poolings must be non-empty")
        shell = cls.__new__(cls)
        shell.n_layers = int(n_layers)
        shell.poolings = pls
        shell.pooling = pls[0]
        shell._merge_shard_hdf5s(shard_paths, out_tmp, full_texts)

    def _merge_shard_hdf5s(
        self,
        shard_paths: List[str],
        out_tmp: str,
        full_texts: List[str],
    ) -> None:
        """
        Merge per-shard HDF5 partial files (same layout as full store) into one HDF5.

        Shards are round-robin splits of ``full_texts`` in the same order as
        the caller's device / shard list.
        """
        if not shard_paths:
            raise ValueError("no shard paths")

        # Load shard texts + pooled arrays (small metadata; big matrices stay in RAM once)
        parts: List[Tuple[List[str], Dict[str, Dict[int, np.ndarray]]]] = []
        for sp in shard_paths:
            with h5py.File(sp, "r") as f:
                stexts = [t.decode("utf-8") for t in f["texts"][:]]
                layers_grp = f["layers"]
                pooled: Dict[str, Dict[int, np.ndarray]] = {}
                if "format_version" in f and int(f["format_version"][0]) == 2:
                    for p in self.poolings:
                        pg = layers_grp[p]
                        pooled[p] = {int(k): pg[k][:] for k in pg.keys()}
                else:
                    p0 = self.poolings[0]
                    pooled[p0] = {int(k): layers_grp[k][:] for k in layers_grp.keys()}
            parts.append((stexts, pooled))

        n = len(full_texts)
        if n == 0:
            self._pooled = {p: {i: np.zeros((0, 1), dtype=np.float32) for i in range(self.n_layers)} for p in self.poolings}
            if len(self.poolings) == 1:
                self._save_to_hdf5_legacy(out_tmp, full_texts, self._pooled[self.poolings[0]])
            else:
                self._save_to_hdf5_multi(out_tmp, full_texts, self._pooled)
            return

        # Infer hidden size from first nonempty shard tensor
        h: Optional[int] = None
        for _st, pmap in parts:
            for p in self.poolings:
                for li in range(self.n_layers):
                    a = pmap.get(p, {}).get(li)
                    if a is not None and a.size:
                        h = int(a.shape[1])
                        break
                if h is not None:
                    break
            if h is not None:
                break
        if h is None:
            h = 1

        merged: Dict[str, Dict[int, np.ndarray]] = {
            p: {i: np.zeros((n, h), dtype=np.float32) for i in range(self.n_layers)} for p in self.poolings
        }

        cursors = [0 for _ in parts]
        for global_i, t in enumerate(full_texts):
            si = global_i % len(parts)
            stexts, pmap = parts[si]
            j = cursors[si]
            if j >= len(stexts) or stexts[j] != t:
                raise RuntimeError(
                    f"shard merge mismatch at global index {global_i}: expected {t!r}, "
                    f"got {stexts[j]!r} in shard {si}"
                )
            for p in self.poolings:
                for li in range(self.n_layers):
                    merged[p][li][global_i] = pmap[p][li][j].astype(np.float32, copy=False)
            cursors[si] += 1

        for si, c in enumerate(cursors):
            if c != len(parts[si][0]):
                raise RuntimeError(
                    f"shard {si} had {len(parts[si][0])} texts but merge consumed {c}"
                )

        self._pooled = merged
        if len(self.poolings) == 1:
            p0 = self.poolings[0]
            self._save_to_hdf5_legacy(out_tmp, full_texts, merged[p0])
        else:
            self._save_to_hdf5_multi(out_tmp, full_texts, merged)

    # ------------------------------------------------------------------ #
    # Internal: HDF5                                                       #
    # ------------------------------------------------------------------ #

    def _save_to_hdf5_legacy(
        self, path: str, texts: List[str], layer_embs: Dict[int, np.ndarray]
    ) -> None:
        with h5py.File(path, "w") as f:
            dt = h5py.special_dtype(vlen=bytes)
            f.create_dataset("texts", data=[t.encode("utf-8") for t in texts], dtype=dt)
            grp = f.create_group("layers")
            for layer_idx, embs in layer_embs.items():
                embs_f32 = np.asarray(embs, dtype=np.float32)
                grp.create_dataset(
                    str(layer_idx),
                    data=embs_f32,
                    dtype=np.float32,
                    compression="lzf",
                    chunks=(min(256, len(texts)), embs_f32.shape[1]),
                )
            f.create_dataset("complete", data=np.array([1], dtype=np.int8))

    def _save_to_hdf5_multi(
        self, path: str, texts: List[str], pooled: Dict[str, Dict[int, np.ndarray]]
    ) -> None:
        with h5py.File(path, "w") as f:
            dt = h5py.special_dtype(vlen=bytes)
            f.create_dataset("texts", data=[t.encode("utf-8") for t in texts], dtype=dt)
            f.create_dataset("format_version", data=np.array([2], dtype=np.int8))
            f.create_dataset(
                "poolings",
                data=[p.encode("utf-8") for p in self.poolings],
                dtype=dt,
            )
            lg = f.create_group("layers")
            for p, layer_embs in pooled.items():
                pg = lg.create_group(p)
                for layer_idx, embs in layer_embs.items():
                    embs_f32 = np.asarray(embs, dtype=np.float32)
                    pg.create_dataset(
                        str(layer_idx),
                        data=embs_f32,
                        dtype=np.float32,
                        compression="lzf",
                        chunks=(min(256, len(texts)), embs_f32.shape[1]),
                    )
            f.create_dataset("complete", data=np.array([1], dtype=np.int8))

    def _load_from_hdf5(self, path: str, texts: List[str]) -> None:
        with h5py.File(path, "r") as f:
            stored_texts = [t.decode("utf-8") for t in f["texts"][:]]
            self._text_index = {t: i for i, t in enumerate(stored_texts)}
            layers_grp = f["layers"]

            if "format_version" in f and int(f["format_version"][0]) == 2:
                raw = f["poolings"][:]
                stored = [
                    x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
                    for x in raw
                ]
                if sorted(stored) != sorted(self.poolings):
                    raise KeyError(
                        f"HDF5 poolings {stored!r} != store poolings {self.poolings!r} — "
                        f"delete {path} to trigger recomputation"
                    )
                self._pooled = {}
                for p in self.poolings:
                    grp = layers_grp[p]
                    self._pooled[p] = {int(k): grp[k][:] for k in grp.keys()}
            else:
                p0 = self.poolings[0]
                self._pooled = {
                    p0: {int(k): layers_grp[k][:] for k in layers_grp.keys()}
                }

        missing = [t for t in texts if t not in self._text_index]
        if missing:
            raise KeyError(
                f"{len(missing)} texts not found in HDF5 store — "
                f"delete {path} to trigger recomputation"
            )

    @staticmethod
    def _hdf5_is_valid(path: str) -> bool:
        try:
            with h5py.File(path, "r") as f:
                if not (
                    "complete" in f
                    and int(f["complete"][0]) == 1
                    and "keys" not in f
                    and "layers" in f
                    and "texts" in f
                ):
                    return False
                layers_grp = f["layers"]
                if "format_version" in f:
                    if int(f["format_version"][0]) != 2:
                        return False
                    if "poolings" not in f:
                        return False
                    subkeys = list(layers_grp.keys())
                    if not subkeys:
                        return False
                    first = layers_grp[subkeys[0]]
                    return isinstance(first, h5py.Group) and len(first.keys()) > 0
                for k in layers_grp.keys():
                    if not str(k).isdigit():
                        return False
                return True
        except Exception:
            return False

    def _cache_file_path(self, dataset_name: str, split_name: str) -> Path:
        return layer_store_hdf5_path(
            self.cache_dir,
            self.model_name,
            dataset_name,
            split_name,
            self.poolings,
            self.n_layers,
        )


def _import_lmdb():
    """Lazy import so ``LayerEmbeddingStore`` / HDF5 paths work without ``lmdb`` installed."""
    try:
        import lmdb
    except ImportError as e:
        raise ImportError(
            "LMDBEmbeddingCache requires the 'lmdb' package (pip install lmdb). "
            "LayerEmbeddingStore / HDF5 does not use LMDB."
        ) from e
    return lmdb


class LMDBEmbeddingCache:
    """
    Fast drop-in replacement for SQLite embedding cache.

    - Memory-mapped reads: ~10-20x faster than SQLite for embedding lookups.
    - Corruption safety: any lmdb.Error during read or open triggers a full
      cache wipe and fresh start — no silent bad data, no crashes.
    - Auto-growing map: doubles map_size on MapFullError instead of crashing.
    - Same get/set interface as existing SQLite cache.
    """

    _NUMPY_MAGIC = b"NP"   # 2-byte magic prefix to detect valid entries

    def __init__(
        self,
        cache_dir: str,
        map_size: int = 10 * 1024 ** 3,    # 10 GB virtual address space
        readonly: bool = False,
    ):
        self.cache_path = cache_dir
        self.map_size   = map_size
        self.readonly   = readonly
        self._env       = None
        self._open()

    # ------------------------------------------------------------------ #
    # Public interface (same as SQLite cache)                              #
    # ------------------------------------------------------------------ #

    def get(self, key: str) -> Optional[np.ndarray]:
        try:
            with self._env.begin() as txn:
                data = txn.get(key.encode())
            if data is None:
                return None
            return self._deserialize(data)
        except Exception as e:
            logger.warning(f"LMDB read error (key={key!r}): {e} — invalidating cache")
            self._invalidate_and_reopen()
            return None

    def set(self, key: str, embedding: np.ndarray) -> None:
        data = self._serialize(embedding)
        self._write(key.encode(), data)

    def __contains__(self, key: str) -> bool:
        try:
            with self._env.begin() as txn:
                return txn.get(key.encode()) is not None
        except Exception:
            return False

    def close(self) -> None:
        if self._env:
            self._env.close()
            self._env = None

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _open(self) -> None:
        lmdb = _import_lmdb()
        try:
            self._env = lmdb.open(
                self.cache_path,
                map_size=self.map_size,
                max_readers=128,
                readonly=self.readonly,
                lock=not self.readonly,
                readahead=False,    # disable for random-access workloads
                meminit=False,      # skip zero-fill for speed
                metasync=False,     # don't fsync meta on every commit
                sync=False,         # async flushes (safe enough; we detect corruption)
            )
        except Exception as e:
            logger.warning(f"LMDB open failed at {self.cache_path!r}: {e} — recreating")
            shutil.rmtree(self.cache_path, ignore_errors=True)
            self._env = lmdb.open(
                self.cache_path,
                map_size=self.map_size,
                max_readers=128,
                readahead=False,
                meminit=False,
            )

    def _write(self, key: bytes, data: bytes) -> None:
        lmdb = _import_lmdb()
        try:
            with self._env.begin(write=True) as txn:
                txn.put(key, data)
        except lmdb.MapFullError:
            # Grow map size and retry
            self._env.close()
            self.map_size *= 2
            logger.info(f"LMDB map full — growing to {self.map_size // 1024**3} GB")
            self._open()
            with self._env.begin(write=True) as txn:
                txn.put(key, data)
        except Exception as e:
            logger.warning(f"LMDB write error: {e} — invalidating cache")
            self._invalidate_and_reopen()

    def _invalidate_and_reopen(self) -> None:
        self.close()
        shutil.rmtree(self.cache_path, ignore_errors=True)
        logger.warning(f"Recreated LMDB cache at {self.cache_path!r}")
        self._open()

    def _serialize(self, arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        buf.write(self._NUMPY_MAGIC)
        np.save(buf, arr, allow_pickle=False)
        return buf.getvalue()

    def _deserialize(self, data: bytes) -> np.ndarray:
        if data[:2] != self._NUMPY_MAGIC:
            raise ValueError("Invalid LMDB cache entry: missing magic bytes")
        return np.load(io.BytesIO(data[2:]), allow_pickle=False)

# cache_manager.py - SQLite version (more robust)


class SQLiteEmbeddingCache:
    """SQLite-based cache - more robust than LMDB"""
    
    def __init__(self, cache_dir: str = "./embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.connections = {}
        self.memory_cache = {}
        self.max_memory_items = 10000
    
    def _get_db_path(self, model_name: str, layer_idx: int, pooling: str) -> Path:
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_name}_L{layer_idx}_{pooling}.db"
    
    def _get_connection(self, model_name: str, layer_idx: int, pooling: str):
        import sqlite3

        db_path = self._get_db_path(model_name, layer_idx, pooling)
        key = str(db_path)
        
        if key not in self.connections:
            try:
                conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
                conn.execute('PRAGMA journal_mode=WAL')  # Better concurrency
                conn.execute('PRAGMA synchronous=NORMAL')  # Faster writes
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS embeddings (
                        sentence_hash TEXT PRIMARY KEY,
                        embedding BLOB
                    )
                ''')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_hash ON embeddings(sentence_hash)')
                conn.commit()
                self.connections[key] = conn
            except sqlite3.Error as e:
                logger.error(f"SQLite error: {e}")
                # Delete corrupted database
                if db_path.exists():
                    db_path.unlink()
                    return self._get_connection(model_name, layer_idx, pooling)
                raise
        
        return self.connections[key]
    
    def _get_key(self, sentence: str) -> str:
        normalized = " ".join(sentence.split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_sentence(self, model_name: str, layer_idx: int, 
                    pooling: str, sentence: str) -> Optional[np.ndarray]:
        mem_key = f"{model_name}_{layer_idx}_{pooling}_{sentence[:50]}"
        if mem_key in self.memory_cache:
            return self.memory_cache[mem_key]
        
        try:
            conn = self._get_connection(model_name, layer_idx, pooling)
            key = self._get_key(sentence)
            
            cursor = conn.execute('SELECT embedding FROM embeddings WHERE sentence_hash = ?', (key,))
            row = cursor.fetchone()
            
            if row:
                embedding = pickle.loads(row[0])
                
                if len(self.memory_cache) >= self.max_memory_items:
                    self.memory_cache.pop(next(iter(self.memory_cache)))
                self.memory_cache[mem_key] = embedding
                
                return embedding
        
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    def set_sentence(self, model_name: str, layer_idx: int, 
                    pooling: str, sentence: str, embedding: np.ndarray):
        mem_key = f"{model_name}_{layer_idx}_{pooling}_{sentence[:50]}"
        self.memory_cache[mem_key] = embedding
        
        try:
            conn = self._get_connection(model_name, layer_idx, pooling)
            key = self._get_key(sentence)
            blob = pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)
            
            conn.execute('''
                INSERT OR REPLACE INTO embeddings (sentence_hash, embedding)
                VALUES (?, ?)
            ''', (key, blob))
            conn.commit()
        
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def close(self):
        for conn in self.connections.values():
            try:
                conn.close()
            except:
                pass
        self.connections.clear()
        self.memory_cache.clear()




class QualityCache:
    """Cache for layer quality scores by (model, pooling, dataset)"""
    
    def __init__(self, cache_dir: str = "./quality_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, model_name: str, pooling: str, task_name: str) -> str:
        """Generate cache key for quality scores"""
        key_str = f"{model_name}_{pooling}_{task_name}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"quality_{key_hash}.pkl"
    
    def get(self, model_name: str, pooling: str, task_name: str) -> Optional[np.ndarray]:
        """Retrieve cached quality scores"""
        cache_file = self.cache_dir / self._get_cache_key(model_name, pooling, task_name)
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                print(f"✓ Quality cache hit: {task_name}, {pooling}")
                return data['layer_quality']
        return None
    
    def set(self, model_name: str, pooling: str, task_name: str, 
            layer_quality: np.ndarray):
        """Store quality scores in cache"""
        cache_file = self.cache_dir / self._get_cache_key(model_name, pooling, task_name)
        
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'layer_quality': layer_quality,
                'model_name': model_name,
                'pooling': pooling,
                'task_name': task_name
            }, f)
        print(f"✓ Quality cached: {task_name}, {pooling}")
