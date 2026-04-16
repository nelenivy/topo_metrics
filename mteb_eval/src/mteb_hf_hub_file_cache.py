"""
Cache MTEB's Hugging Face hub file resolution to avoid repeated network + WARNING spam.

MTEB's ``model_meta._detect_model_type_and_loader`` always tries ``modules.json`` first.
Plain Transformers checkpoints (e.g. ``bert-base-uncased``) do not ship that file, so each
call used to hit the Hub and log ``Can't get file modules.json ... 404``.

Patching ``_get_file_on_hub`` with ``functools.lru_cache`` makes the miss a one-time
network round-trip per (repo, file, type, revision). Disable with env
``MTEB_HF_HUB_FILE_CACHE=0``.
"""

from __future__ import annotations

import functools
import os
from typing import Callable, Optional

_patched = False
_orig_get_file: Optional[Callable[..., Optional[str]]] = None


def patch_mteb_hub_file_cache() -> None:
    """Idempotent: wrap ``mteb.models.model_meta._get_file_on_hub`` with LRU cache."""
    global _patched, _orig_get_file
    if _patched:
        return
    if os.environ.get("MTEB_HF_HUB_FILE_CACHE", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        _patched = True
        return

    from mteb.models import model_meta as mm

    _orig_get_file = mm._get_file_on_hub

    @functools.lru_cache(maxsize=4096)
    def _cached(
        repo_id: str, file_name: str, repo_type: str, revision: str | None
    ) -> str | None:
        assert _orig_get_file is not None
        return _orig_get_file(repo_id, file_name, repo_type, revision)

    mm._get_file_on_hub = _cached  # type: ignore[method-assign]
    _patched = True
