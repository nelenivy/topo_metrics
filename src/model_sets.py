"""
Named HuggingFace model lists for grid runs (analogous to ``task_sets``).

Use ``--model-set`` when you do not pass ``--models``. Passing ``--models``
explicitly overrides the set.

Lists are English-focused, <3B, mixing encoder, sentence-transformer,
retrieval encoders, and small decoders. Edit to taste.
"""

from __future__ import annotations

from typing import Dict, List


def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# Fast, diverse baseline (encoder + ST + E5 + small causal)
CORE_MODELS: List[str] = [
    "bert-base-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-small-v2",
    "distilgpt2",
]

# Adds RoBERTa, MPNet, SimCSE, BGE, GTE, tiny Qwen instruct
STANDARD_MODELS: List[str] = _dedup_preserve_order(
    CORE_MODELS
    + [
        "roberta-base",
        "sentence-transformers/all-mpnet-base-v2",
        "princeton-nlp/sup-simcse-bert-base-uncased",
        "BAAI/bge-small-en-v1.5",
        "Alibaba-NLP/gte-base-en-v1.5",
        "Qwen/Qwen2-0.5B-Instruct",
    ]
)

# Larger English panel: more E5/BGE + small Mamba (needs trust_remote_code)
FULL_MODELS: List[str] = _dedup_preserve_order(
    STANDARD_MODELS
    + [
        "intfloat/e5-base-v2",
        "BAAI/bge-base-en-v1.5",
        "state-spaces/mamba-130m-hf",
    ]
)

MODEL_SET_MAP: Dict[str, List[str]] = {
    "core": CORE_MODELS,
    "standard": STANDARD_MODELS,
    "full": FULL_MODELS,
}
