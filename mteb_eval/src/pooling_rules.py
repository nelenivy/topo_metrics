"""
Explicit rules for (model_name, pooling) compatibility.

We skip invalid pairs by **name heuristics** (predictable logs), not by catching
exceptions during forward — failures can still occur for OOM, network, etc.

``cls`` is only meaningful for typical encoder / bi-encoder checkpoints with a
leading CLS token. Causal LMs (GPT, Llama, Qwen, Mamba, …) should use
``mean`` or ``last_token``.
"""

from __future__ import annotations

from typing import Tuple

import torch

# Substrings in HF ids that indicate decoder-only / causal stacks (no CLS semantics).
_CAUSAL_HINTS: Tuple[str, ...] = (
    "gpt2",
    "gpt-neo",
    "gpt-j",
    "llama",
    "meta-llama",
    "mistralai",
    "mistral",
    "qwen",
    "phi",
    "mamba",
    "gemma",
    "falcon",
    "bloom",
    "opt-",
    "dolly",
    "stablelm",
    "internlm",
    "yi-",
    "rwkv",
    "deepseek",
)


def model_name_suggests_causal_lm(model_name: str) -> bool:
    """Heuristic: True if ``model_name`` likely loads a causal / decoder backbone."""
    n = model_name.lower().replace("\\", "/")
    for h in _CAUSAL_HINTS:
        if h in n:
            return True
    return False


def pooling_supported(model_name: str, pooling: str) -> bool:
    """Return False when this script should skip the (model, pooling) pair without running."""
    if pooling in ("mean", "last_token"):
        return True
    if pooling == "cls":
        if model_name_suggests_causal_lm(model_name):
            return False
        return True
    return False


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """
    Sentence embedding from one layer's hidden states (B, L, H) and a mask (B, L).

    Used after a single model forward to derive several poolings without re-running
    the backbone.
    """
    if pooling == "cls":
        return hidden_states[:, 0, :]

    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    if pooling == "last_token":
        last_pos = attention_mask.sum(dim=1) - 1
        last_pos = last_pos.clamp(min=0)
        b, _, h = hidden_states.shape
        idx = last_pos.view(b, 1, 1).expand(b, 1, h)
        return hidden_states.gather(dim=1, index=idx).squeeze(1)

    raise ValueError(
        f"Unknown pooling: {pooling!r}. Choose from: 'cls', 'mean', 'last_token'."
    )


def skip_reason(model_name: str, pooling: str) -> str:
    """Human-readable reason for skip (for logs)."""
    if pooling_supported(model_name, pooling):
        return ""
    if pooling == "cls" and model_name_suggests_causal_lm(model_name):
        return "causal/decoder LM — no CLS embedding; use mean or last_token"
    return "incompatible pooling"
