"""
Robust HuggingFace loading for encoder-only, decoder (causal), and custom (Mamba, etc.) models.

Used by LayerEncoder: AutoModel + output_hidden_states, tokenizer padding, trust_remote_code.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)

# Typical encoder-only / bi-encoder backbones (right-padded batching is fine)
_ENCODER_MODEL_TYPES = frozenset({
    "bert", "roberta", "mpnet", "xlm-roberta", "distilbert", "camembert",
    "electra", "albert", "deberta", "deberta-v2", "flaubert", "xlnet",
    "nystromformer", "longformer", "big_bird",
})


def _config_num_hidden_layers(config: Any) -> Optional[int]:
    """Best-effort layer count from config (transformers naming varies)."""
    for key in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        v = getattr(config, key, None)
        if isinstance(v, int) and v > 0:
            return v
    return None


def _is_encoder_only_config(config: Any) -> bool:
    mt = (getattr(config, "model_type", None) or "").lower()
    if mt in _ENCODER_MODEL_TYPES:
        return True
    # Explicit flags when present
    is_enc = getattr(config, "is_encoder", None)
    is_dec = getattr(config, "is_decoder", None)
    if is_enc is True and is_dec is not True:
        return True
    if is_dec is True and is_enc is not True:
        return False
    return False


def _parse_torch_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name or name in ("auto", "none", "None"):
        return None
    n = name.lower().strip()
    if n == "float16" or n == "fp16":
        return torch.float16
    if n == "bfloat16" or n == "bf16":
        return torch.bfloat16
    if n == "float32" or n == "fp32":
        return torch.float32
    raise ValueError(f"Unknown torch dtype: {name!r}")


def load_tokenizer(
    model_name: str,
    *,
    trust_remote_code: bool = True,
):
    """Fast tokenizer when possible; fall back to slow. Mamba/custom often need trust_remote_code."""
    kw = dict(trust_remote_code=trust_remote_code)
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kw)
    except Exception as e:
        logger.info("Fast tokenizer failed (%s), retrying slow tokenizer", e)
        return AutoTokenizer.from_pretrained(model_name, use_fast=False, **kw)


def configure_tokenizer(tokenizer: Any, config: Any) -> None:
    """Pad token + padding side for batched encoding (causal LMs need left padding for last-token semantics)."""
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
    if _is_encoder_only_config(config):
        tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "left"


def _load_automodel(
    model_name: str,
    device: str,
    *,
    trust_remote_code: bool,
    torch_dtype: Optional[torch.dtype],
):
    """Load backbone with hidden states. Tries AutoModel, then causal wrapper."""
    common = dict(
        trust_remote_code=trust_remote_code,
        output_hidden_states=True,
    )
    if torch_dtype is not None:
        common["torch_dtype"] = torch_dtype

    try:
        model = AutoModel.from_pretrained(model_name, **common)
        return model
    except Exception as e1:
        logger.info("AutoModel failed (%s), trying AutoModelForCausalLM", e1)
        model = AutoModelForCausalLM.from_pretrained(model_name, **common)
        # Llama/Qwen: .model; GPT-2: .transformer; fallback: full module if it exposes hidden_states
        inner = getattr(model, "model", None) or getattr(model, "transformer", None)
        if inner is not None and hasattr(inner, "forward"):
            return inner
        return model


def probe_num_hidden_tensors(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
) -> int:
    """Run one forward pass and count hidden_states tensors (embedding + blocks)."""
    model.eval()
    inputs = tokenizer(
        ["probe"],
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if not hs:
        raise RuntimeError(
            "Model forward did not return hidden_states; this architecture may not support "
            "per-layer extraction for layer aggregation."
        )
    return len(hs)


def load_encoder_model_for_layers(
    model_name: str,
    device: str,
    *,
    trust_remote_code: bool = True,
    torch_dtype: Optional[str] = None,
) -> Tuple[torch.nn.Module, Any, int]:
    """
    Returns (model, tokenizer, num_layers) where num_layers == len(hidden_states)
    (embedding + transformer blocks, consistent with LayerEncoder indexing).
    """
    dtype = _parse_torch_dtype(torch_dtype)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer = load_tokenizer(model_name, trust_remote_code=trust_remote_code)
    configure_tokenizer(tokenizer, config)

    model = _load_automodel(model_name, device, trust_remote_code=trust_remote_code, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    n_tensors = probe_num_hidden_tensors(model, tokenizer, device)
    cfg_n = _config_num_hidden_layers(config)
    if cfg_n is not None and cfg_n + 1 != n_tensors:
        logger.warning(
            "Config num_hidden_layers=%s implies %s tensors but forward returned %s; using probe count.",
            cfg_n,
            cfg_n + 1,
            n_tensors,
        )
    return model, tokenizer, n_tensors
