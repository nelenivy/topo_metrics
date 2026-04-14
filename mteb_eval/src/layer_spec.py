"""
layer_spec.py
Enumerate all embedding configurations to evaluate:
  - individual layers (layer_0 … layer_N)
  - last-k averages: last_1, last_2, last_4, last_8
  - all-layers mean: all_mean
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

LAST_K_VALUES = [1, 2, 4, 8]


@dataclass
class LayerSpec:
    name: str           # human-readable key used in CSV
    spec_type: str      # "single" | "last_k" | "all_mean"
    layer_idx: int = -1 # valid for spec_type == "single"
    k: int = -1         # valid for spec_type == "last_k"

    def weights(self, n_layers: int) -> np.ndarray:
        """Return normalised weight vector of length n_layers."""
        w = np.zeros(n_layers, dtype=np.float32)
        if self.spec_type == "single":
            assert 0 <= self.layer_idx < n_layers
            w[self.layer_idx] = 1.0
        elif self.spec_type == "last_k":
            k = min(self.k, n_layers)
            w[-k:] = 1.0 / k
        elif self.spec_type == "all_mean":
            w[:] = 1.0 / n_layers
        else:
            raise ValueError(f"Unknown spec_type: {self.spec_type}")
        return w


def build_layer_specs(n_layers: int) -> List[LayerSpec]:
    """
    Build the full list of LayerSpecs for a model with n_layers hidden layers
    (including embedding layer 0).

    Returns specs in order:
        layer_0 … layer_{n_layers-1},
        last_1, last_2, last_4, last_8  (capped at n_layers),
        all_mean
    """
    specs: List[LayerSpec] = []

    for i in range(n_layers):
        specs.append(LayerSpec(name=f"layer_{i}", spec_type="single", layer_idx=i))

    for k in LAST_K_VALUES:
        if k <= n_layers:
            specs.append(LayerSpec(name=f"last_{k}", spec_type="last_k", k=k))

    specs.append(LayerSpec(name="all_mean", spec_type="all_mean"))

    return specs
