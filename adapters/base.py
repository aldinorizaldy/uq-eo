"""Model-agnostic adapter contract.

Every adapter exposes the same `extract` call so the caching stage (and
therefore every eouq UQ layer downstream) never needs to know which model
or decoder produced the outputs. Model-specific code — checkpoint loading,
preprocessing, decoder heads — stays entirely inside the adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ExtractedOutputs:
    probs: np.ndarray  # (n_samples, ..., n_classes) or per-member (n_members, n_samples, ..., n_classes)
    embeddings: np.ndarray | None = None  # (n_samples, embed_dim)
    coords: np.ndarray | None = None  # (n_samples, 2) lat/lon, or region labels
    extra: dict | None = None  # anything adapter-specific worth caching (e.g. logits, per-chip F1)


class Adapter:
    def extract(self, dataloader) -> ExtractedOutputs:
        raise NotImplementedError
