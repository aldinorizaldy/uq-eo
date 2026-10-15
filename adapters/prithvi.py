"""Adapter for Prithvi-EO-2.0-300M-TL (ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11)
via TerraTorch, used for the milestone-4 model-agnostic check. Same Adapter contract as
ssl4eo_vit.py so the rest of the eouq pipeline is unchanged when swapping backbones.
"""
from __future__ import annotations

from adapters.base import Adapter, ExtractedOutputs


class PrithviAdapter(Adapter):
    def __init__(self, checkpoint: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11", device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = device

    def extract(self, dataloader) -> ExtractedOutputs:
        raise NotImplementedError
