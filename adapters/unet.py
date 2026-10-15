"""Adapter for a plain trained-from-scratch U-Net, used as the non-foundation-model
baseline in milestone 4. Same Adapter contract as the FM adapters — the point of
this adapter is to prove the eouq UQ layers don't care whether the backbone is
frozen+pretrained or trained end-to-end.
"""
from __future__ import annotations

from adapters.base import Adapter, ExtractedOutputs


class UNetAdapter(Adapter):
    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = device

    def extract(self, dataloader) -> ExtractedOutputs:
        raise NotImplementedError
