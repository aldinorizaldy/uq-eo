"""Adapter for the frozen SSL4EO-S12 ViT-S/16 (MoCo) backbone + trained CNN decoder,
as used in SHRUG-FM. Loads the frozen backbone, runs the decoder ensemble, and
returns probs/embeddings/coords via the shared Adapter contract. To be filled in
during milestone 2 once the checkpoint source and decoder architecture are confirmed
against the SHRUG-FM paper/repo.
"""
from __future__ import annotations

from adapters.base import Adapter, ExtractedOutputs


class SSL4EOViTAdapter(Adapter):
    def __init__(self, backbone_checkpoint: str, decoder_checkpoints: list[str], device: str = "cuda"):
        self.backbone_checkpoint = backbone_checkpoint
        self.decoder_checkpoints = decoder_checkpoints
        self.device = device

    def extract(self, dataloader) -> ExtractedOutputs:
        raise NotImplementedError
