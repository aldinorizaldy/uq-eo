"""Adapter for the frozen SSL4EO-S12 ViT-S/16 (MoCo) backbone + trained CNN decoder
ensemble, as used in SHRUG-FM. Loads the frozen backbone, runs the decoder
ensemble, and returns probs/embeddings/coords via the shared Adapter contract.

Backbone weights come from torchgeo (see NOTES.md): the exact weights-enum
symbol (`ViTSmall16_Weights.SENTINEL2_ALL_MOCO` below) is a best guess at the
installed torchgeo version's API and has not been executed/verified in this
sandbox — confirm it against the actual installed version on first run and
fix the import if the symbol name differs.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from adapters.base import Adapter, ExtractedOutputs
from adapters.ssl4eo_decoder import DecoderEnsemble


class SSL4EOViTAdapter(Adapter):
    def __init__(
        self,
        backbone_checkpoint: str,
        decoder_checkpoints: list[str],
        device: str = "cuda",
        n_classes: int = 2,
    ):
        self.backbone_checkpoint = backbone_checkpoint
        self.decoder_checkpoints = decoder_checkpoints
        self.device = device
        self.n_classes = n_classes
        self._backbone = None
        self._ensemble = None

    def _load_backbone(self):
        """Frozen SSL4EO-S12 ViT-S/16 MoCo encoder via torchgeo."""
        import torchgeo.models as tgm

        weights = tgm.ViTSmall16_Weights.SENTINEL2_ALL_MOCO
        backbone = tgm.vit_small_patch16_224(weights=weights)
        if self.backbone_checkpoint:
            state_dict = torch.load(self.backbone_checkpoint, map_location=self.device)
            backbone.load_state_dict(state_dict, strict=False)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad_(False)
        return backbone.to(self.device)

    def extract(self, dataloader) -> ExtractedOutputs:
        if self._backbone is None:
            self._backbone = self._load_backbone()
        if self._ensemble is None:
            self._ensemble = DecoderEnsemble.from_checkpoints(
                self.decoder_checkpoints,
                decoder_kwargs={"n_classes": self.n_classes},
                device=self.device,
            )
        backbone, ensemble = self._backbone, self._ensemble

        all_probs, all_embeddings, all_coords, all_chip_ids, all_masks = [], [], [], [], []
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                tokens = backbone.forward_features(images)  # (batch, 1 + n_patches, embed_dim)
                cls_token, patch_tokens = tokens[:, 0, :], tokens[:, 1:, :]

                member_logits = ensemble(patch_tokens)  # (n_members, batch, n_classes, H, W)
                member_probs = F.softmax(member_logits, dim=2).cpu().numpy()

                all_probs.append(member_probs)
                all_embeddings.append(cls_token.cpu().numpy())
                all_coords.append(batch["coords"].cpu().numpy())
                all_chip_ids.extend(batch["chip_id"])
                all_masks.append(batch["mask"].cpu().numpy())

        probs = np.concatenate(all_probs, axis=1)  # (n_members, n_samples, n_classes, H, W)
        embeddings = np.concatenate(all_embeddings, axis=0)
        coords = np.concatenate(all_coords, axis=0)
        masks = np.concatenate(all_masks, axis=0)

        return ExtractedOutputs(
            probs=probs,
            embeddings=embeddings,
            coords=coords,
            extra={"chip_id": np.array(all_chip_ids), "mask": masks},
        )
