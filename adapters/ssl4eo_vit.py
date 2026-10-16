"""Adapter for the frozen SSL4EO-S12 ViT-S/16 (MoCo) backbone + trained CNN decoder
ensemble, as used in SHRUG-FM. Loads the frozen backbone, runs the decoder
ensemble, and returns probs/embeddings/coords via the shared Adapter contract.

Backbone weights come from torchgeo (see NOTES.md): the exact weights-enum
symbol (`ViTSmall16_Weights.SENTINEL2_ALL_MOCO` below) is a best guess at the
installed torchgeo version's API and has not been executed/verified in this
sandbox — confirm it against the actual installed version on first run and
fix the import if the symbol name differs.

Two input-prep steps are needed before HLS chips can go into this backbone,
neither validated against the paper (see NOTES.md):
  1. HLS Burn Scars ships 6 bands; SSL4EO-S12 was pretrained on 13-band
     Sentinel-2. `hls_to_sentinel2` zero-fills the 7 bands HLS doesn't have.
  2. HLS chips are 512x512; the backbone expects 224x224. `resize_images_and_masks`
     resizes both image and mask consistently (bilinear / nearest).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from adapters.base import Adapter, ExtractedOutputs
from adapters.ssl4eo_decoder import DecoderEnsemble

INPUT_SIZE = 224
N_S2_BANDS = 13
# HLS's 6 bands (Blue, Green, Red, NIR-narrow, SWIR1, SWIR2) mapped to their
# indices within 13-band Sentinel-2 L1C ordering [B1,B2,B3,B4,B5,B6,B7,B8,
# B8A,B9,B10,B11,B12]. The other 7 bands (B1,B5,B6,B7,B9,B10,B8) SSL4EO-S12
# saw during pretraining but HLS doesn't provide are left zero-filled — an
# approximation, not validated against the paper.
HLS_TO_S2_BAND_INDEX = {0: 1, 1: 2, 2: 3, 3: 8, 4: 11, 5: 12}


def hls_to_sentinel2(images: torch.Tensor) -> torch.Tensor:
    """images: (batch, 6, H, W) HLS bands -> (batch, 13, H, W), zero-filled Sentinel-2 ordering."""
    batch, _, height, width = images.shape
    s2 = images.new_zeros((batch, N_S2_BANDS, height, width))
    for hls_idx, s2_idx in HLS_TO_S2_BAND_INDEX.items():
        s2[:, s2_idx, :, :] = images[:, hls_idx, :, :]
    return s2


def resize_images_and_masks(images: torch.Tensor, masks: torch.Tensor, size: int = INPUT_SIZE):
    """images: (batch, C, H, W) float; masks: (batch, H, W) int (missing-data=-1).
    Bilinear-resizes images, nearest-resizes masks to keep integer labels intact."""
    resized_images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    resized_masks = F.interpolate(masks.unsqueeze(1).float(), size=(size, size), mode="nearest").squeeze(1).long()
    return resized_images, resized_masks


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
            state_dict = torch.load(self.backbone_checkpoint, map_location=self.device, weights_only=True)
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
                masks = batch["mask"].to(self.device)
                images, masks = resize_images_and_masks(images, masks)
                images = hls_to_sentinel2(images)

                tokens = backbone.forward_features(images)  # (batch, 1 + n_patches, embed_dim)
                cls_token, patch_tokens = tokens[:, 0, :], tokens[:, 1:, :]

                member_logits = ensemble(patch_tokens)  # (n_members, batch, n_classes, H, W)
                member_probs = F.softmax(member_logits, dim=2).cpu().numpy()

                all_probs.append(member_probs)
                all_embeddings.append(cls_token.cpu().numpy())
                all_coords.append(batch["coords"].cpu().numpy())
                all_chip_ids.extend(batch["chip_id"])
                all_masks.append(masks.cpu().numpy())

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
