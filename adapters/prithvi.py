"""Adapter for Prithvi-EO-2.0-300M-TL (ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11)
via TerraTorch, used for the milestone-4 model-agnostic check. Same Adapter contract as
ssl4eo_vit.py so the rest of the eouq pipeline is unchanged when swapping backbones.

Unlike SSL4EO-S12, this checkpoint is already fine-tuned end-to-end for flood
segmentation -- there's no decoder left to train ourselves, so there's no
deep ensemble of independently-trained members either. The predictive-
uncertainty (U_T) signal instead comes from test-time augmentation (TTA):
the same frozen model is run on 5 geometric views of each chip (identity,
horizontal/vertical flip, 90/180-degree rotation), predictions are
inverse-transformed back to the original orientation, and the 5 results are
treated as "ensemble members" by every downstream eouq layer exactly like a
real trained ensemble would be -- the UQ pipeline itself needs no changes
to accept this. See NOTES.md for why TTA was chosen over MC-dropout (this
checkpoint's dropout availability is unconfirmed; TTA works on any frozen
model by construction).

Two unverified assumptions, both flagged in NOTES.md as the highest-risk
parts of Milestone 4:
  - `BACKBONE_REGISTRY.build(...)` is the model card's documented loading
    method, but may return only the encoder rather than a full
    encoder+decode-head segmentation model. If `model(images)` doesn't come
    back as (batch, n_classes, H, W) logits, TerraTorch's task/full-model
    registry (e.g. wrapping the backbone in a SemanticSegmentationTask) is
    probably needed instead.
  - Output class count/order: the model card's "no water (0), water (1), no
    data (-1)" phrasing most likely describes the *label* convention (same
    as the ground-truth raster), not necessarily 3 separate output
    channels. Assumed here: 2 output channels (not-water, water), with
    missing-data handled purely as ground-truth/input masking -- the same
    convention already used for burn scars. Fix `EVENT_CLASS_INDEX` below
    if the real output has 3 channels instead.
"""
from __future__ import annotations

import numpy as np
import torch

from adapters.base import Adapter, ExtractedOutputs

EVENT_CLASS_INDEX = -1  # index of the "water" class within the model's output channels


def _flip_h(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


def _flip_v(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2])


def _rot90(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=1, dims=(-2, -1))


def _rot90_inv(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=-1, dims=(-2, -1))


def _rot180(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=2, dims=(-2, -1))


# (forward transform, inverse transform) pairs -- identity, flips, and 180-deg
# rotation are self-inverse; 90-deg rotation is not.
TTA_TRANSFORMS = [
    (lambda x: x, lambda x: x),
    (_flip_h, _flip_h),
    (_flip_v, _flip_v),
    (_rot90, _rot90_inv),
    (_rot180, _rot180),
]


class PrithviAdapter(Adapter):
    def __init__(
        self, checkpoint: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11", device: str = "cuda"
    ):
        self.checkpoint = checkpoint
        self.device = device
        self._model = None

    def _load_model(self):
        from terratorch.registry import BACKBONE_REGISTRY

        model = BACKBONE_REGISTRY.build(self.checkpoint)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model.to(self.device)

    def _backbone_embedding(self, model, images: torch.Tensor) -> torch.Tensor:
        """Best-guess API: a forward_features-style call returning per-chip
        token features, mean-pooled over tokens if needed. Unverified."""
        features = model.forward_features(images)
        if features.dim() == 3:  # (batch, n_tokens, dim) -> mean-pool tokens
            features = features.mean(dim=1)
        return features

    def extract(self, dataloader) -> ExtractedOutputs:
        if self._model is None:
            self._model = self._load_model()
        model = self._model

        all_probs, all_embeddings, all_coords, all_chip_ids, all_masks = [], [], [], [], []
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)

                member_logits = []
                for forward_tf, inverse_tf in TTA_TRANSFORMS:
                    logits = model(forward_tf(images))
                    member_logits.append(inverse_tf(logits))
                member_logits = torch.stack(member_logits, dim=0)  # (n_members, batch, n_classes, H, W)
                member_probs = torch.softmax(member_logits, dim=2).cpu().numpy()

                embeddings = self._backbone_embedding(model, images)

                all_probs.append(member_probs)
                all_embeddings.append(embeddings.cpu().numpy())
                all_coords.append(batch["coords"].cpu().numpy())
                all_chip_ids.extend(batch["chip_id"])
                all_masks.append(batch["mask"].cpu().numpy())

        probs = np.concatenate(all_probs, axis=1)
        embeddings = np.concatenate(all_embeddings, axis=0)
        coords = np.concatenate(all_coords, axis=0)
        masks = np.concatenate(all_masks, axis=0)

        return ExtractedOutputs(
            probs=probs,
            embeddings=embeddings,
            coords=coords,
            extra={"chip_id": np.array(all_chip_ids), "mask": masks},
        )
