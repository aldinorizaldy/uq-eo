"""Adapter for a plain trained-from-scratch U-Net ensemble, used as the
non-foundation-model baseline in milestone 4. Same Adapter contract as the
FM adapters — the point of this adapter is to prove the eouq UQ layers
don't care whether the backbone is frozen+pretrained or trained end-to-end.

Signature changed from the original stub: `checkpoint: str` became
`checkpoints: list[str]`, since a standard ensemble (per NOTES.md: bootstrap
for burn scars, standard for floods) needs multiple independently-trained
members, not one.

Unlike the FM adapters, a from-scratch U-Net has no pretrained backbone to
extract an embedding from for U_E -- there is no "familiar vs. unfamiliar
to pretraining" question for a model that was never pretrained.
`extract()` returns `embeddings=None`; L1 (embedding OOD) simply doesn't
apply to this adapter. That asymmetry is itself a meaningful part of the
model-agnostic check — the rest of the pipeline should degrade gracefully
(skip U_E, keep U_T/fusion/metrics working) rather than require every
signal from every adapter.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from adapters.base import Adapter, ExtractedOutputs
from adapters.unet_model import UNet


class UNetAdapter(Adapter):
    def __init__(self, checkpoints: list[str], device: str = "cuda", n_classes: int = 2, in_channels: int = 6):
        self.checkpoints = checkpoints
        self.device = device
        self.n_classes = n_classes
        self.in_channels = in_channels
        self._members = None

    def _load_members(self):
        members = []
        for path in self.checkpoints:
            model = UNet(in_channels=self.in_channels, n_classes=self.n_classes)
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            members.append(model.to(self.device))
        return members

    def extract(self, dataloader) -> ExtractedOutputs:
        if self._members is None:
            self._members = self._load_members()
        members = self._members

        all_probs, all_coords, all_chip_ids, all_masks = [], [], [], []
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                member_logits = torch.stack([model(images) for model in members], dim=0)
                member_probs = F.softmax(member_logits, dim=2).cpu().numpy()

                all_probs.append(member_probs)
                all_coords.append(batch["coords"].cpu().numpy())
                all_chip_ids.extend(batch["chip_id"])
                all_masks.append(batch["mask"].cpu().numpy())

        probs = np.concatenate(all_probs, axis=1)
        coords = np.concatenate(all_coords, axis=0)
        masks = np.concatenate(all_masks, axis=0)

        return ExtractedOutputs(
            probs=probs,
            embeddings=None,
            coords=coords,
            extra={"chip_id": np.array(all_chip_ids), "mask": masks},
        )
