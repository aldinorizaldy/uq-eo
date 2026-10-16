"""Lightweight CNN decoder for the frozen SSL4EO-S12 ViT-S/16 backbone.

Takes the ViT's patch-token grid (14x14x384 for a 224x224 input, patch=16)
and upsamples it back to a per-pixel class-logit map. This is our own,
deliberately simple decoder — SHRUG-FM's own decoder architecture is not
reproduced here; only the uncertainty-quantification methodology
(U_I/U_E/U_T, fusion, metrics) is being reimplemented from the paper, not
its exact segmentation head.
"""
from __future__ import annotations

import torch
from torch import nn


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.act(self.bn(self.conv(x)))


class PatchTokenDecoder(nn.Module):
    """embed_dim: ViT token dimension (384 for ViT-S). patch_grid_size: side
    length of the square patch-token grid (14 for a 224x224 input at patch
    size 16). n_classes: output channels (2 for binary burn/no-burn).
    4 upsample blocks (x2 each) take a 14x14 grid back to 224x224.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        patch_grid_size: int = 14,
        n_classes: int = 2,
        channels: tuple = (192, 96, 48, 24),
    ):
        super().__init__()
        self.patch_grid_size = patch_grid_size
        blocks = []
        in_ch = embed_dim
        for out_ch in channels:
            blocks.append(UpsampleBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(in_ch, n_classes, kernel_size=1)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """patch_tokens: (batch, n_patches, embed_dim) -> (batch, n_classes, H, W)."""
        batch, n_patches, embed_dim = patch_tokens.shape
        grid = self.patch_grid_size
        x = patch_tokens.transpose(1, 2).reshape(batch, embed_dim, grid, grid)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class DecoderEnsemble(nn.Module):
    """Wraps N independently-trained decoder heads over the same frozen backbone
    features, used for SHRUG-FM's deep-ensemble U_T signal (bootstrap ensemble
    for burn scars per NOTES.md)."""

    def __init__(self, decoders: list):
        super().__init__()
        self.decoders = nn.ModuleList(decoders)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Returns (n_members, batch, n_classes, H, W) logits, one per member."""
        return torch.stack([decoder(patch_tokens) for decoder in self.decoders], dim=0)

    @classmethod
    def from_checkpoints(cls, checkpoint_paths: list, decoder_kwargs: dict, device: str = "cpu"):
        decoders = []
        for path in checkpoint_paths:
            decoder = PatchTokenDecoder(**decoder_kwargs)
            state_dict = torch.load(path, map_location=device, weights_only=True)
            decoder.load_state_dict(state_dict)
            decoder.eval()
            decoders.append(decoder)
        return cls(decoders).to(device)
