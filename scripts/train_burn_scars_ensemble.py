"""Train the burn-scars decoder ensemble on top of the frozen SSL4EO-S12
ViT-S/16 backbone. Bootstrap ensemble per SHRUG-FM (see NOTES.md): each
member trains independently on a bootstrap resample of the 485 training
chips.

Not runnable in this sandbox (no python/GPU here) — run on the cluster:
  apptainer exec --nv --bind $PWD:/workspace --bind $PWD/data:/data \\
      --bind $PWD/outputs:/outputs container/eouq.sif \\
      python /workspace/scripts/train_burn_scars_ensemble.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from adapters.ssl4eo_decoder import PatchTokenDecoder
from adapters.ssl4eo_vit import hls_to_sentinel2, resize_images_and_masks
from data.burn_scars import HLSBurnScarsDataset, resplit

# SHRUG-FM doesn't publish an exact ensemble size; 5 is a reasonable default
# for a deep ensemble, not validated against the paper.
N_MEMBERS = 5
N_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3


def _bootstrap_indices(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=n)


def _load_frozen_backbone(device: str):
    import torchgeo.models as tgm

    weights = tgm.ViTSmall16_Weights.SENTINEL2_ALL_MOCO
    backbone = tgm.vit_small_patch16_224(weights=weights)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
    return backbone.to(device)


def train_one_member(backbone, train_loader, device, n_epochs=N_EPOCHS, lr=LEARNING_RATE):
    decoder = PatchTokenDecoder(embed_dim=384, patch_grid_size=14, n_classes=2).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # -1 = missing-data pixels

    decoder.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            images, masks = resize_images_and_masks(images, masks)
            images = hls_to_sentinel2(images)

            with torch.no_grad():
                tokens = backbone.forward_features(images)
                patch_tokens = tokens[:, 1:, :]

            logits = decoder(patch_tokens)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"  epoch {epoch + 1}/{n_epochs}: loss={running_loss / len(train_loader):.4f}")

    decoder.eval()
    return decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/burn_scars")
    parser.add_argument("--output-dir", default="/outputs/burn_scars_decoders")
    parser.add_argument("--n-members", type=int, default=N_MEMBERS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    splits = resplit(args.data_root, seed=0)
    train_dataset = HLSBurnScarsDataset(args.data_root, splits["train"])
    backbone = _load_frozen_backbone(args.device)

    for member_idx in range(args.n_members):
        print(f"training member {member_idx + 1}/{args.n_members}")
        bootstrap_idx = _bootstrap_indices(len(train_dataset), seed=member_idx)
        member_loader = DataLoader(Subset(train_dataset, bootstrap_idx), batch_size=BATCH_SIZE, shuffle=True)

        decoder = train_one_member(backbone, member_loader, args.device)

        checkpoint_path = os.path.join(args.output_dir, f"decoder_member_{member_idx}.pt")
        torch.save(decoder.state_dict(), checkpoint_path)
        print(f"  saved {checkpoint_path}")


if __name__ == "__main__":
    main()
