"""Train the Sen1Floods11 U-Net ensemble: a *standard* ensemble (every
member trained on the same full training set, different random seeds per
member) rather than bootstrap -- per SHRUG-FM's own choice for flood/
landslide tasks specifically (see NOTES.md), unlike burn scars' bootstrap
ensemble.

Not runnable in this sandbox -- run on the cluster:
  apptainer exec --nv --bind $PWD:/workspace --bind $PWD/data:/data \\
      --bind $PWD/outputs:/outputs container/eouq.sif \\
      python -m scripts.train_sen1floods11_unet
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from adapters.unet_model import UNet
from data.sen1floods11 import Sen1Floods11Dataset, holdout_continent_split

N_MEMBERS = 5
N_EPOCHS = 20
BATCH_SIZE = 4  # 512x512 chips are much bigger than burn scars' 224x224 working size
LEARNING_RATE = 1e-3


def train_one_member(train_loader, device: str, seed: int, n_epochs: int = N_EPOCHS, lr: float = LEARNING_RATE):
    torch.manual_seed(seed)
    model = UNet(in_channels=6, n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # -1 = missing-data pixels

    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"  epoch {epoch + 1}/{n_epochs}: loss={running_loss / len(train_loader):.4f}")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/sen1floods11")
    parser.add_argument("--output-dir", default="/outputs/sen1floods11_unet")
    parser.add_argument("--holdout-continent", default="Africa")
    parser.add_argument("--n-members", type=int, default=N_MEMBERS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    splits = holdout_continent_split(args.data_root, holdout_continent=args.holdout_continent, seed=0)
    train_dataset = Sen1Floods11Dataset(args.data_root, splits["train"])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for member_idx in range(args.n_members):
        print(f"training member {member_idx + 1}/{args.n_members}")
        model = train_one_member(train_loader, args.device, seed=member_idx)

        checkpoint_path = os.path.join(args.output_dir, f"unet_member_{member_idx}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  saved {checkpoint_path}")


if __name__ == "__main__":
    main()
