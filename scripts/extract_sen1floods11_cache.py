"""Extract and cache per-chip outputs (probs, embeddings if available,
coords, mask, F1) for Sen1Floods11, for either the Prithvi adapter or the
U-Net baseline adapter -- the whole point of milestone 4 is that only the
adapter changes, nothing downstream.

Not runnable in this sandbox -- run on the cluster:
  apptainer exec --nv --bind $PWD:/workspace --bind $PWD/data:/data \\
      --bind $PWD/outputs:/outputs container/eouq.sif \\
      python -m scripts.extract_sen1floods11_cache --model prithvi
  apptainer exec --nv --bind $PWD:/workspace --bind $PWD/data:/data \\
      --bind $PWD/outputs:/outputs container/eouq.sif \\
      python -m scripts.extract_sen1floods11_cache --model unet
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from torch.utils.data import DataLoader

from adapters.prithvi import PrithviAdapter
from adapters.unet import UNetAdapter
from data.sen1floods11 import Sen1Floods11Dataset, holdout_continent_split

BATCH_SIZE = 4


def per_chip_f1(mean_probs: np.ndarray, masks: np.ndarray, prob_threshold: float = 0.5) -> np.ndarray:
    """Same definition as scripts/extract_burn_scars_cache.py's per_chip_f1."""
    event_prob = mean_probs[:, -1, :, :]
    predicted = (event_prob >= prob_threshold).astype(np.int64)

    f1_scores = np.empty(len(masks), dtype=np.float64)
    for i in range(len(masks)):
        valid = masks[i] != -1
        pred, true = predicted[i][valid], masks[i][valid]
        tp = np.sum((pred == 1) & (true == 1))
        fp = np.sum((pred == 1) & (true == 0))
        fn = np.sum((pred == 0) & (true == 1))
        denom = 2 * tp + fp + fn
        f1_scores[i] = (2 * tp / denom) if denom > 0 else float("nan")
    return f1_scores


def build_adapter(model_name: str, decoder_dir: str, device: str, use_location_coords: bool = True):
    if model_name == "prithvi":
        return PrithviAdapter(device=device, use_location_coords=use_location_coords)
    if model_name == "unet":
        checkpoints = sorted(glob.glob(os.path.join(decoder_dir, "unet_member_*.pt")))
        if not checkpoints:
            raise FileNotFoundError(
                f"No U-Net checkpoints found in {decoder_dir} -- run scripts/train_sen1floods11_unet.py first."
            )
        return UNetAdapter(checkpoints=checkpoints, device=device)
    raise ValueError(f"model must be 'prithvi' or 'unet', got {model_name!r}")


def extract_split(adapter, data_root: str, chip_ids: list, output_path: str) -> None:
    dataset = Sen1Floods11Dataset(data_root, chip_ids)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    outputs = adapter.extract(dataloader)
    mean_probs = outputs.probs.mean(axis=0)
    f1_scores = per_chip_f1(mean_probs, outputs.extra["mask"])

    save_kwargs = dict(
        probs=outputs.probs,
        coords=outputs.coords,
        chip_id=outputs.extra["chip_id"],
        mask=outputs.extra["mask"],
        f1=f1_scores,
    )
    if outputs.embeddings is not None:
        save_kwargs["embeddings"] = outputs.embeddings
    np.savez(output_path, **save_kwargs)
    print(f"wrote {output_path}: {len(chip_ids)} chips, mean F1={np.nanmean(f1_scores):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["prithvi", "unet"], required=True)
    parser.add_argument("--data-root", default="/data/sen1floods11")
    parser.add_argument("--decoder-dir", default="/outputs/sen1floods11_unet")
    parser.add_argument("--output-dir", default=None, help="defaults to /outputs/sen1floods11_cache_<model>")
    parser.add_argument("--holdout-continent", default="Africa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-location-coords",
        action="store_true",
        help="Prithvi only: pass location_coords=None (its config.yaml says it was fine-tuned with use_metadata: false)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or f"/outputs/sen1floods11_cache_{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    adapter = build_adapter(args.model, args.decoder_dir, args.device, use_location_coords=not args.no_location_coords)
    splits = holdout_continent_split(args.data_root, holdout_continent=args.holdout_continent, seed=0)

    for split_name, chip_ids in splits.items():
        if len(chip_ids) == 0:
            print(f"skipping {split_name}: no chips (holdout continent {args.holdout_continent} not present?)")
            continue
        output_path = os.path.join(output_dir, f"{split_name}.npz")
        extract_split(adapter, args.data_root, chip_ids, output_path)


if __name__ == "__main__":
    main()
