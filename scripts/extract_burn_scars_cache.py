"""Extract and cache per-chip SSL4EO-S12 + decoder-ensemble outputs for burn
scars: probs, embeddings, coords, and per-chip F1 (used only to build fusion
training labels, never fed into the UQ layers themselves).

Not runnable in this sandbox — run on the cluster after training the
ensemble (scripts/train_burn_scars_ensemble.py):
  apptainer exec --nv --bind $PWD:/workspace --bind $PWD/data:/data \\
      --bind $PWD/outputs:/outputs container/eouq.sif \\
      python /workspace/scripts/extract_burn_scars_cache.py
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from torch.utils.data import DataLoader

from adapters.ssl4eo_vit import SSL4EOViTAdapter
from data.burn_scars import HLSBurnScarsDataset, resplit

BATCH_SIZE = 8


def per_chip_f1(mean_probs: np.ndarray, masks: np.ndarray, prob_threshold: float = 0.5) -> np.ndarray:
    """mean_probs: (n_samples, n_classes, H, W); masks: (n_samples, H, W) with
    missing-data pixels marked -1 (excluded from the F1 computation)."""
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


def extract_split(adapter: SSL4EOViTAdapter, data_root: str, chip_ids: list, output_path: str) -> None:
    dataset = HLSBurnScarsDataset(data_root, chip_ids)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    outputs = adapter.extract(dataloader)
    mean_probs = outputs.probs.mean(axis=0)  # (n_samples, n_classes, H, W)
    f1_scores = per_chip_f1(mean_probs, outputs.extra["mask"])

    np.savez(
        output_path,
        probs=outputs.probs,
        embeddings=outputs.embeddings,
        coords=outputs.coords,
        chip_id=outputs.extra["chip_id"],
        mask=outputs.extra["mask"],
        f1=f1_scores,
    )
    print(f"wrote {output_path}: {len(chip_ids)} chips, mean F1={np.nanmean(f1_scores):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/burn_scars")
    parser.add_argument("--decoder-dir", default="/outputs/burn_scars_decoders")
    parser.add_argument("--output-dir", default="/outputs/burn_scars_cache")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    decoder_checkpoints = sorted(glob.glob(os.path.join(args.decoder_dir, "decoder_member_*.pt")))
    if not decoder_checkpoints:
        raise FileNotFoundError(
            f"No decoder checkpoints found in {args.decoder_dir} — run "
            "scripts/train_burn_scars_ensemble.py first."
        )

    adapter = SSL4EOViTAdapter(backbone_checkpoint=None, decoder_checkpoints=decoder_checkpoints, device=args.device)
    splits = resplit(args.data_root, seed=0)

    for split_name, chip_ids in splits.items():
        output_path = os.path.join(args.output_dir, f"{split_name}.npz")
        extract_split(adapter, args.data_root, chip_ids, output_path)


if __name__ == "__main__":
    main()
