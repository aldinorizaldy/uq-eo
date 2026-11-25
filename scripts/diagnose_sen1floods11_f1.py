"""Diagnose why per-chip mean F1 on Sen1Floods11 is ~0.6 (Prithvi and U-Net
alike), which is low for a model fine-tuned on this exact dataset.

Two hypotheses this tests from the existing caches (CPU only, no model rerun):
  1. Metric effect: published Sen1Floods11 numbers are pooled over all pixels,
     while our per_chip_f1 averages one F1 per chip -- which collapses on
     chips with only a sliver of water. If so, pooled F1 should be much higher
     than per-chip mean, and per-chip F1 should be low mainly on low-water chips.
  2. Pipeline mismatch: our val split is non-African chips that Prithvi very
     likely saw during its own fine-tuning. If our input pipeline matches its
     training setup, pooled F1 on val should be high (~0.9). If val is also
     low, something in our preprocessing/inference is still off.
Also compares any extra cache dirs passed in (e.g. a re-extraction with
location coordinates disabled -- see --no-location-coords in
scripts/extract_sen1floods11_cache.py).

Run on the cluster (CPU-only):
  apptainer exec --bind $PWD:/workspace --bind $PWD/outputs:/outputs \\
      container/eouq.sif python -m scripts.diagnose_sen1floods11_f1
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

PROB_THRESHOLD = 0.5
WATER_FRACTION_BINS = [(0.0, 0.01, "<1% water"), (0.01, 0.10, "1-10% water"), (0.10, 1.01, ">10% water")]


def pixel_counts(event_prob: np.ndarray, masks: np.ndarray, threshold: float = PROB_THRESHOLD) -> tuple:
    """event_prob, masks: (n, H, W). Returns per-chip (tp, fp, fn, n_water, n_valid) arrays,
    ignoring missing-data pixels (mask == -1)."""
    pred = event_prob >= threshold
    valid = masks != -1
    true = masks == 1
    tp = np.sum(pred & true & valid, axis=(1, 2))
    fp = np.sum(pred & ~true & valid, axis=(1, 2))
    fn = np.sum(~pred & true & valid, axis=(1, 2))
    n_water = np.sum(true & valid, axis=(1, 2))
    n_valid = np.sum(valid, axis=(1, 2))
    return tp, fp, fn, n_water, n_valid


def f1_from_counts(tp, fp, fn) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else float("nan")


def iou_from_counts(tp, fp, fn) -> float:
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else float("nan")


def summarize_split(event_prob: np.ndarray, masks: np.ndarray) -> dict:
    tp, fp, fn, n_water, n_valid = pixel_counts(event_prob, masks)
    per_chip_f1 = np.array([f1_from_counts(a, b, c) for a, b, c in zip(tp, fp, fn)])
    water_fraction = np.where(n_valid > 0, n_water / np.maximum(n_valid, 1), 0.0)

    by_water = {}
    for low, high, label in WATER_FRACTION_BINS:
        in_bin = (water_fraction >= low) & (water_fraction < high)
        scores = per_chip_f1[in_bin]
        by_water[label] = {
            "n_chips": int(in_bin.sum()),
            "n_chips_with_defined_f1": int(np.sum(~np.isnan(scores))),
            "mean_per_chip_f1": float(np.nanmean(scores)) if np.any(~np.isnan(scores)) else None,
        }

    return {
        "n_chips": int(len(masks)),
        "per_chip_mean_f1": float(np.nanmean(per_chip_f1)),
        "pooled_f1": f1_from_counts(tp.sum(), fp.sum(), fn.sum()),
        "pooled_iou": iou_from_counts(tp.sum(), fp.sum(), fn.sum()),
        "pooled_water_fraction": float(n_water.sum() / max(n_valid.sum(), 1)),
        "per_chip_f1_by_water_fraction": by_water,
    }


def summarize_cache(cache_dir: str) -> dict:
    results = {}
    for split in ("train", "val", "test"):
        path = os.path.join(cache_dir, f"{split}.npz")
        if not os.path.exists(path):
            continue
        with np.load(path, allow_pickle=True) as data:
            event_prob = data["probs"].mean(axis=0)[:, -1]  # (n, H, W), ensemble/TTA mean of the water class
            masks = data["mask"]
        results[split] = summarize_split(event_prob, masks)
        del event_prob, masks
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dirs",
        nargs="+",
        default=[
            "/outputs/sen1floods11_cache_prithvi",
            "/outputs/sen1floods11_cache_prithvi_nocoords",
            "/outputs/sen1floods11_cache_unet",
        ],
    )
    parser.add_argument("--output", default="/outputs/sen1floods11_f1_diagnosis.json")
    args = parser.parse_args()

    report = {}
    for cache_dir in args.cache_dirs:
        if not os.path.exists(os.path.join(cache_dir, "test.npz")):
            print(f"skipping {cache_dir}: no cache")
            continue
        report[os.path.basename(cache_dir)] = summarize_cache(cache_dir)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
