"""L0 sensor-noise experiment, GPU stage: re-run a model on noise-perturbed
inputs and cache compact per-chip statistics (no full-resolution probs).

Two runs per model, both relative to that model's existing clean cache:
  1. Degradation sweep (test split): one noise draw at each level in
     SWEEP_STDS -> per-chip F1, pixel counts, U_T signals, embeddings,
     and prediction change vs the clean prediction. sigma = 0 is included
     as a reproducibility check (must match the clean cache).
  2. Sensitivity signal (all splits): N_SIGNAL_DRAWS independent draws at
     SIGNAL_STD -> per-chip mean prediction change and label-flip rate,
     averaged over draws. A candidate chip-level L0 uncertainty signal.

Chip ids and their order come from the clean cache, not from re-running the
split: the Sen1Floods11 continent classifier was fixed after those caches
were made (see data/sen1floods11.py), so recomputing the split now would
give a different test set.

Noise: additive Gaussian in reflectance units, per pixel and band, applied
at native resolution before any model preprocessing (see
eouq.layers.l0_input.add_reflectance_noise). Each chip's draw is seeded from
(seed, draw, chip id), so runs are reproducible and identical across models.

Not runnable in this sandbox -- run on the cluster, e.g.:
  python -m scripts.extract_sensor_noise --dataset sen1floods11 --model unet
"""
from __future__ import annotations

import argparse
import glob
import os
import zlib

import numpy as np
from torch.utils.data import DataLoader

from eouq.layers.l0_input import add_reflectance_noise, prediction_change
from scripts.diagnose_sen1floods11_f1 import pixel_counts
from scripts.evaluate_burn_scars import compute_l2_signals
from scripts.extract_burn_scars_cache import per_chip_f1

SWEEP_STDS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
SIGNAL_STD = 0.01
N_SIGNAL_DRAWS = 5
SPLITS = ("train", "val", "test")


class SensorNoiseTransform:
    """Dataset transform adding seeded reflectance noise to sample["image"]."""

    def __init__(self, noise_std: float, seed: int, draw: int, reflectance_scale: float):
        self.noise_std = noise_std
        self.seed = seed
        self.draw = draw
        self.reflectance_scale = reflectance_scale

    def __call__(self, sample: dict) -> dict:
        chip_key = zlib.crc32(str(sample["chip_id"]).encode())
        rng = np.random.default_rng([self.seed, self.draw, chip_key])
        sample["image"] = add_reflectance_noise(sample["image"], self.noise_std, rng, self.reflectance_scale)
        return sample


def infer_reflectance_scale(image: np.ndarray) -> float:
    """1.0 if the image is already [0, 1] reflectance, 10000.0 if it is scaled integers."""
    valid = image[np.isfinite(image) & (image >= 0)]
    return 10000.0 if valid.size and float(np.median(valid)) > 1.0 else 1.0


def build_dataset_factory(dataset_name: str, data_root: str):
    """Returns make_dataset(chip_ids_from_cache, transform)."""
    if dataset_name == "burn_scars":
        from data.burn_scars import HLSBurnScarsDataset, _list_chip_ids

        subdir_of = {chip_id: subdir for subdir, chip_id in _list_chip_ids(data_root)}
        return lambda chip_ids, transform: HLSBurnScarsDataset(
            data_root, [(subdir_of[str(c)], str(c)) for c in chip_ids], transform=transform
        )
    from data.sen1floods11 import Sen1Floods11Dataset

    return lambda chip_ids, transform: Sen1Floods11Dataset(data_root, [str(c) for c in chip_ids], transform=transform)


def build_adapter(dataset_name: str, model_name: str, checkpoint_dir: str, device: str):
    if dataset_name == "burn_scars":
        from adapters.ssl4eo_vit import SSL4EOViTAdapter

        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "decoder_member_*.pt")))
        if not checkpoints:
            raise FileNotFoundError(f"No decoder checkpoints in {checkpoint_dir}")
        return SSL4EOViTAdapter(backbone_checkpoint=None, decoder_checkpoints=checkpoints, device=device)
    from scripts.extract_sen1floods11_cache import build_adapter as build_flood_adapter

    return build_flood_adapter(model_name, checkpoint_dir, device)


def load_clean_reference(cache_path: str) -> dict:
    """Chip ids, masks and ensemble-mean event probability from a clean cache."""
    with np.load(cache_path, allow_pickle=True) as data:
        clean_event_prob = data["probs"].mean(axis=0)[:, -1].astype(np.float32)
        return {"chip_id": data["chip_id"], "mask": data["mask"], "event_prob": clean_event_prob, "f1": data["f1"]}


def run_noisy(adapter, make_dataset, chip_ids, transform, batch_size: int):
    dataloader = DataLoader(make_dataset(chip_ids, transform), batch_size=batch_size, shuffle=False)
    outputs = adapter.extract(dataloader)
    returned_ids = [str(c) for c in outputs.extra["chip_id"]]
    if returned_ids != [str(c) for c in chip_ids]:
        raise RuntimeError("noisy run returned chips in a different order from the clean cache")
    return outputs


def sweep_stats(outputs, clean: dict) -> dict:
    mean_probs = outputs.probs.mean(axis=0)
    event_prob = mean_probs[:, -1]
    masks = outputs.extra["mask"]
    tp, fp, fn, n_event, n_valid = pixel_counts(event_prob, masks)
    mean_abs_change, flip_rate = prediction_change(clean["event_prob"], event_prob, masks)
    stats = {
        "chip_id": clean["chip_id"],
        "f1": per_chip_f1(mean_probs, masks),
        "tp": tp, "fp": fp, "fn": fn, "n_event": n_event, "n_valid": n_valid,
        "mean_abs_change": mean_abs_change,
        "flip_rate": flip_rate,
        **compute_l2_signals(outputs.probs),
    }
    if outputs.embeddings is not None:
        stats["embeddings"] = outputs.embeddings
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["burn_scars", "sen1floods11"], required=True)
    parser.add_argument("--model", choices=["ssl4eo", "prithvi", "unet"], required=True)
    parser.add_argument("--data-root", default=None, help="defaults to /data/<dataset>")
    parser.add_argument("--clean-cache-dir", default=None, help="defaults to the model's existing clean cache")
    parser.add_argument("--checkpoint-dir", default=None, help="decoder / U-Net checkpoints (unused for Prithvi)")
    parser.add_argument("--output-dir", default=None, help="defaults to /outputs/sensor_noise_<dataset>_<model>")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if (args.dataset == "burn_scars") != (args.model == "ssl4eo"):
        parser.error("burn_scars runs with --model ssl4eo; sen1floods11 with --model prithvi or unet")

    data_root = args.data_root or f"/data/{args.dataset}"
    if args.dataset == "burn_scars":
        clean_dir = args.clean_cache_dir or "/outputs/burn_scars_cache"
        checkpoint_dir = args.checkpoint_dir or "/outputs/burn_scars_decoders"
        batch_size = 8
    else:
        clean_dir = args.clean_cache_dir or f"/outputs/sen1floods11_cache_{args.model}"
        checkpoint_dir = args.checkpoint_dir or "/outputs/sen1floods11_unet"
        batch_size = 4
    output_dir = args.output_dir or f"/outputs/sensor_noise_{args.dataset}_{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    make_dataset = build_dataset_factory(args.dataset, data_root)
    adapter = build_adapter(args.dataset, args.model, checkpoint_dir, args.device)

    clean_refs = {split: load_clean_reference(os.path.join(clean_dir, f"{split}.npz")) for split in SPLITS}
    first_image = make_dataset(clean_refs["test"]["chip_id"][:1], None)[0]["image"]
    reflectance_scale = infer_reflectance_scale(first_image)
    print(f"input reflectance scale: {reflectance_scale} (noise std multiplied by this)")

    # 1. Degradation sweep on the test split.
    clean_test = clean_refs["test"]
    for noise_std in SWEEP_STDS:
        transform = SensorNoiseTransform(noise_std, args.seed, draw=0, reflectance_scale=reflectance_scale)
        outputs = run_noisy(adapter, make_dataset, clean_test["chip_id"], transform, batch_size)
        stats = sweep_stats(outputs, clean_test)
        del outputs
        path = os.path.join(output_dir, f"sweep_std{noise_std:g}.npz")
        np.savez(path, noise_std=noise_std, reflectance_scale=reflectance_scale, **stats)
        pooled = 2 * stats["tp"].sum() / max(2 * stats["tp"].sum() + stats["fp"].sum() + stats["fn"].sum(), 1)
        print(
            f"sweep std={noise_std:g}: pooled F1={pooled:.3f}, mean |dp|={stats['mean_abs_change'].mean():.4f}, "
            f"flip rate={stats['flip_rate'].mean():.4f}"
        )
        if noise_std == 0.0:
            same = np.allclose(np.nan_to_num(stats["f1"], nan=-1), np.nan_to_num(clean_test["f1"], nan=-1), atol=1e-4)
            print(f"  reproducibility check vs clean cache: per-chip F1 {'MATCHES' if same else 'DIFFERS'}")

    # 2. Sensitivity signal on every split.
    for split in SPLITS:
        clean = clean_refs[split]
        changes, flips = [], []
        for draw in range(N_SIGNAL_DRAWS):
            transform = SensorNoiseTransform(SIGNAL_STD, args.seed, draw=draw + 1, reflectance_scale=reflectance_scale)
            outputs = run_noisy(adapter, make_dataset, clean["chip_id"], transform, batch_size)
            event_prob = outputs.probs.mean(axis=0)[:, -1]
            mean_abs_change, flip_rate = prediction_change(clean["event_prob"], event_prob, outputs.extra["mask"])
            del outputs, event_prob
            changes.append(mean_abs_change)
            flips.append(flip_rate)
        path = os.path.join(output_dir, f"sensitivity_{split}.npz")
        np.savez(
            path,
            noise_std=SIGNAL_STD,
            n_draws=N_SIGNAL_DRAWS,
            chip_id=clean["chip_id"],
            mean_abs_change=np.mean(changes, axis=0),
            flip_rate=np.mean(flips, axis=0),
            mean_abs_change_draw_std=np.std(changes, axis=0),
        )
        print(f"sensitivity {split}: {len(clean['chip_id'])} chips, mean |dp|={np.mean(changes):.4f}")


if __name__ == "__main__":
    main()
