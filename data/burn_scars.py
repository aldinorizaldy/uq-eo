"""HLS Burn Scars dataset loader (ibm-nasa-geospatial/hls_burn_scars on HuggingFace).

804 chips of 512x512, 6 HLS bands (Blue/Green/Red/NIR/SWIR1/SWIR2) + a 1-band
mask (1=burn, 0=unburned, -1=missing). The public dataset only ships a 2-way
train(2/3)/validation(1/3) split; SHRUG-FM reports a 485/54/263 train/val/test
split instead, so this module does its own fixed-seed resplit of all chips
(see NOTES.md). Expects the dataset already downloaded to `root` with the
HF layout: `{root}/training/*_merged.tif` + `*.mask.tif`,
`{root}/validation/*_merged.tif` + `*.mask.tif`.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import rasterio
from torch.utils.data import Dataset

# Matches SHRUG-FM's reported split; sums to 802 of the 804 published chips
# (2 presumably excluded by the paper — not specified which, so our resplit
# just uses whichever 802 end up in these buckets after a fixed-seed shuffle).
SPLIT_SIZES = {"train": 485, "val": 54, "test": 263}
RESPLIT_SEED = 0


def _list_chip_ids(root: str) -> list[tuple[str, str]]:
    """All (subdir, chip_id) pairs across both HF-provided dirs, sorted for determinism."""
    chip_ids = []
    for subdir in ("training", "validation"):
        for path in sorted(glob.glob(os.path.join(root, subdir, "*_merged.tif"))):
            chip_id = os.path.basename(path)[: -len("_merged.tif")]
            chip_ids.append((subdir, chip_id))
    return sorted(chip_ids)


def split_chip_ids(chip_ids: list, seed: int = RESPLIT_SEED) -> dict:
    """Fixed-seed 485/54/263 resplit of a list of chip ids, ignoring whatever
    train/validation division they came with. Deterministic given the same
    `chip_ids` ordering and `seed`.
    """
    n_total = len(chip_ids)
    n_train, n_val, n_test = SPLIT_SIZES["train"], SPLIT_SIZES["val"], SPLIT_SIZES["test"]
    if n_train + n_val + n_test > n_total:
        raise ValueError(
            f"Got {n_total} chip ids, need at least {n_train + n_val + n_test} "
            f"for the {n_train}/{n_val}/{n_test} split (804 in the published dataset)."
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_total)
    shuffled = [chip_ids[i] for i in order]
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }


def resplit(root: str, seed: int = RESPLIT_SEED) -> dict:
    """Convenience wrapper: list chips under `root` then apply `split_chip_ids`."""
    return split_chip_ids(_list_chip_ids(root), seed=seed)


def _chip_center_lonlat(src) -> tuple:
    """Centre-pixel coordinate of an open rasterio dataset, reprojected to WGS84 lon/lat."""
    from rasterio.warp import transform as warp_transform

    row, col = src.height // 2, src.width // 2
    x, y = src.xy(row, col)
    lon, lat = warp_transform(src.crs, "EPSG:4326", [x], [y])
    return float(lon[0]), float(lat[0])


class HLSBurnScarsDataset(Dataset):
    """chip_ids: list of (subdir, chip_id) pairs, e.g. one value of a `resplit()` dict."""

    def __init__(self, root: str, chip_ids: list, transform=None):
        self.root = root
        self.chip_ids = chip_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.chip_ids)

    def __getitem__(self, idx: int) -> dict:
        subdir, chip_id = self.chip_ids[idx]
        scene_path = os.path.join(self.root, subdir, f"{chip_id}_merged.tif")
        mask_path = os.path.join(self.root, subdir, f"{chip_id}.mask.tif")

        with rasterio.open(scene_path) as src:
            image = src.read().astype(np.float32)  # (6, H, W)
            lon, lat = _chip_center_lonlat(src)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)  # (H, W), values in {-1, 0, 1}

        sample = {
            "chip_id": chip_id,
            "image": image,
            "mask": mask,
            "coords": np.array([lat, lon], dtype=np.float32),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
