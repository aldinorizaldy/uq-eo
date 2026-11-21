"""Sen1Floods11 dataset loader (gs://sen1floods11/, Cloud to Street).

Hand-labeled Sentinel-2 chips (512x512, 13-band TOA reflectance) + ground
truth flood masks (-1=missing, 0=not water, 1=water). We use the 6-band
subset (Blue/Green/Red/NIR-narrow/SWIR1/SWIR2) the Prithvi-EO-2.0 flood
checkpoint expects -- a true subset here, not the zero-fill approximation
burn scars needed, since the full 13-band product is actually available.
Raw values are divided by 10000 to rescale from the dataset's native TOA
reflectance range to [0, 1], matching the scaling the Prithvi checkpoint's
own reference inference script applies -- confirmed by reading that
script directly (see adapters/prithvi.py), not previously done here.

`root` is expected to contain the two layer subdirectories as laid out in
the real bucket (confirmed via the GCS JSON listing API, v1.1 release --
NOT flat `{chip}_S2Hand.tif` files directly under `root` as an earlier
version of this module incorrectly assumed from a model-card example
filename alone):
    {root}/S2Hand/{chip_id}_S2Hand.tif
    {root}/LabelHand/{chip_id}_LabelHand.tif
See scripts/download_sen1floods11.py for a gsutil/gcloud-free downloader
that reproduces this exact layout.

Hold-out-continent split: one continent's chips become the entire test set,
the rest split into train/val. Continent is derived directly from each
chip's own geolocation (a simple lat/lon bounding-box classifier, not the
dataset's metadata geojson's country codes) -- self-contained, no need to
parse/match a separate metadata file. Approximate by construction; good
enough for a built-in distribution-shift check, not scientific-grade
continent attribution. See NOTES.md.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import rasterio
from torch.utils.data import Dataset

# Index of each of the 6 bands the model expects within the 13-band S2
# ordering [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12]: Blue=B2, Green=B3,
# Red=B4, NIR-narrow=B8A, SWIR1=B11, SWIR2=B12.
S2_BAND_INDEX = [1, 2, 3, 8, 11, 12]

# Ordered, first-match-wins approximate bounding boxes (lat_min, lat_max,
# lon_min, lon_max). Oceania checked before Asia since their lon ranges
# overlap; any chip matching neither box of an overlap goes to whichever is
# listed first, which is a known source of imprecision at the Asia/Oceania
# boundary. Not calibrated against the real dataset's actual event list.
CONTINENT_BOUNDING_BOXES = [
    ("Oceania", -50.0, 0.0, 110.0, 180.0),
    ("Asia", -10.0, 77.0, 26.0, 180.0),
    ("Europe", 36.0, 71.0, -25.0, 40.0),
    ("Africa", -35.0, 38.0, -18.0, 52.0),
    ("North America", 7.0, 83.0, -170.0, -50.0),
    ("South America", -56.0, 13.0, -82.0, -34.0),
]


def coords_to_continent(lat: float, lon: float) -> str:
    for name, lat_min, lat_max, lon_min, lon_max in CONTINENT_BOUNDING_BOXES:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return "Unknown"


def split_chip_ids_by_continent(chip_coords: list, holdout_continent: str, val_fraction: float = 0.1, seed: int = 0) -> dict:
    """chip_coords: list of (chip_id, lat, lon) tuples. Returns {"train": [...], "val": [...], "test": [...]}
    (chip ids), where "test" is every chip in `holdout_continent` and the rest split train/val."""
    continents = [coords_to_continent(lat, lon) for _, lat, lon in chip_coords]
    test_ids = [cid for (cid, _, _), c in zip(chip_coords, continents) if c == holdout_continent]
    remaining_ids = [cid for (cid, _, _), c in zip(chip_coords, continents) if c != holdout_continent]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(remaining_ids))
    shuffled = [remaining_ids[i] for i in order]
    n_val = int(len(shuffled) * val_fraction)
    return {"train": shuffled[n_val:], "val": shuffled[:n_val], "test": test_ids}


def _list_chip_ids(root: str) -> list:
    """Chip ids with both an S2Hand and a LabelHand file present."""
    chip_ids = []
    for path in sorted(glob.glob(os.path.join(root, "S2Hand", "*_S2Hand.tif"))):
        chip_id = os.path.basename(path)[: -len("_S2Hand.tif")]
        if os.path.exists(os.path.join(root, "LabelHand", f"{chip_id}_LabelHand.tif")):
            chip_ids.append(chip_id)
    return sorted(chip_ids)


def _chip_center_lonlat(src) -> tuple:
    from rasterio.warp import transform as warp_transform

    row, col = src.height // 2, src.width // 2
    x, y = src.xy(row, col)
    lon, lat = warp_transform(src.crs, "EPSG:4326", [x], [y])
    return float(lon[0]), float(lat[0])


def holdout_continent_split(root: str, holdout_continent: str = "Africa", val_fraction: float = 0.1, seed: int = 0) -> dict:
    """Convenience wrapper: list chips under `root`, read each one's centroid
    coordinate, then apply `split_chip_ids_by_continent`."""
    chip_coords = []
    for chip_id in _list_chip_ids(root):
        with rasterio.open(os.path.join(root, "S2Hand", f"{chip_id}_S2Hand.tif")) as src:
            lon, lat = _chip_center_lonlat(src)
        chip_coords.append((chip_id, lat, lon))
    return split_chip_ids_by_continent(chip_coords, holdout_continent, val_fraction, seed)


class Sen1Floods11Dataset(Dataset):
    def __init__(self, root: str, chip_ids: list, transform=None):
        self.root = root
        self.chip_ids = chip_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.chip_ids)

    def __getitem__(self, idx: int) -> dict:
        chip_id = self.chip_ids[idx]
        with rasterio.open(os.path.join(self.root, "S2Hand", f"{chip_id}_S2Hand.tif")) as src:
            full_image = src.read().astype(np.float32)  # (13, H, W), raw TOA reflectance, scaled by 10000
            image = full_image[S2_BAND_INDEX] / 10000.0  # (6, H, W), rescaled to [0, 1]
            lon, lat = _chip_center_lonlat(src)

        with rasterio.open(os.path.join(self.root, "LabelHand", f"{chip_id}_LabelHand.tif")) as src:
            mask = src.read(1).astype(np.int64)  # (H, W), {-1, 0, 1}

        sample = {
            "chip_id": chip_id,
            "image": image,
            "mask": mask,
            "coords": np.array([lat, lon], dtype=np.float32),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
