"""Download the Sen1Floods11 hand-labeled S2 + label chips directly over
HTTPS from the public GCS bucket, for environments without gsutil/gcloud
installed. Standard library only (urllib + json) -- run with a bare
`python3` on the login node, outside the Apptainer container (this needs
internet access and nothing else; no need for torch/numpy/rasterio).

Confirmed via the bucket's public JSON listing API (2026-11) that the real
v1.1 layout nests each layer in its own subdirectory, NOT flat files at
the bucket root as an early guess (from a model-card example filename
alone) assumed:
    v1.1/data/flood_events/HandLabeled/S2Hand/{chip}_S2Hand.tif
    v1.1/data/flood_events/HandLabeled/LabelHand/{chip}_LabelHand.tif
Only these two layers are downloaded (not S1Hand/JRCWaterHand/
S1OtsuLabelHand) since that's all eouq's Sen1Floods11 pipeline needs --
smaller than the full ~14GB bucket. Downloads into the same
S2Hand/LabelHand subdirectory structure `data/sen1floods11.py` expects.

Usage:
  python3 scripts/download_sen1floods11.py --output-dir data/sen1floods11
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

BUCKET_HTTP_BASE = "https://storage.googleapis.com/sen1floods11"
LISTING_API = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"
HAND_LABELED_PREFIX = "v1.1/data/flood_events/HandLabeled"
LAYERS = ["S2Hand", "LabelHand"]


def list_objects(prefix: str) -> list:
    names = []
    page_token = None
    while True:
        url = f"{LISTING_API}?prefix={prefix}&maxResults=1000"
        if page_token:
            url += f"&pageToken={page_token}"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
        names.extend(item["name"] for item in data.get("items", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return names


def download_object(object_name: str, output_dir: str) -> None:
    relative_path = os.path.relpath(object_name, HAND_LABELED_PREFIX)
    local_path = os.path.join(output_dir, relative_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return  # resumable: skip files already downloaded
    url = f"{BUCKET_HTTP_BASE}/{object_name}"
    urllib.request.urlretrieve(url, local_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/sen1floods11")
    args = parser.parse_args()

    for layer in LAYERS:
        prefix = f"{HAND_LABELED_PREFIX}/{layer}/"
        objects = list_objects(prefix)
        print(f"{layer}: {len(objects)} files")
        for i, name in enumerate(objects):
            download_object(name, args.output_dir)
            if (i + 1) % 50 == 0 or (i + 1) == len(objects):
                print(f"  {i + 1}/{len(objects)}")

    print("done")


if __name__ == "__main__":
    main()
