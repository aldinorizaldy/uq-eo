"""Build L5 Uncertainty Cards from the evaluation JSONs already on disk -- no
recomputation. One card per experiment, merging that experiment's selective-
risk and conformal results:
  - burn_scars:            burn_scars_evaluation.json + burn_scars_conformal_evaluation.json
  - sen1floods11_prithvi:  sen1floods11_evaluation.json["prithvi"] + sen1floods11_conformal_evaluation.json["prithvi"]
  - sen1floods11_unet:     same, ["unet"]
Missing inputs are skipped; a card is written from whatever exists.

Markdown cards go to outputs/report/uncertainty_card_<name>.md (tracked in
git, like the main report); JSON cards to outputs/uncertainty_cards/.

Run on the cluster (CPU, seconds):
  apptainer exec --bind $PWD:/workspace --bind $PWD/outputs:/outputs \\
      container/eouq.sif python -m scripts.build_uncertainty_cards
"""
from __future__ import annotations

import argparse
import json
import os

from eouq.report.uncertainty_card import build_card, render_markdown

EXPERIMENTS = {
    "burn_scars": {
        "title": "Uncertainty Card — Burn scars (SSL4EO-S12 ViT-S/16, 5-member bootstrap ensemble)",
        "sources": [("burn_scars_evaluation.json", None), ("burn_scars_conformal_evaluation.json", None)],
    },
    "sen1floods11_prithvi": {
        "title": "Uncertainty Card — Flood mapping, Prithvi-EO-2.0 (frozen, test-time augmentation), Africa held out",
        "sources": [("sen1floods11_evaluation.json", "prithvi"), ("sen1floods11_conformal_evaluation.json", "prithvi")],
    },
    "sen1floods11_unet": {
        "title": "Uncertainty Card — Flood mapping, U-Net (from scratch, 5-member ensemble), Africa held out",
        "sources": [("sen1floods11_evaluation.json", "unet"), ("sen1floods11_conformal_evaluation.json", "unet")],
    },
}


def merge_sources(outputs_dir: str, sources: list) -> dict:
    merged = {}
    for filename, key in sources:
        path = os.path.join(outputs_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        part = data.get(key, {}) if key else data
        merged.update(part)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="/outputs")
    args = parser.parse_args()

    md_dir = os.path.join(args.outputs_dir, "report")
    json_dir = os.path.join(args.outputs_dir, "uncertainty_cards")
    os.makedirs(md_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    for name, spec in EXPERIMENTS.items():
        merged = merge_sources(args.outputs_dir, spec["sources"])
        if not merged:
            print(f"skipping {name}: no result files found")
            continue
        card = build_card(merged)
        with open(os.path.join(json_dir, f"{name}.json"), "w") as f:
            json.dump(card, f, indent=2, default=str)
        md_path = os.path.join(md_dir, f"uncertainty_card_{name}.md")
        with open(md_path, "w") as f:
            f.write(render_markdown(card, title=spec["title"]) + "\n")
        print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
