"""Conformal prediction under geographic distribution shift, on Sen1Floods11.

Question: if split conformal prediction and conformal risk control are
calibrated on chips from OTHER continents, do their guarantees still hold on
a continent the models never saw (Africa, the held-out test set)? Compared
against an in-distribution baseline calibrated on half of the African chips
themselves -- both evaluated on the SAME other half, so the only difference
is where the calibration data came from.

Per model cache (Prithvi, U-Net):
  - "other continents" calibration = the val split (non-African chips, never
    used for U-Net training or our fusion here).
  - "in distribution" calibration  = a fixed-seed 50% of the African test chips.
  - evaluation = the remaining 50% of the African test chips.
Coverage is reported overall and per flood event (Ghana / Nigeria / Somalia,
read from the chip id prefix), plus event-level coverage disparity.
Conformal risk control (pixel FNR <= alpha) is compared the same way.

Only val + test caches are loaded (Prithvi's train cache alone is >4 GB).
Pixel-level nonconformity uses the same sampling and score as the burn-scars
conformal script, reused from there rather than reimplemented.

Run on the cluster (CPU-only):
  apptainer exec --bind $PWD:/workspace --bind $PWD/outputs:/outputs \\
      container/eouq.sif python -m scripts.evaluate_sen1floods11_conformal
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from eouq.layers.l3_guarantee import (
    conformal_risk_control_threshold,
    coverage,
    coverage_disparity,
    split_conformal_threshold,
)
from scripts.evaluate_burn_scars_conformal import (
    ALPHA_COVERAGE,
    ALPHA_FNR,
    LAMBDA_GRID,
    N_PIXELS_PER_CHIP,
    chip_mean_probs,
    load_cache,
    per_chip_fnr,
    sample_pixel_nonconformity,
    subset_cache,
)


def event_names(chip_ids: np.ndarray) -> np.ndarray:
    """'Ghana_103272' -> 'Ghana'; 'Sri-Lanka_956740' -> 'Sri-Lanka'."""
    return np.array([str(cid).rsplit("_", 1)[0] for cid in chip_ids])


def _risk_control_lambda(mean_probs: np.ndarray, masks: np.ndarray) -> float:
    event_prob = mean_probs[:, -1, :, :]
    has_positive = np.array([(m == 1).any() for m in masks])
    return conformal_risk_control_threshold(
        lambda lam: per_chip_fnr(event_prob[has_positive], masks[has_positive], lam), ALPHA_FNR, LAMBDA_GRID
    )


def _mean_fnr(mean_probs: np.ndarray, masks: np.ndarray, lam: float) -> float:
    event_prob = mean_probs[:, -1, :, :]
    has_positive = np.array([(m == 1).any() for m in masks])
    return float(np.nanmean(per_chip_fnr(event_prob[has_positive], masks[has_positive], lam)))


def run_shift_evaluation(val: dict, test: dict, seed: int = 0) -> dict:
    n_test = len(test["mask"])
    order = np.random.default_rng(seed).permutation(n_test)
    half = n_test // 2
    in_calib, eval_ = subset_cache(test, order[:half]), subset_cache(test, order[half:])

    val_mp, in_mp, eval_mp = (chip_mean_probs(d["probs"]) for d in (val, in_calib, eval_))

    val_scores, _ = sample_pixel_nonconformity(val_mp, val["mask"], N_PIXELS_PER_CHIP, seed)
    in_scores, _ = sample_pixel_nonconformity(in_mp, in_calib["mask"], N_PIXELS_PER_CHIP, seed + 1)
    eval_scores, eval_chip_idx = sample_pixel_nonconformity(eval_mp, eval_["mask"], N_PIXELS_PER_CHIP, seed + 2)
    eval_events = event_names(eval_["chip_id"])[eval_chip_idx]

    results = {
        "alpha_coverage_target": 1 - ALPHA_COVERAGE,
        "n_val_chips": int(len(val["mask"])),
        "n_in_distribution_calib_chips": int(half),
        "n_eval_chips": int(n_test - half),
    }
    for name, calib_scores in (("calibrated_on_other_continents", val_scores), ("calibrated_in_distribution", in_scores)):
        threshold = split_conformal_threshold(calib_scores, ALPHA_COVERAGE)
        cov = coverage(eval_scores <= threshold, eval_events)
        results[name] = {
            "threshold": threshold,
            "coverage": cov,
            "event_coverage_disparity": coverage_disparity(cov["per_group"]),
        }

    lam_other = _risk_control_lambda(val_mp, val["mask"])
    lam_in = _risk_control_lambda(in_mp, in_calib["mask"])
    results["conformal_risk_control"] = {
        "alpha_fnr_target": ALPHA_FNR,
        "calibrated_on_other_continents": {"lambda": lam_other, "mean_fnr_on_eval": _mean_fnr(eval_mp, eval_["mask"], lam_other)},
        "calibrated_in_distribution": {"lambda": lam_in, "mean_fnr_on_eval": _mean_fnr(eval_mp, eval_["mask"], lam_in)},
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/outputs/sen1floods11_conformal_evaluation.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    all_results = {}
    for model_name in ("prithvi", "unet"):
        cache_dir = f"/outputs/sen1floods11_cache_{model_name}"
        if not os.path.exists(os.path.join(cache_dir, "test.npz")):
            print(f"skipping {model_name}: no cache at {cache_dir}")
            continue
        val = load_cache(os.path.join(cache_dir, "val.npz"))
        test = load_cache(os.path.join(cache_dir, "test.npz"))
        all_results[model_name] = run_shift_evaluation(val, test, seed=args.seed)
        del val, test

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
