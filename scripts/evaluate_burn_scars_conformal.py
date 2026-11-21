"""L3 conformal prediction on the burn-scars cache: split conformal, Mondrian
conformal (per geographic quadrant), conformal risk control on pixel FNR, and
a "conformal fusion" variant that normalises the nonconformity score by a
SHRUG-FM-style difficulty signal (predictive entropy). Reports coverage
overall/per-region, coverage disparity, and a selective-risk AURC on the same
difficulty signal, for comparison with the Milestone 2 report.

Design choices not from the paper (also logged in NOTES.md):
  - Calibration/evaluation split: the 263-chip burn-scars TEST split (never
    used to train the decoder ensemble or the Milestone 2 fusion tree) is
    itself split 50/50, fixed seed, into a conformal CALIBRATION half and a
    final EVALUATION half. train/val are untouched -- this keeps calibration
    exchangeable with the held-out evaluation data (standard split-conformal
    practice) rather than reusing val, which the Milestone 2 fusion tree
    already "saw" during fitting.
  - Region definition: four geographic quadrants (N/S x E/W of the median
    chip latitude/longitude), not a fixed-degree grid -- guarantees
    reasonably balanced region sizes regardless of how HLS Burn Scars chips
    happen to be distributed across the continental US.
  - Pixel sampling: a random subset of valid (non-missing-data) pixels per
    chip for the classification-coverage checks, not every pixel, to keep
    this CPU-only script's memory/compute modest. Per-chip FNR (used by
    conformal risk control) uses every valid pixel, not a sample, since FNR
    needs accurate true-positive/false-negative counts.
  - "Conformal fusion" difficulty signal: predictive entropy (roi_entropy),
    the strongest single signal from the Milestone 2 report, not the fitted
    fusion tree itself (Milestone 2's script doesn't persist the tree object
    to disk, only JSON metrics).

Not runnable in this sandbox -- run on the cluster (CPU-only, no GPU needed):
  apptainer exec --bind $PWD:/workspace --bind $PWD/outputs:/outputs \\
      container/eouq.sif python -m scripts.evaluate_burn_scars_conformal
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from eouq.layers.l2_prediction import predictive_entropy, roi_average
from eouq.layers.l3_guarantee import (
    conformal_risk_control_threshold,
    coverage,
    coverage_disparity,
    mondrian_conformal_thresholds,
    split_conformal_threshold,
)
from eouq.metrics.selective_risk import aurc, risk_coverage_curve

ALPHA_COVERAGE = 0.1  # target 90% coverage for split/Mondrian conformal classification
ALPHA_FNR = 0.2  # target pixel false-negative-rate bound for conformal risk control
N_PIXELS_PER_CHIP = 50
ROI_PROB_THRESHOLD = 0.5
LAMBDA_GRID = np.linspace(0.9, 0.1, 17)  # descending, for conformal risk control
TAU = 0.6


def load_cache(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def subset_cache(cache: dict, idx: np.ndarray) -> dict:
    """Index every cached array along the sample axis. `probs` is the one
    exception: its shape is (n_members, n_samples, ...), sample axis 1."""
    result = {}
    for key, value in cache.items():
        result[key] = value[:, idx] if key == "probs" else value[idx]
    return result


def chip_mean_probs(probs: np.ndarray) -> np.ndarray:
    """probs: (n_members, n_samples, n_classes, H, W) -> (n_samples, n_classes, H, W)."""
    return probs.mean(axis=0)


def quadrant_region_labels(coords: np.ndarray) -> np.ndarray:
    """coords: (n, 2) [lat, lon]. Four geographic quadrants around the median
    chip location, guaranteeing reasonably balanced region sizes."""
    median_lat, median_lon = np.median(coords[:, 0]), np.median(coords[:, 1])
    lat_label = np.where(coords[:, 0] >= median_lat, "N", "S")
    lon_label = np.where(coords[:, 1] >= median_lon, "E", "W")
    return np.array([f"{a}{b}" for a, b in zip(lat_label, lon_label)])


def sample_pixel_nonconformity(mean_probs: np.ndarray, masks: np.ndarray, n_per_chip: int, seed: int):
    """mean_probs: (n_samples, n_classes, H, W); masks: (n_samples, H, W), -1=missing.
    Returns (scores, chip_idx) for a random subsample of valid pixels per chip:
    score = 1 - p_model(true_class | pixel), the standard nonconformity score
    for conformal classification."""
    rng = np.random.default_rng(seed)
    n_samples = mean_probs.shape[0]
    scores, chip_idx = [], []
    for i in range(n_samples):
        valid_rows, valid_cols = np.where(masks[i] != -1)
        if len(valid_rows) == 0:
            continue
        n_pick = min(n_per_chip, len(valid_rows))
        pick = rng.choice(len(valid_rows), size=n_pick, replace=False)
        rows, cols = valid_rows[pick], valid_cols[pick]
        true_labels = masks[i, rows, cols]
        p_true = mean_probs[i, true_labels, rows, cols]
        scores.append(1.0 - p_true)
        chip_idx.extend([i] * n_pick)
    return np.concatenate(scores), np.array(chip_idx)


def per_chip_fnr(event_prob: np.ndarray, masks: np.ndarray, threshold: float) -> np.ndarray:
    """event_prob: (n_samples, H, W) prob of the positive/event class;
    masks: (n_samples, H, W), -1=missing. FNR = FN / (TP + FN) per chip,
    NaN if the chip has no true-positive pixels at all."""
    predicted = (event_prob >= threshold).astype(np.int64)
    n_samples = masks.shape[0]
    fnr = np.empty(n_samples)
    for i in range(n_samples):
        valid = masks[i] != -1
        pred, true = predicted[i][valid], masks[i][valid]
        tp = np.sum((pred == 1) & (true == 1))
        fn = np.sum((pred == 0) & (true == 1))
        fnr[i] = fn / (tp + fn) if (tp + fn) > 0 else np.nan
    return fnr


def chip_roi_entropy(mean_probs: np.ndarray) -> np.ndarray:
    """mean_probs: (n_samples, n_classes, H, W) -> per-chip predictive entropy
    averaged over the predicted event region (mean prob >= 0.5); NaN (empty
    ROI) mapped to 0."""
    n_samples = mean_probs.shape[0]
    result = np.empty(n_samples)
    for i in range(n_samples):
        chip_probs = np.moveaxis(mean_probs[i], 0, -1)  # (n_classes,H,W) -> (H,W,n_classes)
        pred_entropy = predictive_entropy(chip_probs)
        event_prob = mean_probs[i, -1, :, :]
        result[i] = roi_average(pred_entropy, event_prob, ROI_PROB_THRESHOLD)
    return np.nan_to_num(result, nan=0.0)


def run_conformal_evaluation(test: dict, seed: int = 0) -> dict:
    n = len(test["f1"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    half = n // 2

    # Regions must come from a single shared partition computed before the
    # calib/eval split, not independently per half -- otherwise "N"/"S"/"E"/"W"
    # would mean a different geographic threshold in calibration vs.
    # evaluation, breaking Mondrian conformal's requirement that calibration
    # and test share the same group definition.
    all_regions = quadrant_region_labels(test["coords"])
    calib_idx, eval_idx = order[:half], order[half:]
    calib, eval_ = subset_cache(test, calib_idx), subset_cache(test, eval_idx)
    calib_regions, eval_regions = all_regions[calib_idx], all_regions[eval_idx]

    calib_mean_probs, eval_mean_probs = chip_mean_probs(calib["probs"]), chip_mean_probs(eval_["probs"])

    # --- 1. Plain split / Mondrian conformal classification coverage ---
    calib_scores, calib_chip_idx = sample_pixel_nonconformity(calib_mean_probs, calib["mask"], N_PIXELS_PER_CHIP, seed)
    eval_scores, eval_chip_idx = sample_pixel_nonconformity(eval_mean_probs, eval_["mask"], N_PIXELS_PER_CHIP, seed + 1)
    calib_pixel_regions, eval_pixel_regions = calib_regions[calib_chip_idx], eval_regions[eval_chip_idx]

    split_threshold = split_conformal_threshold(calib_scores, ALPHA_COVERAGE)
    mondrian_thresholds = mondrian_conformal_thresholds(calib_scores, calib_pixel_regions, ALPHA_COVERAGE)

    split_covered = eval_scores <= split_threshold
    mondrian_covered = eval_scores <= np.array([mondrian_thresholds[r] for r in eval_pixel_regions])

    results = {"alpha_coverage_target": 1 - ALPHA_COVERAGE}
    results["split_conformal"] = {"threshold": split_threshold, "coverage": coverage(split_covered, eval_pixel_regions)}
    results["split_conformal"]["coverage_disparity"] = coverage_disparity(
        results["split_conformal"]["coverage"]["per_group"]
    )
    results["mondrian_conformal"] = {
        "thresholds_per_region": mondrian_thresholds,
        "coverage": coverage(mondrian_covered, eval_pixel_regions),
    }
    results["mondrian_conformal"]["coverage_disparity"] = coverage_disparity(
        results["mondrian_conformal"]["coverage"]["per_group"]
    )

    # --- 2. Conformal-fusion variant: normalise nonconformity by predictive entropy ---
    calib_entropy_by_chip, eval_entropy_by_chip = chip_roi_entropy(calib_mean_probs), chip_roi_entropy(eval_mean_probs)
    calib_norm_scores = calib_scores / (1.0 + calib_entropy_by_chip[calib_chip_idx])
    eval_norm_scores = eval_scores / (1.0 + eval_entropy_by_chip[eval_chip_idx])

    norm_threshold = split_conformal_threshold(calib_norm_scores, ALPHA_COVERAGE)
    norm_covered = eval_norm_scores <= norm_threshold
    results["conformal_fusion"] = {"threshold": norm_threshold, "coverage": coverage(norm_covered, eval_pixel_regions)}
    results["conformal_fusion"]["coverage_disparity"] = coverage_disparity(
        results["conformal_fusion"]["coverage"]["per_group"]
    )

    # --- 3. Conformal risk control on pixel FNR ---
    event_prob_calib = calib_mean_probs[:, -1, :, :]
    has_positive_calib = np.array([(calib["mask"][i] == 1).any() for i in range(len(calib["mask"]))])

    def loss_fn(lam):
        return per_chip_fnr(event_prob_calib[has_positive_calib], calib["mask"][has_positive_calib], lam)

    risk_control_lambda = conformal_risk_control_threshold(loss_fn, ALPHA_FNR, LAMBDA_GRID)

    event_prob_eval = eval_mean_probs[:, -1, :, :]
    has_positive_eval = np.array([(eval_["mask"][i] == 1).any() for i in range(len(eval_["mask"]))])
    eval_fnr = per_chip_fnr(event_prob_eval[has_positive_eval], eval_["mask"][has_positive_eval], risk_control_lambda)

    results["conformal_risk_control"] = {
        "alpha_fnr_target": ALPHA_FNR,
        "selected_lambda": risk_control_lambda,
        "mean_fnr_on_eval": float(np.nanmean(eval_fnr)),
        "n_eval_chips_with_positives": int(has_positive_eval.sum()),
    }

    # --- 4. Selective-risk AURC on the eval half, using the same difficulty signal ---
    y_fail_eval = (eval_["f1"] < TAU).astype(float)
    valid_f1 = ~np.isnan(eval_["f1"])
    coverage_arr, risk_arr = risk_coverage_curve(eval_entropy_by_chip[valid_f1], y_fail_eval[valid_f1])
    results["selective_risk_aurc_on_eval_half"] = aurc(coverage_arr, risk_arr)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/outputs/burn_scars_cache")
    parser.add_argument("--output", default="/outputs/burn_scars_conformal_evaluation.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    test = load_cache(os.path.join(args.cache_dir, "test.npz"))
    results = run_conformal_evaluation(test, seed=args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
