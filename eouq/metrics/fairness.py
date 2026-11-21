"""Spatial-fairness metrics: coverage per group (region) and coverage disparity,
plus calibration error (ECE), reused across L3 guarantee evaluation."""
from __future__ import annotations

import numpy as np


def coverage_per_group(predictions_contain_truth: np.ndarray, groups: np.ndarray) -> dict:
    return {group: float(np.mean(predictions_contain_truth[groups == group])) for group in np.unique(groups)}


def coverage_disparity(per_group_coverage: dict, method: str = "range") -> float:
    """method: 'range' (max - min) or 'std' across groups."""
    values = np.array(list(per_group_coverage.values()), dtype=float)
    if method == "range":
        return float(values.max() - values.min())
    if method == "std":
        return float(values.std())
    raise ValueError(f"method must be 'range' or 'std', got {method!r}")


def expected_calibration_error(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 15) -> float:
    """Standard ECE: weighted average, over equal-width confidence bins, of
    |mean confidence - mean correctness| within each bin."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidences, bin_edges[1:-1], right=True), 0, n_bins - 1)

    n = len(confidences)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        bin_confidence = confidences[mask].mean()
        bin_accuracy = correctness[mask].mean()
        ece += (mask.sum() / n) * abs(bin_confidence - bin_accuracy)
    return float(ece)
