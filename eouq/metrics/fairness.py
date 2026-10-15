"""Spatial-fairness metrics: coverage per group (region) and coverage disparity,
plus calibration error (ECE), reused across L3 guarantee evaluation."""
from __future__ import annotations

import numpy as np


def coverage_per_group(predictions_contain_truth: np.ndarray, groups: np.ndarray) -> dict:
    raise NotImplementedError


def coverage_disparity(per_group_coverage: dict, method: str = "range") -> float:
    """method: 'range' (max - min) or 'std' across groups."""
    raise NotImplementedError


def expected_calibration_error(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 15) -> float:
    raise NotImplementedError
