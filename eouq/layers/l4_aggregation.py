"""L4 aggregation layer: how uncertain is the area total (per event, per
district/region), not just the per-chip prediction?

Two methods, per the original design doc: spatial block bootstrap (resample
whole blocks/regions, not individual chips, to respect within-block spatial
autocorrelation) and conformal prediction on the aggregated per-region/event
total. Consumes per-chip predicted areas and a region/event label per chip
-- model-agnostic, same as every other layer.
"""
from __future__ import annotations

import numpy as np

from eouq.core.uo import UncertaintyObject
from eouq.layers.l3_guarantee import split_conformal_threshold


def per_group_total(values: np.ndarray, groups: np.ndarray) -> dict:
    """Sum `values` within each group (e.g. predicted burned/flooded pixel
    count per chip, summed per region or per event)."""
    return {group: float(values[groups == group].sum()) for group in np.unique(groups)}


def spatial_block_bootstrap_interval(
    values: np.ndarray, groups: np.ndarray, alpha: float, n_bootstrap: int = 1000, seed: int = 0
) -> dict:
    """Confidence interval on the grand total (sum of `values` across all
    groups) via block bootstrap: resample whole groups (blocks) with
    replacement, not individual values, so within-block spatial
    autocorrelation is preserved rather than washed out by i.i.d. resampling.
    Returns {"point_estimate", "lower", "upper"} at level 1 - alpha.
    """
    unique_groups = np.unique(groups)
    group_totals = np.array([values[groups == g].sum() for g in unique_groups])
    n_groups = len(unique_groups)

    rng = np.random.default_rng(seed)
    bootstrap_totals = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        resample_idx = rng.integers(0, n_groups, size=n_groups)
        bootstrap_totals[b] = group_totals[resample_idx].sum()

    lower = np.quantile(bootstrap_totals, alpha / 2)
    upper = np.quantile(bootstrap_totals, 1 - alpha / 2)
    return {"point_estimate": float(group_totals.sum()), "lower": float(lower), "upper": float(upper)}


def conformal_total_interval(
    calibration_predicted_totals: np.ndarray,
    calibration_true_totals: np.ndarray,
    query_predicted_total: float,
    alpha: float,
) -> dict:
    """Conformal prediction interval on a new region/event's true total,
    centred on its predicted total, using the calibration set's realised
    |predicted - true| errors as the nonconformity score (standard
    symmetric split-conformal interval for a real-valued target). Reuses
    `split_conformal_threshold` from L3 -- same finite-sample-corrected
    quantile, just on totals instead of classification scores.
    """
    residuals = np.abs(calibration_predicted_totals - calibration_true_totals)
    half_width = split_conformal_threshold(residuals, alpha)
    return {
        "point_estimate": float(query_predicted_total),
        "lower": float(query_predicted_total - half_width),
        "upper": float(query_predicted_total + half_width),
        "half_width": half_width,
    }


def aggregation_object(method: str, alpha: float, extra: dict, provenance: dict, scale: str = "region") -> UncertaintyObject:
    return UncertaintyObject(
        value=extra,
        type="coverage",
        source=f"l4.{method}",
        scale=scale,
        guarantee={"method": method, "alpha": alpha},
        provenance=provenance,
    )
