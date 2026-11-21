"""L3 guarantee layer: what coverage or risk bound can we actually promise?

Split conformal prediction, Mondrian conformal prediction (per region/event),
localized conformal prediction (generalising GeoConformal, arXiv 2412.08661,
to geography + embedding distance), and conformal risk control (e.g. pixel
false negative rate <= alpha for flood/landslide mapping). Consumes cached
probabilities, optional embeddings, and optional coordinates/region labels —
model-agnostic by construction.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from eouq.core.uo import UncertaintyObject


def split_conformal_threshold(calibration_scores: np.ndarray, alpha: float) -> float:
    """Standard split conformal quantile of nonconformity scores at level 1 - alpha.

    Uses the standard finite-sample-corrected quantile level
    ceil((n+1)(1-alpha))/n (Vovk et al.), clamped to 1.0 for small n.
    """
    n = len(calibration_scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(calibration_scores, q_level, method="higher"))


def mondrian_conformal_thresholds(
    calibration_scores: np.ndarray, calibration_groups: np.ndarray, alpha: float
) -> dict:
    """One conformal threshold per group (e.g. per region or per event)."""
    return {
        group: split_conformal_threshold(calibration_scores[calibration_groups == group], alpha)
        for group in np.unique(calibration_groups)
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Smallest value v such that the weight-mass of {values <= v} is >= q."""
    order = np.argsort(values)
    values_sorted = values[order]
    cum_weights = np.cumsum(weights[order])
    cum_weights /= cum_weights[-1]
    idx = min(np.searchsorted(cum_weights, q, side="left"), len(values_sorted) - 1)
    return float(values_sorted[idx])


def localized_conformal_threshold(
    calibration_scores: np.ndarray,
    calibration_coords: np.ndarray,
    calibration_embeddings: np.ndarray,
    query_coords: np.ndarray,
    query_embeddings: np.ndarray,
    alpha: float,
    bandwidth_geo: float,
    bandwidth_embed: float,
) -> np.ndarray:
    """Per-query threshold from a kernel-weighted calibration set in geo + embedding space.

    Generalises GeoConformal (arXiv:2412.08661) by combining a geographic
    Gaussian kernel with an embedding-space Gaussian kernel: calibration
    points that are both nearby in space and similar in representation get
    the most weight when computing each query's local threshold.
    """
    n_query = len(query_coords)
    thresholds = np.empty(n_query)
    for i in range(n_query):
        geo_dist = np.linalg.norm(calibration_coords - query_coords[i], axis=1)
        embed_dist = np.linalg.norm(calibration_embeddings - query_embeddings[i], axis=1)
        geo_weight = np.exp(-0.5 * (geo_dist / bandwidth_geo) ** 2)
        embed_weight = np.exp(-0.5 * (embed_dist / bandwidth_embed) ** 2)
        weights = geo_weight * embed_weight
        if weights.sum() <= 0:
            # No calibration point has non-negligible weight (query is far from
            # everything, in both spaces) -- fall back to a uniform weighting,
            # equivalent to standard (non-localized) split conformal.
            weights = np.ones_like(weights)
        thresholds[i] = _weighted_quantile(calibration_scores, weights, q=1 - alpha)
    return thresholds


def conformal_risk_control_threshold(
    loss_fn: Callable[[float], np.ndarray],
    alpha: float,
    lambda_grid: np.ndarray,
    loss_upper_bound: float = 1.0,
) -> float:
    """Largest lambda in `lambda_grid` such that the finite-sample-corrected
    calibration risk meets E[loss(lambda)] <= alpha, per Angelopoulos et al.
    (2022) "Conformal Risk Control".

    `loss_fn(lambda)` must return the per-calibration-sample loss (e.g. pixel
    false-negative rate) at that decision threshold, and must be monotone
    non-increasing as lambda decreases (true for FNR vs. an event-probability
    threshold: a lower threshold flags more pixels positive, so fewer false
    negatives). `lambda_grid` should be sorted descending (most permissive /
    largest first) so the first value found satisfying the bound is the
    least conservative valid choice.
    """
    n = None
    for lam in lambda_grid:
        losses = np.asarray(loss_fn(lam))
        if n is None:
            n = len(losses)
        risk_hat = losses.mean()
        corrected_risk = (n * risk_hat + loss_upper_bound) / (n + 1)
        if corrected_risk <= alpha:
            return float(lam)
    return float(lambda_grid[-1])  # nothing satisfied the bound; return the most conservative candidate


def coverage(predictions_contain_truth: np.ndarray, groups: np.ndarray = None) -> dict:
    """Empirical coverage overall and, if groups is given, per group."""
    result = {"overall": float(np.mean(predictions_contain_truth))}
    if groups is not None:
        result["per_group"] = {
            group: float(np.mean(predictions_contain_truth[groups == group])) for group in np.unique(groups)
        }
    return result


def coverage_disparity(per_group_coverage: dict) -> float:
    """Spread (max - min) of per-group coverage, the headline spatial-fairness metric."""
    values = list(per_group_coverage.values())
    return float(max(values) - min(values))


def guarantee_object(method: str, alpha: float, extra: dict, provenance: dict, scale: str = "aggregate") -> UncertaintyObject:
    return UncertaintyObject(
        value=extra,
        type="coverage",
        source=f"l3.{method}",
        scale=scale,
        guarantee={"method": method, "alpha": alpha},
        provenance=provenance,
    )
