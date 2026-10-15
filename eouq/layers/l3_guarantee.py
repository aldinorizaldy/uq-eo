"""L3 guarantee layer: what coverage or risk bound can we actually promise?

Split conformal prediction, Mondrian conformal prediction (per region/event),
localized conformal prediction (generalising GeoConformal, arXiv 2412.08661,
to geography + embedding distance), and conformal risk control (e.g. pixel
false negative rate <= alpha for flood/landslide mapping). Consumes cached
probabilities, optional embeddings, and optional coordinates/region labels —
model-agnostic by construction.
"""
from __future__ import annotations

import numpy as np

from eouq.core.uo import UncertaintyObject


def split_conformal_threshold(calibration_scores: np.ndarray, alpha: float) -> float:
    """Standard split conformal quantile of nonconformity scores at level 1 - alpha."""
    raise NotImplementedError


def mondrian_conformal_thresholds(
    calibration_scores: np.ndarray, calibration_groups: np.ndarray, alpha: float
) -> dict:
    """One conformal threshold per group (e.g. per region or per event)."""
    raise NotImplementedError


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
    """Per-query threshold from a kernel-weighted calibration set in geo + embedding space."""
    raise NotImplementedError


def conformal_risk_control_threshold(
    calibration_losses: np.ndarray, alpha: float
) -> float:
    """Threshold on a monotone loss (e.g. per-chip FNR) guaranteeing E[loss] <= alpha, per Angelopoulos et al."""
    raise NotImplementedError


def coverage(predictions_contain_truth: np.ndarray, groups: np.ndarray = None) -> dict:
    """Empirical coverage overall and, if groups is given, per group."""
    raise NotImplementedError


def coverage_disparity(per_group_coverage: dict) -> float:
    """Spread (e.g. max - min) of per-group coverage, the headline spatial-fairness metric."""
    raise NotImplementedError


def guarantee_object(method: str, alpha: float, extra: dict, provenance: dict) -> UncertaintyObject:
    raise NotImplementedError
