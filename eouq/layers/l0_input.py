"""L0 input layer: is the raw input unusual or noisy, before it reaches the model?

SHRUG-FM-style signal: percentile rank of each geophysical covariate (HydroATLAS
attributes, Copernicus DEM derivatives) against the pretraining-data CDF, plus a
KDE spatial density of pretraining sample locations. Consumes cached covariate
arrays — never touches raw rasters or model internals.

Later: sensor-noise perturbation sensitivity (perturb inputs, re-run through the
model, measure output sensitivity) will live here too.
"""
from __future__ import annotations

import numpy as np

from eouq.core.uo import UncertaintyObject


def covariate_percentile_rank(
    query_covariates: np.ndarray, reference_covariates: np.ndarray
) -> np.ndarray:
    """Percentile rank of each query covariate value against a reference (pretraining) distribution.

    query_covariates: (n_samples, n_covariates)
    reference_covariates: (n_reference, n_covariates)
    returns: (n_samples, n_covariates) percentile ranks in [0, 1]
    """
    n_samples, n_covariates = query_covariates.shape
    ranks = np.empty((n_samples, n_covariates), dtype=float)
    for j in range(n_covariates):
        ref_sorted = np.sort(reference_covariates[:, j])
        ranks[:, j] = np.searchsorted(ref_sorted, query_covariates[:, j], side="right") / len(ref_sorted)
    return ranks


def spatial_kde_density(
    query_coords: np.ndarray, reference_coords: np.ndarray, bandwidth: float
) -> np.ndarray:
    """KDE density of reference (pretraining) locations evaluated at query coordinates.

    Isotropic Gaussian kernel with an explicit (absolute, not Scott/Silverman-scaled)
    bandwidth, so results are directly comparable across datasets of different size.
    O(n_query * n_reference) — fine at chip granularity, not meant for per-pixel use.
    """
    n_reference, n_dims = reference_coords.shape
    diffs = query_coords[:, None, :] - reference_coords[None, :, :]
    sq_dists = np.sum((diffs / bandwidth) ** 2, axis=-1)
    kernel_vals = np.exp(-0.5 * sq_dists)
    norm_const = (2 * np.pi) ** (n_dims / 2) * bandwidth**n_dims
    return kernel_vals.sum(axis=1) / (n_reference * norm_const)


def input_ood_score(
    query_covariates: np.ndarray,
    reference_covariates: np.ndarray,
    provenance: dict,
) -> UncertaintyObject:
    """Covariate-based input OOD score (U_I^Q in SHRUG-FM): percentile rank of each
    covariate against the reference distribution, summarised as extremity (distance
    from the median rank, 0 = typical, 1 = most extreme possible).

    Spatial KDE density (U_I^density) is a separate signal — combine with
    `spatial_kde_density` downstream (e.g. in fusion) since it needs coordinates
    rather than covariate values.
    """
    ranks = covariate_percentile_rank(query_covariates, reference_covariates)
    extremity = np.abs(ranks - 0.5) * 2.0
    value = {
        "percentile_rank": ranks,
        "extremity": extremity,
        "mean_extremity": extremity.mean(axis=1),
    }
    return UncertaintyObject(value=value, type="ood", source="l0.covariate_percentile", scale="chip", provenance=provenance)
