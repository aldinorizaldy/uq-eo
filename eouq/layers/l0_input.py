"""L0 input layer: is the raw input unusual or noisy, before it reaches the model?

SHRUG-FM-style signal: percentile rank of each geophysical covariate (HydroATLAS
attributes, Copernicus DEM derivatives) against the pretraining-data CDF, plus a
KDE spatial density of pretraining sample locations. Consumes cached covariate
arrays — never touches raw rasters or model internals.

Also: sensor-noise perturbation sensitivity (perturb inputs, re-run through the
model, measure output sensitivity) -- this one DOES need to re-run the model, via
a caller-supplied `predict_fn`, kept generic so this layer stays model-agnostic.
"""
from __future__ import annotations

from typing import Callable

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


def sensor_noise_sensitivity(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    images: np.ndarray,
    noise_std: float,
    n_perturbations: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Perturb each image with Gaussian sensor-like noise, re-run through the
    model (`predict_fn`), and measure output sensitivity: mean absolute
    difference between perturbed and original per-pixel predictions,
    averaged over `n_perturbations` independent noise draws.

    A direct probe of "how much does a small, physically-motivated sensor
    perturbation change the output" -- distinct from embedding OOD (is the
    input unfamiliar) or ensemble disagreement (do multiple models/samples
    disagree); this measures sensitivity of a single prediction pipeline to
    its own input noise.

    predict_fn: callable taking a batch of images (n_samples, C, H, W) and
    returning the per-pixel event-class probability (n_samples, H, W).
    Takes a plain callable rather than a specific adapter/model type so
    this layer stays model-agnostic, consistent with the rest of eouq.
    """
    rng = np.random.default_rng(seed)
    baseline_pred = predict_fn(images)
    sensitivity = np.zeros(len(images))
    for _ in range(n_perturbations):
        noise = rng.normal(0.0, noise_std, size=images.shape)
        perturbed_pred = predict_fn(images + noise)
        sensitivity += np.mean(np.abs(perturbed_pred - baseline_pred), axis=(1, 2))
    return sensitivity / n_perturbations


def sensor_noise_ood_score(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    images: np.ndarray,
    provenance: dict,
    noise_std: float = 0.01,
    n_perturbations: int = 10,
    seed: int = 0,
) -> UncertaintyObject:
    sensitivity = sensor_noise_sensitivity(predict_fn, images, noise_std, n_perturbations, seed)
    return UncertaintyObject(
        value={"sensitivity": sensitivity},
        type="aleatoric",
        source="l0.sensor_noise_perturbation",
        scale="chip",
        provenance=provenance,
    )


def add_reflectance_noise(
    image: np.ndarray, noise_std: float, rng: np.random.Generator, reflectance_scale: float = 1.0
) -> np.ndarray:
    """Additive, zero-mean Gaussian sensor noise, independent per pixel and band.

    noise_std is in surface-reflectance units (0.002 ~ Sentinel-2 MSI
    noise-equivalent reflectance at its reference radiance; larger values are
    stress tests); `reflectance_scale` converts it to the image's own units
    (1.0 for [0, 1] reflectance, 10000.0 for scaled integers).

    Valid pixels are clipped at zero (reflectance cannot be negative). Pixels
    that are already negative or non-finite (no-data fill values) are left
    untouched, so the perturbation never alters a no-data flag.
    """
    if noise_std == 0.0:
        return image.copy()
    noise = rng.normal(0.0, noise_std * reflectance_scale, size=image.shape).astype(image.dtype)
    perturbed = np.maximum(image + noise, 0.0)
    valid = np.isfinite(image) & (image >= 0)
    return np.where(valid, perturbed, image).astype(image.dtype)


def prediction_change(
    clean_event_prob: np.ndarray, noisy_event_prob: np.ndarray, masks: np.ndarray, threshold: float = 0.5
) -> tuple:
    """Per-chip output sensitivity of an already-run noisy prediction.

    clean_event_prob, noisy_event_prob, masks: (n, H, W); mask == -1 is
    missing data and excluded. Returns (mean_abs_change, flip_rate), each (n,):
    the mean absolute change in event probability, and the fraction of valid
    pixels whose thresholded label flips. Model-agnostic: it consumes two sets
    of cached predictions rather than calling a model.
    """
    valid = masks != -1
    n_valid = np.maximum(valid.sum(axis=(1, 2)), 1)
    abs_change = np.abs(noisy_event_prob - clean_event_prob) * valid
    flips = ((noisy_event_prob >= threshold) != (clean_event_prob >= threshold)) & valid
    return abs_change.sum(axis=(1, 2)) / n_valid, flips.sum(axis=(1, 2)) / n_valid
