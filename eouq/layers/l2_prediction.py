"""L2 prediction layer: how confident is the model's output, independent of OOD?

SHRUG-FM-style signal: deep ensemble (standard or bootstrap) mean probability,
average predictive entropy, and mutual information (epistemic component),
averaged over the predicted region of interest. Also provides MC-dropout,
test-time augmentation (TTA), and temperature scaling as alternative or
complementary predictive-uncertainty estimators. Consumes cached per-member
probability arrays — never re-runs the model.
"""
from __future__ import annotations

import numpy as np

from eouq.core.uo import UncertaintyObject

_EPS = 1e-12


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) over the last axis of a probability array."""
    p = np.clip(probs, _EPS, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def ensemble_mean_prob(member_probs: np.ndarray) -> np.ndarray:
    """member_probs: (n_members, ..., n_classes) -> mean over members."""
    return member_probs.mean(axis=0)


def predictive_entropy(mean_probs: np.ndarray) -> np.ndarray:
    return _entropy(mean_probs)


def expected_entropy(member_probs: np.ndarray) -> np.ndarray:
    return _entropy(member_probs).mean(axis=0)


def mutual_information(member_probs: np.ndarray) -> np.ndarray:
    """Predictive entropy minus expected entropy (epistemic / BALD score)."""
    mean_probs = ensemble_mean_prob(member_probs)
    mi = predictive_entropy(mean_probs) - expected_entropy(member_probs)
    return np.clip(mi, 0.0, None)  # clamp small negative values from float error


def roi_average(per_pixel_values: np.ndarray, mean_probs: np.ndarray, prob_threshold: float) -> float:
    """Average a per-pixel uncertainty value over the region where mean_probs exceeds prob_threshold.

    mean_probs must already be the positive/event-class probability (same shape
    as per_pixel_values), not the full (..., n_classes) ensemble mean.
    """
    roi_mask = mean_probs >= prob_threshold
    if not np.any(roi_mask):
        return float("nan")
    return float(per_pixel_values[roi_mask].mean())


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


def predictive_uncertainty_score(
    member_probs: np.ndarray, provenance: dict, prob_threshold: float = 0.5
) -> UncertaintyObject:
    """member_probs: (n_members, ..., n_classes), binary or multi-class. ROI averaging
    uses the last class as the "event" class (e.g. burn/flood/landslide positive class)."""
    mean_probs = ensemble_mean_prob(member_probs)
    pred_entropy = predictive_entropy(mean_probs)
    exp_entropy = expected_entropy(member_probs)
    mi = mutual_information(member_probs)
    event_prob = mean_probs[..., -1]

    value = {
        "mean_prob": mean_probs,
        "predictive_entropy": pred_entropy,
        "expected_entropy": exp_entropy,
        "mutual_information": mi,
        "roi_entropy": roi_average(pred_entropy, event_prob, prob_threshold),
        "roi_mutual_information": roi_average(mi, event_prob, prob_threshold),
    }
    return UncertaintyObject(value=value, type="epistemic", source="l2.ensemble", scale="chip", provenance=provenance)
