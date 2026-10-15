import numpy as np

from eouq.layers.l2_prediction import (
    ensemble_mean_prob,
    expected_entropy,
    mutual_information,
    predictive_entropy,
    predictive_uncertainty_score,
    roi_average,
    temperature_scale,
)


def test_ensemble_mean_prob_averages_over_members():
    member_probs = np.array(
        [
            [[0.8, 0.2], [0.4, 0.6]],
            [[0.6, 0.4], [0.6, 0.4]],
        ]
    )  # (n_members=2, n_samples=2, n_classes=2)
    mean_probs = ensemble_mean_prob(member_probs)
    np.testing.assert_allclose(mean_probs, [[0.7, 0.3], [0.5, 0.5]])


def test_predictive_entropy_zero_for_deterministic_distribution():
    mean_probs = np.array([[1.0, 0.0]])
    entropy = predictive_entropy(mean_probs)
    np.testing.assert_allclose(entropy, [0.0], atol=1e-9)


def test_mutual_information_zero_when_members_agree():
    member_probs = np.tile(np.array([[0.3, 0.7]]), (5, 1, 1))  # 5 identical members
    mi = mutual_information(member_probs)
    np.testing.assert_allclose(mi, [0.0], atol=1e-9)


def test_mutual_information_positive_when_members_disagree():
    member_probs = np.array(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 0.0]],
        ]
    )  # 3 members, 1 sample, disagree on the class
    mi = mutual_information(member_probs)
    pred_entropy = predictive_entropy(ensemble_mean_prob(member_probs))
    exp_entropy = expected_entropy(member_probs)
    assert mi[0] > 0.0
    np.testing.assert_allclose(exp_entropy, [0.0], atol=1e-9)  # each member deterministic
    np.testing.assert_allclose(mi, pred_entropy, atol=1e-9)  # MI = pred_entropy - 0


def test_roi_average_only_includes_pixels_above_threshold():
    per_pixel_values = np.array([1.0, 2.0, 3.0, 4.0])
    event_prob = np.array([0.1, 0.6, 0.9, 0.4])
    avg = roi_average(per_pixel_values, event_prob, prob_threshold=0.5)
    assert avg == 2.5  # mean of pixels 1 and 2 (values 2.0 and 3.0)


def test_roi_average_nan_when_roi_is_empty():
    per_pixel_values = np.array([1.0, 2.0])
    event_prob = np.array([0.1, 0.2])
    avg = roi_average(per_pixel_values, event_prob, prob_threshold=0.9)
    assert np.isnan(avg)


def test_temperature_scale_outputs_valid_probabilities():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 5, 3))
    probs = temperature_scale(logits, temperature=2.0)
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones((4, 5)), atol=1e-9)
    assert np.all(probs >= 0.0)


def test_predictive_uncertainty_score_returns_valid_uncertainty_object():
    rng = np.random.default_rng(0)
    member_probs = rng.random(size=(4, 3, 3, 2))  # members, H, W, classes (unnormalised ok for a shape test)
    member_probs = member_probs / member_probs.sum(axis=-1, keepdims=True)
    uo = predictive_uncertainty_score(member_probs, provenance={"seed": 0})
    assert uo.type == "epistemic"
    assert uo.source == "l2.ensemble"
    assert uo.value["mean_prob"].shape == (3, 3, 2)
    assert uo.value["predictive_entropy"].shape == (3, 3)
    assert isinstance(uo.value["roi_entropy"], float)
