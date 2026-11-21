import numpy as np

from eouq.layers.l3_guarantee import (
    conformal_risk_control_threshold,
    coverage,
    coverage_disparity,
    guarantee_object,
    localized_conformal_threshold,
    mondrian_conformal_thresholds,
    split_conformal_threshold,
)


def test_split_conformal_threshold_matches_hand_computed_value():
    calibration_scores = np.arange(1.0, 11.0)  # [1..10]
    threshold = split_conformal_threshold(calibration_scores, alpha=0.3)
    # q_level = ceil(11*0.7)/10 = 0.8; quantile([1..10], 0.8, method="higher") = 9
    assert threshold == 9.0


def test_mondrian_conformal_thresholds_computed_independently_per_group():
    calibration_scores = np.concatenate([np.arange(1.0, 11.0), np.arange(100.0, 110.0)])
    calibration_groups = np.array(["a"] * 10 + ["b"] * 10)

    thresholds = mondrian_conformal_thresholds(calibration_scores, calibration_groups, alpha=0.3)

    assert thresholds["a"] == split_conformal_threshold(np.arange(1.0, 11.0), alpha=0.3)
    assert thresholds["b"] == split_conformal_threshold(np.arange(100.0, 110.0), alpha=0.3)


def test_localized_conformal_threshold_dominated_by_nearest_calibration_point():
    calibration_scores = np.array([5.0, 100.0])
    calibration_coords = np.array([[0.0, 0.0], [1000.0, 1000.0]])
    calibration_embeddings = np.array([[0.0, 0.0], [1000.0, 1000.0]])
    query_coords = np.array([[0.0, 0.0]])
    query_embeddings = np.array([[0.0, 0.0]])

    thresholds = localized_conformal_threshold(
        calibration_scores,
        calibration_coords,
        calibration_embeddings,
        query_coords,
        query_embeddings,
        alpha=0.5,
        bandwidth_geo=1.0,
        bandwidth_embed=1.0,
    )
    # the second calibration point is so far away (in both geo and embedding
    # space) its Gaussian weight underflows to exactly 0.0 -- threshold should
    # be driven entirely by the first (coincident) calibration point.
    assert thresholds[0] == 5.0


def test_localized_conformal_threshold_falls_back_to_uniform_when_all_weights_vanish():
    calibration_scores = np.array([1.0, 2.0, 3.0])
    calibration_coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    calibration_embeddings = np.zeros((3, 2))
    query_coords = np.array([[1e6, 1e6]])  # astronomically far from all calibration points
    query_embeddings = np.zeros((1, 2))

    thresholds = localized_conformal_threshold(
        calibration_scores,
        calibration_coords,
        calibration_embeddings,
        query_coords,
        query_embeddings,
        alpha=0.5,
        bandwidth_geo=1.0,
        bandwidth_embed=1.0,
    )
    assert np.all(np.isfinite(thresholds))


def test_conformal_risk_control_threshold_returns_largest_valid_lambda():
    def loss_fn(lam):
        return np.full(20, lam)

    lambda_grid = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    result = conformal_risk_control_threshold(loss_fn, alpha=0.5, lambda_grid=lambda_grid)
    # corrected_risk(lambda) = (20*lambda + 1)/21; <=0.5 first holds at lambda=0.3
    assert result == 0.3


def test_conformal_risk_control_threshold_returns_most_conservative_if_none_valid():
    def loss_fn(lam):
        return np.full(20, lam)

    lambda_grid = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    result = conformal_risk_control_threshold(loss_fn, alpha=0.01, lambda_grid=lambda_grid)
    assert result == lambda_grid[-1]


def test_coverage_overall_and_per_group():
    predictions_contain_truth = np.array([1, 1, 0, 1, 0, 0])
    groups = np.array(["a", "a", "a", "b", "b", "b"])

    result = coverage(predictions_contain_truth, groups)
    assert result["overall"] == 0.5
    np.testing.assert_allclose(result["per_group"]["a"], 2 / 3)
    np.testing.assert_allclose(result["per_group"]["b"], 1 / 3)


def test_coverage_disparity_is_max_minus_min():
    disparity = coverage_disparity({"a": 2 / 3, "b": 1 / 3})
    np.testing.assert_allclose(disparity, 1 / 3)


def test_guarantee_object_returns_valid_uncertainty_object():
    uo = guarantee_object("split_conformal", alpha=0.1, extra={"threshold": 5.0}, provenance={"seed": 0})
    assert uo.type == "coverage"
    assert uo.source == "l3.split_conformal"
    assert uo.scale == "aggregate"
    assert uo.guarantee == {"method": "split_conformal", "alpha": 0.1}
