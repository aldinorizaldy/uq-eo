import numpy as np

from eouq.layers.l4_aggregation import (
    aggregation_object,
    conformal_total_interval,
    per_group_total,
    spatial_block_bootstrap_interval,
)


def test_per_group_total_matches_hand_computed_values():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    totals = per_group_total(values, groups)
    assert totals == {"a": 6.0, "b": 15.0}


def test_spatial_block_bootstrap_interval_is_exact_when_all_groups_equal():
    # every group totals exactly 10 -> every bootstrap resample also totals
    # n_groups * 10, so the interval collapses to a single exact value.
    values = np.array([5.0, 5.0, 6.0, 4.0, 3.0, 7.0])
    groups = np.array(["a", "a", "b", "b", "c", "c"])  # each group sums to 10
    result = spatial_block_bootstrap_interval(values, groups, alpha=0.1, n_bootstrap=100, seed=0)
    assert result["point_estimate"] == 30.0
    np.testing.assert_allclose(result["lower"], 30.0)
    np.testing.assert_allclose(result["upper"], 30.0)


def test_conformal_total_interval_matches_hand_computed_value():
    calibration_predicted = np.array([10.0, 12.0, 8.0, 11.0])
    calibration_true = np.array([9.0, 13.0, 7.0, 10.0])  # all residuals == 1.0
    result = conformal_total_interval(calibration_predicted, calibration_true, query_predicted_total=20.0, alpha=0.1)
    assert result["half_width"] == 1.0
    assert result["lower"] == 19.0
    assert result["upper"] == 21.0


def test_aggregation_object_returns_valid_uncertainty_object():
    uo = aggregation_object("spatial_block_bootstrap", alpha=0.1, extra={"point_estimate": 30.0}, provenance={})
    assert uo.type == "coverage"
    assert uo.source == "l4.spatial_block_bootstrap"
    assert uo.scale == "region"
    assert uo.guarantee == {"method": "spatial_block_bootstrap", "alpha": 0.1}
