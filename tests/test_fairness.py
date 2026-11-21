import numpy as np

from eouq.metrics.fairness import coverage_disparity, coverage_per_group, expected_calibration_error


def test_coverage_per_group_matches_hand_computed_values():
    predictions_contain_truth = np.array([1, 1, 0, 1, 0, 0])
    groups = np.array(["a", "a", "a", "b", "b", "b"])

    result = coverage_per_group(predictions_contain_truth, groups)
    np.testing.assert_allclose(result["a"], 2 / 3)
    np.testing.assert_allclose(result["b"], 1 / 3)


def test_coverage_disparity_range_and_std():
    per_group = {"a": 0.8, "b": 0.6, "c": 0.5}
    np.testing.assert_allclose(coverage_disparity(per_group, method="range"), 0.3)
    np.testing.assert_allclose(coverage_disparity(per_group, method="std"), np.std([0.8, 0.6, 0.5]))


def test_coverage_disparity_rejects_unknown_method():
    import pytest

    with pytest.raises(ValueError):
        coverage_disparity({"a": 0.5}, method="bogus")


def test_expected_calibration_error_zero_when_perfectly_calibrated():
    confidences = np.array([0.2] * 5 + [0.8] * 5)
    correctness = np.array([1, 0, 0, 0, 0] + [1, 1, 1, 1, 0])  # bin means: 0.2 and 0.8, exactly matching confidence
    ece = expected_calibration_error(confidences, correctness, n_bins=2)
    np.testing.assert_allclose(ece, 0.0, atol=1e-9)


def test_expected_calibration_error_matches_hand_computed_value_when_miscalibrated():
    confidences = np.array([0.2] * 5 + [0.8] * 5)
    correctness = np.array([1, 1, 1, 1, 1] + [0, 0, 0, 0, 0])  # bin accuracies: 1.0 and 0.0
    ece = expected_calibration_error(confidences, correctness, n_bins=2)
    # bin0: |0.2 - 1.0| * 0.5 = 0.4; bin1: |0.8 - 0.0| * 0.5 = 0.4; total 0.8
    np.testing.assert_allclose(ece, 0.8)
