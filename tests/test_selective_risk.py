import numpy as np

from eouq.metrics.selective_risk import aurc, audp, discard_performance_curve, risk_at_coverage, risk_coverage_curve


def test_risk_coverage_curve_matches_hand_computed_values_no_ties():
    risk_scores = np.array([5.0, 3.0, 1.0, 4.0, 2.0])
    losses = np.array([50.0, 30.0, 10.0, 40.0, 20.0])  # loss[i] == 10 * risk_scores[i]
    coverage, risk = risk_coverage_curve(risk_scores, losses, n_tie_permutations=5, seed=0)

    np.testing.assert_allclose(coverage, [0.2, 0.4, 0.6, 0.8, 1.0])
    # ascending by risk_scores: losses in order 10, 20, 30, 40, 50 -> cumulative means
    np.testing.assert_allclose(risk, [10.0, 15.0, 20.0, 25.0, 30.0])


def test_aurc_matches_trapezoid_of_linear_curve():
    coverage = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    risk = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    assert aurc(coverage, risk) == 16.0


def test_risk_at_coverage_interpolates_linearly():
    coverage = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    risk = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    assert risk_at_coverage(coverage, risk, target_coverage=0.5) == 17.5


def test_discard_performance_curve_endpoints_are_order_independent_invariants():
    risk_scores = np.array([5.0, 3.0, 1.0, 4.0, 2.0])
    metric_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fraction_discarded, performance = discard_performance_curve(
        risk_scores, metric_values, n_tie_permutations=5, seed=0
    )

    np.testing.assert_allclose(fraction_discarded, [0.0, 0.2, 0.4, 0.6, 0.8])
    # discarding nothing: mean over all samples, regardless of sort order
    assert performance[0] == metric_values.mean()
    # discarding all but one: only the single lowest-risk sample remains
    idx_min_risk = np.argmin(risk_scores)
    assert performance[-1] == metric_values[idx_min_risk]


def test_audp_sorts_by_fraction_discarded_before_integrating():
    fraction_discarded = np.array([0.4, 0.0, 0.2])
    metric_values = np.array([3.0, 1.0, 2.0])  # aligned: (0.0,1.0), (0.2,2.0), (0.4,3.0)
    sorted_value = audp(fraction_discarded, metric_values)
    expected = np.trapz([1.0, 2.0, 3.0], [0.0, 0.2, 0.4])
    assert sorted_value == expected
