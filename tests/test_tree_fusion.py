import numpy as np

from eouq.fusion.tree_fusion import correlation_filter, fit_tree_fusion, predict_risk


def test_correlation_filter_drops_perfectly_correlated_column():
    col0 = np.array([0.0, 1.0, 2.0, 3.0])
    col1 = 2.0 * col0  # perfectly correlated with col0
    col2 = np.array([3.0, 0.0, 3.0, 0.0])  # correlation with col0 is ~-0.447, well under threshold
    features = np.stack([col0, col1, col2], axis=1)

    keep_idx = correlation_filter(features, threshold=0.95)
    np.testing.assert_array_equal(keep_idx, [0, 2])


def test_fit_tree_fusion_and_predict_risk_on_perfectly_separable_signal():
    n = 40
    risk_signal = np.linspace(0.0, 1.0, n)
    f1_scores = 1.0 - risk_signal
    tau = 0.5
    features = risk_signal.reshape(-1, 1)

    model = fit_tree_fusion(features, f1_scores, tau=tau, max_depth=3)
    predicted = predict_risk(model, features)

    y_fail = (f1_scores < tau).astype(int)
    predicted_label = (predicted > 0.5).astype(int)
    np.testing.assert_array_equal(predicted_label, y_fail)


def test_fit_tree_fusion_handles_degenerate_single_class_labels():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(10, 2))
    f1_scores = np.full(10, 0.9)  # always above tau -> no failures at all
    tau = 0.5

    model = fit_tree_fusion(features, f1_scores, tau=tau)
    predicted = predict_risk(model, features)
    np.testing.assert_allclose(predicted, np.zeros(10))
