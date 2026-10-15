"""SHRUG-FM-style fusion: correlation filter + RFE feature pruning, then a shallow
decision tree predicting failure y_fail = 1{F1 < tau}, trained on train+val only.

Input is the per-chip table of cached UQ signals (U_I, U_E, U_T, ...) plus the
per-chip F1 used only to build training labels. Output is a single scalar risk
score per chip that downstream risk-coverage / AURC metrics consume.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier


def correlation_filter(features: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Return indices of features to keep after dropping one of each highly correlated pair.

    Greedy left-to-right scan: for each feature not yet dropped, drop any later
    feature whose |Pearson correlation| with it exceeds threshold.
    """
    n_features = features.shape[1]
    corr = np.corrcoef(features, rowvar=False)
    corr = np.atleast_2d(np.nan_to_num(corr, nan=0.0))
    dropped = set()
    for i in range(n_features):
        if i in dropped:
            continue
        for j in range(i + 1, n_features):
            if j not in dropped and abs(corr[i, j]) > threshold:
                dropped.add(j)
    return np.array([i for i in range(n_features) if i not in dropped])


def fit_tree_fusion(
    features: np.ndarray, f1_scores: np.ndarray, tau: float, max_depth: int = 3, n_rfe_features: int = None
):
    """Fit correlation filter -> RFE -> depth-limited decision tree on y_fail = 1{F1 < tau}."""
    keep_idx = correlation_filter(features)
    filtered = features[:, keep_idx]
    y_fail = (f1_scores < tau).astype(int)

    if n_rfe_features is None:
        n_rfe_features = max(1, filtered.shape[1] // 2)
    n_rfe_features = min(n_rfe_features, filtered.shape[1])

    if len(np.unique(y_fail)) < 2:
        # Degenerate label set (e.g. tiny synthetic data) — RFE/tree fitting on a
        # single class is meaningless; keep all correlation-filtered features and
        # fit the tree directly, which sklearn handles as a constant predictor.
        selected_idx = keep_idx
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        tree.fit(filtered, y_fail)
        return {"tree": tree, "selected_feature_idx": selected_idx, "tau": tau}

    rfe_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    selector = RFE(rfe_estimator, n_features_to_select=n_rfe_features)
    selector.fit(filtered, y_fail)
    selected_local_idx = np.where(selector.support_)[0]
    selected_idx = keep_idx[selected_local_idx]

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    tree.fit(filtered[:, selected_local_idx], y_fail)

    return {"tree": tree, "selected_feature_idx": selected_idx, "tau": tau}


def predict_risk(model, features: np.ndarray) -> np.ndarray:
    """Predicted probability of failure, used directly as the fused risk score."""
    tree = model["tree"]
    idx = model["selected_feature_idx"]
    probs = tree.predict_proba(features[:, idx])
    if probs.shape[1] == 1:
        # Tree only ever saw one class during fit (degenerate label set) — its
        # single column is P(that class); failure-probability is 0 or 1 accordingly.
        only_class = tree.classes_[0]
        return np.full(len(features), float(only_class))
    return probs[:, 1]
