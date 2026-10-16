import numpy as np

from scripts.evaluate_burn_scars import run_evaluation


def _synthetic_split(n_samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_members, n_classes, height, width = 3, 2, 4, 4

    logits = rng.normal(size=(n_members, n_samples, n_classes, height, width))
    probs = np.exp(logits)
    probs /= probs.sum(axis=2, keepdims=True)

    embeddings = rng.normal(size=(n_samples, 8)).astype(np.float32)
    f1 = rng.uniform(0.0, 1.0, size=n_samples)

    return {"probs": probs, "embeddings": embeddings, "f1": f1}


def test_run_evaluation_end_to_end_on_synthetic_data():
    train = _synthetic_split(40, seed=0)
    val = _synthetic_split(10, seed=1)
    test = _synthetic_split(15, seed=2)

    results = run_evaluation(train, val, test, tau=0.6)

    assert set(results["feature_names"]) == {
        "roi_entropy",
        "roi_mutual_information",
        "normalized_centroid_distance",
        "ncdd",
    }
    assert results["n_test_chips"] == 15

    expected_keys = {"aurc", "risk_at_0.5", "risk_at_0.75", "audp"}
    for group in ("fused", "baseline_random", "baseline_ncdd", "baseline_roi_entropy"):
        assert group in results
        assert set(results[group].keys()) == expected_keys
        for value in results[group].values():
            assert np.isfinite(value)
