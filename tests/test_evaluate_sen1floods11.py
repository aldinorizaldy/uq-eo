import numpy as np

from scripts.evaluate_sen1floods11 import common_valid_test_chips, run_evaluation


def _synthetic_split(n_samples: int, seed: int, with_embeddings: bool) -> dict:
    rng = np.random.default_rng(seed)
    n_members, n_classes, height, width = 3, 2, 4, 4

    logits = rng.normal(size=(n_members, n_samples, n_classes, height, width))
    probs = np.exp(logits)
    probs /= probs.sum(axis=2, keepdims=True)

    f1 = rng.uniform(0.0, 1.0, size=n_samples)
    split = {"probs": probs, "f1": f1}
    if with_embeddings:
        split["embeddings"] = rng.normal(size=(n_samples, 8)).astype(np.float32)
    return split


def test_run_evaluation_with_embeddings_includes_l1_signals():
    train = _synthetic_split(40, seed=0, with_embeddings=True)
    val = _synthetic_split(10, seed=1, with_embeddings=True)
    test = _synthetic_split(15, seed=2, with_embeddings=True)

    results = run_evaluation(train, val, test, tau=0.6)

    assert results["has_embeddings"] is True
    assert set(results["feature_names"]) == {
        "roi_entropy",
        "roi_mutual_information",
        "normalized_centroid_distance",
        "ncdd",
    }
    assert "baseline_ncdd" in results


def test_run_evaluation_without_embeddings_skips_l1_signals():
    train = _synthetic_split(40, seed=0, with_embeddings=False)
    val = _synthetic_split(10, seed=1, with_embeddings=False)
    test = _synthetic_split(15, seed=2, with_embeddings=False)

    results = run_evaluation(train, val, test, tau=0.6)

    assert results["has_embeddings"] is False
    assert set(results["feature_names"]) == {"roi_entropy", "roi_mutual_information"}
    assert "baseline_ncdd" not in results
    assert "fused" in results
    for value in results["fused"].values():
        assert np.isfinite(value)


def test_run_evaluation_restricts_test_scoring_to_eval_chip_ids():
    train = _synthetic_split(40, seed=0, with_embeddings=False)
    val = _synthetic_split(10, seed=1, with_embeddings=False)
    test = _synthetic_split(15, seed=2, with_embeddings=False)
    test["chip_id"] = np.array([f"chip_{i}" for i in range(15)])

    keep = {f"chip_{i}" for i in range(10)}  # first 10 of 15
    results = run_evaluation(train, val, test, tau=0.6, eval_chip_ids=keep)
    assert results["n_test_chips"] == 10


def test_common_valid_test_chips_is_intersection_of_non_nan_f1(tmp_path):
    chip_ids = np.array(["a", "b", "c", "d"])
    model_a = tmp_path / "model_a"
    model_b = tmp_path / "model_b"
    model_a.mkdir()
    model_b.mkdir()
    np.savez(model_a / "test.npz", chip_id=chip_ids, f1=np.array([0.5, np.nan, 0.7, 0.8]))  # drops b
    np.savez(model_b / "test.npz", chip_id=chip_ids, f1=np.array([0.4, 0.6, np.nan, 0.9]))  # drops c

    assert common_valid_test_chips([str(model_a), str(model_b)]) == {"a", "d"}
