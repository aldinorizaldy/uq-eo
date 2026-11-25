import numpy as np

from scripts.evaluate_sen1floods11_conformal import event_names, run_shift_evaluation


def test_event_names_strips_chip_number_and_keeps_hyphenated_names():
    np.testing.assert_array_equal(
        event_names(np.array(["Ghana_103272", "Sri-Lanka_956740"])), ["Ghana", "Sri-Lanka"]
    )


def _synthetic_cache(n: int, events: list, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_members, n_classes, h, w = 3, 2, 6, 6
    logits = rng.normal(size=(n_members, n, n_classes, h, w))
    probs = np.exp(logits)
    probs /= probs.sum(axis=2, keepdims=True)
    masks = rng.integers(0, 2, size=(n, h, w)).astype(np.int64)
    chip_id = np.array([f"{events[i % len(events)]}_{i}" for i in range(n)])
    return {"probs": probs, "mask": masks, "chip_id": chip_id, "coords": rng.normal(size=(n, 2)), "f1": rng.random(n)}


def test_run_shift_evaluation_end_to_end_on_synthetic_data():
    val = _synthetic_cache(12, ["USA", "India"], seed=0)
    test = _synthetic_cache(20, ["Ghana", "Nigeria", "Somalia"], seed=1)
    results = run_shift_evaluation(val, test, seed=0)

    assert results["n_eval_chips"] == 10
    for name in ("calibrated_on_other_continents", "calibrated_in_distribution"):
        cov = results[name]["coverage"]
        assert 0.0 <= cov["overall"] <= 1.0
        assert set(cov["per_group"]) <= {"Ghana", "Nigeria", "Somalia"}
        assert np.isfinite(results[name]["event_coverage_disparity"])
    rc = results["conformal_risk_control"]
    assert np.isfinite(rc["calibrated_on_other_continents"]["mean_fnr_on_eval"])
    assert np.isfinite(rc["calibrated_in_distribution"]["mean_fnr_on_eval"])
