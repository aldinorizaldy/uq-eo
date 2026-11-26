import numpy as np

from scripts.evaluate_sensor_noise import run_sensor_noise_evaluation, valid_chip_ids
from scripts.extract_sensor_noise import SensorNoiseTransform, infer_reflectance_scale


def test_sensor_noise_transform_is_reproducible_per_chip_and_draw():
    image = np.full((6, 16, 16), 0.2, dtype=np.float32)

    def noisy(chip_id, draw):
        transform = SensorNoiseTransform(0.01, seed=0, draw=draw, reflectance_scale=1.0)
        return transform({"chip_id": chip_id, "image": image.copy()})["image"]

    np.testing.assert_array_equal(noisy("a", 0), noisy("a", 0))
    assert not np.array_equal(noisy("a", 0), noisy("b", 0))
    assert not np.array_equal(noisy("a", 0), noisy("a", 1))


def test_infer_reflectance_scale_detects_scaled_integers():
    assert infer_reflectance_scale(np.full((6, 4, 4), 0.15)) == 1.0
    assert infer_reflectance_scale(np.full((6, 4, 4), 1500.0)) == 10000.0


def test_valid_chip_ids_requires_defined_f1_at_every_level():
    clean_test = {"chip_id": np.array(["a", "b", "c"]), "f1": np.array([0.5, np.nan, 0.9])}
    sweep = {
        0.0: {"chip_id": np.array(["a", "b", "c"]), "f1": np.array([0.5, np.nan, 0.9])},
        0.05: {"chip_id": np.array(["a", "b", "c"]), "f1": np.array([np.nan, 0.1, 0.8])},
    }
    assert valid_chip_ids(clean_test, sweep) == {"c"}


def _clean_split(n, seed, prefix):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(3, n, 2, 4, 4))
    probs = np.exp(logits) / np.exp(logits).sum(axis=2, keepdims=True)
    return {
        "chip_id": np.array([f"{prefix}{i}" for i in range(n)]),
        "probs": probs,
        "f1": rng.uniform(0, 1, size=n),
        "embeddings": rng.normal(size=(n, 8)).astype(np.float32),
    }


def _sensitivity(split, seed):
    rng = np.random.default_rng(seed)
    n = len(split["chip_id"])
    return {
        "chip_id": split["chip_id"],
        "noise_std": np.array(0.01),
        "n_draws": np.array(5),
        "mean_abs_change": rng.uniform(0, 0.1, size=n),
        "flip_rate": rng.uniform(0, 0.05, size=n),
        "mean_abs_change_draw_std": rng.uniform(0, 0.01, size=n),
    }


def _sweep_level(test, noise_std, seed):
    rng = np.random.default_rng(seed)
    n = len(test["chip_id"])
    return {
        "chip_id": test["chip_id"],
        "noise_std": np.array(noise_std),
        "f1": np.clip(test["f1"] - noise_std * 5, 0, 1),
        "tp": rng.integers(0, 50, size=n),
        "fp": rng.integers(0, 10, size=n),
        "fn": rng.integers(0, 10, size=n),
        "mean_abs_change": rng.uniform(0, noise_std + 1e-3, size=n),
        "flip_rate": rng.uniform(0, noise_std + 1e-3, size=n),
        "roi_entropy": rng.uniform(0, 1, size=n),
        "roi_mutual_information": rng.uniform(0, 0.2, size=n),
        "embeddings": rng.normal(size=(n, 8)).astype(np.float32),
    }


def test_run_sensor_noise_evaluation_reports_signal_and_sweep_results():
    clean = {"train": _clean_split(60, 0, "tr"), "val": _clean_split(15, 1, "va"), "test": _clean_split(20, 2, "te")}
    sensitivity = {split: _sensitivity(clean[split], i) for i, split in enumerate(("train", "val", "test"))}
    sweep = {std: _sweep_level(clean["test"], std, i) for i, std in enumerate((0.0, 0.01, 0.05))}
    eval_ids = {str(c) for c in clean["test"]["chip_id"][:15]}

    results = run_sensor_noise_evaluation(clean, sweep, sensitivity, eval_ids, tau=0.6)

    assert results["n_test_chips"] == 15
    assert "noise_sensitivity" in results["feature_names_with_l0"]
    assert "noise_sensitivity" not in results["feature_names"]
    signal = results["sensitivity_signal"]
    for key in ("fused_without_l0", "fused_with_l0", "baseline_noise_sensitivity", "baseline_noise_flip_rate"):
        assert 0.0 <= signal[key]["aurc"] <= 1.0
    assert set(signal["spearman_with_existing_signals"]["noise_sensitivity"]) == set(results["feature_names"])

    levels = results["degradation_sweep"]
    assert [lvl["noise_std"] for lvl in levels] == [0.0, 0.01, 0.05]
    assert "spearman_f1_drop_vs_entropy_rise" not in levels[0]
    assert "spearman_f1_drop_vs_entropy_rise" in levels[1]
    assert levels[2]["mean_chip_f1"] <= levels[0]["mean_chip_f1"]


def test_run_sensor_noise_evaluation_rejects_misaligned_chip_order():
    clean = {"train": _clean_split(40, 0, "tr"), "val": _clean_split(10, 1, "va"), "test": _clean_split(10, 2, "te")}
    sensitivity = {split: _sensitivity(clean[split], i) for i, split in enumerate(("train", "val", "test"))}
    sensitivity["test"]["chip_id"] = sensitivity["test"]["chip_id"][::-1]
    try:
        run_sensor_noise_evaluation(clean, {}, sensitivity, set(clean["test"]["chip_id"]))
    except ValueError as err:
        assert "sensitivity_test" in str(err)
    else:
        raise AssertionError("expected a ValueError for misaligned chip order")
