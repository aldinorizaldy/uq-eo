import numpy as np

from scripts.evaluate_burn_scars_conformal import (
    chip_roi_entropy,
    grid_district_labels,
    median_pairwise_distance,
    per_chip_areas,
    per_chip_fnr,
    quadrant_region_labels,
    run_conformal_evaluation,
    sample_pixel_nonconformity,
)


def test_grid_district_labels_groups_chips_in_same_cell():
    coords = np.array([[34.1, -118.2], [35.9, -119.9], [36.1, -118.2]])  # first two share a 2-deg cell
    labels = grid_district_labels(coords, cell_deg=2.0)
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_per_chip_areas_counts_valid_pixels_only():
    event_prob = np.array([[[0.9, 0.9], [0.1, 0.9]]])
    masks = np.array([[[1, -1], [1, 0]]])  # (0,1) missing -> excluded from both counts
    predicted, true = per_chip_areas(event_prob, masks)
    assert predicted[0] == 2.0  # (0,0) and (1,1)
    assert true[0] == 2.0  # (0,0) and (1,0)


def test_median_pairwise_distance_hand_computed():
    points = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])  # distances 5, 10, 5
    assert median_pairwise_distance(points) == 5.0


def test_quadrant_region_labels_matches_hand_computed_assignment():
    coords = np.array([[10.0, 10.0], [10.0, -10.0], [-10.0, 10.0], [-10.0, -10.0], [0.0, 0.0]])
    labels = quadrant_region_labels(coords)
    # median lat = 0, median lon = 0 (ties go to "N"/"E" since >=)
    np.testing.assert_array_equal(labels, ["NE", "NW", "SE", "SW", "NE"])


def test_per_chip_fnr_matches_hand_computed_values_and_nan_for_no_positives():
    event_prob = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],  # chip 0: matches mask perfectly
            [[0.1, 0.1], [0.1, 0.1]],  # chip 1: predicts all-negative
            [[0.1, 0.1], [0.1, 0.1]],  # chip 2: no true positives at all
        ]
    )
    masks = np.array(
        [
            [[1, 0], [0, 1]],  # 2 true positives
            [[1, 1], [0, 0]],  # 2 true positives
            [[0, 0], [0, 0]],  # 0 true positives
        ]
    )
    fnr = per_chip_fnr(event_prob, masks, threshold=0.5)
    np.testing.assert_allclose(fnr[:2], [0.0, 1.0])
    assert np.isnan(fnr[2])


def test_chip_roi_entropy_zero_for_fully_confident_predictions():
    mean_probs = np.zeros((1, 2, 2, 2))
    mean_probs[:, 1, :, :] = 1.0  # class 1 (event) probability 1.0 everywhere -> deterministic
    entropy = chip_roi_entropy(mean_probs)
    assert entropy.shape == (1,)
    np.testing.assert_allclose(entropy, [0.0], atol=1e-9)


def test_sample_pixel_nonconformity_excludes_fully_missing_chips():
    mean_probs = np.random.default_rng(0).random((2, 2, 3, 3))
    mean_probs /= mean_probs.sum(axis=1, keepdims=True)
    masks = np.array(
        [
            np.full((3, 3), -1),  # chip 0: fully missing, must contribute nothing
            np.zeros((3, 3), dtype=np.int64),  # chip 1: fully valid
        ]
    )
    scores, chip_idx = sample_pixel_nonconformity(mean_probs, masks, n_per_chip=50, seed=0)
    assert np.all(chip_idx == 1)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def _synthetic_cache(n_samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_members, n_classes, height, width = 3, 2, 6, 6

    logits = rng.normal(size=(n_members, n_samples, n_classes, height, width))
    probs = np.exp(logits)
    probs /= probs.sum(axis=2, keepdims=True)

    lat = rng.uniform(-50.0, 50.0, size=n_samples)
    lon = rng.uniform(-130.0, -60.0, size=n_samples)
    coords = np.stack([lat, lon], axis=1)

    masks = rng.integers(0, 2, size=(n_samples, height, width)).astype(np.int64)
    f1 = rng.uniform(0.0, 1.0, size=n_samples)
    chip_id = np.array([f"chip_{i}" for i in range(n_samples)])

    return {"probs": probs, "embeddings": rng.normal(size=(n_samples, 8)), "coords": coords, "mask": masks, "f1": f1, "chip_id": chip_id}


def test_run_conformal_evaluation_end_to_end_on_synthetic_data():
    test_cache = _synthetic_cache(40, seed=0)
    results = run_conformal_evaluation(test_cache, seed=0)

    for key in (
        "split_conformal",
        "mondrian_conformal",
        "conformal_fusion",
        "conformal_risk_control",
        "localized_conformal",
        "l4_aggregation",
    ):
        assert key in results

    assert 0.0 <= results["localized_conformal"]["coverage"]["overall"] <= 1.0
    assert np.isfinite(results["localized_conformal"]["coverage_disparity"])
    assert results["l4_aggregation"]["n_eval_districts"] > 0
    assert 0.0 <= results["l4_aggregation"]["district_conformal_coverage"] <= 1.0
    low, high = results["l4_aggregation"]["grand_total_bootstrap_interval"]
    assert low <= high

    assert 0.0 <= results["split_conformal"]["coverage"]["overall"] <= 1.0
    assert 0.0 <= results["mondrian_conformal"]["coverage"]["overall"] <= 1.0
    assert np.isfinite(results["split_conformal"]["coverage_disparity"])
    assert np.isfinite(results["mondrian_conformal"]["coverage_disparity"])
    assert np.isfinite(results["conformal_risk_control"]["mean_fnr_on_eval"])
    assert np.isfinite(results["selective_risk_aurc_on_eval_half"])
