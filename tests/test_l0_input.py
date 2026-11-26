import numpy as np

from eouq.layers.l0_input import (
    covariate_percentile_rank,
    input_ood_score,
    sensor_noise_ood_score,
    sensor_noise_sensitivity,
    spatial_kde_density,
)


def test_covariate_percentile_rank_matches_hand_computed_values():
    reference = np.array([[0.0], [10.0], [20.0], [30.0], [40.0]])
    query = np.array([[-5.0], [0.0], [15.0], [40.0], [100.0]])
    ranks = covariate_percentile_rank(query, reference)
    expected = np.array([[0.0], [0.2], [0.4], [1.0], [1.0]])
    np.testing.assert_allclose(ranks, expected)


def test_covariate_percentile_rank_multi_column_shape():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(50, 3))
    query = rng.normal(size=(5, 3))
    ranks = covariate_percentile_rank(query, reference)
    assert ranks.shape == (5, 3)
    assert np.all((ranks >= 0.0) & (ranks <= 1.0))


def test_spatial_kde_density_higher_near_reference_cluster():
    rng = np.random.default_rng(0)
    reference_coords = rng.normal(loc=0.0, scale=1.0, size=(200, 2))
    query_coords = np.array([[0.0, 0.0], [1000.0, 1000.0]])
    density = spatial_kde_density(query_coords, reference_coords, bandwidth=1.0)
    assert density.shape == (2,)
    assert density[0] > density[1]
    assert density[1] >= 0.0


def test_input_ood_score_returns_valid_uncertainty_object():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(100, 4))
    query = rng.normal(size=(6, 4))
    uo = input_ood_score(query, reference, provenance={"seed": 0})
    assert uo.type == "ood"
    assert uo.source == "l0.covariate_percentile"
    assert uo.scale == "chip"
    assert uo.value["percentile_rank"].shape == (6, 4)
    assert uo.value["mean_extremity"].shape == (6,)
    assert np.all((uo.value["extremity"] >= 0.0) & (uo.value["extremity"] <= 1.0))


def _mean_channel_predict_fn(images: np.ndarray) -> np.ndarray:
    return images.mean(axis=1)  # (n, C, H, W) -> (n, H, W)


def test_sensor_noise_sensitivity_is_exactly_zero_with_no_noise():
    images = np.random.default_rng(0).normal(size=(3, 4, 5, 5))
    sensitivity = sensor_noise_sensitivity(_mean_channel_predict_fn, images, noise_std=0.0, n_perturbations=5, seed=0)
    np.testing.assert_allclose(sensitivity, np.zeros(3))


def test_sensor_noise_sensitivity_increases_with_noise_std():
    images = np.random.default_rng(0).normal(size=(3, 4, 5, 5))
    low = sensor_noise_sensitivity(_mean_channel_predict_fn, images, noise_std=0.01, n_perturbations=200, seed=1)
    high = sensor_noise_sensitivity(_mean_channel_predict_fn, images, noise_std=1.0, n_perturbations=200, seed=1)
    assert np.all(high > low)


def test_sensor_noise_ood_score_returns_valid_uncertainty_object():
    images = np.random.default_rng(0).normal(size=(3, 4, 5, 5))
    uo = sensor_noise_ood_score(_mean_channel_predict_fn, images, provenance={"seed": 0}, noise_std=0.1, n_perturbations=5)
    assert uo.type == "aleatoric"
    assert uo.source == "l0.sensor_noise_perturbation"
    assert uo.scale == "chip"
    assert uo.value["sensitivity"].shape == (3,)


def test_add_reflectance_noise_is_identity_at_zero_std():
    from eouq.layers.l0_input import add_reflectance_noise

    image = np.random.default_rng(0).uniform(0, 1, size=(6, 8, 8)).astype(np.float32)
    np.testing.assert_array_equal(add_reflectance_noise(image, 0.0, np.random.default_rng(1)), image)


def test_add_reflectance_noise_scales_clips_and_preserves_nodata():
    from eouq.layers.l0_input import add_reflectance_noise

    image = np.full((6, 64, 64), 5000.0, dtype=np.float32)
    image[0, 0, 0] = -9999.0  # no-data fill
    image[1, :, :] = 0.0  # dark band, clipping must keep it non-negative
    noisy = add_reflectance_noise(image, 0.01, np.random.default_rng(0), reflectance_scale=10000.0)

    assert noisy[0, 0, 0] == -9999.0
    assert noisy[1].min() >= 0.0
    np.testing.assert_allclose(np.std(noisy[2] - image[2]), 100.0, rtol=0.05)  # 0.01 reflectance x 10000


def test_prediction_change_counts_flips_and_ignores_missing_pixels():
    from eouq.layers.l0_input import prediction_change

    clean = np.array([[[0.4, 0.6], [0.9, 0.1]]])
    noisy = np.array([[[0.6, 0.6], [0.0, 0.1]]])
    masks = np.array([[[0, 1], [-1, 0]]])  # the large change at [1, 0] is a missing pixel
    mean_abs_change, flip_rate = prediction_change(clean, noisy, masks)

    np.testing.assert_allclose(mean_abs_change, [0.2 / 3])
    np.testing.assert_allclose(flip_rate, [1 / 3])
