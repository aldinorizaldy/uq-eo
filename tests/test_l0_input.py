import numpy as np

from eouq.layers.l0_input import covariate_percentile_rank, input_ood_score, spatial_kde_density


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
