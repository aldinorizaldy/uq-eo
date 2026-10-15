import numpy as np

from eouq.layers.l1_representation import (
    embedding_ood_score,
    fit_kmeans_prototypes,
    knn_distance,
    mahalanobis_distance,
    ncdd_score,
    nearest_centroid_distance,
    normalized_centroid_distance,
)


def _centroids_1d_on_a_line():
    # 3 centroids at (0,0), (10,0), (20,0)
    return np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])


def test_nearest_centroid_distance_exact_match_is_zero():
    centroids = _centroids_1d_on_a_line()
    query = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 0.0]])
    dist = nearest_centroid_distance(query, centroids)
    np.testing.assert_allclose(dist, [0.0, 0.0, 5.0])


def test_ncdd_score_matches_hand_computed_values():
    centroids = _centroids_1d_on_a_line()
    query = np.array([[0.0, 0.0], [10.0, 0.0]])
    ncdd = ncdd_score(query, centroids)
    # point at (0,0): dists=[0,10,20], own=0, k=3 -> sum(30) - 3*0 = 30
    # point at (10,0): dists=[10,0,10], own=0, k=3 -> sum(20) - 3*0 = 20
    np.testing.assert_allclose(ncdd, [30.0, 20.0])


def test_normalized_centroid_distance_zero_at_centroid():
    centroids = _centroids_1d_on_a_line()
    cluster_mean_distance = np.array([2.0, 5.0, 1.0])
    query = np.array([[0.0, 0.0], [10.0, 0.0]])
    nd = normalized_centroid_distance(query, centroids, cluster_mean_distance)
    np.testing.assert_allclose(nd, [0.0, 0.0])

    query_offset = np.array([[2.0, 0.0]])  # 2 units from centroid 0, mean dist 2.0
    nd_offset = normalized_centroid_distance(query_offset, centroids, cluster_mean_distance)
    np.testing.assert_allclose(nd_offset, [1.0])


def test_knn_distance_mean_of_k_nearest():
    reference = np.arange(10.0).reshape(-1, 1)
    query = np.array([[0.0]])
    dist = knn_distance(query, reference, k=3)
    # nearest 3 reference points to 0 are 0,1,2 -> distances 0,1,2 -> mean 1.0
    np.testing.assert_allclose(dist, [1.0])


def test_mahalanobis_distance_reduces_to_euclidean_with_identity_covariance():
    query = np.array([[3.0, 4.0]])
    reference_mean = np.array([0.0, 0.0])
    reference_cov_inv = np.eye(2)
    dist = mahalanobis_distance(query, reference_mean, reference_cov_inv)
    np.testing.assert_allclose(dist, [5.0])


def test_fit_kmeans_prototypes_shapes_and_no_nan_with_enough_data():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(200, 4)).astype(np.float32)
    fit = fit_kmeans_prototypes(embeddings, k=4, seed=0)
    assert fit["centroids"].shape == (4, 4)
    assert fit["cluster_mean_distance"].shape == (4,)
    assert not np.any(np.isnan(fit["cluster_mean_distance"]))


def test_embedding_ood_score_returns_valid_uncertainty_object():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(200, 4)).astype(np.float32)
    query = rng.normal(size=(5, 4)).astype(np.float32)
    uo = embedding_ood_score(query, reference, provenance={"seed": 0}, k=4, seed=0)
    assert uo.type == "ood"
    assert uo.source == "l1.kmeans_ncdd"
    assert uo.value["normalized_centroid_distance"].shape == (5,)
    assert uo.value["ncdd"].shape == (5,)
