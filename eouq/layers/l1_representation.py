"""L1 representation layer: is the sample familiar to the model in embedding space?

SHRUG-FM-style signal: FAISS k-means (k=64) on pretraining embeddings, then
normalised nearest-centroid distance and NCDD (nearest centroid distance
deficit) for query embeddings. Also provides kNN distance and Mahalanobis
distance as alternative embedding-OOD scores. Consumes cached embedding
arrays only — model-agnostic.

NCDD follows arXiv:2511.10370 (SHRUG-FM) eq. for U_E^NCDD:
    NCDD(x) = sum_{j != c(x)} d_j(x) - (k - 1) * d_c(x)(x)
            = sum_j d_j(x) - k * d_c(x)(x)
where d_j(x) is the distance from x's embedding to centroid j and c(x) is the
index of the nearest centroid. Low/negative values mean x sits far from its
own cluster relative to the others, i.e. embedding-space OOD.
"""
from __future__ import annotations

import numpy as np

from eouq.core.uo import UncertaintyObject


def _pairwise_distances(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(n_samples, k) Euclidean distance from each embedding to each centroid."""
    diffs = embeddings[:, None, :] - centroids[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def fit_kmeans_prototypes(embeddings: np.ndarray, k: int = 64, seed: int = 0) -> dict:
    """Fit k-means prototypes on pretraining (or reference) embeddings.

    Returns a dict with:
      - "centroids": (k, embed_dim) cluster centroids
      - "cluster_mean_distance": (k,) mean distance of reference points to their
        assigned centroid, used to normalise nearest-centroid distance (U_E^nd).
    """
    import faiss

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    n_dims = embeddings.shape[1]
    kmeans = faiss.Kmeans(n_dims, k, niter=20, seed=seed, verbose=False)
    kmeans.train(embeddings)
    centroids = np.asarray(kmeans.centroids).reshape(k, n_dims)

    dists = _pairwise_distances(embeddings, centroids)
    assignments = np.argmin(dists, axis=1)
    nearest_dist = dists[np.arange(len(embeddings)), assignments]
    cluster_mean_distance = np.full(k, np.nan)
    for c in range(k):
        mask = assignments == c
        if np.any(mask):
            cluster_mean_distance[c] = nearest_dist[mask].mean()

    return {"centroids": centroids, "cluster_mean_distance": cluster_mean_distance}


def nearest_centroid_distance(query_embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Raw (un-normalised) distance from each query embedding to its nearest centroid."""
    return _pairwise_distances(query_embeddings, centroids).min(axis=1)


def normalized_centroid_distance(
    query_embeddings: np.ndarray, centroids: np.ndarray, cluster_mean_distance: np.ndarray
) -> np.ndarray:
    """U_E^nd: nearest-centroid distance normalised by the reference mean distance
    within the assigned cluster: ||e(x) - mu_c(x)|| / d_bar_c(x).

    A near-empty cluster (few reference points, e.g. k too large for a small
    reference set) can have a near-zero mean distance, which would blow this
    ratio up to a huge or infinite value — floor the denominator to avoid that.
    """
    dists = _pairwise_distances(query_embeddings, centroids)
    assignments = np.argmin(dists, axis=1)
    nearest = dists[np.arange(len(query_embeddings)), assignments]
    denom = np.maximum(cluster_mean_distance[assignments], 1e-6)
    return nearest / denom


def ncdd_score(query_embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """NCDD (nearest centroid distance deficit), see module docstring for the formula."""
    dists = _pairwise_distances(query_embeddings, centroids)
    k = centroids.shape[0]
    assignments = np.argmin(dists, axis=1)
    own_dist = dists[np.arange(len(query_embeddings)), assignments]
    return dists.sum(axis=1) - k * own_dist


def knn_distance(query_embeddings: np.ndarray, reference_embeddings: np.ndarray, k: int = 10) -> np.ndarray:
    """Mean distance to the k nearest reference embeddings."""
    dists = _pairwise_distances(query_embeddings, reference_embeddings)
    nearest_k = np.sort(dists, axis=1)[:, :k]
    return nearest_k.mean(axis=1)


def mahalanobis_distance(
    query_embeddings: np.ndarray, reference_mean: np.ndarray, reference_cov_inv: np.ndarray
) -> np.ndarray:
    diff = query_embeddings - reference_mean[None, :]
    left = diff @ reference_cov_inv
    return np.sqrt(np.einsum("ij,ij->i", left, diff))


def embedding_ood_score(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    provenance: dict,
    k: int = 64,
    seed: int = 0,
) -> UncertaintyObject:
    fit = fit_kmeans_prototypes(reference_embeddings, k=k, seed=seed)
    centroids = fit["centroids"]
    nd = normalized_centroid_distance(query_embeddings, centroids, fit["cluster_mean_distance"])
    ncdd = ncdd_score(query_embeddings, centroids)
    value = {"normalized_centroid_distance": nd, "ncdd": ncdd}
    return UncertaintyObject(value=value, type="ood", source="l1.kmeans_ncdd", scale="chip", provenance=provenance)
