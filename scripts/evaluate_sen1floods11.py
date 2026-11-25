"""Evaluate the SHRUG-FM-style UQ pipeline on cached Sen1Floods11 outputs,
for both the Prithvi adapter and the U-Net baseline -- the actual
model-agnostic check (milestone 4): only the adapter changed to produce
these caches; everything from here on (U_T/U_E signals, fusion, metrics) is
identical code for both models.

U-Net has no embeddings (no pretrained backbone, see adapters/unet.py), so
its feature table only has U_T signals; Prithvi's has U_T + U_E. This
script handles that asymmetry by building whatever feature table each
cache supports, rather than requiring every signal from every adapter.

Not runnable in this sandbox -- run on the cluster (CPU-only, no GPU
needed, same as evaluate_burn_scars.py):
  apptainer exec --bind $PWD:/workspace --bind $PWD/outputs:/outputs \\
      container/eouq.sif python -m scripts.evaluate_sen1floods11
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from eouq.fusion.tree_fusion import fit_tree_fusion, predict_risk
from eouq.layers.l1_representation import embedding_ood_score
from eouq.layers.l2_prediction import ensemble_mean_prob, mutual_information, predictive_entropy, roi_average
from eouq.metrics.selective_risk import audp, aurc, discard_performance_curve, risk_at_coverage, risk_coverage_curve

TAU = 0.6  # SHRUG-FM's flood failure threshold
ROI_PROB_THRESHOLD = 0.5


def load_cache(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def compute_l2_signals(probs: np.ndarray) -> dict:
    """Same as scripts/evaluate_burn_scars.py's compute_l2_signals."""
    member_probs = np.moveaxis(probs, 2, -1)  # (n_members, n_samples, H, W, n_classes)
    n_samples = member_probs.shape[1]

    roi_entropy = np.empty(n_samples)
    roi_mi = np.empty(n_samples)
    for i in range(n_samples):
        chip_member_probs = member_probs[:, i]
        mean_probs = ensemble_mean_prob(chip_member_probs)
        pred_entropy = predictive_entropy(mean_probs)
        mi = mutual_information(chip_member_probs)
        event_prob = mean_probs[..., -1]
        roi_entropy[i] = roi_average(pred_entropy, event_prob, ROI_PROB_THRESHOLD)
        roi_mi[i] = roi_average(mi, event_prob, ROI_PROB_THRESHOLD)

    return {
        "roi_entropy": np.nan_to_num(roi_entropy, nan=0.0),
        "roi_mutual_information": np.nan_to_num(roi_mi, nan=0.0),
    }


def compute_l1_signals(embeddings: np.ndarray, reference_embeddings: np.ndarray, k: int = 64) -> dict:
    k = max(2, min(k, len(reference_embeddings) // 20))
    uo = embedding_ood_score(embeddings, reference_embeddings, provenance={}, k=k, seed=0)
    return {
        "normalized_centroid_distance": uo.value["normalized_centroid_distance"],
        "ncdd": uo.value["ncdd"],
    }


def build_feature_table(signals: dict) -> tuple:
    names = sorted(signals.keys())
    features = np.stack([signals[name] for name in names], axis=1)
    return features, names


def evaluate_risk_scores(risk_scores: np.ndarray, f1_scores: np.ndarray, tau: float) -> dict:
    y_fail = (f1_scores < tau).astype(float)
    coverage, risk = risk_coverage_curve(risk_scores, y_fail)
    fraction_discarded, performance = discard_performance_curve(risk_scores, f1_scores)
    return {
        "aurc": aurc(coverage, risk),
        "risk_at_0.5": risk_at_coverage(coverage, risk, 0.5),
        "risk_at_0.75": risk_at_coverage(coverage, risk, 0.75),
        "audp": audp(fraction_discarded, performance),
    }


def run_evaluation(train: dict, val: dict, test: dict, tau: float = TAU) -> dict:
    has_embeddings = "embeddings" in train

    l2_train, l2_val, l2_test = (compute_l2_signals(split["probs"]) for split in (train, val, test))

    if has_embeddings:
        reference_embeddings = train["embeddings"]
        l1_train, l1_val, l1_test = (
            compute_l1_signals(split["embeddings"], reference_embeddings) for split in (train, val, test)
        )
    else:
        l1_train, l1_val, l1_test = {}, {}, {}

    features_train, feature_names = build_feature_table({**l2_train, **l1_train})
    features_val, _ = build_feature_table({**l2_val, **l1_val})
    features_test, _ = build_feature_table({**l2_test, **l1_test})

    fusion_features = np.concatenate([features_train, features_val], axis=0)
    fusion_f1 = np.concatenate([train["f1"], val["f1"]], axis=0)
    fusion_valid = ~np.isnan(fusion_f1)
    model = fit_tree_fusion(fusion_features[fusion_valid], fusion_f1[fusion_valid], tau=tau)

    test_f1 = test["f1"]
    test_valid = ~np.isnan(test_f1)
    fused_risk = predict_risk(model, features_test[test_valid])

    results = {
        "feature_names": feature_names,
        "has_embeddings": has_embeddings,
        "n_test_chips": int(test_valid.sum()),
    }
    results["fused"] = evaluate_risk_scores(fused_risk, test_f1[test_valid], tau)

    for idx, name in enumerate(feature_names):
        baseline_scores = features_test[test_valid, idx]
        if name == "ncdd":
            baseline_scores = -baseline_scores  # low/negative NCDD = OOD = high risk
        results[f"baseline_{name}"] = evaluate_risk_scores(baseline_scores, test_f1[test_valid], tau)

    rng = np.random.default_rng(0)
    results["baseline_random"] = evaluate_risk_scores(rng.random(int(test_valid.sum())), test_f1[test_valid], tau)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/outputs/sen1floods11_evaluation.json")
    args = parser.parse_args()

    all_results = {}
    for model_name in ("prithvi", "unet"):
        cache_dir = f"/outputs/sen1floods11_cache_{model_name}"
        if not os.path.exists(os.path.join(cache_dir, "test.npz")):
            print(f"skipping {model_name}: no cache at {cache_dir}")
            continue
        train = load_cache(os.path.join(cache_dir, "train.npz"))
        val = load_cache(os.path.join(cache_dir, "val.npz"))
        test = load_cache(os.path.join(cache_dir, "test.npz"))
        all_results[model_name] = run_evaluation(train, val, test)
        # Free this model's caches before loading the next: 512x512 x 5 members
        # of full-res probs is ~5.6GB per model (job 884137 was OOM-killed at 8G).
        del train, val, test

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
