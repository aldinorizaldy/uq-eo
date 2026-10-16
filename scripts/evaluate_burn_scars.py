"""Evaluate the SHRUG-FM-style UQ pipeline on cached burn-scars ensemble
outputs: U_T (L2) + U_E (L1) signals, tree fusion, risk-coverage/AURC/AUDP,
compared against single-score baselines and a random-discard baseline.

Two known gaps vs. the full paper design, see NOTES.md:
  - U_I (L0) is not included — HydroATLAS/DEM covariates aren't sourced yet.
  - The U_E reference/pretraining-proxy set is narrower than planned: just
    the burn-scars train split's own embeddings, since floods/landslides
    aren't built yet (the NOTES.md "pretraining reference set" decision
    was to pool all three downstream train splits).

Not runnable in this sandbox — run on the cluster (CPU-only, no GPU needed):
  apptainer exec --bind $PWD:/workspace --bind $PWD/outputs:/outputs \\
      container/eouq.sif python -m scripts.evaluate_burn_scars
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

TAU = 0.6  # burn scars failure threshold, per SHRUG-FM
ROI_PROB_THRESHOLD = 0.5


def load_cache(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def compute_l2_signals(probs: np.ndarray) -> dict:
    """probs: (n_members, n_samples, n_classes, H, W) -> per-chip roi_entropy, roi_mi.

    NaN ROI (model predicted no event pixels in that chip) is treated as zero
    uncertainty contribution — nothing predicted burned, nothing to be
    uncertain about. A modeling choice, not from the paper.
    """
    member_probs = np.moveaxis(probs, 2, -1)  # (n_members, n_samples, H, W, n_classes)
    n_samples = member_probs.shape[1]

    roi_entropy = np.empty(n_samples)
    roi_mi = np.empty(n_samples)
    for i in range(n_samples):
        chip_member_probs = member_probs[:, i]  # (n_members, H, W, n_classes)
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
    k = max(2, min(k, len(reference_embeddings) // 2))
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
    # U_E reference proxy: burn-scars train split's own embeddings (see
    # module docstring — narrower than the planned pooled-dataset proxy).
    reference_embeddings = train["embeddings"]

    l2_train, l2_val, l2_test = (compute_l2_signals(split["probs"]) for split in (train, val, test))
    l1_train, l1_val, l1_test = (
        compute_l1_signals(split["embeddings"], reference_embeddings) for split in (train, val, test)
    )

    features_train, feature_names = build_feature_table({**l2_train, **l1_train})
    features_val, _ = build_feature_table({**l2_val, **l1_val})
    features_test, _ = build_feature_table({**l2_test, **l1_test})

    # Fusion trained on train+val only, per SHRUG-FM.
    fusion_features = np.concatenate([features_train, features_val], axis=0)
    fusion_f1 = np.concatenate([train["f1"], val["f1"]], axis=0)
    fusion_valid = ~np.isnan(fusion_f1)
    model = fit_tree_fusion(fusion_features[fusion_valid], fusion_f1[fusion_valid], tau=tau)

    test_f1 = test["f1"]
    test_valid = ~np.isnan(test_f1)
    fused_risk = predict_risk(model, features_test[test_valid])

    results = {"feature_names": feature_names, "n_test_chips": int(test_valid.sum())}
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
    parser.add_argument("--cache-dir", default="/outputs/burn_scars_cache")
    parser.add_argument("--output", default="/outputs/burn_scars_evaluation.json")
    args = parser.parse_args()

    train = load_cache(os.path.join(args.cache_dir, "train.npz"))
    val = load_cache(os.path.join(args.cache_dir, "val.npz"))
    test = load_cache(os.path.join(args.cache_dir, "test.npz"))

    results = run_evaluation(train, val, test)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print("\nPaper target (burn scars): Risk@0.5 = 0.090 +/- 0.038, best single-score baseline ~= 0.137")


if __name__ == "__main__":
    main()
