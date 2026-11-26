"""L0 sensor-noise experiment, CPU stage: analyse the per-chip statistics
written by scripts/extract_sensor_noise.py against each model's clean cache.

Three questions:
  1. Degradation: as input noise grows, how much does segmentation accuracy
     fall, and do the U_T / U_E uncertainty signals rise with it?
  2. Robustness of the failure detector: does the fusion model fitted on
     CLEAN train+val signals still rank failures correctly when the test
     inputs are noisy (it is never refitted on noisy data)?
  3. Noise sensitivity as a new L0 signal: does the per-chip prediction
     change under small noise predict failure on clean test data, alone and
     added to the U_T + U_E fusion, and is it redundant with U_T?

All test-set scoring uses chips whose F1 is defined for every model and at
every noise level, so levels and models are compared on identical chips.

Not runnable in this sandbox -- run on the cluster (CPU only):
  python -m scripts.evaluate_sensor_noise --dataset sen1floods11
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

from eouq.fusion.tree_fusion import fit_tree_fusion, predict_risk
from scripts.evaluate_sen1floods11 import (
    build_feature_table,
    compute_l1_signals,
    compute_l2_signals,
    evaluate_risk_scores,
    load_cache,
)

TAU = 0.6  # failure threshold, same for burn scars and floods (SHRUG-FM)
L0_SIGNALS = {"noise_sensitivity": "mean_abs_change", "noise_flip_rate": "flip_rate"}
MODELS = {"burn_scars": ["ssl4eo"], "sen1floods11": ["prithvi", "unet"]}


def clean_cache_dir(dataset: str, model: str) -> str:
    return "/outputs/burn_scars_cache" if dataset == "burn_scars" else f"/outputs/sen1floods11_cache_{model}"


def load_noise_dir(noise_dir: str) -> tuple:
    """Returns ({noise_std: sweep stats}, {split: sensitivity stats})."""
    sweep = {}
    for path in glob.glob(os.path.join(noise_dir, "sweep_std*.npz")):
        stats = load_cache(path)
        sweep[float(stats["noise_std"])] = stats
    sensitivity = {split: load_cache(os.path.join(noise_dir, f"sensitivity_{split}.npz")) for split in ("train", "val", "test")}
    return dict(sorted(sweep.items())), sensitivity


def valid_chip_ids(clean_test: dict, sweep: dict) -> set:
    """Test chips whose F1 is defined on clean data and at every noise level."""
    valid = {str(c) for c, f in zip(clean_test["chip_id"], clean_test["f1"]) if not np.isnan(f)}
    for stats in sweep.values():
        valid &= {str(c) for c, f in zip(stats["chip_id"], stats["f1"]) if not np.isnan(f)}
    return valid


def _check_aligned(chip_ids_a, chip_ids_b, what: str) -> None:
    if [str(c) for c in chip_ids_a] != [str(c) for c in chip_ids_b]:
        raise ValueError(f"chip order of {what} does not match the clean cache")


def _signals(split: dict, reference_embeddings) -> dict:
    """U_T (+ U_E if embeddings are available) per-chip signals for a clean cache or a sweep level."""
    if "probs" in split:
        signals = compute_l2_signals(split["probs"])
    else:
        signals = {"roi_entropy": split["roi_entropy"], "roi_mutual_information": split["roi_mutual_information"]}
    if reference_embeddings is not None and "embeddings" in split:
        signals.update(compute_l1_signals(split["embeddings"], reference_embeddings))
    return signals


def _score(scores: np.ndarray, name: str) -> np.ndarray:
    return -scores if name == "ncdd" else scores  # low NCDD = out of distribution = high risk


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def pooled_f1(stats: dict, mask: np.ndarray) -> float:
    tp, fp, fn = (float(stats[k][mask].sum()) for k in ("tp", "fp", "fn"))
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else float("nan")


def run_sensor_noise_evaluation(clean: dict, sweep: dict, sensitivity: dict, eval_chip_ids: set, tau: float = TAU) -> dict:
    """clean: {split: clean cache dict}; sweep: {noise_std: sweep stats}; sensitivity: {split: stats}."""
    for split in ("train", "val", "test"):
        _check_aligned(sensitivity[split]["chip_id"], clean[split]["chip_id"], f"sensitivity_{split}")
    for noise_std, stats in sweep.items():
        _check_aligned(stats["chip_id"], clean["test"]["chip_id"], f"sweep std={noise_std:g}")

    reference_embeddings = clean["train"].get("embeddings")
    clean_signals = {split: _signals(clean[split], reference_embeddings) for split in ("train", "val", "test")}
    l0_signals = {
        split: {name: sensitivity[split][key] for name, key in L0_SIGNALS.items()} for split in ("train", "val", "test")
    }

    fusion_f1 = np.concatenate([clean["train"]["f1"], clean["val"]["f1"]])
    fusion_valid = ~np.isnan(fusion_f1)

    def fit(with_l0: bool):
        tables = [build_feature_table({**clean_signals[s], **(l0_signals[s] if with_l0 else {})}) for s in ("train", "val")]
        features = np.concatenate([tables[0][0], tables[1][0]])
        return fit_tree_fusion(features[fusion_valid], fusion_f1[fusion_valid], tau=tau), tables[0][1]

    base_model, base_names = fit(with_l0=False)
    l0_model, l0_names = fit(with_l0=True)

    test_mask = np.array([str(c) in eval_chip_ids for c in clean["test"]["chip_id"]])
    clean_test_f1 = clean["test"]["f1"][test_mask]
    rng = np.random.default_rng(0)
    random_scores = rng.random(int(test_mask.sum()))

    # 3. Noise sensitivity as a signal, on clean test data.
    base_test, _ = build_feature_table(clean_signals["test"])
    l0_test, _ = build_feature_table({**clean_signals["test"], **l0_signals["test"]})
    signal_results = {
        "noise_std": float(sensitivity["test"]["noise_std"]),
        "n_draws": int(sensitivity["test"]["n_draws"]),
        "fused_without_l0": evaluate_risk_scores(predict_risk(base_model, base_test[test_mask]), clean_test_f1, tau),
        "fused_with_l0": evaluate_risk_scores(predict_risk(l0_model, l0_test[test_mask]), clean_test_f1, tau),
        "baseline_random": evaluate_risk_scores(random_scores, clean_test_f1, tau),
        "spearman_with_existing_signals": {},
    }
    for name in L0_SIGNALS:
        values = l0_signals["test"][name][test_mask]
        signal_results[f"baseline_{name}"] = evaluate_risk_scores(values, clean_test_f1, tau)
        signal_results["spearman_with_existing_signals"][name] = {
            other: _spearman(values, clean_signals["test"][other][test_mask]) for other in base_names
        }
    for name in base_names:
        scores = _score(clean_signals["test"][name][test_mask], name)
        signal_results[f"baseline_{name}"] = evaluate_risk_scores(scores, clean_test_f1, tau)
    draw_std = sensitivity["test"]["mean_abs_change_draw_std"][test_mask]
    signal_results["draw_to_draw_cv"] = float(np.mean(draw_std / np.maximum(l0_signals["test"]["noise_sensitivity"][test_mask], 1e-12)))

    # 1 + 2. Degradation sweep, scored with the clean-fitted fusion model.
    clean_test_entropy = clean_signals["test"]["roi_entropy"][test_mask]
    clean_fused = predict_risk(base_model, base_test[test_mask])
    sweep_results = []
    for noise_std, stats in sweep.items():
        signals = _signals(stats, reference_embeddings)
        features, names = build_feature_table(signals)
        if names != base_names:
            raise ValueError(f"sweep signals {names} differ from clean signals {base_names}")
        f1 = stats["f1"][test_mask]
        fused = predict_risk(base_model, features[test_mask])
        level = {
            "noise_std": noise_std,
            "pooled_f1": pooled_f1(stats, test_mask),
            "mean_chip_f1": float(np.mean(f1)),
            "failure_rate": float(np.mean(f1 < tau)),
            "mean_signal": {name: float(np.mean(signals[name][test_mask])) for name in base_names},
            "mean_abs_change": float(np.mean(stats["mean_abs_change"][test_mask])),
            "flip_rate": float(np.mean(stats["flip_rate"][test_mask])),
            "mean_fused_risk": float(np.mean(fused)),
            "fused": evaluate_risk_scores(fused, f1, tau),
            "baseline_random": evaluate_risk_scores(random_scores, f1, tau),
        }
        for name in base_names:
            level[f"baseline_{name}"] = evaluate_risk_scores(_score(signals[name][test_mask], name), f1, tau)
        if noise_std > 0:
            f1_drop = clean_test_f1 - f1
            level["spearman_f1_drop_vs_entropy_rise"] = _spearman(f1_drop, signals["roi_entropy"][test_mask] - clean_test_entropy)
            level["spearman_f1_drop_vs_fused_risk_rise"] = _spearman(f1_drop, fused - clean_fused)
            level["spearman_f1_drop_vs_flip_rate"] = _spearman(f1_drop, stats["flip_rate"][test_mask])
        sweep_results.append(level)

    return {
        "n_test_chips": int(test_mask.sum()),
        "feature_names": base_names,
        "feature_names_with_l0": l0_names,
        "sensitivity_signal": signal_results,
        "degradation_sweep": sweep_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(MODELS), required=True)
    parser.add_argument("--output", default=None, help="defaults to /outputs/sensor_noise_evaluation_<dataset>.json")
    args = parser.parse_args()

    noise_data = {}
    for model in MODELS[args.dataset]:
        noise_dir = f"/outputs/sensor_noise_{args.dataset}_{model}"
        if not os.path.exists(os.path.join(noise_dir, "sensitivity_test.npz")):
            print(f"skipping {model}: no sensor-noise outputs in {noise_dir}")
            continue
        noise_data[model] = load_noise_dir(noise_dir)

    common = None
    for model, (sweep, _) in noise_data.items():
        with np.load(os.path.join(clean_cache_dir(args.dataset, model), "test.npz"), allow_pickle=True) as data:
            clean_test = {"chip_id": data["chip_id"], "f1": data["f1"]}
        valid = valid_chip_ids(clean_test, sweep)
        common = valid if common is None else common & valid
    common = common or set()
    print(f"scoring on the {len(common)} test chips valid for every model and noise level")

    results = {"dataset": args.dataset, "tau": TAU, "n_common_test_chips": len(common)}
    for model, (sweep, sensitivity) in noise_data.items():
        cache_dir = clean_cache_dir(args.dataset, model)
        clean = {split: load_cache(os.path.join(cache_dir, f"{split}.npz")) for split in ("train", "val", "test")}
        results[model] = run_sensor_noise_evaluation(clean, sweep, sensitivity, common)
        del clean  # full-resolution clean caches are several GB per model

    output = args.output or f"/outputs/sensor_noise_evaluation_{args.dataset}.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
