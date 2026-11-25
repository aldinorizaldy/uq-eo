import numpy as np

from scripts.diagnose_sen1floods11_f1 import pixel_counts, summarize_split


def test_pixel_counts_ignore_missing_pixels():
    event_prob = np.array([[[0.9, 0.9], [0.1, 0.1]]])
    masks = np.array([[[1, -1], [1, 0]]])  # (0,1) missing -> ignored even though predicted positive
    tp, fp, fn, n_water, n_valid = pixel_counts(event_prob, masks)
    assert (tp[0], fp[0], fn[0], n_water[0], n_valid[0]) == (1, 0, 1, 2, 3)


def test_pooled_f1_exceeds_per_chip_mean_when_a_low_water_chip_fails():
    # chip 0: 50/100 water, predicted perfectly -> F1 = 1.0
    # chip 1: 1/100 water, model predicts 3 wrong pixels and misses the water -> F1 = 0.0
    h = w = 10
    masks = np.zeros((2, h, w), dtype=np.int64)
    masks[0, :5, :] = 1
    masks[1, 0, 0] = 1
    event_prob = np.zeros((2, h, w))
    event_prob[0, :5, :] = 1.0
    event_prob[1, 9, 7:] = 1.0

    summary = summarize_split(event_prob, masks)
    assert summary["per_chip_mean_f1"] == 0.5
    # pooled: tp=50, fp=3, fn=1 -> 100 / 104
    np.testing.assert_allclose(summary["pooled_f1"], 100 / 104)
    assert summary["per_chip_f1_by_water_fraction"]["<1% water"]["n_chips"] == 0  # 1% sits in the 1-10% bin
    assert summary["per_chip_f1_by_water_fraction"]["1-10% water"]["mean_per_chip_f1"] == 0.0
    assert summary["per_chip_f1_by_water_fraction"][">10% water"]["mean_per_chip_f1"] == 1.0
