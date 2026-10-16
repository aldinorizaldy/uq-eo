import numpy as np

from scripts.extract_burn_scars_cache import per_chip_f1


def test_per_chip_f1_matches_hand_computed_values_and_excludes_missing_pixels():
    # 2 samples, 2 classes, 2x2 pixels. mean_probs[:, -1] is the event-class prob.
    mean_probs = np.array(
        [
            [[[0.1, 0.9], [0.1, 0.9]], [[0.9, 0.1], [0.9, 0.1]]],  # sample 0: class0, class1
            [[[0.8, 0.2], [0.5, 0.5]], [[0.2, 0.8], [0.5, 0.5]]],  # sample 1: class0, class1
        ]
    )
    masks = np.array(
        [
            [[1, 0], [1, 1]],  # sample 0, no missing pixels
            [[-1, 1], [0, 1]],  # sample 1, (0,0) is missing data
        ]
    )

    f1_scores = per_chip_f1(mean_probs, masks, prob_threshold=0.5)

    # sample 0: event_prob=[[0.9,0.1],[0.9,0.1]] -> predicted=[[1,0],[1,0]]
    #   tp=2 ((0,0),(1,0)), fp=0, fn=1 ((1,1)) -> f1 = 4/(2*2+0+1) = 0.8
    # sample 1: event_prob=[[0.2,0.8],[0.5,0.5]] -> predicted=[[0,1],[1,1]]
    #   (0,0) excluded (missing); tp=2 ((0,1),(1,1)), fp=1 ((1,0)), fn=0 -> f1 = 4/5 = 0.8
    np.testing.assert_allclose(f1_scores, [0.8, 0.8])


def test_per_chip_f1_is_nan_when_no_positives_predicted_or_true():
    mean_probs = np.zeros((1, 2, 2, 2))
    mean_probs[:, 0, :, :] = 1.0  # class0 prob=1 everywhere -> event_prob (class1) = 0
    masks = np.zeros((1, 2, 2), dtype=np.int64)  # all negatives, no missing pixels

    f1_scores = per_chip_f1(mean_probs, masks, prob_threshold=0.5)
    assert np.isnan(f1_scores[0])
