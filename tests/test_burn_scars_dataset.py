import pytest

from data.burn_scars import SPLIT_SIZES, split_chip_ids


def _synthetic_chip_ids(n):
    return [("training", f"chip_{i:04d}") for i in range(n)]


def test_split_chip_ids_sizes_and_disjoint():
    chip_ids = _synthetic_chip_ids(804)
    splits = split_chip_ids(chip_ids, seed=0)

    assert len(splits["train"]) == SPLIT_SIZES["train"]
    assert len(splits["val"]) == SPLIT_SIZES["val"]
    assert len(splits["test"]) == SPLIT_SIZES["test"]

    train_set, val_set, test_set = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    all_assigned = train_set | val_set | test_set
    assert all_assigned.issubset(set(chip_ids))
    assert len(all_assigned) == sum(SPLIT_SIZES.values())  # 802 of 804 used


def test_split_chip_ids_deterministic_given_same_seed():
    chip_ids = _synthetic_chip_ids(804)
    splits_a = split_chip_ids(chip_ids, seed=0)
    splits_b = split_chip_ids(chip_ids, seed=0)
    assert splits_a == splits_b


def test_split_chip_ids_different_seeds_give_different_splits():
    chip_ids = _synthetic_chip_ids(804)
    splits_a = split_chip_ids(chip_ids, seed=0)
    splits_b = split_chip_ids(chip_ids, seed=1)
    assert splits_a["train"] != splits_b["train"]


def test_split_chip_ids_raises_when_not_enough_chips():
    chip_ids = _synthetic_chip_ids(100)  # well short of the 802 required
    with pytest.raises(ValueError):
        split_chip_ids(chip_ids, seed=0)
