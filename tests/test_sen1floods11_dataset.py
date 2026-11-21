from data.sen1floods11 import coords_to_continent, split_chip_ids_by_continent


def test_coords_to_continent_matches_well_known_cities():
    assert coords_to_continent(40.7, -74.0) == "North America"  # New York
    assert coords_to_continent(51.5, -0.1) == "Europe"  # London
    assert coords_to_continent(6.5, 3.4) == "Africa"  # Lagos
    assert coords_to_continent(-33.9, 151.2) == "Oceania"  # Sydney
    assert coords_to_continent(19.1, 72.9) == "Asia"  # Mumbai
    assert coords_to_continent(-34.6, -58.4) == "South America"  # Buenos Aires


def test_coords_to_continent_falls_back_to_unknown():
    assert coords_to_continent(-80.0, 0.0) == "Unknown"  # Antarctica


def test_split_chip_ids_by_continent_holds_out_exactly_the_target_continent():
    chip_coords = [
        ("us_1", 40.7, -74.0),
        ("us_2", 41.0, -75.0),
        ("uk_1", 51.5, -0.1),
        ("uk_2", 52.0, -1.0),
        ("ng_1", 6.5, 3.4),
        ("ng_2", 7.0, 4.0),
        ("ng_3", 8.0, 5.0),
        ("ng_4", 9.0, 6.0),
    ]
    splits = split_chip_ids_by_continent(chip_coords, holdout_continent="Africa", val_fraction=0.25, seed=0)

    assert set(splits["test"]) == {"ng_1", "ng_2", "ng_3", "ng_4"}
    assert set(splits["train"]) | set(splits["val"]) == {"us_1", "us_2", "uk_1", "uk_2"}
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert len(splits["val"]) == 1  # 25% of the 4 non-held-out chips


def test_split_chip_ids_by_continent_deterministic_given_same_seed():
    chip_coords = [("us_1", 40.7, -74.0), ("us_2", 41.0, -75.0), ("us_3", 42.0, -76.0), ("ng_1", 6.5, 3.4)]
    a = split_chip_ids_by_continent(chip_coords, holdout_continent="Africa", seed=0)
    b = split_chip_ids_by_continent(chip_coords, holdout_continent="Africa", seed=0)
    assert a == b
