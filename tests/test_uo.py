import json

import numpy as np
import pytest

from eouq.core.uo import UncertaintyObject


def test_valid_construction_roundtrips_through_json():
    uo = UncertaintyObject(
        value=np.array([0.1, 0.2, 0.3]),
        type="ood",
        source="l1.kmeans_ncdd",
        scale="chip",
        guarantee=None,
        provenance={"git_hash": "abc123", "seed": 0},
    )
    d = uo.to_dict()
    assert d["value"] == [0.1, 0.2, 0.3]
    assert d["type"] == "ood"
    assert d["source"] == "l1.kmeans_ncdd"
    # must be JSON-serialisable end to end
    reloaded = json.loads(uo.to_json())
    assert reloaded["provenance"]["git_hash"] == "abc123"


def test_scalar_and_nested_dict_values_are_jsonable():
    uo = UncertaintyObject(
        value={"mean": np.float64(0.5), "members": np.array([1, 2, 3])},
        type="epistemic",
        source="l2.ensemble",
        scale="pixel",
    )
    d = uo.to_dict()
    assert d["value"]["mean"] == 0.5
    assert d["value"]["members"] == [1, 2, 3]
    assert isinstance(d["value"]["mean"], float)


@pytest.mark.parametrize(
    "field,bad_value",
    [("type", "not_a_type"), ("source", "l9.bogus"), ("scale", "continent")],
)
def test_invalid_fields_raise(field, bad_value):
    kwargs = dict(value=1.0, type="risk", source="l3.split_conformal", scale="region")
    kwargs[field] = bad_value
    with pytest.raises(ValueError):
        UncertaintyObject(**kwargs)


def test_source_accepts_dotted_method_suffix():
    uo = UncertaintyObject(value=1.0, type="coverage", source="l3.mondrian", scale="region")
    assert uo.source == "l3.mondrian"
