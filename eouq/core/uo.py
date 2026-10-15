"""The UncertaintyObject: the common return type for every UQ layer.

Every layer in eouq (L0..L5) returns one or more UncertaintyObject instances
instead of a bare number, so that downstream fusion, reporting, and the
Uncertainty Card can treat all signals uniformly regardless of which layer
or method produced them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json

import numpy as np

# L0 input, L1 representation, L2 prediction, L3 guarantee, L4 aggregation, L5 decision
VALID_SOURCES = {"l0", "l1", "l2", "l3", "l4", "l5"}
VALID_TYPES = {"aleatoric", "epistemic", "ood", "coverage", "risk", "disparity"}
VALID_SCALES = {"pixel", "chip", "event", "region", "aggregate"}


@dataclass
class UncertaintyObject:
    value: Any  # scalar, np.ndarray, or nested dict of arrays
    type: str  # one of VALID_TYPES
    source: str  # one of VALID_SOURCES, optionally suffixed e.g. "l1.kmeans_ncdd"
    scale: str  # one of VALID_SCALES
    guarantee: Optional[dict] = None  # e.g. {"method": "split_conformal", "alpha": 0.1}
    provenance: dict = field(default_factory=dict)  # model id, config hash, git hash, seed, ...

    def __post_init__(self):
        source_root = self.source.split(".", 1)[0]
        if source_root not in VALID_SOURCES:
            raise ValueError(f"source root must be one of {VALID_SOURCES}, got {self.source!r}")
        if self.type not in VALID_TYPES:
            raise ValueError(f"type must be one of {VALID_TYPES}, got {self.type!r}")
        if self.scale not in VALID_SCALES:
            raise ValueError(f"scale must be one of {VALID_SCALES}, got {self.scale!r}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["value"] = _jsonable(self.value)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
