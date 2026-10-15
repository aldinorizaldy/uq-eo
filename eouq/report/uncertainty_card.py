"""The Uncertainty Card: a per-run markdown + JSON report summarising coverage,
per-region coverage disparity, per-source attribution of UncertaintyObjects,
and the abstain/defer decision, for the L5 decision & governance layer."""
from __future__ import annotations


def build_card(run_results: dict) -> dict:
    """Assemble the JSON-serialisable card content from a run's cached metrics/UncertaintyObjects."""
    raise NotImplementedError


def render_markdown(card: dict) -> str:
    raise NotImplementedError
