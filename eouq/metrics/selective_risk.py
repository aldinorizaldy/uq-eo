"""Selective-prediction metrics: risk-coverage curve, AURC, discard-performance
curve / AUDP, Risk@coverage. Ties in the risk score are broken by averaging
over repeated random permutations, per SHRUG-FM.
"""
from __future__ import annotations

import numpy as np


def risk_coverage_curve(risk_scores: np.ndarray, losses: np.ndarray, n_tie_permutations: int = 100, seed: int = 0):
    """Sort by ascending risk_scores (random tie-break, averaged over n_tie_permutations),
    return (coverage, cumulative_mean_loss) arrays.

    At coverage c = i/n, the accepted set is the i lowest-risk samples; risk(c) is
    their mean loss (standard selective-classification risk-coverage curve).
    """
    n = len(risk_scores)
    rng = np.random.default_rng(seed)
    coverage = np.arange(1, n + 1) / n
    risk_accum = np.zeros(n)
    for _ in range(n_tie_permutations):
        tie_break = rng.random(n)
        order = np.lexsort((tie_break, risk_scores))
        cum_loss = np.cumsum(losses[order]) / np.arange(1, n + 1)
        risk_accum += cum_loss
    return coverage, risk_accum / n_tie_permutations


def aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    order = np.argsort(coverage)
    return float(np.trapz(risk[order], coverage[order]))


def risk_at_coverage(coverage: np.ndarray, risk: np.ndarray, target_coverage: float) -> float:
    order = np.argsort(coverage)
    return float(np.interp(target_coverage, coverage[order], risk[order]))


def discard_performance_curve(risk_scores: np.ndarray, metric_values: np.ndarray, n_tie_permutations: int = 100, seed: int = 0):
    """Sort by descending risk_scores (discard most-uncertain first), return (fraction_discarded, mean_metric).

    fraction_discarded ranges over 0, 1/n, ..., (n-1)/n (at least one sample always
    remains). mean_metric[d] is the mean metric_values over the n - d retained,
    least-uncertain samples.
    """
    n = len(risk_scores)
    rng = np.random.default_rng(seed)
    fraction_discarded = np.arange(n) / n
    perf_accum = np.zeros(n)
    for _ in range(n_tie_permutations):
        tie_break = rng.random(n)
        order = np.lexsort((tie_break, -risk_scores))
        sorted_metric = metric_values[order]
        suffix_sum = np.cumsum(sorted_metric[::-1])[::-1]
        remaining_count = np.arange(n, 0, -1)
        perf_accum += suffix_sum / remaining_count
    return fraction_discarded, perf_accum / n_tie_permutations


def audp(fraction_discarded: np.ndarray, metric_values: np.ndarray) -> float:
    order = np.argsort(fraction_discarded)
    return float(np.trapz(metric_values[order], fraction_discarded[order]))
