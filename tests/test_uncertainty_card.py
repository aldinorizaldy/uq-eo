from eouq.report.uncertainty_card import build_card, render_markdown

SELECTIVE_RISK_RESULTS = {
    "feature_names": ["roi_entropy", "ncdd"],
    "fused": {"aurc": 0.10, "risk_at_0.5": 0.05, "risk_at_0.75": 0.08, "audp": 0.75},
    "baseline_roi_entropy": {"aurc": 0.21, "risk_at_0.5": 0.10, "risk_at_0.75": 0.14, "audp": 0.67},
    "baseline_ncdd": {"aurc": 0.39, "risk_at_0.5": 0.38, "risk_at_0.75": 0.31, "audp": 0.58},
    "baseline_random": {"aurc": 0.30, "risk_at_0.5": 0.28, "risk_at_0.75": 0.29, "audp": 0.64},
}

CONFORMAL_RESULTS = {
    "alpha_coverage_target": 0.9,
    "split_conformal": {
        "threshold": 0.5,
        "coverage": {"overall": 0.91, "per_group": {"NE": 0.95, "SW": 0.85}},
        "coverage_disparity": 0.1,
    },
    "conformal_risk_control": {"alpha_fnr_target": 0.2, "selected_lambda": 0.3, "mean_fnr_on_eval": 0.18},
}


def test_build_card_selective_risk_summary_and_attribution():
    card = build_card(SELECTIVE_RISK_RESULTS)

    assert card["selective_risk_summary"]["aurc"] == 0.10
    assert card["signal_attribution"]["best_single_signal"] == "roi_entropy"  # lowest baseline AURC (0.21)
    assert card["signal_attribution"]["fusion_improves_over_best_single_signal"] is True


def test_build_card_coverage_guarantees_and_risk_control():
    card = build_card(CONFORMAL_RESULTS)

    assert card["coverage_guarantees"]["split_conformal"]["empirical_coverage_overall"] == 0.91
    assert card["coverage_guarantees"]["split_conformal"]["coverage_disparity"] == 0.1
    assert card["risk_control"]["mean_fnr_on_eval"] == 0.18


def test_build_card_empty_input_produces_empty_card():
    assert build_card({}) == {}


def test_render_markdown_includes_expected_sections():
    md_selective = render_markdown(build_card(SELECTIVE_RISK_RESULTS))
    assert "Signal attribution" in md_selective
    assert "roi_entropy" in md_selective

    md_conformal = render_markdown(build_card(CONFORMAL_RESULTS))
    assert "Coverage guarantees" in md_conformal
    assert "Conformal risk control" in md_conformal


def test_render_markdown_handles_empty_card():
    md = render_markdown({})
    assert "No results available" in md
