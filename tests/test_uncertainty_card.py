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


SHIFT_RESULTS = {
    "alpha_coverage_target": 0.9,
    "calibrated_on_other_continents": {
        "coverage": {"overall": 0.80, "per_group": {"Ghana": 0.85, "Somalia": 0.70}},
        "event_coverage_disparity": 0.15,
    },
    "calibrated_in_distribution": {
        "coverage": {"overall": 0.90, "per_group": {"Ghana": 0.92, "Somalia": 0.88}},
        "event_coverage_disparity": 0.04,
    },
    "conformal_risk_control": {
        "alpha_fnr_target": 0.2,
        "calibrated_on_other_continents": {"lambda": 0.3, "mean_fnr_on_eval": 0.25},
        "calibrated_in_distribution": {"lambda": 0.25, "mean_fnr_on_eval": 0.18},
    },
}


def test_build_card_deferral_guidance_and_random_excluded_from_attribution():
    card = build_card(SELECTIVE_RISK_RESULTS)
    assert "random" not in card["signal_attribution"]["baseline_aurc_by_signal"]
    assert card["deferral_guidance"]["base_failure_rate"] == 0.28
    assert card["deferral_guidance"]["failure_rate_if_top_50pct_accepted"] == 0.05
    assert "Recommendation" in render_markdown(card)


def test_build_card_and_render_coverage_under_shift_with_nested_risk_control():
    card = build_card(SHIFT_RESULTS)
    assert card["coverage_under_shift"]["calibrated_on_other_continents"]["empirical_coverage_overall"] == 0.80
    md = render_markdown(card, title="Flood mapping")
    assert md.startswith("# Flood mapping")
    assert "Coverage under geographic shift" in md
    assert "calibrated_on_other_continents: lambda = 0.3" in md


def test_build_card_and_render_area_totals():
    card = build_card(
        {
            "l4_aggregation": {
                "district_cell_deg": 2.0,
                "n_eval_districts": 12,
                "district_conformal_coverage": 0.92,
                "district_conformal_half_width_pixels": 500.0,
                "mean_true_district_total_pixels": 3000.0,
                "grand_total_predicted_pixels": 40000.0,
                "grand_total_true_pixels": 42000.0,
                "grand_total_bootstrap_interval": [30000.0, 50000.0],
                "grand_total_true_inside_bootstrap_interval": True,
            }
        }
    )
    md = render_markdown(card)
    assert "Area totals (L4)" in md
    assert "true total inside: True" in md


def test_merge_sources_combines_files_and_skips_missing(tmp_path):
    import json

    from scripts.build_uncertainty_cards import merge_sources

    (tmp_path / "a.json").write_text(json.dumps({"prithvi": {"fused": {"aurc": 0.2}}, "unet": {"x": 1}}))
    (tmp_path / "b.json").write_text(json.dumps({"prithvi": {"alpha_coverage_target": 0.9}}))
    merged = merge_sources(str(tmp_path), [("a.json", "prithvi"), ("b.json", "prithvi"), ("missing.json", None)])
    assert merged == {"fused": {"aurc": 0.2}, "alpha_coverage_target": 0.9}
