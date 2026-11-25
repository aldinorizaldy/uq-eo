"""The Uncertainty Card: a per-run markdown + JSON report summarising coverage,
per-region coverage disparity, per-source attribution of UncertaintyObjects,
and the abstain/defer decision, for the L5 decision & governance layer.

`run_results` is the JSON dict written by any of the evaluation scripts, or
several of them merged into one dict (scripts/build_uncertainty_cards.py
does this per experiment). Recognised sections, each optional:
  - selective risk: `fused` + `baseline_*` (scripts/evaluate_*.py)
  - coverage: `split_conformal` / `mondrian_conformal` / `conformal_fusion` /
    `localized_conformal` (scripts/evaluate_burn_scars_conformal.py)
  - area totals: `l4_aggregation`
  - coverage under geographic shift: `calibrated_on_other_continents` +
    `calibrated_in_distribution` (scripts/evaluate_sen1floods11_conformal.py)
  - risk control: `conformal_risk_control`, in either the single-calibration
    form or the per-calibration-source form used by the shift evaluation
Whatever a run doesn't have is omitted rather than raising.
"""
from __future__ import annotations

COVERAGE_METHODS = ("split_conformal", "mondrian_conformal", "conformal_fusion", "localized_conformal")
SHIFT_KEYS = ("calibrated_on_other_continents", "calibrated_in_distribution")


def _fmt(value, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else str(value)


def build_card(run_results: dict) -> dict:
    """Assemble the JSON-serialisable card content from a run's cached metrics/UncertaintyObjects."""
    card = {}

    if "fused" in run_results:
        fused = run_results["fused"]
        card["selective_risk_summary"] = {
            "aurc": fused.get("aurc"),
            "risk_at_0.5": fused.get("risk_at_0.5"),
            "risk_at_0.75": fused.get("risk_at_0.75"),
            "audp": fused.get("audp"),
        }
        baseline_aurcs = {
            key[len("baseline_") :]: value["aurc"]
            for key, value in run_results.items()
            if key.startswith("baseline_") and key != "baseline_random" and isinstance(value, dict) and "aurc" in value
        }
        if baseline_aurcs:
            best_baseline = min(baseline_aurcs, key=baseline_aurcs.get)
            card["signal_attribution"] = {
                "baseline_aurc_by_signal": baseline_aurcs,
                "best_single_signal": best_baseline,
                "fusion_improves_over_best_single_signal": fused["aurc"] < baseline_aurcs[best_baseline],
            }
        random = run_results.get("baseline_random")
        if isinstance(random, dict) and fused.get("risk_at_0.5") is not None:
            # L5 deferral guidance: what accepting only the most-confident
            # fraction buys, relative to accepting everything blindly (the
            # random-discard risk is ~ the base failure rate at any coverage).
            card["deferral_guidance"] = {
                "base_failure_rate": random.get("risk_at_0.5"),
                "failure_rate_if_top_50pct_accepted": fused.get("risk_at_0.5"),
                "failure_rate_if_top_75pct_accepted": fused.get("risk_at_0.75"),
            }

    for method in COVERAGE_METHODS:
        if method in run_results:
            entry = run_results[method]
            card.setdefault("coverage_guarantees", {})[method] = {
                "target_coverage": run_results.get("alpha_coverage_target"),
                "empirical_coverage_overall": entry["coverage"]["overall"],
                "empirical_coverage_per_region": entry["coverage"].get("per_group"),
                "coverage_disparity": entry.get("coverage_disparity"),
            }

    if all(key in run_results for key in SHIFT_KEYS):
        card["coverage_under_shift"] = {"target_coverage": run_results.get("alpha_coverage_target")}
        for key in SHIFT_KEYS:
            entry = run_results[key]
            card["coverage_under_shift"][key] = {
                "empirical_coverage_overall": entry["coverage"]["overall"],
                "empirical_coverage_per_event": entry["coverage"].get("per_group"),
                "event_coverage_disparity": entry.get("event_coverage_disparity"),
            }

    if "l4_aggregation" in run_results:
        card["area_totals"] = run_results["l4_aggregation"]

    if "conformal_risk_control" in run_results:
        card["risk_control"] = run_results["conformal_risk_control"]

    return card


def _render_risk_control(rc: dict) -> list:
    lines = ["## Conformal risk control", f"- Target: pixel false-negative rate <= {rc.get('alpha_fnr_target')}"]
    if "selected_lambda" in rc:
        lines.append(f"- Selected threshold (lambda): {rc.get('selected_lambda')}")
        lines.append(f"- Mean FNR on evaluation set: {_fmt(rc.get('mean_fnr_on_eval'))}")
    else:
        for source, entry in rc.items():
            if isinstance(entry, dict):
                lines.append(
                    f"- {source}: lambda = {_fmt(entry.get('lambda'), 2)}, mean FNR on evaluation set = "
                    f"{_fmt(entry.get('mean_fnr_on_eval'))}"
                )
    lines.append("")
    return lines


def render_markdown(card: dict, title: str = "Uncertainty Card") -> str:
    lines = [f"# {title}", ""]

    if "deferral_guidance" in card:
        g = card["deferral_guidance"]
        lines.append("## Recommendation (abstain / defer)")
        lines.append(
            f"- Accepting every prediction: expected failure rate ~{_fmt(g['base_failure_rate'])}."
        )
        lines.append(
            f"- Accepting only the 50% most-confident predictions (deferring the rest to review): "
            f"~{_fmt(g['failure_rate_if_top_50pct_accepted'])}."
        )
        lines.append(f"- Accepting the 75% most-confident: ~{_fmt(g['failure_rate_if_top_75pct_accepted'])}.")
        lines.append("")

    if "selective_risk_summary" in card:
        lines.append("## Selective-risk summary")
        for key, value in card["selective_risk_summary"].items():
            lines.append(f"- **{key}**: {_fmt(value)}")
        lines.append("")

    if "signal_attribution" in card:
        attribution = card["signal_attribution"]
        lines.append("## Signal attribution")
        lines.append(f"- Best single signal: **{attribution['best_single_signal']}**")
        lines.append(
            f"- Fusion improves over best single signal: **{attribution['fusion_improves_over_best_single_signal']}**"
        )
        lines.append("")
        lines.append("| Signal | AURC |")
        lines.append("|---|---|")
        for name, aurc_value in attribution["baseline_aurc_by_signal"].items():
            lines.append(f"| {name} | {aurc_value:.3f} |")
        lines.append("")

    if "coverage_guarantees" in card:
        lines.append("## Coverage guarantees")
        for method, entry in card["coverage_guarantees"].items():
            lines.append(f"### {method}")
            lines.append(f"- Target coverage: {entry['target_coverage']}")
            lines.append(f"- Empirical coverage (overall): {entry['empirical_coverage_overall']:.3f}")
            if entry.get("coverage_disparity") is not None:
                lines.append(f"- Coverage disparity across regions: {entry['coverage_disparity']:.3f}")
            if entry.get("empirical_coverage_per_region"):
                lines.append("")
                lines.append("| Region | Coverage |")
                lines.append("|---|---|")
                for region, cov in entry["empirical_coverage_per_region"].items():
                    lines.append(f"| {region} | {cov:.3f} |")
            lines.append("")

    if "coverage_under_shift" in card:
        shift = card["coverage_under_shift"]
        lines.append("## Coverage under geographic shift")
        lines.append(f"- Target coverage: {shift.get('target_coverage')}")
        lines.append("")
        lines.append("| Calibrated on | Coverage (overall) | Event disparity |")
        lines.append("|---|---|---|")
        for key in SHIFT_KEYS:
            entry = shift[key]
            lines.append(
                f"| {key.replace('calibrated_', '').replace('_', ' ')} | "
                f"{_fmt(entry['empirical_coverage_overall'])} | {_fmt(entry.get('event_coverage_disparity'))} |"
            )
        lines.append("")

    if "area_totals" in card:
        a = card["area_totals"]
        lines.append("## Area totals (L4)")
        lines.append(
            f"- Per-district conformal intervals ({a.get('district_cell_deg')}-degree cells): coverage "
            f"{_fmt(a.get('district_conformal_coverage'))} over {a.get('n_eval_districts')} districts "
            f"(half-width {_fmt(a.get('district_conformal_half_width_pixels'), 0)} px vs. mean district total "
            f"{_fmt(a.get('mean_true_district_total_pixels'), 0)} px)."
        )
        interval = a.get("grand_total_bootstrap_interval") or [None, None]
        lines.append(
            f"- Grand total: predicted {_fmt(a.get('grand_total_predicted_pixels'), 0)} px, true "
            f"{_fmt(a.get('grand_total_true_pixels'), 0)} px, block-bootstrap interval "
            f"[{_fmt(interval[0], 0)}, {_fmt(interval[1], 0)}] -- true total inside: "
            f"{a.get('grand_total_true_inside_bootstrap_interval')}."
        )
        lines.append("")

    if "risk_control" in card:
        lines.extend(_render_risk_control(card["risk_control"]))

    if len(lines) <= 2:
        lines.append("_No results available to summarise._")

    return "\n".join(lines)
