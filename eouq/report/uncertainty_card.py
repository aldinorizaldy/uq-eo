"""The Uncertainty Card: a per-run markdown + JSON report summarising coverage,
per-region coverage disparity, per-source attribution of UncertaintyObjects,
and the abstain/defer decision, for the L5 decision & governance layer.

`run_results` is typically the JSON dict one of the `scripts/evaluate_*.py`
(fused + baseline_* keys) or `scripts/evaluate_*_conformal.py`
(split_conformal/mondrian_conformal/conformal_risk_control keys) scripts
writes out. Not every run has every kind of result -- `build_card` omits
whatever section its input doesn't support rather than raising.
"""
from __future__ import annotations


def build_card(run_results: dict) -> dict:
    """Assemble the JSON-serialisable card content from a run's cached metrics/UncertaintyObjects."""
    card = {}

    if "fused" in run_results:
        card["selective_risk_summary"] = {
            "aurc": run_results["fused"].get("aurc"),
            "risk_at_0.5": run_results["fused"].get("risk_at_0.5"),
            "risk_at_0.75": run_results["fused"].get("risk_at_0.75"),
            "audp": run_results["fused"].get("audp"),
        }
        baseline_aurcs = {
            key[len("baseline_") :]: value["aurc"]
            for key, value in run_results.items()
            if key.startswith("baseline_") and isinstance(value, dict) and "aurc" in value
        }
        if baseline_aurcs:
            best_baseline = min(baseline_aurcs, key=baseline_aurcs.get)
            card["signal_attribution"] = {
                "baseline_aurc_by_signal": baseline_aurcs,
                "best_single_signal": best_baseline,
                "fusion_improves_over_best_single_signal": run_results["fused"]["aurc"] < baseline_aurcs[best_baseline],
            }

    for conformal_key in ("split_conformal", "mondrian_conformal", "conformal_fusion"):
        if conformal_key in run_results:
            entry = run_results[conformal_key]
            card.setdefault("coverage_guarantees", {})[conformal_key] = {
                "target_coverage": run_results.get("alpha_coverage_target"),
                "empirical_coverage_overall": entry["coverage"]["overall"],
                "empirical_coverage_per_region": entry["coverage"].get("per_group"),
                "coverage_disparity": entry.get("coverage_disparity"),
            }

    if "conformal_risk_control" in run_results:
        card["risk_control"] = run_results["conformal_risk_control"]

    return card


def render_markdown(card: dict) -> str:
    lines = ["# Uncertainty Card", ""]

    if "selective_risk_summary" in card:
        lines.append("## Selective-risk summary")
        for key, value in card["selective_risk_summary"].items():
            lines.append(f"- **{key}**: {value}")
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

    if "risk_control" in card:
        rc = card["risk_control"]
        lines.append("## Conformal risk control")
        lines.append(f"- Target: alpha_fnr <= {rc.get('alpha_fnr_target')}")
        lines.append(f"- Selected threshold (lambda): {rc.get('selected_lambda')}")
        lines.append(f"- Mean FNR on evaluation set: {rc.get('mean_fnr_on_eval'):.3f}")
        lines.append("")

    if len(lines) <= 2:
        lines.append("_No results available to summarise._")

    return "\n".join(lines)
