"""Turn a list of mismatches into a risk level and a recommended action.

The scoring is deliberately simple and explainable for a proof-of-concept:
the number of mismatches drives the base risk, with one override for the
classic "low-acuity chart, very large bill" red flag.
"""

# Map each risk level to a recommended workflow action.
RECOMMENDED_ACTIONS = {
    "Low": "Auto-approve",
    "Medium": "Send for human review",
    "High": "Escalate to auditor review",
}

# Headline numeric score (0-100) used for the dashboard metric.
RISK_SCORES = {
    "Low": 15,
    "Medium": 55,
    "High": 90,
}


def score_risk(mismatches, chart_facts, bill_facts):
    """Compute the overall risk for a claim.

    Args:
        mismatches:  list of mismatch dicts from `compare_claim`.
        chart_facts: dict from `extract_chart_facts`.
        bill_facts:  dict from `extract_billing_facts`.

    Returns:
        dict with risk_level, risk_score, recommended_action, and a
        plain-English summary.
    """
    num_mismatches = len(mismatches)

    # --- Base risk from the number of mismatches -------------------------
    if num_mismatches == 0:
        risk_level = "Low"
    elif num_mismatches == 1:
        risk_level = "Medium"
    else:  # 2 or more
        risk_level = "High"

    # --- Override: low-severity chart but a very large claim amount ------
    chart_severity = chart_facts.get("severity", "low")
    claim_amount = bill_facts.get("claim_amount", 0) or 0
    amount_override = chart_severity == "low" and claim_amount > 10000
    if amount_override:
        risk_level = "High"

    risk_score = RISK_SCORES[risk_level]
    recommended_action = RECOMMENDED_ACTIONS[risk_level]

    # --- Build a human-readable summary ----------------------------------
    if num_mismatches == 0:
        summary = (
            "No mismatches were detected between the chart and the bill. "
            "The claim appears consistent and can be auto-approved."
        )
    else:
        fields = ", ".join(m["field"] for m in mismatches)
        plural = "mismatch" if num_mismatches == 1 else "mismatches"
        summary = (
            f"Detected {num_mismatches} {plural} ({fields}). "
        )
        if amount_override:
            summary += (
                f"In addition, a low-severity chart was billed at "
                f"${claim_amount:,.0f}, which pushes this claim to High risk. "
            )
        summary += f"Recommended action: {recommended_action}."

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "recommended_action": recommended_action,
        "summary": summary,
    }
