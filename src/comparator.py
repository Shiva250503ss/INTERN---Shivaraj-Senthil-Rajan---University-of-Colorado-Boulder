"""Compare extracted chart facts against billing facts.

Each comparison that looks suspicious is returned as a "mismatch" dictionary.
Mismatches carry both a plain-English `risk_reason` and the supporting
evidence sentences from each side, so a human reviewer can audit the call.

Values are compared by *concept*, not exact string, so the same fact written
differently on each document — e.g. chart "amoxicillin 500mg" vs bill
"amoxicillin", or chart "upper respiratory infection (URI)" vs bill "URI" — is
treated as a match. This matters especially for the LLM engine, which produces
free-form labels rather than the canonical labels the rule-based engine emits.
"""

import re

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}

# Words that carry no diagnostic meaning when matching concepts.
_STOPWORDS = {"the", "a", "an", "of", "and", "with", "for", "to", "in", "on"}


def _format_list(values):
    """Render a list of items as a comma-separated string (or 'None')."""
    return ", ".join(values) if values else "None"


def _tokens(text):
    """Normalize text to a set of meaningful lowercase tokens.

    Strips dosages (e.g. "500mg"), punctuation, and stopwords so that surface
    differences don't cause false mismatches.
    """
    text = str(text).lower()
    text = re.sub(r"\b\d+\s*(?:mg|ml|mcg|g|units?|tabs?|mg/ml)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return {t for t in text.split() if t and t not in _STOPWORDS}


def _same_concept(a, b):
    """True if two free-text values refer to the same concept.

    Matches when the normalized token sets are equal, or when one is fully
    contained in the other (e.g. "uri" ⊆ "upper respiratory infection uri").
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return ta == tb or ta <= tb or tb <= ta


def _extra_items(chart_items, bill_items):
    """Return billed items that have no concept-matching item in the chart."""
    return [
        bill_item
        for bill_item in bill_items
        if not any(_same_concept(bill_item, chart_item) for chart_item in chart_items)
    ]


def _compare_diagnosis(chart_facts, bill_facts):
    """Flag when the billed diagnosis differs from the charted diagnosis."""
    chart_dx = chart_facts.get("diagnosis", "Unknown")
    bill_dx = bill_facts.get("billed_diagnosis", "Unknown")

    if _same_concept(chart_dx, bill_dx):
        return None

    reason = (
        f"Chart documents '{chart_dx}', but the bill claims '{bill_dx}'. "
        "A more severe billed diagnosis can justify a higher reimbursement "
        "level and should be verified against the clinical record."
    )
    return {
        "field": "Diagnosis",
        "chart_value": chart_dx,
        "bill_value": bill_dx,
        "risk_reason": reason,
        "chart_evidence": chart_facts.get("evidence", {}).get("diagnosis", ""),
        "billing_evidence": bill_facts.get("evidence", {}).get("diagnosis", ""),
    }


def _compare_medications(chart_facts, bill_facts):
    """Flag medications that were billed but not documented in the chart."""
    chart_meds = chart_facts.get("medications", [])
    bill_meds = bill_facts.get("billed_medications", [])

    extra_billed = _extra_items(chart_meds, bill_meds)
    if not extra_billed:
        return None

    reason = (
        f"Medication(s) billed but not documented in the chart: "
        f"{_format_list(extra_billed)}. "
        "Billing for un-charted medications may indicate upcoding or a "
        "documentation gap."
    )
    return {
        "field": "Medications",
        "chart_value": _format_list(chart_meds),
        "bill_value": _format_list(bill_meds),
        "risk_reason": reason,
        "chart_evidence": chart_facts.get("evidence", {}).get("medications", ""),
        "billing_evidence": bill_facts.get("evidence", {}).get("medications", ""),
    }


def _compare_procedures(chart_facts, bill_facts):
    """Flag procedures that were billed but not documented in the chart."""
    chart_procs = chart_facts.get("procedures", [])
    bill_procs = bill_facts.get("billed_procedures", [])

    extra_billed = _extra_items(chart_procs, bill_procs)
    if not extra_billed:
        return None

    reason = (
        f"Procedure(s) billed but not documented in the chart: "
        f"{_format_list(extra_billed)}. "
        "Billing for un-performed or un-documented procedures is a common "
        "source of claim error and potential fraud."
    )
    return {
        "field": "Procedures",
        "chart_value": _format_list(chart_procs),
        "bill_value": _format_list(bill_procs),
        "risk_reason": reason,
        "chart_evidence": chart_facts.get("evidence", {}).get("procedures", ""),
        "billing_evidence": bill_facts.get("evidence", {}).get("procedures", ""),
    }


def _compare_severity(chart_facts, bill_facts):
    """Flag when the billed severity is higher than the charted severity."""
    chart_sev = chart_facts.get("severity", "low")
    bill_sev = bill_facts.get("billed_severity", "low")

    chart_rank = SEVERITY_RANK.get(chart_sev, 1)
    bill_rank = SEVERITY_RANK.get(bill_sev, 1)

    if bill_rank <= chart_rank:
        return None

    reason = (
        f"Chart severity is '{chart_sev}', but the claim is billed at "
        f"'{bill_sev}' complexity. Higher billed complexity than the chart "
        "supports can inflate reimbursement."
    )
    return {
        "field": "Severity / Complexity",
        "chart_value": chart_sev,
        "bill_value": bill_sev,
        "risk_reason": reason,
        "chart_evidence": chart_facts.get("evidence", {}).get("diagnosis", ""),
        "billing_evidence": bill_facts.get("evidence", {}).get("diagnosis", ""),
    }


def _compare_amount_vs_severity(chart_facts, bill_facts):
    """Flag a high dollar amount paired with a low-severity chart."""
    chart_sev = chart_facts.get("severity", "low")
    amount = bill_facts.get("claim_amount", 0) or 0

    # A low-severity encounter billed at a large amount is suspicious.
    if chart_sev == "low" and amount > 10000:
        reason = (
            f"Claim amount of ${amount:,.0f} is high for a low-severity "
            "encounter. Large charges on low-acuity visits warrant review."
        )
        return {
            "field": "Claim Amount vs Severity",
            "chart_value": f"severity={chart_sev}",
            "bill_value": f"${amount:,.0f}",
            "risk_reason": reason,
            "chart_evidence": chart_facts.get("evidence", {}).get("diagnosis", ""),
            "billing_evidence": bill_facts.get("evidence", {}).get("amount", ""),
        }
    return None


def compare_claim(chart_facts, bill_facts):
    """Run all comparisons and return a list of mismatch dictionaries.

    An empty list means chart and bill are consistent on every checked field.
    """
    comparisons = [
        _compare_diagnosis,
        _compare_medications,
        _compare_procedures,
        _compare_severity,
        _compare_amount_vs_severity,
    ]

    mismatches = []
    for compare_fn in comparisons:
        result = compare_fn(chart_facts, bill_facts)
        if result is not None:
            mismatches.append(result)
    return mismatches
