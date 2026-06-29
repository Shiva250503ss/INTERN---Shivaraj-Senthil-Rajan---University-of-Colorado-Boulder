"""Rule-based fact extraction for ClaimGuard Lite.

This module turns free-text clinical notes and billing claims into small,
structured dictionaries. For a proof-of-concept we deliberately avoid LLM APIs
and use simple, transparent keyword matching on lowercased text. Every
extraction also captures the *sentence* it was found in ("evidence") so a human
reviewer can see exactly why a fact was pulled out.
"""

import re

# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------
# Each diagnosis maps to a list of trigger phrases. The first phrase is treated
# as the canonical label that we report back.
DIAGNOSIS_KEYWORDS = {
    "Upper Respiratory Infection (URI)": [
        "upper respiratory infection",
        "uri",
        "common cold",
        "cold",
        "cough",
        "fever",
        "sore throat",
    ],
    "Pneumonia": [
        "pneumonia",
        "lung infection",
        "chest infection",
    ],
}

# Medication name -> canonical label.
MEDICATION_KEYWORDS = {
    "amoxicillin": "Amoxicillin",
    "albuterol": "Albuterol",
    "steroid": "Steroid",
    "prednisone": "Steroid",
    "ibuprofen": "Ibuprofen",
    "antibiotic": "Antibiotics",
    "antibiotics": "Antibiotics",
}

# Procedure trigger phrase -> canonical label.
PROCEDURE_KEYWORDS = {
    "surgery": "Surgery",
    "operation": "Surgery",
    "procedure": "Procedure",
    "steroid injection": "Steroid Injection",
    "injection": "Injection",
    "x-ray": "X-Ray",
    "xray": "X-Ray",
    "chest x-ray": "Chest X-Ray",
}

# Diagnoses that are generally considered higher acuity than a simple URI.
HIGH_SEVERITY_DIAGNOSES = {"Pneumonia"}


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------
def _split_sentences(text):
    """Split raw text into a list of trimmed sentences.

    We keep this intentionally naive (split on ., !, ?, and newlines) because
    clinical free text in this demo is short and well-formed.
    """
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _contains(text_lower, phrase):
    """True if `phrase` appears in `text_lower` as a whole word/phrase.

    We use word boundaries so short keywords don't match inside longer words
    (e.g. so "uri" does NOT match inside "d-uri-ng", and "procedure" does not
    match inside "procedures"). This keeps the simple keyword approach honest.
    """
    return re.search(r"\b" + re.escape(phrase) + r"\b", text_lower) is not None


def _find_evidence(text, phrase):
    """Return the first sentence in `text` that contains `phrase` (as a word).

    Matching is case-insensitive. Returns an empty string if nothing matches.
    """
    for sentence in _split_sentences(text):
        if _contains(sentence.lower(), phrase.lower()):
            return sentence
    return ""


def _extract_diagnosis(text_lower, raw_text):
    """Return (diagnosis_label, evidence_sentence) using keyword matching."""
    for label, phrases in DIAGNOSIS_KEYWORDS.items():
        for phrase in phrases:
            if _contains(text_lower, phrase):
                return label, _find_evidence(raw_text, phrase)
    return "Unknown", ""


def _extract_medications(text_lower, raw_text):
    """Return (list_of_medication_labels, evidence_sentence).

    Procedure phrases are stripped out first so that something like
    "steroid injection" (a procedure) is not also double-counted as a
    "steroid" medication.
    """
    # Remove multi-word procedure phrases (e.g. "steroid injection") so their
    # words don't trigger a false medication match.
    med_text = text_lower
    for phrase in PROCEDURE_KEYWORDS:
        if " " in phrase:
            med_text = med_text.replace(phrase, " ")

    found = []
    evidence = ""
    for keyword, label in MEDICATION_KEYWORDS.items():
        if _contains(med_text, keyword) and label not in found:
            found.append(label)
            if not evidence:
                evidence = _find_evidence(raw_text, keyword)
    return found, evidence


def _extract_procedures(text_lower, raw_text):
    """Return (list_of_procedure_labels, evidence_sentence)."""
    found = []
    matched_keywords = []
    evidence = ""
    # Sort by length so multi-word phrases ("steroid injection") win over
    # their single-word substrings ("injection").
    for keyword in sorted(PROCEDURE_KEYWORDS, key=len, reverse=True):
        label = PROCEDURE_KEYWORDS[keyword]
        if not _contains(text_lower, keyword) or label in found:
            continue
        # Skip a sub-phrase if a longer phrase we already matched contains it
        # (e.g. don't report "X-Ray" separately when "Chest X-Ray" matched).
        if any(keyword in longer for longer in matched_keywords):
            continue
        found.append(label)
        matched_keywords.append(keyword)
        if not evidence:
            evidence = _find_evidence(raw_text, keyword)
    return found, evidence


def _extract_length_of_stay(text_lower):
    """Infer a simple length-of-stay description from common phrases."""
    if "same day" in text_lower or "same-day" in text_lower:
        return "Same-day discharge"
    if "discharged" in text_lower and "admitted" not in text_lower:
        return "Discharged (outpatient)"

    # Look for "admitted ... N day(s)" style phrasing.
    match = re.search(r"(\d+)\s*day", text_lower)
    if match:
        return f"{match.group(1)} day(s) inpatient"
    if "admitted" in text_lower:
        return "Admitted (length not specified)"
    return "Not specified"


def _infer_severity(diagnosis, procedures, length_of_stay, text_lower):
    """Derive a coarse low / medium / high severity rating.

    The logic is intentionally simple and explainable:
    - High-acuity diagnoses (e.g. pneumonia) or surgery -> high
    - Inpatient stays or explicit "high complexity" language -> high/medium
    - Otherwise default to low
    """
    if diagnosis in HIGH_SEVERITY_DIAGNOSES:
        return "high"
    if "high complexity" in text_lower or "critical" in text_lower:
        return "high"
    if any(p in ("Surgery",) for p in procedures):
        return "high"
    if "inpatient" in length_of_stay.lower() or "admitted" in text_lower:
        return "medium"
    return "low"


def _extract_claim_amount(text):
    """Pull the largest dollar amount out of the text, e.g. '$12,400' -> 12400."""
    amounts = []
    for match in re.findall(r"\$\s?([0-9][0-9,]*(?:\.\d{1,2})?)", text):
        cleaned = match.replace(",", "")
        try:
            amounts.append(float(cleaned))
        except ValueError:
            continue
    if not amounts:
        return 0
    # Use the largest amount mentioned as the headline claim amount.
    amount = max(amounts)
    # Return an int when the value is whole for cleaner display.
    return int(amount) if amount.is_integer() else amount


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_chart_facts(chart_text):
    """Extract structured facts from a medical chart note.

    Returns a dictionary with diagnosis, medications, procedures, length of
    stay, an inferred severity, and an `evidence` block of supporting sentences.
    """
    chart_text = chart_text or ""
    text_lower = chart_text.lower()

    diagnosis, diagnosis_evidence = _extract_diagnosis(text_lower, chart_text)
    medications, medication_evidence = _extract_medications(text_lower, chart_text)
    procedures, procedure_evidence = _extract_procedures(text_lower, chart_text)
    length_of_stay = _extract_length_of_stay(text_lower)
    severity = _infer_severity(diagnosis, procedures, length_of_stay, text_lower)

    return {
        "diagnosis": diagnosis,
        "medications": medications,
        "procedures": procedures,
        "length_of_stay": length_of_stay,
        "severity": severity,
        "evidence": {
            "diagnosis": diagnosis_evidence,
            "medications": medication_evidence,
            "procedures": procedure_evidence,
        },
    }


def extract_billing_facts(bill_text):
    """Extract structured facts from a billing claim note.

    Mirrors `extract_chart_facts` but also captures a `claim_amount` and the
    sentence that the dollar figure was found in.
    """
    bill_text = bill_text or ""
    text_lower = bill_text.lower()

    diagnosis, diagnosis_evidence = _extract_diagnosis(text_lower, bill_text)
    medications, medication_evidence = _extract_medications(text_lower, bill_text)
    procedures, procedure_evidence = _extract_procedures(text_lower, bill_text)
    claim_amount = _extract_claim_amount(bill_text)

    # Find the sentence that mentions a dollar amount, for evidence.
    amount_evidence = _find_evidence(bill_text, "$")

    # Billing severity can be stated explicitly ("high complexity") or inferred
    # from the billed diagnosis / procedures.
    length_proxy = "inpatient" if "inpatient" in text_lower or "admitted" in text_lower else ""
    billed_severity = _infer_severity(diagnosis, procedures, length_proxy, text_lower)

    return {
        "billed_diagnosis": diagnosis,
        "billed_medications": medications,
        "billed_procedures": procedures,
        "claim_amount": claim_amount,
        "billed_severity": billed_severity,
        "evidence": {
            "diagnosis": diagnosis_evidence,
            "medications": medication_evidence,
            "procedures": procedure_evidence,
            "amount": amount_evidence,
        },
    }
