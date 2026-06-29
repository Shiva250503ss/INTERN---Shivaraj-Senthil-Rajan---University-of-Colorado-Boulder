"""LLM-powered fact extraction using Claude (the "present/future" approach).

This module mirrors the rule-based `extractor.py` but uses Anthropic's Claude
model to read the clinical / billing free text and return the *same* structured
dictionaries. Keeping the output shape identical means `comparator.py` and
`risk_scorer.py` work unchanged regardless of which extraction engine is used.

Design notes for the demo:
- We use Claude Haiku 4.5 (`claude-haiku-4-5`) — the cheapest, fastest Claude
  model — because this is a lightweight extraction task and we want to keep the
  token cost minimal.
- Everything degrades gracefully: if the `anthropic` SDK is missing, no API key
  is provided, or the API call fails, we raise `LLMUnavailable` so the app can
  transparently fall back to the rule-based engine. The demo never breaks.
"""

import json
import re

# Cheapest Claude model — plenty capable for short-text fact extraction.
MODEL = "claude-haiku-4-5"


class LLMUnavailable(Exception):
    """Raised when the LLM path cannot run (no SDK, no key, or API error).

    The Streamlit app catches this and falls back to the rule-based extractor.
    """


def _get_client(api_key=None):
    """Return an Anthropic client, or raise LLMUnavailable if not possible.

    If `api_key` is None, the SDK falls back to the ANTHROPIC_API_KEY env var.
    """
    try:
        import anthropic
    except ImportError as exc:  # SDK not installed
        raise LLMUnavailable(
            "The 'anthropic' package is not installed. Run: pip install anthropic"
        ) from exc

    try:
        # If api_key is None the SDK reads ANTHROPIC_API_KEY from the environment.
        return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    except Exception as exc:  # e.g. no key found anywhere
        raise LLMUnavailable(f"Could not initialize Claude client: {exc}") from exc


def _call_claude(client, prompt):
    """Send a single prompt to Claude and return the raw text response."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            # A tight system prompt keeps Haiku focused on returning clean JSON.
            system=(
                "You are a clinical data extraction assistant for a medical "
                "claim-review tool. Extract structured facts from the text you "
                "are given. Respond with ONLY a single valid JSON object and no "
                "explanation, markdown, or code fences."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # network error, bad key, rate limit, etc.
        raise LLMUnavailable(f"Claude API call failed: {exc}") from exc

    # Concatenate any text blocks in the response.
    return "".join(block.text for block in response.content if block.type == "text")


def _parse_json(text):
    """Parse a JSON object out of Claude's response, tolerating stray text."""
    text = text.strip()
    # Strip ```json ... ``` fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: grab the outermost {...} block.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise LLMUnavailable("Claude did not return parseable JSON.")


# ---------------------------------------------------------------------------
# Normalization helpers — guarantee the exact shape the rest of the app expects
# ---------------------------------------------------------------------------
def _as_list(value):
    """Coerce a value into a clean list of strings."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if value in (None, "", "none", "None"):
        return []
    return [str(value).strip()]


def _norm_severity(value):
    """Normalize any severity-ish string to low / medium / high."""
    text = str(value).strip().lower()
    if "high" in text:
        return "high"
    if "med" in text:
        return "medium"
    return "low"


def _norm_amount(value):
    """Coerce a billed amount into a number (int when whole)."""
    if isinstance(value, (int, float)):
        amount = float(value)
    else:
        # Pull digits out of strings like "$12,400".
        digits = re.sub(r"[^0-9.]", "", str(value))
        amount = float(digits) if digits else 0.0
    return int(amount) if amount.is_integer() else amount


def _norm_evidence(value, keys):
    """Ensure evidence is a dict with the expected string keys."""
    value = value if isinstance(value, dict) else {}
    return {k: str(value.get(k, "")) for k in keys}


# ---------------------------------------------------------------------------
# Public API — same names/shape as the rule-based extractor, with an _llm suffix
# ---------------------------------------------------------------------------
def extract_chart_facts_llm(chart_text, api_key=None):
    """LLM version of `extract_chart_facts`. Returns the same dict shape."""
    client = _get_client(api_key)
    prompt = (
        "Extract the following from this MEDICAL CHART note and return JSON with "
        "exactly these keys: "
        '{"diagnosis": str, "medications": [str], "procedures": [str], '
        '"length_of_stay": str, "severity": "low"|"medium"|"high", '
        '"evidence": {"diagnosis": str, "medications": str, "procedures": str}}. '
        "For each evidence field, quote the exact sentence from the chart that "
        "supports the fact (empty string if none). Judge severity from the "
        "diagnosis and clinical detail.\n\n"
        f"CHART:\n{chart_text}"
    )
    data = _parse_json(_call_claude(client, prompt))
    return {
        "diagnosis": str(data.get("diagnosis", "Unknown")) or "Unknown",
        "medications": _as_list(data.get("medications")),
        "procedures": _as_list(data.get("procedures")),
        "length_of_stay": str(data.get("length_of_stay", "Not specified")),
        "severity": _norm_severity(data.get("severity")),
        "evidence": _norm_evidence(
            data.get("evidence"), ["diagnosis", "medications", "procedures"]
        ),
    }


def extract_billing_facts_llm(bill_text, api_key=None):
    """LLM version of `extract_billing_facts`. Returns the same dict shape."""
    client = _get_client(api_key)
    prompt = (
        "Extract the following from this BILLING CLAIM note and return JSON with "
        "exactly these keys: "
        '{"billed_diagnosis": str, "billed_medications": [str], '
        '"billed_procedures": [str], "claim_amount": number, '
        '"billed_severity": "low"|"medium"|"high", '
        '"evidence": {"diagnosis": str, "medications": str, "procedures": str, '
        '"amount": str}}. '
        "claim_amount is the total dollar figure as a plain number (no $ or "
        "commas). For each evidence field, quote the exact sentence from the bill "
        "that supports the fact (empty string if none).\n\n"
        f"BILL:\n{bill_text}"
    )
    data = _parse_json(_call_claude(client, prompt))
    return {
        "billed_diagnosis": str(data.get("billed_diagnosis", "Unknown")) or "Unknown",
        "billed_medications": _as_list(data.get("billed_medications")),
        "billed_procedures": _as_list(data.get("billed_procedures")),
        "claim_amount": _norm_amount(data.get("claim_amount")),
        "billed_severity": _norm_severity(data.get("billed_severity")),
        "evidence": _norm_evidence(
            data.get("evidence"),
            ["diagnosis", "medications", "procedures", "amount"],
        ),
    }
