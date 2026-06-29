"""ClaimGuard Lite — Streamlit front end.

A proof-of-concept that compares a free-text medical chart against a free-text
billing claim, extracts structured facts from each, highlights mismatches, and
produces a simple fraud-risk score with a recommended review action.

Run with:
    streamlit run app.py
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# Load the Anthropic API key (and any other secrets) from a local .env file so
# it never has to be typed into the UI or hard-coded. The .env file is
# git-ignored. If python-dotenv isn't installed, we just rely on real
# environment variables instead.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Import the core building blocks from the src package.
from src import (
    extract_chart_facts,
    extract_billing_facts,
    compare_claim,
    score_risk,
    # Optional LLM / multimodal engines (degrade gracefully to rule-based).
    extract_chart_facts_llm,
    extract_billing_facts_llm,
    transcribe_image,
    LLMUnavailable,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ClaimGuard Lite",
    page_icon="🛡️",
    layout="wide",
)

SAMPLE_DIR = Path(__file__).parent / "sample_data"


# ---------------------------------------------------------------------------
# Sample-case loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_sample_cases():
    """Load all sample cases from the sample_data folder, sorted by filename."""
    cases = {"— Select a sample case —": None}
    if SAMPLE_DIR.exists():
        for path in sorted(SAMPLE_DIR.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cases[data.get("name", path.stem)] = data
    return cases


# ---------------------------------------------------------------------------
# Risk presentation helpers
# ---------------------------------------------------------------------------
RISK_COLORS = {
    "Low": "#1a7f37",      # green
    "Medium": "#bf8700",   # amber
    "High": "#cf222e",     # red
}


def render_risk_card(risk):
    """Render the fraud risk score card with a colored alert box."""
    level = risk["risk_level"]
    color = RISK_COLORS.get(level, "#444")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Fraud Risk Score", value=f"{risk['risk_score']}/100")
        st.metric(label="Risk Level", value=level)
    with col2:
        # Colored alert box matching the risk level.
        if level == "Low":
            st.success(f"**Recommended Action: {risk['recommended_action']}**")
        elif level == "Medium":
            st.warning(f"**Recommended Action: {risk['recommended_action']}**")
        else:
            st.error(f"**Recommended Action: {risk['recommended_action']}**")

        st.markdown(
            f"<div style='padding:0.75rem;border-left:5px solid {color};"
            f"background:rgba(0,0,0,0.03);border-radius:4px;'>"
            f"{risk['summary']}</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Session state setup (so sample text persists across reruns)
# ---------------------------------------------------------------------------
if "chart_text" not in st.session_state:
    st.session_state.chart_text = ""
if "bill_text" not in st.session_state:
    st.session_state.bill_text = ""


# ---------------------------------------------------------------------------
# Sidebar: extraction engine + OCR upload
# ---------------------------------------------------------------------------
# The app supports two extraction engines, illustrating the evolution of
# clinical NLP (Topic 1: "Past, Present, & Future Approaches"):
#   - Rule-based  : transparent keyword matching (the "past")
#   - AI (Claude) : an LLM reads the text (the "present/future")
#
# The Claude API key is read from the .env file / environment only — there is no
# key entry in the UI. We pass api_key=None everywhere so the SDK resolves it
# from ANTHROPIC_API_KEY.
api_key = None
key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))

with st.sidebar:
    st.header("⚙️ Settings")

    engine = st.radio(
        "Extraction engine",
        options=["Rule-based (offline)", "AI — Claude LLM"],
        help=(
            "Rule-based uses transparent keyword matching and needs no API key. "
            "AI uses Claude (Haiku) to read the text like a human reviewer would."
        ),
    )
    use_llm = engine.startswith("AI")

    # Show whether a key was found in .env / environment (no entry box here).
    if key_present:
        st.caption("🔑 Claude API key: **loaded from .env**")
    else:
        st.caption(
            "🔑 Claude API key: **not found** — add it to a `.env` file to use "
            "the AI engine and OCR. The rule-based engine works without it."
        )

    st.divider()

    # --- OCR: turn an uploaded chart/claim image into text (Claude vision) ----
    st.subheader("🖼️ OCR (image → text)")
    st.caption(
        "Upload a photo/scan of a chart or claim. Claude's vision model "
        "transcribes it (demonstrates OCR + computer vision + LMM)."
    )
    ocr_image = st.file_uploader(
        "Upload a chart/claim image", type=["png", "jpg", "jpeg"]
    )
    ocr_target = st.radio(
        "Send transcribed text to:",
        options=["Medical Chart", "Billing Claim"],
        horizontal=True,
    )
    if st.button("🔎 Run OCR on image"):
        if ocr_image is None:
            st.warning("Upload an image first.")
        else:
            try:
                with st.spinner("Transcribing image with Claude vision..."):
                    text = transcribe_image(
                        ocr_image.getvalue(), ocr_image.type, api_key=api_key
                    )
                target_key = "chart_text" if ocr_target == "Medical Chart" else "bill_text"
                st.session_state[target_key] = text
                st.success(f"Transcribed into the {ocr_target} box.")
            except LLMUnavailable as exc:
                st.error(f"OCR unavailable: {exc}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛡️ ClaimGuard Lite")
st.subheader("Medical Chart vs Billing Claim Mismatch Detector")
st.markdown(
    "This proof-of-concept compares a **patient medical chart** against a "
    "**billing claim**. It extracts structured facts from each, compares them "
    "side by side, flags possible mismatches, and produces a simple fraud-risk "
    "level with a recommended human-review action.\n\n"
    "_This is a demo of how AI/NLP-style extraction can support medical claim "
    "review — it is **not** a production fraud system._"
)

# ---------------------------------------------------------------------------
# Sample-case selector
# ---------------------------------------------------------------------------
sample_cases = load_sample_cases()

st.markdown("#### 1. Load a sample case (optional)")
selected_name = st.selectbox(
    "Choose a pre-built example, or enter your own text below:",
    options=list(sample_cases.keys()),
)
if st.button("📂 Load selected sample"):
    case = sample_cases.get(selected_name)
    if case:
        st.session_state.chart_text = case["chart_text"]
        st.session_state.bill_text = case["bill_text"]
        st.success(f"Loaded: {selected_name}")
    else:
        st.info("Select a named sample case first.")

# Show the description of the currently selected case, if any.
selected_case = sample_cases.get(selected_name)
if selected_case and selected_case.get("description"):
    st.caption(selected_case["description"])

# ---------------------------------------------------------------------------
# Side-by-side text inputs
# ---------------------------------------------------------------------------
st.markdown("#### 2. Enter the chart and billing text")
input_col1, input_col2 = st.columns(2)
with input_col1:
    chart_text = st.text_area(
        "📋 Medical Chart Text",
        height=220,
        key="chart_text",
        placeholder="e.g. Patient diagnosed with an upper respiratory infection...",
    )
with input_col2:
    bill_text = st.text_area(
        "🧾 Billing Claim Text",
        height=220,
        key="bill_text",
        placeholder="e.g. Claim submitted for pneumonia, total amount $12,400...",
    )

# ---------------------------------------------------------------------------
# Analyze button + results
# ---------------------------------------------------------------------------
st.markdown("#### 3. Run the analysis")
analyze = st.button("🔍 Analyze Claim", type="primary")

if analyze:
    if not chart_text.strip() or not bill_text.strip():
        st.warning("Please provide both chart text and billing text before analyzing.")
    else:
        # 1) Extract structured facts from both inputs.
        #    If the AI engine is selected, use Claude; if anything goes wrong
        #    (no key, no network, SDK missing) fall back to the rule-based
        #    engine so the demo always produces a result.
        engine_used = "Rule-based"
        if use_llm:
            try:
                with st.spinner("Extracting facts with Claude..."):
                    chart_facts = extract_chart_facts_llm(chart_text, api_key=api_key)
                    bill_facts = extract_billing_facts_llm(bill_text, api_key=api_key)
                engine_used = "AI — Claude LLM"
            except LLMUnavailable as exc:
                st.warning(
                    f"AI engine unavailable ({exc}). Falling back to the "
                    "rule-based engine."
                )
                chart_facts = extract_chart_facts(chart_text)
                bill_facts = extract_billing_facts(bill_text)
        else:
            chart_facts = extract_chart_facts(chart_text)
            bill_facts = extract_billing_facts(bill_text)

        st.caption(f"Extraction engine used: **{engine_used}**")

        # 2) Compare the two fact sets.
        mismatches = compare_claim(chart_facts, bill_facts)

        # 3) Score the overall risk.
        risk = score_risk(mismatches, chart_facts, bill_facts)

        st.divider()

        # --- Fraud Risk Score Card (shown first for impact) --------------
        st.markdown("### 🚦 Fraud Risk Assessment")
        render_risk_card(risk)

        st.divider()

        # --- Extracted facts side by side --------------------------------
        st.markdown("### 📑 Extracted Facts")
        fact_col1, fact_col2 = st.columns(2)
        with fact_col1:
            st.markdown("**Extracted Chart Facts**")
            st.json(chart_facts)
        with fact_col2:
            st.markdown("**Extracted Billing Facts**")
            st.json(bill_facts)

        st.divider()

        # --- Side-by-side mismatch table ---------------------------------
        st.markdown("### ⚠️ Mismatch Analysis")
        if mismatches:
            df = pd.DataFrame(mismatches)
            df = df.rename(
                columns={
                    "field": "Field",
                    "chart_value": "Chart Value",
                    "bill_value": "Billed Value",
                    "risk_reason": "Why It's Flagged",
                    "chart_evidence": "Chart Evidence",
                    "billing_evidence": "Billing Evidence",
                }
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.success("No mismatches detected — chart and bill are consistent.")
