# 🛡️ ClaimGuard Lite — Medical Chart vs Billing Claim Mismatch Detector

A simple, demo-ready healthcare AI proof-of-concept that compares a patient's
**medical chart** against a submitted **billing claim**, flags possible
mismatches, and produces a plain-English **fraud risk level** with a recommended
human-review action.

> ⚠️ This is a proof-of-concept for an intern assessment — **not** a production
> fraud-detection system.

---

## Problem Statement

Healthcare payers process millions of claims. A claim can be **upcoded** or
**mismatched** when the billed diagnosis, medications, procedures, or complexity
level do not match what the clinical chart actually documents. Reviewing every
claim by hand is slow and expensive, and the most suspicious claims can be hard
to surface.

ClaimGuard Lite shows how lightweight AI/NLP-style **fact extraction** and
**side-by-side comparison** can help reviewers quickly spot claims that deserve a
closer look.

## Why This Matters to Cotiviti

Cotiviti's core business is **payment accuracy** and **healthcare analytics** —
making sure claims are paid correctly and identifying fraud, waste, and abuse.
This prototype mirrors that workflow in miniature:

1. **Extract** structured facts from unstructured chart and claim text.
2. **Compare** the documented care against the billed care.
3. **Score** the risk and recommend an action (auto-approve, human review, or
   auditor escalation).

It demonstrates the *thinking* behind automated claim review — extraction,
comparison, explainable scoring, and a human-in-the-loop recommendation — in a
form that is easy to read and reason about.

## What the App Does

Given two text inputs (a medical chart and a billing claim), the app:

- **Extracts structured facts** from each side (diagnosis, medications,
  procedures, length of stay, severity, claim amount).
- Captures **evidence snippets** — the exact sentence each fact came from — so a
  reviewer can audit every decision.
- Builds a **side-by-side mismatch table** explaining *why* each discrepancy is
  flagged.
- Produces a **fraud risk score card** (Low / Medium / High, 0–100) with a
  **recommended action**.

It offers **two extraction engines** you can switch between in the sidebar, plus
an **OCR** step — these directly demonstrate the Topic 1 technologies (NLP, LLM,
OCR, Computer Vision, LMM).

## Extraction Engines — Past, Present, & Future

Topic 1 is about the *evolution* of clinical NLP. This app demonstrates that
evolution with two interchangeable engines producing the **same** structured
output:

| Engine | Technology | Maps to Topic 1 |
|--------|------------|-----------------|
| **Rule-based (offline)** | Transparent keyword matching with word boundaries | The "past" of clinical NLP — fast, explainable, no API needed |
| **AI — Claude LLM** | Claude Haiku 4.5 reads the text and returns structured facts | The "present/future" — robust to phrasing, synonyms, and context |

Because both engines return an identical dictionary shape, the comparison and
risk-scoring logic is engine-agnostic.

### OCR / Computer Vision (LMM)

The sidebar also includes an **OCR** step: upload a photo or scan of a chart or
claim form, and **Claude's vision model (a Large Multimodal Model)** transcribes
the text straight into the input box. This covers the *OCR* and *Computer Vision*
keywords from Topic 1 with a single multimodal model call.

> **Model choice & cost:** both the AI engine and OCR use **Claude Haiku 4.5**
> (`claude-haiku-4-5`), the cheapest/fastest Claude model, to keep token cost
> minimal. The API key is read **only** from a local `.env` file — there is no
> key entry in the UI. **Everything degrades gracefully:** with no key, no
> network, or any API error, the app automatically falls back to the rule-based
> engine so the demo never breaks.

## How to Run It

```bash
# 1. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Anthropic API key (only needed for the AI engine + OCR)
#    Open the .env file and paste your key after ANTHROPIC_API_KEY=
#    (or copy the template: cp .env.example .env)

# 4. Launch the app
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501), pick a
sample case, click **"Load selected sample"**, and click **"Analyze Claim"**.

### Setting your API key (`.env`)

The key lives in a git-ignored `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Get a key from the [Anthropic Console](https://console.anthropic.com) →
**Settings → API Keys**. The `.env` file is listed in `.gitignore`, so your key
is **never committed**. The rule-based engine works with no key at all.

## Folder Structure

```
.
├── app.py                  # Streamlit UI (the demo front end)
├── requirements.txt        # streamlit, pandas, anthropic, python-dotenv
├── README.md               # this file
├── .env.example            # template for your API key (copy to .env)
├── .gitignore              # keeps .env (your key) out of git
├── src/
│   ├── __init__.py         # exposes the core functions
│   ├── extractor.py        # rule-based extract_chart_facts / extract_billing_facts
│   ├── llm_extractor.py    # Claude LLM versions (same output shape) + fallback
│   ├── ocr.py              # Claude vision OCR (image → text)
│   ├── comparator.py       # compare_claim
│   └── risk_scorer.py      # score_risk
└── sample_data/
    ├── case1_low_risk.json     # consistent claim     -> Low
    ├── case2_high_risk.json    # upcoded claim        -> High
    ├── case3_medium_risk.json  # one extra item       -> Medium
    └── images/                 # chart/claim images for the OCR demo
        ├── case1_chart.png  ├── case1_bill.png
        └── case2_chart.png  └── case2_bill.png
```

## How the Logic Works

| Step        | Module          | What it does                                                              |
|-------------|-----------------|---------------------------------------------------------------------------|
| Extraction  | `extractor.py`  | Keyword matching (with word boundaries) pulls facts + evidence sentences. |
| Comparison  | `comparator.py` | Diffs diagnosis, meds, procedures, severity, and amount-vs-severity.      |
| Scoring     | `risk_scorer.py`| 0 mismatches → Low, 1 → Medium, 2+ → High; low-severity + >$10k → High.    |

**Recommended actions:** Low → *Auto-approve*, Medium → *Send for human review*,
High → *Escalate to auditor review*.

## Example Use Cases

The `sample_data/` folder ships three ready-to-run cases:

1. **Case 1 — Low Risk:** Chart and bill both show a same-day URI treated with
   one medication for a small amount. → **Low / Auto-approve**.
2. **Case 2 — High Risk:** Chart shows a simple same-day URI, but the bill claims
   pneumonia, extra medications, a procedure, high complexity, and **$12,400**. →
   **High / Escalate to auditor review**.
3. **Case 3 — Medium Risk:** Chart and bill agree on pneumonia and antibiotics,
   but the bill adds a **steroid injection** the chart never documents. →
   **Medium / Send for human review**.

## Limitations

This prototype is intentionally simple and avoids production-level complexity:

- **Rule-based engine is keyword-based:** the offline engine uses keyword
  matching, so unusual phrasing or synonyms outside the keyword list may be
  missed. (The AI engine handles these, but needs an API key.)
- **No medical coding validation:** it does not check real **ICD-10 / CPT / DRG**
  codes.
- **No persistence or security:** no database, authentication, or audit logging.
- **English, well-formed text:** the rule-based engine assumes short, clean
  clinical notes like the provided samples.
- **LLM is non-deterministic and costs tokens:** the AI engine's output can vary
  slightly between runs and consumes API credits (kept minimal via Haiku).

## Future Improvements

This prototype intentionally keeps things simple. Some Topic 1 technologies are
already demonstrated here (an **LLM extraction engine** and **Claude vision
OCR**); natural next steps to harden them into production include:

- **Higher-accuracy OCR** tuned for handwritten clinical notes and degraded
  scans (the current OCR is a single general-purpose vision call).
- **Structured-output guarantees** (JSON-schema-constrained LLM responses) for
  fully reliable extraction.
- **ICD-10 / CPT / DRG validation** against official code sets.
- **PDF report export** for reviewers and auditors.
- A **human auditor feedback loop** so confirmed decisions improve the system.
- **Integration with existing claim review systems** and payer workflows.
