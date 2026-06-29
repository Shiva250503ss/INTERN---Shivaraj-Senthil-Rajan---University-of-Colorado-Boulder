"""ClaimGuard Lite source package.

Exposes the three core building blocks of the proof-of-concept:
- extractor:   pull structured facts out of free-text chart / bill notes
- comparator:  diff the two fact sets and surface mismatches
- risk_scorer: turn mismatches into a risk level + recommended action
"""

from .extractor import extract_chart_facts, extract_billing_facts
from .comparator import compare_claim
from .risk_scorer import score_risk
from .llm_extractor import (
    extract_chart_facts_llm,
    extract_billing_facts_llm,
    LLMUnavailable,
)
from .ocr import transcribe_image

__all__ = [
    "extract_chart_facts",
    "extract_billing_facts",
    "compare_claim",
    "score_risk",
    # LLM / multimodal additions (optional, with graceful fallback)
    "extract_chart_facts_llm",
    "extract_billing_facts_llm",
    "transcribe_image",
    "LLMUnavailable",
]
