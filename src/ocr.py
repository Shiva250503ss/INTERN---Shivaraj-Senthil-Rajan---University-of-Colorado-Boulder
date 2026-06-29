"""Image-to-text (OCR) using Claude's vision capability.

Topic 1 of the assessment explicitly covers OCR, Computer Vision, and LMMs
(Large Multimodal Models). Rather than install a separate OCR engine, we use
Claude Haiku 4.5 as a multimodal model: we hand it an image of a chart or claim
form and ask it to transcribe the text. This single call demonstrates OCR +
computer vision + LMM at once, and reuses the same cheap model as the LLM
extractor.

As with the LLM extractor, this degrades gracefully via `LLMUnavailable` so the
rest of the app keeps working if the SDK/key/network is unavailable.
"""

import base64

from .llm_extractor import MODEL, LLMUnavailable, _get_client


def transcribe_image(image_bytes, media_type, api_key=None):
    """Transcribe the text contained in an image using Claude vision.

    Args:
        image_bytes: raw bytes of the uploaded image.
        media_type:  e.g. "image/png" or "image/jpeg".
        api_key:     optional Anthropic API key (falls back to env var).

    Returns:
        The transcribed text as a single string.

    Raises:
        LLMUnavailable: if the SDK/key/network is unavailable or the call fails.
    """
    client = _get_client(api_key)

    # Claude's vision API takes base64-encoded image data.
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=(
                "You are an OCR engine for medical documents. Transcribe ALL the "
                "text in the image exactly as written. Return only the transcribed "
                "text — no commentary, headings, or markdown."
            ),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": "Transcribe the text in this document."},
                    ],
                }
            ],
        )
    except Exception as exc:  # network error, bad key, unsupported media, etc.
        raise LLMUnavailable(f"Claude vision OCR call failed: {exc}") from exc

    return "".join(block.text for block in response.content if block.type == "text").strip()
