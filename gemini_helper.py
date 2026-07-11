"""
Gemini reasoning layer for ClaimPilot.

Used ONLY for reasoning/summarization/explanation/report text — never as a
predictive engine. The CNN and ANN models make the actual predictions;
Gemini reads their outputs plus the claim's free-text description and
produces a human-readable summary and a cross-modal consistency check.
"""
import json
import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

_MODEL_NAME = "gemini-2.5-flash"
_configured = False


def _ensure_configured():
    global _configured
    if not _configured:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Check your .env file.")
        genai.configure(api_key=api_key)
        _configured = True


def analyze_claim_text(description: str, cnn_result: dict, ann_result: dict) -> dict:
    """
    description: free-text claim description typed by the adjuster
    cnn_result: {"severity": str, "confidence": float}
    ann_result: {"risk_score": float}  # 0-1 probability

    Returns: {"summary": str, "consistency_flag": str, "recommendation": str}
    """
    _ensure_configured()
    model = genai.GenerativeModel(_MODEL_NAME)

    prompt = f"""You are an assistant helping a human insurance adjuster review a claim.
You do NOT make the final decision — the CNN and ANN models already produced the
predictions below, and a human will approve/reject/modify. Your job is only to
summarize, cross-check, and explain in plain English.

Claim description (written by the claimant/adjuster):
\"\"\"{description}\"\"\"

Model findings (already computed, not to be re-predicted by you):
- Image damage-severity model says: {cnn_result['severity']} (confidence {cnn_result['confidence']:.0%})
- Tabular risk model says: {ann_result['risk_score']:.0%} estimated fraud/review risk

Respond ONLY with a JSON object with exactly these keys:
- "summary": a 2-3 sentence plain-English summary of the claim description
- "consistency_flag": one of "consistent", "minor mismatch", "significant mismatch" —
  comparing what the description says against the model findings above
- "recommendation": a short paragraph (2-4 sentences) explaining, in plain English,
  what an adjuster should consider, given all three inputs. Do not say "approve" or
  "reject" outright — frame it as considerations for the human reviewer.
"""

    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    return json.loads(response.text)


if __name__ == "__main__":
    result = analyze_claim_text(
        description="Someone hit my car in the parking lot, there's a small dent and scratch on the rear bumper.",
        cnn_result={"severity": "severe", "confidence": 0.91},
        ann_result={"risk_score": 0.72},
    )
    print(json.dumps(result, indent=2))
