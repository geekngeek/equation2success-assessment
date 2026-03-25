# pip install fastapi pydantic uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from typing import Optional, Literal
import json, random, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ■■ MOCK LLM — do not modify ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Intentionally misbehaves depending on the mode you pass in:
#   "good"          → all 5 fields present, scores 1.0–5.0
#   "bad_range"     → scores outside the valid range (e.g. 6.4, 7.9)
#   "missing"       → omits 2 of the 5 required fields
#   "wrong_type"    → returns scores as strings instead of floats
#   "null_category" → one category is null — LLM could not assess it

CATEGORIES = ["clarity", "engagement", "structure", "delivery", "responsiveness"]


def call_llm(transcript: str, mode: str = "good") -> str:
    if mode == "good":
        scores = {c: round(random.uniform(1.0, 5.0), 1) for c in CATEGORIES}
        return json.dumps({
            "scores": scores,
            "confidence": "high",
            "timestamp_evidence": {"clarity": "0:02:14"},
        })
    elif mode == "bad_range":
        scores = {c: round(random.uniform(5.5, 9.0), 1) for c in CATEGORIES}
        return json.dumps({"scores": scores, "confidence": "medium"})
    elif mode == "missing":
        partial = {c: round(random.uniform(1.0, 5.0), 1) for c in CATEGORIES[:3]}
        return json.dumps({"scores": partial, "confidence": "low"})
    elif mode == "wrong_type":
        scores = {c: random.choice(["excellent", "good", "fair"]) for c in CATEGORIES}
        return json.dumps({"scores": scores})
    elif mode == "null_category":
        scores = {c: round(random.uniform(1.0, 5.0), 1) for c in CATEGORIES}
        scores["responsiveness"] = None
        return json.dumps({
            "scores": scores,
            "null_reason": {"responsiveness": "Insufficient evidence"},
        })


# ■■ YOUR TASK ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#
# Implement the three functions below.
# The mock LLM and the API endpoint are already wired up — do not modify them.

def build_prompt(transcript: str) -> str:
    """
    Return the system prompt you would send to the LLM.
    It should instruct the model to return structured JSON containing:
      - A score (float, 1.0-5.0) for each of the 5 categories
      - A confidence level: high | medium | low
      - Optional timestamp evidence for each category
      - Instructions for what to do when a category cannot be assessed
    """
    return f"""\
<role>
You are an expert communication analyst specializing in transcript evaluation.
Your task is to assess a transcript across five predefined categories and return
a strictly structured JSON response. You are precise, evidence-based, and never
fabricate scores when evidence is insufficient.
</role>

<task>
Analyze the transcript below and produce a single JSON object scoring it on
exactly these five categories: clarity, engagement, structure, delivery,
responsiveness.
</task>

<schema>
Return ONLY a valid JSON object matching this exact structure — no markdown
fences, no commentary, no text outside the JSON braces:

{{
  "scores": {{
    "clarity":        <float | null>,
    "engagement":     <float | null>,
    "structure":      <float | null>,
    "delivery":       <float | null>,
    "responsiveness": <float | null>
  }},
  "confidence": "<high | medium | low>",
  "timestamp_evidence": {{
    "<category>": "<HH:MM:SS or MM:SS timestamp>"
  }},
  "null_reason": {{
    "<category>": "<brief explanation>"
  }}
}}
</schema>

<constraints>
1. SCORES — Each score MUST be a float between 1.0 and 5.0 inclusive (e.g. 3.2).
   Never return integers, strings, booleans, or values outside this range.
2. ALL FIVE categories MUST appear in "scores". Never omit a category.
3. CONFIDENCE — Must be exactly one of: "high", "medium", or "low".
   - "high":   transcript is clear and all categories are well-evidenced.
   - "medium": some ambiguity exists or evidence is thin for 1-2 categories.
   - "low":    transcript is short, noisy, or largely unassessable.
4. TIMESTAMP EVIDENCE — Optional. When you can pinpoint a moment in the
   transcript that supports a score, include it in "timestamp_evidence" as
   "category": "timestamp". Omit the field entirely if no timestamps apply.
5. UNASSESSABLE CATEGORIES — If the transcript provides insufficient evidence
   to assess a category, set that category's score to null (not 0, not a
   string — the JSON literal null). Then include a "null_reason" entry mapping
   that category to a one-sentence explanation. Omit "null_reason" entirely
   if all categories are assessable.
6. OUTPUT — Return raw JSON only. No markdown code fences. No explanatory text
   before or after the JSON. The response must be directly parseable by
   json.loads().
</constraints>

<scoring_rubric>
Use this rubric to calibrate your scores consistently:
  1.0 — Very poor: fundamental deficiencies, nearly no evidence of competence.
  2.0 — Below average: significant weaknesses outweigh strengths.
  3.0 — Average: adequate performance with a mix of strengths and weaknesses.
  4.0 — Good: clear strengths with only minor weaknesses.
  5.0 — Excellent: outstanding performance with no meaningful weaknesses.
Intermediate values (e.g. 2.7, 4.3) are encouraged for nuanced assessment.
</scoring_rubric>

<few_shot_example>
INPUT: A 10-minute meeting transcript where the speaker is well-organized but
speaks in monotone and does not respond to audience questions.

OUTPUT:
{{
  "scores": {{
    "clarity": 4.2,
    "engagement": 2.5,
    "structure": 4.6,
    "delivery": 2.1,
    "responsiveness": null
  }},
  "confidence": "medium",
  "timestamp_evidence": {{
    "clarity": "0:02:14",
    "structure": "0:05:30"
  }},
  "null_reason": {{
    "responsiveness": "No audience interaction present in the transcript"
  }}
}}
</few_shot_example>

<transcript>
{transcript}
</transcript>"""


class CategoryScore(BaseModel):
    """
    Pydantic model for a single category result.
    Must handle: valid floats, null values, and wrong types gracefully.
    """
    score: Optional[float] = None

    @field_validator("score", mode="before")
    @classmethod
    def validate_score(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            raise ValueError(f"Expected numeric score, got string: '{v}'")
        v = float(v)
        if not 1.0 <= v <= 5.0:
            raise ValueError(f"Score {v} is outside the valid range [1.0, 5.0]")
        return v


class LLMOutput(BaseModel):
    """
    Full validation model for the LLM response.
    Must include all 5 categories, confidence, and optional timestamp evidence.
    Must reject or handle every failure mode the mock LLM can produce.
    """
    clarity: CategoryScore
    engagement: CategoryScore
    structure: CategoryScore
    delivery: CategoryScore
    responsiveness: CategoryScore
    confidence: Literal["high", "medium", "low"] = "low"
    timestamp_evidence: Optional[dict[str, str]] = None
    null_reason: Optional[dict[str, str]] = None

    @model_validator(mode="before")
    @classmethod
    def unpack_scores(cls, data):
        """Reshape the flat {scores: {cat: val}} dict into per-field CategoryScore dicts."""
        if isinstance(data, dict) and "scores" in data:
            scores = data.pop("scores")
            if not isinstance(scores, dict):
                raise ValueError("'scores' field must be a dictionary")
            for cat in CATEGORIES:
                if cat not in scores:
                    raise ValueError(f"Missing required category: '{cat}'")
                data[cat] = {"score": scores[cat]}
        return data


def process_transcript(transcript: str, llm_mode: str = "good") -> dict | None:
    """
    Call call_llm(), parse the JSON response, validate with LLMOutput.
    Retry up to 3 times if validation fails — log what went wrong each time.
    On the 3rd failure, return a graceful fallback dict instead of raising.
    Never let an unhandled exception reach the caller.
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            raw = call_llm(transcript, mode=llm_mode)
            data = json.loads(raw)
            result = LLMOutput(**data)
            return result.model_dump()
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as e:
            logger.warning("[Attempt %d/%d] Validation failed: %s", attempt, max_retries, e)
        except Exception as e:
            logger.error("[Attempt %d/%d] Unexpected error: %s", attempt, max_retries, e)

    logger.error("All %d attempts failed for mode '%s'; returning fallback", max_retries, llm_mode)
    return {
        "clarity": {"score": None},
        "engagement": {"score": None},
        "structure": {"score": None},
        "delivery": {"score": None},
        "responsiveness": {"score": None},
        "confidence": "none",
        "error": f"Analysis failed after {max_retries} attempts",
    }


# ■■ API ENDPOINT — do not modify ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■


@app.post("/analyze")
def analyze(payload: dict):
    transcript = payload.get("transcript", "sample transcript")
    mode = payload.get("mode", "good")
    result = process_transcript(transcript, llm_mode=mode)
    if result is None:
        raise HTTPException(status_code=422, detail="Analysis unavailable")
    return result


# ■■ TEST ALL 5 MODES (run: python main.py) ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■


if __name__ == "__main__":
    for mode in ["good", "bad_range", "missing", "wrong_type", "null_category"]:
        print(f"\n--- mode: {mode} ---")
        result = process_transcript("test transcript", llm_mode=mode)
        print(result)
