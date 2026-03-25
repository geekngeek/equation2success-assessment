# pip install fastapi pydantic uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import json, random

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
    pass  # TODO


class CategoryScore(BaseModel):
    """
    Pydantic model for a single category result.
    Must handle: valid floats, null values, and wrong types gracefully.
    """
    pass  # TODO


class LLMOutput(BaseModel):
    """
    Full validation model for the LLM response.
    Must include all 5 categories, confidence, and optional timestamp evidence.
    Must reject or handle every failure mode the mock LLM can produce.
    """
    pass  # TODO


def process_transcript(transcript: str, llm_mode: str = "good") -> dict | None:
    """
    Call call_llm(), parse the JSON response, validate with LLMOutput.
    Retry up to 3 times if validation fails — log what went wrong each time.
    On the 3rd failure, return a graceful fallback dict instead of raising.
    Never let an unhandled exception reach the caller.
    """
    pass  # TODO


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
