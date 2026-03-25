"""
Evaluate build_prompt() against multiple OpenAI models.

Sends the prompt to each model, validates the response with LLMOutput,
and reports a pass/fail summary table.

Usage:
    source .venv/bin/activate
    python test_prompt_models.py
"""

import json
import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from main import build_prompt, LLMOutput, CATEGORIES

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODELS = [
    ("gpt-3.5-turbo",  "GPT-3.5 Turbo"),
    ("gpt-4.1-nano",   "GPT-4.1 Nano"),
    ("gpt-4o-mini",    "GPT-4o Mini"),
    ("gpt-4o",         "GPT-4o"),
    ("gpt-5-mini",     "GPT-5 Mini"),
    ("gpt-5",          "GPT-5"),
]

SAMPLE_TRANSCRIPT = (
    "Welcome everyone. Today we'll cover the Q3 results. "
    "Revenue grew 12% quarter over quarter, driven by the enterprise segment. "
    "I want to highlight three areas: first, our customer acquisition improved — "
    "we onboarded 340 new accounts. Second, churn dropped to 4.1%. "
    "Third, our NPS score rose to 62. "
    "Now, I know some of you had questions about the APAC expansion timeline. "
    "Let me address that — we expect to launch in Singapore by Q1 next year. "
    "Any other questions? ... Great, thanks for your time."
)

CHECKS = [
    ("json_parseable",     "Response is valid JSON"),
    ("pydantic_valid",     "Passes LLMOutput validation"),
    ("all_cats_present",   "All 5 categories present in scores"),
    ("scores_are_floats",  "Every non-null score is a float"),
    ("scores_in_range",    "Every non-null score is between 1.0 and 5.0"),
    ("confidence_valid",   "Confidence is high/medium/low"),
    ("no_extra_text",      "No markdown fences or extra text around JSON"),
]


def run_checks(raw_response: str) -> dict[str, bool]:
    results = {check_id: False for check_id, _ in CHECKS}

    raw = raw_response.strip()
    if raw.startswith("```"):
        results["no_extra_text"] = False
        raw = raw.strip("`").removeprefix("json").strip()
    else:
        results["no_extra_text"] = True

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return results
    results["json_parseable"] = True

    try:
        parsed = LLMOutput(**data)
        results["pydantic_valid"] = True
    except (ValidationError, Exception):
        results["pydantic_valid"] = False
        scores = data.get("scores", {})
        results["all_cats_present"] = all(c in scores for c in CATEGORIES)
        results["scores_are_floats"] = all(
            isinstance(scores.get(c), (int, float)) or scores.get(c) is None
            for c in CATEGORIES
        )
        results["scores_in_range"] = all(
            scores.get(c) is None or (isinstance(scores.get(c), (int, float)) and 1.0 <= scores[c] <= 5.0)
            for c in CATEGORIES
        )
        conf = data.get("confidence", "")
        results["confidence_valid"] = conf in ("high", "medium", "low")
        return results

    results["all_cats_present"] = True
    results["scores_are_floats"] = all(
        isinstance(getattr(parsed, c).score, float) or getattr(parsed, c).score is None
        for c in CATEGORIES
    )
    results["scores_in_range"] = all(
        getattr(parsed, c).score is None or 1.0 <= getattr(parsed, c).score <= 5.0
        for c in CATEGORIES
    )
    results["confidence_valid"] = parsed.confidence in ("high", "medium", "low")
    return results


def call_model(model_id: str, prompt: str) -> str | None:
    is_reasoning = model_id.startswith("gpt-5") or model_id.startswith("o")
    try:
        params = dict(
            model=model_id,
            messages=[
                {"role": "developer", "content": prompt} if is_reasoning
                    else {"role": "system", "content": prompt},
                {"role": "user", "content": "Please analyze the transcript provided."},
            ],
        )
        if is_reasoning:
            params["max_completion_tokens"] = 4096
        else:
            params["temperature"] = 0.0
            params["max_tokens"] = 1024
        resp = client.chat.completions.create(**params)
        content = resp.choices[0].message.content
        if not content:
            return f"API_ERROR: Empty response (finish_reason={resp.choices[0].finish_reason})"
        return content
    except Exception as e:
        return f"API_ERROR: {e}"


def main():
    prompt = build_prompt(SAMPLE_TRANSCRIPT)
    all_results: dict[str, dict] = {}
    col_width = max(len(label) for _, label in MODELS) + 2

    print("=" * 70)
    print("  PROMPT EVALUATION ACROSS OPENAI MODELS")
    print("=" * 70)

    for model_id, label in MODELS:
        print(f"\n▶ Testing {label} ({model_id})...", end=" ", flush=True)
        start = time.time()
        raw = call_model(model_id, prompt)
        elapsed = time.time() - start

        if raw and raw.startswith("API_ERROR"):
            print(f"SKIPPED ({raw})")
            all_results[label] = {"error": raw, "elapsed": elapsed}
            continue

        checks = run_checks(raw)
        passed = sum(checks.values())
        total = len(checks)
        print(f"done in {elapsed:.1f}s  ({passed}/{total} checks passed)")

        all_results[label] = {"checks": checks, "elapsed": elapsed, "raw": raw}

    # ── Summary table ─────────────────────────────────────────────────────

    print("\n")
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    header = f"{'Model':<{col_width}}"
    for check_id, _ in CHECKS:
        header += f" {check_id[:12]:>12}"
    header += f" {'PASS RATE':>10}"
    print(header)
    print("-" * len(header))

    for label in [l for _, l in MODELS]:
        entry = all_results.get(label)
        if not entry or "error" in entry:
            err = entry.get("error", "unknown") if entry else "unknown"
            short_err = err[:50]
            print(f"{label:<{col_width}}  !! {short_err}")
            continue

        checks = entry["checks"]
        row = f"{label:<{col_width}}"
        passed = 0
        for check_id, _ in CHECKS:
            ok = checks.get(check_id, False)
            passed += ok
            row += f" {'✓':>12}" if ok else f" {'✗':>12}"
        rate = f"{passed}/{len(CHECKS)}"
        row += f" {rate:>10}"
        print(row)

    # ── Per-model score dump ──────────────────────────────────────────────

    print("\n")
    print("=" * 70)
    print("  PARSED SCORES PER MODEL")
    print("=" * 70)

    for label in [l for _, l in MODELS]:
        entry = all_results.get(label)
        if not entry or "error" in entry:
            continue
        raw = entry.get("raw", "")
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").removeprefix("json").strip()
        try:
            data = json.loads(cleaned)
            scores = data.get("scores", {})
            conf = data.get("confidence", "n/a")
            print(f"\n{label}  (confidence: {conf})")
            for cat in CATEGORIES:
                val = scores.get(cat, "MISSING")
                print(f"  {cat:<20} {val}")
        except json.JSONDecodeError:
            print(f"\n{label}  — could not parse JSON")
            preview = raw[:500] if raw else "(empty)"
            print(f"  Raw response (first 500 chars):\n  {preview}")


if __name__ == "__main__":
    main()
