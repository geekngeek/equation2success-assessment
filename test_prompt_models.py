"""
Evaluate build_prompt() against multiple OpenAI models.

Sends the prompt to each model, validates the response with 22 checks,
and reports a pass/fail summary table.

Usage:
    source .venv/bin/activate
    python test_prompt_models.py
"""

import json
import os
import re
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
    # ── Structural checks ──
    ("no_extra_text",       "No markdown fences or preamble around JSON"),
    ("json_parseable",      "Response is valid JSON"),
    ("top_level_object",    "Top-level JSON is an object, not array/string/number"),
    ("has_scores_key",      "Contains 'scores' top-level key"),
    ("scores_is_dict",      "'scores' value is a dictionary"),
    # ── Category completeness ──
    ("all_cats_present",    "All 5 required categories present in scores"),
    ("no_extra_categories", "No unexpected categories beyond the required 5"),
    # ── Score type checks ──
    ("scores_are_numeric",  "Every non-null score is int or float (not string/bool)"),
    ("scores_are_floats",   "Every non-null score is a float with decimal precision"),
    ("no_string_scores",    "No score is a stringified number like '4.5'"),
    # ── Score range checks ──
    ("scores_in_range",     "Every non-null score is between 1.0 and 5.0"),
    ("no_zero_scores",      "No score is 0 or 0.0"),
    ("no_negative_scores",  "No score is negative"),
    # ── Confidence checks ──
    ("confidence_present",  "'confidence' key exists in the response"),
    ("confidence_valid",    "Confidence is exactly one of: high, medium, low"),
    # ── Null handling checks ──
    ("null_has_reason",     "Every null score has a corresponding null_reason entry"),
    ("reason_matches_null", "null_reason only references categories that are actually null"),
    # ── Optional field checks ──
    ("ts_evidence_valid",   "timestamp_evidence is a dict (or absent), not a list/string"),
    ("ts_format_valid",     "Timestamp values match M:SS, MM:SS, or H:MM:SS format"),
    ("null_reason_valid",   "null_reason is a dict (or absent), not a list/string"),
    # ── Pydantic validation ──
    ("pydantic_valid",      "Full response passes LLMOutput Pydantic validation"),
    # ── Size check ──
    ("response_reasonable", "Response is under 2000 characters (no verbose padding)"),
]


def run_checks(raw_response: str) -> dict[str, bool]:
    results = {cid: False for cid, _ in CHECKS}

    raw = raw_response.strip()

    # ── no_extra_text ──
    results["no_extra_text"] = not raw.startswith("```") and not raw.startswith("{\"") is False
    if raw.startswith("{") or raw.startswith("["):
        results["no_extra_text"] = True
    elif raw.startswith("```"):
        results["no_extra_text"] = False
        raw = raw.strip("`").removeprefix("json").strip()
    else:
        results["no_extra_text"] = False

    # ── response_reasonable ──
    results["response_reasonable"] = len(raw) < 2000

    # ── json_parseable ──
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return results
    results["json_parseable"] = True

    # ── top_level_object ──
    results["top_level_object"] = isinstance(data, dict)
    if not isinstance(data, dict):
        return results

    # ── has_scores_key ──
    results["has_scores_key"] = "scores" in data

    # ── scores_is_dict ──
    scores = data.get("scores", {})
    results["scores_is_dict"] = isinstance(scores, dict)
    if not isinstance(scores, dict):
        return results

    # ── all_cats_present ──
    results["all_cats_present"] = all(c in scores for c in CATEGORIES)

    # ── no_extra_categories ──
    results["no_extra_categories"] = all(c in CATEGORIES for c in scores.keys())

    # ── Score values ──
    score_vals = {c: scores.get(c) for c in CATEGORIES if c in scores}
    non_null = {c: v for c, v in score_vals.items() if v is not None}

    # ── scores_are_numeric ──
    results["scores_are_numeric"] = all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null.values()
    )

    # ── scores_are_floats ──
    results["scores_are_floats"] = all(
        isinstance(v, float) or (isinstance(v, int) and not isinstance(v, bool))
        for v in non_null.values()
    )

    # ── no_string_scores ──
    results["no_string_scores"] = not any(isinstance(v, str) for v in score_vals.values())

    # ── scores_in_range ──
    results["scores_in_range"] = all(
        1.0 <= v <= 5.0 for v in non_null.values() if isinstance(v, (int, float))
    )

    # ── no_zero_scores ──
    results["no_zero_scores"] = all(v != 0 for v in non_null.values())

    # ── no_negative_scores ──
    results["no_negative_scores"] = all(
        v > 0 for v in non_null.values() if isinstance(v, (int, float))
    )

    # ── confidence_present ──
    results["confidence_present"] = "confidence" in data

    # ── confidence_valid ──
    results["confidence_valid"] = data.get("confidence") in ("high", "medium", "low")

    # ── null handling ──
    null_cats = {c for c, v in score_vals.items() if v is None}
    null_reason = data.get("null_reason")

    if not null_cats:
        results["null_has_reason"] = True
        results["reason_matches_null"] = null_reason is None or (
            isinstance(null_reason, dict) and len(null_reason) == 0
        )
    else:
        if isinstance(null_reason, dict):
            results["null_has_reason"] = all(c in null_reason for c in null_cats)
            results["reason_matches_null"] = all(c in null_cats for c in null_reason.keys())
        else:
            results["null_has_reason"] = False
            results["reason_matches_null"] = False

    # ── ts_evidence_valid ──
    ts = data.get("timestamp_evidence")
    if ts is None:
        results["ts_evidence_valid"] = True
        results["ts_format_valid"] = True
    elif isinstance(ts, dict):
        results["ts_evidence_valid"] = True
        ts_pattern = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
        results["ts_format_valid"] = all(
            isinstance(v, str) and ts_pattern.match(v) for v in ts.values()
        ) if ts else True
    else:
        results["ts_evidence_valid"] = False
        results["ts_format_valid"] = False

    # ── null_reason_valid ──
    if null_reason is None:
        results["null_reason_valid"] = True
    else:
        results["null_reason_valid"] = isinstance(null_reason, dict)

    # ── pydantic_valid ──
    try:
        LLMOutput(**data)
        results["pydantic_valid"] = True
    except (ValidationError, Exception):
        results["pydantic_valid"] = False

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

    print("=" * 80)
    print("  PROMPT EVALUATION ACROSS OPENAI MODELS")
    print(f"  {len(CHECKS)} checks per model")
    print("=" * 80)

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
    print("=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    # Group checks by area for readability
    check_groups = [
        ("STRUCTURAL", [
            "no_extra_text", "json_parseable", "top_level_object",
            "has_scores_key", "scores_is_dict",
        ]),
        ("COMPLETENESS", [
            "all_cats_present", "no_extra_categories",
        ]),
        ("SCORE TYPES", [
            "scores_are_numeric", "scores_are_floats", "no_string_scores",
        ]),
        ("SCORE RANGE", [
            "scores_in_range", "no_zero_scores", "no_negative_scores",
        ]),
        ("CONFIDENCE", [
            "confidence_present", "confidence_valid",
        ]),
        ("NULL HANDLING", [
            "null_has_reason", "reason_matches_null",
        ]),
        ("OPTIONAL FIELDS", [
            "ts_evidence_valid", "ts_format_valid", "null_reason_valid",
        ]),
        ("VALIDATION", [
            "pydantic_valid", "response_reasonable",
        ]),
    ]

    model_labels = [l for _, l in MODELS]

    for group_name, check_ids in check_groups:
        print(f"\n  {group_name}")
        print(f"  {'Check':<28}", end="")
        for label in model_labels:
            print(f" {label:>14}", end="")
        print()
        print("  " + "-" * (28 + 15 * len(model_labels)))

        for cid in check_ids:
            desc = next((d for c, d in CHECKS if c == cid), cid)
            row = f"  {cid:<28}"
            for label in model_labels:
                entry = all_results.get(label)
                if not entry or "error" in entry:
                    row += f" {'SKIP':>14}"
                else:
                    ok = entry["checks"].get(cid, False)
                    row += f" {'✓':>14}" if ok else f" {'✗':>14}"
            print(row)

    # ── Pass rate per model ───────────────────────────────────────────────

    print(f"\n  {'PASS RATE':<28}", end="")
    for label in model_labels:
        entry = all_results.get(label)
        if not entry or "error" in entry:
            print(f" {'SKIP':>14}", end="")
        else:
            passed = sum(entry["checks"].values())
            print(f" {f'{passed}/{len(CHECKS)}':>14}", end="")
    print()

    # ── Per-model score dump ──────────────────────────────────────────────

    print("\n")
    print("=" * 80)
    print("  PARSED SCORES PER MODEL")
    print("=" * 80)

    for label in model_labels:
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
            nr = data.get("null_reason", {})
            ts = data.get("timestamp_evidence", {})
            print(f"\n{label}  (confidence: {conf})")
            for cat in CATEGORIES:
                val = scores.get(cat, "MISSING")
                extra = ""
                if val is None and isinstance(nr, dict) and cat in nr:
                    extra = f"  ← {nr[cat]}"
                if isinstance(ts, dict) and cat in ts:
                    extra += f"  [ts: {ts[cat]}]"
                print(f"  {cat:<20} {str(val):<8}{extra}")
        except json.JSONDecodeError:
            print(f"\n{label}  — could not parse JSON")
            preview = raw[:500] if raw else "(empty)"
            print(f"  Raw response (first 500 chars):\n  {preview}")

    # ── Failed checks detail ──────────────────────────────────────────────

    any_failures = False
    for label in model_labels:
        entry = all_results.get(label)
        if not entry or "error" in entry:
            continue
        failed = [cid for cid, ok in entry["checks"].items() if not ok]
        if failed:
            if not any_failures:
                print("\n")
                print("=" * 80)
                print("  FAILED CHECKS DETAIL")
                print("=" * 80)
                any_failures = True
            print(f"\n{label}:")
            for cid in failed:
                desc = next((d for c, d in CHECKS if c == cid), cid)
                print(f"  ✗ {cid:<28} {desc}")


if __name__ == "__main__":
    main()
