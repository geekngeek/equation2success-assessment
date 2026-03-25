import pytest
import json
import math
from unittest.mock import patch
from pydantic import ValidationError
from main import CategoryScore, LLMOutput, process_transcript, CATEGORIES, call_llm


# ── CategoryScore ─────────────────────────────────────────────────────────────

class TestCategoryScoreValid:
    def test_mid_range(self):
        cs = CategoryScore(score=3.5)
        assert cs.score == 3.5

    def test_lower_boundary(self):
        cs = CategoryScore(score=1.0)
        assert cs.score == 1.0

    def test_upper_boundary(self):
        cs = CategoryScore(score=5.0)
        assert cs.score == 5.0

    def test_null_score(self):
        cs = CategoryScore(score=None)
        assert cs.score is None

    def test_default_is_none(self):
        cs = CategoryScore()
        assert cs.score is None

    def test_integer_in_range_coerced(self):
        cs = CategoryScore(score=3)
        assert cs.score == 3.0
        assert isinstance(cs.score, float)

    def test_integer_at_lower_bound(self):
        cs = CategoryScore(score=1)
        assert cs.score == 1.0

    def test_integer_at_upper_bound(self):
        cs = CategoryScore(score=5)
        assert cs.score == 5.0

    def test_one_decimal_place(self):
        cs = CategoryScore(score=2.7)
        assert cs.score == 2.7

    def test_many_decimal_places(self):
        cs = CategoryScore(score=3.14159)
        assert abs(cs.score - 3.14159) < 1e-9

    def test_boolean_true_coerces_to_1(self):
        cs = CategoryScore(score=True)
        assert cs.score == 1.0

    def test_model_dump_null(self):
        cs = CategoryScore(score=None)
        assert cs.model_dump() == {"score": None}

    def test_model_dump_valid(self):
        cs = CategoryScore(score=4.2)
        assert cs.model_dump() == {"score": 4.2}


class TestCategoryScoreInvalid:
    def test_below_range(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            CategoryScore(score=0.5)

    def test_above_range(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            CategoryScore(score=5.1)

    def test_far_above_range(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            CategoryScore(score=8.7)

    def test_zero(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            CategoryScore(score=0.0)

    def test_negative(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            CategoryScore(score=-2.0)

    def test_large_negative(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            CategoryScore(score=-999.0)

    def test_very_large_positive(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            CategoryScore(score=1_000_000.0)

    def test_just_below_lower_boundary(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            CategoryScore(score=0.9999)

    def test_just_above_upper_boundary(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            CategoryScore(score=5.0001)

    def test_string_word(self):
        with pytest.raises(ValidationError, match="got string"):
            CategoryScore(score="excellent")

    def test_string_number(self):
        with pytest.raises(ValidationError, match="got string"):
            CategoryScore(score="3.5")

    def test_empty_string(self):
        with pytest.raises(ValidationError, match="got string"):
            CategoryScore(score="")

    def test_boolean_false_out_of_range(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            CategoryScore(score=False)

    def test_infinity(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            CategoryScore(score=float("inf"))

    def test_negative_infinity(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            CategoryScore(score=float("-inf"))

    def test_nan(self):
        with pytest.raises(ValidationError):
            CategoryScore(score=float("nan"))

    def test_list_as_score(self):
        with pytest.raises(ValidationError):
            CategoryScore(score=[3.5])

    def test_dict_as_score(self):
        with pytest.raises(ValidationError):
            CategoryScore(score={"value": 3.5})


# ── LLMOutput ─────────────────────────────────────────────────────────────────

def _make_scores(value=3.0):
    """Helper: build a valid scores dict, optionally overriding every value."""
    return {c: value for c in CATEGORIES}


class TestLLMOutputValid:
    def test_complete_good_response(self):
        data = {
            "scores": _make_scores(4.0),
            "confidence": "high",
            "timestamp_evidence": {"clarity": "0:02:14"},
        }
        out = LLMOutput(**data)
        assert out.clarity.score == 4.0
        assert out.confidence == "high"
        assert out.timestamp_evidence == {"clarity": "0:02:14"}

    def test_minimal_valid_response(self):
        data = {"scores": _make_scores(2.5), "confidence": "medium"}
        out = LLMOutput(**data)
        assert out.timestamp_evidence is None
        assert out.null_reason is None

    def test_null_category_with_reason(self):
        scores = _make_scores(3.0)
        scores["responsiveness"] = None
        data = {
            "scores": scores,
            "confidence": "low",
            "null_reason": {"responsiveness": "Insufficient evidence"},
        }
        out = LLMOutput(**data)
        assert out.responsiveness.score is None
        assert out.null_reason["responsiveness"] == "Insufficient evidence"

    def test_confidence_defaults_to_low(self):
        data = {"scores": _make_scores(3.0)}
        out = LLMOutput(**data)
        assert out.confidence == "low"

    def test_all_null_scores(self):
        scores = {c: None for c in CATEGORIES}
        data = {"scores": scores, "confidence": "low"}
        out = LLMOutput(**data)
        for cat in CATEGORIES:
            assert getattr(out, cat).score is None

    def test_multiple_null_categories(self):
        scores = _make_scores(3.0)
        scores["delivery"] = None
        scores["responsiveness"] = None
        data = {
            "scores": scores,
            "confidence": "low",
            "null_reason": {
                "delivery": "Audio was muted",
                "responsiveness": "No Q&A segment",
            },
        }
        out = LLMOutput(**data)
        assert out.delivery.score is None
        assert out.responsiveness.score is None
        assert len(out.null_reason) == 2

    def test_all_confidence_levels(self):
        for level in ("high", "medium", "low"):
            data = {"scores": _make_scores(3.0), "confidence": level}
            out = LLMOutput(**data)
            assert out.confidence == level

    def test_extra_fields_ignored(self):
        data = {
            "scores": _make_scores(3.0),
            "confidence": "high",
            "some_unknown_field": "whatever",
            "model_version": "v2",
        }
        out = LLMOutput(**data)
        assert out.confidence == "high"

    def test_timestamp_evidence_multiple_categories(self):
        data = {
            "scores": _make_scores(4.0),
            "confidence": "high",
            "timestamp_evidence": {
                "clarity": "0:02:14",
                "structure": "0:05:30",
                "engagement": "0:10:00",
            },
        }
        out = LLMOutput(**data)
        assert len(out.timestamp_evidence) == 3

    def test_empty_timestamp_evidence_dict(self):
        data = {
            "scores": _make_scores(3.0),
            "confidence": "high",
            "timestamp_evidence": {},
        }
        out = LLMOutput(**data)
        assert out.timestamp_evidence == {}

    def test_empty_null_reason_dict(self):
        data = {
            "scores": _make_scores(3.0),
            "confidence": "high",
            "null_reason": {},
        }
        out = LLMOutput(**data)
        assert out.null_reason == {}

    def test_scores_at_boundaries(self):
        scores = {
            "clarity": 1.0,
            "engagement": 5.0,
            "structure": 1.0,
            "delivery": 5.0,
            "responsiveness": 3.0,
        }
        data = {"scores": scores, "confidence": "medium"}
        out = LLMOutput(**data)
        assert out.clarity.score == 1.0
        assert out.engagement.score == 5.0

    def test_model_dump_round_trip(self):
        data = {
            "scores": _make_scores(4.2),
            "confidence": "high",
            "timestamp_evidence": {"clarity": "0:01:00"},
        }
        out = LLMOutput(**data)
        dumped = out.model_dump()
        assert dumped["clarity"]["score"] == 4.2
        assert dumped["confidence"] == "high"
        assert dumped["timestamp_evidence"]["clarity"] == "0:01:00"

    def test_json_round_trip_good_mode(self):
        raw = call_llm("test", mode="good")
        data = json.loads(raw)
        out = LLMOutput(**data)
        dumped = out.model_dump()
        for cat in CATEGORIES:
            assert cat in dumped

    def test_json_round_trip_null_category_mode(self):
        raw = call_llm("test", mode="null_category")
        data = json.loads(raw)
        out = LLMOutput(**data)
        assert out.responsiveness.score is None


class TestLLMOutputInvalid:
    def test_missing_one_category(self):
        scores = {c: 3.0 for c in CATEGORIES[:4]}
        with pytest.raises(ValidationError, match="Missing required category"):
            LLMOutput(**{"scores": scores, "confidence": "high"})

    def test_missing_two_categories(self):
        scores = {c: 3.0 for c in CATEGORIES[:3]}
        with pytest.raises(ValidationError, match="Missing required category"):
            LLMOutput(**{"scores": scores, "confidence": "high"})

    def test_missing_all_categories(self):
        with pytest.raises(ValidationError, match="Missing required category"):
            LLMOutput(**{"scores": {}, "confidence": "high"})

    def test_single_category_only(self):
        with pytest.raises(ValidationError, match="Missing required category"):
            LLMOutput(**{"scores": {"clarity": 3.0}, "confidence": "high"})

    def test_scores_out_of_range(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            LLMOutput(**{"scores": _make_scores(7.0), "confidence": "medium"})

    def test_scores_as_strings(self):
        scores = {c: "good" for c in CATEGORIES}
        with pytest.raises(ValidationError, match="got string"):
            LLMOutput(**{"scores": scores, "confidence": "high"})

    def test_scores_mixed_strings_and_floats(self):
        scores = _make_scores(3.0)
        scores["clarity"] = "excellent"
        scores["delivery"] = "poor"
        with pytest.raises(ValidationError, match="got string"):
            LLMOutput(**{"scores": scores, "confidence": "high"})

    def test_invalid_confidence_value(self):
        with pytest.raises(ValidationError):
            LLMOutput(**{"scores": _make_scores(3.0), "confidence": "very_high"})

    def test_confidence_numeric(self):
        with pytest.raises(ValidationError):
            LLMOutput(**{"scores": _make_scores(3.0), "confidence": 5})

    def test_confidence_empty_string(self):
        with pytest.raises(ValidationError):
            LLMOutput(**{"scores": _make_scores(3.0), "confidence": ""})

    def test_confidence_none_explicit(self):
        with pytest.raises(ValidationError):
            LLMOutput(**{"scores": _make_scores(3.0), "confidence": None})

    def test_scores_is_a_list(self):
        with pytest.raises(ValidationError, match="must be a dictionary"):
            LLMOutput(**{"scores": [3.0, 3.0, 3.0, 3.0, 3.0], "confidence": "high"})

    def test_scores_is_a_string(self):
        with pytest.raises(ValidationError, match="must be a dictionary"):
            LLMOutput(**{"scores": "all good", "confidence": "high"})

    def test_scores_is_none(self):
        with pytest.raises(ValidationError, match="must be a dictionary"):
            LLMOutput(**{"scores": None, "confidence": "high"})

    def test_scores_is_an_integer(self):
        with pytest.raises(ValidationError, match="must be a dictionary"):
            LLMOutput(**{"scores": 42, "confidence": "high"})

    def test_scores_key_missing_entirely(self):
        with pytest.raises(ValidationError):
            LLMOutput(**{"confidence": "high"})

    def test_mixed_valid_and_invalid_scores(self):
        scores = _make_scores(3.0)
        scores["clarity"] = 9.0
        with pytest.raises(ValidationError, match="outside the valid range"):
            LLMOutput(**{"scores": scores, "confidence": "high"})

    def test_wrong_category_names(self):
        scores = {
            "Clarity": 3.0,
            "Engagement": 3.0,
            "Structure": 3.0,
            "Delivery": 3.0,
            "Responsiveness": 3.0,
        }
        with pytest.raises(ValidationError, match="Missing required category"):
            LLMOutput(**{"scores": scores, "confidence": "high"})

    def test_misspelled_category(self):
        scores = _make_scores(3.0)
        del scores["responsiveness"]
        scores["responsive"] = 3.0
        with pytest.raises(ValidationError, match="Missing required category"):
            LLMOutput(**{"scores": scores, "confidence": "high"})

    def test_completely_empty_dict(self):
        with pytest.raises(ValidationError):
            LLMOutput(**{})

    def test_json_round_trip_bad_range_mode(self):
        raw = call_llm("test", mode="bad_range")
        data = json.loads(raw)
        with pytest.raises(ValidationError, match="outside the valid range"):
            LLMOutput(**data)

    def test_json_round_trip_missing_mode(self):
        raw = call_llm("test", mode="missing")
        data = json.loads(raw)
        with pytest.raises(ValidationError, match="Missing required category"):
            LLMOutput(**data)

    def test_json_round_trip_wrong_type_mode(self):
        raw = call_llm("test", mode="wrong_type")
        data = json.loads(raw)
        with pytest.raises(ValidationError, match="got string"):
            LLMOutput(**data)


# ── process_transcript ────────────────────────────────────────────────────────

class TestProcessTranscriptGoodMode:
    def test_returns_dict(self):
        result = process_transcript("hello", llm_mode="good")
        assert isinstance(result, dict)

    def test_no_error_key(self):
        result = process_transcript("hello", llm_mode="good")
        assert "error" not in result

    def test_has_all_categories(self):
        result = process_transcript("hello", llm_mode="good")
        for cat in CATEGORIES:
            assert cat in result
            assert result[cat]["score"] is not None

    def test_scores_in_range(self):
        result = process_transcript("hello", llm_mode="good")
        for cat in CATEGORIES:
            score = result[cat]["score"]
            assert 1.0 <= score <= 5.0

    def test_confidence_is_valid(self):
        result = process_transcript("hello", llm_mode="good")
        assert result["confidence"] in ("high", "medium", "low")

    def test_has_timestamp_evidence(self):
        result = process_transcript("hello", llm_mode="good")
        assert "timestamp_evidence" in result
        assert result["timestamp_evidence"] is not None

    def test_empty_transcript(self):
        result = process_transcript("", llm_mode="good")
        assert isinstance(result, dict)
        assert "error" not in result

    def test_long_transcript(self):
        long_text = "word " * 10_000
        result = process_transcript(long_text, llm_mode="good")
        assert isinstance(result, dict)
        assert "error" not in result


class TestProcessTranscriptFailureModes:
    def test_bad_range_returns_fallback(self):
        result = process_transcript("hello", llm_mode="bad_range")
        assert "error" in result
        assert "failed" in result["error"].lower()

    def test_missing_returns_fallback(self):
        result = process_transcript("hello", llm_mode="missing")
        assert "error" in result

    def test_wrong_type_returns_fallback(self):
        result = process_transcript("hello", llm_mode="wrong_type")
        assert "error" in result

    def test_fallback_has_all_categories(self):
        result = process_transcript("hello", llm_mode="bad_range")
        for cat in CATEGORIES:
            assert cat in result

    def test_fallback_scores_are_none(self):
        result = process_transcript("hello", llm_mode="bad_range")
        for cat in CATEGORIES:
            assert result[cat]["score"] is None

    def test_fallback_confidence_is_none_string(self):
        result = process_transcript("hello", llm_mode="bad_range")
        assert result["confidence"] == "none"

    def test_fallback_error_mentions_3_attempts(self):
        result = process_transcript("hello", llm_mode="bad_range")
        assert "3" in result["error"]


class TestProcessTranscriptNullCategory:
    def test_returns_valid_dict(self):
        result = process_transcript("hello", llm_mode="null_category")
        assert "error" not in result

    def test_null_score_for_responsiveness(self):
        result = process_transcript("hello", llm_mode="null_category")
        assert result["responsiveness"]["score"] is None

    def test_other_scores_are_valid(self):
        result = process_transcript("hello", llm_mode="null_category")
        for cat in ["clarity", "engagement", "structure", "delivery"]:
            score = result[cat]["score"]
            assert score is not None
            assert 1.0 <= score <= 5.0

    def test_has_null_reason(self):
        result = process_transcript("hello", llm_mode="null_category")
        assert result["null_reason"] is not None
        assert "responsiveness" in result["null_reason"]


class TestProcessTranscriptRobustness:
    def test_never_returns_none(self):
        for mode in ["good", "bad_range", "missing", "wrong_type", "null_category"]:
            result = process_transcript("test", llm_mode=mode)
            assert result is not None, f"Returned None for mode '{mode}'"

    def test_never_raises(self):
        for mode in ["good", "bad_range", "missing", "wrong_type", "null_category"]:
            try:
                process_transcript("test", llm_mode=mode)
            except Exception:
                pytest.fail(f"process_transcript raised on mode '{mode}'")

    def test_unknown_mode_does_not_crash(self):
        try:
            result = process_transcript("test", llm_mode="nonexistent")
            assert result is not None
        except Exception:
            pytest.fail("process_transcript raised on unknown mode")

    def test_multiple_calls_are_independent(self):
        r1 = process_transcript("test", llm_mode="good")
        r2 = process_transcript("test", llm_mode="good")
        assert r1 is not r2
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)

    def test_json_decode_error_handled(self):
        with patch("main.call_llm", return_value="NOT VALID JSON {{{"):
            result = process_transcript("test", llm_mode="good")
            assert result is not None
            assert "error" in result

    def test_call_llm_raises_handled(self):
        with patch("main.call_llm", side_effect=RuntimeError("boom")):
            result = process_transcript("test", llm_mode="good")
            assert result is not None
            assert "error" in result

    def test_call_llm_returns_empty_string(self):
        with patch("main.call_llm", return_value=""):
            result = process_transcript("test", llm_mode="good")
            assert result is not None
            assert "error" in result

    def test_call_llm_returns_null_json(self):
        with patch("main.call_llm", return_value="null"):
            result = process_transcript("test", llm_mode="good")
            assert result is not None
            assert "error" in result

    def test_call_llm_returns_array_json(self):
        with patch("main.call_llm", return_value="[1, 2, 3]"):
            result = process_transcript("test", llm_mode="good")
            assert result is not None
            assert "error" in result
