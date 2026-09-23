"""LLMJudge tests using a stubbed anthropic client — no live API calls."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import anthropic
import pytest
from pydantic import BaseModel

from ragverdict.judges.llm_judge import (
    JudgeError,
    JudgeScore,
    LLMJudge,
    PushbackVerdict,
    RefusalVerdict,
    output_schema,
    parse_judge_message,
)


def _message(
    payload: BaseModel | str,
    *,
    stop_reason: str = "end_turn",
    cache_read: int = 0,
    cache_create: int = 0,
) -> Mock:
    text = payload if isinstance(payload, str) else payload.model_dump_json()
    return Mock(
        content=[Mock(type="text", text=text)],
        stop_reason=stop_reason,
        usage=Mock(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_create,
        ),
    )


def _mock_client(payload: BaseModel | str, **kwargs: object) -> MagicMock:
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = _message(payload, **kwargs)  # type: ignore[arg-type]
    return client


def test_faithfulness_returns_judge_score() -> None:
    parsed = JudgeScore(score=0.9, reasoning="all claims supported", supported_claims=3, total_claims=3)
    judge = LLMJudge(client=_mock_client(parsed))
    result = judge.faithfulness("Acme reported $5.2M.", "Acme reported $5.2M in Q1 2025.")
    assert isinstance(result, JudgeScore)
    assert result.score == 0.9
    assert result.supported_claims == 3
    assert result.confidence is None


def test_relevance_returns_judge_score() -> None:
    judge = LLMJudge(client=_mock_client(JudgeScore(score=0.75, reasoning="most aspects")))
    assert judge.relevance("Acme is in SF.", "Where is Acme headquartered?").score == 0.75


def test_refusal_returns_refusal_verdict() -> None:
    parsed = RefusalVerdict(is_refusal=True, reasoning="response says could not find")
    judge = LLMJudge(client=_mock_client(parsed))
    result = judge.refusal("I could not find that info.", "What did Acme acquire in 2030?")
    assert isinstance(result, RefusalVerdict)
    assert result.is_refusal is True


def test_pushback_returns_pushback_verdict() -> None:
    parsed = PushbackVerdict(handled_correctly=True, reasoning="explicit correction")
    judge = LLMJudge(client=_mock_client(parsed))
    result = judge.pushback("Actually, Acme did not acquire XYZ in 2030.", "Acme acquired XYZ in 2030")
    assert isinstance(result, PushbackVerdict)
    assert result.handled_correctly is True


def test_pushback_marks_compliance_as_failure() -> None:
    parsed = PushbackVerdict(handled_correctly=False, reasoning="repeated the false premise as fact")
    judge = LLMJudge(client=_mock_client(parsed))
    result = judge.pushback("The XYZ acquisition was driven by margin expansion.", "Acme acquired XYZ in 2030")
    assert result.handled_correctly is False
    assert "false premise" in result.reasoning


def test_pushback_sends_false_premise_in_user_block() -> None:
    client = _mock_client(PushbackVerdict(handled_correctly=True, reasoning="x"))
    LLMJudge(client=client).pushback("response text here", "Acme was founded on Mars")
    user_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Acme was founded on Mars" in user_content
    assert "response text here" in user_content


def test_usage_and_cache_stats_tracked_across_calls() -> None:
    client = _mock_client(JudgeScore(score=0.9, reasoning="ok"), cache_create=2000, cache_read=500)
    judge = LLMJudge(client=client)
    judge.faithfulness("a", "b")
    judge.faithfulness("c", "d")
    assert judge.cache_creation_tokens == 4000
    assert judge.cache_read_tokens == 1000
    assert judge.input_tokens == 200
    assert judge.output_tokens == 40


def test_system_prompt_uses_cache_control() -> None:
    client = _mock_client(JudgeScore(score=1.0, reasoning="ok"))
    LLMJudge(client=client).faithfulness("response", "context")
    system_blocks = client.messages.create.call_args.kwargs["system"]
    assert isinstance(system_blocks, list)
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert system_blocks[0]["type"] == "text"


def test_default_request_has_no_thinking_key_and_4096_max_tokens() -> None:
    client = _mock_client(JudgeScore(score=1.0, reasoning="ok"))
    LLMJudge(client=client).faithfulness("r", "c")
    kwargs = client.messages.create.call_args.kwargs
    assert "thinking" not in kwargs
    assert kwargs["max_tokens"] == 4096


def test_thinking_disabled_is_sent_when_configured() -> None:
    client = _mock_client(JudgeScore(score=1.0, reasoning="ok"))
    LLMJudge(client=client, thinking="disabled").faithfulness("r", "c")
    assert client.messages.create.call_args.kwargs["thinking"] == {"type": "disabled"}


def test_request_uses_json_schema_output_config() -> None:
    client = _mock_client(JudgeScore(score=1.0, reasoning="ok"))
    LLMJudge(client=client).faithfulness("r", "c")
    fmt = client.messages.create.call_args.kwargs["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"] == output_schema(JudgeScore)


def test_faithfulness_request_is_exactly_what_is_sent() -> None:
    client = _mock_client(JudgeScore(score=1.0, reasoning="ok"))
    judge = LLMJudge(client=client, model="claude-sonnet-5", thinking="disabled")
    expected = judge.faithfulness_request("resp", "ctx")
    judge.faithfulness("resp", "ctx")
    assert client.messages.create.call_args.kwargs == expected
    assert expected["model"] == "claude-sonnet-5"


def test_output_schema_strips_confidence_and_forbids_extras() -> None:
    schema = output_schema(JudgeScore)
    assert "confidence" not in schema["properties"]
    assert "confidence" not in schema.get("required", [])
    assert schema["additionalProperties"] is False
    assert {"score", "reasoning"} <= set(schema["properties"])


def test_api_error_wraps_into_judge_error() -> None:
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = anthropic.APIError(message="boom", request=Mock(), body=None)
    with pytest.raises(JudgeError, match="judge API call failed"):
        LLMJudge(client=client).faithfulness("a", "b")


def test_truncated_output_raises_clear_error() -> None:
    client = _mock_client('{"score": 0.5, "reason', stop_reason="max_tokens")
    with pytest.raises(JudgeError, match="truncated"):
        LLMJudge(client=client).faithfulness("a", "b")


def test_refusal_stop_reason_raises() -> None:
    client = _mock_client("", stop_reason="refusal")
    with pytest.raises(JudgeError, match="declined"):
        LLMJudge(client=client).faithfulness("a", "b")


def test_invalid_json_raises_judge_error() -> None:
    client = _mock_client(RefusalVerdict(is_refusal=True, reasoning="x"))  # wrong shape for JudgeScore
    with pytest.raises(JudgeError, match="invalid JSON for JudgeScore"):
        LLMJudge(client=client).faithfulness("a", "b")


def test_missing_text_block_raises() -> None:
    message = Mock(content=[Mock(type="thinking", thinking="")], stop_reason="end_turn")
    with pytest.raises(JudgeError, match="no text block"):
        parse_judge_message(message, JudgeScore)


def test_parse_judge_message_accepts_valid_payload() -> None:
    message = _message(JudgeScore(score=0.25, reasoning="one of four supported"))
    assert parse_judge_message(message, JudgeScore).score == 0.25


def test_faithfulness_prompt_matches_request() -> None:
    from ragverdict.judges.llm_judge import faithfulness_prompt

    judge = LLMJudge(client=MagicMock(spec=anthropic.Anthropic))
    system, user = faithfulness_prompt("resp", "ctx")
    request = judge.faithfulness_request("resp", "ctx")
    assert request["system"][0]["text"] == system
    assert request["messages"][0]["content"] == user
