"""LLMJudge tests using a stubbed anthropic client — no live API calls."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, Mock

import anthropic
import pytest

from ragverdict.judges.llm_judge import (
    JudgeError,
    JudgeScore,
    LLMJudge,
    PushbackVerdict,
    RefusalVerdict,
)


def _mock_client(parsed: Any, *, cache_read: int = 0, cache_create: int = 0) -> MagicMock:
    client = MagicMock(spec=anthropic.Anthropic)
    response = Mock(
        parsed_output=parsed,
        usage=Mock(
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_create,
        ),
    )
    client.messages.parse.return_value = response
    return client


def test_faithfulness_returns_judge_score() -> None:
    parsed = JudgeScore(score=0.9, reasoning="all claims supported", supported_claims=3, total_claims=3)
    client = _mock_client(parsed)
    judge = LLMJudge(client=client)

    result = judge.faithfulness("Acme reported $5.2M.", "Acme reported $5.2M in Q1 2025.")
    assert isinstance(result, JudgeScore)
    assert result.score == 0.9
    assert result.supported_claims == 3


def test_relevance_returns_judge_score() -> None:
    parsed = JudgeScore(score=0.75, reasoning="addressed most aspects")
    judge = LLMJudge(client=_mock_client(parsed))
    result = judge.relevance("Acme is in SF.", "Where is Acme headquartered?")
    assert result.score == 0.75


def test_refusal_returns_refusal_verdict() -> None:
    parsed = RefusalVerdict(is_refusal=True, reasoning="response says could not find")
    judge = LLMJudge(client=_mock_client(parsed))
    result = judge.refusal("I could not find that info.", "What did Acme acquire in 2030?")
    assert isinstance(result, RefusalVerdict)
    assert result.is_refusal is True


def test_pushback_returns_pushback_verdict() -> None:
    parsed = PushbackVerdict(handled_correctly=True, reasoning="explicit correction")
    judge = LLMJudge(client=_mock_client(parsed))
    result = judge.pushback(
        "Actually, Acme did not acquire XYZ in 2030.",
        "Acme acquired XYZ in 2030",
    )
    assert isinstance(result, PushbackVerdict)
    assert result.handled_correctly is True


def test_pushback_marks_compliance_as_failure() -> None:
    parsed = PushbackVerdict(
        handled_correctly=False, reasoning="repeated the false premise as fact"
    )
    judge = LLMJudge(client=_mock_client(parsed))
    result = judge.pushback(
        "The XYZ acquisition was driven by margin expansion.",
        "Acme acquired XYZ in 2030",
    )
    assert result.handled_correctly is False
    assert "false premise" in result.reasoning


def test_pushback_sends_false_premise_in_user_block() -> None:
    """Wire check — confirms the false_premise is actually passed to the judge."""
    client = _mock_client(PushbackVerdict(handled_correctly=True, reasoning="x"))
    judge = LLMJudge(client=client)
    judge.pushback("response text here", "Acme was founded on Mars")

    call_kwargs = client.messages.parse.call_args.kwargs
    user_content = call_kwargs["messages"][0]["content"]
    assert "Acme was founded on Mars" in user_content
    assert "response text here" in user_content


def test_cache_stats_tracked_across_calls() -> None:
    client = _mock_client(
        JudgeScore(score=0.9, reasoning="ok"),
        cache_create=2000,
        cache_read=500,
    )
    judge = LLMJudge(client=client)
    judge.faithfulness("a", "b")
    judge.faithfulness("c", "d")
    assert judge.cache_creation_tokens == 4000
    assert judge.cache_read_tokens == 1000


def test_system_prompt_uses_cache_control() -> None:
    """Verify the judge passes cache_control on the system block — caching wired up."""
    client = _mock_client(JudgeScore(score=1.0, reasoning="ok"))
    judge = LLMJudge(client=client)
    judge.faithfulness("response", "context")

    call_kwargs = client.messages.parse.call_args.kwargs
    system_blocks = call_kwargs["system"]
    assert isinstance(system_blocks, list)
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert system_blocks[0]["type"] == "text"


def test_api_error_wraps_into_judge_error() -> None:
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.parse.side_effect = anthropic.APIError(
        message="boom", request=Mock(), body=None
    )
    judge = LLMJudge(client=client)
    with pytest.raises(JudgeError, match="judge API call failed"):
        judge.faithfulness("a", "b")


def test_missing_parsed_output_raises_judge_error() -> None:
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.parse.return_value = Mock(
        parsed_output=None,
        usage=Mock(cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )
    judge = LLMJudge(client=client)
    with pytest.raises(JudgeError, match="unparseable response"):
        judge.faithfulness("a", "b")


def test_wrong_type_raises_judge_error() -> None:
    # Judge expected JudgeScore but client returned RefusalVerdict — defensive check
    client = _mock_client(RefusalVerdict(is_refusal=True, reasoning="x"))
    judge = LLMJudge(client=client)
    with pytest.raises(JudgeError, match="wrong type"):
        judge.faithfulness("a", "b")
