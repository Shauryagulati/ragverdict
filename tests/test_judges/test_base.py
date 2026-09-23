"""Judge protocol + shared result models."""

from __future__ import annotations

from unittest.mock import MagicMock

import anthropic

from ragverdict.judges import base, llm_judge
from ragverdict.judges.base import Judge, JudgeScore, PushbackVerdict, RefusalVerdict
from ragverdict.judges.llm_judge import LLMJudge


def test_llm_judge_satisfies_judge_protocol() -> None:
    judge = LLMJudge(client=MagicMock(spec=anthropic.Anthropic))
    assert isinstance(judge, Judge)


def test_result_models_reexported_from_llm_judge() -> None:
    assert llm_judge.JudgeScore is base.JudgeScore
    assert llm_judge.RefusalVerdict is base.RefusalVerdict
    assert llm_judge.PushbackVerdict is base.PushbackVerdict
    assert llm_judge.JudgeError is base.JudgeError


def test_confidence_defaults_to_none() -> None:
    assert JudgeScore(score=0.5, reasoning="x").confidence is None
    assert RefusalVerdict(is_refusal=True, reasoning="x").confidence is None
    assert PushbackVerdict(handled_correctly=True, reasoning="x").confidence is None


def test_object_missing_a_method_is_not_a_judge() -> None:
    class Partial:
        def faithfulness(self, response_text: str, retrieved_context: str) -> JudgeScore:
            return JudgeScore(score=1.0, reasoning="x")

    assert not isinstance(Partial(), Judge)
