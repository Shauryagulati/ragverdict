"""LLMJudge — Day 3 fills this in. Stub kept here so the typed API is stable."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JudgeScore:
    """A judge's score for one dimension (faithfulness or relevance)."""

    score: float
    reasoning: str
    supported_claims: int = 0
    total_claims: int = 0


@dataclass
class RefusalVerdict:
    """Did the response refuse to answer, and why?"""

    is_refusal: bool
    reasoning: str


class LLMJudge:
    """Anthropic-backed judge. Day 3 implements the actual calls."""

    def __init__(self, *, model: str) -> None:
        self.model = model

    def faithfulness(self, response_text: str, retrieved_context: str) -> JudgeScore:
        raise NotImplementedError("LLMJudge.faithfulness is implemented in Day 3.")

    def relevance(self, response_text: str, query: str) -> JudgeScore:
        raise NotImplementedError("LLMJudge.relevance is implemented in Day 3.")

    def refusal(self, response_text: str, query: str) -> RefusalVerdict:
        raise NotImplementedError("LLMJudge.refusal is implemented in Day 3.")
