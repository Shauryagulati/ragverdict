"""Cascade judge: a cheap probabilistic judge first, an LLM only when it is unsure.

The band is deliberately a fixed interval, not a learned router: measure the
simplest version first. Edges are exclusive — p == lo or p == hi is "confident".
"""

from __future__ import annotations

import threading
from typing import TypeVar

from pydantic import BaseModel

from ragverdict.judges.base import Judge, JudgeScore, PushbackVerdict, RefusalVerdict

R = TypeVar("R", bound=BaseModel)


def _p_yes(flag: bool, confidence: float | None) -> float | None:
    """Recover P(yes) from a boolean verdict and its confidence, if known."""
    if confidence is None:
        return None
    return confidence if flag else 1.0 - confidence


class CascadeJudge:
    """Ask `primary`; if its P(yes) falls strictly inside `band`, ask `fallback` instead."""

    def __init__(
        self, primary: Judge, fallback: Judge, *, band: tuple[float, float] = (0.3, 0.7)
    ) -> None:
        lo, hi = band
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"cascade band must satisfy 0 <= lo < hi <= 1, got {band}")
        self.primary = primary
        self.fallback = fallback
        self.lo, self.hi = lo, hi
        self._lock = threading.Lock()
        self.calls = 0
        self.escalations = 0

    @property
    def escalation_rate(self) -> float:
        return self.escalations / self.calls if self.calls else 0.0

    def faithfulness(self, response_text: str, retrieved_context: str) -> JudgeScore:
        first = self.primary.faithfulness(response_text, retrieved_context)
        if self._escalate(first.score):
            return _mark(self.fallback.faithfulness(response_text, retrieved_context))
        return first

    def relevance(self, response_text: str, query: str) -> JudgeScore:
        first = self.primary.relevance(response_text, query)
        if self._escalate(first.score):
            return _mark(self.fallback.relevance(response_text, query))
        return first

    def refusal(self, response_text: str, query: str) -> RefusalVerdict:
        first = self.primary.refusal(response_text, query)
        if self._escalate(_p_yes(first.is_refusal, first.confidence)):
            return _mark(self.fallback.refusal(response_text, query))
        return first

    def pushback(self, response_text: str, false_premise: str) -> PushbackVerdict:
        first = self.primary.pushback(response_text, false_premise)
        if self._escalate(_p_yes(first.handled_correctly, first.confidence)):
            return _mark(self.fallback.pushback(response_text, false_premise))
        return first

    def _escalate(self, p: float | None) -> bool:
        escalate = p is not None and self.lo < p < self.hi
        with self._lock:
            self.calls += 1
            if escalate:
                self.escalations += 1
        return escalate


def _mark(result: R) -> R:
    reasoning = getattr(result, "reasoning", "")
    return result.model_copy(update={"reasoning": f"escalated: {reasoning}"})
