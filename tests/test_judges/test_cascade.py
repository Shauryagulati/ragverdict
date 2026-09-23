"""CascadeJudge with stub judges."""

from __future__ import annotations

import pytest

from ragverdict.judges.base import Judge, JudgeError, JudgeScore, PushbackVerdict, RefusalVerdict
from ragverdict.judges.cascade import CascadeJudge


class Stub:
    def __init__(self, *, p: float, confidence: float | None = None, fail: bool = False) -> None:
        self.p = p
        self.confidence = confidence
        self.fail = fail
        self.calls = 0

    def _tick(self) -> None:
        self.calls += 1
        if self.fail:
            raise JudgeError("stub failure")

    def faithfulness(self, response_text: str, retrieved_context: str) -> JudgeScore:
        self._tick()
        return JudgeScore(score=self.p, reasoning=f"stub p={self.p}", confidence=self.confidence)

    def relevance(self, response_text: str, query: str) -> JudgeScore:
        self._tick()
        return JudgeScore(score=self.p, reasoning="stub", confidence=self.confidence)

    def refusal(self, response_text: str, query: str) -> RefusalVerdict:
        self._tick()
        return RefusalVerdict(is_refusal=self.p >= 0.5, reasoning="stub", confidence=self.confidence)

    def pushback(self, response_text: str, false_premise: str) -> PushbackVerdict:
        self._tick()
        return PushbackVerdict(handled_correctly=self.p >= 0.5, reasoning="stub", confidence=self.confidence)


def test_is_a_judge() -> None:
    assert isinstance(CascadeJudge(Stub(p=0.9), Stub(p=0.1)), Judge)


def test_confident_primary_is_returned() -> None:
    primary, fallback = Stub(p=0.95), Stub(p=0.1)
    judge = CascadeJudge(primary, fallback)
    assert judge.faithfulness("r", "c").score == 0.95
    assert fallback.calls == 0
    assert judge.escalations == 0 and judge.calls == 1


def test_uncertain_primary_escalates() -> None:
    primary, fallback = Stub(p=0.5), Stub(p=0.25)
    judge = CascadeJudge(primary, fallback)
    result = judge.faithfulness("r", "c")
    assert result.score == 0.25
    assert result.reasoning.startswith("escalated: ")
    assert fallback.calls == 1
    assert judge.escalation_rate == 1.0


def test_band_edges_are_exclusive() -> None:
    fallback = Stub(p=0.0)
    judge = CascadeJudge(Stub(p=0.3), fallback, band=(0.3, 0.7))
    judge.faithfulness("r", "c")
    judge = CascadeJudge(Stub(p=0.7), fallback, band=(0.3, 0.7))
    judge.faithfulness("r", "c")
    assert fallback.calls == 0


def test_verdict_escalation_uses_confidence() -> None:
    # is_refusal=True with confidence 0.55 -> P(yes)=0.55 -> in band
    fallback = Stub(p=0.0)
    judge = CascadeJudge(Stub(p=0.9, confidence=0.55), fallback)
    assert judge.refusal("r", "q").is_refusal is False
    # is_refusal=False with confidence 0.9 -> P(yes)=0.1 -> not in band
    judge2 = CascadeJudge(Stub(p=0.1, confidence=0.9), Stub(p=1.0))
    assert judge2.refusal("r", "q").is_refusal is False
    assert judge2.escalations == 0


def test_verdict_without_confidence_is_never_escalated() -> None:
    fallback = Stub(p=1.0)
    judge = CascadeJudge(Stub(p=0.0, confidence=None), fallback)
    judge.pushback("r", "p")
    assert fallback.calls == 0


def test_fallback_error_propagates() -> None:
    judge = CascadeJudge(Stub(p=0.5), Stub(p=0.5, fail=True))
    with pytest.raises(JudgeError, match="stub failure"):
        judge.faithfulness("r", "c")


def test_invalid_band_rejected() -> None:
    with pytest.raises(ValueError, match="cascade band"):
        CascadeJudge(Stub(p=0.5), Stub(p=0.5), band=(0.7, 0.3))


def test_escalation_rate_with_no_calls_is_zero() -> None:
    assert CascadeJudge(Stub(p=0.5), Stub(p=0.5)).escalation_rate == 0.0
