from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rag_eval.adapters.base import Citation, Message, RagAdapter, RagResponse
from rag_eval.config import TestSpec, Thresholds
from rag_eval.evaluators.base import Verdict
from rag_eval.evaluators.rag_quality import RagQualityEvaluator


@dataclass
class FakeAdapter(RagAdapter):
    handler: Callable[[str], RagResponse] = lambda p: RagResponse(text="")

    def query(
        self,
        prompt: str,
        *,
        conversation: list[Message] | None = None,
    ) -> RagResponse:
        return self.handler(prompt)


def _spec(**kw: Any) -> TestSpec:
    return TestSpec(name="t", evaluator="rag_quality", **kw)


def test_pass_when_must_mention_present() -> None:
    adapter = FakeAdapter(handler=lambda p: RagResponse(text="Acme reported $5.2M in Q1 2025."))
    spec = _spec(cases=[{"query": "Q1 revenue?", "must_mention": ["$5.2M"]}])
    result = RagQualityEvaluator().run(adapter, spec, judge=None, thresholds=Thresholds())
    assert result.verdict == Verdict.PASS


def test_fail_when_must_mention_missing() -> None:
    adapter = FakeAdapter(handler=lambda p: RagResponse(text="Acme reported strong growth."))
    spec = _spec(cases=[{"query": "Q1 revenue?", "must_mention": ["$5.2M"]}])
    result = RagQualityEvaluator().run(adapter, spec, judge=None, thresholds=Thresholds())
    assert result.verdict == Verdict.FAIL
    assert "missing required substring" in result.detail


def test_must_refuse_pass_when_response_refuses() -> None:
    adapter = FakeAdapter(
        handler=lambda p: RagResponse(text="I cannot answer — no information in corpus.")
    )
    spec = _spec(cases=[{"query": "What did Acme acquire in 2030?", "must_refuse": True}])
    result = RagQualityEvaluator().run(adapter, spec, judge=None, thresholds=Thresholds())
    assert result.verdict == Verdict.PASS


def test_must_refuse_fail_when_response_answers_confidently() -> None:
    adapter = FakeAdapter(
        handler=lambda p: RagResponse(text="Acme acquired Globex Corp in March 2030.")
    )
    spec = _spec(cases=[{"query": "What did Acme acquire in 2030?", "must_refuse": True}])
    result = RagQualityEvaluator().run(adapter, spec, judge=None, thresholds=Thresholds())
    assert result.verdict == Verdict.FAIL
    assert "expected refusal" in result.detail


def test_must_not_cite_fail_when_response_cites() -> None:
    # Response correctly refuses, but it cites a source while doing so — must_not_cite catches it.
    adapter = FakeAdapter(
        handler=lambda p: RagResponse(
            text="I cannot answer — no information in corpus [src:UNKNOWN]."
        )
    )
    spec = _spec(cases=[{"query": "what?", "must_refuse": True, "must_not_cite": True}])
    result = RagQualityEvaluator().run(adapter, spec, judge=None, thresholds=Thresholds())
    assert result.verdict == Verdict.FAIL
    assert "no citations" in result.detail


def test_expects_citations_fail_when_none_returned() -> None:
    adapter = FakeAdapter(handler=lambda p: RagResponse(text="acme reported $5.2M", citations=[]))
    spec = _spec(
        cases=[{"query": "Q1?", "must_mention": ["$5.2M"], "expects_citations": True}]
    )
    result = RagQualityEvaluator().run(adapter, spec, judge=None, thresholds=Thresholds())
    assert result.verdict == Verdict.FAIL
    assert "expected at least one citation" in result.detail


def test_expects_citations_pass_when_citations_returned() -> None:
    adapter = FakeAdapter(
        handler=lambda p: RagResponse(
            text="acme reported $5.2M",
            citations=[Citation(id="c1", source_id="REV", span="$5.2M")],
        )
    )
    spec = _spec(
        cases=[{"query": "Q1?", "must_mention": ["$5.2M"], "expects_citations": True}]
    )
    result = RagQualityEvaluator().run(adapter, spec, judge=None, thresholds=Thresholds())
    assert result.verdict == Verdict.PASS


def test_aggregates_worst_verdict_across_cases() -> None:
    def handler(prompt: str) -> RagResponse:
        if "good" in prompt:
            return RagResponse(text="good answer")
        return RagResponse(text="bad answer")  # missing required substring

    adapter = FakeAdapter(handler=handler)
    spec = _spec(
        cases=[
            {"query": "good case?", "must_mention": ["good"]},
            {"query": "missing case?", "must_mention": ["MISSING"]},
        ]
    )
    result = RagQualityEvaluator().run(adapter, spec, judge=None, thresholds=Thresholds())
    assert result.verdict == Verdict.FAIL
    assert result.metrics["cases_passed"] == 1


def test_error_when_no_cases() -> None:
    adapter = FakeAdapter(handler=lambda p: RagResponse(text=""))
    spec = _spec()  # no cases field
    result = RagQualityEvaluator().run(adapter, spec, judge=None, thresholds=Thresholds())
    assert result.verdict == Verdict.ERROR
