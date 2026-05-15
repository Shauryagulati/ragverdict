from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rag_eval.adapters.base import (
    Citation,
    Message,
    RagAdapter,
    RagResponse,
    SourceDoc,
)
from rag_eval.config import TestSpec, Thresholds
from rag_eval.evaluators.base import Verdict
from rag_eval.evaluators.citation_audit import CitationAuditEvaluator
from rag_eval.judges.llm_judge import JudgeScore, RefusalVerdict


CORPUS = [
    SourceDoc(source_id="REV", content="Acme reported $5.2M in Q1 2025."),
    SourceDoc(source_id="LEADERSHIP", content="Jane Smith is CEO. Raj Patel is CTO."),
]


@dataclass
class FakeAdapter(RagAdapter):
    handler: Callable[[str], RagResponse]
    corpus_docs: list[SourceDoc] = field(default_factory=lambda: list(CORPUS))

    def corpus(self) -> list[SourceDoc]:
        return self.corpus_docs

    def query(
        self,
        prompt: str,
        *,
        conversation: list[Message] | None = None,
    ) -> RagResponse:
        return self.handler(prompt)


class StubJudge:
    """Returns a fixed support score; mimics the LLMJudge surface."""

    def __init__(self, score: float = 0.98) -> None:
        self._score = score
        self.calls = 0

    def faithfulness(self, response: str, context: str) -> JudgeScore:
        self.calls += 1
        return JudgeScore(score=self._score, reasoning="stub")

    def relevance(self, response: str, query: str) -> JudgeScore:
        return JudgeScore(score=1.0, reasoning="stub")

    def refusal(self, response: str, query: str) -> RefusalVerdict:
        return RefusalVerdict(is_refusal=False, reasoning="stub")


def _spec(**kw: Any) -> TestSpec:
    return TestSpec(name="t", evaluator="citation_audit", **kw)


def test_pass_when_all_citations_resolve_and_support_is_high() -> None:
    def handler(prompt: str) -> RagResponse:
        return RagResponse(
            text="Acme reported $5.2M. [src:REV]",
            citations=[Citation(id="c1", source_id="REV", span="Acme reported $5.2M.")],
        )

    adapter = FakeAdapter(handler=handler)
    judge = StubJudge(score=0.98)
    result = CitationAuditEvaluator().run(
        adapter,
        _spec(sample_queries=["What was Q1 revenue?"]),
        judge=judge,  # type: ignore[arg-type]
        thresholds=Thresholds(),
    )
    assert result.verdict == Verdict.PASS
    assert result.metrics["citations_dangling"] == 0
    assert result.metrics["citations_audited"] == 1
    assert judge.calls == 1


def test_fail_when_citation_is_dangling() -> None:
    def handler(prompt: str) -> RagResponse:
        return RagResponse(
            text="Some made-up answer [src:BOGUS]",
            citations=[Citation(id="c1", source_id="BOGUS", span="made-up")],
        )

    adapter = FakeAdapter(handler=handler)
    judge = StubJudge()
    result = CitationAuditEvaluator().run(
        adapter,
        _spec(sample_queries=["q"]),
        judge=judge,  # type: ignore[arg-type]
        thresholds=Thresholds(),
    )
    assert result.verdict == Verdict.FAIL
    assert result.metrics["citations_dangling"] == 1
    assert "dangling" in result.detail


def test_weak_when_support_score_is_below_pass_threshold() -> None:
    def handler(prompt: str) -> RagResponse:
        return RagResponse(
            text="Acme reported [src:REV]",
            citations=[Citation(id="c1", source_id="REV", span="some span")],
        )

    adapter = FakeAdapter(handler=handler)
    judge = StubJudge(score=0.85)  # in [weak=0.8, pass=0.95)
    result = CitationAuditEvaluator().run(
        adapter,
        _spec(sample_queries=["q"]),
        judge=judge,  # type: ignore[arg-type]
        thresholds=Thresholds(),
    )
    assert result.verdict == Verdict.WEAK


def test_fail_when_support_score_below_weak_threshold() -> None:
    def handler(prompt: str) -> RagResponse:
        return RagResponse(
            text="Acme reported [src:REV]",
            citations=[Citation(id="c1", source_id="REV", span="some span")],
        )

    adapter = FakeAdapter(handler=handler)
    judge = StubJudge(score=0.3)
    result = CitationAuditEvaluator().run(
        adapter,
        _spec(sample_queries=["q"]),
        judge=judge,  # type: ignore[arg-type]
        thresholds=Thresholds(),
    )
    assert result.verdict == Verdict.FAIL
    assert "support_score" in result.detail


def test_pass_without_judge_only_runs_dangling_check() -> None:
    def handler(prompt: str) -> RagResponse:
        return RagResponse(
            text="Acme reported $5.2M [src:REV]",
            citations=[Citation(id="c1", source_id="REV", span="Acme reported $5.2M")],
        )

    adapter = FakeAdapter(handler=handler)
    result = CitationAuditEvaluator().run(
        adapter,
        _spec(sample_queries=["q"]),
        judge=None,
        thresholds=Thresholds(),
    )
    assert result.verdict == Verdict.PASS
    assert "no judge" in result.detail


def test_error_when_corpus_is_none() -> None:
    class NoCorpus(RagAdapter):
        def query(
            self, prompt: str, *, conversation: list[Message] | None = None
        ) -> RagResponse:
            return RagResponse(text="")
        # corpus() returns None by default

    result = CitationAuditEvaluator().run(
        NoCorpus(),
        _spec(sample_queries=["q"]),
        judge=None,
        thresholds=Thresholds(),
    )
    assert result.verdict == Verdict.ERROR
    assert "corpus" in result.detail


def test_error_when_no_sample_queries() -> None:
    adapter = FakeAdapter(handler=lambda p: RagResponse(text=""))
    result = CitationAuditEvaluator().run(
        adapter,
        _spec(),  # no sample_queries
        judge=None,
        thresholds=Thresholds(),
    )
    assert result.verdict == Verdict.ERROR
