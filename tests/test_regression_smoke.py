"""Regression smoke: induce known broken adapter behaviors and confirm rag-eval flags them.

This is the test that proves the framework is doing its job. If rag-eval fails to fail
on these, something has slipped.
"""

from __future__ import annotations

from rag_eval.adapters.base import (
    Citation,
    Message,
    RagAdapter,
    RagResponse,
    SourceDoc,
    ToolSpec,
)
from rag_eval.config import TestSpec, Thresholds
from rag_eval.evaluators.base import Verdict
from rag_eval.evaluators.citation_audit import CitationAuditEvaluator
from rag_eval.evaluators.rag_quality import RagQualityEvaluator
from rag_eval.evaluators.tool_coverage import ToolCoverageEvaluator


class DanglingCitationAdapter(RagAdapter):
    """Always emits a citation that points to a source_id not in the corpus."""

    def corpus(self) -> list[SourceDoc]:
        return [SourceDoc(source_id="REAL", content="Acme reported $5.2M.")]

    def query(
        self, prompt: str, *, conversation: list[Message] | None = None
    ) -> RagResponse:
        return RagResponse(
            text="The answer is X [src:GHOST].",
            citations=[Citation(id="c1", source_id="GHOST", span="The answer is X")],
        )


class SilentToolAdapter(RagAdapter):
    """Exposes a tool but never actually calls it (the failure mode tool_coverage catches)."""

    def available_tools(self) -> list[ToolSpec]:
        return [ToolSpec(name="search", description="search", trigger_prompt="Use search")]

    def query(
        self, prompt: str, *, conversation: list[Message] | None = None
    ) -> RagResponse:
        return RagResponse(text="I will answer without tools.", tool_calls=[])


class ConfidentlyWrongAdapter(RagAdapter):
    """Answers questions confidently even when it should refuse (hallucination)."""

    def query(
        self, prompt: str, *, conversation: list[Message] | None = None
    ) -> RagResponse:
        return RagResponse(text="Acme acquired Globex Corp in March 2030 for $1.2B.")


def test_citation_audit_catches_dangling_citation() -> None:
    result = CitationAuditEvaluator().run(
        DanglingCitationAdapter(),
        TestSpec(name="t", evaluator="citation_audit", sample_queries=["any"]),
        judge=None,
        thresholds=Thresholds(),
    )
    assert result.verdict == Verdict.FAIL
    assert "dangling" in result.detail
    assert result.metrics["citations_dangling"] == 1


def test_tool_coverage_catches_silent_tool() -> None:
    result = ToolCoverageEvaluator().run(
        SilentToolAdapter(),
        TestSpec(name="t", evaluator="tool_coverage"),
        judge=None,
        thresholds=Thresholds(),
    )
    assert result.verdict == Verdict.FAIL
    assert "search" in result.detail
    # Per-tool artifact records the specific failure
    per_tool = result.artifacts["per_tool"]
    assert per_tool[0]["fired"] is False


def test_rag_quality_catches_unhandled_hallucination() -> None:
    """The agent answers an out-of-corpus question confidently — must_refuse catches this."""
    result = RagQualityEvaluator().run(
        ConfidentlyWrongAdapter(),
        TestSpec(
            name="t",
            evaluator="rag_quality",
            cases=[
                {
                    "query": "What did Acme acquire in 2030?",
                    "must_refuse": True,
                    "must_not_cite": True,
                }
            ],
        ),
        judge=None,  # falls back to refusal-hint heuristic
        thresholds=Thresholds(),
    )
    assert result.verdict == Verdict.FAIL
    assert "expected refusal" in result.detail
