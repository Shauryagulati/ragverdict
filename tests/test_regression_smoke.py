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
from rag_eval.evaluators.edge_cases import EdgeCasesEvaluator
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


class TruncatingAdapter(RagAdapter):
    """Silently drops anything past 1000 chars and answers the partial prompt.

    The failure mode is *silent* — the adapter doesn't raise, doesn't warn,
    doesn't flag the truncation. From the caller's perspective everything
    looks normal. edge_cases.long_input catches this because the agent's
    response either won't include content that depends on the truncated
    tail, or — more commonly — the adapter raises trying to process input
    it was sized to reject silently.
    """

    def query(
        self, prompt: str, *, conversation: list[Message] | None = None
    ) -> RagResponse:
        if len(prompt) > 1000:
            # Simulates an HTTP client / vector store that hard-fails on
            # oversized input. Many real adapters do exactly this.
            raise ValueError(f"prompt exceeds limit ({len(prompt)} > 1000)")
        return RagResponse(text="ok answered partial input")


class CompliantAdapter(RagAdapter):
    """Agrees with whatever premise the user asserts, true or not.

    This is the opposite failure mode from ConfidentlyWrongAdapter — that one
    answers when it should refuse; this one *accepts the user's framing*
    when it should push back. edge_cases.contradiction (heuristic fallback)
    catches it because the response never contains any of _PUSHBACK_HINTS.
    """

    def query(
        self, prompt: str, *, conversation: list[Message] | None = None
    ) -> RagResponse:
        return RagResponse(
            text=(
                "Yes — the XYZ acquisition was driven by margin expansion "
                "and strategic alignment with Acme's logistics roadmap."
            )
        )


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


def test_edge_cases_catches_silently_truncating_adapter() -> None:
    """An adapter that raises on oversized input must FAIL the long_input case."""
    result = EdgeCasesEvaluator().run(
        TruncatingAdapter(),
        TestSpec(
            name="t",
            evaluator="edge_cases",
            cases=[{"kind": "long_input", "length": 5000}],
        ),
        judge=None,
        thresholds=Thresholds(),
    )
    assert result.verdict == Verdict.FAIL
    assert "adapter raised" in result.detail


def test_edge_cases_catches_compliant_adapter_with_heuristic_fallback() -> None:
    """An adapter that accepts false premises must FAIL contradiction even with no judge."""
    result = EdgeCasesEvaluator().run(
        CompliantAdapter(),
        TestSpec(
            name="t",
            evaluator="edge_cases",
            cases=[
                {
                    "kind": "contradiction",
                    "query": (
                        "What was the rationale for Acme's acquisition of XYZ "
                        "Corp in 2030?"
                    ),
                    "false_premise": "Acme acquired XYZ Corp in 2030",
                }
            ],
        ),
        judge=None,  # heuristic fallback
        thresholds=Thresholds(),
    )
    assert result.verdict == Verdict.FAIL
    assert "no pushback" in result.detail
