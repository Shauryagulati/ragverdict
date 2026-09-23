"""citation_audit — verify every [src:ID] citation against the real corpus.

Two layers of check:

1. **Dangling-citation check** — every `Citation.source_id` must resolve to a doc
   returned by `adapter.corpus()`. A dangling citation is a hard FAIL.
2. **Support check** (judge-driven) — for each resolved citation, ask the judge
   whether the source doc actually supports the cited span. Aggregate to a mean
   `support_score`. PASS / WEAK / FAIL based on the configured thresholds.

Without a judge configured, only the dangling check runs.
"""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragverdict.evaluators import register
from ragverdict.evaluators.base import Evaluator, TestResult, Verdict

if TYPE_CHECKING:
    from ragverdict.adapters.base import RagAdapter, SourceDoc
    from ragverdict.config import TestSpec, Thresholds
    from ragverdict.judges.base import Judge


class CitationAuditSpec(BaseModel):
    """Citation audit always sources queries from config (V0)."""

    model_config = ConfigDict(extra="ignore")

    sample_queries: list[str] = Field(default_factory=list)


@register
class CitationAuditEvaluator(Evaluator):
    name = "citation_audit"

    def run(
        self,
        adapter: RagAdapter,
        spec: TestSpec,
        *,
        judge: Judge | None,
        thresholds: Thresholds,
    ) -> TestResult:
        t0 = time.perf_counter()

        try:
            ca_spec = CitationAuditSpec.model_validate(spec.model_dump())
        except ValidationError as exc:
            return TestResult(
                name=spec.name,
                evaluator=self.name,
                verdict=Verdict.ERROR,
                detail=f"invalid citation_audit spec: {exc}",
                duration_ms=_elapsed_ms(t0),
            )

        corpus = adapter.corpus()
        if corpus is None:
            return TestResult(
                name=spec.name,
                evaluator=self.name,
                verdict=Verdict.ERROR,
                detail="adapter.corpus() returned None — citation_audit requires a corpus",
                duration_ms=_elapsed_ms(t0),
            )
        corpus_index: dict[str, SourceDoc] = {doc.source_id: doc for doc in corpus}

        if not ca_spec.sample_queries:
            return TestResult(
                name=spec.name,
                evaluator=self.name,
                verdict=Verdict.ERROR,
                detail="no sample_queries defined for citation_audit",
                duration_ms=_elapsed_ms(t0),
            )

        per_citation: list[dict[str, Any]] = []
        dangling = 0
        support_scores: list[float] = []

        for query in ca_spec.sample_queries:
            try:
                response = adapter.query(query)
            except Exception as exc:
                per_citation.append(
                    {"query": query, "error": f"adapter raised: {exc}"}
                )
                continue

            if not response.citations:
                # No citations to audit — record but don't fail. This is one query's contribution
                # to the run; the rag_quality evaluator already polices expects_citations.
                per_citation.append({"query": query, "note": "no citations returned"})
                continue

            for citation in response.citations:
                source = corpus_index.get(citation.source_id)
                if source is None:
                    dangling += 1
                    per_citation.append(
                        {
                            "query": query,
                            "citation_id": citation.id,
                            "source_id": citation.source_id,
                            "resolved": False,
                            "detail": "source_id not in corpus (dangling citation)",
                        }
                    )
                    continue

                entry: dict[str, Any] = {
                    "query": query,
                    "citation_id": citation.id,
                    "source_id": citation.source_id,
                    "resolved": True,
                    "span": citation.span,
                }
                if judge is not None and citation.span:
                    try:
                        score = judge.faithfulness(citation.span, source.content)
                    except Exception as exc:
                        entry["error"] = f"judge call failed: {exc}"
                    else:
                        entry["support_score"] = score.score
                        entry["reasoning"] = score.reasoning
                        support_scores.append(score.score)
                per_citation.append(entry)

        total_resolved = sum(1 for e in per_citation if e.get("resolved") is True)
        total_audited = total_resolved + dangling
        verdict, detail = _verdict_for_citations(
            dangling=dangling,
            total_audited=total_audited,
            support_scores=support_scores,
            thresholds=thresholds,
            judge_present=judge is not None,
        )

        metrics: dict[str, float] = {
            "citations_audited": float(total_audited),
            "citations_dangling": float(dangling),
        }
        if support_scores:
            metrics["mean_support_score"] = statistics.fmean(support_scores)

        return TestResult(
            name=spec.name,
            evaluator=self.name,
            verdict=verdict,
            detail=detail,
            metrics=metrics,
            duration_ms=_elapsed_ms(t0),
            artifacts={"per_citation": per_citation},
        )


def _verdict_for_citations(
    *,
    dangling: int,
    total_audited: int,
    support_scores: list[float],
    thresholds: Thresholds,
    judge_present: bool,
) -> tuple[Verdict, str]:
    if total_audited == 0:
        return Verdict.ERROR, "no citations were returned across sample queries"

    if dangling > 0:
        return (
            Verdict.FAIL,
            f"{dangling}/{total_audited} citations are dangling (source_id not in corpus)",
        )

    if not judge_present or not support_scores:
        return (
            Verdict.PASS,
            f"{total_audited} citations resolved; no judge configured for support check",
        )

    mean = statistics.fmean(support_scores)
    if mean < thresholds.citation_support_weak:
        return (
            Verdict.FAIL,
            f"{total_audited} resolved, but mean support_score={mean:.2f} < "
            f"{thresholds.citation_support_weak}",
        )
    if mean < thresholds.citation_support_pass:
        return (
            Verdict.WEAK,
            f"{total_audited} resolved, mean support_score={mean:.2f} below "
            f"{thresholds.citation_support_pass} threshold",
        )
    return (
        Verdict.PASS,
        f"{total_audited} citations resolved, mean support_score={mean:.2f}",
    )


def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)
