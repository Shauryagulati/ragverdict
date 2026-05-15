"""rag_quality — hard assertions plus judge-driven WEAK verdicts.

Per case:
  1. Run the adapter against `query`.
  2. Hard assertions (must_mention, must_not_cite, expects_citations) → FAIL on any miss.
  3. must_refuse → judge.refusal() when available, refusal-hint heuristic otherwise.
  4. With a judge: score faithfulness against retrieved_context and relevance against
     the query. PASS / WEAK / FAIL based on thresholds.
  5. Without a judge: hard-assertion-passing cases return PASS (no WEAK transition).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag_eval.evaluators import register
from rag_eval.evaluators.base import Evaluator, TestResult, Verdict

if TYPE_CHECKING:
    from rag_eval.adapters.base import RagAdapter, RagResponse
    from rag_eval.config import TestSpec, Thresholds
    from rag_eval.judges.llm_judge import JudgeScore, LLMJudge


_CITATION_RE = re.compile(r"\[src:[^\]]+\]", re.IGNORECASE)

# Heuristic fallback when no judge is available. Day 3 keeps this as a guard for
# offline / API-keyless runs; with a judge wired, refusal() replaces it.
_REFUSAL_HINTS = (
    "cannot",
    "could not find",
    "i don't have",
    "i do not have",
    "no information",
    "not aware",
    "unable to",
    "i'm not sure",
)


class RagQualityCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    expects_citations: bool = False
    must_mention: list[str] = Field(default_factory=list)
    must_not_cite: bool = False
    must_refuse: bool = False


class RagQualitySpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cases: list[RagQualityCase] = Field(default_factory=list)


@dataclass
class CaseOutcome:
    case: RagQualityCase
    verdict: Verdict
    detail: str
    response_text: str
    metrics: dict[str, float]


@register
class RagQualityEvaluator(Evaluator):
    name = "rag_quality"

    def run(
        self,
        adapter: RagAdapter,
        spec: TestSpec,
        *,
        judge: LLMJudge | None,
        thresholds: Thresholds,
    ) -> TestResult:
        t0 = time.perf_counter()
        try:
            rq_spec = RagQualitySpec.model_validate(spec.model_dump())
        except ValidationError as exc:
            return TestResult(
                name=spec.name,
                evaluator=self.name,
                verdict=Verdict.ERROR,
                detail=f"invalid rag_quality spec: {exc}",
                duration_ms=_elapsed_ms(t0),
            )

        if not rq_spec.cases:
            return TestResult(
                name=spec.name,
                evaluator=self.name,
                verdict=Verdict.ERROR,
                detail="no cases defined for rag_quality",
                duration_ms=_elapsed_ms(t0),
            )

        outcomes: list[CaseOutcome] = []
        for case in rq_spec.cases:
            try:
                response = adapter.query(case.query)
            except Exception as exc:
                outcomes.append(
                    CaseOutcome(
                        case=case,
                        verdict=Verdict.ERROR,
                        detail=f"adapter raised: {exc}",
                        response_text="",
                        metrics={},
                    )
                )
                continue
            outcomes.append(_grade_case(case, response, judge, thresholds))

        worst = _worst_verdict([o.verdict for o in outcomes])
        passed = sum(1 for o in outcomes if o.verdict == Verdict.PASS)
        weak = sum(1 for o in outcomes if o.verdict == Verdict.WEAK)
        detail = f"{passed}/{len(outcomes)} cases passed"
        if weak:
            detail += f", {weak} weak"
        if worst not in (Verdict.PASS, Verdict.WEAK):
            failure_summary = "; ".join(
                f"{o.case.query[:40]!r}: {o.detail}"
                for o in outcomes
                if o.verdict not in (Verdict.PASS, Verdict.WEAK)
            )
            detail += f" — {failure_summary}"

        return TestResult(
            name=spec.name,
            evaluator=self.name,
            verdict=worst,
            detail=detail,
            metrics={
                "cases_total": float(len(outcomes)),
                "cases_passed": float(passed),
                "cases_weak": float(weak),
            },
            duration_ms=_elapsed_ms(t0),
            artifacts={
                "cases": [
                    {
                        "query": o.case.query,
                        "verdict": o.verdict.value,
                        "detail": o.detail,
                        "response_text": o.response_text,
                        "metrics": o.metrics,
                    }
                    for o in outcomes
                ],
            },
        )


def _grade_case(
    case: RagQualityCase,
    response: RagResponse,
    judge: LLMJudge | None,
    thresholds: Thresholds,
) -> CaseOutcome:
    text = response.text

    # 1. must_refuse — judge if available, hint heuristic otherwise.
    if case.must_refuse:
        if judge is not None:
            try:
                refusal_verdict = judge.refusal(text, case.query)
            except Exception as exc:
                return CaseOutcome(
                    case=case,
                    verdict=Verdict.ERROR,
                    detail=f"judge.refusal() failed: {exc}",
                    response_text=text,
                    metrics={},
                )
            if not refusal_verdict.is_refusal:
                return CaseOutcome(
                    case=case,
                    verdict=Verdict.FAIL,
                    detail=f"expected refusal — {refusal_verdict.reasoning}",
                    response_text=text,
                    metrics={},
                )
        else:
            if not any(hint in text.lower() for hint in _REFUSAL_HINTS):
                return CaseOutcome(
                    case=case,
                    verdict=Verdict.FAIL,
                    detail="expected refusal but response answered confidently",
                    response_text=text,
                    metrics={},
                )

    # 2. must_not_cite
    if case.must_not_cite and _CITATION_RE.search(text):
        return CaseOutcome(
            case=case,
            verdict=Verdict.FAIL,
            detail="expected no citations but response cited sources",
            response_text=text,
            metrics={},
        )

    # 3. must_mention
    lower = text.lower()
    missing = [s for s in case.must_mention if s.lower() not in lower]
    if missing:
        return CaseOutcome(
            case=case,
            verdict=Verdict.FAIL,
            detail=f"missing required substring(s): {missing}",
            response_text=text,
            metrics={},
        )

    # 4. expects_citations
    if case.expects_citations and not response.citations:
        return CaseOutcome(
            case=case,
            verdict=Verdict.FAIL,
            detail="expected at least one citation but none returned",
            response_text=text,
            metrics={},
        )

    # 5. Judge-driven WEAK transitions for must_refuse cases with no other failures.
    if case.must_refuse:
        # For refusal cases, hard assertions cover the contract — no judge scoring needed.
        return CaseOutcome(
            case=case,
            verdict=Verdict.PASS,
            detail="refusal verified",
            response_text=text,
            metrics={},
        )

    if judge is None:
        return CaseOutcome(
            case=case,
            verdict=Verdict.PASS,
            detail="hard assertions passed (no judge configured)",
            response_text=text,
            metrics={},
        )

    # 6. Judge scoring for answer cases.
    context_blob = "\n\n---\n\n".join(c.chunk for c in response.retrieved_context)
    try:
        faithfulness = judge.faithfulness(text, context_blob)
        relevance = judge.relevance(text, case.query)
    except Exception as exc:
        return CaseOutcome(
            case=case,
            verdict=Verdict.ERROR,
            detail=f"judge call failed: {exc}",
            response_text=text,
            metrics={},
        )

    metrics = {
        "faithfulness": faithfulness.score,
        "relevance": relevance.score,
    }
    verdict = _verdict_from_scores(faithfulness, relevance, thresholds)
    if verdict == Verdict.PASS:
        detail = f"faithfulness={faithfulness.score:.2f}, relevance={relevance.score:.2f}"
    elif verdict == Verdict.WEAK:
        detail = (
            f"WEAK — faithfulness={faithfulness.score:.2f} ({faithfulness.reasoning}), "
            f"relevance={relevance.score:.2f} ({relevance.reasoning})"
        )
    else:
        detail = (
            f"FAIL — faithfulness={faithfulness.score:.2f} ({faithfulness.reasoning}), "
            f"relevance={relevance.score:.2f} ({relevance.reasoning})"
        )
    return CaseOutcome(
        case=case,
        verdict=verdict,
        detail=detail,
        response_text=text,
        metrics=metrics,
    )


def _verdict_from_scores(
    faithfulness: JudgeScore,
    relevance: JudgeScore,
    thresholds: Thresholds,
) -> Verdict:
    if (
        faithfulness.score < thresholds.faithfulness_weak
        or relevance.score < thresholds.relevance_weak
    ):
        return Verdict.FAIL
    if (
        faithfulness.score < thresholds.faithfulness_pass
        or relevance.score < thresholds.relevance_pass
    ):
        return Verdict.WEAK
    return Verdict.PASS


def _worst_verdict(verdicts: list[Verdict]) -> Verdict:
    if not verdicts:
        return Verdict.ERROR
    rank = {Verdict.PASS: 0, Verdict.WEAK: 1, Verdict.FAIL: 2, Verdict.ERROR: 3}
    return max(verdicts, key=lambda v: rank[v])


def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)
