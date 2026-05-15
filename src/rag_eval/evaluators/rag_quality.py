"""rag_quality — hard assertions over response text + citations.

Day 2 ships hard assertions only (must_mention, must_not_cite, must_refuse via
keyword fallback). Day 3 wires the LLM judge in to add WEAK verdicts based on
faithfulness/relevance scores and replace the must_refuse heuristic.
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
    from rag_eval.config import TestSpec
    from rag_eval.judges.llm_judge import LLMJudge


_CITATION_RE = re.compile(r"\[src:[^\]]+\]", re.IGNORECASE)

# Day 2 fallback for must_refuse. Day 3 replaces with judge.refusal().
_REFUSAL_HINTS = (
    "cannot",
    "cannot answer",
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


@register
class RagQualityEvaluator(Evaluator):
    name = "rag_quality"

    def run(
        self,
        adapter: RagAdapter,
        spec: TestSpec,
        judge: LLMJudge | None,
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
            except Exception as exc:  # noqa: BLE001
                outcomes.append(
                    CaseOutcome(
                        case=case,
                        verdict=Verdict.ERROR,
                        detail=f"adapter raised: {exc}",
                        response_text="",
                    )
                )
                continue
            outcomes.append(_grade_case(case, response))

        worst = _worst_verdict([o.verdict for o in outcomes])
        passed = sum(1 for o in outcomes if o.verdict == Verdict.PASS)
        detail = f"{passed}/{len(outcomes)} cases passed"
        if worst != Verdict.PASS:
            failure_summary = "; ".join(
                f"{o.case.query[:40]!r}: {o.detail}"
                for o in outcomes
                if o.verdict != Verdict.PASS
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
            },
            duration_ms=_elapsed_ms(t0),
            artifacts={
                "cases": [
                    {
                        "query": o.case.query,
                        "verdict": o.verdict.value,
                        "detail": o.detail,
                        "response_text": o.response_text,
                    }
                    for o in outcomes
                ],
            },
        )


def _grade_case(case: RagQualityCase, response: RagResponse) -> CaseOutcome:
    """Hard assertions only. Day 3 layers judge-driven WEAK verdicts on top."""
    text = response.text
    lower = text.lower()

    # must_refuse: response should indicate uncertainty / no answer
    if case.must_refuse:
        looks_like_refusal = any(hint in lower for hint in _REFUSAL_HINTS)
        if not looks_like_refusal:
            return CaseOutcome(
                case=case,
                verdict=Verdict.FAIL,
                detail="expected refusal but response answered confidently",
                response_text=text,
            )

    # must_not_cite: no [src:*] tokens in the response
    if case.must_not_cite:
        if _CITATION_RE.search(text):
            return CaseOutcome(
                case=case,
                verdict=Verdict.FAIL,
                detail="expected no citations but response cited sources",
                response_text=text,
            )

    # must_mention: every substring must appear (case-insensitive)
    missing = [s for s in case.must_mention if s.lower() not in lower]
    if missing:
        return CaseOutcome(
            case=case,
            verdict=Verdict.FAIL,
            detail=f"missing required substring(s): {missing}",
            response_text=text,
        )

    # expects_citations: at least one citation must appear
    if case.expects_citations and not response.citations:
        return CaseOutcome(
            case=case,
            verdict=Verdict.FAIL,
            detail="expected at least one citation but none returned",
            response_text=text,
        )

    return CaseOutcome(
        case=case,
        verdict=Verdict.PASS,
        detail="all hard assertions passed",
        response_text=text,
    )


def _worst_verdict(verdicts: list[Verdict]) -> Verdict:
    """Worst-wins aggregation. Order: ERROR > FAIL > WEAK > PASS."""
    if not verdicts:
        return Verdict.ERROR
    rank = {Verdict.PASS: 0, Verdict.WEAK: 1, Verdict.FAIL: 2, Verdict.ERROR: 3}
    return max(verdicts, key=lambda v: rank[v])


def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)
