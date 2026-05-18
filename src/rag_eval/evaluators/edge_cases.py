"""edge_cases — input-boundary failure modes V0's golden-path evaluators don't catch.

Four kinds, each modeled as its own Pydantic case type and dispatched via a
discriminated union on `kind`:

- `long_input`     — ≥10K-char prompt; PASS = no crash + non-empty response within timeout
- `multi_turn`     — N-turn conversation; PASS = response contains required substrings
                      (verifies the agent uses conversation history)
- `contradiction`  — query with a false premise; PASS = agent pushes back / corrects
                      (judge-graded; heuristic fallback when --no-judge)
- `empty_input`    — empty prompt; PASS = controlled error or refusal-style response

Verdicts are PASS/FAIL/ERROR only — no WEAK. Edge cases are binary safety
properties, not quality scores in a tunable range.

Spec: docs/design/specs/edge-case-evaluator-design.md
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag_eval.adapters.base import Message
from rag_eval.evaluators import register
from rag_eval.evaluators.base import Evaluator, TestResult, Verdict

if TYPE_CHECKING:
    from rag_eval.adapters.base import RagAdapter
    from rag_eval.config import TestSpec, Thresholds
    from rag_eval.judges.llm_judge import LLMJudge


# --- Pydantic case models -------------------------------------------------


class LongInputCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["long_input"]
    length: int = Field(default=10000, gt=0)
    timeout_s: float = Field(default=30.0, gt=0)


class MultiTurnCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["multi_turn"]
    turns: list[str] = Field(min_length=1)
    final_query: str
    must_reference: list[str] = Field(min_length=1)


class ContradictionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["contradiction"]
    query: str
    false_premise: str


class EmptyInputCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["empty_input"]
    prompt: str = ""
    allow_error: bool = True
    allow_refusal: bool = True


# Pydantic v2 discriminated union — looks at the `kind` field, dispatches to
# the matching sub-model, validates accordingly. Cleaner than a manual
# isinstance ladder and gives precise error messages for malformed cases.
EdgeCase = Annotated[
    LongInputCase | MultiTurnCase | ContradictionCase | EmptyInputCase,
    Field(discriminator="kind"),
]


class EdgeCasesSpec(BaseModel):
    # `extra="ignore"` mirrors RagQualitySpec — the runner passes the full TestSpec
    # dump down and we only care about our own fields (`cases`).
    model_config = ConfigDict(extra="ignore")
    cases: list[EdgeCase] = Field(default_factory=list)


# --- Heuristics -----------------------------------------------------------

# Phrases that indicate a clean rejection of empty input. Intentionally separate
# from rag_quality's _REFUSAL_HINTS — semantically different (rejecting *empty*
# input vs declining to answer an *out-of-corpus* question) and the lists may
# evolve independently.
_EMPTY_REJECTION_HINTS = (
    "cannot process",
    "empty",
    "no input",
    "no question",
    "please provide",
    "please ask",
    "i need",
    "missing",
    "i don't have",
    "i do not have",
    "unable to",
)

# Phrases that suggest the agent pushed back on a false premise. Used only as a
# fallback when --no-judge runs are configured; the judge's pushback() rubric is
# the authoritative grader. List intentionally narrow — false positives let bad
# behavior slip; false negatives just nudge the user to install a judge.
_PUSHBACK_HINTS = (
    "actually,",
    "that's not",
    "that is not",
    "incorrect",
    "i don't have",
    "i do not have",
    "no record",
    "no, ",
    "not aware",
    "could you clarify",
    "where did you",
    # A grounded refusal to engage with an unsupported premise IS pushback —
    # the agent is declining to validate the premise by answering. Narrower
    # phrasings ("could not find any documents") count; broad refusals like
    # "I cannot answer" do not (could also fire on legitimate out-of-corpus).
    "could not find",
    "no documents",
)


# --- Evaluator ------------------------------------------------------------


@dataclass
class CaseOutcome:
    case: LongInputCase | MultiTurnCase | ContradictionCase | EmptyInputCase
    verdict: Verdict
    detail: str
    response_text: str


@register
class EdgeCasesEvaluator(Evaluator):
    name = "edge_cases"

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
            ec_spec = EdgeCasesSpec.model_validate(spec.model_dump())
        except ValidationError as exc:
            return TestResult(
                name=spec.name,
                evaluator=self.name,
                verdict=Verdict.ERROR,
                detail=f"invalid edge_cases spec: {exc}",
                duration_ms=_elapsed_ms(t0),
            )

        if not ec_spec.cases:
            return TestResult(
                name=spec.name,
                evaluator=self.name,
                verdict=Verdict.ERROR,
                detail="no cases defined for edge_cases",
                duration_ms=_elapsed_ms(t0),
            )

        outcomes: list[CaseOutcome] = []
        for case in ec_spec.cases:
            if isinstance(case, LongInputCase):
                outcomes.append(_run_long_input(case, adapter))
            elif isinstance(case, MultiTurnCase):
                outcomes.append(_run_multi_turn(case, adapter))
            elif isinstance(case, ContradictionCase):
                outcomes.append(_run_contradiction(case, adapter, judge))
            else:
                # EmptyInputCase — the only branch left after the discriminated union.
                outcomes.append(_run_empty_input(case, adapter))

        worst = _worst_verdict([o.verdict for o in outcomes])
        passed = sum(1 for o in outcomes if o.verdict == Verdict.PASS)
        failed = sum(1 for o in outcomes if o.verdict == Verdict.FAIL)
        errored = sum(1 for o in outcomes if o.verdict == Verdict.ERROR)
        detail = f"{passed}/{len(outcomes)} cases passed"
        if worst is not Verdict.PASS:
            summary = "; ".join(
                f"{o.case.kind}: {o.detail}"
                for o in outcomes
                if o.verdict is not Verdict.PASS
            )
            detail += f" — {summary}"

        return TestResult(
            name=spec.name,
            evaluator=self.name,
            verdict=worst,
            detail=detail,
            metrics={
                "cases_total": float(len(outcomes)),
                "cases_passed": float(passed),
                "cases_failed": float(failed),
                "cases_errored": float(errored),
            },
            duration_ms=_elapsed_ms(t0),
            artifacts={
                "cases": [
                    {
                        "kind": o.case.kind,
                        "verdict": o.verdict.value,
                        "detail": o.detail,
                        "response_text": o.response_text,
                    }
                    for o in outcomes
                ],
            },
        )


# --- Per-kind handlers ----------------------------------------------------


def _run_long_input(case: LongInputCase, adapter: RagAdapter) -> CaseOutcome:
    """Construct a prompt of ~`length` chars and verify the adapter completes within timeout.

    Uses ThreadPoolExecutor for timeout enforcement (works on every platform;
    signal-based timeouts are Unix-only). The filler is `"a "` repeated rather
    than corpus-derived — the test exercises *input length handling*, not
    retrieval quality, and synthetic filler keeps the test self-contained.
    """
    filler = ("a " * ((case.length // 2) + 1))[: case.length - 60]
    prompt = f"{filler}\n\nWhat is the main topic of the text above?"

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(adapter.query, prompt)
            try:
                response = future.result(timeout=case.timeout_s)
            except concurrent.futures.TimeoutError:
                # future.cancel() returns False once running, so this leaves the
                # adapter call orphaned — acceptable for a test runner; the
                # process exits at end-of-run and threads die with it.
                return CaseOutcome(
                    case=case,
                    verdict=Verdict.FAIL,
                    detail=f"timed out after {case.timeout_s}s",
                    response_text="",
                )
    except Exception as exc:  # adapter contract is "anything goes"
        return CaseOutcome(
            case=case,
            verdict=Verdict.FAIL,
            detail=f"adapter raised on {case.length}-char prompt: {exc}",
            response_text="",
        )

    if not response.text.strip():
        return CaseOutcome(
            case=case,
            verdict=Verdict.FAIL,
            detail=f"empty response on {case.length}-char prompt",
            response_text=response.text,
        )

    return CaseOutcome(
        case=case,
        verdict=Verdict.PASS,
        detail=f"handled {case.length}-char prompt in <{case.timeout_s}s",
        response_text=response.text,
    )


def _run_multi_turn(case: MultiTurnCase, adapter: RagAdapter) -> CaseOutcome:
    """Build an N-turn conversation, send `final_query`, verify recall via must_reference.

    The assistant turns are placeholders rather than real model responses — we're
    testing whether the agent *uses* conversation history, not whether it can
    simulate one. Live multi-turn (where each assistant turn comes from the
    adapter) is a v0.3 follow-up.
    """
    conversation: list[Message] = []
    for user_turn in case.turns:
        conversation.append(Message(role="user", content=user_turn))
        conversation.append(
            Message(role="assistant", content="(prior turn — context only)")
        )

    try:
        response = adapter.query(case.final_query, conversation=conversation)
    except Exception as exc:  # adapter contract is "anything goes"
        return CaseOutcome(
            case=case,
            verdict=Verdict.ERROR,
            detail=f"adapter raised on multi_turn call: {exc}",
            response_text="",
        )

    lower = response.text.lower()
    missing = [s for s in case.must_reference if s.lower() not in lower]
    if missing:
        return CaseOutcome(
            case=case,
            verdict=Verdict.FAIL,
            detail=f"missing required reference(s) from earlier turns: {missing}",
            response_text=response.text,
        )

    return CaseOutcome(
        case=case,
        verdict=Verdict.PASS,
        detail=f"recalled all {len(case.must_reference)} required reference(s)",
        response_text=response.text,
    )


def _run_contradiction(
    case: ContradictionCase,
    adapter: RagAdapter,
    judge: LLMJudge | None,
) -> CaseOutcome:
    """Send a query with a false premise; PASS iff the agent pushes back.

    With a judge, the pushback() rubric is the authoritative grader. Without one
    (--no-judge / CI), fall back to a substring heuristic over _PUSHBACK_HINTS.
    The heuristic is intentionally narrow — false positives let bad behavior
    slip, false negatives just nudge the user to install a judge.
    """
    try:
        response = adapter.query(case.query)
    except Exception as exc:  # adapter contract is "anything goes"
        return CaseOutcome(
            case=case,
            verdict=Verdict.ERROR,
            detail=f"adapter raised on contradiction query: {exc}",
            response_text="",
        )

    text = response.text

    if judge is not None:
        try:
            verdict = judge.pushback(text, case.false_premise)
        except Exception as exc:  # judge contract: anything could go wrong
            return CaseOutcome(
                case=case,
                verdict=Verdict.ERROR,
                detail=f"judge.pushback() failed: {exc}",
                response_text=text,
            )
        if verdict.handled_correctly:
            return CaseOutcome(
                case=case,
                verdict=Verdict.PASS,
                detail=f"pushed back: {verdict.reasoning}",
                response_text=text,
            )
        return CaseOutcome(
            case=case,
            verdict=Verdict.FAIL,
            detail=f"accepted false premise: {verdict.reasoning}",
            response_text=text,
        )

    # Heuristic fallback for --no-judge runs.
    lower = text.lower()
    if any(hint in lower for hint in _PUSHBACK_HINTS):
        return CaseOutcome(
            case=case,
            verdict=Verdict.PASS,
            detail="heuristic pushback match (install judge for reliable grading)",
            response_text=text,
        )
    return CaseOutcome(
        case=case,
        verdict=Verdict.FAIL,
        detail="no pushback detected — heuristic fallback; install judge for reliable grading",
        response_text=text,
    )


def _run_empty_input(case: EmptyInputCase, adapter: RagAdapter) -> CaseOutcome:
    """Empty input must be rejected cleanly — either an exception or a refusal-style response."""
    try:
        response = adapter.query(case.prompt)
    except Exception as exc:  # adapter contract is "anything goes"
        if case.allow_error:
            return CaseOutcome(
                case=case,
                verdict=Verdict.PASS,
                detail=f"raised cleanly on empty input: {type(exc).__name__}",
                response_text="",
            )
        return CaseOutcome(
            case=case,
            verdict=Verdict.FAIL,
            detail=f"raised on empty input but allow_error=False: {exc}",
            response_text="",
        )

    text = response.text.strip()
    looks_like_rejection = any(hint in text.lower() for hint in _EMPTY_REJECTION_HINTS)

    if not text:
        # Empty response is treated as a degenerate refusal: not a hard answer.
        if case.allow_refusal:
            return CaseOutcome(
                case=case,
                verdict=Verdict.PASS,
                detail="returned empty response (no hard answer)",
                response_text="",
            )
        return CaseOutcome(
            case=case,
            verdict=Verdict.FAIL,
            detail="returned empty response but allow_refusal=False",
            response_text="",
        )

    if looks_like_rejection:
        if case.allow_refusal:
            return CaseOutcome(
                case=case,
                verdict=Verdict.PASS,
                detail="refused empty input cleanly",
                response_text=response.text,
            )
        return CaseOutcome(
            case=case,
            verdict=Verdict.FAIL,
            detail="refused empty input but allow_refusal=False",
            response_text=response.text,
        )

    return CaseOutcome(
        case=case,
        verdict=Verdict.FAIL,
        detail="answered empty input substantively instead of rejecting",
        response_text=response.text,
    )


# --- Helpers --------------------------------------------------------------


def _worst_verdict(verdicts: list[Verdict]) -> Verdict:
    if not verdicts:
        return Verdict.ERROR
    rank = {Verdict.PASS: 0, Verdict.WEAK: 1, Verdict.FAIL: 2, Verdict.ERROR: 3}
    return max(verdicts, key=lambda v: rank[v])


def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)
