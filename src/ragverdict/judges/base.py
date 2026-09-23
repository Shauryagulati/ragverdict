"""Judge interface and the result models every judge backend returns.

Evaluators depend only on `Judge`, so any backend that implements these four
methods (Anthropic LLM, Jev, a cascade of both, a test stub) plugs in unchanged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class JudgeError(Exception):
    """Raised when a judge call cannot produce a valid result."""


class JudgeTransportError(JudgeError):
    """The call never produced a judge answer — network/HTTP/API error; safe to retry."""


class JudgeScore(BaseModel):
    """Score for one quality dimension (faithfulness or relevance).

    `confidence` is only set by probabilistic judges (Jev): max(p, 1 - p).
    LLM judges leave it None.
    """

    model_config = ConfigDict(extra="ignore")

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    supported_claims: int = 0
    total_claims: int = 0
    confidence: float | None = None


class RefusalVerdict(BaseModel):
    """Did the response explicitly decline to answer?"""

    model_config = ConfigDict(extra="ignore")

    is_refusal: bool
    reasoning: str
    confidence: float | None = None


class PushbackVerdict(BaseModel):
    """Did the response correctly push back on a false premise?

    Pushback ≠ refusal. The agent can correct the premise while still
    answering ("Actually, Acme acquired XYZ in 2018, and the rationale
    was…"). What we're grading is whether the false claim was
    challenged, not whether the agent declined to engage.
    """

    model_config = ConfigDict(extra="ignore")

    handled_correctly: bool
    reasoning: str
    confidence: float | None = None


@runtime_checkable
class Judge(Protocol):
    """Anything that can grade a RAG response on these four dimensions."""

    def faithfulness(self, response_text: str, retrieved_context: str) -> JudgeScore: ...

    def relevance(self, response_text: str, query: str) -> JudgeScore: ...

    def refusal(self, response_text: str, query: str) -> RefusalVerdict: ...

    def pushback(self, response_text: str, false_premise: str) -> PushbackVerdict: ...
