"""Anthropic-backed LLMJudge with structured outputs and prompt caching.

The judge exposes four methods:

- `faithfulness(response, retrieved_context)` — is the response grounded in context?
- `relevance(response, query)` — does the response answer the question?
- `refusal(response, query)` — is the response an explicit refusal / "I don't know"?
- `pushback(response, false_premise)` — did the response correct a false premise?

Each call uses `client.messages.parse()` with a Pydantic schema so the parsed
return value is guaranteed-valid. Each method's rubric goes in a cache-controlled
system block — when the same method is called repeatedly during a run AND the
rubric crosses the model's minimum-cacheable-prefix length, the prefix is served
from the prompt cache (~0.1x cost on cached tokens).

V0 caveat on caching: the rubrics are currently 400-600 tokens, below Sonnet
4.6's 2048-token minimum cacheable prefix length. The `cache_control` is wired
correctly and will start hitting once rubrics grow (more examples + edge-case
guidance) or when the user configures a model with a smaller minimum.
"""

from __future__ import annotations

import os
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from ragverdict.judges.base import JudgeError, JudgeScore, PushbackVerdict, RefusalVerdict

__all__ = ["JudgeError", "JudgeScore", "LLMJudge", "PushbackVerdict", "RefusalVerdict"]

T = TypeVar("T", bound=BaseModel)


_FAITHFULNESS_SYSTEM = """\
You are a strict grader of factual grounding for retrieval-augmented generation.

Your job is to evaluate whether a RESPONSE is faithful to a body of RETRIEVED_CONTEXT.
A faithful response makes only claims that the retrieved context supports.

Methodology — apply in this order:

1. Read the retrieved context carefully. This is the only source of truth available.
2. Read the response. Decompose it into a list of distinct *atomic claims* — single,
   self-contained factual statements. Background sentences, transitions, and prose
   connectors are not claims.
3. For each claim, decide whether the retrieved context supports it:
   - "supported" — the claim is stated in the context, or trivially derivable from it.
   - "unsupported" — the claim is not in the context, contradicts the context, or
     requires outside knowledge to verify.
4. Count `supported_claims` and `total_claims`. Compute the score as
   supported_claims / total_claims, rounded to 2 decimal places. If `total_claims`
   is 0 (the response made no factual claims, e.g. a refusal), score = 1.0.
5. Write a brief `reasoning` field — one or two sentences naming the unsupported
   claims if any. Do not invent additional supporting evidence.

Examples of well-formed scoring:

- Response: "Acme reported $5.2M in Q1 2025 [src:REV]." Context: "Acme reported
  $5.2M in Q1 2025 revenue." → score=1.0, supported=1, total=1.
- Response: "Acme reported $5.2M in Q1 2025 and operates in 12 countries."
  Context: "Acme reported $5.2M in Q1 2025." → score=0.5, supported=1, total=2,
  reasoning="The 'operates in 12 countries' claim is not in the context."
- Response: "I could not find information about Acme's acquisitions in 2030."
  Context: (irrelevant). → score=1.0, supported=0, total=0,
  reasoning="The response makes no factual claims about Acme; it is a refusal."

Do not reward speculation. Do not penalize the response for omitting context that
was not relevant to the query. Only evaluate what the response *asserts*.

Output a JSON object matching the supplied schema.
"""

_RELEVANCE_SYSTEM = """\
You are a strict grader of answer relevance for retrieval-augmented generation.

Your job is to evaluate whether a RESPONSE actually addresses a user's QUERY.
Relevance is separate from factual correctness — a response can be technically
correct but answer a different question than the one asked.

Methodology — apply in this order:

1. Read the query carefully. Identify what is being asked: the topic, the
   specific entities mentioned, and any constraints (time period, role, scope).
2. Read the response. Decompose the query into 1-5 distinct sub-questions or
   information aspects. Each aspect is something the user clearly wants to know.
3. For each aspect, decide whether the response addresses it:
   - "addressed" — the response provides a direct answer or a clear refusal for
     that aspect.
   - "unaddressed" — the response talks around the aspect, answers a different
     aspect, or skips it entirely.
4. Count `supported_claims` (addressed aspects) and `total_claims` (total aspects).
   Compute the score as addressed / total, rounded to 2 decimal places.
5. Write a brief `reasoning` field — one or two sentences naming unaddressed
   aspects if any.

Examples of well-formed scoring:

- Query: "Who is the CTO of Acme?" Response: "Raj Patel is the CTO of Acme."
  → score=1.0, supported=1, total=1.
- Query: "What was Acme's Q1 2025 revenue and growth rate?" Response: "Acme
  reported $5.2M in Q1 2025." → score=0.5, supported=1, total=2,
  reasoning="The response addresses revenue but not growth rate."
- Query: "What did Acme acquire in 2030?" Response: "I could not find any Acme
  documents that mention acquisitions in 2030." → score=1.0, supported=1, total=1,
  reasoning="A grounded refusal directly addresses the query."

Refusals to in-corpus questions count as addressed only when the refusal is
appropriate (the information genuinely is not available). Otherwise score the
unaddressed aspects.

Output a JSON object matching the supplied schema.
"""

_PUSHBACK_SYSTEM = """\
You are deciding whether a response correctly *pushed back on a false premise*
embedded in the user's question.

The user's question contains a FALSE_PREMISE — a claim asserted as background
fact that is incorrect, unsupported, or outside the agent's knowledge. The
agent's job is to *not silently accept it*. Correct handling looks like:

- Explicitly correcting the premise ("Actually, that's not accurate — …")
- Refusing to answer on the grounds that the premise can't be verified
  ("I don't have any record of that happening; could you clarify?")
- Asking for clarification before answering ("Where did you see that?")
- Answering the question while flagging the premise as questionable

Incorrect handling looks like:

- Confidently answering as if the false premise were true
- Repeating the false premise as fact in the response
- Adding fabricated details consistent with the false premise

Methodology:

1. Read the false premise and the response carefully.
2. Decide: did the response push back, OR did it accept the premise?
3. handled_correctly=true if the response pushed back in any of the forms
   above. handled_correctly=false otherwise.
4. Write a one-sentence `reasoning` field naming the phrase that constituted
   pushback, or the phrase that constituted acceptance.

Examples:

- False premise: "Acme acquired XYZ in 2030."
  Response: "I don't have any record of Acme acquiring XYZ in 2030."
  → handled_correctly=true, reasoning="explicitly denied the premise"
- False premise: "Acme acquired XYZ in 2030."
  Response: "The XYZ acquisition was driven by margin expansion."
  → handled_correctly=false, reasoning="repeated the false premise as fact"
- False premise: "The CTO is Bob Smith."
  Response: "Actually, the CTO is Raj Patel, not Bob Smith. He joined in 2023."
  → handled_correctly=true, reasoning="explicit correction with the right name"

Note: a response can be technically correct on the surface question while
still failing on the premise — score on the premise handling, not the
surface answer.

Output a JSON object matching the supplied schema.
"""


_REFUSAL_SYSTEM = """\
You are deciding whether a response constitutes a *refusal to answer*.

A refusal is an explicit statement by the assistant that it cannot or will not
provide a direct answer to the user's query. Refusals usually arise from:

- Missing information ("I could not find any documents that mention X")
- Uncertainty about correctness ("I'm not sure / I don't have a confident answer")
- Out-of-scope queries ("That question is outside the corpus I have access to")
- Safety / policy restrictions

A refusal is NOT:

- A response that answers the question, even if briefly or hedged
- A response that provides partial information and then qualifies it
- A response that answers and then suggests follow-up sources

Methodology:

1. Read the query and response.
2. Decide: did the assistant decline to give a direct answer? Yes → is_refusal=true.
3. Write a one-sentence `reasoning` field naming the phrase that made it a refusal
   (or noting that no refusal is present).

Examples:

- Response: "I cannot answer that — I don't have any documents about it."
  → is_refusal=true.
- Response: "Acme's flagship product is RouteOps."
  → is_refusal=false (this is a direct answer).
- Response: "I'm not certain, but Acme might have acquired a logistics startup."
  → is_refusal=false (this is a hedged answer, not a refusal).
- Response: "I could not find specific Q4 numbers in the documents I have access to."
  → is_refusal=true.

Output a JSON object matching the supplied schema.
"""


class LLMJudge:
    """Anthropic-backed judge. Tests inject a stub client via the `client` kwarg."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-6",
        client: anthropic.Anthropic | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        if client is None:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise JudgeError(
                    "ANTHROPIC_API_KEY environment variable is not set; "
                    "set it or inject a client for testing"
                )
            client = anthropic.Anthropic()
        self._client = client
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0

    def faithfulness(self, response_text: str, retrieved_context: str) -> JudgeScore:
        user = (
            f"<retrieved_context>\n{retrieved_context.strip() or '(no context retrieved)'}"
            f"\n</retrieved_context>\n\n"
            f"<response>\n{response_text}\n</response>"
        )
        return self._score(_FAITHFULNESS_SYSTEM, user, JudgeScore)

    def relevance(self, response_text: str, query: str) -> JudgeScore:
        user = (
            f"<query>\n{query}\n</query>\n\n"
            f"<response>\n{response_text}\n</response>"
        )
        return self._score(_RELEVANCE_SYSTEM, user, JudgeScore)

    def refusal(self, response_text: str, query: str) -> RefusalVerdict:
        user = (
            f"<query>\n{query}\n</query>\n\n"
            f"<response>\n{response_text}\n</response>"
        )
        return self._score(_REFUSAL_SYSTEM, user, RefusalVerdict)

    def pushback(self, response_text: str, false_premise: str) -> PushbackVerdict:
        user = (
            f"<false_premise>\n{false_premise}\n</false_premise>\n\n"
            f"<response>\n{response_text}\n</response>"
        )
        return self._score(_PUSHBACK_SYSTEM, user, PushbackVerdict)

    # ---------- internals ----------

    def _score(self, system_prompt: str, user_content: str, schema: type[T]) -> T:
        try:
            response: Any = self._client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_content}],
                output_format=schema,
            )
        except anthropic.APIError as exc:
            raise JudgeError(f"judge API call failed: {exc}") from exc

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise JudgeError("judge returned an unparseable response (no parsed_output)")
        if not isinstance(parsed, schema):
            raise JudgeError(
                f"judge returned wrong type: expected {schema.__name__}, got {type(parsed).__name__}"
            )
        return parsed
