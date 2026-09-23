"""Anthropic-backed LLMJudge with structured outputs and prompt caching.

The judge exposes four methods:

- `faithfulness(response, retrieved_context)` — is the response grounded in context?
- `relevance(response, query)` — does the response answer the question?
- `refusal(response, query)` — is the response an explicit refusal / "I don't know"?
- `pushback(response, false_premise)` — did the response correct a false premise?

Every call is built by `_request()` as plain `messages.create` kwargs with an
`output_config` JSON schema derived from the Pydantic result model, and every
response is validated by `parse_judge_message()`. The Batch API accepts the same
kwargs as a request's `params`, so batch and live runs use identical prompts,
schemas, and validation — only the transport differs.

`thinking="disabled"` exists because models that think by default (Sonnet 5)
can spend the output budget on reasoning and truncate the JSON answer.

Caching caveat: rubrics are 400-600 tokens, below the minimum cacheable prefix
on current models, so `cache_control` is wired but rarely hits.
"""

from __future__ import annotations

import os
from typing import Any, Literal, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from ragverdict.judges.base import (
    JudgeError,
    JudgeScore,
    JudgeTransportError,
    PushbackVerdict,
    RefusalVerdict,
)

__all__ = [
    "JudgeError",
    "JudgeScore",
    "JudgeTransportError",
    "LLMJudge",
    "PushbackVerdict",
    "RefusalVerdict",
    "Thinking",
    "faithfulness_prompt",
    "output_schema",
    "parse_judge_message",
]

T = TypeVar("T", bound=BaseModel)

Thinking = Literal["model_default", "disabled"]


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


def faithfulness_prompt(response_text: str, retrieved_context: str) -> tuple[str, str]:
    """(system prompt, user message) for a faithfulness judgment — shared by every LLM judge."""
    user = (
        f"<retrieved_context>\n{retrieved_context.strip() or '(no context retrieved)'}"
        f"\n</retrieved_context>\n\n"
        f"<response>\n{response_text}\n</response>"
    )
    return _FAITHFULNESS_SYSTEM, user


def output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema the API enforces for `model`, minus fields the LLM must not fill.

    `confidence` belongs to probabilistic judges (Jev); an LLM judge never sets it,
    so it is removed from the schema the model is constrained to.
    """
    schema: dict[str, Any] = anthropic.transform_schema(model.model_json_schema())
    schema.get("properties", {}).pop("confidence", None)
    if "required" in schema:
        schema["required"] = [name for name in schema["required"] if name != "confidence"]
    return schema


def parse_judge_message(message: Any, schema: type[T]) -> T:
    """Validate one Messages API response against `schema`. Shared by live and batch paths."""
    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason == "max_tokens":
        raise JudgeError(
            "judge output truncated (stop_reason=max_tokens); raise the judge's max_tokens "
            "or set thinking: disabled"
        )
    if stop_reason == "refusal":
        raise JudgeError("judge declined to answer (stop_reason=refusal)")
    text = next(
        (block.text for block in message.content if getattr(block, "type", None) == "text"),
        None,
    )
    if text is None:
        raise JudgeError("judge returned no text block")
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        raise JudgeError(f"judge returned invalid JSON for {schema.__name__}: {exc}") from exc


class LLMJudge:
    """Anthropic-backed judge. Tests inject a stub client via the `client` kwarg."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-6",
        client: anthropic.Anthropic | None = None,
        max_tokens: int = 4096,
        thinking: Thinking = "model_default",
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
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
        self.input_tokens = 0
        self.output_tokens = 0
        self.last_response_id = ""  # the Anthropic message id from the most recent call

    def faithfulness_request(self, response_text: str, retrieved_context: str) -> dict[str, Any]:
        """The exact `messages.create` kwargs for a faithfulness call (also valid batch params)."""
        system, user = faithfulness_prompt(response_text, retrieved_context)
        return self._request(system, user, JudgeScore)

    def faithfulness(self, response_text: str, retrieved_context: str) -> JudgeScore:
        return self._call(self.faithfulness_request(response_text, retrieved_context), JudgeScore)

    def relevance(self, response_text: str, query: str) -> JudgeScore:
        user = f"<query>\n{query}\n</query>\n\n<response>\n{response_text}\n</response>"
        return self._call(self._request(_RELEVANCE_SYSTEM, user, JudgeScore), JudgeScore)

    def refusal(self, response_text: str, query: str) -> RefusalVerdict:
        user = f"<query>\n{query}\n</query>\n\n<response>\n{response_text}\n</response>"
        return self._call(self._request(_REFUSAL_SYSTEM, user, RefusalVerdict), RefusalVerdict)

    def pushback(self, response_text: str, false_premise: str) -> PushbackVerdict:
        user = (
            f"<false_premise>\n{false_premise}\n</false_premise>\n\n"
            f"<response>\n{response_text}\n</response>"
        )
        return self._call(self._request(_PUSHBACK_SYSTEM, user, PushbackVerdict), PushbackVerdict)

    # ---------- internals ----------

    def _request(
        self, system_prompt: str, user_content: str, schema: type[BaseModel]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [
                {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": user_content}],
            "output_config": {
                "format": {"type": "json_schema", "schema": output_schema(schema)}
            },
        }
        if self.thinking == "disabled":
            params["thinking"] = {"type": "disabled"}
        return params

    def _call(self, params: dict[str, Any], schema: type[T]) -> T:
        self.last_response_id = ""
        try:
            message: Any = self._client.messages.create(**params)
        except anthropic.APIError as exc:
            raise JudgeTransportError(f"judge API call failed: {exc}") from exc
        self.last_response_id = str(getattr(message, "id", "") or "")
        self._record_usage(getattr(message, "usage", None))
        return parse_judge_message(message, schema)

    def _record_usage(self, usage: Any) -> None:
        if usage is None:
            return
        self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
