"""Jev (TypeSafe AI) judge backend: typed yes/no questions answered with probabilities.

Jev returns a probability instead of text, so there is no explanation to show;
`reasoning` records the model and the raw probability. TypeSafe's API and
OpenRouter expose the same `POST /v1/systemone` shape, so one client serves both.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ragverdict.judges.base import (
    JudgeError,
    JudgeScore,
    JudgeTransportError,
    PushbackVerdict,
    RefusalVerdict,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api"
TYPESAFE_BASE_URL = "https://api.typesafe.ai"
DEFAULT_JEV_MODEL = "typesafe/jev-1.13"

FAITHFULNESS_QUESTION = "Is every factual claim in the response supported by the source?"
RELEVANCE_QUESTION = "Does the response directly answer the query?"
REFUSAL_QUESTION = "Does the response decline to answer or say the information is unavailable?"
PUSHBACK_QUESTION = "Does the response challenge or correct the false premise?"

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 529})


@dataclass(frozen=True)
class JevAnswer:
    """One Jev answer plus what it cost."""

    p_yes: float
    input_tokens: int
    cost_usd: float
    served_model: str
    latency_s: float
    response_id: str = ""


def reorient_to_supported(answer: JevAnswer) -> JevAnswer:
    """Re-orient a JevAnswer from P(yes) to P(supported) = 1 - P(yes) — used whenever the
    configured question means "is it unsupported?" rather than "is it supported?". A module-
    level function, not a method, so every caller that needs this re-orientation (JevJudge
    itself, and the bench runner's pilot-state arm, which calls `ask()` directly) shares one
    copy instead of each hand-rolling the JevAnswer field list."""
    return JevAnswer(
        p_yes=1.0 - answer.p_yes,
        input_tokens=answer.input_tokens,
        cost_usd=answer.cost_usd,
        served_model=answer.served_model,
        latency_s=answer.latency_s,
        response_id=answer.response_id,
    )


class JevJudge:
    """Judge backed by Jev. Thread-safe: counters are updated under a lock."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_JEV_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        max_attempts: int = 3,
        backoff_s: float = 1.0,
        faithfulness_question: str = FAITHFULNESS_QUESTION,
        question_means_unsupported: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        if api_key is None:
            env_var = "OPENROUTER_API_KEY" if "openrouter.ai" in self.base_url else "TYPESAFE_API_KEY"
            api_key = os.getenv(env_var)
            if not api_key:
                raise JudgeError(f"{env_var} environment variable is not set; set it or pass api_key")
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=60.0)
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.faithfulness_question = faithfulness_question
        # True when the question asks "is it hallucinated?" — P(yes) is then P(unsupported).
        self.question_means_unsupported = question_means_unsupported
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.cost_usd = 0.0
        self.served_models: set[str] = set()

    # ---------- Judge protocol ----------

    def faithfulness(self, response_text: str, retrieved_context: str) -> JudgeScore:
        p = self.faithfulness_answer(response_text, retrieved_context).p_yes
        return JudgeScore(score=p, confidence=max(p, 1.0 - p), reasoning=self._reason("supported", p))

    def relevance(self, response_text: str, query: str) -> JudgeScore:
        p = self.ask({"query": query, "response": response_text}, RELEVANCE_QUESTION).p_yes
        return JudgeScore(score=p, confidence=max(p, 1.0 - p), reasoning=self._reason("relevant", p))

    def refusal(self, response_text: str, query: str) -> RefusalVerdict:
        p = self.ask({"query": query, "response": response_text}, REFUSAL_QUESTION).p_yes
        return RefusalVerdict(
            is_refusal=p >= 0.5, confidence=max(p, 1.0 - p), reasoning=self._reason("refusal", p)
        )

    def pushback(self, response_text: str, false_premise: str) -> PushbackVerdict:
        state = {"false_premise": false_premise, "response": response_text}
        p = self.ask(state, PUSHBACK_QUESTION).p_yes
        return PushbackVerdict(
            handled_correctly=p >= 0.5,
            confidence=max(p, 1.0 - p),
            reasoning=self._reason("pushback", p),
        )

    # ---------- lower level ----------

    def faithfulness_answer(self, response_text: str, retrieved_context: str) -> JevAnswer:
        """Faithfulness answer with `p_yes` oriented as P(response is fully supported)."""
        answer = self.ask(
            {"source": retrieved_context, "response": response_text}, self.faithfulness_question
        )
        if not self.question_means_unsupported:
            return answer
        return reorient_to_supported(answer)

    def ask(self, state: dict[str, str], question: str) -> JevAnswer:
        """Ask one yes/no question about `state`; returns P(yes) and usage."""
        body = {
            "model": self.model,
            "state": state,
            "questions": {"q": {"type": "noul", "instructions": question}},
        }
        started = time.perf_counter()
        data = self._post(body)
        latency = time.perf_counter() - started
        try:
            p = float(data["answers"]["q"]["noul"])
        except (KeyError, TypeError, ValueError) as exc:
            # Jev is a typed classifier — it cannot itself emit a malformed answer, so a
            # missing/malformed `answers.q.noul` means the gateway mangled the response.
            raise JudgeTransportError(
                f"jev returned an unexpected response shape: {str(data)[:200]}"
            ) from exc
        if not 0.0 <= p <= 1.0:
            raise JudgeTransportError(f"jev returned probability out of range: {p}")
        usage = data.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        answer = JevAnswer(
            p_yes=p,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            cost_usd=float(usage.get("cost", 0.0) or 0.0),
            served_model=str(data.get("model", "")),
            latency_s=latency,
            response_id=str(data.get("id", "") or ""),
        )
        with self._lock:
            self.calls += 1
            self.input_tokens += answer.input_tokens
            self.cost_usd += answer.cost_usd
            if answer.served_model:
                self.served_models.add(answer.served_model)
        return answer

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/v1/systemone"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._client.post(url, json=body, headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"network error: {exc}"
            else:
                if response.status_code == 200:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise JudgeTransportError("jev returned a non-JSON body") from exc
                    if not isinstance(data, dict):
                        raise JudgeTransportError(
                            f"jev returned an unexpected response shape: {str(data)[:200]}"
                        )
                    return data
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                if response.status_code not in _RETRYABLE_STATUS:
                    raise JudgeTransportError(f"jev API call failed: {last_error}")
            if attempt < self.max_attempts:
                time.sleep(self.backoff_s * 2 ** (attempt - 1))
        raise JudgeTransportError(
            f"jev API call failed after {self.max_attempts} attempts: {last_error}"
        )

    def _reason(self, what: str, p: float) -> str:
        return f"jev {self.model}: P({what})={p:.2f}"
