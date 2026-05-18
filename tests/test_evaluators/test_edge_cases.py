"""Tests for the edge_cases evaluator.

Layered top-down:
  1. Pydantic spec validation (step 1 of the design spec).
  2. Per-kind handler behavior using fake adapters (steps 2-5).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from rag_eval.adapters.base import Message, RagAdapter, RagResponse
from rag_eval.config import TestSpec, Thresholds
from rag_eval.evaluators.base import Verdict
from rag_eval.evaluators.edge_cases import EdgeCasesEvaluator, EdgeCasesSpec


@dataclass
class FakeAdapter(RagAdapter):
    handler: Callable[[str], RagResponse] = lambda p: RagResponse(text="")

    def query(
        self,
        prompt: str,
        *,
        conversation: list[Message] | None = None,
    ) -> RagResponse:
        return self.handler(prompt)


def _spec(**kw: Any) -> TestSpec:
    return TestSpec(name="t", evaluator="edge_cases", **kw)


def _run(adapter: RagAdapter, **spec_kw: Any) -> Any:
    return EdgeCasesEvaluator().run(
        adapter, _spec(**spec_kw), judge=None, thresholds=Thresholds()
    )


def test_empty_cases_list_validates() -> None:
    """A spec with no cases is structurally valid (the evaluator will flag it later)."""
    spec = EdgeCasesSpec.model_validate({"cases": []})
    assert spec.cases == []


def test_long_input_case_validates_with_defaults() -> None:
    spec = EdgeCasesSpec.model_validate({"cases": [{"kind": "long_input"}]})
    assert len(spec.cases) == 1
    case = spec.cases[0]
    assert case.kind == "long_input"
    # type narrowing — defaults present
    assert case.length == 10000  # type: ignore[union-attr]
    assert case.timeout_s == 30.0  # type: ignore[union-attr]


def test_long_input_length_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        EdgeCasesSpec.model_validate({"cases": [{"kind": "long_input", "length": 0}]})


def test_long_input_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        EdgeCasesSpec.model_validate({"cases": [{"kind": "long_input", "timeout_s": 0}]})


def test_multi_turn_requires_turns() -> None:
    with pytest.raises(ValidationError) as exc:
        EdgeCasesSpec.model_validate(
            {
                "cases": [
                    {
                        "kind": "multi_turn",
                        "final_query": "what?",
                        "must_reference": ["x"],
                    }
                ]
            }
        )
    assert "turns" in str(exc.value)


def test_multi_turn_requires_non_empty_turns() -> None:
    with pytest.raises(ValidationError):
        EdgeCasesSpec.model_validate(
            {
                "cases": [
                    {
                        "kind": "multi_turn",
                        "turns": [],
                        "final_query": "what?",
                        "must_reference": ["x"],
                    }
                ]
            }
        )


def test_multi_turn_requires_final_query() -> None:
    with pytest.raises(ValidationError) as exc:
        EdgeCasesSpec.model_validate(
            {
                "cases": [
                    {
                        "kind": "multi_turn",
                        "turns": ["who?"],
                        "must_reference": ["x"],
                    }
                ]
            }
        )
    assert "final_query" in str(exc.value)


def test_multi_turn_requires_non_empty_must_reference() -> None:
    with pytest.raises(ValidationError):
        EdgeCasesSpec.model_validate(
            {
                "cases": [
                    {
                        "kind": "multi_turn",
                        "turns": ["who?"],
                        "final_query": "what?",
                        "must_reference": [],
                    }
                ]
            }
        )


def test_contradiction_requires_query_and_false_premise() -> None:
    with pytest.raises(ValidationError):
        EdgeCasesSpec.model_validate(
            {"cases": [{"kind": "contradiction", "query": "only a query"}]}
        )
    with pytest.raises(ValidationError):
        EdgeCasesSpec.model_validate(
            {"cases": [{"kind": "contradiction", "false_premise": "only a premise"}]}
        )


def test_empty_input_validates_with_defaults() -> None:
    spec = EdgeCasesSpec.model_validate({"cases": [{"kind": "empty_input"}]})
    case = spec.cases[0]
    assert case.kind == "empty_input"
    assert case.prompt == ""  # type: ignore[union-attr]
    assert case.allow_error is True  # type: ignore[union-attr]
    assert case.allow_refusal is True  # type: ignore[union-attr]


def test_invalid_kind_is_rejected() -> None:
    """The discriminator catches unknown kinds at parse time, not at run time."""
    with pytest.raises(ValidationError) as exc:
        EdgeCasesSpec.model_validate({"cases": [{"kind": "bogus_kind"}]})
    assert "kind" in str(exc.value)


def test_extra_fields_on_case_rejected() -> None:
    """extra='forbid' on each sub-model — catches typo'd field names early."""
    with pytest.raises(ValidationError):
        EdgeCasesSpec.model_validate(
            {"cases": [{"kind": "long_input", "lenght": 100}]}  # typo
        )


def test_discriminator_dispatches_heterogeneous_cases() -> None:
    """A real-world config mixes kinds; verify the union resolves each correctly."""
    spec = EdgeCasesSpec.model_validate(
        {
            "cases": [
                {"kind": "long_input", "length": 5000},
                {
                    "kind": "multi_turn",
                    "turns": ["a"],
                    "final_query": "b",
                    "must_reference": ["c"],
                },
                {"kind": "contradiction", "query": "q", "false_premise": "p"},
                {"kind": "empty_input"},
            ]
        }
    )
    assert [c.kind for c in spec.cases] == [
        "long_input",
        "multi_turn",
        "contradiction",
        "empty_input",
    ]


# --- Evaluator behavior: long_input ---------------------------------------


def test_long_input_pass_when_adapter_handles_it() -> None:
    seen_lengths: list[int] = []

    def handler(prompt: str) -> RagResponse:
        seen_lengths.append(len(prompt))
        return RagResponse(text="The text is about the letter 'a'.")

    result = _run(
        FakeAdapter(handler=handler),
        cases=[{"kind": "long_input", "length": 5000}],
    )
    assert result.verdict == Verdict.PASS
    assert result.metrics["cases_passed"] == 1
    # Prompt should be close to requested length (within ~60 chars for the
    # question suffix). Verifies the filler construction is right.
    assert 4900 <= seen_lengths[0] <= 5100


def test_long_input_fail_when_adapter_raises() -> None:
    def truncating(prompt: str) -> RagResponse:
        if len(prompt) > 1000:
            raise ValueError("input too long")
        return RagResponse(text="ok")

    result = _run(
        FakeAdapter(handler=truncating),
        cases=[{"kind": "long_input", "length": 5000}],
    )
    assert result.verdict == Verdict.FAIL
    assert "adapter raised" in result.detail


def test_long_input_fail_on_empty_response() -> None:
    result = _run(
        FakeAdapter(handler=lambda p: RagResponse(text="   ")),
        cases=[{"kind": "long_input", "length": 2000}],
    )
    assert result.verdict == Verdict.FAIL
    assert "empty response" in result.detail


def test_long_input_fail_on_timeout() -> None:
    """A slow adapter must trip the timeout path. Uses a tiny timeout (0.1s) +
    a sleep(0.5) so the test stays fast."""

    def slow(prompt: str) -> RagResponse:
        time.sleep(0.5)
        return RagResponse(text="too late")

    result = _run(
        FakeAdapter(handler=slow),
        cases=[{"kind": "long_input", "length": 500, "timeout_s": 0.1}],
    )
    assert result.verdict == Verdict.FAIL
    assert "timed out" in result.detail


# --- Evaluator behavior: empty_input --------------------------------------


def test_empty_input_pass_when_adapter_raises_and_allow_error_default() -> None:
    def raising(prompt: str) -> RagResponse:
        raise ValueError("empty prompt")

    result = _run(
        FakeAdapter(handler=raising),
        cases=[{"kind": "empty_input"}],
    )
    assert result.verdict == Verdict.PASS
    # On full PASS the test-level detail is just the count; per-case detail lives in artifacts.
    assert "raised cleanly" in result.artifacts["cases"][0]["detail"]


def test_empty_input_pass_when_adapter_refuses_and_allow_refusal_default() -> None:
    result = _run(
        FakeAdapter(handler=lambda p: RagResponse(text="Please provide a question.")),
        cases=[{"kind": "empty_input"}],
    )
    assert result.verdict == Verdict.PASS
    case_detail = result.artifacts["cases"][0]["detail"]
    assert "refused" in case_detail or "empty" in case_detail


def test_empty_input_fail_when_adapter_answers_substantively() -> None:
    result = _run(
        FakeAdapter(
            handler=lambda p: RagResponse(text="Sure — Acme's CEO is Jane Smith.")
        ),
        cases=[{"kind": "empty_input"}],
    )
    assert result.verdict == Verdict.FAIL
    assert "substantively" in result.detail


def test_empty_input_fail_when_raises_but_allow_error_false() -> None:
    def raising(prompt: str) -> RagResponse:
        raise RuntimeError("kaboom")

    result = _run(
        FakeAdapter(handler=raising),
        cases=[{"kind": "empty_input", "allow_error": False}],
    )
    assert result.verdict == Verdict.FAIL
    assert "allow_error=False" in result.detail


def test_empty_input_fail_when_refuses_but_allow_refusal_false() -> None:
    result = _run(
        FakeAdapter(handler=lambda p: RagResponse(text="Please provide a question.")),
        cases=[{"kind": "empty_input", "allow_refusal": False}],
    )
    assert result.verdict == Verdict.FAIL
    assert "allow_refusal=False" in result.detail


def test_empty_input_pass_when_response_is_blank_and_allow_refusal_default() -> None:
    """A blank string back is treated as a degenerate refusal (no hard answer)."""
    result = _run(
        FakeAdapter(handler=lambda p: RagResponse(text="")),
        cases=[{"kind": "empty_input"}],
    )
    assert result.verdict == Verdict.PASS


# --- Evaluator behavior: rollup, errors, dispatch -------------------------


def test_aggregates_worst_verdict_across_kinds() -> None:
    """A PASS + a FAIL should aggregate to FAIL."""

    def handler(prompt: str) -> RagResponse:
        # Long-input branch will see a long prompt; empty-input branch sees "".
        if prompt == "":
            return RagResponse(text="Sure — here's a substantive answer.")  # FAIL
        return RagResponse(text="Acknowledged.")  # PASS

    result = _run(
        FakeAdapter(handler=handler),
        cases=[
            {"kind": "long_input", "length": 500},
            {"kind": "empty_input"},
        ],
    )
    assert result.verdict == Verdict.FAIL
    assert result.metrics["cases_passed"] == 1
    assert result.metrics["cases_failed"] == 1


def test_error_when_no_cases() -> None:
    result = _run(FakeAdapter(handler=lambda p: RagResponse(text="")))
    assert result.verdict == Verdict.ERROR
    assert "no cases" in result.detail


def test_error_on_invalid_spec() -> None:
    """A malformed case dict surfaces as Verdict.ERROR with the validation message."""
    result = _run(
        FakeAdapter(handler=lambda p: RagResponse(text="ok")),
        cases=[{"kind": "long_input", "length": -5}],
    )
    assert result.verdict == Verdict.ERROR
    assert "invalid edge_cases spec" in result.detail


def test_contradiction_kind_surfaces_as_error_for_now() -> None:
    """contradiction lands in step 5. Until then it's ERROR, not silent-pass.
    This test is replaced by real behavior tests in step 5."""
    result = _run(
        FakeAdapter(handler=lambda p: RagResponse(text="x")),
        cases=[{"kind": "contradiction", "query": "q", "false_premise": "p"}],
    )
    assert result.verdict == Verdict.ERROR
    assert "not yet implemented" in result.detail


# --- Evaluator behavior: multi_turn ---------------------------------------


@dataclass
class _ConversationAwareAdapter(RagAdapter):
    """Records the conversation it received and returns based on what it saw."""

    response_builder: Callable[[list[Message], str], str] = (
        lambda conv, prompt: ""
    )
    captured: list[Message] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.captured = []

    def query(
        self,
        prompt: str,
        *,
        conversation: list[Message] | None = None,
    ) -> RagResponse:
        self.captured = list(conversation or [])
        return RagResponse(text=self.response_builder(self.captured, prompt))


def test_multi_turn_pass_when_adapter_recalls_earlier_turn() -> None:
    """RecallAdapter behavior: echo content from the first prior turn."""

    def recall(conv: list[Message], prompt: str) -> str:
        first_user = next((m.content for m in conv if m.role == "user"), "")
        return f"You earlier asked about Jane Smith and {first_user}"

    adapter = _ConversationAwareAdapter(response_builder=recall)
    result = _run(
        adapter,
        cases=[
            {
                "kind": "multi_turn",
                "turns": ["Who is the CEO of Acme?"],
                "final_query": "What was the name you mentioned?",
                "must_reference": ["Jane Smith"],
            }
        ],
    )
    assert result.verdict == Verdict.PASS
    assert "recalled all 1" in result.artifacts["cases"][0]["detail"]


def test_multi_turn_fail_when_adapter_forgets() -> None:
    """AmnesiaAdapter behavior: ignore conversation entirely."""

    adapter = _ConversationAwareAdapter(
        response_builder=lambda conv, prompt: "I have no context from prior turns."
    )
    result = _run(
        adapter,
        cases=[
            {
                "kind": "multi_turn",
                "turns": ["Who is the CEO of Acme?"],
                "final_query": "What was their name?",
                "must_reference": ["Jane Smith"],
            }
        ],
    )
    assert result.verdict == Verdict.FAIL
    assert "missing required reference" in result.detail


def test_multi_turn_builds_conversation_with_alternating_roles() -> None:
    """Verify the conversation shape: N user turns + N placeholder assistant turns."""

    adapter = _ConversationAwareAdapter(
        response_builder=lambda conv, prompt: "ok found Jane Smith"
    )
    _run(
        adapter,
        cases=[
            {
                "kind": "multi_turn",
                "turns": ["Q1", "Q2", "Q3"],
                "final_query": "F",
                "must_reference": ["Jane Smith"],
            }
        ],
    )
    # 3 user turns + 3 placeholder assistant turns = 6 messages
    assert len(adapter.captured) == 6
    assert [m.role for m in adapter.captured] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [m.content for m in adapter.captured if m.role == "user"] == ["Q1", "Q2", "Q3"]


def test_multi_turn_error_when_adapter_raises() -> None:
    def raising(conv: list[Message], prompt: str) -> str:
        raise ConnectionError("boom")

    adapter = _ConversationAwareAdapter(response_builder=raising)
    result = _run(
        adapter,
        cases=[
            {
                "kind": "multi_turn",
                "turns": ["a"],
                "final_query": "b",
                "must_reference": ["c"],
            }
        ],
    )
    assert result.verdict == Verdict.ERROR
    assert "adapter raised on multi_turn" in result.detail


def test_registered_under_edge_cases_name() -> None:
    """Confirms the @register decorator wired the evaluator into the registry."""
    from rag_eval.evaluators import EVALUATORS

    assert EVALUATORS["edge_cases"] is EdgeCasesEvaluator
