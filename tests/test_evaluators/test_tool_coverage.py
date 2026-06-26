from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ragverdict.adapters.base import (
    Message,
    RagAdapter,
    RagResponse,
    ToolCall,
    ToolSpec,
)
from ragverdict.config import TestSpec, Thresholds
from ragverdict.evaluators.base import Verdict
from ragverdict.evaluators.tool_coverage import ToolCoverageEvaluator


@dataclass
class FakeAdapter(RagAdapter):
    tools: list[ToolSpec] = field(default_factory=list)
    handler: Callable[[str], RagResponse] | None = None
    raise_on: set[str] = field(default_factory=set)

    def available_tools(self) -> list[ToolSpec]:
        return self.tools

    def query(
        self,
        prompt: str,
        *,
        conversation: list[Message] | None = None,
    ) -> RagResponse:
        if any(token in prompt for token in self.raise_on):
            raise RuntimeError("adapter exploded")
        assert self.handler is not None
        return self.handler(prompt)


def _spec(**kw: Any) -> TestSpec:
    return TestSpec(name="t", evaluator="tool_coverage", **kw)


def test_pass_when_all_tools_fire_cleanly() -> None:
    tools = [
        ToolSpec(name="search", description="search", trigger_prompt="Use search"),
        ToolSpec(name="profile", description="profile", trigger_prompt="Use profile"),
    ]

    def handler(prompt: str) -> RagResponse:
        if "search" in prompt:
            return RagResponse(text="ok", tool_calls=[ToolCall(name="search", latency_ms=10)])
        return RagResponse(text="ok", tool_calls=[ToolCall(name="profile", latency_ms=5)])

    adapter = FakeAdapter(tools=tools, handler=handler)
    result = ToolCoverageEvaluator().run(adapter, _spec(), judge=None, thresholds=Thresholds())

    assert result.verdict == Verdict.PASS
    assert result.metrics["tools_passed"] == 2
    assert result.metrics["tools_total"] == 2


def test_fail_when_required_tool_never_fires() -> None:
    tools = [
        ToolSpec(name="search", description="search"),
        ToolSpec(name="profile", description="profile"),
    ]

    def handler(prompt: str) -> RagResponse:
        return RagResponse(text="ok", tool_calls=[])  # never fires anything

    adapter = FakeAdapter(tools=tools, handler=handler)
    result = ToolCoverageEvaluator().run(adapter, _spec(), judge=None, thresholds=Thresholds())

    assert result.verdict == Verdict.FAIL
    assert "search" in result.detail and "profile" in result.detail


def test_fail_when_tool_errors() -> None:
    tools = [ToolSpec(name="search", description="search")]

    def handler(prompt: str) -> RagResponse:
        return RagResponse(
            text="ok",
            tool_calls=[ToolCall(name="search", error="500 internal", latency_ms=3)],
        )

    adapter = FakeAdapter(tools=tools, handler=handler)
    result = ToolCoverageEvaluator().run(adapter, _spec(), judge=None, thresholds=Thresholds())

    assert result.verdict == Verdict.FAIL
    per_tool = result.artifacts["per_tool"]
    assert per_tool[0]["error"] == "500 internal"


def test_error_when_adapter_has_no_tools() -> None:
    adapter = FakeAdapter(tools=[], handler=lambda p: RagResponse(text=""))
    result = ToolCoverageEvaluator().run(adapter, _spec(), judge=None, thresholds=Thresholds())

    assert result.verdict == Verdict.ERROR
    assert "no tools" in result.detail


def test_trigger_prompts_override_toolspec() -> None:
    seen_prompts: list[str] = []

    def handler(prompt: str) -> RagResponse:
        seen_prompts.append(prompt)
        return RagResponse(text="ok", tool_calls=[ToolCall(name="search", latency_ms=1)])

    tools = [ToolSpec(name="search", description="d", trigger_prompt="default")]
    adapter = FakeAdapter(tools=tools, handler=handler)
    spec = _spec(trigger_prompts={"search": "OVERRIDE"})

    result = ToolCoverageEvaluator().run(adapter, spec, judge=None, thresholds=Thresholds())
    assert result.verdict == Verdict.PASS
    assert seen_prompts == ["OVERRIDE"]


def test_adapter_exception_treated_as_tool_failure() -> None:
    tools = [ToolSpec(name="search", description="d", trigger_prompt="use search")]
    adapter = FakeAdapter(tools=tools, handler=lambda p: RagResponse(text=""), raise_on={"search"})

    result = ToolCoverageEvaluator().run(adapter, _spec(), judge=None, thresholds=Thresholds())
    assert result.verdict == Verdict.FAIL
    assert "adapter raised" in str(result.artifacts["per_tool"][0]["error"])
