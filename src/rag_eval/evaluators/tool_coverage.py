"""tool_coverage — fires every tool the adapter exposes; checks each one works."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag_eval.evaluators import register
from rag_eval.evaluators.base import Evaluator, TestResult, Verdict

if TYPE_CHECKING:
    from rag_eval.adapters.base import RagAdapter
    from rag_eval.config import TestSpec, Thresholds
    from rag_eval.judges.llm_judge import LLMJudge


class ToolCoverageSpec(BaseModel):
    """Fields parsed off a TestSpec for this evaluator."""

    model_config = ConfigDict(extra="ignore")

    require_all_tools: bool = True
    trigger_prompts: dict[str, str] = Field(default_factory=dict)


@register
class ToolCoverageEvaluator(Evaluator):
    name = "tool_coverage"

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
            tc_spec = ToolCoverageSpec.model_validate(spec.model_dump())
        except ValidationError as exc:
            return TestResult(
                name=spec.name,
                evaluator=self.name,
                verdict=Verdict.ERROR,
                detail=f"invalid tool_coverage spec: {exc}",
                duration_ms=_elapsed_ms(t0),
            )

        tools = adapter.available_tools()
        if not tools:
            return TestResult(
                name=spec.name,
                evaluator=self.name,
                verdict=Verdict.ERROR,
                detail="adapter.available_tools() returned no tools — nothing to test",
                duration_ms=_elapsed_ms(t0),
            )

        per_tool: list[dict[str, object]] = []
        any_failed = False
        for tool in tools:
            prompt = (
                tc_spec.trigger_prompts.get(tool.name)
                or tool.trigger_prompt
                or f"Please use the {tool.name} tool. ({tool.description})"
            )
            tool_t0 = time.perf_counter()
            try:
                response = adapter.query(prompt)
            except Exception as exc:  # noqa: BLE001 - any adapter error is a tool failure
                per_tool.append(
                    {
                        "tool": tool.name,
                        "fired": False,
                        "error": f"adapter raised: {exc}",
                        "latency_ms": _elapsed_ms(tool_t0),
                    }
                )
                any_failed = True
                continue

            calls = [tc for tc in response.tool_calls if tc.name == tool.name]
            fired_ok = bool(calls) and all(c.error is None for c in calls)
            errored = bool(calls) and any(c.error is not None for c in calls)
            latency = max((c.latency_ms for c in calls), default=_elapsed_ms(tool_t0))
            per_tool.append(
                {
                    "tool": tool.name,
                    "fired": fired_ok,
                    "error": next((c.error for c in calls if c.error), None)
                    if errored
                    else (None if fired_ok else "tool was not called"),
                    "latency_ms": latency,
                }
            )
            if not fired_ok:
                any_failed = True

        passed_tools = sum(1 for r in per_tool if r["fired"])
        total_tools = len(per_tool)

        if tc_spec.require_all_tools:
            verdict = Verdict.PASS if not any_failed else Verdict.FAIL
        else:
            verdict = Verdict.PASS if passed_tools > 0 else Verdict.FAIL

        detail = f"{passed_tools}/{total_tools} tools fired cleanly"
        if any_failed:
            failed_names = [str(r["tool"]) for r in per_tool if not r["fired"]]
            detail += f" — failed: {', '.join(failed_names)}"

        return TestResult(
            name=spec.name,
            evaluator=self.name,
            verdict=verdict,
            detail=detail,
            metrics={
                "tools_total": float(total_tools),
                "tools_passed": float(passed_tools),
            },
            duration_ms=_elapsed_ms(t0),
            artifacts={"per_tool": per_tool},
        )


def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)
