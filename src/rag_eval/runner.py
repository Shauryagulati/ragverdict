"""Orchestrate: load config, wire adapter + judge + evaluators, execute, finalize."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal

from rag_eval.adapters.loader import load_adapter
from rag_eval.config import Config, load_config
from rag_eval.evaluators import EVALUATORS
from rag_eval.evaluators.base import TestResult, Verdict
from rag_eval.judges.llm_judge import JudgeError, LLMJudge

# Sentinel for "auto-init the judge from config" (vs. None which means "no judge").
_AUTO: Literal["auto"] = "auto"


class RunnerError(Exception):
    """Raised for non-recoverable runner errors (bad config, unknown evaluator, etc.)."""


class Runner:
    def __init__(
        self,
        config: Config,
        out_dir: Path,
        *,
        judge: LLMJudge | None | Literal["auto"] = "auto",
    ) -> None:
        self.config = config
        self.out_dir = out_dir
        self.judge: LLMJudge | None = (
            self._maybe_init_judge() if judge == _AUTO else judge
        )

    @classmethod
    def from_config_path(
        cls,
        config_path: Path,
        out_dir: Path,
        *,
        judge: LLMJudge | None | Literal["auto"] = "auto",
    ) -> Runner:
        return cls(load_config(config_path), out_dir, judge=judge)

    def execute(self) -> tuple[list[TestResult], int]:
        """Run every test in config. Returns (results, exit_code)."""
        from rag_eval.report import Reporter  # local import to break the cycle

        adapter = load_adapter(self.config.adapter)

        unknown = [t.evaluator for t in self.config.tests if t.evaluator not in EVALUATORS]
        if unknown:
            raise RunnerError(f"unknown evaluator(s): {', '.join(sorted(set(unknown)))}")

        reporter = Reporter(out_dir=self.out_dir, config=self.config)
        reporter.start(test_count=len(self.config.tests))

        results: list[TestResult] = []
        for spec in self.config.tests:
            cls = EVALUATORS[spec.evaluator]
            evaluator = cls()
            t0 = time.perf_counter()
            try:
                result = evaluator.run(
                    adapter,
                    spec,
                    judge=self.judge,
                    thresholds=self.config.thresholds,
                )
            except Exception as exc:
                result = TestResult(
                    name=spec.name,
                    evaluator=spec.evaluator,
                    verdict=Verdict.ERROR,
                    detail=f"evaluator crashed: {exc}",
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                )
            results.append(result)
            reporter.on_result(result)

        _, exit_code = reporter.finalize(results)
        return results, exit_code

    def _maybe_init_judge(self) -> LLMJudge | None:
        """Best-effort judge instantiation. Returns None if no API key is set."""
        if self.config.judge.provider != "anthropic":
            return None
        try:
            return LLMJudge(model=self.config.judge.model)
        except JudgeError as exc:
            print(
                f"warning: {exc}\nproceeding without judge — WEAK verdicts and "
                "citation-audit support scoring will be skipped",
                file=sys.stderr,
            )
            return None
