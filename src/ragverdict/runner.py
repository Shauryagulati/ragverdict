"""Orchestrate: load config, wire adapter + judge + evaluators, execute, finalize."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal

from ragverdict.adapters.loader import load_adapter
from ragverdict.config import Config, load_config
from ragverdict.evaluators import EVALUATORS
from ragverdict.evaluators.base import TestResult, Verdict
from ragverdict.judges.base import Judge, JudgeError
from ragverdict.judges.factory import build_judge
from ragverdict.judges.llm_judge import LLMJudge

# Sentinel for "auto-init the judge from config" (vs. None which means "no judge").
_AUTO: Literal["auto"] = "auto"

# Thresholds.faithfulness_pass / faithfulness_weak defaults — calibrated for Claude's
# claim-fraction score. _maybe_init_judge warns when a jev/cascade judge still has these.
_DEFAULT_FAITHFULNESS_PASS = 0.85
_DEFAULT_FAITHFULNESS_WEAK = 0.7


class RunnerError(Exception):
    """Raised for non-recoverable runner errors (bad config, unknown evaluator, etc.)."""


class Runner:
    def __init__(
        self,
        config: Config,
        out_dir: Path,
        *,
        judge: Judge | None | Literal["auto"] = "auto",
    ) -> None:
        self.config = config
        self.out_dir = out_dir
        self.judge: Judge | None = (
            self._maybe_init_judge() if judge == _AUTO else judge
        )

    @classmethod
    def from_config_path(
        cls,
        config_path: Path,
        out_dir: Path,
        *,
        judge: Judge | None | Literal["auto"] = "auto",
    ) -> Runner:
        return cls(load_config(config_path), out_dir, judge=judge)

    def execute(self) -> tuple[list[TestResult], int]:
        """Run every test in config. Returns (results, exit_code)."""
        from ragverdict.report import Reporter  # local import to break the cycle

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
        if isinstance(self.judge, LLMJudge) and (
            self.judge.cache_read_tokens or self.judge.cache_creation_tokens
        ):
            reporter.console.print(
                f"[dim]Judge prompt cache: "
                f"{self.judge.cache_read_tokens} read, "
                f"{self.judge.cache_creation_tokens} written.[/dim]"
            )
        return results, exit_code

    def _maybe_init_judge(self) -> Judge | None:
        """Best-effort judge instantiation. Returns None if credentials are missing."""
        try:
            judge = build_judge(self.config.judge)
        except JudgeError as exc:
            print(
                f"warning: {exc}\nproceeding without judge — WEAK verdicts and "
                "citation-audit support scoring will be skipped",
                file=sys.stderr,
            )
            return None
        self._maybe_warn_claude_calibrated_thresholds()
        return judge

    def _maybe_warn_claude_calibrated_thresholds(self) -> None:
        """Jev's faithfulness score is a probability, not Claude's claim-fraction score —
        the default thresholds are calibrated for Claude and are usually too strict for
        Jev. Warn once, to stderr, rather than silently misgrading every jev/cascade run."""
        thresholds = self.config.thresholds
        if self.config.judge.provider not in ("jev", "cascade"):
            return
        if (
            thresholds.faithfulness_pass != _DEFAULT_FAITHFULNESS_PASS
            or thresholds.faithfulness_weak != _DEFAULT_FAITHFULNESS_WEAK
        ):
            return
        print(
            "warning: judge.provider is "
            f"'{self.config.judge.provider}' but thresholds.faithfulness_pass/"
            f"faithfulness_weak are still the defaults ({_DEFAULT_FAITHFULNESS_PASS}/"
            f"{_DEFAULT_FAITHFULNESS_WEAK}). Those were calibrated for an LLM's "
            "claim-fraction score and are likely too strict for Jev's probability score. "
            "See the README's 'Judge backends' section for guidance on setting "
            "thresholds for Jev.",
            file=sys.stderr,
        )
