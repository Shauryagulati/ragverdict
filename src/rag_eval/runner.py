"""Orchestrate: load config, wire adapter + evaluators, execute, finalize."""

from __future__ import annotations

import time
from pathlib import Path

from rag_eval.adapters.loader import load_adapter
from rag_eval.config import Config, load_config
from rag_eval.evaluators import EVALUATORS
from rag_eval.evaluators.base import TestResult, Verdict


class RunnerError(Exception):
    """Raised for non-recoverable runner errors (bad config, unknown evaluator, etc.)."""


class Runner:
    def __init__(self, config: Config, out_dir: Path) -> None:
        self.config = config
        self.out_dir = out_dir

    @classmethod
    def from_config_path(cls, config_path: Path, out_dir: Path) -> Runner:
        return cls(load_config(config_path), out_dir)

    def execute(self) -> tuple[list[TestResult], int]:
        """Run every test in config. Returns (results, exit_code)."""
        from rag_eval.report import Reporter  # imported lazily to avoid circular imports

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
                # Judge is wired in Day 3; pass None until then.
                result = evaluator.run(adapter, spec, judge=None)
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
