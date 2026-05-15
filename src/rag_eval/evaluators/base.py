"""Evaluator ABC, verdict enum, and the TestResult dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from rag_eval.adapters.base import RagAdapter
    from rag_eval.config import TestSpec, Thresholds
    from rag_eval.judges.llm_judge import LLMJudge


class Verdict(str, Enum):
    PASS = "PASS"
    WEAK = "WEAK"
    FAIL = "FAIL"
    ERROR = "ERROR"


@dataclass
class TestResult:
    name: str
    evaluator: str
    verdict: Verdict
    detail: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    duration_ms: int = 0
    artifacts: dict[str, Any] = field(default_factory=dict)


class Evaluator(ABC):
    """Subclass and implement `run`. Register with `@register` in evaluators/__init__.py."""

    name: ClassVar[str]

    @abstractmethod
    def run(
        self,
        adapter: RagAdapter,
        spec: TestSpec,
        *,
        judge: LLMJudge | None,
        thresholds: Thresholds,
    ) -> TestResult: ...
