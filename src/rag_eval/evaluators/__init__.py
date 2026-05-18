"""Evaluators run behavioral tests against a RagAdapter."""

from rag_eval.evaluators.base import Evaluator, TestResult, Verdict

EVALUATORS: dict[str, type[Evaluator]] = {}


def register(cls: type[Evaluator]) -> type[Evaluator]:
    """Register an Evaluator subclass by its `name` class attribute."""
    EVALUATORS[cls.name] = cls
    return cls


def _autoload() -> None:
    # Import the bundled evaluator modules so they self-register on the registry.
    # Kept inside a function (not module-level) to avoid circular imports.
    from rag_eval.evaluators import (  # noqa: F401
        citation_audit,
        edge_cases,
        rag_quality,
        tool_coverage,
    )


_autoload()

__all__ = ["EVALUATORS", "Evaluator", "TestResult", "Verdict", "register"]
