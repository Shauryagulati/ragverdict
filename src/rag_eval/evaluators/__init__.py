"""Evaluators run behavioral tests against a RagAdapter."""

from rag_eval.evaluators.base import Evaluator, TestResult, Verdict

EVALUATORS: dict[str, type[Evaluator]] = {}


def register(cls: type[Evaluator]) -> type[Evaluator]:
    """Register an Evaluator subclass by its `name` class attribute."""
    EVALUATORS[cls.name] = cls
    return cls


__all__ = ["EVALUATORS", "Evaluator", "TestResult", "Verdict", "register"]
