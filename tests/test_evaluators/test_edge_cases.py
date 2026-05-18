"""Step 1 — Pydantic spec validation for the edge_cases evaluator.

Verifies the discriminated-union dispatch and the per-kind field constraints
defined in the v0.2.0 design spec. Evaluator behavior tests land in later steps.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_eval.evaluators.edge_cases import EdgeCasesSpec


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
