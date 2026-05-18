"""edge_cases — input-boundary failure modes V0's golden-path evaluators don't catch.

Four kinds, each modeled as its own Pydantic case type and dispatched via a
discriminated union on `kind`:

- `long_input`     — ≥10K-char prompt; PASS = no crash + non-empty response within timeout
- `multi_turn`     — N-turn conversation; PASS = response contains required substrings
                      (verifies the agent uses conversation history)
- `contradiction`  — query with a false premise; PASS = agent pushes back / corrects
                      (judge-graded; heuristic fallback when --no-judge)
- `empty_input`    — empty prompt; PASS = controlled error or refusal-style response

Verdicts are PASS/FAIL/ERROR only — no WEAK. Edge cases are binary safety
properties, not quality scores in a tunable range.

Spec: docs/superpowers/specs/2026-05-18-edge-case-evaluator-design.md
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class LongInputCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["long_input"]
    length: int = Field(default=10000, gt=0)
    timeout_s: float = Field(default=30.0, gt=0)


class MultiTurnCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["multi_turn"]
    turns: list[str] = Field(min_length=1)
    final_query: str
    must_reference: list[str] = Field(min_length=1)


class ContradictionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["contradiction"]
    query: str
    false_premise: str


class EmptyInputCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["empty_input"]
    prompt: str = ""
    allow_error: bool = True
    allow_refusal: bool = True


# Pydantic v2 discriminated union — looks at the `kind` field, dispatches to
# the matching sub-model, validates accordingly. Cleaner than a manual
# isinstance ladder and gives precise error messages for malformed cases.
EdgeCase = Annotated[
    LongInputCase | MultiTurnCase | ContradictionCase | EmptyInputCase,
    Field(discriminator="kind"),
]


class EdgeCasesSpec(BaseModel):
    # `extra="ignore"` mirrors RagQualitySpec — the runner passes the full TestSpec
    # dump down and we only care about our own fields (`cases`).
    model_config = ConfigDict(extra="ignore")
    cases: list[EdgeCase] = Field(default_factory=list)
