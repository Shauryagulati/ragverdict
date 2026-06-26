"""Head-to-head comparison driver: runs ragverdict against deliberately-broken RAG
adapters to make the differentiation vs metric-centric tools concrete.

Each scenario uses an adapter that exhibits a real production failure mode. We
run ragverdict and show what verdict it returns. The writeup in README.md walks
through what a metric-centric tool (RAGAs) would have reported on the same
response, and why.

Run:
    python examples/comparison/run_comparison.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

# These are the same regression-smoke adapters tests/test_regression_smoke.py uses —
# importing them here keeps a single source of truth for "what known-broken behavior
# looks like."
sys.path.insert(0, ".")

from ragverdict.config import TestSpec, Thresholds
from ragverdict.evaluators.citation_audit import CitationAuditEvaluator
from ragverdict.evaluators.edge_cases import EdgeCasesEvaluator
from ragverdict.evaluators.tool_coverage import ToolCoverageEvaluator
from tests.test_regression_smoke import (
    CompliantAdapter,
    DanglingCitationAdapter,
    SilentToolAdapter,
)


@dataclass
class Scenario:
    name: str
    failure_mode: str
    what_ragas_would_likely_say: str
    ragverdict_evaluator: str
    ragverdict_caught_it: bool
    ragverdict_detail: str


def run() -> list[Scenario]:
    scenarios: list[Scenario] = []
    thresholds = Thresholds()

    # ── Scenario 1: Dangling citation ─────────────────────────────────────
    result = CitationAuditEvaluator().run(
        DanglingCitationAdapter(),
        TestSpec(name="t", evaluator="citation_audit", sample_queries=["any query"]),
        judge=None,
        thresholds=thresholds,
    )
    scenarios.append(
        Scenario(
            name="Dangling citation",
            failure_mode=(
                "Agent emits `[src:GHOST]` — looks well-formed but the source_id "
                "doesn't exist in the corpus (only `REAL` is present)."
            ),
            what_ragas_would_likely_say=(
                "Faithfulness ~0.9. The response IS grounded in the retrieved "
                "context (`REAL` was retrieved). RAGAs's faithfulness scores "
                "groundedness, not citation-vs-corpus integrity. The dangling "
                "[src:GHOST] is invisible to metric scoring."
            ),
            ragverdict_evaluator="citation_audit",
            ragverdict_caught_it=(result.verdict.value == "FAIL"),
            ragverdict_detail=result.detail,
        )
    )

    # ── Scenario 2: Silent tool ───────────────────────────────────────────
    result = ToolCoverageEvaluator().run(
        SilentToolAdapter(),
        TestSpec(name="t", evaluator="tool_coverage"),
        judge=None,
        thresholds=thresholds,
    )
    scenarios.append(
        Scenario(
            name="Silent tool no-op",
            failure_mode=(
                "Agent advertises a `search` tool in its manifest but never "
                "actually calls it — answers from prior knowledge instead."
            ),
            what_ragas_would_likely_say=(
                "RAGAs has no concept of tool firing. It scores the response only. "
                "If the response happens to be plausible, faithfulness/relevance "
                "look fine. The silent tool is completely invisible."
            ),
            ragverdict_evaluator="tool_coverage",
            ragverdict_caught_it=(result.verdict.value == "FAIL"),
            ragverdict_detail=result.detail,
        )
    )

    # ── Scenario 3: Compliant agent (false-premise acceptance) ────────────
    result = EdgeCasesEvaluator().run(
        CompliantAdapter(),
        TestSpec(
            name="t",
            evaluator="edge_cases",
            cases=[
                {
                    "kind": "contradiction",
                    "query": (
                        "What was the rationale for Acme's acquisition of XYZ "
                        "Corp in 2030?"
                    ),
                    "false_premise": "Acme acquired XYZ Corp in 2030",
                }
            ],
        ),
        judge=None,  # heuristic fallback — no API key required
        thresholds=thresholds,
    )
    scenarios.append(
        Scenario(
            name="False-premise acceptance",
            failure_mode=(
                "User question embeds a false premise ('Acme's 2030 acquisition'). "
                "The agent agrees with the premise and invents a rationale."
            ),
            what_ragas_would_likely_say=(
                "Answer Relevance ~0.85. The response addresses the question that "
                "was asked. RAGAs doesn't grade premise-handling — an agent that "
                "confidently agrees with a false premise gets a high relevance "
                "score because it answered the question on-topic."
            ),
            ragverdict_evaluator="edge_cases.contradiction",
            ragverdict_caught_it=(result.verdict.value == "FAIL"),
            ragverdict_detail=result.detail,
        )
    )

    return scenarios


def print_report(scenarios: list[Scenario]) -> None:
    print()
    print("=" * 78)
    print("HEAD-TO-HEAD: ragverdict vs metric-centric tools (illustrated with RAGAs)")
    print("=" * 78)
    for i, s in enumerate(scenarios, start=1):
        print()
        print(f"── Scenario {i}: {s.name}")
        print(f"   Failure mode:   {s.failure_mode}")
        print(f"   RAGAs likely:   {s.what_ragas_would_likely_say}")
        print(f"   ragverdict:       {s.ragverdict_evaluator}")
        verdict_label = "✅ CAUGHT" if s.ragverdict_caught_it else "❌ MISSED"
        print(f"   Verdict:        {verdict_label} — {s.ragverdict_detail}")
    print()
    print("=" * 78)
    caught = sum(1 for s in scenarios if s.ragverdict_caught_it)
    print(f"Summary: ragverdict caught {caught}/{len(scenarios)} failures that metric")
    print("scoring would have given high scores to.")
    print("=" * 78)
    print()


if __name__ == "__main__":
    print_report(run())
