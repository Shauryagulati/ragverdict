# rag-eval vs metric-centric eval tools

A concrete, runnable comparison: three real production failure modes, what
metric-centric RAG eval tools see, and what rag-eval catches.

## The setup

Three deliberately-broken adapters (the same ones in
[`tests/test_regression_smoke.py`](../../tests/test_regression_smoke.py)),
each exhibiting a real production failure pattern. We run rag-eval against each and
reason about what a metric-centric tool like RAGAs would report on the same response.

## Run it yourself

```bash
python examples/comparison/run_comparison.py
```

No API key required — uses the heuristic-fallback path that rag-eval ships for `--no-judge` runs.

## The three scenarios

### 1. Dangling citation

**The bug:** Agent emits `[src:GHOST]` in its response. Looks well-formed. The
`source_id` doesn't actually exist in the corpus — only `REAL` is present.

| | Says about this response |
|---|---|
| **RAGAs faithfulness** | ~0.9 — the response IS grounded in the retrieved context (the `REAL` document was retrieved and the answer reflects it). Faithfulness measures groundedness in retrieved context, not citation-vs-corpus integrity. |
| **rag-eval `citation_audit`** | **FAIL** — "1/1 citations are dangling (source_id not in corpus)". |

**Why metric scoring misses it:** RAGAs checks "is the answer grounded in the chunks
that came back from retrieval." The chunks that came back are real. The CITATION the
agent wrote points at something else entirely. Different question.

### 2. Silent tool no-op

**The bug:** Agent's manifest advertises a `search` tool. The agent never actually
calls it. Answers from prior knowledge instead.

| | Says about this response |
|---|---|
| **RAGAs / ARES / TruLens** | Has no concept of "did the tool fire." Scores the response text only. If the response is plausible, faithfulness and relevance look fine. |
| **rag-eval `tool_coverage`** | **FAIL** — "0/1 tools fired cleanly — failed: search". |

**Why metric scoring misses it:** All five metric-centric tools (RAGAs, ARES,
TruLens, Phoenix, DeepEval) operate on `(query, response, retrieved_context)` tuples.
The tool-call layer is invisible to them. A tool that silently no-ops is the kind of
production bug that ships for weeks without anyone noticing because every metric
looks green.

### 3. False-premise acceptance

**The bug:** User question embeds a false premise — "What was the rationale for
Acme's acquisition of XYZ Corp in 2030?" — when Acme never acquired XYZ. The agent
agrees with the premise and confidently invents a rationale.

| | Says about this response |
|---|---|
| **RAGAs answer relevance** | ~0.85 — the response addresses the question that was asked. Relevance measures "does the answer cover what was asked"; an agent that confidently agrees with the false premise gets a high score because it stayed on-topic. |
| **rag-eval `edge_cases.contradiction`** | **FAIL** — "no pushback detected". |

**Why metric scoring misses it:** None of the major rubrics distinguish "did the
agent push back on the premise" from "did the agent answer the question." rag-eval
has a dedicated `pushback` judge rubric for this exact distinction (with a heuristic
fallback for offline runs).

## The pattern

Each failure mode shares the same shape: **the response itself looks fine.** It's
grounded, it's on-topic, it answers the question. Metric scoring grades the response.
The bug isn't in the response — it's in the agent's *behavior*: a tool that didn't
fire, a citation that doesn't resolve, a premise that wasn't challenged.

rag-eval tests behavior. That's the gap.

## When to use each

This isn't either/or. A mature RAG team uses both:

| Use case | Right tool |
|---|---|
| Track faithfulness / relevance / context-recall metrics over time on a benchmark | **RAGAs**, **ARES** (with confidence intervals), **TruLens** |
| Hosted dashboards, traces, span-level debugging | **TruLens**, **Arize Phoenix** |
| pytest-style assertions on metric thresholds | **DeepEval** |
| Verify the agent **behaves** correctly — tools fire, citations resolve, premises challenged, edges handled — with CI exit codes | **rag-eval** |

The metric-centric tools tell you whether your responses are *good*. rag-eval tells
you whether your agent is *correct*. Both matter.

## One-liner you can quote

> Metric-centric RAG eval scores how a response looks. rag-eval tests whether the
> agent behaves correctly — tools firing, citations resolving, false premises
> challenged, edges handled. A response can score 0.9 faithfulness while citing a
> document that doesn't exist; rag-eval catches that. The two compose: RAGAs for
> quality tracking, rag-eval for behavioral regression in CI.
