# rag-eval

[![CI](https://github.com/Shauryagulati/rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Shauryagulati/rag-eval/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/Shauryagulati/rag-eval/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**pytest for RAG agents.** Behavioral audits of any RAG system — tool coverage, retrieval
quality, citation verification, hallucination guardrails — with PASS / FAIL / WEAK verdicts,
not floating-point metric averages.

```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Test               ┃ Evaluator      ┃ Verdict ┃ Latency ┃ Detail                          ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ tool_coverage_all  │ tool_coverage  │ PASS    │     8ms │ 2/2 tools fired cleanly         │
│ direct_retrieval   │ rag_quality    │ PASS    │ 12662ms │ 3/3 cases passed                │
│ hallucination_g…   │ rag_quality    │ PASS    │  3985ms │ 2/2 cases passed                │
│ citation_audit     │ citation_audit │ PASS    │  5535ms │ mean support_score=1.00         │
└────────────────────┴────────────────┴─────────┴─────────┴─────────────────────────────────┘
```

## Why rag-eval

Existing RAG evaluation tools score metrics. RAGAs, DeepEval, TruLens, and Arize Phoenix
all answer "how faithful was the response *on average*" via LLM-as-judge — they tell you
the mean of a fleet of scores. They do not answer **does the agent actually work
end-to-end**.

| Tool       | Does                                                          | Doesn't                                                                       |
|------------|---------------------------------------------------------------|-------------------------------------------------------------------------------|
| **RAGAs**  | LLM-as-judge metric scores (faithfulness, context P/R)        | No tool-call testing, no citation-vs-corpus verification, no assertions       |
| **DeepEval** | pytest-style assertions on the RAGAs metric family          | Same metric-centric model                                                     |
| **TruLens**  | RAG Triad + OpenTelemetry tracing                           | Observability-centric                                                         |
| **Phoenix**  | Tracing platform that wraps the above                       | Heavy infra, not a CLI                                                        |

**The gap rag-eval fills:** behavioral audits of RAG *agents* — assertions about whether
the system *behaves correctly*, with PASS/FAIL/WEAK verdicts that map cleanly to CI.

### What it checks

- **`tool_coverage`** — Fires every tool the agent exposes and confirms it returns without
  error. Reports per-tool pass/fail + latency. **None of the four competitors do this.**
- **`rag_quality`** — Hard assertions (`must_mention`, `must_not_cite`, `must_refuse`,
  `expects_citations`) plus LLM-as-judge faithfulness + relevance scoring for `WEAK` /
  `FAIL` verdicts when hard checks pass.
- **`citation_audit`** — Verifies every `[src:ID]` citation resolves to a real document in
  the agent's corpus, then asks the judge whether the cited claim is actually supported by
  the source. Dangling citations are a hard `FAIL`.
- **`edge_cases`** *(v0.2)* — Input-boundary failure modes: `long_input` (10K-char
  prompts, timeout-bounded), `multi_turn` (conversation-context recall), `contradiction`
  (false premises must be pushed back on, judge-graded with `--no-judge` heuristic
  fallback), and `empty_input` (clean rejection). **None of the four competitors do this
  either.**

## Quickstart

### 1. Install

```bash
pip install -e ".[dev]"  # from a clone; PyPI release pending
export ANTHROPIC_API_KEY=sk-ant-…  # required for the LLM judge
```

### 2. Run the bundled demo

```bash
rag-eval run examples/demo_rag/config.yaml
```

This runs five tests against a tiny reference RAG agent (`DemoAdapter`) over a fictional
"Acme Corp" corpus, exercising all four evaluators.

To run without burning API tokens:

```bash
rag-eval run examples/demo_rag/config.yaml --no-judge
```

(Hard assertions still run; WEAK verdicts and citation support scoring are skipped.)

### 3. Write a config for your own RAG system

`config.yaml`:

```yaml
adapter:
  type: python
  module: my_app.rag_adapter
  class: MyRagAdapter

judge:
  provider: anthropic
  model: claude-sonnet-4-6

tests:
  - name: tool_coverage_all
    evaluator: tool_coverage

  - name: golden_path
    evaluator: rag_quality
    cases:
      - query: "What was Q1 2025 revenue?"
        must_mention: ["$5.2M"]
        expects_citations: true

  - name: out_of_corpus
    evaluator: rag_quality
    cases:
      - query: "Predict 2030 revenue."
        must_refuse: true
        must_not_cite: true

  - name: citations
    evaluator: citation_audit
    sample_queries:
      - "Summarize Q1 2025 risks."
      - "Who is the CTO?"
```

### 4. Write your adapter

Subclass `RagAdapter` and implement `query()`:

```python
from rag_eval import RagAdapter, RagResponse, Citation, ToolCall, ToolSpec, SourceDoc

class MyRagAdapter(RagAdapter):
    def query(self, prompt, *, conversation=None) -> RagResponse:
        # Call your real RAG pipeline:
        text, retrieved, citations, tool_calls = my_pipeline.run(prompt)
        return RagResponse(
            text=text,
            citations=[Citation(id=c.id, source_id=c.source, span=c.span) for c in citations],
            tool_calls=[ToolCall(name=t.name, args=t.args, latency_ms=t.ms) for t in tool_calls],
            retrieved_context=retrieved,
        )

    def available_tools(self) -> list[ToolSpec]:
        return [ToolSpec(name="search_kb", description="Knowledge-base lookup")]

    def corpus(self):
        for doc in my_pipeline.iter_docs():
            yield SourceDoc(source_id=doc.id, content=doc.text, title=doc.title)
```

The runner inserts the current working directory into `sys.path` before resolving your
`module:` import, so a project-local `my_app/` package just works.

See [`examples/demo_rag/adapter.py`](./examples/demo_rag/adapter.py) for a complete
reference adapter and [`examples/README.md`](./examples/README.md) for a walkthrough.

## Verdicts

- **PASS** — All hard assertions hold; judge scores (if configured) are at or above the
  pass threshold (defaults: faithfulness 0.85, relevance 0.85, citation support 0.95).
- **WEAK** — Hard assertions hold but a judge score falls in `[weak, pass)` (defaults:
  0.7–0.85 for faithfulness/relevance, 0.8–0.95 for citation support).
- **FAIL** — A hard assertion failed, or a judge score fell below the weak threshold.
- **ERROR** — The evaluator crashed or the judge returned unparseable output.

Tune thresholds via the `thresholds:` section of `config.yaml`. Exit codes:

| Code | Meaning                                                |
|------|--------------------------------------------------------|
| 0    | All tests PASS or WEAK                                 |
| 1    | At least one FAIL or ERROR                             |
| 2    | Config error / adapter load failure / unknown evaluator |
| 3    | All tests ERROR (typically: judge unavailable)         |

## Reports

After each run, two files land in `./report/` (override with `--out-dir`):

- **`report.json`** — Machine-readable: full per-test verdicts, metrics, judge artifacts,
  per-citation audit detail. Stable shape — see [`docs/json-report-schema.md`](./docs/json-report-schema.md).
- **`report.md`** — Human-readable summary table.

## Roadmap

v0.2 ships the edge-case battery. Next up:

- Write-tool safety evaluator (preview-only verification, version chain checks)
- `auth_negative` kind for the `edge_cases` evaluator (requires adapter ABC extension)
- Native `OpenAI` / `LangChain` adapters
- Concurrent test execution
- Hosted dashboard with regression tracking across runs

## Methodology

The methodology behind rag-eval comes from running a 34-turn behavioral audit against a
17-tool RAG copilot serving private-equity firms for commercial due diligence.
The audit caught what metric scoring did not: tools that silently no-op'd, citations that
looked clean but pointed at truncated document IDs, refusals that fired when they
shouldn't, and answers that confabulated when they couldn't ground.

## License

MIT — see [LICENSE](./LICENSE).
