# ragverdict

[![CI](https://github.com/Shauryagulati/ragverdict/actions/workflows/ci.yml/badge.svg)](https://github.com/Shauryagulati/ragverdict/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/Shauryagulati/ragverdict/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**pytest for RAG agents.** Behavioral audits of any RAG system — tool coverage, retrieval
quality, citation verification, hallucination guardrails — with PASS / FAIL / WEAK verdicts,
not floating-point metric averages.

![ragverdict running the bundled demo — all green](docs/demo.gif)

```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Test               ┃ Evaluator      ┃ Verdict ┃ Latency ┃ Detail                          ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ tool_coverage_all  │ tool_coverage  │ PASS    │     8ms │ 2/2 tools fired cleanly         │
│ direct_retrieval   │ rag_quality    │ PASS    │ 12662ms │ 3/3 cases passed                │
│ hallucination_g…   │ rag_quality    │ PASS    │  3985ms │ 2/2 cases passed                │
│ citation_audit     │ citation_audit │ PASS    │  5535ms │ mean support_score=1.00         │
│ edge_cases_battery │ edge_cases     │ PASS    │  2380ms │ 4/4 cases passed                │
└────────────────────┴────────────────┴─────────┴─────────┴─────────────────────────────────┘
```

## Why ragverdict

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

**The gap ragverdict fills:** behavioral audits of RAG *agents* — assertions about whether
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
# Use it on your own RAG system:
pip install ragverdict

# Or clone to run the bundled demo below / hack on it:
git clone https://github.com/Shauryagulati/ragverdict && cd ragverdict
pip install -e ".[dev]"

export ANTHROPIC_API_KEY=sk-ant-…  # required for the LLM judge
```

### 2. Run the bundled demo

```bash
ragverdict run examples/demo_rag/config.yaml
```

This runs five tests against a tiny reference RAG agent (`DemoAdapter`) over a fictional
"Acme Corp" corpus, exercising all four evaluators.

To run without burning API tokens:

```bash
ragverdict run examples/demo_rag/config.yaml --no-judge
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
from ragverdict import RagAdapter, RagResponse, Citation, ToolCall, ToolSpec, SourceDoc

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

## Judge backends

`judge.provider` in `config.yaml` selects the scorer. All three backends speak the same
`Judge` protocol (`faithfulness`, `relevance`, `refusal`, `pushback`), so nothing else in
your config changes when you switch.

### `anthropic` (default)

```yaml
judge:
  provider: anthropic
  model: claude-sonnet-4-6
```

Requires `ANTHROPIC_API_KEY`. Unchanged from earlier releases.

### `jev`

```yaml
judge:
  provider: jev
  jev_model: typesafe/jev-1.13
  jev_base_url: https://openrouter.ai/api

thresholds:
  faithfulness_pass: 0.5
  faithfulness_weak: 0.2
```

Requires `OPENROUTER_API_KEY` (or `TYPESAFE_API_KEY` if you point `jev_base_url` at
`https://api.typesafe.ai` instead). Jev is a lightweight probabilistic classifier, not an
LLM — read this before switching:

- **(a)** With `jev`, the faithfulness score is Jev's probability that the response is
  fully supported. It is **not** a claim fraction the way Claude's score is.
- **(b)** The default `thresholds` (`faithfulness_pass: 0.85`, `faithfulness_weak: 0.7`)
  were set for Claude's claim-fraction score and are too strict for Jev. Set them from a
  labeled sample of your own data; a conservative starting point for the default
  faithfulness question is `faithfulness_pass: 0.5`, `faithfulness_weak: 0.2`. `ragverdict`
  warns on stderr if you run `jev`/`cascade` with the untouched Claude-calibrated defaults.
- **(c)** The [RAGTruth benchmark](#benchmark) validated only the faithfulness question.
  Jev's relevance, refusal, and pushback questions are unvalidated.
- **(d)** Jev returns no explanation — `reasoning` on its results just records the model
  and raw probability, not a rationale.
- **(e)** If you use `cascade`, `cascade_band` must straddle your decision thresholds,
  e.g. band `[0.1, 0.6]` with the thresholds above.
- **(f)** The benchmark's own tuned configuration used a different question wording
  ("B") and threshold `0.165`, chosen by fitting on the benchmark's train split — not
  something to copy blind onto your data. See
  [docs/jev-ragtruth-benchmark.md](docs/jev-ragtruth-benchmark.md) for the wording and
  method.

### `cascade`

```yaml
judge:
  provider: cascade
  model: claude-sonnet-4-6       # fallback, asked only when Jev is unsure
  jev_model: typesafe/jev-1.13
  jev_base_url: https://openrouter.ai/api
  cascade_band: [0.1, 0.6]        # straddles the thresholds below

thresholds:
  faithfulness_pass: 0.5
  faithfulness_weak: 0.2
```

Asks Jev first; escalates to the Anthropic model only when Jev's probability falls
strictly inside `cascade_band`. Requires both `OPENROUTER_API_KEY` (or
`TYPESAFE_API_KEY`) and `ANTHROPIC_API_KEY`. See the Jev guidance above — it applies to
`cascade` too whenever Jev's score is used directly.

See [`examples/demo_rag/config.jev.yaml`](./examples/demo_rag/config.jev.yaml) for a
complete worked example.

## Benchmark

**Benchmark: Jev vs LLM judges on RAGTruth.** On 2,666 human-labeled RAG answers (RAGTruth
test set), with each judge at its default setting, Jev's F1 was 0.758 [95% CI 0.734–0.782] vs
Claude Sonnet 5 0.721 [0.697–0.748], GLM Flash 0.753 and DeepSeek Flash 0.708. The registered
Jev-vs-Claude test was inconclusive (+0.037, 90% CI +0.023 to +0.050, margin ±0.03), Jev tied
GLM and beat DeepSeek. At each LLM's own operating point the LLM was equal or slightly better:
Jev's edge comes from its default cutoff and a finer-grained score. Jev cost $0.046 per 1,000
judgments vs $6.54 for Claude at standard list price, but it caught far fewer hallucinations
written by GPT-4/GPT-3.5 (small n), and its default cutoff needs calibrating on your own data.
A Jev→Claude cascade added no F1. Full method, registration, limitations and reproduction:
[docs/jev-ragtruth-benchmark.md](docs/jev-ragtruth-benchmark.md). Interactive case explorer:
`docs/bench/index.html`.

![Jev vs Claude Sonnet 5, GLM Flash and DeepSeek Flash: F1 vs cost per 1,000 judgments](docs/bench/f1_vs_cost.png)

To reproduce or explore the benchmark yourself:

```bash
pip install "ragverdict[bench]"
```

`ragverdict bench ragtruth ...` must be run from the root of a clone (not a `pip install`
of the package) — it reads `bench/frozen_config.json` and `scripts/verify_bench.py`, which
aren't shipped in the PyPI distribution.

## FAQ

### When should I use ragverdict vs RAGAs / DeepEval / TruLens?

They're complementary, not competing. The metric-centric tools (RAGAs, ARES, TruLens,
Phoenix, DeepEval) score response quality dimensions like faithfulness and relevance —
useful for tracking quality over time. ragverdict tests *agent behavior* — did the tools
fire, do the citations resolve to real documents, did the agent push back on a false
premise, does it survive a 10K-character prompt. A mature RAG team uses both:
RAGAs-style scoring for quality tracking + ragverdict for behavioral regression in CI.

### Does it work without an API key?

Yes. Pass `--no-judge` (or set no `ANTHROPIC_API_KEY` and the runner degrades
automatically). Hard assertions still run — `tool_coverage`, citation-vs-corpus
dangling checks, `must_mention` / `must_refuse` / `must_not_cite`,
long-input/multi-turn/empty-input edge cases. The `contradiction` edge case falls back
to a narrow regex heuristic (`_PUSHBACK_HINTS`) with a clear caveat in the FAIL detail
when it can't confidently grade.

### Can I write my own evaluator?

Yes. Subclass `Evaluator`, set a class-level `name`, decorate with `@register`, and
implement `run(adapter, spec, *, judge, thresholds) -> TestResult`. Then `import` your
module before `ragverdict run` or add it to the package's autoload. The bundled
evaluators (`src/ragverdict/evaluators/`) are reference implementations.

### Can I use it with a RAG system written in another language?

Yes — use the `HttpAdapter`. Set `adapter.type: http` + an `endpoint` URL in your
config. The runner POSTs `{prompt, conversation}` and expects a JSON response matching
the `RagResponse` shape. Your Rust / Go / Node / TypeScript / etc. service just needs
to speak that protocol.

### What's the difference between `WEAK` and `FAIL`?

`FAIL` = a hard assertion failed (a required substring was missing, a citation didn't
resolve, an edge case crashed). `WEAK` = all hard assertions held but a judge score
fell into the configurable weak band (default: faithfulness or relevance in `[0.7,
0.85)`). `WEAK` is "watch this," `FAIL` is "fix this." Both `PASS` and `WEAK` give
exit code 0; `FAIL` gives exit code 1.

### Why four-state verdicts instead of floating-point scores?

So they map cleanly to CI exit codes and a 5-second scan of the terminal table. Raw
judge scores still live in `report.json` for users who want them — but the headline
output is a verdict, not a number you have to threshold yourself. The pitch is "pytest
for RAG, not metrics for RAG."

### Can I use a model other than Claude for the judge?

The judge is configurable via `judge.model` in `config.yaml` (defaults to
`claude-sonnet-4-6`). Any current Anthropic model works out of the box. Set
`judge.provider` to switch backends entirely — `jev` (a lightweight probabilistic
judge) and `cascade` (Jev first, escalating to Claude when unsure) ship built in.
See [Judge backends](#judge-backends) below. Any object that satisfies the `Judge`
protocol (`faithfulness`, `relevance`, `refusal`, `pushback`) works, so you can also
write your own.

### How do I integrate this into GitHub Actions?

```yaml
- name: RAG behavioral audit
  run: |
    pip install ragverdict
    ragverdict run config.yaml --no-judge
```

CI exit code propagates naturally — `PASS`/`WEAK` is exit 0, any `FAIL` is exit 1,
config errors are exit 2, all-`ERROR` (typically: judge unreachable) is exit 3. For
live-judge CI runs, set `ANTHROPIC_API_KEY` as a repo secret and drop the
`--no-judge` flag.

### Does prompt caching actually fire?

Yes. Verified 2026-09-22: on Claude Sonnet 5, the ~1,200-token faithfulness rubric is
served from cache (`cache_read_input_tokens=1202` on repeat calls). The wiring is
`cache_control={"type": "ephemeral"}` on every judge rubric. The older caveat — that a
model's minimum cacheable prefix (e.g. 2048 tokens on some models) can exceed a short
rubric — still applies on models with a larger minimum than the rubric length; caching
activates as rubrics grow or on models with a smaller minimum.

## Roadmap

v0.2 shipped the edge-case battery. Next up:

- Write-tool safety evaluator (preview-only verification, version chain checks)
- `auth_negative` kind for the `edge_cases` evaluator (requires adapter ABC extension)
- Native `OpenAI` / `LangChain` adapters
- Concurrent test execution
- Hosted dashboard with regression tracking across runs

## License

MIT — see [LICENSE](./LICENSE).
