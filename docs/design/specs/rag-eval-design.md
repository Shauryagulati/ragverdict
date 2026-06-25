# rag-eval Design Spec

## 1. Problem & Positioning

### The gap

Existing RAG evaluation tools score *metrics* against a fixed dataset. They answer "how faithful is the response on average" — they do not answer "does the agent actually work end-to-end."

| Tool | What it does | What it misses |
|------|--------------|----------------|
| **RAGAs** | LLM-as-judge metrics: faithfulness, answer relevance, context precision/recall on (query, ground-truth, retrieved-context) tuples. | No agent/tool-call testing. No citation-against-source verification. No behavioral assertions. |
| **DeepEval** | pytest-style assertions on the same RAGAs metric family, plus contextual recall. CI-integrated. | Same metric-centric model. Tool coverage, citation audit against a corpus, and write-tool safety are out of scope. |
| **TruLens** | RAG Triad (groundedness, context relevance, answer relevance) + OpenTelemetry tracing. | Observability-centric. Same metric family. |
| **Arize Phoenix** | Tracing platform with LLM evaluators (often wrapping RAGAs/DeepEval). | Heavy infra. Not a CLI you bolt onto an agent. |

The **uncovered gap**: behavioral audits of RAG-*agents* (agents that retrieve + cite + call tools), shaped like pytest, that answer concrete pass/fail questions about whether the system *behaves correctly* — not just whether some metric scores well on average.

### Pitch

**`rag-eval`: pytest for RAG agents.** Point it at any RAG endpoint or Python adapter, give it a YAML config, get a PASS / FAIL / WEAK report covering tool coverage, retrieval quality, citation verification, hallucination, and edge cases.

### Differentiation (one line each)

1. **Tool coverage matrix** — exercises every tool the agent exposes; reports per-tool pass/fail + latency. *None of the four competitors do this.*
2. **Citation audit against the real source corpus** — not just "is the response grounded in retrieved context" (RAGAs/TruLens) but "do the `[src:ID]` citations point to documents that actually exist in the user's corpus, and do those docs contain the cited claim."
3. **Behavioral edge-case battery** — long input, multi-turn coherence, contradiction handling, auth/validation negatives. Assertion-based, not metric-based.
4. **PASS / FAIL / WEAK report** — actionable verdicts, not floating-point scores. Non-zero exit code on FAIL for CI.

---

## 2. V0 Scope

### In scope

| Component | V0 deliverable |
|-----------|----------------|
| CLI | `rag-eval run <config.yaml>` |
| Adapters | (a) Python adapter (user subclasses `RagAdapter`), (b) HTTP endpoint adapter |
| Evaluators | `tool_coverage`, `rag_quality` (direct + hallucination guardrail), `citation_audit` |
| Judges | Anthropic-based `faithfulness` + `relevance` (single module, two prompts, prompt caching enabled) |
| Reporting | Rich terminal table (live updates) + Markdown export + JSON export |
| Demo | One reference RAG agent under `examples/demo_rag/` with a corpus + config |

### Cut from V0 (defer to V1)

- FastAPI dashboard
- LangChain / OpenAI built-in adapters (HTTP adapter handles them generically; native built-ins land in V1)
- Write-tool safety evaluator (specialized, smaller audience)
- Edge-case battery (long input, multi-turn coherence, contradiction, auth-negative) — all V1
- Cloud dashboard / regression tracking (monetization track)

**Rationale:** The initial scope cannot accommodate the full plan and stay polished. V0 must be small enough to demo cleanly and rough-enough-edged that the gap is obvious to early users.

---

## 3. Architecture

### File layout

```
rags-eval/                              # repo root
├── pyproject.toml                      # hatch/uv project, deps, entry point
├── README.md                           # pitch + quickstart + demo gif
├── LICENSE                             # MIT
├── .gitignore
├── docs/
│   └── design/specs/                   # this doc + future specs
├── examples/
│   ├── README.md
│   └── demo_rag/                       # reference agent for the bundled demo
│       ├── adapter.py                  # subclass of RagAdapter
│       ├── corpus/                     # 5-10 markdown source docs
│       └── config.yaml                 # rag-eval config exercising the demo
├── src/rag_eval/
│   ├── __init__.py                     # version, public exports
│   ├── cli.py                          # Click entry point
│   ├── config.py                       # Pydantic schemas: Config, TestCase, EvaluatorRef
│   ├── runner.py                       # orchestrator: loads config, runs evaluators, collects results
│   ├── report.py                       # Rich live table + Markdown/JSON serializers
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py                     # RagAdapter ABC, RagResponse dataclass
│   │   ├── http.py                     # generic HTTP-endpoint adapter
│   │   └── loader.py                   # importlib loader for user Python adapters
│   ├── evaluators/
│   │   ├── __init__.py                 # registry: name -> evaluator class
│   │   ├── base.py                     # Evaluator ABC, TestResult dataclass, Verdict enum
│   │   ├── tool_coverage.py
│   │   ├── rag_quality.py
│   │   └── citation_audit.py
│   └── judges/
│       ├── __init__.py
│       └── llm_judge.py                # Anthropic client, faithfulness + relevance prompts
└── tests/                              # rag-eval's own pytest tests
    ├── test_config.py
    ├── test_runner.py
    ├── test_evaluators/
    │   ├── test_tool_coverage.py
    │   ├── test_rag_quality.py
    │   └── test_citation_audit.py
    └── test_judges/
        └── test_llm_judge.py           # uses recorded fixtures, no live calls
```

### Core types

```python
# adapters/base.py
@dataclass
class RagResponse:
    text: str
    citations: list[Citation]            # [{id, span, source_id}]
    tool_calls: list[ToolCall]           # [{name, args, result, latency_ms, error}]
    retrieved_context: list[ContextDoc]  # [{source_id, chunk, score}]
    raw: dict                            # passthrough for adapter-specific fields

class RagAdapter(ABC):
    @abstractmethod
    def query(self, prompt: str, *, conversation: list[Message] | None = None) -> RagResponse: ...

    def available_tools(self) -> list[ToolSpec]:  # for tool_coverage
        return []

    def corpus(self) -> Iterable[SourceDoc] | None:  # for citation_audit
        return None
```

```python
# evaluators/base.py
class Verdict(str, Enum):
    PASS = "PASS"
    WEAK = "WEAK"   # quality below threshold but no hard assertion failure
    FAIL = "FAIL"   # hard assertion failed
    ERROR = "ERROR" # evaluator crashed

@dataclass
class TestResult:
    name: str
    evaluator: str
    verdict: Verdict
    detail: str             # short human-readable explanation
    metrics: dict[str, float]
    duration_ms: int
    artifacts: dict         # full response, judge output, etc. (for JSON report)

class Evaluator(ABC):
    name: str               # registry key
    @abstractmethod
    def run(self, adapter: RagAdapter, case: TestCase, judge: LLMJudge) -> TestResult: ...
```

### Config shape (YAML)

```yaml
# config.yaml
adapter:
  type: python              # or "http"
  module: examples.demo_rag.adapter
  class: DemoAdapter

judge:
  provider: anthropic
  model: claude-sonnet-4-6
  max_concurrency: 4

thresholds:
  faithfulness_weak: 0.7
  relevance_weak: 0.7

tests:
  - name: tool_coverage_all
    evaluator: tool_coverage
    require_all_tools: true        # every available_tool() must succeed at least once

  - name: direct_retrieval_basics
    evaluator: rag_quality
    cases:
      - query: "What revenue did Acme report in Q1 2025?"
        expects_citations: true
        must_mention: ["$5.2M"]    # substring assertion
      - query: "Summarize the risk factors."
        expects_citations: true

  - name: hallucination_guardrail
    evaluator: rag_quality
    cases:
      - query: "What did Acme acquire in 2030?"   # out-of-corpus
        must_refuse: true            # response must indicate uncertainty / no answer
        must_not_cite: true

  - name: citation_audit
    evaluator: citation_audit
    sample_size: 20                  # run N queries from rag_quality + cross-check citations
```

### Data flow

1. **Load** — `cli.py` parses args, `config.py` validates the YAML into a `Config` Pydantic model.
2. **Wire** — `runner.py` instantiates the adapter (via `adapters/loader.py`), the judge, and each evaluator named in `tests:`.
3. **Execute** — for each test, runner calls `evaluator.run(adapter, case, judge)`. Evaluators may issue multiple adapter queries (e.g., `tool_coverage` fires one per tool) and may consult the judge.
4. **Stream** — runner publishes per-test results to `report.py`, which updates a live Rich table.
5. **Finalize** — runner aggregates verdicts, writes `report.md` + `report.json`, and exits non-zero if any FAIL.

### Evaluator semantics (V0)

- **`tool_coverage`**: calls `adapter.available_tools()`; for each tool, issues a synthesized prompt that should trigger it; records whether `tool_calls` contains that tool with no `error`. Verdict: PASS if all tools fire, FAIL if any required tool errors or is never called.
- **`rag_quality`**: for each case, runs the query, then:
  - hard-asserts `must_mention` substrings, `must_not_cite` flag (no `[src:*]` tokens in response), and `must_refuse` flag (judge `refusal` prompt returns `is_refusal=true`);
  - invokes judge for `faithfulness` (vs `retrieved_context`) and `relevance` (vs `query`);
  - PASS if all hard assertions hold and both judge scores ≥ threshold; WEAK if hard assertions hold but a score falls below the `*_weak` threshold; FAIL otherwise.
- **`citation_audit`**: samples N responses (from prior `rag_quality` runs or fresh queries), then for each citation:
  - checks `source_id` exists in `adapter.corpus()` (dangling-citation check);
  - extracts the cited span, asks judge whether the source document supports the cited claim (per-citation `support_score` in [0,1]).
  - Verdict: PASS if 100% citations resolve AND mean `support_score` ≥ 0.95; WEAK if all resolve but mean `support_score` is in [`threshold_weak`, 0.95); FAIL if any dangling `source_id` or mean `support_score` below `threshold_weak`.

### LLM-as-judge (Anthropic SDK)

- Single `LLMJudge` class with three methods: `faithfulness(response, context) -> JudgeScore`, `relevance(response, query) -> JudgeScore`, and `refusal(response, query) -> RefusalVerdict` (returns `{is_refusal: bool, reasoning: str}`).
- Uses **prompt caching** on the system prompt + rubric (large, reused across N calls). Sonnet 4.6 by default.
- `JudgeScore { score: float (0–1), reasoning: str, supported_claims: int, total_claims: int }`.
- Recorded-fixture mode for unit tests (no live calls in CI).

### Reporting

- **Live table** (Rich): rows = tests, columns = evaluator / verdict / latency / detail. Colors: green PASS, yellow WEAK, red FAIL.
- **Markdown report**: hierarchical sections per evaluator, expandable per-case details, full judge reasoning.
- **JSON report**: machine-readable; suitable for diffing across runs (lays groundwork for V1 regression tracking).

### Error handling

- Adapter errors → `Verdict.ERROR` for the affected test; runner continues.
- Judge API errors → retry with exponential backoff (3 attempts), then mark `ERROR`.
- Config validation errors → fail fast before any tests run.
- No fallbacks for missing API keys — fail fast with a clear message at startup.

---

## 4. Testing strategy (for rag-eval itself)

- `tests/` uses pytest. Fast, no live LLM calls in CI.
- Evaluator tests use **fake adapters** that return canned `RagResponse` objects — lets us assert verdict logic deterministically.
- Judge tests use **recorded fixtures** (JSON files of past Anthropic responses) replayed via a stub client.
- One **integration test** exercises the full CLI against the bundled `examples/demo_rag/` adapter with a fake judge — guards the end-to-end wire.
- Type-check with `mypy --strict` on `src/`.

---

## 5. Build sequence

| Phase | Deliverable | Verification |
|-------|-------------|--------------|
| 1 | Repo scaffold, `pyproject.toml`, `RagAdapter`/`RagResponse`, Pydantic `Config`, `cli.py` skeleton, `runner.py` happy path, example demo_rag adapter stub. | `rag-eval run examples/demo_rag/config.yaml` exits 0 with empty test list. |
| 2 | `tool_coverage` evaluator, `rag_quality` evaluator (hard assertions only, no judge yet), Rich live table. | Demo config exercises tool coverage and hard-assertion rag_quality; terminal table renders. |
| 3 | `LLMJudge` (faithfulness + relevance, prompt caching), wire into `rag_quality` for WEAK verdicts, `citation_audit` evaluator. | Full demo run produces PASS/WEAK/FAIL mix and a Markdown report. |
| 4 | JSON report, exit codes, README with quickstart, `examples/` polish, mypy clean, integration test green. | Fresh-clone install works; README quickstart succeeds. |
| 5 | Package the distribution and publish v0.1.0. | Release tagged; fresh-clone install + quickstart succeed. |

---

## 6. Open questions intentionally deferred

These are V1 concerns. Listed so they don't get smuggled into V0:

- Concurrency model for parallel test execution (V0 = serial; fine for ≤50 tests).
- Adapter authoring docs beyond a single example.
- Built-in OpenAI / LangChain adapters.
- Web dashboard, regression tracking, cloud sync.
- Pricing / billing surface.

---

## 7. Non-goals

- We are **not** building a metric-scoring library that competes head-to-head with RAGAs on faithfulness math. We use LLM-as-judge similarly, but the product surface is assertions and verdicts.
- We are **not** building observability/tracing. TruLens and Phoenix own that.
- We are **not** building a hosted product in V0. Open-source CLI only.
