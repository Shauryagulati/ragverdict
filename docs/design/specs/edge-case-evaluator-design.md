# Edge-Case Battery Evaluator — Design Spec

Builds on the [base ragverdict design spec](./ragverdict-design.md).

## 1. Problem & Motivation

V0 ships three evaluators (`tool_coverage`, `rag_quality`, `citation_audit`) that cover the
*golden paths* a RAG agent must handle. They do not exercise the *boundaries* where most
production failures actually live:

- An agent that quietly truncates a 50KB prompt and answers a partial question.
- An agent that loses prior-turn context partway through a multi-turn conversation.
- An agent that confidently confirms a false premise embedded in the user's question
  ("…when Acme acquired XYZ in 2030, what was the rationale?").
- An agent that crashes — or worse, answers — when given empty input.

None of the four incumbents (RAGAs, DeepEval, TruLens, Phoenix) test these directly. They
are operational failures that score-based evals miss. This evaluator covers the four
input-boundary categories most likely to surface in production.

### Pitch

`edge_cases` — a fourth evaluator in the ragverdict battery that exercises four input-
boundary failure modes (`long_input`, `multi_turn`, `contradiction`, `empty_input`) with
hard-assertion verdicts. One new judge method (`pushback`) to grade contradiction
handling; everything else is assertion-based.

---

## 2. Scope

### In scope (v0.2.0)

| Kind | What it tests |
|---|---|
| `long_input` | Adapter handles a ≥10K-char prompt without crash, returns within timeout, response non-empty |
| `multi_turn` | Adapter retains context across an N-turn conversation; final response contains required substrings |
| `contradiction` | Adapter pushes back on a false premise rather than agreeing (judge-graded) |
| `empty_input` | Adapter rejects empty input cleanly — either a controlled error or a refusal-style response |

### Deferred (v0.3.0+)

- `auth_negative` — requires extending the `RagAdapter` ABC with an auth surface. None of
  the bundled adapters need auth today; we will design and ship this when the first
  adapter that uses authentication is contributed. Cleaner to skip the parameter than to
  add API surface every adapter must ignore.

---

## 3. Architecture

### File layout

```
src/ragverdict/
├── evaluators/
│   └── edge_cases.py            ← NEW: evaluator module
└── judges/
    └── llm_judge.py             ← MODIFIED: +pushback() method + PushbackVerdict schema

tests/
├── test_evaluators/
│   └── test_edge_cases.py       ← NEW: per-kind unit tests with fake adapters
├── test_judges/
│   └── test_llm_judge.py        ← MODIFIED: +pushback() tests with stub client
└── test_regression_smoke.py     ← MODIFIED: +TruncatingAdapter, +CompliantAdapter

examples/demo_rag/
├── adapter.py                   ← MODIFIED: tolerate long + empty input cleanly
└── config.yaml                  ← MODIFIED: +edge_cases test entry exercising all 4 kinds
```

### Pydantic spec (discriminated union)

```python
from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Field

class LongInputCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["long_input"]
    length: int = 10000
    timeout_s: float = 30.0

class MultiTurnCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["multi_turn"]
    turns: list[str] = Field(min_length=1)   # at least 1 prior user turn
    final_query: str                         # the question that requires recalling earlier turns
    must_reference: list[str] = Field(min_length=1)  # substrings the response must contain

class ContradictionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["contradiction"]
    query: str                          # the full query embedding the false premise
    false_premise: str                  # the specific claim the agent must push back on

class EmptyInputCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["empty_input"]
    prompt: str = ""                    # default empty; configurable to whitespace-only
    allow_error: bool = True            # controlled exception counts as PASS
    allow_refusal: bool = True          # refusal-style response counts as PASS

EdgeCase = Annotated[
    Union[LongInputCase, MultiTurnCase, ContradictionCase, EmptyInputCase],
    Field(discriminator="kind"),
]

class EdgeCasesSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cases: list[EdgeCase] = Field(default_factory=list)
```

### YAML shape

```yaml
- name: edge_cases
  evaluator: edge_cases
  cases:
    - kind: long_input
      length: 10000
    - kind: multi_turn
      turns:
        - "Who is the CEO of Acme?"
        - "What is their background?"
      final_query: "Remind me — what was their name again?"
      must_reference: ["Jane Smith"]
    - kind: contradiction
      query: "What was the rationale for Acme's acquisition of XYZ Corp in 2030?"
      false_premise: "Acme acquired XYZ Corp in 2030"
    - kind: empty_input
```

### Per-kind semantics

#### `long_input`

- **Prompt construction:** A filler prefix (`"a " * (length // 2)` trimmed to `length`)
  followed by `"\n\nSummarize the company's revenue in one sentence."` so there is a real
  question at the end. Total prompt length ≈ `length` chars.
- **Execution:** Run via `concurrent.futures.ThreadPoolExecutor(max_workers=1)` with
  `timeout_s` (cross-platform; signal-based timeouts don't work on Windows).
- **PASS:** Adapter returns a `RagResponse` with non-empty `text` within `timeout_s`.
- **FAIL:** Adapter raises, returns empty `text`, or exceeds `timeout_s`.

#### `multi_turn`

- **Prompt construction:** Build a `list[Message]` of length `2 * len(turns)` alternating
  user/assistant: `[user(turns[0]), assistant("(prior turn — context only)"), user(turns[1]),
  assistant("(prior turn — context only)"), …]`. (`Message` is the existing dataclass in
  `src/ragverdict/adapters/base.py` with `role: str` and `content: str` fields.) Then call
  `adapter.query(final_query, conversation=<built list>)`.
- **PASS:** Response text contains every substring in `must_reference` (case-insensitive).
- **FAIL:** Any required substring missing.

> **Note on the placeholder assistant turns:** This is a known simplification. A more
> thorough multi-turn test would use the agent's actual prior responses — but capturing
> those requires multi-call orchestration that complicates the case for marginal gain.
> The placeholder approach still verifies the agent *uses* the conversation history,
> which is the property under test. Documented in the evaluator docstring.

#### `contradiction`

- **Execution:** Send `query` (which embeds `false_premise`) to the adapter via the normal
  `query()` call.
- **With judge:** Call `judge.pushback(response_text, false_premise)`. PASS if
  `handled_correctly=True`; FAIL otherwise.
- **Without judge (`--no-judge`):** Fall back to a regex heuristic over the response text.
  The hint list lives as a module-level constant `_PUSHBACK_HINTS` in `edge_cases.py` and
  is a starting tuple of: `"actually,"`, `"that's not"`, `"that is not"`, `"incorrect"`,
  `"i don't have"`, `"i do not have"`, `"no record"`, `"no, "`, `"not aware"`
  (case-insensitive). PASS on any match, FAIL otherwise with detail noting "heuristic
  fallback — install judge for reliable contradiction grading."

#### `empty_input`

- **Execution:** Call `query(case.prompt)` (default `""`).
- **PASS:** Either (a) adapter raises any `Exception` and `allow_error=True`, or
  (b) returns a refusal-style response (matched via the same `_REFUSAL_HINTS` heuristic
  rag_quality already uses) and `allow_refusal=True`.
- **FAIL:** Adapter returns a substantive non-refusal response, or raises but
  `allow_error=False`, or refuses but `allow_refusal=False`.

### Judge extension: `pushback()`

```python
# In src/ragverdict/judges/llm_judge.py

class PushbackVerdict(BaseModel):
    handled_correctly: bool
    reasoning: str

class LLMJudge:
    # existing: faithfulness(), relevance(), refusal()
    def pushback(self, response: str, false_premise: str) -> PushbackVerdict:
        """Grade whether `response` correctly pushed back on `false_premise`.

        Pushback = explicitly correcting the false premise, OR declining to answer
        because the premise is unsupported. Confidently answering as if the premise
        were true = handled_correctly: false.
        """
        ...
```

Rubric is a ~300-token system block under `cache_control={"type": "ephemeral"}` — same
shape as the existing three rubrics. Structured output via `messages.parse()` + the
`PushbackVerdict` Pydantic schema. Same retry/error semantics as `judge.refusal()`.

### Verdict rollup

- Per case: `Verdict.PASS / FAIL / ERROR`. **No WEAK** — these are binary safety
  properties, not quality scores in a tunable range.
- Test verdict: `_worst_verdict([case verdicts])` — reuses the same helper as
  `rag_quality.py`. Order: ERROR > FAIL > PASS.
- `TestResult.metrics`: `cases_total`, `cases_passed`, `cases_failed`, `cases_errored`.
- `TestResult.artifacts.cases[]`: per-case `{kind, verdict, detail, response_text}` for
  the JSON report.

### Error handling

| Failure | Outcome |
|---|---|
| Adapter raises during a non-empty-input case | Case = `ERROR` with `f"adapter raised: {exc}"`, evaluator continues |
| `long_input` exceeds `timeout_s` | Case = `FAIL` with `f"timed out after {timeout_s}s"` |
| Judge call fails on `contradiction` | Case = `ERROR` with `f"judge.pushback() failed: {exc}"` |
| Pydantic validation fails on the spec (incl. empty `turns` or empty `must_reference` for `multi_turn`) | Whole test = `ERROR` (same as `rag_quality`/`citation_audit`) |

---

## 4. Test strategy

### Unit tests (`tests/test_evaluators/test_edge_cases.py`)

Fake adapters, one per pass/fail axis per kind. Each test instantiates the adapter,
invokes the evaluator directly with a spec, asserts on the returned `TestResult`.

| Adapter | Behavior | Asserts |
|---|---|---|
| `LongTolerantAdapter` | Echoes back length info regardless of input size | `long_input` → PASS |
| `TruncatingAdapter` | Raises on prompts > 1000 chars | `long_input` → FAIL ("adapter raised") |
| `SlowAdapter` | `time.sleep(timeout_s + 1)` then returns | `long_input` → FAIL ("timed out") |
| `RecallAdapter` | Reads `conversation`, includes the first user turn's content in response | `multi_turn` → PASS |
| `AmnesiaAdapter` | Ignores `conversation`, always responds the same way | `multi_turn` → FAIL ("missing required substring") |
| `CorrectingAdapter` + stub judge `handled_correctly=True` | Returns "Actually, that didn't happen" | `contradiction` → PASS |
| `CompliantAdapter` + stub judge `handled_correctly=False` | Echoes the false premise | `contradiction` → FAIL |
| `CorrectingAdapter` + no judge | Returns "Actually, that's not in my records" | `contradiction` → PASS (regex match) |
| `CompliantAdapter` + no judge | Returns "Yes, in 2030 Acme acquired XYZ" | `contradiction` → FAIL (no regex match) |
| `EmptyRefusingAdapter` | Returns "I cannot process empty input" | `empty_input` → PASS |
| `EmptyAnsweringAdapter` | Returns "Sure, here's a response..." | `empty_input` → FAIL |
| `EmptyRaisingAdapter` | Raises `ValueError("empty prompt")` | `empty_input` → PASS (allow_error=True) |

Plus one spec-validation test (missing `must_reference` on `multi_turn` → `ERROR`).

### Judge unit tests (`tests/test_judges/test_llm_judge.py`)

Stub the `anthropic.Anthropic` client (existing pattern). Add tests:
- `pushback` returns the parsed `PushbackVerdict` from a stubbed `messages.parse()` response.
- `pushback` retries on transient errors (mirrors existing retry tests for other methods).
- `pushback` raises if all retries exhausted.

### Regression smoke (`tests/test_regression_smoke.py`)

Add two adapters and the assertions that prove ragverdict flags them:
- `TruncatingAdapter` — silently drops anything past 1000 chars and answers the partial
  prompt. Run with `edge_cases.long_input` → must produce FAIL.
- `CompliantAdapter` — agrees with any false premise without pushback. Run with
  `edge_cases.contradiction` (no judge, heuristic fallback) → must produce FAIL.

These join `DanglingCitationAdapter` / `SilentToolAdapter` / `ConfidentlyWrongAdapter` as
the "framework catches what it claims" guarantee.

### Integration (`examples/demo_rag/config.yaml` + `test_cli.py`)

- Add an `edge_cases` test to the demo config exercising all 4 kinds against
  `DemoAdapter`. Adapter modifications (likely small) to handle long input cleanly and
  refuse empty input.
- Verify the existing `test_cli.py` end-to-end test still passes; coverage of the new
  evaluator is exercised through the demo run.

### Coverage target

The new evaluator + judge method should land at ≥90% line coverage in their own modules.
Overall repo coverage should hold ≥88% (current) or improve.

---

## 5. Demo wiring

`DemoAdapter` modifications:
- `query()`: if `prompt` is empty/whitespace, return `RagResponse(text="I cannot process empty input.", ...)`.
- `query()`: tolerate long input by truncating internally for retrieval but acknowledging
  the question — keeps `long_input` PASS without ABC changes.

`examples/demo_rag/config.yaml` gets a new test entry:
```yaml
- name: edge_cases
  evaluator: edge_cases
  cases:
    - kind: long_input
      length: 10000
    - kind: multi_turn
      turns:
        - "Who is the CEO of Acme?"
      final_query: "What was the name you just told me?"
      must_reference: ["Jane Smith"]
    - kind: contradiction
      query: "When did Acme acquire XYZ Corp in 2030?"
      false_premise: "Acme acquired XYZ Corp in 2030"
    - kind: empty_input
```

Expected result on `ragverdict run examples/demo_rag/config.yaml`: 5 tests instead of 4, all
PASS.

---

## 6. Open questions intentionally deferred

- **`auth_negative` kind.** Needs an adapter API change. Ship in v0.3 with a real
  auth-using adapter.
- **Concurrency.** Edge-case cases (especially `long_input` with 30s timeout) are
  serial today. V1 spec's deferred concurrency item still applies.
- **Custom kinds.** Some users will want their own edge cases (e.g., "agent must respect
  rate limits"). A `custom` kind with user-supplied callable is V0.3+.
- **Multi-turn with real assistant responses.** Current design uses placeholder assistant
  turns. A "live" multi-turn (where the adapter generates each turn's response and the
  next turn references it) is a meaningfully more expensive case — defer to v0.3.

---

## 7. Non-goals

- We are **not** adding adversarial / jailbreak testing. Edge cases ≠ red-teaming. That
  is a separate evaluator family (and a different threat model) that needs its own
  thinking.
- We are **not** instrumenting the adapter for timing/profiling. `long_input` measures
  *did it complete within timeout*, not *how long it took*. Latency profiling is a
  separate V1 item.
- We are **not** auto-generating long-input filler from the corpus. Repeated filler is
  sufficient to stress input handling; corpus-derived filler would couple the evaluator
  to `adapter.corpus()` for no test-power gain.

---

## 8. Build sequence (estimated 1-2 days)

| Step | Deliverable | Verification |
|---|---|---|
| 1 | `LongInputCase`, `MultiTurnCase`, `ContradictionCase`, `EmptyInputCase` + `EdgeCasesSpec` Pydantic models with discriminated union; `tests/test_evaluators/test_edge_cases.py` spec-validation tests | Pydantic round-trips YAML; invalid `kind` rejected |
| 2 | `EdgeCasesEvaluator.run()` skeleton + per-kind handlers for `long_input` and `empty_input` (no-judge kinds); unit tests for both | All 4 fake adapters for these kinds produce expected verdicts |
| 3 | `multi_turn` handler + unit tests | `RecallAdapter` PASS, `AmnesiaAdapter` FAIL |
| 4 | `pushback()` added to `LLMJudge` + judge unit tests | Stubbed client returns parsed `PushbackVerdict` |
| 5 | `contradiction` handler (judge + heuristic fallback) + unit tests | All 4 fake adapter × judge-present/absent combinations produce expected verdicts |
| 6 | Regression smoke: `TruncatingAdapter` + `CompliantAdapter` | Both produce FAIL when run through the evaluator |
| 7 | `DemoAdapter` + `examples/demo_rag/config.yaml` updates; `test_cli.py` still green | Full demo run produces 5/5 PASS with judge, 5/5 PASS with `--no-judge` |
| 8 | README update: bump evaluator count to 4 in the pitch and competitor-comparison sections; CHANGELOG entry for v0.2.0 | `mypy --strict`, `ruff`, `pytest --cov` all clean |

All steps gated on `mypy --strict` + `ruff` + `pytest` green before advancing.
