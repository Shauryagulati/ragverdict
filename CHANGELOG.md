# Changelog

All notable changes to `rag-eval` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-18

Fourth evaluator: input-boundary failure modes that V0's golden-path evaluators don't
catch.

### Added

- `edge_cases` evaluator with four kinds:
  - `long_input` — verifies the adapter handles ≥10K-char prompts without crash and
    within a configurable timeout (default 30s). Cross-platform timeout via
    `concurrent.futures.ThreadPoolExecutor`.
  - `multi_turn` — verifies the adapter uses conversation history. Builds an N-turn
    conversation from `turns: [...]` with placeholder assistant turns, sends `final_query`,
    asserts response contains every `must_reference` substring.
  - `contradiction` — verifies the adapter pushes back on false premises rather than
    silently accepting them. Judge-graded via the new `pushback()` rubric; heuristic
    fallback (substring match against `_PUSHBACK_HINTS`) for `--no-judge` runs.
  - `empty_input` — verifies the adapter rejects empty/whitespace input cleanly (either
    via controlled exception or refusal-style response).
- `LLMJudge.pushback(response, false_premise) -> PushbackVerdict` — fourth rubric.
  Semantically distinct from `refusal()`: an agent can correct a premise while still
  answering, which the refusal rubric wouldn't grade as such.
- YAML config: `edge_cases` test entries use a Pydantic v2 discriminated union on
  `kind` so each case-type validates against the right sub-model.
- Regression smoke: `TruncatingAdapter` and `CompliantAdapter` join the existing three
  known-broken adapters that prove the framework catches what it claims.
- GitHub Actions CI workflow (ruff + mypy --strict + pytest, py3.10/3.11/3.12 matrix).
- CI / Python-version / MIT-license badges on the README.

### Changed

- `DemoAdapter` now folds prior user-turn content into retrieval when a conversation is
  supplied (keeps `multi_turn` deterministic against the corpus). Also handles empty
  input with an explicit rejection response.
- README: pitch updated to four evaluators; competitor-gap callout extended to cover
  the `edge_cases` differentiation.

## [0.1.0] — 2026-05-15

Initial release.

### Added

- `RagAdapter` ABC and `RagResponse` contract — connect any RAG agent by subclassing or
  pointing at an HTTP endpoint
- YAML configuration with Pydantic validation
- Three evaluators:
  - `tool_coverage` — fires every `available_tools()` entry, reports per-tool pass/fail + latency
  - `rag_quality` — hard assertions (`must_mention`, `must_not_cite`, `must_refuse`,
    `expects_citations`) plus optional LLM-as-judge faithfulness + relevance scoring for
    `WEAK`/`FAIL` verdicts
  - `citation_audit` — verifies every `[src:ID]` resolves to the corpus and (with judge)
    scores per-citation support
- `LLMJudge` backed by the Anthropic SDK with `messages.parse()` + Pydantic schemas; three
  rubrics (faithfulness, relevance, refusal) under `cache_control={"type":"ephemeral"}`
- Click CLI: `rag-eval run <config.yaml>` with `--no-judge` and `--out-dir` options
- Rich live table during the run; Markdown + JSON reports at the end
- Bundled `examples/demo_rag/` reference agent over a fictional "Acme Corp" markdown corpus
- 41 tests; mypy `--strict` clean; ruff clean
