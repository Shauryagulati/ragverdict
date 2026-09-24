# Changelog

All notable changes to `ragverdict` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-09-23

Adds a `Judge` protocol so any scorer — not just `LLMJudge` — can back ragverdict, plus
two new backends (Jev, and a Jev→Claude cascade), and a RAGTruth benchmark comparing them.

### Added

- `Judge` protocol (`faithfulness`, `relevance`, `refusal`, `pushback`) that any judge
  backend implements; `JudgeTransportError` for transport-level failures distinct from
  judge-level errors; results can now carry an optional `confidence`.
- `judge.provider: jev | cascade` in config, alongside the existing `anthropic`. New
  `judge` fields: `thinking`, `jev_model`, `jev_base_url`, `cascade_band`.
- `JevJudge`: a lightweight probabilistic judge backend (TypeSafe's Jev, via OpenRouter or
  TypeSafe directly).
- `CascadeJudge`: Jev first, escalating to an LLM judge only when Jev's probability falls
  inside `cascade_band`.
- `ragverdict bench ragtruth` subcommands and the `[bench]` extra (`numpy`, `scikit-learn`,
  `matplotlib`) for reproducing the Jev-vs-LLM-judge benchmark against RAGTruth. See
  [docs/jev-ragtruth-benchmark.md](docs/jev-ragtruth-benchmark.md) for method,
  pre-registration, and results.
- A warning printed once to stderr when a `jev`/`cascade` judge is configured with the
  still-default `faithfulness_pass`/`faithfulness_weak` thresholds, which were calibrated
  for an LLM's claim-fraction score and are usually too strict for Jev's probability score.
- `examples/demo_rag/config.jev.yaml`: a worked example of a Jev-backed cascade config with
  thresholds set for Jev's score.

### Changed

- `LLMJudge`'s `max_tokens` default raised from 1024 to 4096, with clear truncation and
  refusal errors instead of a bare JSON-parse failure — fixes truncated judge output on
  thinking-by-default models such as Claude Sonnet 5.
- Every `LLMJudge` request is now built as `messages.create` kwargs plus an `output_config`
  JSON schema (via `anthropic.transform_schema`) rather than `messages.parse()`; the same
  request body is reusable against the Batch API.
- `anthropic` dependency bumped to `>=0.102` (required for structured `output_config` and
  `anthropic.transform_schema`).

## [0.2.1] — 2026-06-25

### Changed

- **Renamed the project from `rag-eval` to `ragverdict`.** The PyPI name `rag-eval` was
  already taken by an unrelated project, and `ragverdict` better reflects the output model
  — PASS / WEAK / FAIL **verdicts**, not metric averages. No functional changes: the CLI is
  now `ragverdict run …` and the public import is `import ragverdict`.

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
- Click CLI: `ragverdict run <config.yaml>` with `--no-judge` and `--out-dir` options
- Rich live table during the run; Markdown + JSON reports at the end
- Bundled `examples/demo_rag/` reference agent over a fictional "Acme Corp" markdown corpus
- 41 tests; mypy `--strict` clean; ruff clean
