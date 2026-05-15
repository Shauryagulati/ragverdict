# Changelog

All notable changes to `rag-eval` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
