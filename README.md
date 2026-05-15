# rag-eval

**pytest for RAG agents.** Point it at any RAG endpoint or Python adapter, give it a YAML config, get a PASS / FAIL / WEAK report covering tool coverage, retrieval quality, citation verification, hallucination, and edge cases.

> Status: pre-release scaffold. Full quickstart, demo, and asciinema land Day 4 of the build week. See [the design doc](./docs/superpowers/specs/2026-05-15-rag-eval-design.md) for the V0 plan.

## Why rag-eval

Existing RAG evaluation tools (RAGAs, DeepEval, TruLens, Phoenix) score *metrics* — faithfulness, relevance, context precision. They answer "how faithful was the response on average." They do not answer "does the agent actually work end-to-end."

`rag-eval` answers the behavioral question: does every tool fire, do citations point to real source documents, does the agent refuse to hallucinate when it should? Verdicts, not scores. Built around the same `pass / weak / fail` model as `pytest`, so it slots into CI without inventing new ceremony.

## License

MIT — see [LICENSE](./LICENSE).
