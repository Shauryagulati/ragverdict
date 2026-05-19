# OpenAI RAG starter

A copy-paste template for wiring an OpenAI-backed RAG into rag-eval.

## What's here

| File | Purpose |
|---|---|
| `adapter.py` | `OpenAIRagAdapter` — subclass of `RagAdapter` showing the four methods you'd replace with real OpenAI calls (embeddings, chat completion, citation parsing, tool extraction). Runs offline against a mock corpus. |
| `config.yaml` | rag-eval config that exercises all four evaluators against the adapter. |

## Try it (offline, no API key needed)

```bash
rag-eval run examples/openai_rag/config.yaml --no-judge
```

This runs the mock adapter against the bundled corpus and shows the test surface.

## Make it real

The mock adapter has four methods marked `# REPLACE:` with sketches of the real OpenAI
implementations:

1. **`_retrieve()`** — replace with `openai.embeddings.create()` + your vector store
   (Qdrant, Pinecone, pgvector, etc.) `.search()`.
2. **`_generate()`** — replace with `openai.chat.completions.create()` using a system
   prompt instructing the model to cite sources as `[src:ID]`.
3. **`_tool_calls_for()`** — replace with parsing
   `resp.choices[0].message.tool_calls` from the OpenAI response.
4. **`available_tools()`** — replace the placeholder with your real function-calling
   tool schemas.

The four methods are intentionally small so the diff between the mock and the real
adapter is exactly the parts that connect to OpenAI.

## What rag-eval will catch (worth knowing before you wire it up)

| Failure mode | Caught by |
|---|---|
| OpenAI function-calling spec changed and a tool silently no-ops | `tool_coverage` |
| Your citation parser emits `[src:abc-1]` for a `source_id` that was actually `abc-12345` | `citation_audit` (dangling check) |
| The model confidently answers an out-of-corpus question | `rag_quality` with `must_refuse: true` |
| The model agrees with a false premise in the user's question | `edge_cases.contradiction` |
| Large input (10K+ chars) trips your embedding or chat-completion request | `edge_cases.long_input` |
| Multi-turn conversation loses earlier context | `edge_cases.multi_turn` |
| Empty input gets a substantive made-up response | `edge_cases.empty_input` |
