# LangChain RAG starter

A copy-paste template for wiring a LangChain-backed RAG into rag-eval.

## What's here

| File | Purpose |
|---|---|
| `adapter.py` | `LangChainRagAdapter` — subclass of `RagAdapter` showing the four methods you'd replace with real LangChain calls (retriever, RetrievalQA chain, citation parsing, agent tool calls). Runs offline against a mock corpus. |
| `config.yaml` | rag-eval config that exercises all four evaluators against the adapter. |

## Try it (offline, no dependencies)

```bash
rag-eval run examples/langchain_rag/config.yaml --no-judge
```

LangChain itself is **not** required to run this example — the adapter mocks the
chain so you can see the wiring. To use against a real chain, install LangChain
separately.

## Make it real

The mock adapter has four methods marked `# REPLACE:` with sketches of the real
LangChain implementations:

1. **`__init__()`** — replace the placeholder with construction of your `RetrievalQA`
   chain (or whichever LCEL composition you use). Example sketches a
   `RetrievalQA.from_chain_type(llm=ChatOpenAI(...), retriever=vectorstore.as_retriever(...))`.
2. **`_retrieve()`** — replace with `self._chain.retriever.invoke(query)` and map
   each `Document` to a `ContextDoc`. Be sure to populate `source_id` from your
   document metadata at ingestion time so `citation_audit` works.
3. **`_generate()`** — replace with `self._chain.invoke({"query": prompt})` and map
   `source_documents` to `Citation` objects.
4. **`_tool_calls_for()`** — if you use `AgentExecutor` with tools, replace with
   parsing `intermediate_steps` from the agent result. If you only use a
   non-agentic chain, return `[]`.

The four methods are small so the diff between the mock and the real adapter is
exactly the parts that connect to LangChain.

## Tips for LangChain users

- **`source_id` is your friend.** When you ingest documents into your vector store,
  attach a stable `source_id` to each chunk's metadata. `citation_audit` relies on it
  to verify citations resolve.
- **Conversation memory.** The mock folds prior user turns into the search query;
  real LangChain adapters typically use `ConversationBufferMemory` or similar
  attached to the chain. Make sure `multi_turn` edge cases pass with your real
  memory wiring.
- **Streaming.** If you use streaming chains (`chain.stream()`), collect the chunks
  into the final response before returning from `query()` — `RagResponse` expects a
  complete answer.

## What rag-eval will catch (worth knowing before you wire it up)

| Failure mode | Caught by |
|---|---|
| `RetrievalQA` returns `source_documents` with metadata missing `source_id` (citations point at nothing) | `citation_audit` (dangling check) |
| An `AgentExecutor` tool was removed from the registry but still referenced in the agent's prompt | `tool_coverage` |
| Your prompt template doesn't constrain the model to refuse out-of-corpus queries | `rag_quality` with `must_refuse: true` |
| The chain happily answers a question with a false premise instead of pushing back | `edge_cases.contradiction` |
| Long input (10K+ chars) exceeds the underlying model's context window and the chain raises | `edge_cases.long_input` |
| `ConversationBufferMemory` isn't wired into your chain so multi-turn context is lost | `edge_cases.multi_turn` |
