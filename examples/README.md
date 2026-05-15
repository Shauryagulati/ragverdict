# Examples

## `demo_rag/`

A reference RAG agent that runs against a tiny in-tree markdown corpus about a fictional
"Acme Corp." It exists to (a) make rag-eval's `rag-eval run` go on a fresh clone without
any setup, and (b) demonstrate what a `RagAdapter` looks like.

### What's in here

```
demo_rag/
├── adapter.py        # DemoAdapter — substring-retrieval over the markdown corpus
├── config.yaml       # exercises all three V0 evaluators
└── corpus/
    ├── intro.md      # Acme overview
    ├── revenue.md    # $5.2M Q1 2025, segment breakdown, bookings
    ├── products.md   # RouteOps (flagship), InventoryMind (secondary)
    ├── risks.md      # customer concentration, FX, regulatory, talent
    └── leadership.md # Jane Smith (CEO), Raj Patel (CTO), Maria Chen (VPS)
```

### Run it

```bash
# With the LLM judge (requires ANTHROPIC_API_KEY)
rag-eval run examples/demo_rag/config.yaml

# Offline — hard assertions only, no judge calls
rag-eval run examples/demo_rag/config.yaml --no-judge
```

### What each test exercises

| Test                       | Evaluator        | What it asserts                                                                         |
|----------------------------|------------------|-----------------------------------------------------------------------------------------|
| `tool_coverage_all`        | `tool_coverage`  | Fires the two tools (`search_corpus`, `get_company_profile`) and checks both succeed.   |
| `direct_retrieval_basics`  | `rag_quality`    | Three queries with substring + citation assertions; judge scores faithfulness + relevance. |
| `hallucination_guardrail`  | `rag_quality`    | Two out-of-corpus queries; `must_refuse` + `must_not_cite` to verify clean refusals.    |
| `citation_audit_basics`    | `citation_audit` | Audits citations from three sample queries against the markdown corpus.                  |

### How DemoAdapter "retrieves"

Substring matching with a stopword filter — score each paragraph by token hits, pick the
densest, refuse if no paragraph crosses a minimum-hit threshold. It is deliberately simple.
That is the point: a real RAG system will have higher recall and better citation
fidelity, and rag-eval's job is to verify *that*.

### Adapting this to your project

Copy `adapter.py` and replace `_retrieve` / `_tool_calls_for` / the response synthesis
with calls into your real RAG pipeline. Then point `config.yaml`'s `adapter.module` at
your file. The runner inserts the current working directory into `sys.path` before
resolving your adapter module, so a project-local Python package just works.
