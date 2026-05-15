# `report.json` schema

`rag-eval` writes `report.json` to `--out-dir` (default `./report/`) at the end of every
run. The shape is stable: V1's regression-tracking story rests on diffing successive
reports, so additions are backwards-compatible and removals require a major bump.

## Top-level shape

```jsonc
{
  "tests": [ /* TestResult, one per spec in config.tests */ ],
  "summary": {
    "PASS":  3,
    "WEAK":  0,
    "FAIL":  1,
    "ERROR": 0
  }
}
```

## `TestResult`

```jsonc
{
  "name": "direct_retrieval_basics",   // matches config.tests[i].name
  "evaluator": "rag_quality",          // matches config.tests[i].evaluator
  "verdict": "PASS",                   // PASS | WEAK | FAIL | ERROR
  "detail": "3/3 cases passed",        // short human-readable summary
  "metrics": {                          // evaluator-specific numbers
    "cases_total": 3.0,
    "cases_passed": 3.0,
    "cases_weak":   0.0
  },
  "duration_ms": 12662,
  "artifacts": { /* evaluator-specific — see per-evaluator sections below */ }
}
```

## `metrics` and `artifacts` by evaluator

### `tool_coverage`

```jsonc
{
  "metrics": {
    "tools_total":  2.0,
    "tools_passed": 2.0
  },
  "artifacts": {
    "per_tool": [
      {
        "tool":       "search_corpus",
        "fired":      true,
        "error":      null,             // string when fired=false or the tool errored
        "latency_ms": 8
      }
    ]
  }
}
```

### `rag_quality`

```jsonc
{
  "metrics": {
    "cases_total":  3.0,
    "cases_passed": 3.0,
    "cases_weak":   0.0
  },
  "artifacts": {
    "cases": [
      {
        "query":         "What revenue did Acme report in Q1 2025?",
        "verdict":       "PASS",
        "detail":        "faithfulness=1.00, relevance=1.00",
        "response_text": "Acme reported $5.2M in Q1 2025 revenue… [src:REVENUE]",
        "metrics": {
          "faithfulness": 1.0,
          "relevance":    1.0
        }
      }
    ]
  }
}
```

`metrics` on each case is empty when hard assertions fail before the judge runs, or
when the judge is not configured.

### `citation_audit`

```jsonc
{
  "metrics": {
    "citations_audited":    3.0,
    "citations_dangling":   0.0,
    "mean_support_score":   1.0    // present only when judge is configured
  },
  "artifacts": {
    "per_citation": [
      {
        "query":         "Who is the CTO of Acme?",
        "citation_id":   "cite-LEADERSHIP",
        "source_id":     "LEADERSHIP",
        "resolved":      true,
        "span":          "Raj Patel is CTO. He joined Acme in 2020 from a fintech startup…",
        "support_score": 1.0,
        "reasoning":     "Span is verbatim from the LEADERSHIP source."
      }
    ]
  }
}
```

A dangling citation entry has `"resolved": false` and a `detail` describing the failure.
Per-citation entries without a `support_score` indicate either no judge was configured
or the citation had no `span` to evaluate.

## Stability notes

- **`verdict`** values (`PASS`/`WEAK`/`FAIL`/`ERROR`) are stable.
- New keys may appear inside `metrics` or `artifacts.*` in any version. Consumers should
  ignore unknown keys.
- The top-level `summary` always contains all four verdict counts, including zeros.
- `metrics` values are always JSON numbers (floats). Integer-valued counters are encoded
  as `2.0`, not `2`.
- `duration_ms` is integer milliseconds (rounded down).
