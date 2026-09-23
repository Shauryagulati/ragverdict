"""Export the data behind docs/bench/index.html.

`export` writes `results.json`: the untuned headline table (Judge, n, AUROC CI, F1 CI, $/1k
plus the cascade row), the H1/H9 verdicts, the sensitivity analyses, the pilot-replication
table, the case explorer's disagreement cases, and the label audit (if one has been run). Every
number the page shows comes from this file — nothing is computed client-side beyond filtering.

Cases are examples where untuned Jev (run "jev-para-A" @ the threshold the caller passes — the
frozen pre-registration's untuned threshold, 0.5) disagrees with Claude's pre-registered rule
(`llm_hallucinated`: score < 1.0 or an unsupported claim), or where both are wrong against the
human label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragverdict.bench.predict import PredictionStore
from ragverdict.bench.ragtruth import Example
from ragverdict.bench.summary import HEADLINE_JEV_RUN, llm_hallucinated


def _headline_table(summary: dict[str, Any]) -> dict[str, Any]:
    """{is_headline, cohort_n, judges: {key: {run, n, f1, auroc, cost_usd_per_1k}}, cascade}.

    `is_headline` is True once the untuned cohort (headline) exists; on a partial run (only
    some judges scored so far) it falls back to each judge's own block and its own n.
    """
    judges = summary.get("judges") or {}
    headline = summary.get("headline")
    rows: dict[str, Any] = {}
    if headline:
        for key, block in headline["judges"].items():
            src = "jev" if key == "jev_tuned" else key  # jev_tuned's cost lives on run "jev"
            cost_block = judges.get(src)
            if cost_block is None:
                continue
            rows[key] = {
                "run": block["run"], "n": block["n"], "f1": block["f1"],
                "macro_f1": block["macro_f1"],
                "auroc": cost_block["auroc"], "cost_usd_per_1k": cost_block["cost_usd_per_1k"],
            }
    else:
        for key in ("jev_untuned", "jev", "claude", "deepseek", "glm"):
            block = judges.get(key)
            if block is None:
                continue
            rows[key] = {
                "run": block["run"], "n": block["n"], "f1": block["f1"],
                "macro_f1": block["macro_f1"],
                "auroc": block["auroc"], "cost_usd_per_1k": block["cost_usd_per_1k"],
            }
    return {
        "is_headline": bool(headline),
        "cohort_n": headline["cohort"]["n"] if headline else summary.get("n_test"),
        "judges": rows,
        "cascade": (summary.get("cascade") or {}).get("frozen_band"),
    }


def _cases(
    examples: list[Example], store: PredictionStore, frozen_threshold: float
) -> list[dict[str, Any]]:
    jev = {
        eid: p for (eid, rep), p in store.load(HEADLINE_JEV_RUN).items()
        if rep == 0 and p.score is not None
    }
    claude = {
        eid: p for (eid, rep), p in store.load("claude").items()
        if rep == 0 and p.score is not None
    }
    cases: list[dict[str, Any]] = []
    for e in examples:
        if e.id not in jev or e.id not in claude:
            continue
        jev_p, claude_p = jev[e.id], claude[e.id]
        assert jev_p.score is not None  # filtered above
        jev_says = jev_p.score < frozen_threshold
        claude_says = llm_hallucinated(claude_p)
        disagree = jev_says != claude_says
        both_wrong = jev_says != e.hallucinated and claude_says != e.hallucinated
        if not (disagree or both_wrong):
            continue
        cases.append({
            "id": e.id, "task": e.task, "generator": e.generator, "label": e.hallucinated,
            "severity": e.severity, "span_texts": list(e.span_texts),
            "jev_p": jev_p.score, "jev_verdict": jev_says,
            "claude_score": claude_p.score, "claude_verdict": claude_says,
            "claude_reasoning": claude_p.reasoning,
            "source": e.source, "response": e.response,
        })
    return cases


SOURCE_TRUNCATE_CHARS = 4_000
TRUNCATION_SUFFIX = "… [truncated]"


def export(
    examples: list[Example],
    store: PredictionStore,
    summary: dict[str, Any],
    frozen_threshold: float,
    out_dir: Path,
    audit_path: Path | None = None,
    max_bytes: int = 8_000_000,
) -> Path:
    audit = json.loads(audit_path.read_text()) if audit_path and audit_path.exists() else []
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    cases = _cases(examples, store, frozen_threshold)
    payload: dict[str, Any] = {
        "n_test": summary.get("n_test"),
        "headline": _headline_table(summary),
        "hypotheses": summary.get("hypotheses"),
        "sensitivity": summary.get("sensitivity"),
        "qa_replication_pilot_rules": summary.get("qa_replication_pilot_rules"),
        "pilot_reference_qa": summary.get("pilot_reference_qa"),
        "cases": cases,
        "audit": audit,
        "sources_truncated": False,
    }
    encoded = json.dumps(payload).encode()
    if len(encoded) > max_bytes:
        # Spans are highlighted in `response`, so only `source` (not `response`) is truncated.
        for case in cases:
            source = case["source"]
            if len(source) > SOURCE_TRUNCATE_CHARS:
                case["source"] = source[:SOURCE_TRUNCATE_CHARS] + TRUNCATION_SUFFIX
        payload["sources_truncated"] = True
        encoded = json.dumps(payload).encode()
    path.write_bytes(encoded)
    return path
