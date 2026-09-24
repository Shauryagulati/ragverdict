"""Charts render and the results-page export has the right cases.

`SUMMARY` is a real `build_summary()` output on a small synthetic dataset (not a hand-written
fixture) so its shape always matches the actual summary.py — headline/hypotheses/sensitivity
included — rather than drifting from it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from ragverdict.bench.charts import render_all
from ragverdict.bench.page import export
from ragverdict.bench.predict import Prediction, PredictionStore
from ragverdict.bench.ragtruth import Example
from ragverdict.bench.runs import FrozenConfig
from ragverdict.bench.summary import HEADLINE_JEV_RUN, build_summary

FROZEN = FrozenConfig(
    jev_model="typesafe/jev-1.13", jev_paraphrase="A", jev_threshold=0.6,
    claude_model="claude-sonnet-5", claude_rule="score<1.0 or supported<total", cascade_band=(0.3, 0.7),
    cascade_band_sweep=[(0.4, 0.6), (0.3, 0.7)], dataset_commit="abc",
    bootstrap_resamples=50, bootstrap_seed=0, registered=True,
)
POSITIVES = {"1", "3", "5", "7", "9", "11"}  # ids 1..12 alternate hallucinated / clean
TASKS = ["QA", "QA", "QA", "QA", "Summary", "Summary", "Summary", "Summary",
        "Data2txt", "Data2txt", "Data2txt", "Data2txt"]


def _ex(i: int) -> Example:
    hallucinated = str(i) in POSITIVES
    return Example(
        id=str(i), split="test", task=TASKS[i - 1], generator="gpt-4-0613" if i % 2 else "llama",
        source=f"source text {i} " * 5, response=f"response text {i}",
        hallucinated=hallucinated, span_types=("Evident Conflict",) if hallucinated else (),
        span_texts=(f"span {i}",) if hallucinated else (), numeric=hallucinated,
        hallucinated_any_span=hallucinated,
    )


def _examples() -> list[Example]:
    return [_ex(i) for i in range(1, 13)]


def _full_store(tmp_path: Path) -> PredictionStore:
    """A store with jev (tuned), jev-para-A (headline/untuned), claude, deepseek, glm all
    scoring every example — everything render_all/export need is present."""
    store = PredictionStore(tmp_path)
    for e in _examples():
        jev_p = 0.1 if e.hallucinated else 0.9
        store.append(Prediction(run="jev", example_id=e.id, repeat=0, score=jev_p,
                                cost_usd=0.00005, latency_s=0.3))
        store.append(Prediction(run=HEADLINE_JEV_RUN, example_id=e.id, repeat=0, score=jev_p,
                                cost_usd=0.00005, latency_s=0.3))
        claude_score = 0.5 if e.hallucinated else 1.0
        store.append(Prediction(run="claude", example_id=e.id, repeat=0, score=claude_score,
                                supported_claims=1 if e.hallucinated else 2, total_claims=2,
                                cost_usd=0.002, reasoning=f"reasoning for {e.id}"))
        for run in ("deepseek", "glm"):
            store.append(Prediction(run=run, example_id=e.id, repeat=0, score=claude_score,
                                    supported_claims=1 if e.hallucinated else 2, total_claims=2,
                                    cost_usd=0.0002))
    return store


def _full_summary(tmp_path: Path) -> dict:
    store = _full_store(tmp_path)
    return build_summary({"test": _examples()}, store, FROZEN, "sha")


def test_render_all_writes_pngs(tmp_path: Path) -> None:
    summary = _full_summary(tmp_path / "store")
    paths = render_all(summary, tmp_path / "charts")
    assert {p.name for p in paths} == {
        "f1_vs_cost.png", "reliability.png", "cascade_curve.png", "slices.png", "pr_curve.png",
    }
    assert all(p.stat().st_size > 1000 for p in paths)


def test_render_all_skips_missing_blocks_on_partial_run(tmp_path: Path) -> None:
    """No cascade/head_to_head (claude hasn't scored anything yet) — those two charts are
    skipped rather than raising; the other three still render from what's present."""
    store = PredictionStore(tmp_path / "store")
    for e in _examples():
        jev_p = 0.1 if e.hallucinated else 0.9
        store.append(Prediction(run=HEADLINE_JEV_RUN, example_id=e.id, repeat=0, score=jev_p,
                                cost_usd=0.00005))
    summary = build_summary({"test": _examples()}, store, FROZEN, "sha")
    assert summary["cascade"] is None and summary["head_to_head"] is None
    paths = render_all(summary, tmp_path / "charts")
    names = {p.name for p in paths}
    assert names == {"f1_vs_cost.png", "reliability.png"}


def test_export_keeps_disagreements_and_joint_errors(tmp_path: Path) -> None:
    exs = [_ex(i) for i in (1, 2, 3, 4)]  # 1, 3 hallucinated; 2, 4 clean
    summary = _full_summary(tmp_path / "summary_store")
    store = PredictionStore(tmp_path / "cases_store")
    # 1 (hallucinated): both right. 2 (clean): disagree. 3 (hallucinated): both wrong.
    # 4 (clean): both right.
    rows = {
        "1": (0.1, 0.5, 1, 2),   # jev says halluc, claude score<1 -> halluc: both right
        "2": (0.2, 1.0, 2, 2),   # jev says halluc, claude fully supported: disagree
        "3": (0.9, 1.0, 2, 2),   # jev says clean, claude says clean: both wrong (positive)
        "4": (0.9, 1.0, 2, 2),   # both say clean: both right
    }
    for eid, (jev, claude, sup, tot) in rows.items():
        store.append(Prediction(run=HEADLINE_JEV_RUN, example_id=eid, repeat=0, score=jev))
        store.append(Prediction(run="claude", example_id=eid, repeat=0, score=claude,
                                supported_claims=sup, total_claims=tot, reasoning="why"))
    path = export(exs, store, summary, 0.5, tmp_path / "docs")
    data = json.loads(path.read_text())
    assert sorted(c["id"] for c in data["cases"]) == ["2", "3"]
    case = next(c for c in data["cases"] if c["id"] == "2")
    assert case["jev_p"] == 0.2 and case["claude_score"] == 1.0 and case["claude_reasoning"] == "why"
    assert case["jev_verdict"] is True and case["claude_verdict"] is False
    assert case["label"] is False and case["task"] == "QA"
    assert data["audit"] == []


def test_export_uses_llm_hallucinated_rule_not_just_score(tmp_path: Path) -> None:
    """Controller update: Claude's verdict is `llm_hallucinated` (score < 1 OR an unsupported
    claim) — score alone (the old rule) would miss this case."""
    exs = [_ex(1)]  # hallucinated
    summary = _full_summary(tmp_path / "summary_store")
    store = PredictionStore(tmp_path / "cases_store")
    store.append(Prediction(run=HEADLINE_JEV_RUN, example_id="1", repeat=0, score=0.9))  # jev: clean
    # score == 1.0 (old rule says clean) but a claim is unsupported -> llm_hallucinated is True
    store.append(Prediction(run="claude", example_id="1", repeat=0, score=1.0,
                            supported_claims=1, total_claims=2, reasoning="unsupported claim"))
    data = json.loads(export(exs, store, summary, 0.5, tmp_path / "docs").read_text())
    assert [c["id"] for c in data["cases"]] == ["1"]  # jev says clean, claude (new rule) says halluc
    assert data["cases"][0]["claude_verdict"] is True


def test_export_reads_run_jev_para_a_not_tuned_jev(tmp_path: Path) -> None:
    """The headline/case comparison is untuned Jev (run "jev-para-A"), never the tuned run
    "jev" — predictions only under "jev" must not produce any cases."""
    exs = [_ex(2)]  # clean
    summary = _full_summary(tmp_path / "summary_store")
    store = PredictionStore(tmp_path / "cases_store")
    store.append(Prediction(run="jev", example_id="2", repeat=0, score=0.1))  # wrong run
    store.append(Prediction(run="claude", example_id="2", repeat=0, score=1.0,
                            supported_claims=2, total_claims=2))
    data = json.loads(export(exs, store, summary, 0.5, tmp_path / "docs").read_text())
    assert data["cases"] == []


def test_export_writes_headline_hypotheses_and_pilot_tables(tmp_path: Path) -> None:
    summary = _full_summary(tmp_path / "summary_store")
    store = _full_store(tmp_path / "cases_store")
    data = json.loads(export(_examples(), store, summary, 0.5, tmp_path / "docs").read_text())
    assert data["headline"]["is_headline"] is True
    assert set(data["headline"]["judges"]) >= {"jev_untuned", "claude", "deepseek", "glm"}
    for row in data["headline"]["judges"].values():
        assert "f1" in row and "auroc" in row and "cost_usd_per_1k" in row
    assert data["headline"]["cascade"] is not None
    assert data["hypotheses"]["H1"] is not None
    assert data["hypotheses"]["H9"] is not None
    assert "invalid_as_wrong" in data["sensitivity"]
    assert "jev" in data["qa_replication_pilot_rules"]
    assert "jev" in data["pilot_reference_qa"]
    assert data["n_test"] == summary["n_test"]


def test_export_reads_audit_file_when_present(tmp_path: Path) -> None:
    summary = _full_summary(tmp_path / "summary_store")
    store = PredictionStore(tmp_path / "cases_store")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps([{"id": "1", "verdict": "hallucinated"}]))
    data = json.loads(export([], store, summary, 0.5, tmp_path / "docs", audit_path).read_text())
    assert data["audit"] == [{"id": "1", "verdict": "hallucinated"}]


def test_export_missing_audit_file_is_empty_list(tmp_path: Path) -> None:
    summary = _full_summary(tmp_path / "summary_store")
    store = PredictionStore(tmp_path / "cases_store")
    missing = tmp_path / "no-such-audit.json"
    data = json.loads(export([], store, summary, 0.5, tmp_path / "docs", missing).read_text())
    assert data["audit"] == []


def test_headline_table_includes_macro_f1_value_and_ci(tmp_path: Path) -> None:
    """Each headline judge row carries macro_f1 (value + 95% CI) straight from
    summary["headline"]["judges"][key]["macro_f1"], alongside the existing f1/auroc fields."""
    summary = _full_summary(tmp_path / "summary_store")
    store = _full_store(tmp_path / "cases_store")
    data = json.loads(export(_examples(), store, summary, 0.5, tmp_path / "docs").read_text())
    assert data["headline"]["is_headline"] is True
    for key, row in data["headline"]["judges"].items():
        expected = summary["headline"]["judges"][key]["macro_f1"]
        assert row["macro_f1"] == expected
        assert "value" in row["macro_f1"] and "ci95" in row["macro_f1"]
        assert "f1" in row  # existing field kept


def test_headline_table_partial_run_includes_macro_f1(tmp_path: Path) -> None:
    """On a partial run (only run "jev" scored, no untuned "jev-para-A" yet), the headline
    cohort is empty so _headline_table falls back to each judge's own block in
    summary["judges"] — macro_f1 must still be present there too."""
    store = PredictionStore(tmp_path / "store")
    for e in _examples():
        jev_p = 0.1 if e.hallucinated else 0.9
        store.append(Prediction(run="jev", example_id=e.id, repeat=0, score=jev_p,
                                cost_usd=0.00005))
    summary = build_summary({"test": _examples()}, store, FROZEN, "sha")
    assert summary["headline"] is None
    data = json.loads(export(_examples(), store, summary, 0.5, tmp_path / "docs").read_text())
    assert data["headline"]["is_headline"] is False
    row = data["headline"]["judges"]["jev"]
    assert row["macro_f1"] == summary["judges"]["jev"]["macro_f1"]
    assert "value" in row["macro_f1"] and "ci95" in row["macro_f1"]


def test_export_truncates_sources_past_size_limit(tmp_path: Path) -> None:
    """A tiny max_bytes still forces the legacy size-guard truncation path (source cut to
    4,000 chars, sources_truncated recorded) — even though every case source is already a
    600-char excerpt by then, so this guard is now a no-op on source length specifically."""
    huge_source = "x" * 20_000
    exs = [
        replace(_ex(1), source=huge_source),  # hallucinated
        replace(_ex(2), source=huge_source),  # clean
    ]
    summary = _full_summary(tmp_path / "summary_store")
    store = PredictionStore(tmp_path / "cases_store")
    # id 1: jev says halluc, claude fully supported -> disagree. id 2: jev says halluc,
    # claude fully supported (clean) -> disagree too. Both land in cases.
    for eid, jev_score in (("1", 0.1), ("2", 0.1)):
        store.append(Prediction(run=HEADLINE_JEV_RUN, example_id=eid, repeat=0, score=jev_score))
        store.append(Prediction(run="claude", example_id=eid, repeat=0, score=1.0,
                                supported_claims=2, total_claims=2, reasoning="fine"))
    path = export(exs, store, summary, 0.5, tmp_path / "docs", max_bytes=2_000)
    data = json.loads(path.read_text())
    assert data["sources_truncated"] is True
    assert len(data["cases"]) == 2
    for case in data["cases"]:
        assert case["source"] == "x" * 600 + "… [excerpt; full source in RAGTruth @ c103204b]"
        assert case["response"] == f"response text {case['id']}"


def test_export_excerpts_case_sources(tmp_path: Path) -> None:
    """Every case source is shipped as a 600-char excerpt with the licensing note, never
    the full source, regardless of the size guard — and sources_excerpted is always true."""
    exs = [_ex(1), _ex(2)]  # 1: hallucinated, 2: clean
    summary = _full_summary(tmp_path / "summary_store")
    store = PredictionStore(tmp_path / "cases_store")
    for eid, jev_score in (("1", 0.1), ("2", 0.1)):
        store.append(Prediction(run=HEADLINE_JEV_RUN, example_id=eid, repeat=0, score=jev_score))
        store.append(Prediction(run="claude", example_id=eid, repeat=0, score=1.0,
                                supported_claims=2, total_claims=2, reasoning="fine"))
    data = json.loads(export(exs, store, summary, 0.5, tmp_path / "docs").read_text())
    assert data["sources_excerpted"] is True
    for case in data["cases"]:
        original_source = next(e.source for e in exs if e.id == case["id"])
        assert case["source"] == (
            original_source[:600] + "… [excerpt; full source in RAGTruth @ c103204b]"
        )


def test_export_no_truncation_flag_false_when_under_limit(tmp_path: Path) -> None:
    summary = _full_summary(tmp_path / "summary_store")
    store = _full_store(tmp_path / "cases_store")
    data = json.loads(export(_examples(), store, summary, 0.5, tmp_path / "docs").read_text())
    assert data["sources_truncated"] is False
    for case in data["cases"]:
        assert not case["source"].endswith("… [truncated]")
