"""build_summary on a small synthetic dataset with known answers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragverdict.bench.predict import Prediction, PredictionStore
from ragverdict.bench.ragtruth import Example
from ragverdict.bench.runs import FrozenConfig
from ragverdict.bench.summary import build_summary

FROZEN = FrozenConfig(
    jev_model="typesafe/jev-1.13", jev_paraphrase="A", jev_threshold=0.5,
    claude_model="claude-sonnet-5", claude_rule="score<1.0", cascade_band=(0.3, 0.7),
    cascade_band_sweep=[(0.4, 0.6), (0.3, 0.7)], dataset_commit="abc",
    bootstrap_resamples=200, bootstrap_seed=0,
)


def _ex(i: int, hallucinated: bool, task: str = "QA") -> Example:
    return Example(id=str(i), split="test", task=task, generator="gpt-4-0613" if i % 2 else "llama",
                   source="s" * (10 * i), response="r", hallucinated=hallucinated,
                   span_types=("Evident Conflict",) if hallucinated else (),
                   span_texts=("in 2022",) if hallucinated else (), numeric=hallucinated)


def _examples() -> list[Example]:
    labels = [True, False, True, False, True, False, True, False]
    tasks = ["QA", "QA", "Summary", "Summary", "Data2txt", "Data2txt", "QA", "Summary"]
    return [_ex(i + 1, y, t) for i, (y, t) in enumerate(zip(labels, tasks, strict=True))]


def _store(tmp_path: Path, jev: dict[str, float | None], claude: dict[str, float | None]) -> PredictionStore:
    store = PredictionStore(tmp_path)
    for eid, p in jev.items():
        store.append(Prediction(run="jev", example_id=eid, repeat=0, score=p, cost_usd=0.00005,
                                latency_s=0.4, error=None if p is not None else "boom"))
    for eid, s in claude.items():
        store.append(Prediction(run="claude", example_id=eid, repeat=0, score=s, cost_usd=0.002,
                                error=None if s is not None else "boom"))
    return store


def test_perfect_jev_and_claude(tmp_path: Path) -> None:
    exs = _examples()
    jev = {e.id: (0.1 if e.hallucinated else 0.9) for e in exs}
    claude = {e.id: (0.5 if e.hallucinated else 1.0) for e in exs}
    summary = build_summary({"test": exs}, _store(tmp_path, jev, claude), FROZEN, "sha")
    j, c = summary["judges"]["jev"], summary["judges"]["claude"]
    assert j["n"] == 8 and j["n_failed"] == 0 and j["n_missing"] == 0
    assert j["total_spend_usd"] == pytest.approx(8 * 0.00005)
    assert j["auroc"]["value"] == pytest.approx(1.0)
    assert j["f1"]["value"] == pytest.approx(1.0)
    assert c["f1"]["value"] == pytest.approx(1.0)
    assert j["cost_usd_total"] == pytest.approx(8 * 0.00005)
    assert set(j["by_task"]) == {"QA", "Summary", "Data2txt"}
    assert j["by_task"]["QA"]["auroc_ci95"] is not None
    assert "recall_ci95" in j["recall_by_severity"]["evident"]
    # untuned Jev is run jev-para-A @ 0.5 (spec §6.6 item 1), never run "jev" at 0.5
    assert "jev_untuned" not in summary["judges"]
    assert c["frac_score_eq_1"] == pytest.approx(0.5)
    assert j["cost_usd_per_1m"] == pytest.approx(50.0)
    assert j["latency_s"]["p50"] == pytest.approx(0.4)  # all rows share latency_s=0.4
    h2h = summary["head_to_head"]
    assert h2h["claude_operating_point"] == {"precision": 1.0, "recall": 1.0}
    assert h2h["jev_recall_at_claude_precision"] == 1.0
    assert summary["wall_clock_s"] == {}
    assert summary["frozen_config_sha256"] == "sha"


def test_head_to_head_uses_intersection(tmp_path: Path) -> None:
    exs = _examples()
    jev: dict[str, float | None] = {e.id: (0.1 if e.hallucinated else 0.9) for e in exs}
    jev["1"] = None  # Jev errored on example 1
    claude = {e.id: (0.5 if e.hallucinated else 1.0) for e in exs}
    summary = build_summary({"test": exs}, _store(tmp_path, jev, claude), FROZEN, "sha")
    assert summary["judges"]["jev"]["n"] == 7
    assert summary["judges"]["jev"]["n_failed"] == 1 and summary["judges"]["jev"]["n_missing"] == 0
    assert summary["judges"]["claude"]["n"] == 8
    assert summary["head_to_head"]["n"] == 7
    assert summary["cascade"]["n"] == 7


def test_operating_points_use_jev_intersection(tmp_path: Path) -> None:
    """head_to_head.operating_points must reflect test ∩ jev ∩ that judge, not the judge's
    own (larger) block — a Jev error changes the cohort deepseek is scored against."""
    exs = _examples()
    jev: dict[str, float | None] = {e.id: (0.1 if e.hallucinated else 0.9) for e in exs}
    jev["1"] = None  # Jev errored on example "1" (hallucinated=True)
    claude = {e.id: (0.5 if e.hallucinated else 1.0) for e in exs}
    store = _store(tmp_path, jev, claude)
    for e in exs:  # deepseek flags everything
        store.append(Prediction(run="deepseek", example_id=e.id, repeat=0, score=0.5, cost_usd=0.0002))
    summary = build_summary({"test": exs}, store, FROZEN, "sha")

    ds_block = summary["judges"]["deepseek"]
    assert ds_block["classification"]["precision"] == pytest.approx(0.5)  # full 8-example block

    op = summary["head_to_head"]["operating_points"]["deepseek"]
    assert op["precision"] == pytest.approx(3 / 7)  # test ∩ jev ∩ deepseek excludes id "1"
    assert op["recall"] == pytest.approx(1.0)


def test_cascade_escalation_and_agreement(tmp_path: Path) -> None:
    exs = _examples()
    jev = {e.id: 0.5 for e in exs}  # always unsure -> always escalated
    claude = {e.id: (0.5 if e.hallucinated else 1.0) for e in exs}
    summary = build_summary({"test": exs}, _store(tmp_path, jev, claude), FROZEN, "sha")
    point = summary["cascade"]["frozen_band"]
    assert point["escalation_rate"] == pytest.approx(1.0)
    assert point["f1"] == pytest.approx(1.0)
    assert [p["band"] for p in summary["cascade"]["sweep"]] == [[0.4, 0.6], [0.3, 0.7]]
    assert "both_agree" in summary["agreement"]


def test_wall_clock_read_from_meta(tmp_path: Path) -> None:
    exs = _examples()
    store = _store(tmp_path, {e.id: 0.9 for e in exs}, {e.id: 1.0 for e in exs})
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "jev.json").write_text(json.dumps({
        "run": "jev",
        "invocations": [
            {"started_at": "2026-09-22T00:00:00+00:00", "wall_clock_s": 12.5,
             "n_computed": 8, "n_cached": 0},
        ],
    }))
    summary = build_summary({"test": exs}, store, FROZEN, "sha")
    assert summary["wall_clock_s"] == {"jev": {"seconds": 12.5, "n_computed": 8}}


def test_wall_clock_ignores_noop_rerun(tmp_path: Path) -> None:
    """A rerun that hits the cache entirely (n_computed == 0) must not move the published
    wall-clock number — only invocations that actually computed something count."""
    exs = _examples()
    store = _store(tmp_path, {e.id: 0.9 for e in exs}, {e.id: 1.0 for e in exs})
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "jev.json").write_text(json.dumps({
        "run": "jev",
        "invocations": [
            {"started_at": "2026-09-22T00:00:00+00:00", "wall_clock_s": 12.5,
             "n_computed": 8, "n_cached": 0},
            {"started_at": "2026-09-22T01:00:00+00:00", "wall_clock_s": 0.02,
             "n_computed": 0, "n_cached": 8},
        ],
    }))
    summary = build_summary({"test": exs}, store, FROZEN, "sha")
    assert summary["wall_clock_s"] == {"jev": {"seconds": 12.5, "n_computed": 8}}


def test_cheap_llm_judges_paired_and_qa_rows(tmp_path: Path) -> None:
    exs = _examples()
    jev = {e.id: (0.1 if e.hallucinated else 0.9) for e in exs}
    claude = {e.id: (0.5 if e.hallucinated else 1.0) for e in exs}
    store = _store(tmp_path, jev, claude)
    for e in exs:  # deepseek: flags everything
        store.append(Prediction(run="deepseek", example_id=e.id, repeat=0, score=0.5, cost_usd=0.0002))
    summary = build_summary({"test": exs}, store, FROZEN, "sha")
    ds = summary["judges"]["deepseek"]
    assert ds["false_alarms"] == 4 and ds["missed"] == 0
    assert ds["macro_f1"]["value"] == pytest.approx(1 / 3)  # pos F1 2/3, clean F1 0
    diff = summary["paired_vs_jev"]["deepseek"]["macro_f1_diff"]
    assert diff[0] == pytest.approx(1 - 1 / 3)
    assert summary["qa_replication"]["jev"]["missed"] == 0
    assert summary["baselines"]["always_clean"]["recall"] == 0.0
    assert set(summary["head_to_head"]["operating_points"]) == {"claude", "deepseek"}
    assert summary["pilot_reference_qa"]["jev"]["macro_f1"] == 0.6667
    assert summary["models"]["deepseek"]["requested"] == "deepseek/deepseek-v4.1-flash"


def test_missing_optional_runs_are_absent(tmp_path: Path) -> None:
    exs = _examples()
    jev = {e.id: (0.1 if e.hallucinated else 0.9) for e in exs}
    claude = {e.id: (0.5 if e.hallucinated else 1.0) for e in exs}
    summary = build_summary({"test": exs}, _store(tmp_path, jev, claude), FROZEN, "sha")
    assert summary["paraphrases"] == {}
    assert summary["flip"] == {}
    assert summary["thinking"] is None
    assert "deepseek" not in summary["judges"] and summary["paired_vs_jev"].keys() == {"claude"}


def test_non_good_rows_excluded_from_primary_but_in_pilot_rules(tmp_path: Path) -> None:
    import dataclasses

    exs = _examples()
    refusal = dataclasses.replace(_ex(99, False, "QA"), quality="incorrect_refusal",
                                  hallucinated_any_span=True)
    jev = {e.id: (0.1 if e.hallucinated else 0.9) for e in exs} | {"99": 0.2}
    claude = {e.id: (0.5 if e.hallucinated else 1.0) for e in exs} | {"99": 0.5}
    summary = build_summary({"test": [*exs, refusal]}, _store(tmp_path, jev, claude), FROZEN, "sha")
    assert summary["n_test"] == 8 and summary["judges"]["jev"]["n"] == 8
    pilot = summary["qa_replication_pilot_rules"]
    assert pilot["n"] == 4  # 3 QA "good" examples in _examples() + the refusal
    assert pilot["jev"]["n_scored"] == 4
    assert pilot["jev"]["n_failed"] == 0 and pilot["jev"]["n_missing"] == 0
    assert pilot["jev"]["paraphrase"] == FROZEN.jev_paraphrase
    assert pilot["jev"]["threshold"] == FROZEN.jev_threshold


def test_qa_pilot_rules_includes_pilot_setting_row(tmp_path: Path) -> None:
    exs = _examples()
    jev = {e.id: (0.1 if e.hallucinated else 0.9) for e in exs}
    claude = {e.id: (0.5 if e.hallucinated else 1.0) for e in exs}
    store = _store(tmp_path, jev, claude)
    for e in exs:  # a different paraphrase/threshold than the frozen jev row
        store.append(Prediction(run="jev-para-P", example_id=e.id, repeat=0,
                                score=(0.1 if e.hallucinated else 0.9), cost_usd=0.00005))
    summary = build_summary({"test": exs}, store, FROZEN, "sha")
    pilot = summary["qa_replication_pilot_rules"]
    assert pilot["jev_pilot_setting"]["n_scored"] == 3  # 3 QA examples in _examples()
    assert pilot["jev_pilot_setting"]["missed"] == 0


def test_failure_and_missing_counts_and_total_spend(tmp_path: Path) -> None:
    exs = _examples()
    store = PredictionStore(tmp_path)
    for e in exs:
        if e.id in {"7", "8"}:
            continue  # id "8": no row at all -> n_missing
        store.append(Prediction(run="jev", example_id=e.id, repeat=0,
                                score=(0.1 if e.hallucinated else 0.9), cost_usd=0.00005))
    store.append(Prediction(run="jev", example_id="7", repeat=0, score=None, cost_usd=0.00002,
                            error="network blip", error_kind="transport"))
    claude = {e.id: (0.5 if e.hallucinated else 1.0) for e in exs}
    for eid, s in claude.items():
        store.append(Prediction(run="claude", example_id=eid, repeat=0, score=s, cost_usd=0.002))

    summary = build_summary({"test": exs}, store, FROZEN, "sha")
    j = summary["judges"]["jev"]
    assert j["n"] == 6
    assert j["n_failed"] == 1
    assert j["n_failed_by_kind"] == {"transport": 1, "judge": 0, "unknown": 0}
    assert j["n_missing"] == 1
    assert j["total_spend_usd"] == pytest.approx(6 * 0.00005 + 0.00002)


def test_n_failed_by_kind_buckets_legacy_rows_as_unknown(tmp_path: Path) -> None:
    """A row with an error but no error_kind (written before error_kind existed) must be
    counted somewhere, so the buckets always sum to n_failed."""
    exs = _examples()
    store = PredictionStore(tmp_path)
    for e in exs:
        if e.id == "1":
            store.append(Prediction(run="jev", example_id=e.id, repeat=0, score=None,
                                    cost_usd=0.00001, error="pre-error_kind failure"))
            continue
        store.append(Prediction(run="jev", example_id=e.id, repeat=0,
                                score=(0.1 if e.hallucinated else 0.9), cost_usd=0.00005))
    claude = {e.id: (0.5 if e.hallucinated else 1.0) for e in exs}
    for eid, s in claude.items():
        store.append(Prediction(run="claude", example_id=eid, repeat=0, score=s, cost_usd=0.002))

    summary = build_summary({"test": exs}, store, FROZEN, "sha")
    j = summary["judges"]["jev"]
    assert j["n_failed"] == 1
    assert j["n_failed_by_kind"] == {"transport": 0, "judge": 0, "unknown": 1}
    assert sum(j["n_failed_by_kind"].values()) == j["n_failed"]


def test_partial_progress_only_jev_present(tmp_path: Path) -> None:
    """build_summary must not raise when only one judge has any predictions yet."""
    exs = _examples()
    store = PredictionStore(tmp_path)
    for e in exs:
        store.append(Prediction(run="jev", example_id=e.id, repeat=0,
                                score=(0.1 if e.hallucinated else 0.9), cost_usd=0.00005))
    summary = build_summary({"test": exs}, store, FROZEN, "sha")
    assert set(summary["judges"]) == {"jev"}
    assert summary["headline"] is None
    assert summary["head_to_head"] is None
    assert summary["cascade"] is None
    assert summary["agreement"] is None
    assert summary["paired_vs_jev"] == {}
    assert summary["models"].keys() == {"jev"}


def test_paraphrase_and_thinking_have_bootstrap_cis(tmp_path: Path) -> None:
    exs = _examples()
    jev = {e.id: (0.1 if e.hallucinated else 0.9) for e in exs}
    claude = {e.id: (0.5 if e.hallucinated else 1.0) for e in exs}
    store = _store(tmp_path, jev, claude)
    for e in exs:
        store.append(Prediction(run="jev-para-A", example_id=e.id, repeat=0,
                                score=(0.1 if e.hallucinated else 0.9), cost_usd=0.00005))
        store.append(Prediction(run="claude-thinking", example_id=e.id, repeat=0,
                                score=claude[e.id], cost_usd=0.008))
    summary = build_summary({"test": exs}, store, FROZEN, "sha")

    assert summary["judges"]["jev_untuned"]["run"] == "jev-para-A"
    assert summary["judges"]["jev_untuned"]["n"] == 8

    para = summary["paraphrases"]["A"]
    assert para["auroc"]["value"] == pytest.approx(1.0)
    assert len(para["auroc"]["ci95"]) == 2

    think = summary["thinking"]
    assert think is not None
    assert think["auroc_thinking"]["value"] == pytest.approx(1.0)
    assert len(think["auroc_no_thinking"]["ci95"]) == 2
    diff, lo, hi = think["auroc_diff_thinking_minus_no_thinking"]
    assert diff == pytest.approx(0.0)
    assert lo <= diff + 1e-9 and diff - 1e-9 <= hi
