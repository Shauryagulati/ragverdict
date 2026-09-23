"""Task 9c-2 red-team statistics in build_summary: verdict rule, headline/H1, threshold-free,
sensitivity, subtypes, calibration extremes, H9, replication rows. Expected values are
hand-computed in the comments next to each assertion."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragverdict.bench.metrics import wilson_ci
from ragverdict.bench.predict import Prediction, PredictionStore
from ragverdict.bench.ragtruth import Example
from ragverdict.bench.runs import FrozenConfig
from ragverdict.bench.summary import build_summary, h1_verdict, h9_verdict, llm_hallucinated

FROZEN = FrozenConfig(
    jev_model="typesafe/jev-1.13", jev_paraphrase="A", jev_threshold=0.5,
    claude_model="claude-sonnet-5", claude_rule="score<1.0", cascade_band=(0.3, 0.7),
    cascade_band_sweep=[(0.4, 0.6), (0.3, 0.7)], dataset_commit="abc",
    bootstrap_resamples=200, bootstrap_seed=0,
)
POSITIVES = {"1", "3", "5", "7"}  # ids 1..8 alternate hallucinated / clean


def _ex(i: int, *, span: str = "Evident Conflict", conv: bool = False) -> Example:
    hallucinated = str(i) in POSITIVES
    task = ["QA", "QA", "Summary", "Summary", "Data2txt", "Data2txt", "QA", "Summary"][i - 1]
    return Example(id=str(i), split="test", task=task, generator="gpt-4-0613" if i % 2 else "llama",
                   source="s" * (10 * i), response="r", hallucinated=hallucinated,
                   span_types=(span,) if hallucinated else (),
                   span_texts=("in 2022",) if hallucinated else (), numeric=hallucinated,
                   hallucinated_any_span=hallucinated, convention_dependent=conv)


def _examples() -> list[Example]:
    return [_ex(i) for i in range(1, 9)]


def _jev(store: PredictionStore, run: str, scores: dict[str, float | None],
         cost: float = 0.00005) -> None:
    for eid, p in scores.items():
        store.append(Prediction(run=run, example_id=eid, repeat=0, score=p, cost_usd=cost,
                                error=None if p is not None else "bad", error_kind=None if p is not None else "judge"))


def _llm(store: PredictionStore, run: str, rows: dict[str, tuple[float, int, int] | None],
         **kw: float) -> None:
    """rows: id -> (score, supported_claims, total_claims), or None for a judge failure."""
    for eid, row in rows.items():
        if row is None:
            store.append(Prediction(run=run, example_id=eid, repeat=0, score=None, error="bad",
                                    error_kind="judge"))
            continue
        s, sup, tot = row
        store.append(Prediction(run=run, example_id=eid, repeat=0, score=s, supported_claims=sup,
                                total_claims=tot, cost_usd=0.002, **kw))  # type: ignore[arg-type]


def _perfect_jev() -> dict[str, float | None]:
    return {str(i): (0.1 if str(i) in POSITIVES else 0.9) for i in range(1, 9)}


def _perfect_llm() -> dict[str, tuple[float, int, int] | None]:
    return {str(i): ((0.5, 1, 2) if str(i) in POSITIVES else (1.0, 2, 2)) for i in range(1, 9)}


def _flag_all_llm() -> dict[str, tuple[float, int, int] | None]:
    return {str(i): (0.5, 1, 2) for i in range(1, 9)}


# ---------- §1 verdict rule ----------

def test_llm_rule_counts_claims_even_when_score_is_one() -> None:
    def p(score: float, sup: int, tot: int) -> Prediction:
        return Prediction(run="claude", example_id="x", repeat=0, score=score,
                          supported_claims=sup, total_claims=tot)

    assert llm_hallucinated(p(1.0, 2, 3)) is True  # score says clean, claims say not
    assert llm_hallucinated(p(0.5, 3, 3)) is True  # score alone flags it
    assert llm_hallucinated(p(1.0, 3, 3)) is False
    assert llm_hallucinated(p(1.0, 0, 0)) is False  # pilot prompt: no claim counts -> score < 1
    assert llm_hallucinated(p(0.9, 0, 0)) is True


def test_llm_judge_block_uses_rule_and_reports_inconsistency(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    _jev(store, "jev", _perfect_jev())
    rows = _perfect_llm()
    rows["1"] = (1.0, 2, 3)  # positive; score 1.0 but a claim unsupported -> flagged (inconsistent)
    rows["8"] = (0.5, 3, 3)  # negative; score < 1 but every claim supported -> flagged (inconsistent)
    _llm(store, "claude", rows)
    c = build_summary({"test": _examples()}, store, FROZEN, "sha")["judges"]["claude"]
    # tp=4 fp=1 fn=0 -> precision 4/5, recall 1, F1 = 8/9 (the old score<1 rule would miss id 1)
    assert c["classification"]["tp"] == 4 and c["classification"]["fp"] == 1
    assert c["f1"]["value"] == pytest.approx(8 / 9)
    assert c["verdict_inconsistency_rate"] == pytest.approx(2 / 8)
    assert c["n_verdict_inconsistent"] == 2
    assert c["run"] == "claude"
    assert "verdict_inconsistency_rate" not in build_summary(
        {"test": _examples()}, store, FROZEN, "sha")["judges"]["jev"]


# ---------- §2 headline / H1 ----------

def test_h1_verdict_rule() -> None:
    assert h1_verdict(-0.02, 0.02, 0.03) == "equivalent"
    assert h1_verdict(-0.03, 0.03, 0.03) == "equivalent"  # closed interval
    assert h1_verdict(0.031, 0.2, 0.03) == "jev_better"
    assert h1_verdict(-0.2, -0.031, 0.03) == "jev_worse"
    assert h1_verdict(-0.02, 0.05, 0.03) == "inconclusive"
    assert h1_verdict(0.03, 0.2, 0.03) == "inconclusive"  # lower bound must exceed +delta


def test_h9_verdict_rule() -> None:
    assert h9_verdict(-0.01, 0.019, 0.02) == "confirmed"
    assert h9_verdict(0.02, 0.05, 0.02) == "refuted"
    assert h9_verdict(0.0, 0.02, 0.02) == "inconclusive"


def test_headline_is_untuned_on_common_intersection(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    para_a = _perfect_jev()
    para_a["1"] = None  # jev-para-A judge failure on id 1
    _jev(store, "jev-para-A", para_a)
    _jev(store, "jev", _perfect_jev())
    _llm(store, "claude", _perfect_llm())
    ds = _flag_all_llm()
    del ds["2"]  # deepseek never scored id 2
    _llm(store, "deepseek", ds)
    h = build_summary({"test": _examples()}, store, FROZEN, "sha")["headline"]
    assert h["cohort"]["n"] == 6  # ids 3..8
    assert h["cohort"]["n_positive"] == 3
    assert h["cohort"]["runs_intersected"] == ["jev-para-A", "claude", "deepseek"]
    j = h["judges"]
    assert j["jev_untuned"]["run"] == "jev-para-A" and j["jev_untuned"]["threshold"] == 0.5
    for key in ("precision", "recall", "f1", "macro_f1", "accuracy"):
        assert len(j["jev_untuned"][key]["ci95"]) == 2
    assert j["jev_untuned"]["f1"]["value"] == pytest.approx(1.0)
    assert j["claude"]["f1"]["value"] == pytest.approx(1.0)
    # deepseek flags all 6: precision 1/2, recall 1, F1 2/3, clean F1 0 -> macro 1/3, accuracy 1/2
    assert j["deepseek"]["precision"]["value"] == pytest.approx(0.5)
    assert j["deepseek"]["f1"]["value"] == pytest.approx(2 / 3)
    assert j["deepseek"]["macro_f1"]["value"] == pytest.approx(1 / 3)
    assert j["deepseek"]["accuracy"]["value"] == pytest.approx(0.5)
    tuned = j["jev_tuned"]
    assert tuned["label"] == "tuned_on_train" and tuned["run"] == "jev"
    assert tuned["threshold"] == FROZEN.jev_threshold and tuned["n"] == 6
    h1 = h["h1"]  # both perfect: diff 0 on every resample
    assert h1["diff"] == pytest.approx(0.0) and h1["ci90"] == [0.0, 0.0]
    assert h1["verdict"] == "equivalent" and h1["delta"] == 0.03


def test_h1_jev_better_when_claude_flags_everything(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    _jev(store, "jev-para-A", _perfect_jev())
    _llm(store, "claude", _flag_all_llm())
    s = build_summary({"test": _examples()}, store, FROZEN, "sha")
    h1 = s["headline"]["h1"]
    assert h1["diff"] == pytest.approx(1 - 2 / 3)  # Jev F1 1 minus flag-all F1 2/3
    assert h1["ci90"][0] > 0.03 and h1["verdict"] == "jev_better"
    assert s["hypotheses"]["H1"] == h1


def test_headline_absent_without_para_a(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    _jev(store, "jev", _perfect_jev())
    _llm(store, "claude", _perfect_llm())
    s = build_summary({"test": _examples()}, store, FROZEN, "sha")
    assert s["headline"] is None  # never falls back to the tuned run "jev"
    assert s["hypotheses"]["H1"] is None
    assert s["subtypes"] is None
    assert s["sensitivity"]["excluding_convention_dependent"] is None


def test_threshold_free_bootstraps_llm_operating_point(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    _jev(store, "jev-para-A", _perfect_jev())
    rows = _perfect_llm()
    rows["2"] = (0.5, 1, 2)  # claude: precision 4/5, recall 1
    _llm(store, "claude", rows)
    tf = build_summary({"test": _examples()}, store, FROZEN, "sha")["headline"]["threshold_free"]
    c = tf["claude"]
    assert c["llm_precision"] == pytest.approx(0.8) and c["llm_recall"] == pytest.approx(1.0)
    # Jev separates perfectly, so its curve reaches precision 1 at recall 1
    assert c["jev_recall_at_llm_precision"]["value"] == pytest.approx(1.0)
    assert c["jev_precision_at_llm_recall"]["value"] == pytest.approx(1.0)
    assert c["jev_recall_at_llm_precision"]["ci95"][1] == pytest.approx(1.0)
    assert c["jev_recall_at_llm_precision"]["n_resamples_undefined"] == 0


def test_cost_basis_standard_uncached(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    _jev(store, "jev-para-A", _perfect_jev(), cost=0.00005)
    _llm(store, "claude", _perfect_llm(), input_tokens=1000, cache_read_tokens=500,
         cache_write_tokens=100, output_tokens=200)
    cb = build_summary({"test": _examples()}, store, FROZEN, "sha")["headline"]["cost_basis"]
    # (1000 + 500 + 100) * $2/MTok + 200 * $10/MTok = $0.0052 per example
    assert cb["claude_usd_per_example_standard_uncached"] == pytest.approx(0.0052)
    assert cb["claude_usd_per_example_billed"] == pytest.approx(0.002)
    assert cb["jev_usd_per_example_billed"] == pytest.approx(0.00005)
    assert cb["claude_to_jev_cost_ratio_standard_uncached"] == pytest.approx(104.0)
    assert "notes" in cb


# ---------- §3 sensitivity ----------

def test_invalid_as_wrong_counts_failures_and_missing(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    _jev(store, "jev-para-A", _perfect_jev())
    rows: dict[str, tuple[float, int, int] | None] = dict(_perfect_llm())
    rows["1"] = None  # judge failure on a positive -> counted as "clean" (wrong)
    del rows["2"]  # missing on a negative -> counted as "hallucinated" (wrong)
    _llm(store, "claude", rows)
    inv = build_summary({"test": _examples()}, store, FROZEN, "sha")["sensitivity"]["invalid_as_wrong"]
    c = inv["claude"]
    assert c["n"] == 8 and c["n_invalid"] == 2
    assert c["n_failed"] == 1 and c["n_missing"] == 1
    # tp=3 fn=1 fp=1 tn=3: F1 0.75 and clean-class F1 0.75
    assert c["f1"]["value"] == pytest.approx(0.75)
    assert c["macro_f1"]["value"] == pytest.approx(0.75)
    assert inv["jev_untuned"]["n_invalid"] == 0 and inv["jev_untuned"]["f1"]["value"] == 1.0


def test_excluding_convention_dependent_drops_rows(tmp_path: Path) -> None:
    exs = [_ex(i, conv=(i == 8)) for i in range(1, 9)]
    store = PredictionStore(tmp_path)
    _jev(store, "jev-para-A", _perfect_jev())
    rows = _perfect_llm()
    rows["8"] = (0.5, 1, 2)  # claude's only error is on the convention-dependent row
    _llm(store, "claude", rows)
    s = build_summary({"test": exs}, store, FROZEN, "sha")
    assert s["headline"]["judges"]["claude"]["f1"]["value"] == pytest.approx(8 / 9)
    ex_cd = s["sensitivity"]["excluding_convention_dependent"]
    assert ex_cd["n"] == 7 and ex_cd["n_dropped"] == 1
    assert ex_cd["judges"]["claude"]["f1"]["value"] == pytest.approx(1.0)
    assert set(ex_cd["judges"]["claude"]) >= {"f1", "macro_f1"}
    assert ex_cd["h1"]["diff"] == pytest.approx(0.0)


# ---------- §4 subtypes ----------

def _subtype_store(tmp_path: Path) -> tuple[list[Example], PredictionStore]:
    # ids 1, 3: subtle baseless; 5, 7: evident conflict. Jev P(supported) by id:
    # sorted .1(1,T) .2(2,F) .3(3,T) .5(4,F) .6(5,T) .7(6,F) .8(7,T) .9(8,F)
    exs = [_ex(i, span="Subtle Baseless Info" if i in (1, 3) else "Evident Conflict")
           for i in range(1, 9)]
    store = PredictionStore(tmp_path)
    _jev(store, "jev-para-A", {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.5, "5": 0.6, "6": 0.7,
                               "7": 0.8, "8": 0.9})
    rows = {str(i): (1.0, 2, 2) for i in range(1, 9)}
    for eid in ("2", "5", "7"):  # claude flags 5, 7 and clean 2 -> FPR 1/4
        rows[eid] = (0.5, 1, 2)
    _llm(store, "claude", rows)  # type: ignore[arg-type]
    return exs, store


def test_matched_fpr_subtype_recall(tmp_path: Path) -> None:
    exs, store = _subtype_store(tmp_path)
    m = build_summary({"test": exs}, store, FROZEN, "sha")["subtypes"]["matched_fpr"]["claude"]
    assert m["llm_fpr"] == pytest.approx(0.25)
    # midpoints .15 .25 .4 .55 ...; FPR 0, 1/4, 1/4, 2/4 -> largest with FPR <= 1/4 is 0.4
    assert m["jev_threshold"] == pytest.approx(0.4)
    assert m["jev_fpr"] == pytest.approx(0.25)
    subtle = m["subtypes"]["severity"]["subtle"]  # Jev flags 1, 3 (p < .4); claude flags neither
    assert subtle["n_pos"] == 2 and subtle["descriptive"] is True
    assert subtle["jev_recall"] == pytest.approx(1.0) and subtle["llm_recall"] == pytest.approx(0.0)
    assert subtle["recall_diff_jev_minus_llm"]["diff"] == pytest.approx(1.0)
    assert len(subtle["recall_diff_jev_minus_llm"]["ci95"]) == 2
    evident = m["subtypes"]["severity"]["evident"]  # Jev flags neither of 5, 7; claude both
    assert evident["jev_recall"] == pytest.approx(0.0) and evident["llm_recall"] == pytest.approx(1.0)
    assert m["subtypes"]["kind"]["baseless"]["jev_recall"] == pytest.approx(1.0)
    assert m["subtypes"]["numeric"]["numeric"]["n_pos"] == 4
    gpt4 = m["subtypes"]["generator"]["gpt-4-0613"]  # all positives: Jev 2/4, claude 2/4
    assert gpt4["jev_recall"] == pytest.approx(0.5) and gpt4["llm_recall"] == pytest.approx(0.5)


def test_subtype_auroc_positives_vs_all_negatives(tmp_path: Path) -> None:
    exs, store = _subtype_store(tmp_path)
    sa = build_summary({"test": exs}, store, FROZEN, "sha")["subtypes"]["subtype_auroc"]
    subtle, evident = sa["severity"]["subtle"], sa["severity"]["evident"]
    # Jev subtle {.1,.3} vs negatives {.2,.5,.7,.9}: 4 + 3 of 8 pairs ranked right
    assert subtle["jev_untuned"]["value"] == pytest.approx(7 / 8)
    # Jev evident {.6,.8}: 2 + 1 of 8
    assert evident["jev_untuned"]["value"] == pytest.approx(3 / 8)
    # claude 1-score: subtle {0,0} vs negatives {.5,0,0,0}: 2 * (0 + 3 * 0.5) / 8
    assert subtle["claude"]["value"] == pytest.approx(3 / 8)
    # evident {.5,.5}: 2 * (0.5 + 3) / 8
    assert evident["claude"]["value"] == pytest.approx(7 / 8)
    assert subtle["n_pos"] == 2 and subtle["n_neg"] == 4
    assert subtle["diff_jev_minus"]["claude"]["diff"] == pytest.approx(7 / 8 - 3 / 8)


def test_length_quartiles_are_within_task(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    _jev(store, "jev", _perfect_jev())
    q = build_summary({"test": _examples()}, store, FROZEN, "sha")["judges"]["jev"]["by_length_quartile"]
    assert set(q) == {"QA", "Summary", "Data2txt"}
    assert sum(s["n"] for s in q["QA"].values()) == 3  # ids 1, 2, 7
    assert sum(s["n"] for s in q["Summary"].values()) == 3  # ids 3, 4, 8
    assert sum(s["n"] for s in q["Data2txt"].values()) == 2  # ids 5, 6


# ---------- §5 calibration / flip / agreement / H9 ----------

def test_calibration_extremes_on_untuned_jev(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    # P(hallucinated) = 1 - p. high bin (> .9): ids 1 (.95), 3 (.98), both hallucinated.
    # low bin (< .1): 5 (.05, hallucinated), 2 (.03), 4 (.01) -> 1 of 3 hallucinated.
    _jev(store, "jev-para-A", {"1": 0.05, "3": 0.02, "5": 0.95, "2": 0.97, "4": 0.99,
                               "6": 0.5, "7": 0.5, "8": 0.5})
    cal = build_summary({"test": _examples()}, store, FROZEN, "sha")["calibration_extremes"]
    assert cal["run"] == "jev-para-A"
    low, high = cal["low"], cal["high"]
    assert low["n"] == 3 and low["mean_predicted"] == pytest.approx(0.03)
    assert low["observed_rate"] == pytest.approx(1 / 3)
    assert low["wilson_ci95"] == pytest.approx(list(wilson_ci(1, 3)))
    assert len(low["bootstrap_ci95"]) == 2
    assert high["n"] == 2 and high["mean_predicted"] == pytest.approx(0.965)
    assert high["observed_rate"] == 1.0 and high["bootstrap_ci95"] == [1.0, 1.0]


def test_flip_rate_has_wilson_ci(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    _jev(store, "jev", _perfect_jev())
    for rep, row in enumerate([(0.5, 1, 2), (1.0, 1, 2), (1.0, 2, 2)]):  # flagged, flagged, clean
        store.append(Prediction(run="claude-flip", example_id="1", repeat=rep, score=row[0],
                                supported_claims=row[1], total_claims=row[2]))
    for rep in range(3):
        store.append(Prediction(run="claude-flip", example_id="2", repeat=rep, score=1.0,
                                supported_claims=2, total_claims=2))
    flip = build_summary({"test": _examples()}, store, FROZEN, "sha")["flip"]["claude-flip"]
    assert flip["n"] == 2 and flip["n_flipped"] == 1 and flip["flip_rate"] == 0.5
    assert flip["flip_ci95"] == pytest.approx(list(wilson_ci(1, 2)))


def test_agreement_has_wilson_cis(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    _jev(store, "jev", _perfect_jev())
    _llm(store, "claude", _perfect_llm())
    agree = build_summary({"test": _examples()}, store, FROZEN, "sha")["agreement"]["both_agree"]
    assert agree["n"] == 8
    assert agree["precision_ci95"] == pytest.approx(list(wilson_ci(4, 4)))
    assert agree["accuracy_ci95"] == pytest.approx(list(wilson_ci(8, 8)))


def test_h9_cascade_vs_best_single_judge(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    _jev(store, "jev", {str(i): 0.5 for i in range(1, 9)})  # all in band -> all escalated
    _jev(store, "jev-para-A", {str(i): 0.1 for i in range(1, 9)})  # flags all: F1 2/3
    _llm(store, "claude", _perfect_llm())
    s = build_summary({"test": _examples()}, store, FROZEN, "sha")
    h9 = s["hypotheses"]["H9"]
    # tuned Jev at 0.5 flags nothing (F1 0); untuned 2/3; claude 1 -> best is claude.
    assert h9["single_judge_f1"] == pytest.approx({"jev_tuned": 0.0, "jev_untuned": 2 / 3,
                                                  "claude": 1.0})
    assert h9["best_single_judge"] == "claude"
    assert h9["cascade_f1"] == pytest.approx(1.0) and h9["escalation_rate"] == 1.0
    assert h9["diff"] == pytest.approx(0.0) and h9["ci95"] == [0.0, 0.0]
    assert h9["verdict"] == "confirmed" and h9["margin"] == 0.02 and h9["n"] == 8
    assert s["hypotheses"]["exploratory"] == ["H2", "H3", "H4", "H5", "H6", "H7", "H8", "H10",
                                             "H11", "H12"]


# ---------- §7 replication rows ----------

def test_pilot_replication_rows(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    _jev(store, "jev", _perfect_jev())
    qa = {"1": 0.1, "2": 0.9, "7": 0.9}  # QA ids 1 (T), 2 (F), 7 (T): misses 7
    _jev(store, "jev-pilot-state", qa)  # type: ignore[arg-type]
    _llm(store, "deepseek-pilot", {"1": (0.0, 0, 0), "2": (0.0, 0, 0), "7": (1.0, 0, 0)})
    rows = build_summary({"test": _examples()}, store, FROZEN, "sha")["qa_replication_pilot_rules"]
    assert rows["jev_pilot_state"]["n_scored"] == 3 and rows["jev_pilot_state"]["missed"] == 1
    assert rows["jev_pilot_state"]["accuracy"] == pytest.approx(2 / 3)
    ds = rows["deepseek_pilot"]  # flags 1 and 2, passes 7: one miss, one false alarm
    assert ds["missed"] == 1 and ds["false_alarms"] == 1
    assert ds["n_failed"] == 0 and ds["n_missing"] == 0
    assert "glm_pilot" not in rows


def test_missing_optional_sections_do_not_crash(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path)
    _jev(store, "jev-para-A", _perfect_jev())
    s = build_summary({"test": _examples()}, store, FROZEN, "sha")
    assert s["headline"]["h1"] is None and s["headline"]["threshold_free"] == {}
    assert s["headline"]["cost_basis"] is None
    assert s["hypotheses"]["H9"] is None
    assert s["subtypes"]["matched_fpr"] == {}
    assert set(s["subtypes"]["subtype_auroc"]["severity"]["evident"]) >= {"jev_untuned", "n_pos"}
    assert s["sensitivity"]["invalid_as_wrong"].keys() == {"jev_untuned"}
