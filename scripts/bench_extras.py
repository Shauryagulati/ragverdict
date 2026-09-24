"""Supplementary benchmark numbers for the write-up, computed from the cached predictions.

Every supplementary figure quoted in the write-up (beyond summary.json) comes from here, so
each one is traceable to code. Reuses ragverdict.bench for loading, verdict rules and metric
primitives; this is NOT the independent verifier (that is scripts/verify_bench.py).

Conventions (same as summary.py): positive = hallucinated; every stored score is
P(supported) (inverted Jev wordings are already reoriented); repeat 0, successful rows only.
Verdict rules are the pre-registered ones:
  jev_untuned  run "jev-para-A", hallucinated iff score < 0.5
  jev_tuned    run "jev",        hallucinated iff score < frozen.jev_threshold
  LLM judges   runs "claude", "deepseek", "glm": summary.llm_hallucinated
The headline cohort is the primary (quality == "good") test cohort restricted to examples
scored by all five of those runs.

Usage: python scripts/bench_extras.py --out bench_results --frozen bench/frozen_config.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from ragverdict.bench import metrics
from ragverdict.bench.predict import CLAUDE_PRICES, Prediction, PredictionStore
from ragverdict.bench.ragtruth import Example, ensure_downloaded, load_examples
from ragverdict.bench.runs import RUNS, FrozenConfig, load_frozen, select_examples
from ragverdict.bench.summary import (
    HEADLINE_JEV_RUN,
    LLM_RUNS,
    UNTUNED_JEV_THRESHOLD,
    llm_hallucinated,
)

TUNED_JEV_RUN = "jev"
FLIP_RUN = "jev-flip"
FLIP_REPEATS = (0, 1, 2)
PARAPHRASE_RUNS = tuple(f"jev-para-{p}" for p in "ABCDEP")
NEAR_THRESHOLD_WINDOW = 0.03  # "near the tuned threshold" = within ±0.03 of it
BASE_RATES = (0.05, 0.10)
JUDGES = ("jev_untuned", "jev_tuned", *LLM_RUNS)


# ---------- pure helpers (unit-tested) ----------

def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default), q in [0, 1]."""
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def flag_thresholds(p_supported: Sequence[float]) -> list[tuple[float, int]]:
    """Every distinct operating point of 'hallucinated iff p < t' as (t, n_flagged), from
    flag-nothing (t = min score) up to flag-everything. Thresholds are midpoints between
    distinct scores (as in metrics), so tied scores are always flagged together."""
    values = sorted(set(p_supported))
    counts = {v: 0 for v in values}
    for p in p_supported:
        counts[p] += 1
    out = [(values[0], 0)]
    flagged = 0
    uppers = [(a + b) / 2 for a, b in pairwise(values)] + [values[-1] + 1e-9]
    for v, t in zip(values, uppers, strict=True):
        flagged += counts[v]
        out.append((t, flagged))
    return out


def oracle_f1(p_supported: Sequence[float], labels: Sequence[bool]) -> dict[str, float]:
    """Max F1 over every threshold of 'hallucinated iff p < t' (test-optimal: optimistic).
    Ties in F1 go to the lowest threshold."""
    n_pos = sum(labels)
    order = sorted(range(len(p_supported)), key=lambda i: p_supported[i])
    best = {"threshold": p_supported[order[0]], "f1": 0.0, "n_flagged": 0}
    tp = 0
    i = 0
    for t, n_flagged in flag_thresholds(p_supported)[1:]:
        while i < n_flagged:
            tp += labels[order[i]]
            i += 1
        f1 = 2 * tp / (n_flagged + n_pos) if n_flagged + n_pos else 0.0
        if f1 > best["f1"]:
            best = {"threshold": t, "f1": f1, "n_flagged": n_flagged}
    return best


def matched_flag_threshold(p_supported: Sequence[float], k: int) -> tuple[float, int]:
    """The threshold whose flag count is nearest `k` (ties go to the smaller count)."""
    return min(flag_thresholds(p_supported), key=lambda tn: (abs(tn[1] - k), tn[1]))


def jitter_stats(scores_per_example: Sequence[Sequence[float]]) -> dict[str, float]:
    """Spread (max - min) of each example's repeated scores, summarised."""
    ranges = [max(s) - min(s) for s in scores_per_example]
    return {
        "n": len(ranges),
        "n_changed": sum(r > 0 for r in ranges),
        "median": statistics.median(ranges),
        "p90": quantile(ranges, 0.9),
        "max": max(ranges),
    }


def base_rate_precision(tpr: float, fpr: float, prevalence: float) -> float | None:
    """Precision of a flag at `prevalence`, given the judge's TPR/FPR (Bayes' rule)."""
    flagged = tpr * prevalence + fpr * (1 - prevalence)
    return tpr * prevalence / flagged if flagged else None


def disagreement(a: Sequence[bool], b: Sequence[bool], labels: Sequence[bool]) -> dict[str, Any]:
    """Where judges a and b disagree: who is right, split by which one flags."""
    idx = [i for i in range(len(labels)) if a[i] != b[i]]
    b_only = [i for i in idx if b[i]]  # b flags, a passes
    a_only = [i for i in idx if a[i]]
    return {
        "n": len(idx),
        "a_right": sum(a[i] == labels[i] for i in idx),
        "b_right": sum(b[i] == labels[i] for i in idx),
        "b_flags_a_passes": {"n": len(b_only), "n_human_clean": sum(not labels[i] for i in b_only)},
        "a_flags_b_passes": {"n": len(a_only), "n_human_clean": sum(not labels[i] for i in a_only)},
    }


def agreement(a: Sequence[bool], b: Sequence[bool], labels: Sequence[bool]) -> dict[str, Any]:
    """Where judges a and b agree: accuracy, and precision of a joint flag / joint pass."""
    both_flag = [i for i in range(len(labels)) if a[i] and b[i]]
    both_pass = [i for i in range(len(labels)) if not a[i] and not b[i]]
    n_agree = len(both_flag) + len(both_pass)
    correct = sum(labels[i] for i in both_flag) + sum(not labels[i] for i in both_pass)
    return {
        "n_agree": n_agree,
        "agree_rate": n_agree / len(labels),
        "accuracy_when_agree": correct / n_agree if n_agree else None,
        "n_both_flag": len(both_flag),
        "precision_when_both_flag": (
            sum(labels[i] for i in both_flag) / len(both_flag) if both_flag else None
        ),
        "n_both_pass": len(both_pass),
        "clean_rate_when_both_pass": (
            sum(not labels[i] for i in both_pass) / len(both_pass) if both_pass else None
        ),
    }


def recall_by(
    keys: Sequence[str | None], verdicts: dict[str, list[bool]], labels: Sequence[bool]
) -> dict[str, dict[str, Any]]:
    """Recall of each judge within each key (positives only; key None = skipped)."""
    out: dict[str, dict[str, Any]] = {}
    for key in sorted({k for k, y in zip(keys, labels, strict=True) if y and k is not None}):
        pos = [i for i, (k, y) in enumerate(zip(keys, labels, strict=True)) if y and k == key]
        out[key] = {
            "n_pos": len(pos),
            **{j: sum(v[i] for i in pos) / len(pos) for j, v in verdicts.items()},
        }
    return out


def claude_standard_uncached_usd(p: Prediction, price_in: float, price_out: float) -> float:
    """Live list price with no caching: every input token (incl. cache reads/writes) is input."""
    tokens_in = p.input_tokens + p.cache_read_tokens + p.cache_write_tokens
    return (tokens_in * price_in + p.output_tokens * price_out) / 1_000_000


# ---------- loading ----------

def _ok(store: PredictionStore, run: str, repeat: int = 0) -> dict[str, Prediction]:
    return {
        eid: p for (eid, rep), p in store.load(run).items()
        if rep == repeat and p.score is not None
    }


def _s(p: Prediction) -> float:
    assert p.score is not None
    return float(p.score)


def _f1(verdicts: Sequence[bool], labels: Sequence[bool]) -> float:
    return metrics.classification(verdicts, labels)["f1"]


# ---------- the analysis ----------

def compute(
    test: list[Example], tune: list[Example], store: PredictionStore, frozen: FrozenConfig
) -> dict[str, Any]:
    para_a = _ok(store, HEADLINE_JEV_RUN)
    tuned = _ok(store, TUNED_JEV_RUN)
    llms = {run: _ok(store, run) for run in LLM_RUNS}
    runs = {"jev_untuned": para_a, "jev_tuned": tuned, **llms}
    rules: dict[str, Callable[[Prediction], bool]] = {
        "jev_untuned": lambda p: _s(p) < UNTUNED_JEV_THRESHOLD,
        "jev_tuned": lambda p: _s(p) < frozen.jev_threshold,
        **{run: llm_hallucinated for run in LLM_RUNS},
    }
    cohort = [e for e in test if all(e.id in preds for preds in runs.values())]
    labels = [e.hallucinated for e in cohort]
    n_pos = sum(labels)
    verdicts = {j: [rules[j](runs[j][e.id]) for e in cohort] for j in JUDGES}
    # P(supported)-oriented continuous scores (LLM score = supported-claim fraction)
    p_sup = {j: [_s(runs[j][e.id]) for e in cohort] for j in JUDGES}
    cls = {j: metrics.classification(verdicts[j], labels) for j in JUDGES}
    out: dict[str, Any] = {
        "frozen_config": frozen.model_dump(),
        "primary_cohort": {"n": len(test), "n_positive": sum(e.hallucinated for e in test)},
        "headline_cohort": {"n": len(cohort), "n_positive": n_pos,
                            "runs": [HEADLINE_JEV_RUN, TUNED_JEV_RUN, *LLM_RUNS]},
    }

    # 1
    d = disagreement(verdicts["jev_untuned"], verdicts["claude"], labels)
    out["disagreement_untuned_vs_claude"] = {
        "n": d["n"], "jev_right": d["a_right"], "claude_right": d["b_right"],
        "claude_flags_jev_passes": d["b_flags_a_passes"],
        "jev_flags_claude_passes": d["a_flags_b_passes"],
        "notes": "n_human_clean = RAGTruth label is clean; for claude_flags_jev_passes that "
                 "means Jev is right, for jev_flags_claude_passes Claude is right.",
    }
    # 2
    out["agreement_precision"] = {
        "jev_untuned_vs_claude": agreement(verdicts["jev_untuned"], verdicts["claude"], labels),
        "jev_tuned_vs_claude": agreement(verdicts["jev_tuned"], verdicts["claude"], labels),
    }
    # 3, 4
    out["generator_recall_at_operating_points"] = recall_by(
        [e.generator for e in cohort], verdicts, labels)
    out["severity_recall_at_operating_points"] = recall_by(
        [e.severity for e in cohort], verdicts, labels)
    # 5
    n_neg = len(cohort) - n_pos
    out["false_alarms_on_headline_cohort"] = {
        j: {"fp": cls[j]["fp"], "n_negative": n_neg, "fpr": cls[j]["fp"] / n_neg} for j in JUDGES
    }

    # 6
    def always_flag(exs: list[Example]) -> dict[str, Any]:
        def block(sub: list[Example]) -> dict[str, float]:
            ys = [e.hallucinated for e in sub]
            c = metrics.classification([True] * len(sub), ys)
            return {"n": len(sub), **{k: c[k] for k in ("precision", "recall", "f1", "macro_f1")}}
        tasks = sorted({e.task for e in exs})
        return {"overall": block(exs),
                "by_task": {t: block([e for e in exs if e.task == t]) for t in tasks}}

    out["always_flag_baseline"] = {"headline_cohort": always_flag(cohort),
                                   "primary_cohort": always_flag(test)}
    # 7
    out["oracle_best_f1"] = {
        "notes": "test-optimal threshold chosen on the very data it is scored on: optimistic "
                 "upper bound, not a deployable number. Swept rule: hallucinated iff "
                 "P(supported) < t (LLMs: supported-claim fraction < t; ties flagged together).",
        **{j: {**oracle_f1(p_sup[j], labels),
               "f1_at_own_rule": cls[j]["f1"]} for j in JUDGES},
    }
    # 8
    matched: dict[str, Any] = {}
    for run in LLM_RUNS:
        k = sum(verdicts[run])
        t, n_flagged = matched_flag_threshold(p_sup["jev_untuned"], k)
        matched[run] = {
            "llm_n_flagged": k, "llm_f1": cls[run]["f1"],
            "jev_untuned_threshold": t, "jev_untuned_n_flagged": n_flagged,
            "jev_untuned_f1": _f1([p < t for p in p_sup["jev_untuned"]], labels),
        }
    out["f1_at_matched_flag_rate"] = matched

    # 9
    flip = {r: _ok(store, FLIP_RUN, r) for r in FLIP_REPEATS}
    flip_ids = sorted(set.intersection(*(set(f) for f in flip.values())))
    per_example = [[_s(flip[r][eid]) for r in FLIP_REPEATS] for eid in flip_ids]
    tuned_primary = [_s(tuned[e.id]) for e in test if e.id in tuned]
    n_near = sum(abs(s - frozen.jev_threshold) <= NEAR_THRESHOLD_WINDOW for s in tuned_primary)
    out["jev_score_jitter"] = {
        "run": FLIP_RUN, "repeats": list(FLIP_REPEATS),
        **jitter_stats(per_example),
        "n_verdict_flips_at_tuned_threshold": sum(
            len({s < frozen.jev_threshold for s in scores}) > 1 for scores in per_example),
        "near_tuned_threshold": {
            "run": TUNED_JEV_RUN, "window": NEAR_THRESHOLD_WINDOW,
            "n_scored": len(tuned_primary), "n_within": n_near,
            "fraction": n_near / len(tuned_primary),
        },
    }

    # 10
    cas_v, escalated = metrics.cascade(p_sup["jev_tuned"], verdicts["claude"],
                                       threshold=frozen.jev_threshold, band=frozen.cascade_band)
    cas = metrics.classification(cas_v, labels)
    band = [i for i, esc in enumerate(escalated) if esc]
    price_in, price_out = CLAUDE_PRICES[frozen.claude_model]
    claude_cost = [claude_standard_uncached_usd(llms["claude"][e.id], price_in, price_out)
                   for e in cohort]
    jev_cost = [tuned[e.id].cost_usd for e in cohort]
    jev_1k = 1000 * sum(jev_cost) / len(cohort)
    claude_esc_1k = 1000 * sum(claude_cost[i] for i in band) / len(cohort)
    out["cascade_detail"] = {
        "band": list(frozen.cascade_band), "threshold": frozen.jev_threshold,
        "cascade": {"recall": cas["recall"], "fp": cas["fp"], "fpr": cas["fp"] / n_neg,
                    "f1": cas["f1"]},
        "jev_tuned": {"recall": cls["jev_tuned"]["recall"], "fp": cls["jev_tuned"]["fp"],
                      "fpr": cls["jev_tuned"]["fp"] / n_neg, "f1": cls["jev_tuned"]["f1"]},
        "within_band": {
            "n": len(band), "escalation_rate": len(band) / len(cohort),
            "claude_flag_rate": sum(verdicts["claude"][i] for i in band) / len(band),
            "claude_accuracy": sum(verdicts["claude"][i] == labels[i] for i in band) / len(band),
            "jev_tuned_accuracy": sum(verdicts["jev_tuned"][i] == labels[i] for i in band)
            / len(band),
        },
        "cost_usd_per_1k": {
            "jev_billed": jev_1k,
            "claude_escalations_standard_uncached": claude_esc_1k,
            "cascade_total": jev_1k + claude_esc_1k,
            "claude_alone_standard_uncached": 1000 * sum(claude_cost) / len(cohort),
            "notes": f"Claude rows' tokens at ${price_in}/${price_out} per MTok, every input "
                     "token (incl. cache reads/writes) at the input price; Jev = run 'jev' "
                     "OpenRouter-billed cost on every example.",
        },
    }

    # 11
    def ece_block(preds: dict[str, Prediction]) -> dict[str, Any]:
        exs = [e for e in test if e.id in preds]
        return {"n": len(exs), "ece_10_bins": metrics.ece(
            [1 - _s(preds[e.id]) for e in exs], [e.hallucinated for e in exs], 10)}

    out["ece"] = {"notes": "P(hallucinated) = 1 - P(supported), primary cohort",
                  HEADLINE_JEV_RUN: ece_block(para_a), TUNED_JEV_RUN: ece_block(tuned)}
    # 12
    out["tune_set_prevalence"] = {
        "tune": {"n": len(tune), "n_positive": sum(e.hallucinated for e in tune),
                 "prevalence": sum(e.hallucinated for e in tune) / len(tune)},
        "primary_test": {"n": len(test), "n_positive": sum(e.hallucinated for e in test),
                         "prevalence": sum(e.hallucinated for e in test) / len(test)},
    }
    # 13
    by_task: dict[str, Any] = {}
    for task in sorted({e.task for e in cohort}):
        idx = [i for i, e in enumerate(cohort) if e.task == task]
        ys = [labels[i] for i in idx]
        by_task[task] = {
            "n": len(idx), "n_pos": sum(ys), "share_of_positives": sum(ys) / n_pos,
            "f1": {j: _f1([verdicts[j][i] for i in idx], ys) for j in JUDGES},
        }
    out["pooled_by_task_contribution"] = {
        "pooled_f1": {j: cls[j]["f1"] for j in JUDGES}, "by_task": by_task}
    # 14
    out["production_base_rate_precision"] = {
        j: {"tpr": cls[j]["recall"], "fpr": cls[j]["fp"] / n_neg,
            **{f"precision_at_{round(pi * 100)}pct": base_rate_precision(
                cls[j]["recall"], cls[j]["fp"] / n_neg, pi) for pi in BASE_RATES}}
        for j in JUDGES
    }

    # 15
    def f1_on(v: list[bool]) -> Callable[[Sequence[int]], float]:
        return lambda idx: _f1([v[i] for i in idx], [labels[i] for i in idx])

    groups = [e.source_id for e in cohort]
    paired: dict[str, Any] = {"resamples": frozen.bootstrap_resamples,
                              "seed": frozen.bootstrap_seed, "groups": "source_id"}
    for run in ("glm", "deepseek"):
        diff, lo, hi = metrics.paired_bootstrap_diff(
            f1_on(verdicts["jev_untuned"]), f1_on(verdicts[run]), len(cohort), groups=groups,
            n_resamples=frozen.bootstrap_resamples, seed=frozen.bootstrap_seed)
        paired[f"jev_untuned_minus_{run}"] = {"diff": diff, "ci95": [lo, hi]}
    out["paired_untuned_vs_glm_deepseek"] = paired

    # 16
    wording: dict[str, Any] = {"threshold": UNTUNED_JEV_THRESHOLD,
                               "notes": "scores are P(supported) for every wording (inverted "
                                        "wordings reoriented at scoring time)"}
    for run in PARAPHRASE_RUNS:
        preds = _ok(store, run)
        sub = [e for e in cohort if e.id in preds]
        wording[run] = {
            "n": len(sub), "n_missing": len(cohort) - len(sub),
            "f1": _f1([_s(preds[e.id]) < UNTUNED_JEV_THRESHOLD for e in sub],
                      [e.hallucinated for e in sub]) if sub else None,
        }
    out["wording_f1_at_default_cutoff"] = wording
    return out


def _print_summary(x: dict[str, Any]) -> None:
    def r(v: float | None) -> str:
        return "None" if v is None else f"{v:.3f}"

    hc = x["headline_cohort"]
    print(f"headline cohort n={hc['n']} pos={hc['n_positive']}")
    d = x["disagreement_untuned_vs_claude"]
    print(f"1 disagree untuned-vs-claude n={d['n']} jev_right={d['jev_right']} "
          f"claude_right={d['claude_right']}")
    print("7 oracle F1: " + ", ".join(
        f"{j} {r(v['f1'])}@{r(v['threshold'])} (rule {r(v['f1_at_own_rule'])})"
        for j, v in x["oracle_best_f1"].items() if j != "notes"))
    j = x["jev_score_jitter"]
    print(f"9 jitter n={j['n']} changed={j['n_changed']} median={j['median']:.4f} "
          f"p90={j['p90']:.4f} max={j['max']:.4f} near-threshold={r(j['near_tuned_threshold']['fraction'])}")
    c = x["cascade_detail"]
    print(f"10 cascade recall={r(c['cascade']['recall'])} fp={c['cascade']['fp']} "
          f"band n={c['within_band']['n']} cost/1k=${c['cost_usd_per_1k']['cascade_total']:.3f}")
    p = x["paired_untuned_vs_glm_deepseek"]
    print("15 " + ", ".join(f"{k} {r(v['diff'])} [{r(v['ci95'][0])}, {r(v['ci95'][1])}]"
                            for k, v in p.items() if isinstance(v, dict)))
    w = x["wording_f1_at_default_cutoff"]
    print("16 " + ", ".join(f"{k[-1]} {r(v['f1'])}" for k, v in w.items() if isinstance(v, dict)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="bench results directory")
    parser.add_argument("--frozen", type=Path, required=True, help="frozen_config.json")
    parser.add_argument("--dest", type=Path, default=None, help="default: <out>/extras.json")
    args = parser.parse_args()
    frozen, _ = load_frozen(args.frozen)
    data_dir = ensure_downloaded()
    test = load_examples(data_dir, split="test")
    tune = select_examples(RUNS["tune-A"], data_dir)
    extras = compute(test, tune, PredictionStore(args.out), frozen)
    dest = args.dest or args.out / "extras.json"
    dest.write_text(json.dumps(extras, indent=2) + "\n")
    _print_summary(extras)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
