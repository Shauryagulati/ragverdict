"""Turn cached predictions into summary.json — the single source for every published number.

Conventions: positive = hallucinated; Jev's stored score is P(supported); every CI is a
source-grouped percentile bootstrap with the frozen resamples/seed; "paired" means both arms
are scored on the same resampled groups. The primary (confirmatory) analyses are H1 and H9;
every other hypothesis is exploratory and reported through its numbers only.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from typing import Any

from ragverdict.bench import metrics
from ragverdict.bench.predict import CLAUDE_PRICES, Prediction, PredictionStore, total_spend
from ragverdict.bench.ragtruth import Example
from ragverdict.bench.runs import MODELS, PARAPHRASES, PRICES_CHECKED, FrozenConfig

Rule = Callable[[Prediction], bool]  # verdict rule: True = "hallucinated"
Stat = Callable[[Sequence[int]], float]  # statistic over a list of example indices

LLM_RUNS = ("claude", "deepseek", "glm")
# Headline Jev: wording A (the product's default question, never tuned) at the default 0.5.
# Run "jev" uses the train-tuned wording/threshold and is only the secondary "tuned" row.
HEADLINE_JEV_RUN = "jev-para-A"
UNTUNED_JEV_THRESHOLD = 0.5
H1_DELTA = 0.03  # equivalence margin on the F1 difference (TOST, 90% CI)
H9_MARGIN = 0.02  # H9 predicts the cascade beats the best single judge by < 0.02 F1
EXPLORATORY = ["H2", "H3", "H4", "H5", "H6", "H7", "H8", "H10", "H11", "H12"]
DESCRIPTIVE_BELOW_N_POS = 50  # subtype comparisons with fewer positives are descriptive only
CLASSIFICATION_KEYS = ("precision", "recall", "f1", "macro_f1", "accuracy")

# Subtype families for H2-H4. Each maps a positive to its subtype; clean examples map to None.
SUBTYPE_FAMILIES: dict[str, Callable[[Example], str | None]] = {
    "severity": lambda e: e.severity,
    "kind": lambda e: e.kind,
    "numeric": lambda e: None if not e.hallucinated else ("numeric" if e.numeric else "non_numeric"),
    "generator": lambda e: e.generator if e.hallucinated else None,
}


# ---------- verdict rules ----------

def _score(p: Prediction) -> float:
    if p.score is None:
        raise ValueError(f"{p.run}/{p.example_id} has no score")
    return float(p.score)


def _claims_unsupported(p: Prediction) -> bool:
    return p.total_claims > 0 and p.supported_claims < p.total_claims


def llm_hallucinated(p: Prediction) -> bool:
    """Pre-registered LLM verdict: hallucinated iff the self-reported
    score is below 1 OR the claim counts show an unsupported claim. Pilot-prompt runs record
    no claim counts (total_claims == 0), so for them this reduces to score < 1."""
    return _score(p) < 1.0 or _claims_unsupported(p)


def verdict_inconsistent(p: Prediction) -> bool:
    """The self-reported score and the claim counts disagree about "anything unsupported"."""
    return (_score(p) < 1.0) != _claims_unsupported(p)


def jev_rule(threshold: float) -> Rule:
    """Jev verdict: hallucinated iff P(supported) < threshold."""

    def hallucinated(p: Prediction) -> bool:
        return _score(p) < threshold

    return hallucinated


def h1_verdict(lo: float, hi: float, delta: float) -> str:
    """TOST reading of the 90% CI of F1(Jev untuned) - F1(Claude) against ±delta."""
    if -delta <= lo and hi <= delta:
        return "equivalent"
    if lo > delta:
        return "jev_better"
    if hi < -delta:
        return "jev_worse"
    return "inconclusive"


def h9_verdict(lo: float, hi: float, margin: float) -> str:
    """95% CI of F1(cascade) - F1(best single judge) against the predicted gain < margin."""
    if hi < margin:
        return "confirmed"
    if lo >= margin:
        return "refuted"
    return "inconclusive"


# ---------- loading and bootstrap helpers ----------

def _ok(store: PredictionStore, run: str) -> dict[str, Prediction]:
    """Successful repeat-0 predictions for `run`, keyed by example id."""
    return {
        eid: p for (eid, rep), p in store.load(run).items() if rep == 0 and p.score is not None
    }


def _repeat0(store: PredictionStore, run: str) -> dict[str, Prediction]:
    """Every repeat-0 row for `run` (successes and failures), keyed by example id."""
    return {eid: p for (eid, rep), p in store.load(run).items() if rep == 0}


def _failure_split(
    ids: list[Example], all_rows: dict[str, Prediction]
) -> dict[str, Any]:
    """n_failed (row exists with an error), by error_kind, and n_missing (no row at all)."""
    kinds = [
        all_rows[e.id].error_kind
        for e in ids
        if e.id in all_rows and all_rows[e.id].error is not None
    ]
    return {
        "n_failed": len(kinds),
        "n_failed_by_kind": {
            "transport": kinds.count("transport"),
            "judge": kinds.count("judge"),
            # rows with an error but no error_kind (pre-error_kind legacy rows) — counted here
            # so the buckets always sum to n_failed.
            "unknown": sum(1 for k in kinds if k not in ("transport", "judge")),
        },
        "n_missing": sum(1 for e in ids if e.id not in all_rows),
    }


def _percentile_50_95(values: Sequence[float]) -> dict[str, float]:
    """p50/p95 via `statistics.quantiles(n=100)`; a single value stands in for both when n < 2."""
    if len(values) < 2:
        v = values[0]
        return {"p50": v, "p95": v}
    q = statistics.quantiles(values, n=100, method="inclusive")
    return {"p50": q[49], "p95": q[94]}


def _groups(exs: Sequence[Example]) -> list[str]:
    """Bootstrap clusters: responses sharing a source are resampled together."""
    return [e.source_id or e.id for e in exs]


def _ci(fn: Stat, groups: list[str], frozen: FrozenConfig) -> dict[str, Any]:
    n = len(groups)
    value = fn(list(range(n)))
    lo, hi = metrics.bootstrap_ci(
        fn, n, groups=groups, n_resamples=frozen.bootstrap_resamples, seed=frozen.bootstrap_seed
    )
    return {"value": value, "ci95": [lo, hi]}


def _metric_on(verdicts: Sequence[bool], labels: Sequence[bool], key: str) -> Stat:
    """Classification metric `key` of `verdicts` vs `labels` on a list of indices."""

    def stat(idx: Sequence[int]) -> float:
        return metrics.classification([verdicts[i] for i in idx], [labels[i] for i in idx])[key]

    return stat


def _classification_cis(
    verdicts: Sequence[bool],
    labels: Sequence[bool],
    groups: list[str],
    frozen: FrozenConfig,
    keys: Sequence[str] = CLASSIFICATION_KEYS,
) -> dict[str, Any]:
    """{metric: {"value", "ci95"}} for each classification metric in `keys`."""
    return {key: _ci(_metric_on(verdicts, labels, key), groups, frozen) for key in keys}


def _counting(fn: Stat, dropped: list[int]) -> Stat:
    """Wrap `fn` so each ValueError (statistic undefined on a resample) is tallied."""

    def wrapped(idx: Sequence[int]) -> float:
        try:
            return fn(idx)
        except ValueError:
            dropped[0] += 1
            raise

    return wrapped


def _stat_ci(stat: Stat, groups: list[str], frozen: FrozenConfig) -> dict[str, Any]:
    """Value + 95% bootstrap CI for a statistic that raises ValueError where undefined (a
    one-class AUROC; Jev recall at a precision its curve never reaches). Undefined resamples
    are skipped, as in metrics.bootstrap_ci, and counted so the CI's support is visible."""
    n = len(groups)
    dropped = [0]
    try:
        value: float | None = stat(list(range(n)))
    except ValueError:
        value = None
    try:
        lo, hi = metrics.bootstrap_ci(
            _counting(stat, dropped), n, groups=groups,
            n_resamples=frozen.bootstrap_resamples, seed=frozen.bootstrap_seed,
        )
        ci: list[float] | None = [lo, hi]
    except ValueError:  # undefined on every resample
        ci = None
    return {"value": value, "ci95": ci, "n_resamples_undefined": dropped[0]}


def _diff_ci(
    a: Stat, b: Stat, groups: list[str], frozen: FrozenConfig, *, alpha: float = 0.05
) -> dict[str, Any] | None:
    """Paired a - b with a (1 - alpha) bootstrap CI (key "ci95" or "ci90"); None if undefined
    on the full cohort or on every resample. Undefined resamples are skipped and counted."""
    dropped = [0]
    try:
        diff, lo, hi = metrics.paired_bootstrap_diff(
            _counting(a, dropped), _counting(b, dropped), len(groups), groups=groups,
            n_resamples=frozen.bootstrap_resamples, seed=frozen.bootstrap_seed, alpha=alpha,
        )
    except ValueError:
        return None
    return {"diff": diff, f"ci{round(100 * (1 - alpha))}": [lo, hi],
            "n_resamples_undefined": dropped[0]}


# ---------- per-judge blocks (each judge on its own scored examples) ----------

def _judge_block(
    examples: list[Example],
    preds: dict[str, Prediction],
    halluc_pred: Rule,
    frozen: FrozenConfig,
    store: PredictionStore,
    run: str,
    *,
    probabilistic: bool,
) -> dict[str, Any]:
    exs = [e for e in examples if e.id in preds]
    labels = [e.hallucinated for e in exs]
    groups = _groups(exs)
    h_scores = [1.0 - _score(preds[e.id]) for e in exs]
    verdicts = [halluc_pred(preds[e.id]) for e in exs]

    def auroc_on(idx: Sequence[int]) -> float:
        return metrics.auroc([h_scores[i] for i in idx], [labels[i] for i in idx])

    all_rows = _repeat0(store, run)  # every attempted row, success or failure, for this run

    block: dict[str, Any] = {
        "run": run,
        "n": len(exs),
        **_failure_split(examples, all_rows),  # n_failed / n_failed_by_kind / n_missing
        "total_spend_usd": total_spend(store, run),  # all attempts, incl. superseded/retried
        "auroc": _ci(auroc_on, groups, frozen),
        **_classification_cis(verdicts, labels, groups, frozen, keys=("f1", "macro_f1")),
        "classification": metrics.classification(verdicts, labels),
        # Pilot-style counts: hallucinations missed, clean answers falsely flagged
        "missed": sum(y and not v for y, v in zip(labels, verdicts, strict=True)),
        "false_alarms": sum(v and not y for y, v in zip(labels, verdicts, strict=True)),
        "n_positive": sum(labels),
        "n_negative": len(labels) - sum(labels),
        "cost_usd_total": sum(preds[e.id].cost_usd for e in exs),
        "cost_usd_per_1k": 1000 * sum(preds[e.id].cost_usd for e in exs) / max(len(exs), 1),
        "cost_usd_per_1m": 1_000_000 * sum(preds[e.id].cost_usd for e in exs) / max(len(exs), 1),
        "distinct_scores": len({preds[e.id].score for e in exs}),
        "frac_score_eq_1": sum(preds[e.id].score == 1.0 for e in exs) / max(len(exs), 1),
    }
    if not probabilistic:  # LLM judge: how often the score and the claim counts disagree
        n_inconsistent = sum(verdict_inconsistent(preds[e.id]) for e in exs)
        block["n_verdict_inconsistent"] = n_inconsistent
        block["verdict_inconsistency_rate"] = n_inconsistent / max(len(exs), 1)
    # Latency over primary-cohort examples that were actually scored — not every raw row
    # (a full-test run scores "all" quality; only "good" rows belong in the reported cohort).
    scored_latencies = [
        lat for e in exs if (lat := preds[e.id].latency_s) is not None
    ]
    if scored_latencies:
        block["latency_s"] = _percentile_50_95(scored_latencies)
    if probabilistic:
        block["ece"] = metrics.ece(h_scores, labels)
        block["reliability"] = metrics.reliability(h_scores, labels)
    block["by_task"] = _slices(exs, labels, h_scores, verdicts, frozen, key=lambda e: e.task)
    block["by_generator"] = _slices(
        exs, labels, h_scores, verdicts, frozen, key=lambda e: e.generator
    )
    # H5: length quartiles cut within each task. Global cuts mostly re-slice by task (54% of
    # QA falls in the global Q1), so they'd confound length with task difficulty.
    block["by_length_quartile"] = {
        task: _slices(
            exs, labels, h_scores, verdicts, frozen,
            key=_quartile_key([e for e in examples if e.task == task], task),
        )
        for task in sorted({e.task for e in exs})
    }
    block["recall_by_severity"] = _recall(exs, verdicts, key=lambda e: e.severity)
    block["recall_by_kind"] = _recall(exs, verdicts, key=lambda e: e.kind)
    block["recall_numeric"] = _recall(exs, verdicts, key=SUBTYPE_FAMILIES["numeric"])
    return block


def _quartile_key(task_examples: list[Example], task: str) -> Callable[[Example], str | None]:
    """Length quartile of an example of `task`, with cuts from that task's primary-cohort
    examples (every example, scored or not); None for examples of any other task."""
    lengths = [e.n_chars for e in task_examples]
    cuts = statistics.quantiles(lengths, n=4) if len(lengths) >= 2 else []

    def key(e: Example) -> str | None:
        if e.task != task:
            return None
        return f"Q{1 + sum(e.n_chars > c for c in cuts)}"

    return key


def _slices(
    exs: list[Example],
    labels: list[bool],
    h_scores: list[float],
    verdicts: list[bool],
    frozen: FrozenConfig,
    *,
    key: Callable[[Example], str | None],
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for i, e in enumerate(exs):
        k = key(e)
        if k is not None:
            groups.setdefault(k, []).append(i)
    out: dict[str, Any] = {}
    for name, idx in sorted(groups.items()):
        sub_labels = [labels[i] for i in idx]
        entry: dict[str, Any] = {
            "n": len(idx),
            "n_hallucinated": sum(sub_labels),
            **metrics.classification([verdicts[i] for i in idx], sub_labels),
        }
        sub_scores = [h_scores[i] for i in idx]
        sub_groups = [exs[i].source_id or exs[i].id for i in idx]

        def sub_auroc(j: Sequence[int], s: list[float] = sub_scores, y: list[bool] = sub_labels) -> float:
            return metrics.auroc([s[k] for k in j], [y[k] for k in j])

        try:
            entry["auroc"] = sub_auroc(list(range(len(idx))))
            entry["auroc_ci95"] = list(metrics.bootstrap_ci(
                sub_auroc, len(idx), groups=sub_groups, n_resamples=frozen.bootstrap_resamples,
                seed=frozen.bootstrap_seed,
            ))
        except ValueError:  # slice has only one class
            entry["auroc"] = None
            entry["auroc_ci95"] = None
        out[name] = entry
    return out


def _recall(
    exs: list[Example], verdicts: list[bool], *, key: Callable[[Example], str | None]
) -> dict[str, Any]:
    groups: dict[str, list[bool]] = {}
    for e, v in zip(exs, verdicts, strict=True):
        k = key(e)
        if k is not None:
            groups.setdefault(k, []).append(v)
    return {
        k: {"n": len(v), "recall": sum(v) / len(v), "recall_ci95": list(metrics.wilson_ci(sum(v), len(v)))}
        for k, v in sorted(groups.items())
    }


# slavadubrov/sgr-judge-bench @ 5e142707 results/pilot/README.md (2026-09-19): RAGTruth QA
# test, 900 responses (740 clean / 160 hallucinated), Jev thresholded at 0.5. Cited, not re-run.
PILOT_QA: dict[str, dict[str, float]] = {
    "jev": {"accuracy": 0.7200, "macro_f1": 0.6667, "missed": 16, "false_alarms": 236},
    "gpt-5.6-luna": {"accuracy": 0.7111, "macro_f1": 0.6539, "missed": 23, "false_alarms": 237},
    "deepseek-flash": {"accuracy": 0.8044, "macro_f1": 0.7303, "missed": 34, "false_alarms": 142},
    "glm-5.3-flash": {"accuracy": 0.8322, "macro_f1": 0.7661, "missed": 25, "false_alarms": 121},
}


def _paired_vs_jev(
    test: list[Example],
    jev: dict[str, Prediction],
    other: dict[str, Prediction],
    jev_halluc: Rule,
    other_rule: Rule,
    frozen: FrozenConfig,
) -> dict[str, Any]:
    """Jev minus comparator on the examples both scored, source-grouped paired bootstrap."""
    both = [e for e in test if e.id in jev and e.id in other]
    y = [e.hallucinated for e in both]
    jv = [jev_halluc(jev[e.id]) for e in both]
    ov = [other_rule(other[e.id]) for e in both]
    out: dict[str, Any] = {"n": len(both)}
    for key in ("macro_f1", "f1"):
        out[f"{key}_diff"] = list(metrics.paired_bootstrap_diff(
            _metric_on(jv, y, key), _metric_on(ov, y, key), len(both), groups=_groups(both),
            n_resamples=frozen.bootstrap_resamples, seed=frozen.bootstrap_seed,
        ))
    return out


def _qa_rows(judges: dict[str, Any]) -> dict[str, Any]:
    """QA-task rows in the 2026-09-19 pilot's format, for a side-by-side replication table."""
    rows = {}
    for name, block in judges.items():
        qa = block["by_task"].get("QA")
        if qa is None:
            continue
        rows[name] = {
            "n": qa["n"],
            "accuracy": qa["accuracy"],
            "macro_f1": qa["macro_f1"],
            "missed": qa["fn"],
            "false_alarms": qa["fp"],
            "n_positive": qa["n_hallucinated"],
            "n_negative": qa["n"] - qa["n_hallucinated"],
            "latency_s": block.get("latency_s"),
            "cost_usd_per_1k": block["cost_usd_per_1k"],
        }
    return rows


def _qa_pilot_rules(
    everything: list[Example],
    store: PredictionStore,
    jev_halluc: Rule,
    frozen: FrozenConfig,
) -> dict[str, Any]:
    """QA rows under the pilot's cohort rules: all 900 QA test responses (no quality filter),
    hallucinated = any span incl. implicit_true. Unscored responses split into n_failed
    (a row exists with an error) and n_missing (no row at all), same as the primary judge
    blocks."""
    qa = [e for e in everything if e.task == "QA"]
    rows: dict[str, Any] = {"n": len(qa), "n_positive": sum(e.hallucinated_any_span for e in qa)}

    def row_for(run: str, rule: Rule) -> dict[str, Any] | None:
        preds = _ok(store, run)
        if not preds:
            return None
        all_rows = _repeat0(store, run)
        scored = [e for e in qa if e.id in preds]
        m = metrics.classification(
            [rule(preds[e.id]) for e in scored],
            [e.hallucinated_any_span for e in scored],
        )
        return {
            "n_scored": len(scored),
            **_failure_split(qa, all_rows),
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "missed": m["fn"], "false_alarms": m["fp"],
        }

    untuned = jev_rule(UNTUNED_JEV_THRESHOLD)
    for name, run, rule in (
        ("jev", "jev", jev_halluc),
        ("claude", "claude", llm_hallucinated),
        ("deepseek", "deepseek", llm_hallucinated),
        ("glm", "glm", llm_hallucinated),
        # The 2026-09-19 pilot's exact Jev setting: paraphrase P, hallucinated iff p < 0.5,
        # independent of whatever threshold/paraphrase is currently frozen.
        ("jev_pilot_setting", "jev-para-P", untuned),
        # Replication arm: the pilot's state keys for Jev, the pilot's
        # binary prompt for DeepSeek/GLM (no claim counts, so the LLM rule is score < 1).
        ("jev_pilot_state", "jev-pilot-state", untuned),
        ("deepseek_pilot", "deepseek-pilot", llm_hallucinated),
        ("glm_pilot", "glm-pilot", llm_hallucinated),
    ):
        row = row_for(run, rule)
        if row is not None:
            rows[name] = row
    if "jev" in rows:
        rows["jev"]["paraphrase"] = frozen.jev_paraphrase
        rows["jev"]["threshold"] = frozen.jev_threshold
    return rows


def _wall_clock(store: PredictionStore) -> dict[str, dict[str, float]]:
    """Published wall-clock per run: sum of `wall_clock_s` over invocations that actually
    computed something (n_computed > 0), plus the total n_computed. A no-op rerun (everything
    already cached) writes an invocation with n_computed == 0, which never moves this number —
    a cache hit's near-zero wall clock would otherwise misleadingly avg into the real timing."""
    meta_dir = store.root / "meta"
    if not meta_dir.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    for p in sorted(meta_dir.glob("*.json")):
        data = json.loads(p.read_text())
        computed = [inv for inv in data.get("invocations", []) if inv.get("n_computed", 0) > 0]
        if not computed:
            continue
        out[p.stem] = {
            "seconds": sum(float(inv["wall_clock_s"]) for inv in computed),
            "n_computed": sum(int(inv["n_computed"]) for inv in computed),
        }
    return out


# ---------- headline (primary analysis on the common intersection) ----------

def _headline_judges(
    cohort: list[Example],
    para_a: dict[str, Prediction],
    llms: dict[str, dict[str, Prediction]],
    tuned: dict[str, Prediction],
    frozen: FrozenConfig,
    keys: Sequence[str] = CLASSIFICATION_KEYS,
) -> dict[str, Any]:
    """Untuned Jev and each LLM on `cohort`, plus tuned Jev (run "jev") as a secondary row.
    Run "jev", when present, is part of the cohort's intersection, so every row shares one n."""
    labels = [e.hallucinated for e in cohort]
    groups = _groups(cohort)
    untuned = jev_rule(UNTUNED_JEV_THRESHOLD)
    out: dict[str, Any] = {
        "jev_untuned": {
            "run": HEADLINE_JEV_RUN, "question": PARAPHRASES["A"][0],
            "threshold": UNTUNED_JEV_THRESHOLD, "n": len(cohort),
            **_classification_cis([untuned(para_a[e.id]) for e in cohort], labels, groups,
                                  frozen, keys),
        },
    }
    for run, preds in llms.items():
        out[run] = {
            "run": run, "rule": "score < 1.0 or supported_claims < total_claims",
            "n": len(cohort),
            **_classification_cis([llm_hallucinated(preds[e.id]) for e in cohort], labels,
                                  groups, frozen, keys),
        }
    if tuned:
        tuned_rule = jev_rule(frozen.jev_threshold)
        out["jev_tuned"] = {
            "label": "tuned_on_train", "run": "jev", "paraphrase": frozen.jev_paraphrase,
            "threshold": frozen.jev_threshold, "n": len(cohort),
            **_classification_cis([tuned_rule(tuned[e.id]) for e in cohort], labels, groups,
                                  frozen, keys),
        }
    return out


def _h1(
    cohort: list[Example],
    para_a: dict[str, Prediction],
    claude: dict[str, Prediction],
    frozen: FrozenConfig,
) -> dict[str, Any]:
    """H1 (primary): paired F1(Jev untuned) - F1(Claude) with a 90% CI, read as TOST."""
    labels = [e.hallucinated for e in cohort]
    untuned = jev_rule(UNTUNED_JEV_THRESHOLD)
    jev_v = [untuned(para_a[e.id]) for e in cohort]
    claude_v = [llm_hallucinated(claude[e.id]) for e in cohort]
    diff, lo, hi = metrics.paired_bootstrap_diff(
        _metric_on(jev_v, labels, "f1"), _metric_on(claude_v, labels, "f1"), len(cohort),
        groups=_groups(cohort), n_resamples=frozen.bootstrap_resamples,
        seed=frozen.bootstrap_seed, alpha=0.10,
    )
    return {
        "metric": "f1", "comparison": "jev_untuned - claude", "n": len(cohort),
        "diff": diff, "ci90": [lo, hi], "delta": H1_DELTA, "verdict": h1_verdict(lo, hi, H1_DELTA),
    }


def _threshold_free(
    cohort: list[Example],
    para_a: dict[str, Prediction],
    llm: dict[str, Prediction],
    frozen: FrozenConfig,
) -> dict[str, Any]:
    """Jev recall at the LLM's precision and Jev precision at the LLM's recall. Each bootstrap
    resample recomputes both the LLM's operating point and Jev's PR curve."""
    labels = [e.hallucinated for e in cohort]
    p_sup = [_score(para_a[e.id]) for e in cohort]
    llm_v = [llm_hallucinated(llm[e.id]) for e in cohort]
    groups = _groups(cohort)

    def curve_and_point(idx: Sequence[int]) -> tuple[list[dict[str, float]], dict[str, float]]:
        y = [labels[i] for i in idx]
        return (metrics.pr_curve([p_sup[i] for i in idx], y),
                metrics.classification([llm_v[i] for i in idx], y))

    def raw_recall_at_llm_precision(idx: Sequence[int]) -> float | None:
        curve, point = curve_and_point(idx)
        return metrics.recall_at_precision(curve, point["precision"])

    def recall_at_llm_precision(idx: Sequence[int]) -> float:
        # No Jev threshold reaches the LLM's precision: the only operating point that does
        # not fall short of it is flagging nothing (precision undefined), whose recall is 0.
        value = raw_recall_at_llm_precision(idx)
        return 0.0 if value is None else value

    def raw_precision_at_llm_recall(idx: Sequence[int]) -> float | None:
        curve, point = curve_and_point(idx)
        return metrics.precision_at_recall(curve, point["recall"])

    def precision_at_llm_recall(idx: Sequence[int]) -> float:
        # Jev's last threshold flags everything (recall 1), so this is None only on a
        # degenerate resample; that is genuinely undefined and counted as such.
        value = raw_precision_at_llm_recall(idx)
        if value is None:
            raise ValueError("Jev's curve never reaches the LLM's recall")
        return value

    everyone = list(range(len(cohort)))
    point = metrics.classification(llm_v, labels)
    return {
        "llm_precision": point["precision"],
        "llm_recall": point["recall"],
        "jev_recall_at_llm_precision": {
            **_stat_ci(recall_at_llm_precision, groups, frozen),
            "flag_nothing": raw_recall_at_llm_precision(everyone) is None,
        },
        "jev_precision_at_llm_recall": {
            **_stat_ci(precision_at_llm_recall, groups, frozen),
            "flag_nothing": False,  # flagging nothing never reaches a recall target > 0
        },
    }


def _cost_basis(
    cohort: list[Example],
    para_a: dict[str, Prediction],
    claude: dict[str, Prediction],
    frozen: FrozenConfig,
) -> dict[str, Any]:
    """H1's cost statement on a comparable basis: Claude at standard live, uncached list price
    (every input token — uncached, cache-read and cache-write — at the standard input price)
    vs Jev's billed cost. Billed/batch costs stay in the judge blocks."""
    price_in, price_out = CLAUDE_PRICES[frozen.claude_model]
    n = len(cohort)

    def standard_uncached(p: Prediction) -> float:
        tokens_in = p.input_tokens + p.cache_read_tokens + p.cache_write_tokens
        return (tokens_in * price_in + p.output_tokens * price_out) / 1_000_000

    claude_std = sum(standard_uncached(claude[e.id]) for e in cohort) / n
    jev_billed = sum(para_a[e.id].cost_usd for e in cohort) / n
    return {
        "n": n,
        "claude_usd_per_example_standard_uncached": claude_std,
        "claude_usd_per_example_billed": sum(claude[e.id].cost_usd for e in cohort) / n,
        "jev_usd_per_example_billed": jev_billed,
        "claude_to_jev_cost_ratio_standard_uncached": claude_std / jev_billed if jev_billed else None,
        "notes": {
            "claude_standard_uncached": (
                f"{frozen.claude_model} at standard live list price, no batch discount, no "
                f"prompt caching: input + cache-read + cache-write tokens at ${price_in}/MTok, "
                f"output at ${price_out}/MTok (prices checked {PRICES_CHECKED})."
            ),
            "claude_billed": (
                "Computed from tokens x hard-coded prices at the Batch API's 50% rate, cache "
                "reads at 0.1x and writes at 1.25x input; not reconciled with the console."
            ),
            "jev_billed": f"OpenRouter-reported cost of run {HEADLINE_JEV_RUN}.",
        },
    }


def _headline(
    cohort: list[Example],
    para_a: dict[str, Prediction],
    llms: dict[str, dict[str, Prediction]],
    tuned: dict[str, Prediction],
    frozen: FrozenConfig,
) -> dict[str, Any]:
    claude = llms.get("claude")
    return {
        "cohort": {
            "n": len(cohort),
            "n_positive": sum(e.hallucinated for e in cohort),
            "runs_intersected": [HEADLINE_JEV_RUN, *(["jev"] if tuned else []), *llms],
        },
        "judges": _headline_judges(cohort, para_a, llms, tuned, frozen),
        "h1": _h1(cohort, para_a, claude, frozen) if claude else None,
        "threshold_free": {
            run: _threshold_free(cohort, para_a, preds, frozen) for run, preds in llms.items()
        },
        "cost_basis": _cost_basis(cohort, para_a, claude, frozen) if claude else None,
    }


# ---------- H9 ----------

def _h9(
    cohort: list[Example],
    tuned: dict[str, Prediction],
    para_a: dict[str, Prediction],
    claude: dict[str, Prediction],
    frozen: FrozenConfig,
) -> dict[str, Any]:
    """H9 (primary): cascade F1 at the frozen band minus the best single judge's F1, on the
    headline cohort (which includes run "jev": the cascade's primary is the tuned Jev its band
    was frozen for). The best single judge is picked by point F1 on this cohort and then held
    fixed across resamples (ties go to the first in jev_tuned, jev_untuned, claude order)."""
    labels = [e.hallucinated for e in cohort]
    p_tuned = [_score(tuned[e.id]) for e in cohort]
    claude_v = [llm_hallucinated(claude[e.id]) for e in cohort]
    cascade_v, escalated = metrics.cascade(
        p_tuned, claude_v, threshold=frozen.jev_threshold, band=frozen.cascade_band
    )
    untuned = jev_rule(UNTUNED_JEV_THRESHOLD)
    singles = {
        "jev_tuned": [p < frozen.jev_threshold for p in p_tuned],
        "jev_untuned": [untuned(para_a[e.id]) for e in cohort],
        "claude": claude_v,
    }
    single_f1 = {name: metrics.classification(v, labels)["f1"] for name, v in singles.items()}
    best = max(single_f1, key=lambda name: single_f1[name])
    diff = _diff_ci(_metric_on(cascade_v, labels, "f1"), _metric_on(singles[best], labels, "f1"),
                    _groups(cohort), frozen)
    assert diff is not None  # F1 is defined on every resample
    lo, hi = diff["ci95"]
    return {
        "n": len(cohort),
        "band": list(frozen.cascade_band),
        "threshold": frozen.jev_threshold,
        "cascade_f1": metrics.classification(cascade_v, labels)["f1"],
        "escalation_rate": sum(escalated) / len(escalated),
        "single_judge_f1": single_f1,
        "best_single_judge": best,
        "diff": diff["diff"],
        "ci95": [lo, hi],
        "margin": H9_MARGIN,
        "verdict": h9_verdict(lo, hi, H9_MARGIN),
        "notes": (
            "best single judge chosen by point F1 on the full cohort and held fixed across "
            "resamples; point estimate carries winner's-curse bias toward 'confirmed'"
        ),
    }


# ---------- sensitivity analyses ----------

def _invalid_as_wrong(
    test: list[Example], store: PredictionStore, rules: dict[str, tuple[str, Rule]],
    frozen: FrozenConfig,
) -> dict[str, Any]:
    """The pilot's rule on the whole primary cohort: every example a judge did not score
    (judge failure, transport failure, or no row at all) counts as the wrong verdict."""
    labels = [e.hallucinated for e in test]
    groups = _groups(test)
    out: dict[str, Any] = {}
    for name, (run, rule) in rules.items():
        rows = _repeat0(store, run)
        if not any(p.score is not None for p in rows.values()):
            continue
        verdicts: list[bool] = []
        n_invalid = 0
        for e in test:
            p = rows.get(e.id)
            if p is not None and p.score is not None:
                verdicts.append(rule(p))
            else:
                n_invalid += 1
                verdicts.append(not e.hallucinated)
        out[name] = {
            "run": run, "n": len(test), "n_invalid": n_invalid, **_failure_split(test, rows),
            **_classification_cis(verdicts, labels, groups, frozen, keys=("f1", "macro_f1")),
        }
    return out


# ---------- subtypes (exploratory, H2-H4) ----------

def _subtype_members(cohort: list[Example]) -> dict[str, dict[str, list[bool]]]:
    """family -> subtype -> membership mask over `cohort` (positives of that subtype)."""
    out: dict[str, dict[str, list[bool]]] = {}
    for family, key in SUBTYPE_FAMILIES.items():
        names = sorted({k for e in cohort if (k := key(e)) is not None})
        out[family] = {name: [key(e) == name for e in cohort] for name in names}
    return out


def _matched_fpr(
    cohort: list[Example],
    para_a: dict[str, Prediction],
    llm: dict[str, Prediction],
    frozen: FrozenConfig,
) -> dict[str, Any] | None:
    """Subtype recall at a matched false-positive rate: Jev's threshold is the largest one
    whose FPR on the cohort's clean examples is <= the LLM's, then subtype recall is compared
    as a paired difference. Each resample recomputes the LLM's FPR and Jev's threshold. When no
    threshold fits the budget, Jev flags nothing (FPR 0, recall 0) — a real, always-feasible
    operating point, so those resamples are kept rather than skipped."""
    labels = [e.hallucinated for e in cohort]
    if all(labels):
        return None  # FPR undefined without clean examples
    p_sup = [_score(para_a[e.id]) for e in cohort]
    llm_v = [llm_hallucinated(llm[e.id]) for e in cohort]
    groups = _groups(cohort)

    def jev_threshold(idx: Sequence[int]) -> float:
        neg = [i for i in idx if not labels[i]]
        if not neg:
            raise ValueError("no clean examples in this resample")
        llm_fpr = sum(llm_v[i] for i in neg) / len(neg)
        return metrics.threshold_at_fpr([p_sup[i] for i in idx], [labels[i] for i in idx], llm_fpr)

    everyone = list(range(len(cohort)))
    negatives = [i for i in everyone if not labels[i]]
    t = jev_threshold(everyone)
    out: dict[str, Any] = {
        "llm_fpr": sum(llm_v[i] for i in negatives) / len(negatives),
        "jev_threshold": t,
        "jev_flags_nothing": t <= min(p_sup),
        "jev_fpr": sum(p_sup[i] < t for i in negatives) / len(negatives),
        "subtypes": {},
    }
    for family, subtypes in _subtype_members(cohort).items():
        fam: dict[str, Any] = {}
        for name, member in subtypes.items():
            def jev_recall(idx: Sequence[int], m: list[bool] = member) -> float:
                pos = [i for i in idx if m[i]]
                if not pos:
                    raise ValueError("no positives of this subtype in this resample")
                t_r = jev_threshold(idx)
                return sum(p_sup[i] < t_r for i in pos) / len(pos)

            def llm_recall(idx: Sequence[int], m: list[bool] = member) -> float:
                pos = [i for i in idx if m[i]]
                if not pos:
                    raise ValueError("no positives of this subtype in this resample")
                return sum(llm_v[i] for i in pos) / len(pos)

            n_pos = sum(member)
            fam[name] = {
                "n_pos": n_pos,
                "descriptive": n_pos < DESCRIPTIVE_BELOW_N_POS,
                "jev_recall": jev_recall(everyone),
                "llm_recall": llm_recall(everyone),
                "recall_diff_jev_minus_llm": _diff_ci(jev_recall, llm_recall, groups, frozen),
            }
        out["subtypes"][family] = fam
    return out


def _subtype_auroc(
    cohort: list[Example],
    h_scores: dict[str, list[float]],
    frozen: FrozenConfig,
) -> dict[str, Any]:
    """AUROC of each subtype's positives against ALL negatives in the cohort, per judge
    (`h_scores` maps judge -> P(hallucinated)-like scores; its first entry is Jev), plus the
    paired Jev-minus-judge difference for every other judge."""
    labels = [e.hallucinated for e in cohort]
    groups = _groups(cohort)
    jev_name, *others = h_scores
    n_neg = len(labels) - sum(labels)
    out: dict[str, Any] = {}
    for family, subtypes in _subtype_members(cohort).items():
        fam: dict[str, Any] = {}
        for name, member in subtypes.items():
            keep = [m or not y for m, y in zip(member, labels, strict=True)]

            def auroc_of(scores: list[float], k: list[bool] = keep) -> Stat:
                def stat(idx: Sequence[int]) -> float:
                    sub = [i for i in idx if k[i]]
                    return metrics.auroc([scores[i] for i in sub], [labels[i] for i in sub])

                return stat

            n_pos = sum(member)
            entry: dict[str, Any] = {
                "n_pos": n_pos, "n_neg": n_neg, "descriptive": n_pos < DESCRIPTIVE_BELOW_N_POS,
            }
            for judge, scores in h_scores.items():
                entry[judge] = _stat_ci(auroc_of(scores), groups, frozen)
            entry["diff_jev_minus"] = {
                judge: _diff_ci(auroc_of(h_scores[jev_name]), auroc_of(h_scores[judge]),
                                groups, frozen)
                for judge in others
            }
            fam[name] = entry
        out[family] = fam
    return out


# ---------- calibration (H6) ----------

def _calibration_extremes(test: list[Example], para_a: dict[str, Prediction],
                          frozen: FrozenConfig) -> dict[str, Any]:
    """H6 on untuned Jev: in the bins P(hallucinated) < 0.1 and > 0.9, mean predicted vs the
    observed hallucination rate, with Wilson and cluster-bootstrap 95% CIs of the rate.
    Bins are taken on P(supported) (> 0.9 / < 0.1) to avoid 1 - p rounding at the edges."""
    exs = [e for e in test if e.id in para_a]
    p_sup = [_score(para_a[e.id]) for e in exs]
    groups = _groups(exs)

    def bin_block(members: list[int], label: str) -> dict[str, Any]:
        n = len(members)
        if n == 0:
            return {"range": label, "n": 0, "mean_predicted": None, "observed_rate": None,
                    "wilson_ci95": None, "bootstrap_ci95": None}
        y = [exs[i].hallucinated for i in members]

        def rate(idx: Sequence[int]) -> float:
            return sum(y[i] for i in idx) / len(idx)

        return {
            "range": label,
            "n": n,
            "mean_predicted": sum(1 - p_sup[i] for i in members) / n,
            "observed_rate": sum(y) / n,
            "wilson_ci95": list(metrics.wilson_ci(sum(y), n)),
            "bootstrap_ci95": _ci(rate, [groups[i] for i in members], frozen)["ci95"],
        }

    return {
        "run": HEADLINE_JEV_RUN,
        "low": bin_block([i for i, p in enumerate(p_sup) if p > 0.9], "P(hallucinated) < 0.1"),
        "high": bin_block([i for i, p in enumerate(p_sup) if p < 0.1], "P(hallucinated) > 0.9"),
    }


# ---------- flip test (H8) ----------

def _cache_hit_check(scored: list[Prediction]) -> dict[str, Any]:
    """Were the flip repeats real, independent calls? A provider-side cache hit would return
    the same answer for free, faking a 0% flip rate. Ruled out iff every repeat of an example
    has its own non-empty response id and nonzero cost; None when any id is missing."""
    ids = [p.response_id for p in scored if p.response_id]
    n_zero_cost = sum(p.cost_usd == 0 for p in scored)
    n_missing = len(scored) - len(ids)
    ids_by_example: dict[str, list[str]] = {}
    for p in scored:
        ids_by_example.setdefault(p.example_id, []).append(p.response_id)
    distinct_per_example = all(len(set(v)) == len(v) for v in ids_by_example.values())
    return {
        "n_rows": len(ids),
        "n_distinct_response_ids": len(set(ids)),
        "n_rows_missing_response_id": n_missing,
        "n_zero_cost_rows": n_zero_cost,
        "cache_hits_ruled_out": (
            None if n_missing else (distinct_per_example and n_zero_cost == 0)
        ),
    }


# ---------- build_summary ----------

def build_summary(
    examples_by_split: dict[str, list[Example]],
    store: PredictionStore,
    frozen: FrozenConfig,
    frozen_sha: str,
) -> dict[str, Any]:
    everything = examples_by_split["test"]  # may include non-"good" rows (quality="all")
    test = [e for e in everything if e.quality == "good"]  # primary analysis cohort
    jev = _ok(store, "jev")
    claude = _ok(store, "claude")
    para_a = _ok(store, HEADLINE_JEV_RUN)
    llms = {run: preds for run in LLM_RUNS if (preds := _ok(store, run))}
    jev_halluc = jev_rule(frozen.jev_threshold)

    summary: dict[str, Any] = {
        "frozen_config": frozen.model_dump(),
        "frozen_config_sha256": frozen_sha,
        "n_test": len(test),
        "n_test_hallucinated": sum(e.hallucinated for e in test),
        "judges": {},
        "wall_clock_s": _wall_clock(store),
    }

    # Partial progress (only some judges scored yet) must not crash build_summary: every
    # judge block, and everything that compares two judges, is added only when its inputs
    # are non-empty.
    if jev:
        summary["judges"]["jev"] = _judge_block(
            test, jev, jev_halluc, frozen, store, "jev", probabilistic=True
        )
    if para_a:
        # Untuned Jev = the headline's Jev: wording A at 0.5 (not run "jev" at 0.5, which
        # would still carry the train-tuned wording).
        summary["judges"]["jev_untuned"] = _judge_block(
            test, para_a, jev_rule(UNTUNED_JEV_THRESHOLD), frozen, store, HEADLINE_JEV_RUN,
            probabilistic=True,
        )
    for run, preds in llms.items():
        summary["judges"][run] = _judge_block(
            test, preds, llm_hallucinated, frozen, store, run, probabilistic=False
        )

    all_labels = [e.hallucinated for e in test]
    summary["baselines"] = {
        "always_clean": metrics.classification([False] * len(test), all_labels),
    }
    summary["paired_vs_jev"] = {
        run: _paired_vs_jev(test, jev, preds, jev_halluc, llm_hallucinated, frozen)
        for run, preds in llms.items()
        if jev
    }

    # Headline: common intersection of the untuned Jev, the tuned Jev (when run "jev" has
    # predictions) and every present LLM judge — one cohort, one n for every primary.
    intersected = [*([jev] if jev else []), *llms.values()]
    cohort = [e for e in test if e.id in para_a and all(e.id in p for p in intersected)]
    if cohort:
        summary["headline"] = _headline(cohort, para_a, llms, jev, frozen)
        no_conv = [e for e in cohort if not e.convention_dependent]
        summary["sensitivity"] = {
            "excluding_convention_dependent": {
                "n": len(no_conv),
                "n_dropped": len(cohort) - len(no_conv),
                "judges": _headline_judges(no_conv, para_a, llms, jev, frozen,
                                           keys=("f1", "macro_f1")),
                "h1": (
                    {**_h1(no_conv, para_a, llms["claude"], frozen), "sensitivity": True}
                    if "claude" in llms else None
                ),
            } if no_conv else None,
        }
        jev_h = {"jev_untuned": [1 - _score(para_a[e.id]) for e in cohort]}
        llm_h = {run: [1 - _score(p[e.id]) for e in cohort] for run, p in llms.items()}
        summary["subtypes"] = {
            "exploratory": True,
            "jev_run": HEADLINE_JEV_RUN,
            "matched_fpr": {
                run: block for run, preds in llms.items()
                if (block := _matched_fpr(cohort, para_a, preds, frozen)) is not None
            },
            "subtype_auroc": _subtype_auroc(cohort, jev_h | llm_h, frozen),
        }
    else:  # never falls back to run "jev": the headline is defined on the untuned wording
        summary["headline"] = None
        summary["sensitivity"] = {"excluding_convention_dependent": None}
        summary["subtypes"] = None
    summary["sensitivity"]["invalid_as_wrong"] = _invalid_as_wrong(test, store, {
        "jev_untuned": (HEADLINE_JEV_RUN, jev_rule(UNTUNED_JEV_THRESHOLD)),
        "jev": ("jev", jev_halluc),
        **{run: (run, llm_hallucinated) for run in LLM_RUNS},
    }, frozen)
    summary["calibration_extremes"] = (
        _calibration_extremes(test, para_a, frozen) if para_a else None
    )
    h9 = (
        _h9(cohort, jev, para_a, llms["claude"], frozen)
        if cohort and jev and "claude" in llms else None
    )
    summary["hypotheses"] = {
        "H1": summary["headline"]["h1"] if summary["headline"] else None,
        "H9": h9,
        "exploratory": EXPLORATORY,
    }

    summary["qa_replication"] = _qa_rows(summary["judges"])
    summary["qa_replication_pilot_rules"] = _qa_pilot_rules(everything, store, jev_halluc, frozen)
    summary["pilot_reference_qa"] = PILOT_QA
    summary["models"] = {
        run: {
            **MODELS[run],
            "served": sorted({p.served_model for p in _ok(store, run).values() if p.served_model}),
            "prices_checked": PRICES_CHECKED,
        }
        for run in MODELS
        if _ok(store, run)
    }

    # Head-to-head, cascade, agreement (tuned Jev vs Claude, exploratory): only meaningful
    # once both have scored at least one overlapping example. None rather than raised on
    # partial progress. The pre-registered cascade test is hypotheses.H9 on the headline cohort.
    both = [e for e in test if e.id in jev and e.id in claude] if (jev and claude) else []
    if both:
        labels = [e.hallucinated for e in both]
        p_sup = [_score(jev[e.id]) for e in both]
        c_score = [_score(claude[e.id]) for e in both]
        c_halluc = [llm_hallucinated(claude[e.id]) for e in both]

        def jev_auroc(idx: Sequence[int]) -> float:
            return metrics.auroc([1 - p_sup[i] for i in idx], [labels[i] for i in idx])

        def claude_auroc(idx: Sequence[int]) -> float:
            return metrics.auroc([1 - c_score[i] for i in idx], [labels[i] for i in idx])

        diff, lo, hi = metrics.paired_bootstrap_diff(
            jev_auroc, claude_auroc, len(both), groups=_groups(both),
            n_resamples=frozen.bootstrap_resamples, seed=frozen.bootstrap_seed,
        )
        curve = metrics.pr_curve(p_sup, labels)
        claude_point = metrics.classification(c_halluc, labels)

        def _op_point(run_name: str) -> dict[str, float] | None:
            """Precision/recall for `run_name` on test ∩ jev ∩ run_name — the same cohort the
            cascade/head-to-head comparison uses, not the judge's own (possibly larger) block."""
            preds_llm = llms.get(run_name)
            if not preds_llm:
                return None
            sub = [e for e in test if e.id in jev and e.id in preds_llm]
            if not sub:
                return None
            v = [llm_hallucinated(preds_llm[e.id]) for e in sub]
            y = [e.hallucinated for e in sub]
            m = metrics.classification(v, y)
            return {"precision": m["precision"], "recall": m["recall"]}

        operating_points: dict[str, dict[str, float]] = {}
        for name in LLM_RUNS:
            pt = _op_point(name)
            if pt is not None:
                operating_points[name] = pt

        summary["head_to_head"] = {
            "n": len(both),
            "auroc_diff_jev_minus_claude": [diff, lo, hi],
            "claude_operating_point": {
                "precision": claude_point["precision"], "recall": claude_point["recall"]
            },
            # Point estimates on the tuned-wording run; the untuned, bootstrapped version is
            # headline.threshold_free.
            "jev_recall_at_claude_precision": metrics.recall_at_precision(curve, claude_point["precision"]),
            "jev_precision_at_claude_recall": metrics.precision_at_recall(curve, claude_point["recall"]),
            "jev_pr_curve": curve[:: max(1, len(curve) // 200)],  # ≤ ~200 points for charts
            "operating_points": operating_points,
        }

        claude_cost = {e.id: claude[e.id].cost_usd for e in both}
        jev_cost = {e.id: jev[e.id].cost_usd for e in both}

        def cascade_point(band: tuple[float, float]) -> dict[str, Any]:
            preds, escalated = metrics.cascade(p_sup, c_halluc, threshold=frozen.jev_threshold, band=band)
            cost = sum(jev_cost.values()) + sum(
                claude_cost[e.id] for e, esc in zip(both, escalated, strict=True) if esc
            )
            return {
                "band": list(band),
                "escalation_rate": sum(escalated) / len(escalated),
                "cost_usd_per_1k": 1000 * cost / len(both),
                **metrics.classification(preds, labels),
            }

        summary["cascade"] = {
            "n": len(both),
            "frozen_band": cascade_point(frozen.cascade_band),
            "sweep": [cascade_point(tuple(b)) for b in frozen.cascade_band_sweep],  # type: ignore[arg-type]
        }

        j_halluc = [jev_halluc(jev[e.id]) for e in both]
        agree = [i for i in range(len(both)) if j_halluc[i] == c_halluc[i]]
        disagree = [i for i in range(len(both)) if j_halluc[i] != c_halluc[i]]
        agree_m = metrics.classification([j_halluc[i] for i in agree], [labels[i] for i in agree])
        tp, fp, tn = int(agree_m["tp"]), int(agree_m["fp"]), int(agree_m["tn"])
        summary["agreement"] = {
            "both_agree": {  # H10; Wilson CIs as the brief specifies (not source-clustered)
                "n": len(agree),
                **agree_m,
                "precision_ci95": list(metrics.wilson_ci(tp, tp + fp)),
                "accuracy_ci95": list(metrics.wilson_ci(tp + tn, len(agree))),
            },
            "disagree": {
                "n": len(disagree),
                "jev_correct": sum(j_halluc[i] == labels[i] for i in disagree),
                "claude_correct": sum(c_halluc[i] == labels[i] for i in disagree),
            },
        }
    else:
        summary["head_to_head"] = None
        summary["cascade"] = None
        summary["agreement"] = None

    summary["paraphrases"] = {}
    for pid in PARAPHRASES:
        preds_p = _ok(store, f"jev-para-{pid}")
        if not preds_p:
            continue
        exs = [e for e in test if e.id in preds_p]
        labels_p = [e.hallucinated for e in exs]
        h_scores_p = [1 - _score(preds_p[e.id]) for e in exs]

        def auroc_on(
            idx: Sequence[int], s: list[float] = h_scores_p, y: list[bool] = labels_p
        ) -> float:
            return metrics.auroc([s[i] for i in idx], [y[i] for i in idx])

        summary["paraphrases"][pid] = {
            "question": PARAPHRASES[pid][0],
            "n": len(exs),
            "auroc": _ci(auroc_on, _groups(exs), frozen),
            "f1_at_frozen_threshold": metrics.classification(
                [jev_halluc(preds_p[e.id]) for e in exs], labels_p,
            )["f1"],
        }

    summary["flip"] = {}
    for run, rule in (("jev-flip", jev_halluc), ("claude-flip", llm_hallucinated)):
        per_example: dict[str, list[bool]] = {}
        scored: list[Prediction] = []
        for (eid, _rep), p in store.load(run).items():
            if p.score is not None:
                per_example.setdefault(eid, []).append(rule(p))
                scored.append(p)
        complete = [v for v in per_example.values() if len(v) == 3]
        if complete:
            n_flipped = sum(len(set(v)) > 1 for v in complete)
            summary["flip"][run] = {
                "n": len(complete),
                "n_flipped": n_flipped,
                "flip_rate": metrics.flip_rate(complete),
                "flip_ci95": list(metrics.wilson_ci(n_flipped, len(complete))),
                **_cache_hit_check(scored),
            }

    thinking = _ok(store, "claude-thinking")
    if thinking and claude:
        exs = [e for e in test if e.id in thinking and e.id in claude]
        t_scores = [_score(thinking[e.id]) for e in exs]
        o_scores = [_score(claude[e.id]) for e in exs]
        y = [e.hallucinated for e in exs]
        groups_t = _groups(exs)

        def thinking_auroc(
            idx: Sequence[int], s: list[float] = t_scores, yy: list[bool] = y
        ) -> float:
            return metrics.auroc([1 - s[i] for i in idx], [yy[i] for i in idx])

        def no_thinking_auroc(
            idx: Sequence[int], s: list[float] = o_scores, yy: list[bool] = y
        ) -> float:
            return metrics.auroc([1 - s[i] for i in idx], [yy[i] for i in idx])

        diff_t, lo_t, hi_t = metrics.paired_bootstrap_diff(
            thinking_auroc, no_thinking_auroc, len(exs), groups=groups_t,
            n_resamples=frozen.bootstrap_resamples, seed=frozen.bootstrap_seed,
        )
        summary["thinking"] = {
            "n": len(exs),
            "auroc_thinking": _ci(thinking_auroc, groups_t, frozen),
            "auroc_no_thinking": _ci(no_thinking_auroc, groups_t, frozen),
            "auroc_diff_thinking_minus_no_thinking": [diff_t, lo_t, hi_t],
            "cost_per_1k_thinking": 1000 * sum(thinking[e.id].cost_usd for e in exs) / len(exs),
            "cost_per_1k_no_thinking": 1000 * sum(claude[e.id].cost_usd for e in exs) / len(exs),
            "verdict_inconsistency_rate_thinking": (
                sum(verdict_inconsistent(thinking[e.id]) for e in exs) / len(exs)
            ),
        }
    else:
        summary["thinking"] = None

    live = _ok(store, "claude-live")
    if live:
        lat = [p.latency_s for p in live.values() if p.latency_s is not None]
        summary["claude_live_latency_s"] = {"n": len(lat), **_percentile_50_95(lat)}
    return summary
