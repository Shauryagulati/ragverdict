"""Turn cached predictions into summary.json — the single source for every published number."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from typing import Any

from ragverdict.bench import metrics
from ragverdict.bench.predict import Prediction, PredictionStore
from ragverdict.bench.ragtruth import Example
from ragverdict.bench.runs import MODELS, PARAPHRASES, PRICES_CHECKED, FrozenConfig


def _ok(store: PredictionStore, run: str) -> dict[str, Prediction]:
    """Successful repeat-0 predictions for `run`, keyed by example id."""
    return {
        eid: p for (eid, rep), p in store.load(run).items() if rep == 0 and p.score is not None
    }


def _ci(
    fn: Callable[[Sequence[int]], float], groups: list[str], frozen: FrozenConfig
) -> dict[str, Any]:
    n = len(groups)
    value = fn(list(range(n)))
    lo, hi = metrics.bootstrap_ci(
        fn, n, groups=groups, n_resamples=frozen.bootstrap_resamples, seed=frozen.bootstrap_seed
    )
    return {"value": value, "ci95": [lo, hi]}


def _judge_block(
    examples: list[Example],
    preds: dict[str, Prediction],
    halluc_pred: Callable[[float], bool],
    frozen: FrozenConfig,
    run_rows: int,
    *,
    probabilistic: bool,
) -> dict[str, Any]:
    exs = [e for e in examples if e.id in preds]
    labels = [e.hallucinated for e in exs]
    groups = [e.source_id or e.id for e in exs]
    h_scores = [1.0 - float(preds[e.id].score) for e in exs]  # type: ignore[arg-type]
    verdicts = [halluc_pred(float(preds[e.id].score)) for e in exs]  # type: ignore[arg-type]

    def auroc_on(idx: Sequence[int]) -> float:
        return metrics.auroc([h_scores[i] for i in idx], [labels[i] for i in idx])

    def f1_on(idx: Sequence[int]) -> float:
        return metrics.classification([verdicts[i] for i in idx], [labels[i] for i in idx])["f1"]

    def macro_f1_on(idx: Sequence[int]) -> float:
        return metrics.classification(
            [verdicts[i] for i in idx], [labels[i] for i in idx]
        )["macro_f1"]

    block: dict[str, Any] = {
        "n": len(exs),
        "n_errors": run_rows - len(exs),
        "auroc": _ci(auroc_on, groups, frozen),
        "f1": _ci(f1_on, groups, frozen),
        "macro_f1": _ci(macro_f1_on, groups, frozen),
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
    latencies = sorted(p.latency_s for p in preds.values() if p.latency_s is not None)
    if latencies:
        block["latency_s"] = {
            "p50": statistics.median(latencies),
            "p95": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
        }
    if probabilistic:
        block["ece"] = metrics.ece(h_scores, labels)
        block["reliability"] = metrics.reliability(h_scores, labels)
    block["by_task"] = _slices(exs, labels, h_scores, verdicts, frozen, key=lambda e: e.task)
    block["by_generator"] = _slices(
        exs, labels, h_scores, verdicts, frozen, key=lambda e: e.generator
    )
    block["by_length_quartile"] = _slices(
        exs, labels, h_scores, verdicts, frozen, key=_quartile_key(examples)
    )
    block["recall_by_severity"] = _recall(exs, verdicts, key=lambda e: e.severity)
    block["recall_by_kind"] = _recall(exs, verdicts, key=lambda e: e.kind)
    block["recall_numeric"] = _recall(
        exs, verdicts, key=lambda e: None if not e.hallucinated else ("numeric" if e.numeric else "non_numeric")
    )
    return block


def _quartile_key(examples: list[Example]) -> Callable[[Example], str]:
    cuts = statistics.quantiles([e.n_chars for e in examples], n=4)

    def key(e: Example) -> str:
        return f"Q{1 + sum(e.n_chars > c for c in cuts)}"

    return key


def _slices(
    exs: list[Example],
    labels: list[bool],
    h_scores: list[float],
    verdicts: list[bool],
    frozen: FrozenConfig,
    *,
    key: Callable[[Example], str],
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for i, e in enumerate(exs):
        groups.setdefault(key(e), []).append(i)
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
    jev_rule: Callable[[float], bool],
    other_rule: Callable[[float], bool],
    frozen: FrozenConfig,
) -> dict[str, Any]:
    """Jev minus comparator on the examples both scored, source-grouped paired bootstrap."""
    both = [e for e in test if e.id in jev and e.id in other]
    y = [e.hallucinated for e in both]
    jv = [jev_rule(float(jev[e.id].score)) for e in both]  # type: ignore[arg-type]
    ov = [other_rule(float(other[e.id].score)) for e in both]  # type: ignore[arg-type]
    groups = [e.source_id or e.id for e in both]
    out: dict[str, Any] = {"n": len(both)}
    for key in ("macro_f1", "f1"):
        def a(idx: Sequence[int], k: str = key) -> float:
            return metrics.classification([jv[i] for i in idx], [y[i] for i in idx])[k]

        def b(idx: Sequence[int], k: str = key) -> float:
            return metrics.classification([ov[i] for i in idx], [y[i] for i in idx])[k]

        out[f"{key}_diff"] = list(metrics.paired_bootstrap_diff(
            a, b, len(both), groups=groups,
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
    jev_rule: Callable[[float], bool],
    llm_rule: Callable[[float], bool],
) -> dict[str, Any]:
    """QA rows under the pilot's cohort rules: all 900 QA test responses (no quality filter),
    hallucinated = any span incl. implicit_true. Unscored responses are reported as invalid."""
    qa = [e for e in everything if e.task == "QA"]
    rows: dict[str, Any] = {"n": len(qa), "n_positive": sum(e.hallucinated_any_span for e in qa)}
    for run, rule in (("jev", jev_rule), ("claude", llm_rule), ("deepseek", llm_rule),
                      ("glm", llm_rule)):
        preds = _ok(store, run)
        if not preds:
            continue
        scored = [e for e in qa if e.id in preds]
        m = metrics.classification(
            [rule(float(preds[e.id].score)) for e in scored],  # type: ignore[arg-type]
            [e.hallucinated_any_span for e in scored],
        )
        rows[run] = {"n_scored": len(scored), "invalid": len(qa) - len(scored),
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "missed": m["fn"], "false_alarms": m["fp"]}
    return rows


def _wall_clock(store: PredictionStore) -> dict[str, float]:
    """Wall-clock seconds per run, written by the CLI to <out>/meta/<run>.json."""
    meta_dir = store.root / "meta"
    if not meta_dir.exists():
        return {}
    return {
        p.stem: float(json.loads(p.read_text())["wall_clock_s"]) for p in sorted(meta_dir.glob("*.json"))
    }


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

    def jev_halluc(p: float) -> bool:
        return p < frozen.jev_threshold

    def claude_halluc(s: float) -> bool:
        return s < 1.0

    summary: dict[str, Any] = {
        "frozen_config": frozen.model_dump(),
        "frozen_config_sha256": frozen_sha,
        "n_test": len(test),
        "n_test_hallucinated": sum(e.hallucinated for e in test),
        "judges": {
            "jev": _judge_block(test, jev, jev_halluc, frozen, len(test), probabilistic=True),
            "claude": _judge_block(test, claude, claude_halluc, frozen, len(test), probabilistic=False),
            # Fairness check: Jev at the untuned default threshold (Claude's rule is untuned too).
            "jev_untuned": _judge_block(
                test, jev, lambda p: p < 0.5, frozen, len(test), probabilistic=True
            ),
        },
        "wall_clock_s": _wall_clock(store),
    }

    for run in ("deepseek", "glm"):
        preds_llm = _ok(store, run)
        if preds_llm:
            summary["judges"][run] = _judge_block(
                test, preds_llm, claude_halluc, frozen, len(test), probabilistic=False
            )

    all_labels = [e.hallucinated for e in test]
    summary["baselines"] = {
        "always_clean": metrics.classification([False] * len(test), all_labels),
    }
    summary["paired_vs_jev"] = {
        run: _paired_vs_jev(test, jev, _ok(store, run), jev_halluc, claude_halluc, frozen)
        for run in ("claude", "deepseek", "glm")
        if _ok(store, run)
    }
    summary["qa_replication"] = _qa_rows(summary["judges"])
    summary["qa_replication_pilot_rules"] = _qa_pilot_rules(
        everything, store, jev_halluc, claude_halluc
    )
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

    # Head-to-head, cascade, agreement: only examples both judges scored.
    both = [e for e in test if e.id in jev and e.id in claude]
    labels = [e.hallucinated for e in both]
    p_sup = [float(jev[e.id].score) for e in both]  # type: ignore[arg-type]
    c_score = [float(claude[e.id].score) for e in both]  # type: ignore[arg-type]
    c_halluc = [claude_halluc(s) for s in c_score]

    def jev_auroc(idx: Sequence[int]) -> float:
        return metrics.auroc([1 - p_sup[i] for i in idx], [labels[i] for i in idx])

    def claude_auroc(idx: Sequence[int]) -> float:
        return metrics.auroc([1 - c_score[i] for i in idx], [labels[i] for i in idx])

    diff, lo, hi = metrics.paired_bootstrap_diff(
        jev_auroc, claude_auroc, len(both), groups=[e.source_id or e.id for e in both],
        n_resamples=frozen.bootstrap_resamples, seed=frozen.bootstrap_seed,
    )
    curve = metrics.pr_curve(p_sup, labels)
    claude_point = metrics.classification(c_halluc, labels)
    summary["head_to_head"] = {
        "n": len(both),
        "auroc_diff_jev_minus_claude": [diff, lo, hi],
        "claude_operating_point": {
            "precision": claude_point["precision"], "recall": claude_point["recall"]
        },
        # Fairest single comparison: Jev tuned to Claude's precision (or recall).
        "jev_recall_at_claude_precision": metrics.recall_at_precision(curve, claude_point["precision"]),
        "jev_precision_at_claude_recall": metrics.precision_at_recall(curve, claude_point["recall"]),
        "jev_pr_curve": curve[:: max(1, len(curve) // 200)],  # ≤ ~200 points for charts
        "operating_points": {
            name: {
                "precision": block["classification"]["precision"],
                "recall": block["classification"]["recall"],
            }
            for name, block in summary["judges"].items()
            if name in ("claude", "deepseek", "glm")
        },
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

    j_halluc = [jev_halluc(p) for p in p_sup]
    agree = [i for i in range(len(both)) if j_halluc[i] == c_halluc[i]]
    disagree = [i for i in range(len(both)) if j_halluc[i] != c_halluc[i]]
    summary["agreement"] = {
        "both_agree": {
            "n": len(agree),
            **metrics.classification([j_halluc[i] for i in agree], [labels[i] for i in agree]),
        },
        "disagree": {
            "n": len(disagree),
            "jev_correct": sum(j_halluc[i] == labels[i] for i in disagree),
            "claude_correct": sum(c_halluc[i] == labels[i] for i in disagree),
        },
    }

    summary["paraphrases"] = {}
    for pid in PARAPHRASES:
        preds_p = _ok(store, f"jev-para-{pid}")
        if not preds_p:
            continue
        exs = [e for e in test if e.id in preds_p]
        summary["paraphrases"][pid] = {
            "question": PARAPHRASES[pid][0],
            "n": len(exs),
            "auroc": metrics.auroc(
                [1 - float(preds_p[e.id].score) for e in exs],  # type: ignore[arg-type]
                [e.hallucinated for e in exs],
            ),
            "f1_at_frozen_threshold": metrics.classification(
                [float(preds_p[e.id].score) < frozen.jev_threshold for e in exs],  # type: ignore[arg-type]
                [e.hallucinated for e in exs],
            )["f1"],
        }

    summary["flip"] = {}
    for run, rule in (("jev-flip", jev_halluc), ("claude-flip", claude_halluc)):
        rows = store.load(run)
        per_example: dict[str, list[bool]] = {}
        for (eid, _rep), p in rows.items():
            if p.score is not None:
                per_example.setdefault(eid, []).append(rule(p.score))
        complete = [v for v in per_example.values() if len(v) == 3]
        if complete:
            summary["flip"][run] = {"n": len(complete), "flip_rate": metrics.flip_rate(complete)}

    thinking = _ok(store, "claude-thinking")
    if thinking:
        exs = [e for e in test if e.id in thinking and e.id in claude]
        t_scores = [float(thinking[e.id].score) for e in exs]  # type: ignore[arg-type]
        o_scores = [float(claude[e.id].score) for e in exs]  # type: ignore[arg-type]
        y = [e.hallucinated for e in exs]
        summary["thinking"] = {
            "n": len(exs),
            "auroc_thinking": metrics.auroc([1 - s for s in t_scores], y),
            "auroc_no_thinking": metrics.auroc([1 - s for s in o_scores], y),
            "cost_per_1k_thinking": 1000 * sum(thinking[e.id].cost_usd for e in exs) / len(exs),
            "cost_per_1k_no_thinking": 1000 * sum(claude[e.id].cost_usd for e in exs) / len(exs),
        }
    else:
        summary["thinking"] = None

    live = _ok(store, "claude-live")
    if live:
        lat = sorted(p.latency_s for p in live.values() if p.latency_s is not None)
        summary["claude_live_latency_s"] = {
            "n": len(lat),
            "p50": statistics.median(lat),
            "p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
        }
    return summary
