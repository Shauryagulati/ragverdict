"""Metric functions for judge benchmarks. Pure Python so every line is explainable;
`scripts/verify_bench.py` independently recomputes the headline numbers with scikit-learn.

Conventions: the positive class is "hallucinated". `p_supported` is a judge's
probability/score that the response is fully supported (higher = more faithful).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from itertools import pairwise


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under ROC via the Mann-Whitney rank formula (ties get average ranks)."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes present")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    rank_sum_pos = sum(r for r, y in zip(ranks, labels, strict=True) if y)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def classification(preds: Sequence[bool], labels: Sequence[bool]) -> dict[str, float]:
    tp = sum(p and y for p, y in zip(preds, labels, strict=True))
    fp = sum(p and not y for p, y in zip(preds, labels, strict=True))
    fn = sum((not p) and y for p, y in zip(preds, labels, strict=True))
    tn = sum((not p) and (not y) for p, y in zip(preds, labels, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # F1 of the negative ("clean") class, for macro-F1
    neg_precision = tn / (tn + fn) if tn + fn else 0.0
    neg_recall = tn / (tn + fp) if tn + fp else 0.0
    neg_f1 = (
        2 * neg_precision * neg_recall / (neg_precision + neg_recall)
        if neg_precision + neg_recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": (f1 + neg_f1) / 2,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def reliability(
    probs: Sequence[float], outcomes: Sequence[bool], n_bins: int = 10
) -> list[dict[str, float]]:
    """Equal-width bins over [0, 1]; p == 1.0 lands in the last bin. Empty bins omitted."""
    buckets: list[list[int]] = [[] for _ in range(n_bins)]
    for i, p in enumerate(probs):
        buckets[min(int(p * n_bins), n_bins - 1)].append(i)
    out: list[dict[str, float]] = []
    for b, idx in enumerate(buckets):
        if not idx:
            continue
        out.append(
            {
                "lo": b / n_bins,
                "hi": (b + 1) / n_bins,
                "n": len(idx),
                "mean_prob": sum(probs[i] for i in idx) / len(idx),
                "frac_positive": sum(outcomes[i] for i in idx) / len(idx),
            }
        )
    return out


def ece(probs: Sequence[float], outcomes: Sequence[bool], n_bins: int = 10) -> float:
    """Expected calibration error: sample-weighted |mean predicted - observed rate|."""
    total = len(probs)
    return sum(
        b["n"] / total * abs(b["mean_prob"] - b["frac_positive"])
        for b in reliability(probs, outcomes, n_bins)
    )


def _resample_values(
    metric: Callable[[Sequence[int]], float],
    n: int,
    n_resamples: int,
    seed: int,
    groups: Sequence[str] | None = None,
) -> list[float]:
    rng = random.Random(seed)
    members: dict[str, list[int]] = {}
    if groups is not None:
        for i, g in enumerate(groups):
            members.setdefault(g, []).append(i)
    keys = sorted(members)
    values: list[float] = []
    for _ in range(n_resamples):
        if groups is None:
            idx = [rng.randrange(n) for _ in range(n)]
        else:  # cluster bootstrap: resample whole groups, keep their members together
            idx = [i for _ in keys for i in members[keys[rng.randrange(len(keys))]]]
        try:
            values.append(metric(idx))
        except ValueError:  # metric undefined on this resample (e.g. one class only)
            continue
    if not values:
        raise ValueError("metric undefined on every bootstrap resample")
    return sorted(values)


def _percentile(sorted_values: list[float], q: float) -> float:
    k = (len(sorted_values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def bootstrap_ci(
    metric: Callable[[Sequence[int]], float],
    n: int,
    *,
    groups: Sequence[str] | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI. `metric` receives a list of example indices."""
    values = _resample_values(metric, n, n_resamples, seed, groups)
    return _percentile(values, alpha / 2), _percentile(values, 1 - alpha / 2)


def paired_bootstrap_diff(
    metric_a: Callable[[Sequence[int]], float],
    metric_b: Callable[[Sequence[int]], float],
    n: int,
    *,
    groups: Sequence[str] | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(a - b on all examples, CI lo, CI hi), resampling the same indices for both."""
    all_idx = list(range(n))
    diff = metric_a(all_idx) - metric_b(all_idx)
    values = _resample_values(
        lambda idx: metric_a(idx) - metric_b(idx), n, n_resamples, seed, groups
    )
    return diff, _percentile(values, alpha / 2), _percentile(values, 1 - alpha / 2)


def flip_rate(verdicts_per_example: Sequence[Sequence[bool]]) -> float:
    """Fraction of examples whose verdict is not identical across repeats."""
    flipped = sum(len(set(v)) > 1 for v in verdicts_per_example)
    return flipped / len(verdicts_per_example)


def cascade(
    p_supported: Sequence[float],
    fallback_hallucinated: Sequence[bool],
    *,
    threshold: float,
    band: tuple[float, float],
) -> tuple[list[bool], list[bool]]:
    """Offline replay of CascadeJudge: primary decides unless lo < p < hi."""
    lo, hi = band
    preds: list[bool] = []
    escalated: list[bool] = []
    for p, fallback in zip(p_supported, fallback_hallucinated, strict=True):
        esc = lo < p < hi
        escalated.append(esc)
        preds.append(fallback if esc else p < threshold)
    return preds, escalated


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; behaves well for small n and 0%/100%."""
    if n == 0:
        return 0.0, 1.0
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _thresholds(p_supported: Sequence[float]) -> list[float]:
    values = sorted(set(p_supported))
    return [(a + b) / 2 for a, b in pairwise(values)] + [values[-1] + 1e-9]


def pr_curve(p_supported: Sequence[float], labels: Sequence[bool]) -> list[dict[str, float]]:
    """Precision/recall for 'hallucinated iff p < t' at every distinct threshold."""
    out: list[dict[str, float]] = []
    for t in _thresholds(p_supported):
        m = classification([p < t for p in p_supported], labels)
        out.append({"threshold": t, "precision": m["precision"], "recall": m["recall"]})
    return out


def recall_at_precision(curve: list[dict[str, float]], target: float) -> float | None:
    ok = [c["recall"] for c in curve if c["precision"] >= target]
    return max(ok) if ok else None


def precision_at_recall(curve: list[dict[str, float]], target: float) -> float | None:
    ok = [c["precision"] for c in curve if c["recall"] >= target]
    return max(ok) if ok else None


def best_f1_threshold(
    p_supported: Sequence[float], labels: Sequence[bool]
) -> tuple[float, float]:
    """Threshold t maximizing F1 for 'hallucinated iff p < t' (midpoints between scores)."""
    best = (0.5, -1.0)
    for t in _thresholds(p_supported):
        f1 = classification([p < t for p in p_supported], labels)["f1"]
        if f1 > best[1]:
            best = (t, f1)
    return best
