"""Metric functions checked against hand-computed values."""

from __future__ import annotations

import pytest

from ragverdict.bench.metrics import (
    auroc,
    best_f1_threshold,
    bootstrap_ci,
    cascade,
    classification,
    ece,
    flip_rate,
    paired_bootstrap_diff,
    pr_curve,
    precision_at_recall,
    recall_at_precision,
    reliability,
    wilson_ci,
)

T, F = True, False


def test_auroc_hand_computed() -> None:
    # positives {0.35, 0.8} vs negatives {0.1, 0.4}: 3 of 4 pairs ranked right
    assert auroc([0.1, 0.4, 0.35, 0.8], [F, F, T, T]) == pytest.approx(0.75)


def test_auroc_ties_count_half() -> None:
    assert auroc([0.5, 0.5], [T, F]) == pytest.approx(0.5)
    assert auroc([0.2, 0.5, 0.5, 0.9], [F, F, T, T]) == pytest.approx(0.875)


def test_auroc_perfect_and_inverted() -> None:
    assert auroc([0.1, 0.9], [F, T]) == 1.0
    assert auroc([0.9, 0.1], [F, T]) == 0.0


def test_auroc_single_class_raises() -> None:
    with pytest.raises(ValueError, match="both classes"):
        auroc([0.1, 0.2], [T, T])


def test_classification_counts() -> None:
    m = classification([T, T, F, F], [T, F, T, F])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == m["accuracy"] == pytest.approx(0.5)
    assert m["macro_f1"] == pytest.approx(0.5)


def test_classification_no_positive_predictions() -> None:
    m = classification([F, F], [T, F])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    # clean class: precision 1/2, recall 1/1 -> F1 2/3; macro = (0 + 2/3) / 2
    assert m["macro_f1"] == pytest.approx(1 / 3)


def test_ece_hand_computed() -> None:
    # bin 9: mean .9, frac .5 -> .4 * 2/4 ; bin 1: mean .1, frac 0 -> .1 * 2/4  => .25
    assert ece([0.9, 0.9, 0.1, 0.1], [T, F, F, F]) == pytest.approx(0.25)


def test_reliability_puts_one_in_last_bin() -> None:
    bins = reliability([1.0, 0.0], [T, F])
    assert [(b["lo"], b["n"]) for b in bins] == [(0.0, 1), (0.9, 1)]


def test_bootstrap_ci_constant_metric() -> None:
    assert bootstrap_ci(lambda idx: 0.7, 50) == (0.7, 0.7)


def test_bootstrap_ci_brackets_mean_and_is_deterministic() -> None:
    values = [float(i % 2) for i in range(200)]
    metric = lambda idx: sum(values[i] for i in idx) / len(idx)  # noqa: E731
    lo, hi = bootstrap_ci(metric, 200, seed=3)
    assert lo < 0.5 < hi
    assert (lo, hi) == bootstrap_ci(metric, 200, seed=3)


def test_bootstrap_skips_resamples_where_metric_undefined() -> None:
    labels = [T] + [F] * 9
    scores = [0.9] + [0.1] * 9
    lo, hi = bootstrap_ci(lambda idx: auroc([scores[i] for i in idx], [labels[i] for i in idx]), 10)
    assert lo == hi == 1.0


def test_group_bootstrap_keeps_groups_together() -> None:
    # 4 groups of 3 identical rows: resampled sizes are always multiples of 3
    seen_sizes: set[int] = set()

    def metric(idx: list[int]) -> float:
        seen_sizes.add(len(idx))
        return 0.0

    bootstrap_ci(metric, 12, groups=[str(i // 3) for i in range(12)], n_resamples=50)
    assert seen_sizes == {12}


def test_group_bootstrap_is_wider_for_correlated_rows() -> None:
    # 20 groups x 5 identical rows: row resampling pretends n=100, groups know n=20
    values = [float(g % 2) for g in range(20) for _ in range(5)]
    groups = [str(g) for g in range(20) for _ in range(5)]
    mean = lambda idx: sum(values[i] for i in idx) / len(idx)  # noqa: E731
    row_lo, row_hi = bootstrap_ci(mean, 100, seed=1)
    grp_lo, grp_hi = bootstrap_ci(mean, 100, groups=groups, seed=1)
    assert (grp_hi - grp_lo) > (row_hi - row_lo)


def test_paired_bootstrap_diff() -> None:
    diff, lo, hi = paired_bootstrap_diff(lambda idx: 0.8, lambda idx: 0.6, 30)
    assert diff == pytest.approx(0.2) and lo == pytest.approx(0.2) and hi == pytest.approx(0.2)


def test_flip_rate() -> None:
    assert flip_rate([[T, T, T], [T, F, T], [F, F, F]]) == pytest.approx(1 / 3)


def test_cascade() -> None:
    preds, escalated = cascade([0.1, 0.5, 0.9], [T, F, T], threshold=0.5, band=(0.3, 0.7))
    assert preds == [T, F, F]
    assert escalated == [F, T, F]


def test_wilson_ci_hand_computed() -> None:
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_pr_curve_and_operating_points() -> None:
    # p_supported low = flagged. labels: T,T,F,F
    curve = pr_curve([0.1, 0.3, 0.6, 0.9], [T, T, F, F])
    by_t = {round(c["threshold"], 2): c for c in curve}
    assert by_t[0.2]["precision"] == 1.0 and by_t[0.2]["recall"] == 0.5
    assert by_t[0.45]["precision"] == 1.0 and by_t[0.45]["recall"] == 1.0
    assert by_t[0.75]["precision"] == pytest.approx(2 / 3)
    assert recall_at_precision(curve, 0.9) == 1.0
    assert precision_at_recall(curve, 1.0) == 1.0
    assert recall_at_precision(pr_curve([0.9, 0.1], [T, F]), 0.99) is None


def test_best_f1_threshold() -> None:
    threshold, f1 = best_f1_threshold([0.1, 0.2, 0.8, 0.9], [T, T, F, F])
    assert f1 == pytest.approx(1.0)
    assert 0.2 < threshold <= 0.8
