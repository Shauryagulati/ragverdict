"""Metric functions checked against hand-computed values."""

from __future__ import annotations

from collections import Counter

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
    threshold_at_fpr,
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


@pytest.mark.parametrize("bad", [-0.2, 1.3, float("nan")])
def test_reliability_rejects_out_of_range_probs(bad: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        reliability([bad], [T])


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
    # Unequal group sizes so row-resampling (which always yields n=11) can't accidentally
    # pass: "a" x1, "b" x5, "c" x2, "d" x3.
    groups = ["a"] + ["b"] * 5 + ["c"] * 2 + ["d"] * 3
    members = {"a": [0], "b": [1, 2, 3, 4, 5], "c": [6, 7], "d": [8, 9, 10]}
    seen_idx: list[list[int]] = []

    def metric(idx: list[int]) -> float:
        seen_idx.append(list(idx))
        return 0.0

    bootstrap_ci(metric, len(groups), groups=groups, n_resamples=50)

    assert len(seen_idx) == 50
    for idx in seen_idx:
        counts = Counter(idx)
        total_draws = 0
        for name, member_idx in members.items():
            # every member of a drawn group must occur the same number of times
            occurrences = {counts.get(i, 0) for i in member_idx}
            assert len(occurrences) == 1, f"group {name} split across a resample: {idx}"
            total_draws += occurrences.pop()
        assert total_draws == 4  # one draw per of the 4 distinct groups, every resample


def test_group_bootstrap_is_wider_for_correlated_rows() -> None:
    # 20 groups x 5 identical rows: row resampling pretends n=100, groups know n=20
    values = [float(g % 2) for g in range(20) for _ in range(5)]
    groups = [str(g) for g in range(20) for _ in range(5)]
    mean = lambda idx: sum(values[i] for i in idx) / len(idx)  # noqa: E731
    row_lo, row_hi = bootstrap_ci(mean, 100, seed=1)
    grp_lo, grp_hi = bootstrap_ci(mean, 100, groups=groups, seed=1)
    assert (grp_hi - grp_lo) > (row_hi - row_lo)


def test_group_bootstrap_validates_groups_length() -> None:
    with pytest.raises(ValueError, match="groups has"):
        bootstrap_ci(lambda idx: 0.0, 5, groups=["a", "b"])


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


def test_pr_curve_with_ties_matches_brute_force() -> None:
    # Tied scores must move together: at t=0.4 both 0.3s (one T, one F) are flagged.
    p = [0.3, 0.3, 0.1, 0.7, 0.7, 0.9]
    y = [T, F, T, F, T, F]
    curve = pr_curve(p, y)
    assert [round(c["threshold"], 9) for c in curve] == [0.2, 0.5, 0.8, 0.900000001]
    # t=0.2 flags {0.1}: tp=1 fp=0; t=0.5 flags {0.1,0.3,0.3}: tp=2 fp=1;
    # t=0.8 adds {0.7,0.7}: tp=3 fp=2; last flags everything: tp=3 fp=3.
    assert [(c["precision"], c["recall"]) for c in curve] == [
        (1.0, 1 / 3), (2 / 3, 2 / 3), (3 / 5, 1.0), (0.5, 1.0),
    ]
    for c in curve:  # identical to classification() at every threshold
        m = classification([v < c["threshold"] for v in p], y)
        assert (c["precision"], c["recall"]) == (m["precision"], m["recall"])


def test_pr_curve_without_positives_has_zero_recall() -> None:
    assert [c["recall"] for c in pr_curve([0.2, 0.8], [F, F])] == [0.0, 0.0]


def test_threshold_at_fpr_hand_computed() -> None:
    # sorted: .1T .2F .3T .5F .6T .7F .8T .9F; midpoints .15 .25 .4 .55 .65 .75 .85, then .9+1e-9
    # FPR (4 negatives) at each:            0   1/4 1/4 2/4 2/4 3/4 3/4, 1
    p = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    y = [T, F, T, F, T, F, T, F]
    assert threshold_at_fpr(p, y, 0.25) == pytest.approx(0.4)  # largest t with FPR <= 1/4
    assert threshold_at_fpr(p, y, 0.0) == pytest.approx(0.15)
    assert threshold_at_fpr(p, y, 0.6) == pytest.approx(0.65)
    assert threshold_at_fpr(p, y, 1.0) == pytest.approx(0.9 + 1e-9)


def test_threshold_at_fpr_none_when_even_lowest_threshold_exceeds() -> None:
    # the lowest score is a negative, so every threshold flags >= 1 of 2 negatives (FPR >= 1/2)
    assert threshold_at_fpr([0.1, 0.5, 0.9], [F, T, F], 0.25) is None


def test_threshold_at_fpr_needs_negatives() -> None:
    with pytest.raises(ValueError):
        threshold_at_fpr([0.1, 0.5], [T, T], 0.1)


def test_best_f1_threshold() -> None:
    threshold, f1 = best_f1_threshold([0.1, 0.2, 0.8, 0.9], [T, T, F, F])
    assert f1 == pytest.approx(1.0)
    assert 0.2 < threshold <= 0.8
