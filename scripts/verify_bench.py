"""Independently recompute headline benchmark numbers from raw files with scikit-learn.

Deliberately imports nothing from ragverdict.bench: it re-parses RAGTruth, re-applies
the label and filter rules and the pre-registered verdict rules, and recomputes metrics
with sklearn/numpy. Any disagreement with summary.json beyond 1e-3 fails the check.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

TOL = 1e-3
LLM_RUNS = ("claude", "deepseek", "glm")
Row = dict[str, float]


def load_labels(data: Path) -> dict[str, bool]:
    labels = {}
    with (data / "response.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["split"] == "test" and row["quality"] == "good":
                labels[str(row["id"])] = any(not s.get("implicit_true") for s in row["labels"])
    return labels


def load_scores(out: Path, run: str) -> dict[str, Row]:
    """Last repeat-0 row per example; a failed (score null) last row removes the example."""
    rows: dict[str, Row] = {}
    path = out / "raw" / f"{run}.jsonl"
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            p = json.loads(line)
            if p["repeat"] == 0:
                if p["score"] is None:
                    rows.pop(p["example_id"], None)
                else:
                    rows[p["example_id"]] = {
                        "score": float(p["score"]),
                        "cost": float(p["cost_usd"]),
                        "supported": float(p.get("supported_claims", 0)),
                        "total": float(p.get("total_claims", 0)),
                    }
    return rows


def llm_flags(row: Row) -> bool:
    """Pre-registered LLM verdict: hallucinated iff score < 1 or a claim is unsupported."""
    return row["score"] < 1.0 or (row["total"] > 0 and row["supported"] < row["total"])


def llm_inconsistent(row: Row) -> bool:
    return (row["score"] < 1.0) != (row["total"] > 0 and row["supported"] < row["total"])


def jev_flags(threshold: float) -> Callable[[Row], bool]:
    return lambda row: row["score"] < threshold


def ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(probs[mask].mean() - outcomes[mask].mean())
    return float(total)


def f1s(y: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    return (float(f1_score(y, preds, zero_division=0)),
            float(f1_score(y, preds, average="macro", zero_division=0)))


def h1_verdict(lo: float, hi: float, delta: float = 0.03) -> str:
    """TOST on the 90% CI of F1(Jev untuned) - F1(Claude)."""
    if -delta <= lo and hi <= delta:
        return "equivalent"
    if lo > delta:
        return "jev_better"
    if hi < -delta:
        return "jev_worse"
    return "inconclusive"


def h9_verdict(lo: float, hi: float, margin: float = 0.02) -> str:
    """H9 predicts the cascade beats the best single judge by < margin F1 (95% CI)."""
    if hi < margin:
        return "confirmed"
    if lo >= margin:
        return "refuted"
    return "inconclusive"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.out / "summary.json").read_text())
    frozen = json.loads(args.frozen.read_text())
    labels = load_labels(args.data)
    checks: list[tuple[str, float, float]] = []

    # summary judge name -> (raw run, verdict rule, probabilistic)
    judges: dict[str, tuple[str, Callable[[Row], bool], bool]] = {
        "jev": ("jev", jev_flags(frozen["jev_threshold"]), True),
        "jev_untuned": ("jev-para-A", jev_flags(0.5), True),
        **{run: (run, llm_flags, False) for run in LLM_RUNS},
    }
    for name, (run, rule, probabilistic) in judges.items():
        if name not in summary["judges"]:
            continue
        scores = load_scores(args.out, run)
        ids = [i for i in labels if i in scores]
        y = np.array([labels[i] for i in ids])
        s = np.array([scores[i]["score"] for i in ids])
        block = summary["judges"][name]
        checks.append((f"{name}.n", float(len(ids)), float(block["n"])))
        checks.append((f"{name}.auroc", float(roc_auc_score(y, 1 - s)), block["auroc"]["value"]))
        f1, macro = f1s(y, np.array([rule(scores[i]) for i in ids]))
        checks.append((f"{name}.f1", f1, block["f1"]["value"]))
        checks.append((f"{name}.macro_f1", macro, block["macro_f1"]["value"]))
        checks.append((f"{name}.cost_total", float(sum(scores[i]["cost"] for i in ids)),
                       block["cost_usd_total"]))
        if probabilistic:
            checks.append((f"{name}.ece", ece(1 - s, y.astype(float)), block["ece"]))
        else:
            rate = float(np.mean([llm_inconsistent(scores[i]) for i in ids]))
            checks.append((f"{name}.inconsistency", rate, block["verdict_inconsistency_rate"]))

    labeled: list[tuple[str, str, str]] = []  # (check, independent, summary) for verdicts
    headline: dict[str, Any] | None = summary.get("headline")
    if headline is not None:
        # Headline cohort: good test examples scored by jev-para-A, by run "jev" when it has
        # data, and by every LLM run with data.
        para_a = load_scores(args.out, "jev-para-A")
        tuned = load_scores(args.out, "jev")
        llms = {run: sc for run in LLM_RUNS if (sc := load_scores(args.out, run))}
        others = [*([tuned] if tuned else []), *llms.values()]
        cohort = [i for i in labels if i in para_a and all(i in sc for sc in others)]
        y = np.array([labels[i] for i in cohort])
        checks.append(("headline.n", float(len(cohort)), float(headline["cohort"]["n"])))
        verdicts = {"jev_untuned": np.array([para_a[i]["score"] < 0.5 for i in cohort])}
        verdicts |= {run: np.array([llm_flags(sc[i]) for i in cohort]) for run, sc in llms.items()}
        for name, preds in verdicts.items():
            f1, macro = f1s(y, preds)
            block = headline["judges"][name]
            checks.append((f"headline.{name}.f1", f1, block["f1"]["value"]))
            checks.append((f"headline.{name}.macro_f1", macro, block["macro_f1"]["value"]))
        if "claude" in verdicts:
            h1 = headline["h1"]
            diff = f1s(y, verdicts["jev_untuned"])[0] - f1s(y, verdicts["claude"])[0]
            checks.append(("headline.h1_diff", diff, h1["diff"]))
            labeled.append(("headline.h1_verdict", h1_verdict(*h1["ci90"]), h1["verdict"]))

        h9 = summary.get("hypotheses", {}).get("H9")
        if h9 is not None:
            # Cascade replay: Jev (tuned) decides unless lo < p < hi, then Claude decides.
            p_t = np.array([tuned[i]["score"] for i in cohort])
            lo_band, hi_band = frozen["cascade_band"]
            escalate = (p_t > lo_band) & (p_t < hi_band)
            cascade = np.where(escalate, verdicts["claude"], p_t < frozen["jev_threshold"])
            singles = {"jev_tuned": p_t < frozen["jev_threshold"],
                       "jev_untuned": verdicts["jev_untuned"], "claude": verdicts["claude"]}
            single_f1 = {name: f1s(y, v)[0] for name, v in singles.items()}
            top = max(single_f1.values())
            best = next(name for name, f in single_f1.items() if f >= top - 1e-12)  # first wins
            cascade_f1 = f1s(y, cascade)[0]
            checks.append(("h9.n", float(len(cohort)), float(h9["n"])))
            checks.append(("h9.cascade_f1", cascade_f1, h9["cascade_f1"]))
            for name, f in single_f1.items():
                checks.append((f"h9.single.{name}", f, h9["single_judge_f1"][name]))
            checks.append(("h9.diff", cascade_f1 - single_f1[best], h9["diff"]))
            labeled.append(("h9.best_single_judge", best, h9["best_single_judge"]))
            labeled.append(("h9.verdict", h9_verdict(*h9["ci95"]), h9["verdict"]))

    mismatches = [(name, mine, theirs) for name, mine, theirs in checks if abs(mine - theirs) > TOL]
    for name, mine, theirs in checks:
        flag = "MISMATCH" if (name, mine, theirs) in mismatches else "ok"
        print(f"{flag:8} {name:28} independent={mine:.6f} summary={theirs:.6f}")
    label_mismatches = [c for c in labeled if c[1] != c[2]]
    for name, mine_s, theirs_s in labeled:
        flag = "MISMATCH" if mine_s != theirs_s else "ok"
        print(f"{flag:8} {name:28} independent={mine_s} summary={theirs_s}")
    if mismatches or label_mismatches:
        return 1
    print("VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
