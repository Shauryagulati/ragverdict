"""Independently recompute headline benchmark numbers from raw files with scikit-learn.

Deliberately imports nothing from ragverdict.bench: it re-parses RAGTruth, re-applies
the label and filter rules, and recomputes metrics with sklearn/numpy. Any
disagreement with summary.json beyond 1e-3 fails the check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

TOL = 1e-3


def load_labels(data: Path) -> dict[str, bool]:
    labels = {}
    with (data / "response.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["split"] == "test" and row["quality"] == "good":
                labels[str(row["id"])] = any(not s.get("implicit_true") for s in row["labels"])
    return labels


def load_scores(out: Path, run: str) -> dict[str, tuple[float, float]]:
    rows: dict[str, tuple[float, float]] = {}
    path = out / "raw" / f"{run}.jsonl"
    with path.open() as fh:
        for line in fh:
            p = json.loads(line)
            if p["repeat"] == 0:
                if p["score"] is None:
                    rows.pop(p["example_id"], None)
                else:
                    rows[p["example_id"]] = (float(p["score"]), float(p["cost_usd"]))
    return rows


def ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(probs[mask].mean() - outcomes[mask].mean())
    return float(total)


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

    rules = {"jev": lambda s: s < frozen["jev_threshold"], "claude": lambda s: s < 1.0,
             "deepseek": lambda s: s < 1.0, "glm": lambda s: s < 1.0}
    for run, rule in rules.items():
        if run not in summary["judges"]:
            continue
        scores = load_scores(args.out, run)
        ids = [i for i in labels if i in scores]
        y = np.array([labels[i] for i in ids])
        s = np.array([scores[i][0] for i in ids])
        block = summary["judges"][run]
        checks.append((f"{run}.n", float(len(ids)), float(block["n"])))
        checks.append((f"{run}.auroc", float(roc_auc_score(y, 1 - s)), block["auroc"]["value"]))
        preds = np.array([rule(v) for v in s])
        checks.append((f"{run}.f1", float(f1_score(y, preds, zero_division=0)), block["f1"]["value"]))
        checks.append((f"{run}.macro_f1", float(f1_score(y, preds, average="macro", zero_division=0)),
                       block["macro_f1"]["value"]))
        checks.append((f"{run}.cost_total", float(sum(scores[i][1] for i in ids)), block["cost_usd_total"]))
        if run == "jev":
            checks.append((f"{run}.ece", ece(1 - s, y.astype(float)), block["ece"]))

    mismatches = [(name, mine, theirs) for name, mine, theirs in checks if abs(mine - theirs) > TOL]
    for name, mine, theirs in checks:
        flag = "MISMATCH" if (name, mine, theirs) in mismatches else "ok"
        print(f"{flag:8} {name:20} independent={mine:.6f} summary={theirs:.6f}")
    if mismatches:
        return 1
    print("VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
