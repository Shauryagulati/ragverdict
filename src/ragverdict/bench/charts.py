"""PNG charts for the write-up and posts. Requires the `bench` extra (matplotlib).

Every chart reads only `summary.json` (never raw predictions) and None-guards every block
that can be missing on a partial run (`head_to_head`, `cascade`, `headline`). The F1-vs-cost
chart prefers the untuned headline cohort (Jev wording A @ 0.5 vs every LLM judge on their
common intersection); when there is no such cohort yet (only some judges have run), it falls
back to each judge's own block and says so in the title.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Published RAGTruth response-level F1 (LettuceDetect, arXiv 2502.17125, Table 2).
# Context only: different sampling/filters from this benchmark.
PUBLISHED_F1 = {
    "GPT-4-turbo prompt": 63.4,
    "Luna": 65.4,
    "LettuceDetect-large": 79.2,
    "RAG-HAT": 83.9,
}
COLORS = {
    "jev_untuned": "#2563eb",
    "jev": "#1d4ed8",
    "claude": "#d97706",
    "cascade": "#059669",
    "deepseek": "#7c3aed",
    "glm": "#db2777",
}
LABELS = {
    "jev_untuned": "Jev (untuned)",
    "jev": "Jev (tuned on train)",
    "claude": "Claude Sonnet 5",
    "deepseek": "DeepSeek Flash",
    "glm": "GLM Flash",
}


def _save(fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _f1_cost_points(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Per-judge (key, F1 value+CI, cost/1k, n) for the F1-vs-cost chart, plus whether this
    is the untuned headline cohort (True) or a per-judge fallback for a partial run (False —
    no common intersection across judges yet, so each point is on its own cohort)."""
    judges = summary.get("judges") or {}
    headline = summary.get("headline")
    points: list[dict[str, Any]] = []
    if headline:
        for key, block in headline["judges"].items():
            src = "jev" if key == "jev_tuned" else key  # jev_tuned's cost lives on run "jev"
            cost_block = judges.get(src)
            if cost_block is None:
                continue
            points.append({
                "key": src, "f1": block["f1"]["value"], "ci": block["f1"]["ci95"],
                "cost": cost_block["cost_usd_per_1k"], "n": block["n"],
            })
        return points, True
    for key in ("jev_untuned", "jev", "claude", "deepseek", "glm"):
        block = judges.get(key)
        if block is None:
            continue
        points.append({
            "key": key, "f1": block["f1"]["value"], "ci": block["f1"]["ci95"],
            "cost": block["cost_usd_per_1k"], "n": block["n"],
        })
    return points, False


def _f1_vs_cost(summary: dict[str, Any], out_dir: Path) -> Path | None:
    points, is_headline = _f1_cost_points(summary)
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pt in points:
        key = pt["key"]
        marker = "^" if key == "jev" else "o"  # tuned Jev drawn as a secondary marker
        ax.errorbar(
            pt["cost"], pt["f1"],
            yerr=[[pt["f1"] - pt["ci"][0]], [pt["ci"][1] - pt["f1"]]],
            fmt=marker, color=COLORS.get(key, "#111827"), capsize=4, label=LABELS.get(key, key),
        )
    cascade = (summary.get("cascade") or {}).get("frozen_band")
    if cascade:
        ax.plot(cascade["cost_usd_per_1k"], cascade["f1"], "s", color=COLORS["cascade"],
                label="Cascade")
    ax.set_xscale("log")
    for label, f1 in PUBLISHED_F1.items():
        ax.axhline(f1 / 100, color="#9ca3af", linestyle=":", linewidth=1)
        # x in axes coordinates (right edge), y in data coordinates: labels stay inside the plot
        ax.text(0.99, f1 / 100 + 0.003, f"{label} (published; not a controlled comparison)",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=7, color="#6b7280")
    ax.set_xlabel("Cost per 1,000 judgments (USD, log scale)")
    ax.set_ylabel("F1 (hallucination detection)")
    if is_headline:
        n = summary["headline"]["cohort"]["n"]
        ax.set_title(f"RAGTruth test set (n={n}): each judge at its default setting")
    else:
        ax.set_title(
            f"RAGTruth test set (n={summary.get('n_test')}) — partial run, each judge's own "
            "cohort (no common intersection yet)"
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.9)
    fig.text(0.01, -0.02, "Cost: billed (Jev, DeepSeek, GLM via OpenRouter); Claude at Batch API +\n"
             "prompt-cache prices (standard live about 2.9x higher). Tuned Jev is not comparable\n"
             "to the untuned rows.", fontsize=6, color="#6b7280", va="top")
    return _save(fig, out_dir / "f1_vs_cost.png")


def _reliability(summary: dict[str, Any], out_dir: Path) -> Path | None:
    judges = summary.get("judges") or {}
    key = "jev_untuned" if (judges.get("jev_untuned") or {}).get("reliability") else "jev"
    block = judges.get(key)
    if not block or not block.get("reliability"):
        return None
    bins = block["reliability"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="Perfect calibration")
    ax.plot([b["mean_prob"] for b in bins], [b["frac_positive"] for b in bins], "o-",
            color=COLORS[key], label=f"{LABELS[key]} (ECE {block['ece']:.3f})")
    ax.set_xlabel(f"{LABELS[key]} predicted P(hallucinated)")
    ax.set_ylabel("Observed fraction hallucinated")
    ax.set_title("Is Jev's confidence calibrated?")
    ax.legend()
    return _save(fig, out_dir / "reliability.png")


def _cascade_curve(summary: dict[str, Any], out_dir: Path) -> Path | None:
    cascade = summary.get("cascade")
    judges = summary.get("judges") or {}
    if not cascade or "jev" not in judges or "claude" not in judges:
        return None
    sweep = cascade["sweep"]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    rates = [p["escalation_rate"] * 100 for p in sweep]
    ax1.plot(rates, [p["f1"] for p in sweep], "o-", color=COLORS["cascade"], label="Cascade F1")
    ax1.axhline(judges["jev"]["f1"]["value"], color=COLORS["jev"], linestyle="--",
               label="Jev alone")
    ax1.axhline(judges["claude"]["f1"]["value"], color=COLORS["claude"], linestyle="--",
               label="Claude alone")
    ax1.set_xlabel("% of cases escalated to Claude")
    ax1.set_ylabel("F1")
    ax2 = ax1.twinx()
    ax2.plot(rates, [p["cost_usd_per_1k"] for p in sweep], "s:", color="#6b7280", label="$ / 1k")
    ax2.set_ylabel("Cost per 1,000 (USD)")
    ax1.set_title("Jev first, Claude when unsure")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    return _save(fig, out_dir / "cascade_curve.png")


def _slices(summary: dict[str, Any], out_dir: Path) -> Path | None:
    judges = summary.get("judges") or {}
    jev_key = "jev" if "jev" in judges else ("jev_untuned" if "jev_untuned" in judges else None)
    if jev_key is None or "claude" not in judges:
        return None
    tasks = sorted(judges[jev_key]["by_task"])
    if not tasks:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.38
    for k, name in enumerate((jev_key, "claude")):
        vals = [judges[name]["by_task"][t]["auroc"] or 0 for t in tasks]
        ax.bar([i + (k - 0.5) * width for i in range(len(tasks))], vals, width,
               color=COLORS[name], label=LABELS[name])
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("By task type")
    ax.legend()
    return _save(fig, out_dir / "slices.png")


def _pr_curve(summary: dict[str, Any], out_dir: Path) -> Path | None:
    h2h = summary.get("head_to_head")
    if not h2h:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    curve = h2h["jev_pr_curve"]
    # head_to_head is computed from run "jev" (the tuned wording) — the only Jev PR curve
    # summary.json carries; labeled honestly rather than claimed as the untuned wording.
    ax.plot([c["recall"] for c in curve], [c["precision"] for c in curve], "-",
            color=COLORS["jev"], label="Jev (all thresholds, tuned wording)")
    for name, point in h2h["operating_points"].items():
        ax.plot(point["recall"], point["precision"], "o", color=COLORS.get(name, "#111827"),
                markersize=9, label=LABELS.get(name, name))
    ax.set_xlabel("Recall (share of hallucinations caught)")
    ax.set_ylabel("Precision (share of flags that were right)")
    ax.set_title("Jev at every threshold vs the LLM judges' operating points")
    ax.legend()
    return _save(fig, out_dir / "pr_curve.png")


def render_all(summary: dict[str, Any], out_dir: Path) -> list[Path]:
    """Render every chart whose inputs are present in `summary`; skip the rest (partial run)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = (_f1_vs_cost, _reliability, _cascade_curve, _slices, _pr_curve)
    paths: list[Path] = []
    for fn in charts:
        path = fn(summary, out_dir)
        if path is not None:
            paths.append(path)
    return paths
