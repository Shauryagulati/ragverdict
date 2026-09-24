"""Render the results page's story chart and its social preview card.

Reads the published summary (docs/bench/data/summary.json) and writes:
  docs/bench/story_chart_light.png, docs/bench/story_chart_dark.png  (page figure)
  docs/bench/social-card.png                                         (1200x630 link preview)

The chart shows the four judges at their default settings: F1 with its 95% CI
against cost per 1,000 checks. Every judge is priced at its standard rate
(Claude at list price, uncached) so the x-axis matches the page text.

    python scripts/story_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "docs/bench/data/summary.json"
OUT = ROOT / "docs/bench"

THEMES = {
    "light": {"bg": "#ffffff", "fg": "#1a1a1a", "muted": "#5b6472", "grid": "#e3e6ea", "accent": "#2555a4"},
    "dark": {"bg": "#12151a", "fg": "#e6e8eb", "muted": "#9aa3ad", "grid": "#262c34", "accent": "#7fb0ff"},
}
LABELS = {"jev_untuned": "Jev", "glm": "GLM Flash", "claude": "Claude Sonnet 5", "deepseek": "DeepSeek Flash"}

def _system_font(path: str, fallback: str) -> str:
    """Register a macOS system font if present; otherwise use matplotlib's default."""
    if not Path(path).exists():
        return fallback
    font_manager.fontManager.addfont(path)
    return font_manager.FontProperties(fname=path).get_name()


_SERIF = _system_font("/System/Library/Fonts/NewYork.ttf", "serif")
_SANS = _system_font("/System/Library/Fonts/HelveticaNeue.ttc", "sans-serif")
plt.rcParams["font.family"] = _SANS


def load_points() -> list[dict]:
    """F1, CI and standard-rate cost per 1,000 checks for each headline judge."""
    summary = json.loads(SUMMARY.read_text())
    headline = summary["headline"]
    claude_list_per_1k = 1000 * headline["cost_basis"]["claude_usd_per_example_standard_uncached"]
    points = []
    for key, label in LABELS.items():
        judge = headline["judges"][key]
        billed = summary["judges"][key]["cost_usd_per_1k"]
        cost = claude_list_per_1k if key == "claude" else billed
        points.append({
            "key": key, "label": label, "f1": judge["f1"]["value"],
            "lo": judge["f1"]["ci95"][0], "hi": judge["f1"]["ci95"][1], "cost": cost,
        })
    return points


def draw_chart(ax: plt.Axes, points: list[dict], theme: dict, fontsize: float) -> None:
    ax.set_facecolor(theme["bg"])
    for p in points:
        color = theme["accent"] if p["key"] == "jev_untuned" else theme["muted"]
        ax.errorbar(p["cost"], p["f1"], yerr=[[p["f1"] - p["lo"]], [p["hi"] - p["f1"]]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.6, capsize=0, markersize=7)
        is_jev = p["key"] == "jev_untuned"
        # Jev sits at the far left, next to GLM; label it above its whisker to avoid overlap.
        anchor, offset, va = ((p["cost"], p["hi"]), (-6, 6), "bottom") if is_jev else \
            ((p["cost"], p["f1"]), (9, 0), "center")
        ax.annotate(f"{p['label']}\nF1 {p['f1']:.3f}, ${p['cost']:.2f}/1k", anchor,
                    xytext=offset, textcoords="offset points", va=va, ha="left",
                    fontsize=fontsize, color=theme["fg"],
                    fontweight="bold" if is_jev else "normal")
    ax.set_xscale("log")
    ax.set_xlim(0.025, 40)
    ax.set_ylim(0.675, 0.815)
    ax.set_xticks([0.05, 0.1, 0.3, 1, 3, 10])
    ax.set_xticklabels(["$0.05", "$0.10", "$0.30", "$1", "$3", "$10"])
    ax.minorticks_off()
    ax.set_xlabel("Cost per 1,000 checks (log scale)", color=theme["muted"], fontsize=fontsize)
    ax.set_ylabel("F1 against human labels", color=theme["muted"], fontsize=fontsize)
    ax.tick_params(colors=theme["muted"], labelsize=fontsize - 1, length=0)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])


def render_story_chart(points: list[dict], name: str) -> Path:
    theme = THEMES[name]
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=200)
    fig.patch.set_facecolor(theme["bg"])
    draw_chart(ax, points, theme, fontsize=10.5)
    fig.tight_layout()
    path = OUT / f"story_chart_{name}.png"
    fig.savefig(path, facecolor=theme["bg"])
    plt.close(fig)
    return path


def render_social_card(points: list[dict]) -> Path:
    theme = THEMES["dark"]
    fig = plt.figure(figsize=(12, 6.3), dpi=100)  # 1200x630 px
    fig.patch.set_facecolor(theme["bg"])
    fig.text(0.05, 0.88, "A ragverdict benchmark", color=theme["muted"], fontsize=15, va="top")
    fig.text(0.05, 0.80, "Jev vs LLM judges\non 2,666 RAG answers\nlabeled by humans",
             color=theme["fg"], fontsize=31, family=_SERIF, linespacing=1.2, va="top")
    fig.text(0.05, 0.43, "Highest F1 at default settings,\nfor about 1/140 of Claude's price.\n"
                         "At equal strictness, the LLMs\nare as good or better.",
             color=theme["muted"], fontsize=16, linespacing=1.4, va="top")
    fig.text(0.05, 0.07, "shauryagulati.github.io/ragverdict/bench", color=theme["accent"],
             fontsize=14)
    ax = fig.add_axes((0.50, 0.17, 0.44, 0.72))
    draw_chart(ax, points, theme, fontsize=12)
    path = OUT / "social-card.png"
    fig.savefig(path, facecolor=theme["bg"])
    plt.close(fig)
    return path


def main() -> None:
    points = load_points()
    for path in (render_story_chart(points, "light"), render_story_chart(points, "dark"),
                 render_social_card(points)):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
