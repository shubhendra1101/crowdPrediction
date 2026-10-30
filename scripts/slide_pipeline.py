"""Simplified CrowdSafe pipeline for presentation slides (16:9).

    python scripts/slide_pipeline.py --out results/figures/slide_pipeline_simple.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE, BLUE_EDGE = "#e3eefc", "#2f6fdb"
ORANGE, ORANGE_EDGE = "#fdeede", "#e0782a"
GREY, GREY_EDGE = "#f4f4f4", "#7a7a7a"
TXT = "#1d2330"


def draw(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    W, H = 3.15, 2.05
    xs = [0.45, 4.35, 8.25, 12.15]          # four columns
    y_top, y_bot = 4.55, 0.85
    box = {}

    def b(key, x, y, title, lines, fc, ec, dashed=False, step=None):
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.03,rounding_size=0.18", fc=fc, ec=ec,
                                    lw=2.2, ls=(0, (5, 3)) if dashed else "-"))
        if step is not None:
            ax.add_patch(plt.Circle((x + 0.02, y + H - 0.02), 0.26, color=ec, zorder=3))
            ax.text(x + 0.02, y + H - 0.03, str(step), ha="center", va="center", fontsize=13, color="white",
                    weight="bold", zorder=4)
        ax.text(x + W / 2, y + H - 0.36, title, ha="center", va="center",
                fontsize=15.5, weight="bold", color=TXT)
        ax.text(x + W / 2, y + H - 0.78, "\n".join(lines), ha="center", va="top", fontsize=12.5, color=TXT,
                linespacing=1.45)
        box[key] = (x, y)

    def arrow(p0, p1, color="#333333", dashed=False, rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=24, lw=2.4, color=color,
                                     ls=(0, (5, 3)) if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=2, shrinkB=2))

    # row labels
    ax.text(0.45, 6.95, "SEE THE CROWD  ·  every 1 s", fontsize=14, weight="bold", color=BLUE_EDGE)
    ax.text(0.45, 3.2, "PREDICT AND WARN  ·  every 5 s", fontsize=14, weight="bold", color=ORANGE_EDGE)

    # one-time setup
    cx = 7.6
    ax.add_patch(FancyBboxPatch((cx, 7.35), 2 * W + 1.4, 1.3, boxstyle="round,pad=0.03,rounding_size=0.18",
                                fc=GREY, ec=GREY_EDGE, lw=2.0, ls=(0, (5, 3))))
    ax.text(cx + (2 * W + 1.4) / 2, 8.35, "Camera Calibration  (one-time setup)", ha="center", va="center",
            fontsize=15, weight="bold", color=TXT)
    ax.text(cx + (2 * W + 1.4) / 2, 7.72, "Desk and table sizes + straight edges → pixels mapped to\nfloor positions in metres",
            ha="center", va="center", fontsize=12.5, color=TXT, linespacing=1.3)

    # row 1
    b("in", xs[0], y_top, "CCTV Input", ["Fixed camera video", "1 frame per second"], BLUE, BLUE_EDGE, step=1)
    b("count", xs[1], y_top, "People Counting", ["CLIP-EBC density map", "of people in each frame", "zero-shot, no training"],
      BLUE, BLUE_EDGE, step=2)
    b("move", xs[2], y_top, "Movement", ["RAFT optical flow", "walking speed and", "direction (2 frames, 0.2 s)"],
      BLUE, BLUE_EDGE, step=3)
    b("zone", xs[3], y_top, "Zone Density", ["people per m²", "for every floor zone,", "stored each second"],
      BLUE, BLUE_EDGE, step=4)
    # row 2 (right to left, continuing the path)
    b("fc", xs[3], y_bot, "Forecasting", ["Chronos-2: density", "10–60 s ahead", "+ safe upper limit"],
      ORANGE, ORANGE_EDGE, step=5)
    b("risk", xs[2], y_bot, "Risk Level", ["Low < 2  ·  Medium 2–4", "High 4–5  ·  Critical ≥ 5", "persons/m²"],
      ORANGE, ORANGE_EDGE, step=6)
    b("alert", xs[1], y_bot, "Alert & Dashboard", ["alert if High for 10 s", "cleared after 30 s below", "zones shown on dashboard"],
      ORANGE, ORANGE_EDGE, step=7)

    mid = y_top + H / 2
    for a, c in ((xs[0], xs[1]), (xs[1], xs[2]), (xs[2], xs[3])):
        arrow((a + W, mid), (c, mid))
    arrow((xs[3] + W / 2, y_top), (xs[3] + W / 2, y_bot + H))                    # zone -> forecasting
    midb = y_bot + H / 2
    arrow((xs[3], midb), (xs[2] + W, midb))                                       # forecasting -> risk
    arrow((xs[2], midb), (xs[1] + W, midb))                                       # risk -> alert
    arrow((xs[2] + W / 2, 7.35), (xs[2] + W / 2, y_top + H), color=GREY_EDGE, dashed=True)   # calibration -> movement
    arrow((xs[3] + W / 2, 7.35), (xs[3] + W / 2, y_top + H), color=GREY_EDGE, dashed=True)  # -> zone

    # repeat note in the free bottom-left cell
    ax.add_patch(FancyBboxPatch((xs[0], y_bot), W, H, boxstyle="round,pad=0.03,rounding_size=0.18", fc="white",
                                ec="#c9ced8", lw=1.5))
    ax.text(xs[0] + W / 2, y_bot + H - 0.45, "↻  Continuous loop", ha="center", va="center", fontsize=15,
            weight="bold", color=TXT)
    ax.text(xs[0] + W / 2, y_bot + H - 0.95, "counting and movement:\nevery second\nforecast, risk, alerts:\nevery 5 seconds",
            ha="center", va="top", fontsize=12, color="#4a5568", linespacing=1.35)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results/figures/slide_pipeline_simple.png"))
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    draw(a.out)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
