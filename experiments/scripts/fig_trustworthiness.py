#!/usr/bin/env python3
"""Fig. 9: Trustworthiness Across Road Geometries — 2×2 layout.
Output: SVG to docs/.../figs/trustworthiness.svg
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
FIGS_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
COLOR_BLUE = "#2166ac"
COLOR_RED = "#b2182b"
COLOR_GREEN = "#4dac26"
COLOR_ORANGE = "#e66101"
COLOR_PURPLE = "#5e3c99"
COLOR_GRAY = "#777777"
COLOR_ROAD = "#555555"
TW_CMAP = plt.cm.RdYlGn  # Red (low TW) → Green (high TW)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9.5, "axes.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "text.usetex": False,
})


def draw_road_segment(ax, start, end, color=COLOR_ROAD, lw=2.5, **kwargs):
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=lw, **kwargs)


def draw_gnss_point(ax, pos, tw_color=None):
    fc = tw_color if tw_color else COLOR_RED
    ax.plot(pos[0], pos[1], "o", color=fc, markersize=8,
            markeredgecolor="white", markeredgewidth=1, zorder=5)


def add_tw_label(ax, tw_val, h_val, pos="upper right"):
    color = TW_CMAP(tw_val)  # higher TW = greener
    x, y = {"upper right": (0.96, 0.96), "upper left": (0.04, 0.96),
            "lower right": (0.96, 0.04), "lower left": (0.04, 0.04)}[pos]
    ha = "right" if "right" in pos else "left"
    va = "top" if "upper" in pos else "bottom"
    ax.text(x, y, f"TW = {tw_val:.2f}\n$H$ = {h_val:.2f} bits",
            transform=ax.transAxes, ha=ha, va=va, fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.9))


def main():
    fig = plt.figure(figsize=(7.5, 6.5))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30,
                          left=0.05, right=0.97, top=0.93, bottom=0.04)

    # ═══════════════════════════════════════════
    # PANEL (a): Highway — High TW
    # ═══════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_aspect("equal")
    ax_a.set_xlim(-3, 3); ax_a.set_ylim(-2, 2)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_a.set_title("(a) Highway — High TW", fontsize=9.5)

    # Single road
    draw_road_segment(ax_a, (-3, 0.1), (3, -0.1), lw=3)
    # GNSS near road
    pos = (0.0, 0.15)
    # Small isotropic error
    from matplotlib.patches import Ellipse
    ell = Ellipse(pos, 0.4, 0.3, angle=5, fill=True,
                  facecolor=COLOR_BLUE, alpha=0.15, edgecolor=COLOR_BLUE, linewidth=1)
    ax_a.add_patch(ell)
    draw_gnss_point(ax_a, pos, TW_CMAP(0.97))
    # Candidate
    ax_a.plot(0.0, 0.0, "D", color=COLOR_GREEN, markersize=7, markeredgecolor="darkgreen", markeredgewidth=0.5, zorder=4)
    add_tw_label(ax_a, 0.97, 0.12, "upper right")

    # ═══════════════════════════════════════════
    # PANEL (b): Complex Junction — Low TW
    # ═══════════════════════════════════════════
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_aspect("equal")
    ax_b.set_xlim(-3, 3); ax_b.set_ylim(-2, 2)
    ax_b.set_xticks([]); ax_b.set_yticks([])
    ax_b.set_title("(b) Complex Junction — Low TW", fontsize=9.5)

    # Multiple roads meeting at center
    roads_j = [
        ((-3, 1.5), (0, 0)), ((3, 1.5), (0, 0)), ((-3, -1.5), (0, 0)),
        ((3, -1.5), (0, 0)), ((-2, 2), (0, 0)), ((2, -2), (0, 0)),
    ]
    for s, e in roads_j:
        draw_road_segment(ax_b, s, e, lw=1.8, alpha=0.8)

    pos_b = (0.05, -0.05)
    ell_b = Ellipse(pos_b, 1.8, 1.2, angle=35, fill=True,
                    facecolor=COLOR_BLUE, alpha=0.12, edgecolor=COLOR_BLUE, linewidth=1)
    ax_b.add_patch(ell_b)
    draw_gnss_point(ax_b, pos_b, TW_CMAP(0.34))

    # Candidate points on multiple edges
    cand_colors = [TW_CMAP(v) for v in [0.34, 0.28, 0.22, 0.18, 0.08]]
    cand_positions = [
        (0.30, 0.25), (-0.15, -0.35), (0.40, -0.20), (-0.35, 0.15), (0.05, 0.50),
    ]
    for (cx, cy), cc in zip(cand_positions, cand_colors):
        ax_b.plot(cx, cy, "s", color=cc, markersize=6, markeredgecolor="white",
                  markeredgewidth=0.5, zorder=4)

    add_tw_label(ax_b, 0.34, 2.81, "upper right")

    # ═══════════════════════════════════════════
    # PANEL (c): Simple Intersection — Medium TW
    # ═══════════════════════════════════════════
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_aspect("equal")
    ax_c.set_xlim(-3, 3); ax_c.set_ylim(-2, 2)
    ax_c.set_xticks([]); ax_c.set_yticks([])
    ax_c.set_title("(c) Simple Intersection — Medium TW", fontsize=9.5)

    # Crossing roads
    draw_road_segment(ax_c, (-3, 0.3), (3, -0.3), lw=2.2)
    draw_road_segment(ax_c, (-0.3, -2), (0.3, 2), lw=2.2)

    pos_c = (0.15, 0.1)
    ell_c = Ellipse(pos_c, 1.2, 0.7, angle=20, fill=True,
                    facecolor=COLOR_BLUE, alpha=0.15, edgecolor=COLOR_BLUE, linewidth=1)
    ax_c.add_patch(ell_c)
    draw_gnss_point(ax_c, pos_c, TW_CMAP(0.72))

    # Candidates on both roads
    ax_c.plot(0.05, 0.0, "s", color=TW_CMAP(0.72), markersize=7, markeredgecolor="white", markeredgewidth=0.5, zorder=4)
    ax_c.plot(0.20, -0.08, "s", color=TW_CMAP(0.28), markersize=5, markeredgecolor="white", markeredgewidth=0.5, zorder=4)

    add_tw_label(ax_c, 0.72, 1.54, "upper right")

    # ═══════════════════════════════════════════
    # PANEL (d): How TW is Computed
    # ═══════════════════════════════════════════
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_xlim(0, 1); ax_d.set_ylim(0, 1)
    ax_d.axis("off")
    ax_d.set_title("(d) How Trustworthiness is Computed", fontsize=9.5)

    # Forward recursion mini diagram
    ax_d.text(0.5, 0.92, "Forward Algorithm + Softmax", ha="center", fontsize=9,
              fontweight="bold", color=COLOR_BLUE)

    # Trellis sketch
    for t in range(3):
        x0 = 0.15 + t * 0.30
        for c in range(4):
            y0 = 0.55 - c * 0.08
            ax_d.plot(x0, y0, "o", color=COLOR_GRAY, markersize=4, alpha=0.6)
        if t > 0:
            for c1 in range(4):
                for c2 in range(4):
                    ax_d.plot([0.15 + (t - 1) * 0.30, x0],
                              [0.55 - c1 * 0.08, 0.55 - c2 * 0.08],
                              color=COLOR_GRAY, linewidth=0.3, alpha=0.3)

    ax_d.text(0.15, 0.64, "$t-1$", ha="center", fontsize=7, color=COLOR_GRAY)
    ax_d.text(0.45, 0.64, "$t$", ha="center", fontsize=7, color=COLOR_GRAY)
    ax_d.text(0.75, 0.64, "$t+1$", ha="center", fontsize=7, color=COLOR_GRAY)

    # Highlight one candidate at layer t
    ax_d.plot(0.45, 0.55, "o", color=COLOR_GREEN, markersize=8, zorder=5)
    ax_d.annotate("$i^*$", (0.45, 0.55), textcoords="offset points",
                  xytext=(8, 3), fontsize=8, color="darkgreen", fontweight="bold")

    # Formula
    ax_d.text(0.5, 0.39,
              "$\\log\\alpha_t(b) = \\log\\sum_a \\exp(\\log\\alpha_{t-1}(a) + \\log t_{a\\to b}) + \\log p(z_t|s_t^{(b)})$",
              ha="center", fontsize=7, transform=ax_d.transAxes,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f4f8", edgecolor=COLOR_BLUE, alpha=0.5))

    # Softmax bar chart
    ax_bar = ax_d.inset_axes([0.08, 0.02, 0.84, 0.32])
    candidates = ["i*", "A", "B", "C", "D", "bg"]
    tw_vals = [0.72, 0.12, 0.08, 0.04, 0.03, 0.01]
    colors_bar = [COLOR_GREEN] + [COLOR_BLUE] * 3 + [COLOR_GRAY] * 2
    bars = ax_bar.bar(range(len(candidates)), tw_vals, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax_bar.set_xticks(range(len(candidates)))
    ax_bar.set_xticklabels(candidates, fontsize=6.5)
    ax_bar.set_ylabel("$P(s_t^{(i)}|z_{1:t})$", fontsize=7)
    ax_bar.set_ylim(0, 0.85)
    ax_bar.set_title("Softmax Posterior Distribution", fontsize=7.5)

    # Arrow from selected candidate to bar
    ax_bar.annotate("TW = max", xy=(0, tw_vals[0] + 0.02), fontsize=7,
                    ha="center", color="darkgreen", fontweight="bold")

    fig.suptitle("Trustworthiness: A Calibrated Per-Epoch Confidence Score from the HMM Forward Algorithm",
                 fontsize=10, fontweight="bold", y=0.99)

    out = FIGS_DIR / "trustworthiness.svg"
    fig.savefig(out, dpi=DPI, format="svg")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
