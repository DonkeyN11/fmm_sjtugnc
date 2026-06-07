#!/usr/bin/env python3
"""Fig.8: Trustworthiness Across Road Geometries — 2x2 intuition diagram."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import matplotlib.patches as mpatches
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
FIGS = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS.mkdir(parents=True, exist_ok=True)

DPI = 300; C = {"blue":"#2471A3","red":"#C0392B","green":"#27AE60","road":"#566573","gray":"#95A5A6","orange":"#E67E22"}

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.5,"axes.titlesize":9,
    "axes.labelsize":7.5,"legend.fontsize":6.5,"figure.dpi":DPI,"savefig.dpi":DPI,
    "savefig.bbox":"tight","text.usetex":False})

def draw_road(ax, segments, color=C["road"], lw=2.5, alpha=1.0, **kw):
    for (s,e) in segments: ax.plot([s[0],e[0]],[s[1],e[1]],color=color,lw=lw,alpha=alpha,**kw)

def draw_gnss(ax, pts, sigma=0.10):
    for p in pts: ax.plot(*p,"o",color=C["red"],ms=6,mec="white",mew=0.7,zorder=5)
    t = np.linspace(0, 2*np.pi, 50)
    for p in pts:
        ax.fill(p[0]+sigma*np.cos(t), p[1]+sigma*np.sin(t), fc=C["red"], alpha=0.06, ec=C["red"], lw=0.4, ls="--")

def add_tw_box(ax, tw_val, label, color, detail):
    """Compact TW label at top with subtle background."""
    bbox = dict(boxstyle="round,pad=0.2", fc=color+"18", ec=color, lw=1.2, alpha=0.9)
    ax.text(0.5, 0.94, label, ha="center", va="top", transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", color=color, bbox=bbox)
    ax.text(0.5, 0.82, detail, ha="center", va="top", transform=ax.transAxes,
            fontsize=6.5, color=C["gray"])

def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    axes = axes.flatten()
    for ax in axes: ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    # ═══ (a) Highway ═══
    ax = axes[0]
    road = [((-4,0),(4,0))]
    draw_road(ax, road, lw=3)
    pts = [(-2,0.06),(0,0.0),(2,0.04)]
    draw_gnss(ax, pts)
    ax.set_xlim(-4.5,4.5); ax.set_ylim(-1.5,1.5)
    add_tw_box(ax, "$\\mathrm{tw}_t\\!\\approx\\!0.99$", "High Trustworthiness", "#1E8449",
               "Unambiguous: single road, tight constraints")
    ax.set_title("(a) Highway — Unambiguous", pad=6, fontweight="bold")

    # ═══ (b) Junction ═══
    ax = axes[1]
    roads = [((-4,1),(0,0)),((0,0),(3,2.5)),((0,0),(3,-1))]
    draw_road(ax, roads, lw=2.5)
    pts = [(-2.3,0.7),(-1.0,0.4),(0.2,-0.3)]
    draw_gnss(ax, pts)
    ax.set_xlim(-4.5,3.5); ax.set_ylim(-2,3.5)
    add_tw_box(ax, "$\\mathrm{tw}_t\\!\\approx\\!0.55$", "Low Trustworthiness", "#CA6F1E",
               "Ambiguous: multiple paths consistent")
    ax.set_title("(b) Junction — Branching Ambiguity", pad=6, fontweight="bold")

    # ═══ (c) Dense grid ═══
    ax = axes[2]
    roads = [[(-4,y),(4,y)] for y in [-2,-1,0,1,2]] + [[(x,-2.5),(x,2.5)] for x in [-3,-1,1,3]]
    draw_road(ax, roads, lw=1.5, alpha=0.35)
    ax.plot([-4,4],[0,0],color=C["green"],lw=3,zorder=3,alpha=0.75)
    pts = [(-2,-0.05),(0,0.02),(2,0.0)]
    draw_gnss(ax, pts)
    ax.set_xlim(-4.5,4.5); ax.set_ylim(-2.8,2.8)
    add_tw_box(ax, "$\\mathrm{tw}_t\\!\\approx\\!0.92$", "Medium-High Trustworthiness", "#1E8449",
               "Grid: topology constrains candidates")
    ax.set_title("(c) Dense Urban Grid — Constrained", pad=6, fontweight="bold")

    # ═══ (d) Parallel edges ═══
    ax = axes[3]
    roads = [[(-4,0.5),(4,0.5)],[(-4,-0.5),(4,-0.5)]]
    draw_road(ax, roads[:1], color=C["green"], lw=3, zorder=3)
    draw_road(ax, roads[1:], lw=2, alpha=0.45)
    pts = [(-2,0.45),(-0.3,-0.05),(2,-0.35)]
    draw_gnss(ax, pts)
    ax.set_xlim(-4.5,4.5); ax.set_ylim(-1.5,1.5)
    add_tw_box(ax, "$\\mathrm{tw}_t\\!\\approx\\!0.60$", "Medium Trustworthiness (risk)", "#CA6F1E",
               "Parallel edges: emission ambiguity")
    ax.set_title("(d) Dual Carriageway — Parallel Ambiguity", pad=6, fontweight="bold")

    fig.suptitle("Trustworthiness $\\mathrm{tw}_t = P(x_t=i^*\\mid z_{1:t})$ Across Road Geometries",
                 fontsize=10.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.8, w_pad=1.8)
    for fmt in ["svg","png"]: fig.savefig(FIGS/f"trustworthiness.{fmt}", dpi=DPI, format=fmt)
    plt.close(fig); print("Saved trustworthiness.svg/.png")

if __name__ == "__main__": main()
