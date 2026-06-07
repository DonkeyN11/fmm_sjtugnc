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

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9.5,
    "axes.labelsize":8,"legend.fontsize":6.5,"figure.dpi":DPI,"savefig.dpi":DPI,
    "savefig.bbox":"tight","text.usetex":False})

def draw_road(ax, segments, color=C["road"], lw=2.5, alpha=1.0, **kw):
    for (s,e) in segments: ax.plot([s[0],e[0]],[s[1],e[1]],color=color,lw=lw,alpha=alpha,**kw)

def draw_gnss(ax, pts, sigma=0.12):
    for p in pts: ax.plot(*p,"o",color=C["red"],ms=7,mec="white",mew=0.8,zorder=5)
    for p in pts:
        t = np.linspace(0,2*np.pi,60)
        ax.fill(p[0]+sigma*np.cos(t), p[1]+sigma*np.sin(t), fc=C["red"], alpha=0.08, ec=C["red"], lw=0.5, ls="--")

def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    axes = axes.flatten()
    for ax in axes: ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    # ═══ (a) Highway — simple, unambiguous → HIGH TW ═══
    ax = axes[0]
    road = [((-4,0),(4,0))]
    draw_road(ax, road)
    pts = [(-2,0.08),(0,-0.05),(2,0.03)]
    draw_gnss(ax, pts)
    ax.set_xlim(-4.5,4.5); ax.set_ylim(-2,2)
    ax.add_patch(mpatches.FancyBboxPatch((0.02,0.78),0.96,0.2,boxstyle="round,pad=0.15",
                 transform=ax.transAxes,fc="#27AE6020",ec=C["green"],lw=1.5))
    ax.text(0.5,0.88,"High Trustworthiness  $\\mathrm{tw}_t\\!\\approx\\!0.99$",
            ha="center",va="center",transform=ax.transAxes,fontsize=10,fontweight="bold",color="#1E8449")
    ax.text(0.5,0.78,"Unambiguous: single road, isotropic error",ha="center",va="center",
            transform=ax.transAxes,fontsize=7.5,color=C["gray"])
    ax.set_title("(a) Highway — Unambiguous Matching", pad=8, fontweight="bold")

    # ═══ (b) Junction — branching → LOW TW ═══
    ax = axes[1]
    roads = [((-4,1),(0,0)),((0,0),(3,2.5)),((0,0),(3,-1))]
    draw_road(ax, roads)
    pts = [(-2.5,0.8),(-1.2,0.45),(0.3,-0.3)]
    draw_gnss(ax, pts)
    ax.set_xlim(-4.5,3.5); ax.set_ylim(-2,3.5)
    ax.add_patch(mpatches.FancyBboxPatch((0.02,0.78),0.96,0.2,boxstyle="round,pad=0.15",
                 transform=ax.transAxes,fc="#E67E2220",ec=C["orange"],lw=1.5))
    ax.text(0.5,0.88,"Low Trustworthiness  $\\mathrm{tw}_t\\!\\approx\\!0.50$",
            ha="center",va="center",transform=ax.transAxes,fontsize=10,fontweight="bold",color="#CA6F1E")
    ax.text(0.5,0.78,"Ambiguous: multiple paths consistent with GNSS",ha="center",va="center",
            transform=ax.transAxes,fontsize=7.5,color=C["gray"])
    ax.set_title("(b) Junction — Branching Ambiguity", pad=8, fontweight="bold")

    # ═══ (c) Dense grid → MEDIUM-HIGH TW ═══
    ax = axes[2]
    roads = [[(-4,y),(4,y)] for y in [-2,-1,0,1,2]] + [[(x,-2.5),(x,2.5)] for x in [-3,-1,1,3]]
    draw_road(ax, roads, alpha=0.4)
    # Highlight the correct path
    ax.plot([-4,4],[0,0],color=C["green"],lw=3,zorder=3,alpha=0.8)
    pts = [(-2,-0.1),(0,0.05),(2,-0.03)]
    draw_gnss(ax, pts)
    ax.set_xlim(-4.5,4.5); ax.set_ylim(-2.8,2.8)
    ax.add_patch(mpatches.FancyBboxPatch((0.02,0.78),0.96,0.2,boxstyle="round,pad=0.15",
                 transform=ax.transAxes,fc="#27AE6020",ec=C["green"],lw=1.5))
    ax.text(0.5,0.88,"Medium-High TW  $\\mathrm{tw}_t\\!\\approx\\!0.92$",
            ha="center",va="center",transform=ax.transAxes,fontsize=10,fontweight="bold",color="#1E8449")
    ax.text(0.5,0.78,"Grid: topology constrains possibilities, TW recovers",ha="center",va="center",
            transform=ax.transAxes,fontsize=7.5,color=C["gray"])
    ax.set_title("(c) Dense Urban Grid — Topology-Constrained", pad=8, fontweight="bold")

    # ═══ (d) Parallel edges → MEDIUM TW + risk ═══
    ax = axes[3]
    roads = [[(-4,0.5),(4,0.5)],[(-4,-0.5),(4,-0.5)]]
    draw_road(ax, roads[:1], color=C["green"], lw=3, zorder=3)
    draw_road(ax, roads[1:], alpha=0.5)
    pts = [(-2,0.55),(-0.5,-0.1),(2,-0.45)]
    draw_gnss(ax, pts)
    ax.set_xlim(-4.5,4.5); ax.set_ylim(-2,2)
    ax.add_patch(mpatches.FancyBboxPatch((0.02,0.78),0.96,0.2,boxstyle="round,pad=0.15",
                 transform=ax.transAxes,fc="#E67E2220",ec=C["orange"],lw=1.5))
    ax.text(0.5,0.88,"Medium TW  $\\mathrm{tw}_t\\!\\approx\\!0.65$  (false-lock risk)",
            ha="center",va="center",transform=ax.transAxes,fontsize=10,fontweight="bold",color="#CA6F1E")
    ax.text(0.5,0.78,"Parallel edges: emission model ambiguity",ha="center",va="center",
            transform=ax.transAxes,fontsize=7.5,color=C["gray"])
    ax.set_title("(d) Dual Carriageway — Parallel-Edge Ambiguity", pad=8, fontweight="bold")

    fig.suptitle("Trustworthiness $\\mathrm{tw}_t = P(x_t=i^*\\mid z_{1:t})$ Across Road Geometries",
                 fontsize=10.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0,0,1,0.96))
    for fmt in ["svg","png"]: fig.savefig(FIGS/f"trustworthiness.{fmt}",dpi=DPI,format=fmt)
    plt.close(fig); print("Saved trustworthiness.svg/.png")

if __name__ == "__main__": main()
