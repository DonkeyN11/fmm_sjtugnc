#!/usr/bin/env python3
"""Fig: Trustworthiness on Real Road Network — 3 panels from Hainan-06 data.
(a) High TW (~0.99): traj 11 highway — unambiguous
(b) Medium TW (~0.59): traj 22 junction — branching ambiguity
(c) Low TW (0.01~0.3): traj 22 parallel-road false lock — emission model failure
"""
from pathlib import Path
import csv, json
import numpy as np
import shapefile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "experiments/data/real_data"
MAP = PROJECT / "input/map/hainan/edges.shp"
OUT = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 300
C = {"road":"#D5D8DC","tw_high":"#27AE60","tw_med":"#F39C12","tw_low":"#C0392B",
     "obs":"#2471A3","obs_a":"#2471A320","match":"#1A5276","gt":"#1E8449"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.5,"axes.titlesize":9.5,
    "axes.labelsize":8,"legend.fontsize":7,"figure.dpi":DPI,"savefig.dpi":DPI,
    "savefig.bbox":"tight","text.usetex":False})

# ── Load road network ──
sf = shapefile.Reader(str(MAP))
all_roads = []
for shape in sf.shapes():
    pts = np.array(shape.points)
    if len(pts) >= 2: all_roads.append(pts)

# ── Load data ──
REV = json.load(open(PROJECT / "experiments/config/reverse_edge_map.json"))
def em(m,t): return str(m)==str(t) or REV.get(str(m))==str(t)

gt_edges = {}
with open(DATA / "ground_truth.csv") as f:
    for row in csv.DictReader(f, delimiter=";"):
        gt_edges[(row["id"].strip(), int(row["seq"]))] = row["edge_id"].strip()

rows = []
with open(DATA / "aligned.csv") as f:
    for row in csv.DictReader(f, delimiter=";"):
        tid = row["id"].strip(); useq = int(row["uni_seq"])
        if (tid, useq) not in gt_edges: continue
        gt_eid = gt_edges[(tid, useq)]
        if gt_eid in ("0","-1"): continue
        x = row.get("cmm_x",""); tw = float(row.get("cmm_tw","0") or 0)
        if not x: continue
        rows.append({"id":tid,"seq":useq,"x":float(x),"y":float(row["cmm_y"]),
                     "ox":float(row["obs_x"]),"oy":float(row["obs_y"]),
                     "tw":tw,"correct":em(row.get("cmm_cpath",""),gt_eid)})

# ── Select exemplar segments ──
def get_range(data, tid, seq_s, seq_e):
    return [r for r in data if r["id"]==tid and seq_s<=r["seq"]<=seq_e]

# (a) Traj 11 highway: seq 498-604, TW~0.999
seg_a = get_range(rows, "11", 498, 604)
# (b) Traj 22 junction: seq 333-388, TW~0.59
seg_b = get_range(rows, "22", 333, 388)
# (c) Traj 21 wrong-match with TW drop then recovery: seq 2315-2350
# Wrong at 2324-2337 (TW~0.05-0.79), recovers at 2338 (TW→1.0)
seg_c = get_range(rows, "21", 2315, 2350)

def plot_panel(ax, seg, roads, bbox_pad=0.005, title=""):
    """Plot road network + trajectory colored by TW."""
    if not seg: return
    xs = [s["x"] for s in seg]; ys = [s["y"] for s in seg]
    oxs = [s["ox"] for s in seg]; oys = [s["oy"] for s in seg]
    tws = np.array([s["tw"] for s in seg])
    cx, cy = np.mean(xs), np.mean(ys)
    lo, hi = cx-bbox_pad, cx+bbox_pad; bo, to_ = cy-bbox_pad, cy+bbox_pad

    # Filter roads in view
    for r in roads:
        rx, ry = np.mean(r[:,0]), np.mean(r[:,1])
        if lo <= rx <= hi and bo <= ry <= to_:
            ax.plot(r[:,0], r[:,1], color=C["road"], lw=0.5, alpha=0.7, zorder=1)

    ax.set_xlim(lo, hi); ax.set_ylim(bo, to_); ax.set_aspect("equal")

    # GNSS observations as faint dots
    ax.scatter(oxs, oys, s=3, c=C["obs"], alpha=0.15, zorder=2)

    # Matched path colored by TW
    norm = plt.Normalize(0, 1)
    points = np.array([xs, ys]).T.reshape(-1,1,2)
    segments_lc = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments_lc, cmap="RdYlGn", norm=norm, linewidth=2.5, zorder=4)
    lc.set_array(tws[1:])
    ax.add_collection(lc)

    # Colorbar — small
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label("tw$_t$", fontsize=7); cbar.ax.tick_params(labelsize=6)

    # Start marker
    ax.scatter(xs[0], ys[0], s=40, c=C["match"], marker="o", zorder=5, ec="white", lw=0.8)
    ax.scatter(xs[-1], ys[-1], s=30, c=C["match"], marker="s", zorder=5, ec="white", lw=0.8)

    # Info box
    mean_tw = np.mean(tws)
    corr_pct = np.mean([s["correct"] for s in seg])*100
    color_tag = C["tw_high"] if mean_tw>0.9 else (C["tw_med"] if mean_tw>0.5 else C["tw_low"])
    tag = "HIGH" if mean_tw>0.9 else ("MEDIUM" if mean_tw>0.5 else "LOW")
    ax.text(0.03, 0.97, f"TW: {tag}\n$\\langle\\mathrm{{tw}}\\rangle$={mean_tw:.3f}\n{corr_pct:.0f}% correct",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, fontweight="bold", color=color_tag,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_tag, alpha=0.9))

    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, pad=6, fontweight="bold")

# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

plot_panel(axes[0], seg_a, all_roads, 0.004, "(a) Highway — High TW, Unambiguous")
plot_panel(axes[1], seg_b, all_roads, 0.003, "(b) Junction — Medium TW, Branching")
plot_panel(axes[2], seg_c, all_roads, 0.004, "(c) Dual Carriageway — Low TW, False Lock")

fig.suptitle("Trustworthiness on Real Road Networks: Hainan-06 Dataset (SPP, 7 Trajectories)",
             fontsize=10, fontweight="bold", y=1.01)
fig.tight_layout(rect=(0,0,1,0.94))

for fmt in ["svg","png"]: fig.savefig(OUT/f"trustworthiness.{fmt}", dpi=DPI, format=fmt)
plt.close(fig); print("Saved trustworthiness.svg/.png")
