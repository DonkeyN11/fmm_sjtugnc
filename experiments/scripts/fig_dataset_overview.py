#!/usr/bin/env python3
"""Fig. dataset_overview: Haikou road network + 7 SPP trajectories + HPL context."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import shapefile

OUT = Path(__file__).resolve().parents[2] / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 7, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight"})

# ── 1. Load road network ──────────────────────────────────────────
ROAD = Path("/home/ncz/fmm_sjtugnc/input/map/hainan/edges.shp")
sf = shapefile.Reader(str(ROAD))
roads = []
for shape in sf.shapes():
    pts = np.array(shape.points)
    if len(pts) >= 2:
        roads.append(pts)

print(f"Loaded {len(roads)} road segments")

# ── 2. Load 7 trajectories ────────────────────────────────────────
import csv
DATA = Path("/home/ncz/fmm_sjtugnc/experiments/data/real_data")
TRAJ_IDS = [11, 12, 13, 14, 21, 22, 23]
COLORS = plt.cm.tab10(np.linspace(0, 1, 7))

all_lons, all_lats = [], []
trajs = {}
for tid in TRAJ_IDS:
    lons, lats = [], []
    with open(DATA / "cmm_input_points.csv") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if int(row["id"]) == tid:
                lons.append(float(row["x"]))
                lats.append(float(row["y"]))
    trajs[tid] = (np.array(lons), np.array(lats))
    all_lons.extend(lons); all_lats.extend(lats)
    print(f"Traj {tid}: {len(lons)} epochs")

# ── 3. Plot ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

# Road network — light gray
for r in roads[:5000]:  # subsample for speed
    ax.plot(r[:, 0], r[:, 1], color="#d0d0d0", linewidth=0.2, alpha=0.6)

# Trajectories — colored, with direction arrows
bbox = None
for i, tid in enumerate(TRAJ_IDS):
    lons, lats = trajs[tid]
    color = COLORS[i]
    ax.plot(lons, lats, color=color, linewidth=0.8, label=f"Traj {tid}", alpha=0.9)
    # Start marker
    ax.scatter(lons[0], lats[0], color=color, marker="o", s=50, edgecolors="white", linewidth=0.5, zorder=5)
    # Direction arrow at midpoint
    mid = len(lons) // 2
    ax.annotate("", xy=(lons[mid+5], lats[mid+5]), xytext=(lons[mid-5], lats[mid-5]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.7))
    if bbox is None:
        bbox = [lons.min(), lons.max(), lats.min(), lats.max()]
    else:
        bbox[0] = min(bbox[0], lons.min())
        bbox[1] = max(bbox[1], lons.max())
        bbox[2] = min(bbox[2], lats.min())
        bbox[3] = max(bbox[3], lats.max())

# Padding
pad_lon = (bbox[1] - bbox[0]) * 0.08
pad_lat = (bbox[3] - bbox[2]) * 0.08
ax.set_xlim(bbox[0] - pad_lon, bbox[1] + pad_lon)
ax.set_ylim(bbox[2] - pad_lat, bbox[3] + pad_lat)

ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.legend(loc="upper left", ncol=4, framealpha=0.9)

# ── 4. Inset: Hainan island location ──────────────────────────────
ax_inset = inset_axes(ax, width="28%", height="28%", loc="lower right",
                       bbox_to_anchor=(0.02, 0.02, 1, 1), bbox_transform=ax.transAxes)
# Simplified Hainan outline
hainan_lon = [108.6, 111.0, 111.0, 108.6, 108.6]
hainan_lat = [18.2, 18.2, 20.2, 20.2, 18.2]
ax_inset.fill(hainan_lon, hainan_lat, color="#e8f4e8", edgecolor="#4a7c4a", linewidth=1)
# Haikou marker
ax_inset.scatter(110.35, 20.02, color="red", s=80, marker="*", edgecolors="darkred", linewidth=0.5, zorder=5)
ax_inset.text(110.35, 19.88, "Haikou", ha="center", fontsize=7, fontweight="bold")
# South China Sea label
ax_inset.text(109.8, 18.6, "Hainan", ha="center", fontsize=7, color="#4a7c4a")
ax_inset.set_xlim(108.3, 111.3)
ax_inset.set_ylim(17.9, 20.5)
ax_inset.set_xticks([]); ax_inset.set_yticks([])
ax_inset.set_title("Location", fontsize=7, pad=2)

# ── 5. Annotations ────────────────────────────────────────────────
ax.text(0.02, 0.97, f"Haikou Road Network\n{len(roads):,} edges\n7 GNSS trajectories\n16,155 epochs",
        transform=ax.transAxes, fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"))

ax.set_title("Hainan-06 Dataset: Haikou Road Network with 7 SPP GNSS Trajectories", fontweight="bold")

fig.savefig(OUT / "dataset_overview.png")
fig.savefig(OUT / "dataset_overview.svg")
print(f"Saved: {OUT / 'dataset_overview.png'}")
