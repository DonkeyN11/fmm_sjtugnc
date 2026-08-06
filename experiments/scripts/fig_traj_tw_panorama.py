#!/usr/bin/env python3
"""
Generate traj_tw_panorama figure — panoramic view of all 7 real-vehicle trajectories.

Layout (3 rows × 4 columns):
  Row 1: Traj 11 | Traj 12 | Traj 13 | Traj 14
  Row 2:     Haikou Road Network Overview (all 7 trajectories)
  Row 3: Traj 21 | Traj 22 | Traj 23 | Colorbar / Legend

Each trajectory panel:
  - Satellite basemap (ESRI World Imagery)
  - Road network (white lines)
  - TW-colored matched path (RdYlGn, LineCollection)
  - Gold dashed ellipse around lowest-TW region
  - Inset showing local road detail at lowest-TW spot
  - Start/end markers
  - Title with trajectory ID, mean TW, accuracy

Uses aligned_0729.csv (direction-aware emission, July 29, 2026).
"""

import csv
import json
import math
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import requests
from PIL import Image
import shapefile

# ── Paths ──
PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data/real_vehicle/hainan_06/processed"
MAP_SHP = PROJECT / "input/map/hainan/edges.shp"
REV_MAP = PROJECT / "experiments/config/reverse_edge_map.json"
OUT_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 200
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "figure.dpi": DPI, "savefig.dpi": DPI,
})

# ── Tile helpers ─────────────────────────────────────────────────────────
def deg2num(lon, lat, zoom):
    n = 2 ** zoom
    return int((lon + 180.0) / 360.0 * n), int((1.0 - math.log(math.tan(math.radians(lat))
                + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    return xtile / n * 360.0 - 180.0, math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n))))

TILE_CACHE = {}
def get_sat(lo, hi, bo, to, zoom):
    x0, y0 = deg2num(lo, bo, zoom); x1, y1 = deg2num(hi, to, zoom)
    x0, x1 = min(x0, x1), max(x0, x1); y0, y1 = min(y0, y1), max(y0, y1)
    c, r = x1 - x0 + 1, y1 - y0 + 1
    if c * r > 64:
        return get_sat(lo, hi, bo, to, zoom - 1)
    st = Image.new('RGB', (c * 256, r * 256))
    for dy in range(r):
        for dx in range(c):
            tx, ty = x0 + dx, y0 + dy; k = (zoom, tx, ty)
            if k not in TILE_CACHE:
                url = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}'
                try:
                    resp = requests.get(url, timeout=10, headers={'User-Agent': 'CMM-Panorama/1.0'})
                    TILE_CACHE[k] = Image.open(BytesIO(resp.content)) if resp.status_code == 200 else None
                except Exception:
                    TILE_CACHE[k] = None
            if TILE_CACHE[k]:
                st.paste(TILE_CACHE[k], (dx * 256, dy * 256))
    l, t = num2deg(x0, y0, zoom); r, b = num2deg(x1 + 1, y1 + 1, zoom)
    return st, (l, r, b, t)


# ── Data loading ─────────────────────────────────────────────────────────
def load_data():
    sf = shapefile.Reader(str(MAP_SHP))
    all_roads = [np.array(s.points) for s in sf.shapes() if len(s.points) >= 2]
    print(f"Loaded {len(all_roads)} road segments")

    with open(REV_MAP) as f:
        rev_map = json.load(f)

    def em(m, t):
        return str(m) == str(t) or rev_map.get(str(m)) == str(t)

    gt_edges = {}
    with open(DATA / "ground_truth.csv") as f:
        for row in csv.DictReader(f, delimiter=";"):
            gt_edges[(row["id"].strip(), int(row["seq"]))] = row["edge_id"].strip()

    # Use aligned_0729.csv (latest, with direction-aware emission)
    aligned_path = DATA / "aligned_0729.csv"
    print(f"Loading: {aligned_path}")

    traj_data = {}
    with open(aligned_path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = row["id"].strip()
            useq = int(row["uni_seq"])
            if (tid, useq) not in gt_edges:
                continue
            gt_eid = gt_edges[(tid, useq)]
            if gt_eid in ("0", "-1"):
                continue
            x = row.get("cmm_x", "").strip()
            if not x:
                continue
            tw = float(row.get("cmm_tw", "0") or 0)
            if tid not in traj_data:
                traj_data[tid] = []
            traj_data[tid].append({
                "seq": useq, "x": float(x), "y": float(row["cmm_y"]),
                "ox": float(row["obs_x"]), "oy": float(row["obs_y"]),
                "tw": tw,
                "correct": em(row.get("cmm_cpath", "").strip(), gt_eid),
            })

    for tid in sorted(traj_data.keys(), key=int):
        traj_data[tid].sort(key=lambda r: r["seq"])
        tws = [r["tw"] for r in traj_data[tid]]
        acc = sum(1 for r in traj_data[tid] if r["correct"]) / len(traj_data[tid]) * 100
        print(f"  Traj {tid}: {len(traj_data[tid])} eps, TW [{min(tws):.4f}, {max(tws):.4f}], acc={acc:.1f}%")

    return traj_data, all_roads


# ── Lowest-TW window finder ──────────────────────────────────────────────
def find_lowest(seg, w=50):
    if len(seg) < w:
        return 0, len(seg)
    tws = np.array([s["tw"] for s in seg])
    bs, bm = 0, float('inf')
    for i in range(len(seg) - w + 1):
        m = tws[i:i + w].mean()
        if m < bm:
            bm, bs = m, i
    return bs, bs + w


# ── Individual trajectory panel ──────────────────────────────────────────
R50M = 50.0 / 111320.0  # ~50m in degrees

def plot_traj_panel(ax, seg, tid, all_roads):
    """Plot one trajectory's satellite map on the given axes."""
    if not seg:
        return
    xs = [s["x"] for s in seg]; ys = [s["y"] for s in seg]
    tws = np.array([s["tw"] for s in seg])
    corrects = np.array([s["correct"] for s in seg])
    min_idx = int(np.argmin(tws))
    mtx, mty = xs[min_idx], ys[min_idx]
    ws, we = find_lowest(seg, min(60, len(seg)))
    low_seg = seg[ws:we]

    # Viewport
    xs_a, ys_a = np.array(xs), np.array(ys)
    cx = (xs_a.min() + xs_a.max()) / 2.0
    cy = (ys_a.min() + ys_a.max()) / 2.0
    pad = max(xs_a.max() - xs_a.min(), ys_a.max() - ys_a.min()) * 0.65 + 0.001
    lo, hi = cx - pad, cx + pad
    bo, to = cy - pad, cy + pad

    ax.set_xlim(lo, hi); ax.set_ylim(bo, to); ax.set_aspect("equal")

    # Satellite basemap
    try:
        span = max(hi - lo, to - bo)
        z = 18 if span < 0.002 else (17 if span < 0.005 else 16)
        img, ext = get_sat(lo, hi, bo, to, z)
        ax.imshow(img, extent=ext, zorder=0, alpha=0.9, interpolation='bilinear')
    except Exception:
        pass

    # Road network
    for r in all_roads:
        rmx, rmx2 = float(r[:, 0].min()), float(r[:, 0].max())
        rmy, rmy2 = float(r[:, 1].min()), float(r[:, 1].max())
        if rmx <= hi and rmx2 >= lo and rmy <= to and rmy2 >= bo:
            ax.plot(r[:, 0], r[:, 1], color="#ffffff", lw=0.8, alpha=0.35, zorder=1)

    # TW-colored matched path
    norm = plt.Normalize(0, 1)
    pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1),
                        cmap="RdYlGn", norm=norm, linewidth=5.0, zorder=3)
    lc.set_array(tws[1:])
    ax.add_collection(lc)

    # Gold ellipse around lowest-TW spot
    ell = Ellipse((mtx, mty), R50M * 2, R50M * 2,
                  facecolor='none', edgecolor='#FFD700',
                  linewidth=3.5, linestyle='--', zorder=6)
    ax.add_patch(ell)

    # Start marker
    ax.scatter(xs[0], ys[0], s=60, c="#FFD700", marker="o", zorder=5,
               edgecolors="black", lw=1.2)

    # Inset: zoomed view of lowest-TW region
    ir = R50M * 2.5
    axi = inset_axes(ax, width="28%", height="28%", loc="lower left",
                     bbox_to_anchor=(0.02, 0.02, 1, 1),
                     bbox_transform=ax.transAxes)
    axi.set_xlim(mtx - ir, mtx + ir); axi.set_ylim(mty - ir, mty + ir)
    axi.set_aspect("equal")
    for r in all_roads:
        rmx, rmx2 = float(r[:, 0].min()), float(r[:, 0].max())
        rmy, rmy2 = float(r[:, 1].min()), float(r[:, 1].max())
        if rmx <= mtx + ir and rmx2 >= mtx - ir and rmy <= mty + ir and rmy2 >= mty - ir:
            axi.plot(r[:, 0], r[:, 1], color="#555555", lw=1.0, alpha=0.6)
    axi.scatter(xs, ys, c=tws, cmap="RdYlGn", norm=norm, s=10, zorder=5, edgecolors="none")
    axi.scatter(mtx, mty, s=25, c="none", marker="o", edgecolors="#FFD700",
                linewidth=1.8, zorder=6)
    axi.set_xticks([]); axi.set_yticks([])
    ml = np.mean([s["tw"] for s in low_seg])
    axi.set_title(f"Min TW={ml:.3f}", fontsize=5, color="#C0392B", pad=1)

    ax.set_xticks([]); ax.set_yticks([])
    mt = np.mean(tws); acc = np.mean(corrects) * 100
    ax.set_title(f"Traj {tid}  |  TW={mt:.3f}  |  Acc={acc:.0f}%  |  {len(seg)} ep",
                 fontsize=8, fontweight="bold")


# ── Overview panel ───────────────────────────────────────────────────────
def plot_overview_panel(ax, traj_data, all_roads):
    """Plot all 7 trajectories on the Haikou road network with satellite."""
    COLS = plt.cm.tab10(np.linspace(0, 1, 7))

    # Compute combined extent
    all_xs, all_ys = [], []
    for tid in sorted(traj_data.keys(), key=int):
        seg = traj_data[tid]
        all_xs.extend(s["x"] for s in seg)
        all_ys.extend(s["y"] for s in seg)

    xs_a, ys_a = np.array(all_xs), np.array(all_ys)
    cx = (xs_a.min() + xs_a.max()) / 2.0
    cy = (ys_a.min() + ys_a.max()) / 2.0
    pad = max(xs_a.max() - xs_a.min(), ys_a.max() - ys_a.min()) * 0.15 + 0.01
    lo, hi = cx - pad, cx + pad
    bo, to = cy - pad, cy + pad

    ax.set_xlim(lo, hi); ax.set_ylim(bo, to); ax.set_aspect("equal")

    # Satellite basemap (use zoom 12 or 13 for full Haikou)
    try:
        span = max(hi - lo, to - bo)
        z = 13 if span > 0.2 else 14
        img, ext = get_sat(lo, hi, bo, to, z)
        ax.imshow(img, extent=ext, zorder=0, alpha=0.8, interpolation='bilinear')
    except Exception:
        pass

    # Road network
    for r in all_roads:
        rmx, rmx2 = float(r[:, 0].min()), float(r[:, 0].max())
        rmy, rmy2 = float(r[:, 1].min()), float(r[:, 1].max())
        if rmx <= hi and rmx2 >= lo and rmy <= to and rmy2 >= bo:
            ax.plot(r[:, 0], r[:, 1], color="#cccccc", lw=0.3, alpha=0.5, zorder=1)

    # Plot each trajectory
    for i, tid in enumerate(sorted(traj_data.keys(), key=int)):
        seg = traj_data[tid]
        xs = np.array([s["x"] for s in seg])
        ys = np.array([s["y"] for s in seg])
        ax.plot(xs, ys, color=COLS[i], lw=1.8, alpha=0.85, zorder=3,
                label=f"T{tid}", solid_capstyle='round')
        # Start marker
        ax.scatter(xs[0], ys[0], s=50, c=COLS[i], marker="o", zorder=5,
                   edgecolors="white", lw=1.0)

    ax.legend(loc="upper right", ncol=4, fontsize=7, framealpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Haikou Road Network — All 7 Trajectories Overview",
                 fontsize=10, fontweight="bold")


# ── Colorbar panel ───────────────────────────────────────────────────────
def plot_colorbar_panel(ax):
    """imshow-based RdYlGn colorbar."""
    gradient = np.linspace(1, 0, 256).reshape(256, 1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn', extent=[0, 1, 0, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.yaxis.set_ticks_position('right')
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.0', '0.25', '0.50', '0.75', '1.0'], fontsize=8)
    ax.yaxis.set_label_position('right')
    ax.set_ylabel('TW', fontsize=10, fontweight='bold', rotation=0,
                  labelpad=18, va='center')


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    traj_data, all_roads = load_data()
    TRAJ_IDS = sorted(traj_data.keys(), key=int)
    print(f"Trajectories: {TRAJ_IDS}")

    # Figure: 3 rows × 4 columns
    fig = plt.figure(figsize=(28, 20), dpi=DPI)

    # GridSpec layout
    gs = fig.add_gridspec(3, 4, hspace=0.25, wspace=0.15,
                          left=0.02, right=0.95, top=0.96, bottom=0.03)

    # Row 1: Traj 11, 12, 13, 14
    for col, tid in enumerate([11, 12, 13, 14]):
        tid_str = str(tid)
        if tid_str in traj_data:
            ax = fig.add_subplot(gs[0, col])
            plot_traj_panel(ax, traj_data[tid_str], tid, all_roads)
            print(f"  Panel: Traj {tid} (row 1, col {col+1})")

    # Row 2: Overview (spans all 4 columns)
    ax_overview = fig.add_subplot(gs[1, :])
    plot_overview_panel(ax_overview, traj_data, all_roads)
    print(f"  Panel: Overview (row 2, cols 1-4)")

    # Row 3: Traj 21, 22, 23 + Colorbar
    for col, tid in enumerate([21, 22, 23]):
        tid_str = str(tid)
        if tid_str in traj_data:
            ax = fig.add_subplot(gs[2, col])
            plot_traj_panel(ax, traj_data[tid_str], tid, all_roads)
            print(f"  Panel: Traj {tid} (row 3, col {col+1})")

    ax_cbar = fig.add_subplot(gs[2, 3])
    plot_colorbar_panel(ax_cbar)
    print(f"  Panel: Colorbar (row 3, col 4)")

    # Save
    for fmt, ext in [('png', '.png'), ('svg', '.svg')]:
        out_path = OUT_DIR / f"traj_tw_panorama{ext}"
        kwargs = dict(facecolor='white', edgecolor='none')
        if ext == '.png':
            kwargs['dpi'] = 150
        fig.savefig(out_path, format=fmt, **kwargs)
        print(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    plt.close(fig)
    print("\nDone!")


if __name__ == '__main__':
    main()
