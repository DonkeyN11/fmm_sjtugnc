#!/usr/bin/env python3
"""
Generate 1×2 case-study visualizations for CMM trustworthiness manuscript.

Follows the approach from experiments/scripts/fig_traj_tw_individual.py:
  - deg2num/num2deg tile math (no pyproj)
  - LineCollection with RdYlGn colormap
  - White road network overlay from shapefile
  - Golden start marker

Layout:
  Left panel:  TW variation curve
  Right panel: Satellite basemap + trajectory with TW-colored lines
               + road network overlay + color bar

Output: SVG + PNG for each scenario.
"""

import csv
import json
import math
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import numpy as np
import requests
from PIL import Image

# ── Paths ──
PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data/real_vehicle/hainan_06/processed"
MAP_SHP = PROJECT / "input/map/hainan/edges.shp"
REV_MAP_PATH = PROJECT / "experiments/config/reverse_edge_map.json"
OUT_DIR = DATA / "case_studies"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Matplotlib settings ──
DPI = 200
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})

# ── Slippy Map Tile Functions (Google/Bing convention) ──

def deg2num(lon, lat, zoom):
    """Lat/Lon to tile x,y at given zoom."""
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(math.radians(lat))
                + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile, ytile, zoom):
    """Tile x,y to lat/lon of NW corner."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_deg = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n))))
    return lon_deg, lat_deg


TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_tile_cache = {}


def get_satellite(west, east, south, north, zoom):
    """
    Download satellite tiles covering [west,east]×[south,north] at given zoom.
    Returns (PIL.Image, extent_tuple) where extent = (left, right, bottom, top).
    Falls back to lower zoom if >64 tiles.
    """
    x0, y0 = deg2num(west, north, zoom)   # NW corner
    x1, y1 = deg2num(east, south, zoom)    # SE corner
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    n_cols = x1 - x0 + 1
    n_rows = y1 - y0 + 1
    n_total = n_cols * n_rows
    if n_total > 64:
        return get_satellite(west, east, south, north, zoom - 1)

    print(f"    Zoom={zoom}, tiles: ({x0},{y0})→({x1},{y1}), "
          f"{n_cols}×{n_rows}={n_total} tiles")

    mosaic = Image.new('RGB', (n_cols * 256, n_rows * 256))
    for dy in range(n_rows):
        for dx in range(n_cols):
            tx, ty = x0 + dx, y0 + dy
            key = (zoom, tx, ty)
            if key not in _tile_cache:
                url = TILE_URL.format(z=zoom, y=ty, x=tx)
                try:
                    resp = requests.get(url, timeout=10,
                                        headers={'User-Agent': 'CMM-CaseStudy/1.0'})
                    if resp.status_code == 200:
                        _tile_cache[key] = Image.open(BytesIO(resp.content))
                    else:
                        _tile_cache[key] = None
                except Exception:
                    _tile_cache[key] = None
            tile = _tile_cache[key]
            if tile is not None:
                if tile.mode == 'RGBA':
                    tile = tile.convert('RGB')
                mosaic.paste(tile, (dx * 256, dy * 256))

    # Compute extent in lat/lon (left, right, bottom, top)
    left, top = num2deg(x0, y0, zoom)
    right, bottom = num2deg(x1 + 1, y1 + 1, zoom)
    return mosaic, (left, right, bottom, top)


# ── Data Loading ──

def load_data():
    """Load aligned data, ground truth, reverse edge map, and road network."""
    # Reverse edge map
    with open(REV_MAP_PATH) as f:
        rev_map = json.load(f)

    def edge_match(m, t):
        """Check if matched edge m matches ground truth t (bidirectional)."""
        return str(m) == str(t) or rev_map.get(str(m)) == str(t)

    # Ground truth
    gt_edges = {}
    with open(DATA / "ground_truth.csv") as f:
        for row in csv.DictReader(f, delimiter=";"):
            gt_edges[(row["id"].strip(), int(row["seq"]))] = row["edge_id"].strip()

    # Aligned CMM results
    traj_data = {}
    with open(DATA / "aligned.csv") as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = row["id"].strip()
            useq = int(row["uni_seq"])
            if (tid, useq) not in gt_edges:
                continue
            gt = gt_edges[(tid, useq)]
            # Skip ground truth = 0 or -1 (off-road)
            if gt in ("0", "-1"):
                continue

            x = row.get("cmm_x", "").strip()
            if not x:
                continue
            tw = float(row.get("cmm_tw", "0") or 0)
            correct = edge_match(row.get("cmm_cpath", "").strip(), gt)

            if tid not in traj_data:
                traj_data[tid] = []
            traj_data[tid].append({
                "seq": useq,
                "x": float(x),
                "y": float(row["cmm_y"]),
                "tw": tw,
                "correct": correct,
                "cpath": row.get("cmm_cpath", "").strip(),
                "gt_edge": gt,
            })

    for tid in traj_data:
        traj_data[tid].sort(key=lambda r: r["seq"])

    return traj_data


# ── Plotting Functions ──

def plot_case_study(seg, title, out_stem, figsize=(16, 6.5)):
    """
    Create 1×2 figure: left=TW curve + candidate context, right=satellite map.

    Parameters
    ----------
    seg : list of dicts with keys: seq, x, y, tw, correct
    title : str
    out_stem : Path
    """
    xs = [s["x"] for s in seg]
    ys = [s["y"] for s in seg]
    tws = np.array([s["tw"] for s in seg])
    corrects = np.array([s["correct"] for s in seg])
    seqs = [s["seq"] for s in seg]

    # ── Viewport for satellite map ──
    xs_a = np.array(xs)
    ys_a = np.array(ys)
    cx = (xs_a.min() + xs_a.max()) / 2.0
    cy = (ys_a.min() + ys_a.max()) / 2.0
    span = max(xs_a.max() - xs_a.min(), ys_a.max() - ys_a.min())
    pad = span * 0.65 + 0.0005
    lo, hi = cx - pad, cx + pad
    bo, to = cy - pad, cy + pad

    # Zoom selection: higher zoom for smaller areas
    if span < 0.001:
        zoom = 18
    elif span < 0.003:
        zoom = 17
    else:
        zoom = 16

    # ── Figure ──
    fig = plt.figure(figsize=figsize, dpi=DPI)

    # GridSpec: left panel | right panel
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                          left=0.04, right=0.90, top=0.92, bottom=0.12,
                          wspace=0.08)

    ax_tw = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])

    # ── Left Panel: TW Curve ──
    _plot_tw_curve(ax_tw, seqs, tws)

    # ── Right Panel: Satellite Map + Colorbar ──
    _plot_satellite_map(ax_map, seg, xs, ys, tws, corrects, lo, hi, bo, to, zoom)

    # ── Title ──
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.97)

    # ── Save ──
    for fmt, ext in [('svg', '.svg'), ('png', '.png')]:
        out_path = Path(str(out_stem) + ext)
        kwargs = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
        if ext == '.png':
            kwargs['dpi'] = 200
        fig.savefig(out_path, format=fmt, **kwargs)
        print(f"  Saved: {out_path}")

    plt.close(fig)


def _plot_tw_curve(ax, seqs, tws):
    """TW line plot with fill and reference lines."""
    ax.plot(seqs, tws, 'o-', color='#2c3e50', linewidth=1.8, markersize=5,
            markerfacecolor='#3498db', markeredgewidth=0, zorder=3)
    ax.fill_between(seqs, 0, tws, alpha=0.08, color='#3498db')

    # High / Low reference lines
    ax.axhline(y=0.9, color='#27ae60', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.axhline(y=0.3, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.4)

    ax.set_xlabel('Epoch Sequence', fontsize=10)
    ax.set_ylabel('Trustworthiness (TW)', fontsize=10, color='#2c3e50')
    ax.set_ylim(-0.05, 1.08)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xticks(seqs[::max(1, len(seqs) // 8)])
    ax.grid(True, alpha=0.3, linestyle='--')


def _plot_satellite_map(ax, seg, xs, ys, tws, corrects, lo, hi, bo, to, zoom):
    """Satellite basemap + road network + TW-colored trajectory lines."""
    ax.set_xlim(lo, hi)
    ax.set_ylim(bo, to)
    ax.set_aspect('equal')

    # ── Satellite basemap ──
    try:
        img, extent = get_satellite(lo, hi, bo, to, zoom)
        ax.imshow(img, extent=extent, zorder=0, alpha=0.9,
                  interpolation='bilinear')
    except Exception as e:
        print(f"    [WARN] Satellite download failed: {e}")

    # ── TW-colored trajectory (LineCollection, same as paper) ──
    norm = plt.Normalize(0, 1)
    pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segments, cmap='RdYlGn', norm=norm,
                        linewidth=6.0, zorder=3)
    lc.set_array(tws[1:])
    ax.add_collection(lc)

    # ── Start marker (golden dot) ──
    ax.scatter(xs[0], ys[0], s=80, c='#FFD700', marker='o', zorder=5,
               edgecolors='black', linewidth=1.5)

    # ── Axis labels ──
    ax.set_xlabel('Longitude (°)', fontsize=9)
    ax.set_ylabel('Latitude (°)', fontsize=9)
    ax.tick_params(labelsize=7)
    from matplotlib.ticker import ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', useOffset=False)

    # ── Colorbar (overlaid on map, using LineCollection as mappable) ──
    cbar = plt.colorbar(lc, ax=ax, shrink=0.55, aspect=20, pad=0.025)
    cbar.set_label('TW', fontsize=9, fontweight='bold')
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    traj_data = load_data()
    print(f"Loaded trajectories: {sorted(traj_data.keys())}")

    # ── Scenario 1: Simple Highway, High TW ──
    # Traj 21, uni_seq 3630 epochs total. Pick a highway section: seq 330-350
    t21 = traj_data.get('21', [])
    s1_seg = [s for s in t21 if 330 <= s['seq'] <= 350]
    if s1_seg:
        print(f"\nS1: Traj 21, {len(s1_seg)} epochs, "
              f"TW=[{min(s['tw'] for s in s1_seg):.4f}, {max(s['tw'] for s in s1_seg):.4f}], "
              f"correct={sum(s['correct'] for s in s1_seg)}")
        plot_case_study(
            s1_seg,
            'Case 1: Simple Highway — High TW (Traj 21)',
            OUT_DIR / 'case1_highway_high_tw',
        )

    # ── Scenario 2: Complex Junction, Low TW but Correct ──
    # Traj 13, uni_seq around 1714-1748
    t13 = traj_data.get('13', [])
    s2_seg = [s for s in t13 if 1714 <= s['seq'] <= 1748]
    if s2_seg:
        print(f"\nS2: Traj 13, {len(s2_seg)} epochs, "
              f"TW=[{min(s['tw'] for s in s2_seg):.4f}, {max(s['tw'] for s in s2_seg):.4f}], "
              f"correct={sum(s['correct'] for s in s2_seg)}")
        plot_case_study(
            s2_seg,
            'Case 2: Complex Junction — Low TW, Correct Match (Traj 13)',
            OUT_DIR / 'case2_complex_low_tw',
        )

    # ── Scenario 4: Wrong Match, Very Low TW ──
    # Traj 22, uni_seq around 575-640
    t22 = traj_data.get('22', [])
    s4_seg = [s for s in t22 if 575 <= s['seq'] <= 630]
    if s4_seg:
        print(f"\nS4: Traj 22, {len(s4_seg)} epochs, "
              f"TW=[{min(s['tw'] for s in s4_seg):.4f}, {max(s['tw'] for s in s4_seg):.4f}], "
              f"correct={sum(s['correct'] for s in s4_seg)}")
        plot_case_study(
            s4_seg,
            'Case 4: Parallel Road Confusion — Low TW, Wrong Match (Traj 22)',
            OUT_DIR / 'case4_wrong_low_tw',
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
