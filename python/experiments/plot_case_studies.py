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

    # The label and the smoothed score both come from aligned.csv. Earlier this
    # read gt_edge out of ground_truth.csv and joined it to aligned.csv's
    # `uni_seq` positionally, which is correct only while the two files enumerate
    # the same epochs -- align_real_data.py's docstring explains why that join
    # breaks silently. aligned.csv already carries gt_edge, so the second file is
    # not needed and the positional join is gone.
    traj_data = {}
    with open(DATA / "aligned.csv") as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = row["id"].strip()
            useq = int(row["uni_seq"])
            gt = row.get("gt_edge", "").strip()
            # Skip ground truth = 0 or -1 (off-road / no label)
            if gt in ("0", "-1", ""):
                continue

            x = row.get("cmm_x", "").strip()
            if not x:
                continue
            tw = float(row.get("cmm_tw", "0") or 0)
            tw_s = row.get("cmm_tw_smoothed", "").strip()
            correct = edge_match(row.get("cmm_cpath", "").strip(), gt)

            if tid not in traj_data:
                traj_data[tid] = []
            traj_data[tid].append({
                "seq": useq,
                "x": float(x),
                "y": float(row["cmm_y"]),
                "gx": float(row["gt_x"]) if row.get("gt_x", "").strip() else None,
                "gy": float(row["gt_y"]) if row.get("gt_y", "").strip() else None,
                "tw": tw,
                "tw_smoothed": float(tw_s) if tw_s else tw,
                "correct": correct,
                "cpath": row.get("cmm_cpath", "").strip(),
                "gt_edge": gt,
            })

    for tid in traj_data:
        traj_data[tid].sort(key=lambda r: r["seq"])

    return traj_data


# ── Plotting Functions ──

CASE_WINDOWS = [
    ('21', 330, 350, 'Case 1: straight highway $\\to$ saturated',
     'all 21 epochs correct, tw = 1.000'),
    ('13', 1714, 1748, 'Case 2: interchange $\\to$ false alarm',
     'all 35 correct; filtering falls to 0.025'),
    ('11', 1730, 1749, 'Case 3: slip road $\\to$ confident error',
     'all 20 wrong, tw = 1.000'),
]


def plot_case_composite(traj_data, out_stem, figsize=(7.16, 6.2)):
    """One 2x3 figure: the three case windows side by side, curves above maps below.

    Replaces three separate 1x2 figures. In a two-column layout those cost about 1.6
    pages for content that is read as a set, so the panels are stacked into one figure
    that spans both columns.
    """
    import numpy as np
    fig = plt.figure(figsize=figsize, dpi=DPI)
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.55,
                          left=0.075, right=0.895, top=0.90, bottom=0.075)
    ax_cbar = fig.add_axes([0.925, 0.30, 0.016, 0.38])
    for col, (tid, lo, hi, title, sub) in enumerate(CASE_WINDOWS):
        seg = [s for s in traj_data.get(tid, []) if lo <= s['seq'] <= hi]
        if not seg:
            continue
        xs = [s['x'] for s in seg]
        ys = [s['y'] for s in seg]
        tws = np.array([s['tw'] for s in seg])
        tws_s = np.array([s['tw_smoothed'] for s in seg])
        corrects = np.array([s['correct'] for s in seg])
        seqs = [s['seq'] for s in seg]
        ax_tw = fig.add_subplot(gs[0, col])
        _plot_tw_curve(ax_tw, seqs, tws, tws_s)
        ax_tw.set_title(title, fontsize=7.5, pad=3)
        ax_tw.set_xlabel(sub, fontsize=6.0)
        ax_tw.set_xticks([])
        if col:
            ax_tw.set_ylabel('')
        ax_map = fig.add_subplot(gs[1, col])
        gts = [(s['gx'], s['gy']) for s in seg
               if s.get('gx') is not None and s.get('gy') is not None]
        _plot_satellite_map_sized(ax_map, xs, ys, tws, lo, hi, gts)
        if col:
            ax_map.set_ylabel('')
    _plot_colorbar(ax_cbar)
    for ext in ('png', 'svg'):
        out = Path(str(out_stem) + '.' + ext)
        fig.savefig(out, format=ext, facecolor='white', dpi=DPI)
        print(f'  Saved: {out}')
    plt.close(fig)


def _plot_satellite_map_sized(ax, xs, ys, tws, lo, hi, gts):
    """The satellite panel of the composite: same viewport rule as the standalone one."""
    import numpy as np
    xs_a, ys_a = np.array(xs), np.array(ys)
    cx = (xs_a.min() + xs_a.max()) / 2.0
    cy = (ys_a.min() + ys_a.max()) / 2.0
    span = max(xs_a.max() - xs_a.min(), ys_a.max() - ys_a.min())
    pad = span * 0.65 + 0.0005
    zoom = 18 if span < 0.001 else (17 if span < 0.003 else 16)
    _plot_satellite_map(ax, None, xs, ys, tws, None, cx - pad, cx + pad,
                        cy - pad, cy + pad, zoom, gts)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # Longitude ticks at three significant figures collide at this panel width, so they
    # are reduced to two and thinned; latitude labels stay on the left column only.
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.tick_params(labelsize=5.0, pad=1.5)


def plot_case_study(seg, title, out_stem, figsize=(16, 6.5)):
    """
    Create 1×2 figure: left=TW curve + candidate context, right=satellite map.

    Parameters
    ----------
    seg : list of dicts with keys: seq, x, y, tw, tw_smoothed, correct
    title : str
    out_stem : Path
    """
    xs = [s["x"] for s in seg]
    ys = [s["y"] for s in seg]
    tws = np.array([s["tw"] for s in seg])
    tws_s = np.array([s["tw_smoothed"] for s in seg])
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

    # GridSpec: left panel | right panel, leave room for colorbar on right
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                          left=0.04, right=0.88, top=0.92, bottom=0.12,
                          wspace=0.08)

    ax_tw = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])

    # Colorbar axes: wider strip on far right (must be wide enough to render)
    ax_cbar = fig.add_axes([0.905, 0.22, 0.035, 0.56])

    # ── Left Panel: TW Curve ──
    _plot_tw_curve(ax_tw, seqs, tws, tws_s)

    # ── Right Panel: Satellite Map ──
    gts = [(s["gx"], s["gy"]) for s in seg
           if s.get("gx") is not None and s.get("gy") is not None]
    _plot_satellite_map(ax_map, seg, xs, ys, tws, corrects, lo, hi, bo, to, zoom, gts)

    # ── Colorbar (imshow-based, the only approach that works in mpl 3.11) ──
    _plot_colorbar(ax_cbar)

    # ── Title ──
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.97)

    # ── Save ──
    for fmt, ext, use_tight in [('svg', '.svg', False), ('png', '.png', False)]:
        out_path = Path(str(out_stem) + ext)
        kwargs = dict(facecolor='white', edgecolor='none')
        if use_tight:
            kwargs['bbox_inches'] = 'tight'
        if ext == '.png':
            kwargs['dpi'] = 200
        fig.savefig(out_path, format=fmt, **kwargs)
        print(f"  Saved: {out_path}")

    plt.close(fig)


def _plot_tw_curve(ax, seqs, tws, tws_s=None):
    """TW line plot with fill and reference lines.

    Both arms are drawn. They share a path and differ only in their evidence set,
    so the gap between them IS the value of the whole-trajectory evidence, and on
    the Case-2 window that gap is the whole point of the panel.
    """
    if tws_s is not None and not np.allclose(tws, tws_s):
        ax.plot(seqs, tws_s, 's-', color='#c0392b', linewidth=1.8, markersize=4.5,
                markerfacecolor='#e67e22', markeredgewidth=0, zorder=4,
                label='smoothed  $\\alpha_t\\beta_t$')
    ax.plot(seqs, tws, 'o-', color='#2c3e50', linewidth=1.8, markersize=5,
            markerfacecolor='#3498db', markeredgewidth=0, zorder=3,
            label='filtering  $\\alpha_t$')
    ax.fill_between(seqs, 0, tws, alpha=0.08, color='#3498db')

    # High / Low reference lines
    ax.axhline(y=0.9, color='#27ae60', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.axhline(y=0.3, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.4)

    ax.set_xlabel('Epoch Sequence', fontsize=10)
    ax.set_ylabel('Trustworthiness (TW)', fontsize=10, color='#2c3e50')
    if tws_s is not None and not np.allclose(tws, tws_s):
        ax.legend(loc='center left', fontsize=7.5, framealpha=0.9)
    ax.set_ylim(-0.05, 1.08)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xticks(seqs[::max(1, len(seqs) // 8)])
    ax.grid(True, alpha=0.3, linestyle='--')


def _plot_satellite_map(ax, seg, xs, ys, tws, corrects, lo, hi, bo, to, zoom, gts=None):
    """Satellite basemap + road network + TW-colored trajectory lines.

    The ground-truth trace is drawn underneath when it is available. Without it the
    panel shows only where the matcher put the vehicle, so a window whose whole
    point is that the match is WRONG reads as a competent-looking drive: on
    Case 3 the two lines separate by up to 50 m and the eye needs both of them.
    """
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

    # ── Ground truth (underneath, so it stays visible where the two coincide) ──
    if gts:
        ax.plot([p[0] for p in gts], [p[1] for p in gts], '--',
                color='#000000', linewidth=2.2, alpha=0.95, zorder=6,
                label='RTK ground truth')
        ax.legend(loc='lower right', fontsize=7.5, framealpha=0.85)

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


def _plot_colorbar(ax):
    """imshow-based RdYlGn colorbar (the only approach that renders in mpl 3.11)."""
    gradient = np.linspace(1, 0, 256).reshape(256, 1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn',
              extent=[0, 1, 0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.yaxis.set_ticks_position('right')
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.0', '0.25', '0.50', '0.75', '1.0'], fontsize=7)
    ax.yaxis.set_label_position('right')
    ax.set_ylabel('TW', fontsize=9, fontweight='bold', rotation=0,
                  labelpad=12, va='center')


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
    # Traj 13, uni_seq 1714-1748. All 35 epochs are matched correctly while the
    # FILTERING score collapses over ~29 consecutive epochs and bottoms out at
    # 0.025. It used to be described as a six-epoch dip; at the current operating
    # point the depressed stretch is the whole middle of the window, and what the
    # panel shows is that the smoothed arm largely removes it.
    t13 = traj_data.get('13', [])
    s2_seg = [s for s in t13 if 1714 <= s['seq'] <= 1748]
    if s2_seg:
        print(f"\nS2: Traj 13, {len(s2_seg)} epochs, "
              f"TW=[{min(s['tw'] for s in s2_seg):.4f}, {max(s['tw'] for s in s2_seg):.4f}], "
              f"correct={sum(s['correct'] for s in s2_seg)}")
        plot_case_study(
            s2_seg,
            'Case 2: Interchange — Filtering Collapse, Correct Match (Traj 13)',
            OUT_DIR / 'case2_complex_low_tw',
        )

    # ── Scenario 3: Wrong Match, HIGH TW ──
    # Traj 11, uni_seq 1730-1749: the vehicle takes a slip road (edge 11152) and
    # the matcher stays on the main carriageway (edge 101366) for the whole run.
    # Every one of the 20 epochs is wrong, and trustworthiness is 1.000 on all of
    # them in BOTH arms -- beta cannot rescue this one, because the wrong path is
    # confidently wrong over its whole length rather than briefly uncertain. The
    # observation is 0.07-3.3 m from the labelled slip road and 11-50 m from the
    # matched carriageway, so the geometry is decisive: this is a matcher error,
    # not a ground-truth defect.
    #
    # This replaces the Traj 22 window (uni_seq 575-630) that used to carry the
    # case. Under the current observation model that window is 55 of 56 correct
    # with no confident error left in it: the map-registration term and the
    # corrected heading term between them removed the failure it illustrated, so
    # keeping it would have meant illustrating a regime the model no longer has.
    # The two are not the same geometry -- this one is ramp against mainline, the
    # old one was carriageway against carriageway -- so the caption names the
    # geometry rather than claiming the parallel-road case.
    #
    t11 = traj_data.get('11', [])
    s3_seg = [s for s in t11 if 1730 <= s['seq'] <= 1749]
    if s3_seg:
        print(f"\nS3: Traj 11, {len(s3_seg)} epochs, "
              f"TW=[{min(s['tw'] for s in s3_seg):.4f}, {max(s['tw'] for s in s3_seg):.4f}], "
              f"correct={sum(s['correct'] for s in s3_seg)}")
        plot_case_study(
            s3_seg,
            'Case 3: Slip Road vs Main Carriageway — High TW, Wrong Match (Traj 11)',
            OUT_DIR / 'case3_parallel_high_tw_wrong',
        )

    # ── Composite of the three windows, for the compressed manuscript ──
    try:
        plot_case_composite(
            traj_data,
            OUT_DIR / 'case_composite',
        )
    except Exception as e:  # a satellite fetch failure must not lose the other figures
        print(f"  [WARN] composite failed: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
