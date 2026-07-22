#!/usr/bin/env python3
"""Generate per-trajectory TW visualization figures for all 7 real-vehicle trajectories.
Each figure: 1×2 layout — left: TW timeline, right: satellite map of lowest-TW region.
"""

import csv, json, math, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import shapefile, requests
from PIL import Image
from io import BytesIO

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data/real_vehicle/processed"
MAP_SHP = PROJECT / "input/map/hainan/edges.shp"
OUT_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs/traj_tw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight"})

# ── Satellite tile helpers ────────────────────────────────────────────
def deg2num(lon, lat, zoom):
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return xtile, ytile

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon = xtile / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n))))
    return lon, lat

TILE_CACHE = {}
def get_satellite_background(lo, hi, bo, to, zoom):
    x0, y0 = deg2num(lo, bo, zoom)
    x1, y1 = deg2num(hi, to, zoom)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    cols, rows = x1 - x0 + 1, y1 - y0 + 1
    if cols * rows > 64:
        return get_satellite_background(lo, hi, bo, to, zoom - 1)
    stitched = Image.new('RGB', (cols * 256, rows * 256))
    for dy in range(rows):
        for dx in range(cols):
            tx, ty = x0 + dx, y0 + dy
            key = (zoom, tx, ty)
            if key not in TILE_CACHE:
                url = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}'
                try:
                    resp = requests.get(url, timeout=5)
                    TILE_CACHE[key] = Image.open(BytesIO(resp.content)) if resp.status_code == 200 else None
                except Exception:
                    TILE_CACHE[key] = None
            tile = TILE_CACHE[key]
            if tile:
                stitched.paste(tile, (dx * 256, dy * 256))
    left, top = num2deg(x0, y0, zoom)
    right, bottom = num2deg(x1 + 1, y1 + 1, zoom)
    return stitched, (left, right, bottom, top)

# ── Load road network ─────────────────────────────────────────────────
sf = shapefile.Reader(str(MAP_SHP))
all_roads = [np.array(s.points) for s in sf.shapes() if len(s.points) >= 2]
print(f"Loaded {len(all_roads)} road segments")

# ── Load aligned data + ground truth ──────────────────────────────────
REV = json.load(open(PROJECT / "experiments/config/reverse_edge_map.json"))
def em(m, t):
    return str(m) == str(t) or REV.get(str(m)) == str(t)

gt_edges = {}
with open(DATA / "ground_truth.csv") as f:
    for row in csv.DictReader(f, delimiter=";"):
        gt_edges[(row["id"].strip(), int(row["seq"]))] = row["edge_id"].strip()

traj_data = {}
with open(DATA / "aligned.csv") as f:
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
            "seq": useq,
            "x": float(x), "y": float(row["cmm_y"]),
            "ox": float(row["obs_x"]), "oy": float(row["obs_y"]),
            "tw": tw,
            "correct": em(row.get("cmm_cpath", "").strip(), gt_eid),
        })

for tid in sorted(traj_data.keys(), key=int):
    traj_data[tid].sort(key=lambda r: r["seq"])
    print(f"Traj {tid}: {len(traj_data[tid])} epochs, "
          f"TW [{min(r['tw'] for r in traj_data[tid]):.4f}, {max(r['tw'] for r in traj_data[tid]):.4f}], "
          f"acc={np.mean([r['correct'] for r in traj_data[tid]])*100:.1f}%")

# ── Generate per-trajectory figures ───────────────────────────────────
def find_lowest_tw_window(seg, window=50):
    """Find the window with the lowest mean TW."""
    if len(seg) < window:
        return 0, len(seg)
    tws = np.array([s["tw"] for s in seg])
    best_start, best_mean = 0, float('inf')
    for i in range(len(seg) - window + 1):
        m = tws[i:i+window].mean()
        if m < best_mean:
            best_mean, best_start = m, i
    return best_start, best_start + window

for tid in sorted(traj_data.keys(), key=int):
    seg = traj_data[tid]
    seqs = [r["seq"] for r in seg]
    tws = np.array([r["tw"] for r in seg])
    corrects = np.array([r["correct"] for r in seg], dtype=float)

    # Find the lowest-TW window for the map panel
    ws, we = find_lowest_tw_window(seg, window=min(80, len(seg)))
    map_seg = seg[ws:we]

    fig, (ax_timeline, ax_map) = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: TW timeline ──
    colors_t = ["#27AE60" if c else "#C0392B" for c in corrects]
    ax_timeline.scatter(seqs, tws, c=colors_t, s=8, alpha=0.6, edgecolors="none")
    ax_timeline.plot(seqs, tws, color="#888888", alpha=0.3, lw=0.5)
    ax_timeline.axhline(y=0.5, color="gray", ls="--", lw=0.8)
    # Highlight the map window
    if we > ws:
        ax_timeline.axvspan(seqs[ws], seqs[we-1], alpha=0.15, color="#2471A3")
        ax_timeline.text((seqs[ws] + seqs[we-1]) / 2, 0.02, "map region →",
                         ha="center", fontsize=7, color="#2471A3")

    ax_timeline.set_xlabel("Sequence"); ax_timeline.set_ylabel("Trustworthiness")
    mean_tw, acc = float(np.mean(tws)), float(np.mean(corrects)) * 100
    ax_timeline.set_title(f"Traj {tid}: TW Timeline | mean TW={mean_tw:.3f} | acc={acc:.1f}%")
    ax_timeline.set_ylim(-0.02, 1.05)
    ax_timeline.grid(alpha=0.3)

    # ── Right: Satellite map of lowest-TW region ──
    xs = [r["x"] for r in map_seg]; ys = [r["y"] for r in map_seg]
    oxs = [r["ox"] for r in map_seg]; oys = [r["oy"] for r in map_seg]
    tws_map = np.array([r["tw"] for r in map_seg])

    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    pad = 0.004
    lo, hi = cx - pad, cx + pad; bo, to = cy - pad, cy + pad
    ax_map.set_xlim(lo, hi); ax_map.set_ylim(bo, to); ax_map.set_aspect("equal")

    # Satellite background
    try:
        span = max(hi - lo, to - bo)
        zoom = 18 if span < 0.002 else (17 if span < 0.005 else 16)
        img, extent = get_satellite_background(lo, hi, bo, to, zoom)
        ax_map.imshow(img, extent=extent, zorder=0, alpha=0.9, interpolation='bilinear')
    except Exception:
        pass

    # Roads
    for r in all_roads:
        rx, ry = float(np.mean(r[:, 0])), float(np.mean(r[:, 1]))
        if lo <= rx <= hi and bo <= ry <= to:
            ax_map.plot(r[:, 0], r[:, 1], color="#ffffff", lw=1.2, alpha=0.45, zorder=1)

    # GNSS observations
    ax_map.scatter(oxs, oys, s=20, c="#00FFFF", alpha=0.6, zorder=5, ec="white", lw=0.8)
    ax_map.plot(oxs, oys, color="#00FFFF", alpha=0.35, lw=1.5, zorder=2, ls="--")

    # Matched path colored by TW
    norm = plt.Normalize(0, 1)
    pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments_lc = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segments_lc, cmap="RdYlGn", norm=norm, linewidth=10.0, zorder=3)
    lc.set_array(tws_map[1:])
    ax_map.add_collection(lc)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_map, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label("tw$_t$", fontsize=8); cbar.ax.tick_params(labelsize=7)

    # Start/end markers
    ax_map.scatter(xs[0], ys[0], s=150, c="#FFD700", marker="o", zorder=6, ec="black", lw=2.0)
    ax_map.scatter(xs[-1], ys[-1], s=120, c="#FFD700", marker="s", zorder=6, ec="black", lw=1.5)

    map_mean = float(np.mean(tws_map))
    map_corr = float(np.mean([r["correct"] for r in map_seg])) * 100
    ax_map.set_xticks([]); ax_map.set_yticks([])
    ax_map.set_title(f"Lowest TW region (seq {map_seg[0]['seq']}–{map_seg[-1]['seq']}) | "
                     f"TW={map_mean:.3f} | {map_corr:.0f}% correct", fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / f"traj{tid}_tw.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out} ({fig.get_size_inches()[0]:.0f}×{fig.get_size_inches()[1]:.0f} in)")

print(f"\nDone. All figures saved to {OUT_DIR}/")
