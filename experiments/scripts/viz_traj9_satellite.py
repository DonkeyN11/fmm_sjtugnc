#!/usr/bin/env python3
"""Satellite-map comparison of Trajectory 9 (sigma = 15 m): CaMM vs HMM matches.

Shows the core paper narrative: CaMM (blue) follows the correct ground-truth
route while HMM (red) locks onto wrong roads (visible as a detour beyond the
raw GNSS point cloud extent).

GT route handling:
  The simulation ground truth (ground_truth.csv / metadata.json) records Traj 9's
  true route as OSM ways 77390/77392/77393. Those IDs refer to the network
  version used at simulation time; the current input/map/hainan/edges.shp has
  since been re-exported from OSM and renumbered those same physical roads
  (the truth points still snap to them within millimetres). GT edges are
  therefore derived geometrically: ground_truth_points.csv is snapped onto the
  current network and the union of snapped edges is highlighted in green.

Outputs:
  - PNG with Esri.WorldImagery satellite basemap (contextily, WebMercator)
  - SVG without the raster basemap (clean vector lines, editable in Inkscape)

Dependencies: numpy, matplotlib, pyproj, shapely, pyshp, contextily.
"""
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely import LineString, Point, STRtree
import shapefile

try:
    import contextily as cx
    HAS_CONTEXTILY = True
except ImportError:  # pragma: no cover - env fallback
    cx = None
    HAS_CONTEXTILY = False

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse"
SIM = ROOT / "data/simulation/sigma_15/no_occlusion/no_fault"
EDGES_SHP = ROOT / "input/map/hainan/edges.shp"
OUT_PNG = DOCS / "figs/traj9_comparison.png"
OUT_SVG = DOCS / "figs_svg/traj9_comparison.svg"

TRAJ = "9"
SIGMA = 15
# True-route ways per the simulation metadata; superseded by geometric snapping
# (see module docstring) because the current network renumbered these roads.
GT_WAY_IDS = {"77390", "77392", "77393"}
SNAP_TOL_DEG = 5.0e-4          # ~55 m; truth route snaps at ~1e-8 deg
BASEMAP_ZOOM = 14              # Esri WorldImagery zoom; ~2.3 km/tile at 19.4N

# colors consistent with other paper figures (fig_ep_comparison.py)
C_CMM = "#2166ac"
C_HMM = "#b2182b"
C_GT = "#1a9850"
C_ROAD = "0.78"
C_RAW = "0.35"

DPI = 300
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
})

PT_RE = re.compile(r"POINT\s*\(([-+0-9.eE]+)\s+([-+0-9.eE]+)")


# ---------------------------------------------------------------- readers ----
def read_observations(path, traj=TRAJ):
    """Raw GNSS fixes (degrees): (x, y) arrays for one trajectory id."""
    xs, ys = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row.get("id") != traj:
                continue
            try:
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))
            except (TypeError, ValueError):
                continue
    return np.asarray(xs), np.asarray(ys)


def read_matched_points(path, traj=TRAJ):
    """Matched points parsed from the pgeom column (status == SUCCESS)."""
    xs, ys = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row.get("id") != traj or row.get("status", "SUCCESS") != "SUCCESS":
                continue
            m = PT_RE.search(row.get("pgeom", ""))
            if m:
                xs.append(float(m.group(1)))
                ys.append(float(m.group(2)))
    return np.asarray(xs), np.asarray(ys)


def read_truth_points(path, traj=TRAJ):
    """Ground-truth route points (degrees) used to derive GT edges."""
    pts = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row.get("id") != traj:
                continue
            try:
                pts.append((float(row["x"]), float(row["y"])))
            except (TypeError, ValueError):
                continue
    return np.asarray(pts)


def load_network_and_gt(shape_path, xmin, ymin, xmax, ymax, truth_pts):
    """Single pass over edges.shp: index all edges, snap truth points, filter
    edges intersecting the padded view bbox.

    Returns (roads, gt_roads) as lists of (N, 2) arrays in DEGREES, plus the
    set of GT osmid strings actually used.
    """
    reader = shapefile.Reader(str(shape_path))
    geoms, osmids = [], []
    for shape, rec in zip(reader.iterShapes(), reader.iterRecords()):
        if len(shape.points) >= 2:
            geoms.append(LineString(shape.points))
            osmids.append(str(rec["osmid"]))

    tree = STRtree(geoms)
    gt_osmids = set()
    snap_dists = []
    for px, py in truth_pts:
        idx = tree.nearest(Point(px, py))
        d = geoms[idx].distance(Point(px, py))
        snap_dists.append(d)
        if d <= SNAP_TOL_DEG:
            gt_osmids.add(osmids[idx])
    # include any GT ids still present in the current network (id-based path)
    for i, osmid in enumerate(osmids):
        tokens = {t.strip() for t in osmid.split(",") if t.strip()}
        if tokens & GT_WAY_IDS:
            gt_osmids.add(osmid)

    pad_x = 0.10 * (xmax - xmin)
    pad_y = 0.10 * (ymax - ymin)
    bx0, by0, bx1, by1 = xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y

    roads, gt_roads = [], []
    for i, geom in enumerate(geoms):
        sb = geom.bounds
        if sb[2] < bx0 or sb[0] > bx1 or sb[3] < by0 or sb[1] > by1:
            continue
        pts = np.asarray(geom.coords)
        if osmids[i] in gt_osmids:
            gt_roads.append(pts)
        else:
            roads.append(pts)

    snap_dists = np.asarray(snap_dists)
    return roads, gt_roads, gt_osmids, snap_dists


# ---------------------------------------------------------------- plotting ----
def nice_ticks(vmin, vmax, target=6):
    """Degree ticks at 'nice' multiples covering [vmin, vmax]."""
    step = 0.005
    while (vmax - vmin) / step > target:
        step *= 2
    start = np.floor(vmin / step) * step
    return np.arange(start, vmax + step / 2, step).round(8)


def draw_overlay(ax, roads, gt_roads, rx, ry, cx_, cy_, fx, fy):
    """Vector layers in WebMercator metres; shared by PNG/SVG renders."""
    for seg in roads:                       # road network, light gray
        ax.plot(seg[:, 0], seg[:, 1], color=C_ROAD, lw=0.7, zorder=2)
    for k, seg in enumerate(gt_roads):      # ground-truth route, green
        ax.plot(seg[:, 0], seg[:, 1], color=C_GT, lw=2.4, zorder=3,
                label="GT route (true)" if k == 0 else None)
    ax.scatter(rx, ry, s=5, c=C_RAW, alpha=0.55, linewidths=0, zorder=4,
               label="GNSS observations")
    ax.plot(fx, fy, color=C_HMM, lw=1.1, zorder=5)              # HMM, red
    ax.scatter(fx[::20], fy[::20], s=16, c=C_HMM, marker="^", zorder=6,
               label="HMM match")
    ax.plot(cx_, cy_, color=C_CMM, lw=1.5, zorder=7)            # CaMM, blue
    ax.scatter(cx_[::20], cy_[::20], s=14, c=C_CMM, marker="o", zorder=8,
               label="CaMM match")


def make_figure(with_basemap, tr, roads, gt_roads, rx, ry, cx_, cy_, fx, fy,
                lon_ticks, lat_ticks):
    """One figure. with_basemap=True adds the satellite imagery (PNG); the SVG
    render reuses the same layout without the raster."""
    x0, y0 = tr.transform(lon_ticks[0], lat_ticks[0])
    x1, y1 = tr.transform(lon_ticks[-1], lat_ticks[-1])
    w = x1 - x0
    h = y1 - y0

    fig, ax = plt.subplots(figsize=(8.0, 8.0 * h / w))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")

    if with_basemap and HAS_CONTEXTILY:
        for zoom in (BASEMAP_ZOOM, BASEMAP_ZOOM - 1):
            try:
                cx.add_basemap(ax, crs=None,
                               source=cx.providers.Esri.WorldImagery,
                               zoom=zoom)
                break
            except Exception as exc:  # pragma: no cover - network fallback
                print(f"  basemap zoom={zoom} failed: {exc}")
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
        else:
            print("  WARNING: no satellite basemap; plotting on white background")
    elif with_basemap:
        print("  WARNING: contextily not installed; plotting on white background")

    draw_overlay(ax, roads, gt_roads, rx, ry, cx_, cy_, fx, fy)

    lon_pos = [tr.transform(v, lat_ticks[0])[0] for v in lon_ticks]
    lat_pos = [tr.transform(lon_ticks[0], v)[1] for v in lat_ticks]
    ax.set_xticks(lon_pos)
    ax.set_yticks(lat_pos)
    ax.set_xticklabels([f"{v:.3f}°E" for v in lon_ticks])
    ax.set_yticklabels([f"{v:.3f}°N" for v in lat_ticks])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if not with_basemap:                    # grid only on the vector SVG
        ax.grid(True, ls=":", lw=0.4, alpha=0.6, zorder=1)

    ax.set_title(f"Trajectory {TRAJ}, $\\sigma$ = {SIGMA} m: "
                 "CaMM follows the correct route, HMM locks onto wrong roads",
                 fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, fancybox=True)
    fig.tight_layout()
    return fig


def main():
    print("Reading trajectory data ...")
    rx, ry = read_observations(SIM / "observations.csv")
    cx_, cy_ = read_matched_points(SIM / "cmm_result.csv")
    fx, fy = read_matched_points(SIM / "fmm_result.csv")
    truth = read_truth_points(SIM / "ground_truth_points.csv")
    print(f"  raw {len(rx)} pts | CaMM {len(cx_)} pts | HMM {len(fx)} pts | "
          f"truth {len(truth)} pts")

    xmin = min(rx.min(), cx_.min(), fx.min())
    xmax = max(rx.max(), cx_.max(), fx.max())
    ymin = min(ry.min(), cy_.min(), fy.min())
    ymax = max(ry.max(), cy_.max(), fy.max())

    print("Loading road network + deriving GT route ...")
    roads, gt_roads, gt_osmids, snap = load_network_and_gt(
        EDGES_SHP, xmin, ymin, xmax, ymax, truth)
    print(f"  {len(roads)} normal edges, {len(gt_roads)} GT edges in view")
    print(f"  GT osmid strings: {sorted(gt_osmids)}")
    print(f"  truth snap distance: median {np.median(snap):.2e} deg, "
          f"max {snap.max():.2e} deg (tol {SNAP_TOL_DEG})")

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def to_m(xs, ys):
        return tr.transform(np.asarray(xs), np.asarray(ys))

    roads_m = [np.column_stack(to_m(s[:, 0], s[:, 1])) for s in roads]
    gt_m = [np.column_stack(to_m(s[:, 0], s[:, 1])) for s in gt_roads]
    rx_m, ry_m = to_m(rx, ry)
    cx_m, cy_m = to_m(cx_, cy_)
    fx_m, fy_m = to_m(fx, fy)

    lon_ticks = nice_ticks(xmin, xmax)
    lat_ticks = nice_ticks(ymin, ymax)

    print("Rendering PNG (satellite basemap) ...")
    fig_png = make_figure(True, tr, roads_m, gt_m, rx_m, ry_m, cx_m, cy_m,
                          fx_m, fy_m, lon_ticks, lat_ticks)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig_png.savefig(OUT_PNG, dpi=DPI)
    plt.close(fig_png)
    print(f"Saved {OUT_PNG}")

    print("Rendering SVG (vector, no basemap) ...")
    fig_svg = make_figure(False, tr, roads_m, gt_m, rx_m, ry_m, cx_m, cy_m,
                          fx_m, fy_m, lon_ticks, lat_ticks)
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig_svg.savefig(OUT_SVG, format="svg")
    plt.close(fig_svg)
    print(f"Saved {OUT_SVG}")


if __name__ == "__main__":
    main()
