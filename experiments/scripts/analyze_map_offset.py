#!/usr/bin/env python3
"""Map-vs-positioning offset audit on the Haikou real-vehicle dataset.

Motivation
----------
With the ECEF->ENU-corrected protection levels, 703 of 16155 epochs (4.35 %)
find *no* real road candidate inside the search radius, where the legacy
(inflated) PL had none.  One possible cause is that the road network itself is
offset from the positioning solution -- surveying error, or the map and the
receiver using different coordinate systems.  If so, no choice of search radius
below that offset can ever recover those epochs.

Method
------
Take the RTK ground truth (WGS84 lat/lon parsed from the receiver's own $GNGGA
solutions) and, for every epoch, locate the nearest point on the nearest road
edge.  The offset

    o = q_road - p_rtk

is the map-minus-positioning displacement at that epoch.

Two mechanisms produce a non-zero o and they are cleanly separable:

  * map bias (surveying / datum / CRS) -- o is fixed in the EARTH frame; the
    mean of o over many epochs with many different headings stays non-zero and
    points in a constant compass direction.
  * antenna / lane offset           -- o is fixed in the VEHICLE frame
    (roughly perpendicular to the heading, magnitude ~ half a lane); averaged
    over headings spread over 360 deg it cancels in the earth frame.

So the script decomposes o in both frames, tests each for concentration, and
also fits a 2-D similarity transform (translation + rotation + scale) to see
whether any datum-level misregistration survives.

It also cross-checks against the SPP solutions, and against the receiver's own
GGA fix quality, because a float or autonomous solution is itself several
metres off and would masquerade as a map offset.

Outputs (experiments/output/map_offset/):
  summary.json          -- all scalar results
  per_epoch.csv         -- per-epoch offsets, headings, fix quality, status
  grid_offset.csv       -- mean offset vector per 500 m cell (spatial coherence)
  report.txt            -- human-readable dump

Usage:
  python experiments/scripts/analyze_map_offset.py
"""

import csv
import json
import math
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from osgeo import ogr
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from shapely.strtree import STRtree

BASE = Path(__file__).resolve().parents[2]
REAL = BASE / "data/real_vehicle/hainan_06"
PROC = REAL / "processed"
EDGES = BASE / "input/map/hainan/edges.shp"
OUT = BASE / "experiments/output/map_offset"

TRAJ_MAP = {11: "1.1", 12: "1.2", 13: "1.3", 14: "1.4",
            21: "2.1", 22: "2.2", 23: "2.3"}
EXCLUDE = {"12"}          # excluded from the paper's metric run
R_EARTH = 6371008.8

# Fix-quality buckets, from the GGA field 6 indicator.
FIX_NAME = {4: "RTK_fixed", 5: "RTK_float", 2: "DGPS", 1: "autonomous"}


# ─────────────────────────────────────────────────────────────────────────────
# RTK raw parsing: recover the GGA fix quality per (traj, unix timestamp)
# ─────────────────────────────────────────────────────────────────────────────
def _dm_to_dd(s):
    try:
        v = float(s)
        d = int(v / 100)
        return d + (v - d * 100) / 60.0
    except ValueError:
        return None


def parse_rtk_quality_and_pos(filepath):
    """(traj-agnostic) -> {unix_ts: (lon, lat, fix_quality)}.

    Mirrors build_real_data_gt.py's two-pass date resolution: GGA precedes RMC
    in this receiver's stream, so the date has to be collected first.
    """
    time_to_date = {}
    current_date = None
    with open(filepath, encoding="ascii", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue
            i = line.find("$")
            if i < 0:
                continue
            fields = line[i:].split("*")[0].split(",")
            if len(fields) < 2:
                continue
            mt = fields[0][3:]
            if mt == "RMC" and len(fields) >= 10:
                try:
                    current_date = date(int(fields[9][4:6]) + 2000,
                                        int(fields[9][2:4]), int(fields[9][0:2]))
                    time_to_date[fields[1]] = current_date
                except (ValueError, IndexError):
                    pass
            elif mt == "GGA" and len(fields) >= 2 and current_date:
                time_to_date[fields[1]] = current_date

    out = {}
    with open(filepath, encoding="ascii", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue
            i = line.find("$")
            if i < 0:
                continue
            fields = line[i:].split("*")[0].split(",")
            if len(fields) < 10 or fields[0][3:] != "GGA":
                continue
            ts = fields[1]
            if ts not in time_to_date:
                continue
            lat, lon = _dm_to_dd(fields[2]), _dm_to_dd(fields[4])
            if lat is None or lon is None:
                continue
            if fields[3] == "S":
                lat = -lat
            if fields[5] == "W":
                lon = -lon
            try:
                d = time_to_date[ts]
                h, m = int(ts[0:2]), int(ts[2:4])
                s = float(ts[4:])
                dt = datetime(d.year, d.month, d.day, h, m,
                              int(s), int((s - int(s)) * 1e6), tzinfo=timezone.utc)
            except (ValueError, IndexError):
                continue
            try:
                q = int(fields[6])
            except ValueError:
                q = 0
            out[round(dt.timestamp())] = (lon, lat, q)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────
def load_gt():
    """aligned.csv -> list of epochs with the RTK position and the GT edge."""
    rows = []
    with open(PROC / "aligned.csv", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            tid = r["id"].strip()
            try:
                gx, gy = float(r["gt_x"]), float(r["gt_y"])
            except (ValueError, KeyError):
                continue
            ts = r["timestamp"].strip().rstrip("0").rstrip(".")
            rows.append({
                "traj": tid, "ts": ts,
                "lon": gx, "lat": gy,
                "gt_edge": r["gt_edge"].strip(),
                "obs_lon": float(r["obs_x"]) if r["obs_x"].strip() else None,
                "obs_lat": float(r["obs_y"]) if r["obs_y"].strip() else None,
            })
    return rows


def load_edges():
    """Read every edge as a WGS84 LineString, keeping the network's edge id.

    fmm's edge id is the feature index, which is also the shapefile's `key`
    field -- gt_edge values index into that same space.
    """
    ds = ogr.Open(str(EDGES))
    lyr = ds.GetLayer(0)
    lines, ids = [], []
    for feat in lyr:
        g = feat.GetGeometryRef()
        if g is None or g.GetPointCount() < 2:
            continue
        coords = [g.GetPoint_2D(i) for i in range(g.GetPointCount())]
        lines.append(LineString(coords))
        ids.append(int(feat.GetField("key")))
    ds = None
    return lines, ids


def load_zero_candidate_epochs():
    """Epochs whose candidate list in a CMM run holds only the background node.

    The writer serialises candidates as ((...),(...)); the background entry is
    (0,0,0.1).  An epoch with a single entry is an epoch with no real road
    candidate.  Returns {filename: set[(traj, ts)]}.
    """
    out = {}
    for fn in ("cmm_result_fixed_en.csv", "cmm_result_enu.csv"):
        p = PROC / fn
        if not p.exists():
            continue
        s = set()
        with open(p, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                cand = (r.get("candidates") or "").strip()
                if not cand:
                    continue
                # count top-level entries of the outer tuple
                depth, n = 0, 0
                for ch in cand:
                    if ch == "(":
                        depth += 1
                        if depth == 2:
                            n += 1
                    elif ch == ")":
                        depth -= 1
                if n <= 1:
                    s.add((r["id"].strip(), r["timestamp"].strip().rstrip("0").rstrip(".")))
        out[fn] = s
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def project_meta(lon, lat):
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    mlat = math.radians(float(np.mean(lat)))
    return lon, lat, mlat


def offset_stats(oe, on):
    """Earth-frame offset statistics. oe/on are east/north components (m)."""
    oe, on = np.asarray(oe, float), np.asarray(on, float)
    n = len(oe)
    if n == 0:
        return {}
    mag = np.hypot(oe, on)
    bearing = (np.degrees(np.arctan2(oe, on))) % 360.0   # compass, 0=N 90=E
    # Rayleigh test of uniformity on the offset direction
    th = np.arctan2(on, oe)
    C, S = np.cos(th).mean(), np.sin(th).mean()
    Rbar = math.hypot(C, S)
    Z = n * Rbar * Rbar
    p_rayleigh = math.exp(-Z) * (1 + (2 * Z - Z * Z) / (4 * n)
                                 - (24 * Z - 132 * Z * Z + 76 * Z ** 3 - 9 * Z ** 4) / (288 * n * n))
    return {
        "n": int(n),
        "mean_east_m": float(oe.mean()),
        "mean_north_m": float(on.mean()),
        "mean_vec_mag_m": float(math.hypot(oe.mean(), on.mean())),
        "mean_vec_bearing_deg": float(math.degrees(math.atan2(oe.mean(), on.mean())) % 360.0),
        "median_east_m": float(np.median(oe)),
        "median_north_m": float(np.median(on)),
        "median_vec_mag_m": float(math.hypot(np.median(oe), np.median(on))),
        "median_vec_bearing_deg": float(math.degrees(math.atan2(np.median(oe), np.median(on))) % 360.0),
        "std_east_m": float(oe.std(ddof=1)) if n > 1 else None,
        "std_north_m": float(on.std(ddof=1)) if n > 1 else None,
        "dist_mean_m": float(mag.mean()),
        "dist_median_m": float(np.median(mag)),
        "dist_p68_m": float(np.percentile(mag, 68)),
        "dist_p90_m": float(np.percentile(mag, 90)),
        "dist_p95_m": float(np.percentile(mag, 95)),
        "dist_max_m": float(mag.max()),
        "frac_within_5m": float((mag <= 5).mean()),
        "frac_within_10m": float((mag <= 10).mean()),
        "frac_within_20m": float((mag <= 20).mean()),
        "frac_within_30m": float((mag <= 30).mean()),
        "frac_within_50m": float((mag <= 50).mean()),
        "rayleigh_Rbar": float(Rbar),
        "rayleigh_Z": float(Z),
        "rayleigh_p": float(min(1.0, p_rayleigh)),
    }


def fit_similarity(pe, pn, qe, qn):
    """Least-squares 2-D similarity  q = T + s R(theta) p  (four parameters).

    Linearisation:  qE = TE + a pE - b pN ;  qN = TN + b pE + a pN
    with a = s cos(theta), b = s sin(theta).
    """
    A = np.column_stack([np.ones(len(pe)), np.zeros(len(pe)), pe, -pn])
    B = np.column_stack([np.zeros(len(pe)), np.ones(len(pe)), pn, pe])
    M = np.vstack([A, B])
    rhs = np.concatenate([qe, qn])
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    te, tn, a, b = sol
    s = math.hypot(a, b)
    theta = math.degrees(math.atan2(b, a))
    return te, tn, s, theta


def robust_similarity(pe, pn, qe, qn, iters=5, trim=3.0):
    """Trimmed similarity fit; returns parameters on the final inlier set."""
    keep = np.ones(len(pe), bool)
    te = tn = 0.0
    s, theta = 1.0, 0.0
    for _ in range(iters):
        if keep.sum() < 10:
            break
        te, tn, s, theta = fit_similarity(pe[keep], pn[keep], qe[keep], qn[keep])
        ca, sa = s * math.cos(math.radians(theta)), s * math.sin(math.radians(theta))
        # apply forward to every point and measure residual
        fE = te + ca * pe - sa * pn
        fN = tn + sa * pe + ca * pn
        res = np.hypot(fE - qe, fN - qn)
        med = np.median(res)
        mad = np.median(np.abs(res - med)) + 1e-9
        new = res <= med + trim * 1.4826 * mad
        if new.sum() == keep.sum() and (new == keep).all():
            keep = new
            break
        keep = new
    return te, tn, s, theta, int(keep.sum()), int(len(pe))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = []

    def say(s=""):
        print(s)
        rep.append(s)

    say("=" * 78)
    say("Map-vs-positioning offset audit  --  Haikou real-vehicle dataset")
    say("=" * 78)

    # ── 1. inputs ───────────────────────────────────────────────────────────
    gt = load_gt()
    say(f"\nRTK epochs with a valid position (aligned.csv): {len(gt)}")

    quality = {}
    for tid, folder in TRAJ_MAP.items():
        p = REAL / folder / "实时定位结果" / "rtk_solution_clean.txt"
        if p.exists():
            quality[tid] = parse_rtk_quality_and_pos(p)

    qcount = defaultdict(int)
    for e in gt:
        tid = e["traj"]
        ts = str(round(float(e["ts"])))
        q = quality.get(tid, {}).get(int(ts), (None, None, 0))[2]
        e["fix"] = q
        qcount[q] += 1
    say("GGA fix quality of those epochs:")
    for q in sorted(qcount, reverse=True):
        say(f"   {FIX_NAME.get(q, 'unknown(%s)' % q):<12} {qcount[q]:>6}  "
            f"({qcount[q] / len(gt) * 100:5.1f} %)")

    lines, eids = load_edges()
    say(f"\nRoad edges loaded: {len(lines)}  (id range {min(eids)}..{max(eids)})")

    # ── 2. project everything to a local metric plane ───────────────────────
    lon0 = float(np.mean([e["lon"] for e in gt]))
    lat0 = float(np.mean([e["lat"] for e in gt]))
    aeqd = (f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 "
            f"+datum=WGS84 +units=m +no_defs")
    tr = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
    say(f"Local projection: AEQD centred on ({lat0:.6f}, {lon0:.6f})")

    proj_lines = []
    for ln in lines:
        xs, ys = tr.transform(*zip(*ln.coords))
        proj_lines.append(LineString(zip(xs, ys)))
    tree = STRtree(proj_lines)
    eid2idx = {e: i for i, e in enumerate(eids)}

    # ── 3. per-epoch nearest-road offset ────────────────────────────────────
    recs = []
    miss = 0
    for e in gt:
        px, py = tr.transform(e["lon"], e["lat"])
        pt = Point(px, py)
        try:
            idx = int(tree.query_nearest(pt, all_matches=False)[0])
        except TypeError:
            idx = int(tree.query_nearest(pt)[0])
        lng = proj_lines[idx]
        qx, qy = nearest_points(pt, lng)[1].coords[0]
        e["px"], e["py"] = px, py
        e["qx"], e["qy"] = qx, qy
        e["oe"], e["on"] = qx - px, qy - py
        e["dist"] = math.hypot(e["oe"], e["on"])
        e["nearest_eid"] = eids[idx]
        # distance to the annotated GT edge, when there is one
        if e["gt_edge"] not in ("", "0", "-1"):
            j = eid2idx.get(int(e["gt_edge"]))
            if j is not None:
                gx, gy = nearest_points(pt, proj_lines[j])[1].coords[0]
                e["gt_eid_dist"] = math.hypot(gx - px, gy - py)
            else:
                miss += 1
        recs.append(e)
    say(f"Epochs with an unprojectable gt_edge id: {miss}")

    # SPP offset, same treatment, for contrast
    for e in recs:
        if e["obs_lon"] is None:
            e["spp_dist"] = None
            continue
        ox, oy = tr.transform(e["obs_lon"], e["obs_lat"])
        pt = Point(ox, oy)
        try:
            idx = int(tree.query_nearest(pt, all_matches=False)[0])
        except TypeError:
            idx = int(tree.query_nearest(pt)[0])
        qx, qy = nearest_points(pt, proj_lines[idx])[1].coords[0]
        e["spp_oe"], e["spp_on"] = qx - ox, qy - oy
        e["spp_dist"] = math.hypot(e["spp_oe"], e["spp_on"])

    # ── 4. heading from the RTK track itself ────────────────────────────────
    by_traj = defaultdict(list)
    for e in recs:
        by_traj[e["traj"]].append(e)
    HALF = 3   # +-3 s centred difference -> ~24 m baseline at 8 m/s
    for tid, lst in by_traj.items():
        lst.sort(key=lambda r: float(r["ts"]))
        n = len(lst)
        for i, e in enumerate(lst):
            a = lst[max(0, i - HALF)]
            b = lst[min(n - 1, i + HALF)]
            dx, dy = b["px"] - a["px"], b["py"] - a["py"]
            d = math.hypot(dx, dy)
            # require a real displacement so a parked vehicle has no heading
            e["speed"] = d / max(1e-9, float(b["ts"]) - float(a["ts"]))
            if d > 3.0:
                e["he"], e["hn"] = dx / d, dy / d
            else:
                e["he"], e["hn"] = None, None

    for e in recs:
        if e["he"] is None:
            e["along"] = e["cross"] = None
            continue
        # right-hand perpendicular of the heading
        re_, rn_ = e["hn"], -e["he"]
        e["along"] = e["oe"] * e["he"] + e["on"] * e["hn"]
        e["cross"] = e["oe"] * re_ + e["on"] * rn_

    # ── 5. aggregate ────────────────────────────────────────────────────────
    zero = load_zero_candidate_epochs()

    def subset(pred):
        return [e for e in recs if pred(e)]

    blocks = {}
    blocks["ALL epochs"] = recs
    blocks["traj 12 excluded"] = subset(lambda e: e["traj"] not in EXCLUDE)
    blocks["RTK fixed only (q=4)"] = subset(lambda e: e["fix"] == 4)
    blocks["RTK float (q=5)"] = subset(lambda e: e["fix"] == 5)
    blocks["autonomous (q=1)"] = subset(lambda e: e["fix"] == 1)
    blocks["GT edge known (on a mapped road)"] = subset(
        lambda e: e["gt_edge"] not in ("", "0", "-1"))
    blocks["GT edge == 0 (no road in map)"] = subset(lambda e: e["gt_edge"] == "0")

    say("\n" + "=" * 78)
    say("A. DISTANCE FROM RTK GROUND TRUTH TO THE NEAREST ROAD EDGE")
    say("=" * 78)
    say(f"{'subset':<36} {'n':>6} {'mean':>7} {'med':>7} {'p68':>7} "
        f"{'p90':>7} {'p95':>7} {'<=5m':>7} {'<=10m':>7} {'<=20m':>7} {'<=30m':>7}")
    say("-" * 78)
    stats = {}
    for name, lst in blocks.items():
        if not lst:
            continue
        st = offset_stats([e["oe"] for e in lst], [e["on"] for e in lst])
        stats[name] = st
        say(f"{name:<36} {st['n']:>6} {st['dist_mean_m']:>7.2f} {st['dist_median_m']:>7.2f} "
            f"{st['dist_p68_m']:>7.2f} {st['dist_p90_m']:>7.2f} {st['dist_p95_m']:>7.2f} "
            f"{st['frac_within_5m'] * 100:>6.1f}% {st['frac_within_10m'] * 100:>6.1f}% "
            f"{st['frac_within_20m'] * 100:>6.1f}% {st['frac_within_30m'] * 100:>6.1f}%")

    say("\n" + "=" * 78)
    say("B. OFFSET VECTOR  o = nearest road point - RTK position   (EARTH frame)")
    say("=" * 78)
    say("   A map/datum/CRS bias would show up here as a non-zero mean vector")
    say("   pointing in a constant compass direction.")
    say(f"{'subset':<36} {'n':>6} {'meanE':>7} {'meanN':>7} {'|mean|':>7} {'brg':>6} "
        f"{'medE':>7} {'medN':>7} {'|med|':>7} {'stdE':>7} {'stdN':>7} {'RaylP':>8}")
    say("-" * 78)
    for name, lst in blocks.items():
        st = stats.get(name)
        if not st:
            continue
        say(f"{name:<36} {st['n']:>6} {st['mean_east_m']:>7.2f} {st['mean_north_m']:>7.2f} "
            f"{st['mean_vec_mag_m']:>7.2f} {st['mean_vec_bearing_deg']:>5.0f}° "
            f"{st['median_east_m']:>7.2f} {st['median_north_m']:>7.2f} "
            f"{st['median_vec_mag_m']:>7.2f} {st['std_east_m']:>7.2f} {st['std_north_m']:>7.2f} "
            f"{st['rayleigh_p']:>8.2e}")

    say("\n" + "=" * 78)
    say("C. OFFSET IN THE VEHICLE FRAME  (along + = ahead, cross + = to the driver's right)")
    say("=" * 78)
    say(f"{'subset':<36} {'n':>6} {'along mean':>10} {'along std':>10} "
        f"{'cross mean':>11} {'cross std':>10} {'cross med':>10}")
    say("-" * 78)
    vframe = {}
    for name, lst in blocks.items():
        v = [e for e in lst if e["along"] is not None]
        if not v:
            continue
        al = np.array([e["along"] for e in v])
        cr = np.array([e["cross"] for e in v])
        vframe[name] = {"n": len(v),
                        "along_mean": float(al.mean()), "along_std": float(al.std(ddof=1)),
                        "cross_mean": float(cr.mean()), "cross_std": float(cr.std(ddof=1)),
                        "cross_median": float(np.median(cr))}
        say(f"{name:<36} {len(v):>6} {al.mean():>10.2f} {al.std(ddof=1):>10.2f} "
            f"{cr.mean():>11.2f} {cr.std(ddof=1):>10.2f} {np.median(cr):>10.2f}")

    say("\n" + "=" * 78)
    say("D. PER-TRAJECTORY MEAN OFFSET VECTOR  (spatial coherence test)")
    say("=" * 78)
    say("   A datum/CRS misregistration is smooth over the city, so different")
    say("   trajectories should share the same offset vector.")
    say(f"{'traj':>5} {'n':>6} {'meanE':>8} {'meanN':>8} {'|mean|':>7} {'brg':>6} "
        f"{'med|d|':>7} {'p90|d|':>7} {'zero-cand':>10} {'fix4%':>7}")
    say("-" * 78)
    per_traj = {}
    for tid in sorted(by_traj, key=lambda x: int(x)):
        lst = [e for e in by_traj[tid] if e["traj"] not in EXCLUDE]
        if not lst:
            continue
        st = offset_stats([e["oe"] for e in lst], [e["on"] for e in lst])
        nz = sum(1 for e in lst
                 if (e["traj"], e["ts"]) in zero.get("cmm_result_enu.csv", set()))
        f4 = sum(1 for e in lst if e["fix"] == 4) / len(lst) * 100
        per_traj[tid] = st
        per_traj[tid]["n_zero_candidate"] = nz
        per_traj[tid]["frac_fix4"] = f4
        say(f"{tid:>5} {st['n']:>6} {st['mean_east_m']:>8.2f} {st['mean_north_m']:>8.2f} "
            f"{st['mean_vec_mag_m']:>7.2f} {st['mean_vec_bearing_deg']:>5.0f}° "
            f"{st['dist_median_m']:>7.2f} {st['dist_p90_m']:>7.2f} {nz:>10} {f4:>6.1f}%")

    # ── 6. spatial coherence in 500 m cells ─────────────────────────────────
    say("\n" + "=" * 78)
    say("E. SPATIAL COHERENCE OF THE OFFSET  (500 m cells, >= 20 epochs each)")
    say("=" * 78)
    core = [e for e in recs if e["traj"] not in EXCLUDE and e["dist"] <= 20.0]
    cells = defaultdict(list)
    for e in core:
        cells[(int(e["px"] // 500), int(e["py"] // 500))].append(e)
    cell_rows = []
    for k, lst in cells.items():
        if len(lst) < 20:
            continue
        me = float(np.mean([e["oe"] for e in lst]))
        mn = float(np.mean([e["on"] for e in lst]))
        cell_rows.append({"cx": k[0], "cy": k[1], "n": len(lst),
                          "mean_east_m": me, "mean_north_m": mn,
                          "mag_m": math.hypot(me, mn),
                          "bearing_deg": math.degrees(math.atan2(me, mn)) % 360.0,
                          "median_dist_m": float(np.median([e["dist"] for e in lst]))})
    if cell_rows:
        mags = np.array([c["mag_m"] for c in cell_rows])
        ang = np.radians([c["bearing_deg"] for c in cell_rows])
        Rb = math.hypot(np.cos(ang).mean(), np.sin(ang).mean())
        say(f"cells with >=20 epochs: {len(cell_rows)}")
        say(f"cell-mean |offset|: median {np.median(mags):.2f} m, "
            f"p90 {np.percentile(mags, 90):.2f} m, max {mags.max():.2f} m")
        say(f"cell-mean direction concentration R_bar = {Rb:.3f}  "
            f"(0 = random directions, 1 = all cells agree)")
        wmE = float(np.average([c["mean_east_m"] for c in cell_rows],
                               weights=[c["n"] for c in cell_rows]))
        wmN = float(np.average([c["mean_north_m"] for c in cell_rows],
                               weights=[c["n"] for c in cell_rows]))
        say(f"epoch-weighted city-wide offset vector: "
            f"({wmE:+.2f} E, {wmN:+.2f} N) m, |v| = {math.hypot(wmE, wmN):.2f} m, "
            f"bearing {math.degrees(math.atan2(wmE, wmN)) % 360:.0f}°")

    # ── 7. similarity transform (translation + rotation + scale) ────────────
    say("\n" + "=" * 78)
    say("F. BEST-FIT 2-D SIMILARITY  q = T + s·R(theta)·p   (trimmed LS)")
    say("=" * 78)
    sim = {}
    for name, lst in (("ALL epochs", recs),
                      ("traj 12 excluded", [e for e in recs if e["traj"] not in EXCLUDE]),
                      ("RTK fixed only", [e for e in recs if e["fix"] == 4])):
        if len(lst) < 50:
            continue
        pe = np.array([e["px"] for e in lst]); pn = np.array([e["py"] for e in lst])
        qe = np.array([e["qx"] for e in lst]); qn = np.array([e["qy"] for e in lst])
        te, tn, s, th, ninl, ntot = robust_similarity(pe, pn, qe, qn)
        sim[name] = {"translate_east_m": te, "translate_north_m": tn,
                     "scale": s, "rotation_deg": th,
                     "n_inlier": ninl, "n_total": ntot}
        say(f"{name:<22} T=({te:+7.2f} E, {tn:+7.2f} N) m   "
            f"s={s:.6f} ({(s - 1) * 1e6:+.1f} ppm)   theta={th:+.4f}°   "
            f"inliers {ninl}/{ntot}")
    say("   Reference: a GCJ-02 (火星坐标) offset would be O(100 m) here;")
    say("              a WGS84/CGCS2000 difference is < 0.01 m.")

    # ── 8. SPP as a control ─────────────────────────────────────────────────
    say("\n" + "=" * 78)
    say("G. CONTROL: the same statistic for the SPP solutions")
    say("=" * 78)
    say("   A map bias is present in both RTK and SPP; GNSS error affects only SPP.")
    spp = {}
    for tag, key_d, key_oe, key_on in (("RTK", "dist", "oe", "on"),
                                       ("SPP", "spp_dist", "spp_oe", "spp_on")):
        lst = [e for e in recs if e["traj"] not in EXCLUDE and e.get(key_d) is not None]
        st = offset_stats([e[key_oe] for e in lst], [e[key_on] for e in lst])
        spp[tag] = st
        say(f"{tag}: n={st['n']}  mean dist {st['dist_mean_m']:.2f} m  "
            f"median {st['dist_median_m']:.2f} m  p90 {st['dist_p90_m']:.2f} m  "
            f"|mean vec| {st['mean_vec_mag_m']:.2f} m @ {st['mean_vec_bearing_deg']:.0f}°  "
            f"Rayleigh p={st['rayleigh_p']:.2e}")

    # ── 9. link to the zero-candidate epochs ────────────────────────────────
    say("\n" + "=" * 78)
    say("H. THE ZERO-CANDIDATE EPOCHS")
    say("=" * 78)
    zlink = {}
    for fn, zs in zero.items():
        zs_f = {(t, s) for (t, s) in zs if t not in EXCLUDE}
        in_set = [e for e in recs if e["traj"] not in EXCLUDE and (e["traj"], e["ts"]) in zs_f]
        out_set = [e for e in recs if e["traj"] not in EXCLUDE and (e["traj"], e["ts"]) not in zs_f]
        if not in_set:
            say(f"{fn}: no zero-candidate epochs among the analysed set")
            continue
        a = offset_stats([e["oe"] for e in in_set], [e["on"] for e in in_set])
        b = offset_stats([e["oe"] for e in out_set], [e["on"] for e in out_set])
        q4 = sum(1 for e in in_set if e["fix"] == 4) / len(in_set) * 100
        q45 = sum(1 for e in out_set if e["fix"] == 4) / len(out_set) * 100
        zlink[fn] = {"n_zero": len(in_set), "zero": a, "nonzero": b,
                     "frac_fix4_zero": q4, "frac_fix4_nonzero": q45}
        say(f"\n{fn}:  {len(in_set)} zero-candidate epochs "
            f"(vs {len(out_set)} with a candidate)")
        say(f"  nearest-road distance : median {a['dist_median_m']:.2f} m, "
            f"p90 {a['dist_p90_m']:.2f} m   |   "
            f"others median {b['dist_median_m']:.2f} m, p90 {b['dist_p90_m']:.2f} m")
        say(f"  offset vector         : ({a['median_east_m']:+.2f} E, "
            f"{a['median_north_m']:+.2f} N) m   |   others "
            f"({b['median_east_m']:+.2f} E, {b['median_north_m']:+.2f} N) m")
        say(f"  RTK-fixed fraction    : {q4:.1f} %  |  others {q45:.1f} %")
        # how much of the deficit is explained by the offset alone?
        need = a["dist_median_m"]
        say(f"  -> to reach a candidate by radius alone the search radius would")
        say(f"     have to exceed {a['dist_p90_m']:.1f} m for 90 % of these epochs "
            f"(p50 needs {need:.1f} m).")

    # ── 10. persist ─────────────────────────────────────────────────────────
    summary = {"blocks": stats, "vehicle_frame": vframe, "per_traj": per_traj,
               "similarity": sim, "spp_control": spp, "zero_candidate": zlink,
               "n_epochs": len(recs), "fix_quality_counts": dict(qcount),
               "projection": {"lat_0": lat0, "lon_0": lon0, "proj": aeqd}}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    with open(OUT / "per_epoch.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["traj", "timestamp", "fix_quality", "gt_edge", "nearest_eid",
                    "gt_eid_dist_m", "dist_m", "offset_east_m", "offset_north_m",
                    "bearing_deg", "speed_mps", "along_m", "cross_m",
                    "spp_dist_m", "spp_offset_east_m", "spp_offset_north_m",
                    "zero_candidate_enu"])
        zs = zero.get("cmm_result_enu.csv", set())
        for e in sorted(recs, key=lambda r: (int(r["traj"]), float(r["ts"]))):
            w.writerow([e["traj"], e["ts"], e["fix"], e["gt_edge"], e["nearest_eid"],
                        f"{e.get('gt_eid_dist', float('nan')):.4f}",
                        f"{e['dist']:.4f}", f"{e['oe']:.4f}", f"{e['on']:.4f}",
                        f"{math.degrees(math.atan2(e['oe'], e['on'])) % 360:.2f}",
                        f"{e.get('speed', float('nan')):.3f}",
                        "" if e["along"] is None else f"{e['along']:.4f}",
                        "" if e["cross"] is None else f"{e['cross']:.4f}",
                        "" if e.get("spp_dist") is None else f"{e['spp_dist']:.4f}",
                        "" if e.get("spp_oe") is None else f"{e['spp_oe']:.4f}",
                        "" if e.get("spp_on") is None else f"{e['spp_on']:.4f}",
                        int((e["traj"], e["ts"]) in zs)])

    with open(OUT / "grid_offset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cx", "cy", "n", "mean_east_m",
                                          "mean_north_m", "mag_m", "bearing_deg",
                                          "median_dist_m"], delimiter=";")
        w.writeheader()
        for c in sorted(cell_rows, key=lambda r: -r["n"]):
            w.writerow(c)

    with open(OUT / "report.txt", "w") as f:
        f.write("\n".join(rep) + "\n")
    say(f"\nWrote {OUT}/summary.json, per_epoch.csv, grid_offset.csv, report.txt")


if __name__ == "__main__":
    main()
