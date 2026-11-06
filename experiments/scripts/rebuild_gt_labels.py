#!/usr/bin/env python3
"""Rebuild the ground-truth label table (four classes, traceable, reconciled).

Rules, in order:
  a. no RTK position                          -> no_rtk
  b. gt_edge == '-1'                          -> stationary  (the traj14 frozen block)
  c. in MANUAL_ON2OFF                         -> off         (user reclassified #1/#3/#4)
  d. gt_edge == '0' and stationary            -> stationary
  e. gt_edge == '0' and moving                -> off
  f. real gt_edge, stationary                 -> on   (stationary flag kept)
  g. real gt_edge, moving                     -> on

stationary := 11-epoch rolling mean of per-epoch RTK displacement < 0.05 m
The original gt_edge is preserved verbatim in gt_edge_raw; gt_class is new.
"""
import csv
import math
from collections import Counter, defaultdict

import numpy as np
from osgeo import ogr

SRC = "data/real_vehicle/hainan_06/processed/aligned.csv"
SHP = "input/map/hainan/edges.shp"
OUT = "docs/calib/figs/gt_labels_v2.csv"

# block -> (tid, seq0, seq1) for the three blocks the user reclassified as off-road
MANUAL_ON2OFF = [("12", 351, 356), ("21", 1605, 1636), ("21", 1852, 1885)]

lat0, lon0 = 19.82168, 110.63658
kx = 111320.0 * math.cos(math.radians(lat0))
ky = 110540.0


def xy(a, b):
    return ((a - lon0) * kx, (b - lat0) * ky)


rows = []
with open(SRC, newline="") as fh:
    for ln, r in enumerate(csv.DictReader(fh, delimiter=";"), start=2):
        if not r["uni_seq"].strip():
            continue
        r["_ln"] = ln
        rows.append(r)

# ---- duplicate key audit -------------------------------------------------
keys = Counter((r["id"].strip(), r["uni_seq"].strip()) for r in rows)
dups = {k: v for k, v in keys.items() if v > 1}
print("rows=%d  unique (id,uni_seq)=%d  duplicate keys=%d (extra rows=%d)"
      % (len(rows), len(keys), len(dups), sum(v - 1 for v in dups.values())))
for k, v in list(dups.items())[:6]:
    print("   dup %s x%d" % (k, v))
seen = set()
keep = []
for r in rows:
    k = (r["id"].strip(), r["uni_seq"].strip())
    if k in seen:
        continue
    seen.add(k)
    keep.append(r)
print("kept %d unique epochs" % len(keep))

# ---- edges ---------------------------------------------------------------
AX, AY, BX, BY = [], [], [], []
ds = ogr.Open(SHP)
ly = ds.GetLayer(0)
NF = ly.GetFeatureCount()
for f in ly:
    g = f.GetGeometryRef()
    if g is None:
        continue
    for gi in range(g.GetGeometryCount() or 1):
        gg = g.GetGeometryRef(gi) if g.GetGeometryCount() else g
        if gg is None:
            continue
        for i in range(gg.GetPointCount() - 1):
            p1 = xy(gg.GetX(i), gg.GetY(i))
            p2 = xy(gg.GetX(i + 1), gg.GetY(i + 1))
            AX.append(p1[0]); AY.append(p1[1]); BX.append(p2[0]); BY.append(p2[1])
AX = np.array(AX); AY = np.array(AY); BX = np.array(BX); BY = np.array(BY)
DX = BX - AX; DY = BY - AY
L2 = DX * DX + DY * DY
L2S = np.where(L2 > 1e-12, L2, 1.0)
MX = (AX + BX) / 2
MY = (AY + BY) / 2
print("features=%d segments=%d" % (NF, len(AX)))


def nearest_dist(px, py):
    m = (np.abs(MX - px) < 400.0) & (np.abs(MY - py) < 400.0)
    if not m.any():
        return float("nan")
    ax, ay, dx, dy, l2 = AX[m], AY[m], DX[m], DY[m], L2S[m]
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / l2, 0.0, 1.0)
    return float(np.min(np.hypot(px - (ax + t * dx), py - (ay + t * dy))))


def feature_dist(idx, px, py):
    f = ly.GetFeature(int(idx))
    if f is None:
        return float("nan")
    g = f.GetGeometryRef()
    if g is None:
        return float("nan")
    best = 1e18
    for gi in range(g.GetGeometryCount() or 1):
        gg = g.GetGeometryRef(gi) if g.GetGeometryCount() else g
        if gg is None:
            continue
        for i in range(gg.GetPointCount() - 1):
            a = xy(gg.GetX(i), gg.GetY(i))
            b = xy(gg.GetX(i + 1), gg.GetY(i + 1))
            dx, dy = b[0] - a[0], b[1] - a[1]
            l2 = dx * dx + dy * dy
            t = 0.0 if l2 <= 1e-12 else max(0.0, min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy) / l2))
            best = min(best, math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy)))
    return best


# ---- per-trajectory stationarity ----------------------------------------
stat = {}
for tid in sorted({r["id"].strip() for r in keep}):
    rs = sorted([r for r in keep if r["id"].strip() == tid and r["gt_x"].strip()],
                key=lambda r: int(r["uni_seq"]))
    if not rs:
        continue
    P = np.array([xy(float(r["gt_x"]), float(r["gt_y"])) for r in rs])
    step = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    step = np.concatenate([[step[0] if len(step) else 0.0], step])
    roll = np.convolve(step, np.ones(11) / 11.0, mode="same")
    for i, r in enumerate(rs):
        stat[(tid, r["uni_seq"].strip())] = bool(roll[i] < 0.05)

# ---- classify ------------------------------------------------------------
out = []
cnt = Counter()
for r in keep:
    tid = r["id"].strip()
    sq = r["uni_seq"].strip()
    raw = (r.get("gt_edge") or "").strip()
    if not r["gt_x"].strip() or not r["gt_y"].strip():
        cls, src, edge, dist = "no_rtk", "no_rtk", "", float("nan")
    else:
        px, py = xy(float(r["gt_x"]), float(r["gt_y"]))
        is_stat = stat.get((tid, sq), False)
        manual_off = any(tid == t and s0 <= int(sq) <= s1 for (t, s0, s1) in MANUAL_ON2OFF)
        if raw == "-1":
            cls, src, edge, dist = "stationary", "orig_minus1", "", nearest_dist(px, py)
        elif manual_off:
            cls, src, edge, dist = "off", "manual_on2off", "", nearest_dist(px, py)
        elif raw in ("0", "0.0"):
            if is_stat:
                cls, src, edge, dist = "stationary", "stationary_off", "", nearest_dist(px, py)
            else:
                cls, src, edge, dist = "off", "orig_0", "", nearest_dist(px, py)
        else:
            try:
                idx = int(raw)
            except ValueError:
                cls, src, edge, dist = "on", "orig_edge_unparsed", raw, float("nan")
            else:
                cls, src = "on", "orig_edge"
                edge = raw
                dist = feature_dist(idx, px, py) if 0 <= idx < NF else float("nan")
                if is_stat:
                    src = "orig_edge_stationary"
    cnt[cls] += 1
    out.append([tid, sq, r.get("timestamp", "").strip(), cls, edge, raw,
                "%.2f" % dist if np.isfinite(dist) else "", int(bool(stat.get((tid, sq), False))),
                src, r["_ln"]])

with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "uni_seq", "timestamp", "gt_class", "gt_edge", "gt_edge_raw",
                "gt_dist_m", "gt_stationary", "gt_source", "src_line"])
    w.writerows(out)

print("\n== four-class counts ==")
for k in ("on", "off", "stationary", "no_rtk"):
    print("   %-11s %6d  (%.2f%%)" % (k, cnt[k], 100.0 * cnt[k] / max(len(out), 1)))
print("   %-11s %6d" % ("TOTAL", sum(cnt.values())))
print("\n== gt_source breakdown ==")
for k, v in sorted(Counter(o[8] for o in out).items()):
    print("   %-24s %6d" % (k, v))
print("\n== per trajectory ==")
print("   %-4s %6s %6s %6s %9s" % ("id", "on", "off", "stat", "no_rtk"))
for tid in sorted({o[0] for o in out}):
    c = Counter(o[3] for o in out if o[0] == tid)
    print("   %-4s %6d %6d %6d %9d" % (tid, c["on"], c["off"], c["stationary"], c["no_rtk"]))
print("\nwrote %s (%d rows)" % (OUT, len(out)))
