#!/usr/bin/env python3
"""v2: off-candidate HMM + speed-magnitude emission channel.

Runs each window twice (speed channel off / on) so the comparison is paired.
Speed model p(v|class) is estimated on the OTHER trajectories (no leakage from
traj 11), with cross-class shrinkage on the log-sd.
"""
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT / "data/real_vehicle/hainan_06/processed/aligned.csv"
SHP = PROJECT / "input/map/hainan/edges.shp"
OUT_CSV = PROJECT / "docs/calib/figs/proto_speed_dump.csv"

EVAL_ID = "11"
_GEO_CACHE = None
WINDOWS = [("ctrl", 1000, 1053), ("off-A", 1653, 1696), ("off-B", 2502, 2511)]
PAD = 6

SIG_MAP_CROSS = 3.0
SIG_MAP_ALONG = 1.0
ETA = float(os.environ.get("PROTO_ETA", "0.02"))
SIG_V = 2.0
Q_DR = 0.5
S_GRID = np.exp(np.linspace(math.log(0.25), math.log(3.0), 14))
S_STEP = 1.0
S_PRIOR_SD = 0.35
R_CAND = 300.0
K_MAX = 16

# speed channel
SD_FLOOR = 0.35      # m/s -> log units; a floor on the class log-sd (anti over-penalty)
SHRINK_K = 20.0      # pseudo-count shrinking a class mean toward the global mean
MIN_N = 15
MU0, SD0 = math.log(5.0), 0.75    # x0 speed prior: broad, centred on a slow crawl

CLASS_ORDER = ["living_street", "service", "residential", "unclassified", "tertiary",
               "secondary", "primary_link", "primary", "trunk_link", "trunk",
               "motorway_link", "motorway", "unknown"]


def norm_class(hw):
    if isinstance(hw, (list, tuple)):
        toks = [str(t) for t in hw]
    elif isinstance(hw, str):
        toks = [t.strip().strip("'\" ") for t in hw.strip("[]").split(",")]
    else:
        toks = []
    toks = [t for t in toks if t]
    for c in CLASS_ORDER:            # pick the slowest class present (conservative)
        if c in toks:
            return c
    return "unknown"


def load_edges(bbox):
    """Read the shapefile ONCE into a module cache, then bbox-filter cheaply."""
    global _GEO_CACHE
    if _GEO_CACHE is None:
        from osgeo import ogr
        ds = ogr.Open(str(SHP))
        ly = ds.GetLayer(0)
        d = ly.GetLayerDefn()
        names = [d.GetFieldDefn(i).GetName() for i in range(d.GetFieldCount())]
        idf = "osmid" if "osmid" in names else names[0]
        cache = []
        for row_i, f in enumerate(ly):
            g = f.GetGeometryRef()
            if g is None:
                continue
            hw = norm_class(f.GetField("highway") if "highway" in names else None)
            eid = str(row_i)   # gt_edge in aligned.csv is the shapefile ROW INDEX, not osmid
            for gi in range(g.GetGeometryCount() or 1):
                gg = g.GetGeometryRef(gi) if g.GetGeometryCount() else g
                if gg is None:
                    continue
                pts = [(gg.GetX(i), gg.GetY(i)) for i in range(gg.GetPointCount())]
                if len(pts) >= 2:
                    cache.append((eid, pts, hw))
        _GEO_CACHE = cache
        print("   [cache] read %d linestrings from the shapefile" % len(cache), flush=True)
    x0, x1, y0, y1 = bbox
    return [e for e in _GEO_CACHE
            if any(x0 <= px <= x1 and y0 <= py <= y1 for px, py in e[1])]


def to_xy(lon, lat, lon0, lat0, kx, ky):
    return (lon - lon0) * kx, (lat - lat0) * ky


def seg_project(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 <= 1e-12 else ((px - ax) * dx + (py - ay) * dy) / L2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    qx, qy = ax + t * dx, ay + t * dy
    return qx, qy, math.hypot(px - qx, py - qy), math.atan2(dy, dx), math.sqrt(L2)


def build_edges(edges, lon0, lat0, kx, ky):
    out = []
    for eid, pts, hw in edges:
        xy = [to_xy(a, b, lon0, lat0, kx, ky) for a, b in pts]
        L = sum(math.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1])
                for i in range(len(xy) - 1))
        out.append((eid, xy, L, hw))
    return out


def candidates(px, py, edges_xy, r_cand, k_max):
    out = []
    for eid, pts, L, hw in edges_xy:
        best = None
        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            qx, qy, dd, ang, sl = seg_project(px, py, ax, ay, bx, by)
            if best is None or dd < best[2]:
                best = (qx, qy, dd, ang, L, sl)
        if best is not None and best[2] <= r_cand:
            out.append((eid, best[0], best[1], best[2], best[3], L, hw))
    out.sort(key=lambda z: z[3])
    return out[:k_max]


def log_gauss2(dx, dy, S):
    det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
    if det <= 1e-12:
        return -1e9
    inv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
    q = inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    return -0.5 * q - 0.5 * math.log(2 * math.pi) - 0.5 * math.log(det)


def logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis) if axis is not None else float(out)


def lp_speed(v, mu, sd):
    v = max(v, 0.5)
    return -0.5 * ((math.log(v) - mu) / sd) ** 2 - math.log(v * sd * math.sqrt(2 * math.pi))


def read_rows():
    per = defaultdict(dict)
    with open(CSV_PATH, newline="") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if not r["uni_seq"].strip() or not r["obs_x"].strip() or not r["obs_y"].strip():
                continue
            per[r["id"].strip()][int(r["uni_seq"])] = r
    return per


def frame(rows, kx_pad=400.0):
    lon0 = float(np.mean([float(r["obs_x"]) for r in rows.values()]))
    lat0 = float(np.mean([float(r["obs_y"]) for r in rows.values()]))
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110540.0
    pts = [to_xy(float(r["obs_x"]), float(r["obs_y"]), lon0, lat0, kx, ky) for r in rows.values()]
    bbox = (min(p[0] for p in pts) - kx_pad, max(p[0] for p in pts) + kx_pad,
            min(p[1] for p in pts) - kx_pad, max(p[1] for p in pts) + kx_pad)
    return lon0, lat0, kx, ky, bbox


def fit_speed_model(per, eval_id):
    acc = defaultdict(list)
    g_all = []
    n_fit = 0
    for tid, rows in per.items():
        if tid == eval_id:
            continue
        seqs = sorted(rows)
        if len(seqs) < 50:
            continue
        n_fit += 1
        if n_fit > 3:
            break
        print("   [fit] traj %s (%d epochs)" % (tid, len(seqs)), flush=True)
        lon0, lat0, kx, ky, bbox = frame(rows)
        edges_xy = build_edges(load_edges(bbox), lon0, lat0, kx, ky)
        P = {s: to_xy(float(rows[s]["obs_x"]), float(rows[s]["obs_y"]), lon0, lat0, kx, ky) for s in seqs}
        TS = {s: float(rows[s]["timestamp"]) for s in seqs}
        idx = {s: i for i, s in enumerate(seqs)}
        for i, s in enumerate(seqs):
            if i % 10 or i < 2 or i >= len(seqs) - 2:
                continue
            a, b = seqs[i - 2], seqs[i + 2]
            path = sum(math.hypot(P[seqs[j + 1]][0] - P[seqs[j]][0],
                                  P[seqs[j + 1]][1] - P[seqs[j]][1])
                       for j in range(i - 2, i + 2))
            v = path / max(TS[b] - TS[a], 1e-6)
            c = candidates(P[s][0], P[s][1], edges_xy, R_CAND, 1)
            if not c or c[0][3] > 3.0:
                continue
            acc[c[0][6]].append(v)
            g_all.append(v)
        print("      -> %d samples so far" % len(g_all), flush=True)
    g = np.log(np.maximum(np.array(g_all), 0.5))
    mu_g, sd_g = float(g.mean()), max(float(g.std()), SD_FLOOR)
    model = {}
    print("== speed model: p(v | highway class), fitted on trajs != %s ==" % eval_id)
    print("   global: n=%d median=%.2f m/s log-sd=%.2f" % (len(g_all), math.exp(mu_g), sd_g))
    for c, vs in sorted(acc.items(), key=lambda kv: -len(kv[1])):
        if len(vs) < MIN_N:
            continue
        lv = np.log(np.maximum(np.array(vs), 0.5))
        n = len(lv)
        mu = (n * float(lv.mean()) + SHRINK_K * mu_g) / (n + SHRINK_K)
        sd = max(float(lv.std()), SD_FLOOR)
        model[c] = (mu, sd)
        print("   %-14s n=%5d median=%5.2f m/s  mu_shrunk->%5.2f m/s  log-sd=%.2f"
              % (c, n, float(np.median(np.exp(lv))), math.exp(mu), sd))
    return model, (mu_g, sd_g)


def run_window(tag, rows, edges_xy, lon0, lat0, kx, ky, spd_model, glob, use_speed,
               drop_x0=False):
    T = len(rows)
    z = np.array([to_xy(float(r["obs_x"]), float(r["obs_y"]), lon0, lat0, kx, ky) for r in rows])
    ts = np.array([float(r["timestamp"]) for r in rows])
    Sobs = []
    for r in rows:
        se = (float(r["obs_sde"]) if r["obs_sde"].strip() else 2.5 / kx) * kx
        sn = (float(r["obs_sdn"]) if r["obs_sdn"].strip() else 2.5 / ky) * ky
        sne = (float(r["obs_sdne"]) if r["obs_sdne"].strip() else 0.0) * kx * ky
        Sobs.append(np.array([[max(se, 0.5) ** 2, sne], [sne, max(sn, 0.5) ** 2]]))
    # smoothed speed
    v = np.zeros(T)
    for t in range(T):
        a, b = max(0, t - 2), min(T - 1, t + 2)
        if b > a:                      # PATH speed, not chord: a U-turn makes the chord
            path = sum(math.hypot(*(z[j + 1] - z[j])) for j in range(a, b))
            v[t] = path / max(ts[b] - ts[a], 1e-6)

    ns = len(S_GRID)
    cands = [candidates(z[t][0], z[t][1], edges_xy, R_CAND, K_MAX) for t in range(T)]
    Kmax = max(1, max(len(c) for c in cands))

    logE = np.full((T, Kmax + 1, ns), -1e9)
    for t in range(T):
        for k, (eid, qx, qy, dd, ang, L, hw) in enumerate(cands[t]):
            dx, dy = z[t][0] - qx, z[t][1] - qy
            ca, sa = math.cos(ang), math.sin(ang)
            R = np.array([[ca, -sa], [sa, ca]])
            M = R @ np.diag([SIG_MAP_ALONG ** 2, SIG_MAP_CROSS ** 2]) @ R.T
            spd = lp_speed(v[t], *spd_model.get(hw, glob)) if use_speed else 0.0
            for j, s in enumerate(S_GRID):
                Sig = s * s * Sobs[t] + M
                logE[t, k, j] = log_gauss2(dx, dy, Sig) + spd
        if t == 0:
            xi = z[t].copy()
            Sdr = Sobs[t].copy() + 50.0 * np.eye(2)
        else:
            vv = ((z[t - 1] - z[t - 3]) / 2.0 if t >= 3 else
                  (z[t - 1] - z[t - 2] if t >= 2 else np.zeros(2)))
            xi = z[t - 1] + vv
            Sdr = Sobs[t - 1] + SIG_V ** 2 * np.eye(2) + Q_DR ** 2 * np.eye(2)
        spd0 = lp_speed(v[t], MU0, SD0) if use_speed else 0.0
        for j, s in enumerate(S_GRID):
            Sig = s * s * (Sdr + Sobs[t])
            logE[t, Kmax, j] = log_gauss2(z[t][0] - xi[0], z[t][1] - xi[1], Sig) + spd0

    A = np.zeros((T, Kmax + 1, Kmax + 1))

    def cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    for t in range(1, T):
        dg = float(np.hypot(*(z[t] - z[t - 1])))
        n_prev, n_cur = len(cands[t - 1]), len(cands[t])
        for j in range(n_prev):
            for k in range(n_cur):
                dij = float(np.hypot(cands[t][k][1] - cands[t - 1][j][1],
                                     cands[t][k][2] - cands[t - 1][j][2]))
                A[t, j, k] = min(1.0, dg / dij) if dij > 1e-6 else 1.0
        A[t, :n_prev, Kmax] = ETA
        vv = ((z[t - 1] - z[t - 3]) / 2.0 if t >= 3 else
              (z[t - 1] - z[t - 2] if t >= 2 else np.zeros(2)))
        xi_ = z[t - 1] + vv
        Sdr_ = Sobs[t - 1] + (SIG_V ** 2 + Q_DR ** 2) * np.eye(2)
        dmin = min(c[3] for c in cands[t]) if n_cur else 1e9
        sd = math.sqrt(max(0.5 * (Sdr_[0, 0] + Sdr_[1, 1]), 1.0))
        P_re = min(1.0, max(0.0, cdf((5.0 - dmin) / sd) - cdf((-5.0 - dmin) / sd)))
        w = np.zeros(n_cur)
        for k in range(n_cur):
            w[k] = math.exp(log_gauss2(cands[t][k][1] - xi_[0], cands[t][k][2] - xi_[1], Sdr_))
        if w.sum() > 0:
            A[t, Kmax, :n_cur] = (w / w.sum()) * P_re
        A[t, Kmax, Kmax] = 1.0 - P_re
        for j in range(Kmax + 1):
            if j == Kmax:
                A[t, Kmax] /= max(A[t, Kmax].sum(), 1e-12)
                continue
            ss = A[t, j].sum()
            if ss <= 0:
                A[t, j, :n_cur] = 1.0
                ss = A[t, j].sum()
            A[t, j] /= ss
    A[0] = 1.0 / (Kmax + 1)
    if drop_x0:                       # MST-style model: no background state at all
        A[:, :, Kmax] = 0.0
        A[:, Kmax, :] = 0.0
        A[:, Kmax, Kmax] = 1.0
        for _t in range(T):
            _ss = A[_t, :Kmax, :Kmax].sum(axis=1, keepdims=True)
            A[_t, :Kmax, :Kmax] = A[_t, :Kmax, :Kmax] / np.maximum(_ss, 1e-300)
        logE[:, Kmax, :] = -1e9

    prior_s = np.exp(-0.5 * (np.log(S_GRID) / S_PRIOR_SD) ** 2)
    As = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            if abs(i - j) <= 4:
                As[i, j] = math.exp(-((i - j) ** 2) / (2 * S_STEP ** 2)) * prior_s[j]
    As = As / As.sum(axis=1, keepdims=True)

    logA = np.where(A > 0, np.log(np.maximum(A, 1e-300)), -1e9)
    logAs = np.where(As > 0, np.log(np.maximum(As, 1e-300)), -1e9)
    la = np.full((T, Kmax + 1, ns), -1e9)
    la[0] = logE[0] - math.log(Kmax + 1) - math.log(ns)
    for t in range(1, T):
        tmp = la[t - 1][:, :, None] + logAs[None, :, :]
        tmp = logsumexp(tmp, axis=1)
        tmp = logsumexp(tmp[:, None, :] + logA[t][:, :, None], axis=0)
        la[t] = tmp + logE[t]
    ga = logsumexp(la.reshape(T, -1), axis=1)
    pa = np.exp(la - ga[:, None, None])
    lb = np.zeros((T, Kmax + 1, ns))
    for t in range(T - 2, -1, -1):
        core = logE[t + 1] + lb[t + 1]
        ys = logsumexp(core[None, :, :] + logAs[:, None, :], axis=2)
        out = logsumexp(ys[:, None, :] + logA[t + 1][None, :, :], axis=2)
        lb[t] = out.T
    gab = la + lb
    g2 = logsumexp(gab.reshape(T, -1), axis=1)
    pab = np.exp(gab - g2[:, None, None])
    ld = np.full((T, Kmax + 1, ns), -1e9)
    ld[0] = logE[0]
    for t in range(1, T):
        tmp = ld[t - 1][:, :, None] + logAs[None, :, :]
        tmp = np.max(tmp, axis=1)
        tmp = np.max(tmp[:, None, :] + logA[t][:, :, None], axis=0)
        ld[t] = tmp + logE[t]
    ld_m = np.max(ld, axis=2)
    p00 = np.exp(ld_m[:, Kmax] - logsumexp(ld_m, axis=1))

    ld_all = np.exp(ld_m - logsumexp(ld_m, axis=1)[:, None])
    pt_d = ld_all[:, :Kmax].max(axis=1)
    pt_a = pa[:, :Kmax, :].sum(axis=2).max(axis=1)
    pt_ab = pab[:, :Kmax, :].sum(axis=2).max(axis=1)
    topid = []
    for t in range(T):
        if not cands[t]:
            topid.append("")
            continue
        marg = pa[t, :len(cands[t]), :].sum(axis=1)
        topid.append(cands[t][int(np.argmax(marg))][0])
    best_d = np.array([c[0][3] if c else np.nan for c in cands])
    best_c = [c[0][6] if c else "-" for c in cands]
    return dict(t=np.array([int(r["uni_seq"]) for r in rows]), v=v, bd=best_d, cls=best_c,
                p0d=p00, p0a=pa[:, Kmax, :].sum(axis=1), p0ab=pab[:, Kmax, :].sum(axis=1),
                ptd=pt_d, pta=pt_a, ptab=pt_ab, topid=topid,
                gt=[r.get("gt_edge", "") for r in rows])


def _score(cfg, labels):
    P = np.column_stack([cfg["ptd"], cfg["p0d"], 1.0 - cfg["ptd"] - cfg["p0d"]])
    P = np.clip(P, 1e-12, None)
    P = P / P.sum(axis=1, keepdims=True)
    keys = ("top", "x0", "other")
    Y = np.array([[1.0 if l == k else 0.0 for k in keys] for l in labels])
    return (-np.log((P * Y).sum(axis=1)), ((P - Y) ** 2).sum(axis=1))


def main():
    per = read_rows()
    print("ids:", sorted(per), "| eval:", EVAL_ID)
    model = {"primary": (math.log(18.98), 0.62),
             "motorway": (math.log(22.11), 0.35),
             "motorway_link": (math.log(17.72), 0.35)}
    glob = (math.log(17.97), 0.63)
    print("speed model CACHED from the fitted run; global", glob)
    rows_all = per[EVAL_ID]
    lon0, lat0, kx, ky, bbox = frame(rows_all)
    edges_xy = build_edges(load_edges(bbox), lon0, lat0, kx, ky)
    print("edges in bbox: %d" % len(edges_xy))

    CFGS = [("no_x0 MST-style", dict(drop_x0=True, use_speed=False)),
            ("x0 no speed", dict(drop_x0=False, use_speed=False)),
            ("x0 + speed", dict(drop_x0=False, use_speed=True))]
    res = {}
    keeps = {}
    for tag, lo, hi in WINDOWS:
        padseq = [i for i in range(lo - PAD, hi + PAD + 1) if i in rows_all]
        rrp = [rows_all[i] for i in padseq]
        keeps[tag] = [j for j, i in enumerate(padseq) if lo <= i <= hi]
        for nm, kw in CFGS:
            res[(tag, nm)] = run_window(tag, rrp, edges_xy, lon0, lat0, kx, ky,
                                        model, glob, **kw)

    print("\n== three-class proper scores on the evaluation epochs (pad excluded) ==")
    print("   %-6s %-16s %4s %10s %8s | %s" % ("win", "model", "n", "logscore", "Brier",
                                              "split by outcome"))
    for tag, _, _ in WINDOWS:
        keep = keeps[tag]
        for nm, _ in CFGS:
            r = res[(tag, nm)]
            gt = [(r["gt"][j] or "").strip() for j in keep]
            tid = [r["topid"][j] for j in keep]
            labels = []
            for g, tp in zip(gt, tid):
                if g in ("0", "0.0", ""):
                    labels.append("x0")
                elif any(g == x.strip() for x in tp.split(",")):
                    labels.append("top")
                else:
                    labels.append("other")
            sub = dict(t=r["t"][keep], ptd=r["ptd"][keep], p0d=r["p0d"][keep])
            ls, br = _score(sub, labels)
            lab = np.array(labels)
            mt, mx = lab == "top", lab == "x0"
            print("   %-6s %-16s %4d %10.3f %8.4f | y=top n=%2d ls=%7.3f  y=x0 n=%2d ls=%s"
                  % (tag, nm, len(labels), ls.mean(), br.mean(), int(mt.sum()),
                     ls[mt].mean() if mt.any() else float("nan"), int(mx.sum()),
                     ("%7.3f" % ls[mx].mean()) if mx.any() else "   -   "))


if __name__ == "__main__":
    main()
