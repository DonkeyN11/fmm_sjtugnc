#!/usr/bin/env python3
"""Prototype: does the background state x0 fire on the two off-candidate windows?

Standalone reimplementation of the model in ALGO_SPEC_v1 (no C++ change).
Reads: input/map/hainan/edges.shp, data/real_vehicle/hainan_06/processed/aligned.csv
Runs : traj 11, three windows -- two off-candidate (1653-1696, 2502-2511) plus one
       on-network control of the same length.

Outputs the three posteriors for x0 per epoch: delta (Viterbi), alpha (forward),
alphabeta (smoothed), all marginalised over the s grid.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT / "data/real_vehicle/hainan_06/processed/aligned.csv"
SHP_PATH = PROJECT / "input/map/hainan/edges.shp"

TRAJ = "11"
DUMP = []
WINDOWS = [("off-A", 1653, 1696), ("off-B", 2502, 2511), ("ctrl", 1000, 1053)]
PAD = 6                     # extra epochs on each side for DR / transition warm-up

# model knobs (swept below)
SIG_MAP_CROSS = 3.0         # m, across-track map error
SIG_MAP_ALONG = 1.0         # m, along-track
SIG_GEO = 0.0               # m, isotropic: polyline decimation / geometry fidelity
ETA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02   # per-epoch candidate -> x0
SIG_V = 2.0                 # m/s, velocity uncertainty for the DR prior
Q_DR = 0.5                  # m/sqrt(s), extra DR random walk
SIG_TP = 12.0               # m, transition spread on the position discrepancy
S_GRID = np.exp(np.linspace(math.log(0.25), math.log(3.0), 14))
S_STEP = 1.0                # random-walk sd, in grid steps
S_PRIOR_SD = 0.35           # stationary prior on log s (mean-reverting)
R_CAND = 300.0              # m
K_MAX = 16


# ---------------------------------------------------------------- geometry
def load_edges(shp_path):
    from osgeo import ogr
    ds = ogr.Open(str(shp_path))
    if ds is None:
        raise RuntimeError("cannot open %s" % shp_path)
    layer = ds.GetLayer(0)
    defn = layer.GetLayerDefn()
    names = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
    id_field = None
    for cand in ("osmid", "edge_id", "id", "eid", "fid", "edge"):
        for n in names:
            if n.lower() == cand.lower():
                id_field = n
                break
        if id_field:
            break
    if id_field is None:
        id_field = names[0]
    edges = []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        eid = str(feat.GetField(id_field))
        for gi in range(geom.GetGeometryCount() or 1):
            g = geom.GetGeometryRef(gi) if geom.GetGeometryCount() else geom
            if g is None:
                continue
            pts = [(g.GetX(i), g.GetY(i)) for i in range(g.GetPointCount())]
            if len(pts) >= 2:
                edges.append((eid, pts))
    print("edges: %d linestrings, id field = %r, fields = %s"
          % (len(edges), id_field, names[:8]))
    return edges


def to_xy(lon, lat, lon0, lat0, kx, ky):
    return (lon - lon0) * kx, (lat - lat0) * ky


def seg_project(px, py, ax, ay, bx, by):
    """closest point on segment + distance."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1e-12:
        t = 0.0
    else:
        t = ((px - ax) * dx + (py - ay) * dy) / L2
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    qx, qy = ax + t * dx, ay + t * dy
    return qx, qy, math.hypot(px - qx, py - qy), math.atan2(dy, dx), math.sqrt(L2)


def candidates(px, py, edges_xy, r_cand, k_max):
    out = []
    for eid, pts, L in edges_xy:
        best = None
        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            qx, qy, d, ang, segL = seg_project(px, py, ax, ay, bx, by)
            if best is None or d < best[2]:
                best = (qx, qy, d, ang, segL)
        if best is not None and best[2] <= r_cand:
            out.append((eid, best[0], best[1], best[2], best[3], L))
    out.sort(key=lambda z: z[3])
    return out[:k_max]


def build_edges_xy(edges, lon0, lat0, kx, ky):
    out = []
    for eid, pts in edges:
        xy = [to_xy(a, b, lon0, lat0, kx, ky) for a, b in pts]
        L = sum(math.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1])
                for i in range(len(xy) - 1))
        out.append((eid, xy, L))
    return out


# ---------------------------------------------------------------- model
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


def run_window(rows, edges_xy, lon0, lat0, kx, ky, tag):
    T = len(rows)
    z = np.array([to_xy(float(r["obs_x"]), float(r["obs_y"]), lon0, lat0, kx, ky)
                  for r in rows])
    Sobs = []
    for r in rows:
        se = (float(r["obs_sde"]) if r["obs_sde"].strip() else 2.5 / kx) * kx
        sn = (float(r["obs_sdn"]) if r["obs_sdn"].strip() else 2.5 / ky) * ky
        sne = (float(r["obs_sdne"]) if r["obs_sdne"].strip() else 0.0) * kx * ky
        Sobs.append(np.array([[max(se, 0.5) ** 2, sne], [sne, max(sn, 0.5) ** 2]]))
    ns = len(S_GRID)
    cands = [candidates(z[t][0], z[t][1], edges_xy, R_CAND, K_MAX) for t in range(T)]
    Kmax = max(len(c) for c in cands)

    # emission: [T, Kmax+1, ns]
    logE = np.full((T, Kmax + 1, ns), -1e9)
    for t in range(T):
        for k, (eid, qx, qy, d, ang, L) in enumerate(cands[t]):
            dx, dy = z[t][0] - qx, z[t][1] - qy
            ca, sa = math.cos(ang), math.sin(ang)
            R = np.array([[ca, -sa], [sa, ca]])
            M = (R @ np.diag([SIG_MAP_ALONG ** 2, SIG_MAP_CROSS ** 2]) @ R.T
                 + SIG_GEO ** 2 * np.eye(2))
            for j, s in enumerate(S_GRID):
                Sig = s * s * Sobs[t] + M
                logE[t, k, j] = log_gauss2(dx, dy, Sig)
        # background: one-step DR re-anchored at t-1
        if t == 0:
            xi = z[t].copy()
            Sdr = Sobs[t].copy() + 50.0 * np.eye(2)   # no velocity yet -> keep diffuse
        else:
            v = ((z[t - 1] - z[t - 3]) / 2.0 if t >= 3 else
                 (z[t - 1] - z[t - 2] if t >= 2 else np.zeros(2)))
            xi = z[t - 1] + v
            Sdr = Sobs[t - 1] + SIG_V ** 2 * np.eye(2) + Q_DR ** 2 * np.eye(2)
        for j, s in enumerate(S_GRID):
            Sig = s * s * (Sdr + Sobs[t])   # s scales the observation noise in BOTH hypotheses
            logE[t, Kmax, j] = log_gauss2(z[t][0] - xi[0], z[t][1] - xi[1], Sig)

    # transitions, row-normalised over (Kmax+1)
    A = np.zeros((T, Kmax + 1, Kmax + 1))
    pexit = np.zeros(T)

    def _cdf(x):
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
        # background -> candidate: use the ONE-STEP PREDICTION, not the last fix
        v_ = ((z[t - 1] - z[t - 3]) / 2.0 if t >= 3 else
              (z[t - 1] - z[t - 2] if t >= 2 else np.zeros(2)))
        xi_ = z[t - 1] + v_
        Sdr_ = Sobs[t - 1] + (SIG_V ** 2 + Q_DR ** 2) * np.eye(2)
        dmin = min(c[3] for c in cands[t]) if n_cur else 1e9
        sd = math.sqrt(max(0.5 * (Sdr_[0, 0] + Sdr_[1, 1]), 1.0))
        W_BAND = 5.0
        P_re = min(1.0, max(0.0, _cdf((W_BAND - dmin) / sd) - _cdf((-W_BAND - dmin) / sd)))
        w = np.zeros(n_cur)
        for k in range(n_cur):
            eid, qx, qy, d, ang, L = cands[t][k]
            w[k] = math.exp(log_gauss2(qx - xi_[0], qy - xi_[1], Sdr_))
        if w.sum() > 0:
            A[t, Kmax, :n_cur] = (w / w.sum()) * P_re
        A[t, Kmax, Kmax] = 1.0 - P_re
        pexit[t] = P_re
        for j in range(Kmax + 1):
            if j == Kmax:
                A[t, Kmax] /= max(A[t, Kmax].sum(), 1e-12)
                continue
            ssum = A[t, j].sum()
            if ssum <= 0:
                A[t, j, :n_cur] = 1.0
                ssum = A[t, j].sum()
            A[t, j] /= ssum
    A[0] = 1.0 / (Kmax + 1)

    # s random walk
    prior_s = np.exp(-0.5 * (np.log(S_GRID) / S_PRIOR_SD) ** 2)
    As = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            if abs(i - j) <= 4:
                As[i, j] = (math.exp(-((i - j) ** 2) / (2 * S_STEP ** 2))
                            * prior_s[j])
    As = As / As.sum(axis=1, keepdims=True)

    # forward
    logA = np.where(A > 0, np.log(np.maximum(A, 1e-300)), -1e9)
    logAs = np.where(As > 0, np.log(np.maximum(As, 1e-300)), -1e9)
    la = np.full((T, Kmax + 1, ns), -1e9)
    la[0] = logE[0] - math.log(Kmax + 1) - math.log(ns)
    for t in range(1, T):
        tmp = la[t - 1][:, :, None] + logAs[None, :, :]        # (m', s', s)
        tmp = logsumexp(tmp, axis=1)                           # (m', s)
        tmp = logsumexp(tmp[:, None, :] + logA[t][:, :, None], axis=0)   # (m, s)
        la[t] = tmp + logE[t]
    ga = logsumexp(la.reshape(T, -1), axis=1)
    pa = np.exp(la - ga[:, None, None])

    # backward
    lb = np.zeros((T, Kmax + 1, ns))
    for t in range(T - 2, -1, -1):
        core = logE[t + 1] + lb[t + 1]                          # (m', s')
        ys = logsumexp(core[None, :, :] + logAs[:, None, :], axis=2)      # (s, m')
        out = logsumexp(ys[:, None, :] + logA[t + 1][None, :, :], axis=2)  # (s, m)
        lb[t] = out.T
    gab = la + lb
    g2 = logsumexp(gab.reshape(T, -1), axis=1)
    pab = np.exp(gab - g2[:, None, None])

    # Viterbi
    ld = np.full((T, Kmax + 1, ns), -1e9)
    ld[0] = logE[0]
    for t in range(1, T):
        tmp = ld[t - 1][:, :, None] + logAs[None, :, :]
        tmp = np.max(tmp, axis=1)                               # (m', s)
        tmp = np.max(tmp[:, None, :] + logA[t][:, :, None], axis=0)  # (m, s)
        ld[t] = tmp + logE[t]

    p0_a = pa[:, Kmax, :].sum(axis=1)
    p0_ab = pab[:, Kmax, :].sum(axis=1)
    ld_m = np.max(ld, axis=2)
    p0_d = np.exp(ld_m[:, Kmax] - logsumexp(ld_m, axis=1))
    s_hat = [int(np.argmax(ld[t, Kmax, :])) for t in range(T)]

    # diagnostics: DR residual, and the s=1 log Bayes factor vs the best candidate
    j1 = int(np.argmin(np.abs(np.log(S_GRID))))
    bg_d = np.zeros(T)
    lbf = np.zeros(T)
    for t in range(T):
        if t == 0:
            xi = z[t].copy()
        else:
            v = ((z[t - 1] - z[t - 3]) / 2.0 if t >= 3 else
                 (z[t - 1] - z[t - 2] if t >= 2 else np.zeros(2)))
            xi = z[t - 1] + v
        bg_d[t] = float(np.hypot(*(z[t] - xi)))
        lbf[t] = logE[t, Kmax, j1] - logE[t, 0, j1]

    ld_all = np.exp(ld_m - logsumexp(ld_m, axis=1)[:, None])
    pt_d = ld_all[:, :Kmax].max(axis=1)
    pt_a = pa[:, :Kmax, :].sum(axis=2).max(axis=1)
    pt_ab = pab[:, :Kmax, :].sum(axis=2).max(axis=1)
    for t in range(T):
        DUMP.append((tag, rows[t]["uni_seq"], (cands[t][0][3] if cands[t] else float("nan")),
                     float(bg_d[t]), float(lbf[t]), float(1.0 - pexit[t]),
                     float(p0_d[t]), float(p0_a[t]), float(p0_ab[t]),
                     float(pt_d[t]), float(pt_a[t]), float(pt_ab[t]),
                     float(S_GRID[s_hat[t]]), rows[t].get("gt_edge", "")))
    best_d = [cands[t][0][3] if cands[t] else float("nan") for t in range(T)]
    best_edge = [cands[t][0][0] if cands[t] else "-" for t in range(T)]
    matched = [r.get("cmm_cpath", "") for r in rows]
    gt_edge = [r.get("gt_edge", "") for r in rows]

    print("\n=== %s : uni_seq %s..%s (%d epochs) ===" % (tag, rows[0]["uni_seq"],
                                                         rows[-1]["uni_seq"], T))
    print(" t     bestD    bgD  logBF1  Pexit  p0_delta  p0_alpha  p0_alphabeta   sMap  gtEdge")
    stride = max(1, T // 16)
    for t in range(0, T, stride):
        print("%5s %7.1f %6.1f %7.2f %5.3f  %8.3f  %8.3f  %12.3f  %5.2f  %-7s"
              % (rows[t]["uni_seq"], best_d[t], bg_d[t], lbf[t], pexit[t],
                 p0_d[t], p0_a[t], p0_ab[t], S_GRID[s_hat[t]], gt_edge[t][:6]))
    bd_arr = np.array(best_d, dtype=float)
    sl = slice(6, max(7, T - 6))          # drop the pad epochs from the metrics
    bd_s, d_s, a_s = bd_arr[sl], p0_d[sl], p0_a[sl]
    m_near, m_far = bd_s < 3.0, bd_s > 8.0
    fp = float(d_s[m_near].mean()) if m_near.any() else float("nan")
    det = float(d_s[m_far].mean()) if m_far.any() else float("nan")
    fph = float((d_s[m_near] > 0.5).mean() * 100.0) if m_near.any() else float("nan")
    print("  -- summary: mean p0 d=%.3f a=%.3f ab=%.3f | median bestD=%.1f bgD=%.1f m"
          " | mean logBF1=%.2f | s %.2f..%.2f"
          % (d_s.mean(), a_s.mean(), p0_ab[sl].mean(), float(np.nanmedian(bd_arr[sl])),
             float(np.nanmedian(bg_d[sl])), float(np.nanmean(lbf[sl])),
             float(min(S_GRID[q] for q in s_hat)), float(max(S_GRID[q] for q in s_hat))))
    print("     TRADE n_near=%d fp=%.3f fp>0.5=%.0f%% | n_far=%d det=%.3f"
          % (int(m_near.sum()), fp, fph, int(m_far.sum()), det))
    return p0_d, p0_a, p0_ab, best_d


def main():
    rows_all = {}
    with open(CSV_PATH, newline="") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if (r["id"].strip() != TRAJ or not r["uni_seq"].strip()
                    or not r["obs_x"].strip() or not r["obs_y"].strip()):
                continue
            rows_all[int(r["uni_seq"])] = r
    print("traj %s: %d epochs loaded, uni_seq %d..%d"
          % (TRAJ, len(rows_all), min(rows_all), max(rows_all)))
    have = set(rows_all)

    lon0 = float(np.mean([float(r["obs_x"]) for r in rows_all.values()]))
    lat0 = float(np.mean([float(r["obs_y"]) for r in rows_all.values()]))
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110540.0
    print("origin lon/lat = %.6f, %.6f ; kx=%.1f m/deg, ky=%.1f m/deg" % (lon0, lat0, kx, ky))

    edges = load_edges(SHP_PATH)
    edges_xy = build_edges_xy(edges, lon0, lat0, kx, ky)
    xs = [to_xy(float(r["obs_x"]), float(r["obs_y"]), lon0, lat0, kx, ky)
          for r in rows_all.values()]
    x0b, x1b = min(q[0] for q in xs) - 400, max(q[0] for q in xs) + 400
    y0b, y1b = min(q[1] for q in xs) - 400, max(q[1] for q in xs) + 400
    edges_xy = [e for e in edges_xy
                if any(x0b <= x <= x1b and y0b <= y <= y1b for x, y in e[1])]
    print("edges after bbox filter: %d" % len(edges_xy))

    import csv as _csv
    for tag, lo, hi in WINDOWS:
        seq = [i for i in range(lo - PAD, hi + PAD + 1) if i in have]
        if not seq:
            print("!! %s: no rows for %d..%d" % (tag, lo, hi))
            continue
        run_window([rows_all[i] for i in seq], edges_xy, lon0, lat0, kx, ky, tag)
    outp = PROJECT / "docs/calib/figs/proto_offcand_dump.csv"
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["window", "t", "bestD", "bgD", "logBF1", "P0self",
                    "p0_d", "p0_a", "p0_ab", "ptop_d", "ptop_a", "ptop_ab",
                    "sMap", "gt_edge"])
        for r in DUMP:
            w.writerow([r[0], r[1], "%.4f" % r[2]] + ["%.6f" % v for v in r[3:13]] + [r[13]])
    print("dumped", len(DUMP), "rows ->", outp)


if __name__ == "__main__":
    main()
