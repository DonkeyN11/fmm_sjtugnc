"""Fit the heading-residual model that sets kappa, on the real data and on the corpus.

THE MODEL, AND WHY c IS NOT A FREE PARAMETER.

The direction term scores the observed displacement against the candidate segment's
tangent. Write the displacement as the difference of two fixes,

    v_i = z_i - z_{i-1} = (p_i - p_{i-1}) + (eps_i - eps_{i-1}),

so the displacement error is the DIFFERENCE of two GNSS errors, not one:

    Var(dv) = 2 Sigma - 2 rho Sigma = 2(1 - rho) Sigma  ==  c Sigma.

c is therefore exactly 2(1 - rho), where rho is the coefficient between the GNSS errors
at two CONSECUTIVE epochs. It is not a tuning knob and it has one physical meaning: it
converts a one-epoch covariance into a two-epoch difference covariance. rho is high when
the error is dominated by slowly varying sources (multipath, ionosphere, orbit) and near
zero when it is dominated by per-epoch thermal noise.

rho is NOT in the reported covariance: Sigma_i is the marginal distribution of one epoch,
and rho is a property of the pair. It is however MEASURABLE, and the measurement is a
regression rather than a fit to the metric being reported:

    sigma_theta^2 = a * (n^T Sigma n / |v|^2) + b,

where theta is the angle between the observed displacement and the tangent of the segment
the candidate sits on, n is the unit normal to that tangent, and the sample is restricted
to epochs the matcher got right on a gap-free step. The slope a is the empirical c; the
intercept b is everything the model does not contain.

WHY THE INTERCEPT MATTERS MORE THAN THE SLOPE. If b = 0 the model is complete and a single
constant c suffices for a dataset. If b > 0 there is a heading error with no counterpart in
Sigma at all -- the vehicle's own lateral motion within the epoch, or a breakdown of the
small-angle step theta ~ dv_perp / |v| -- and no value of c can absorb it, because b does
not scale with the covariance. Reporting only the slope hides that, and reporting a slope
fit at one noise level cannot see it.

THE SMALL-ANGLE CAVEAT, stated because it is the first thing to check when b > 0. The
relation sigma_theta^2 = c * n^T Sigma n / |v|^2 linearises theta about zero. It is good
while the perpendicular position error is small against the inter-epoch displacement, and
a corpus sampled at 1 Hz on a slow trajectory can violate that: at 12 m/s the step is
~12 m, and a pseudorange noise of 30 m puts several metres of error across it.

Usage: python3 experiments/scripts/fit_direction_kappa.py
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

M_PER_DEG = 111320.0
REAL_OBS = BASE / 'data/real_vehicle/hainan_06/cmm_input_points_k1.csv'
SIM = BASE / 'data/simulation'
SIM_LEVELS = ['sigma_01', 'sigma_05', 'sigma_10', 'sigma_15', 'sigma_20', 'sigma_25', 'sigma_30']

_TANGENT_CACHE = {}


def read_network():
    """edge id -> polyline in lon/lat, from the same shapefile both pipelines use."""
    from osgeo import ogr
    ds = ogr.Open(str(BASE / 'input/map/hainan/edges.shp'))
    lyr = ds.GetLayer(0)
    out = {}
    for f in lyr:
        g = f.GetGeometryRef()
        out[str(f.GetField('key')).strip()] = [(g.GetX(i), g.GetY(i))
                                               for i in range(g.GetPointCount())]
    return out, ds


def local_tangent(pts, p):
    """Perpendicular distance and tangent angle of the polyline SEGMENT nearest p.

    The segment, not the edge's start->end chord: on a 3.5 km edge the chord is not the
    road's direction where the vehicle is, and using it makes the apparent heading
    scatter grow with speed, which is the opposite of the 1/|v| law.
    """
    best = None
    px, py = p
    k = math.cos(math.radians(py))
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        axk, bxk, pxk = ax * k, bx * k, px * k
        vx, vy = bxk - axk, by - ay
        wx, wy = pxk - axk, py - ay
        L2 = vx * vx + vy * vy
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / L2)) if L2 > 0 else 0.0
        d2 = (wx - t * vx) ** 2 + (wy - t * vy) ** 2
        if best is None or d2 < best[0]:
            best = (d2, math.atan2(vy, vx))
    return best


def samples(obs_rows, run_rows, gt_edges, geo, key_of):
    """theta, |v|, and n^T Sigma n over correctly matched, gap-free, moving steps."""
    obs = {}
    for r in obs_rows:
        k = key_of(r)
        if k is None:
            continue
        obs[k] = (float(r['x']), float(r['y']),
                  float(r['sde']), float(r['sdn']), float(r['sdne']))
    run = {}
    for r in run_rows:
        k = key_of(r)
        if k is not None:
            run[k] = r.get('cpath', '').strip()

    th, vlen, spec = [], [], []
    for tid in sorted({k[0] for k in obs}):
        ks = sorted(k[1] for k in obs if k[0] == tid)
        for i in range(1, len(ks)):
            a, b = ks[i - 1], ks[i]
            if (tid, a) not in gt_edges or (tid, b) not in gt_edges:
                continue
            if gt_edges[(tid, a)] != gt_edges[(tid, b)]:
                continue                     # gap-free: same edge both ends
            e = run.get((tid, b), '')
            if not e or not v.is_edge_match(e, gt_edges[(tid, b)]):
                continue                     # the step must be RIGHT or it tests nothing
            if e not in geo:
                continue
            pa, pb = obs[(tid, a)], obs[(tid, b)]
            lat = (pa[1] + pb[1]) / 2.0
            kx = M_PER_DEG * math.cos(math.radians(lat))
            dx = (pb[0] - pa[0]) * kx
            dy = (pb[1] - pa[1]) * M_PER_DEG
            L = math.hypot(dx, dy)
            if L < 1.0:
                continue                     # standing still is not a heading
            t = local_tangent(geo[e], (pb[0], pb[1]))
            if t is None:
                continue
            ang = math.atan2(math.sin(math.atan2(dy, dx) - t[1]),
                             math.cos(math.atan2(dy, dx) - t[1]))
            # Sigma in metres, same local frame
            sx, sy = pb[2] * kx, pb[3] * M_PER_DEG
            sxy = pb[4] * kx * M_PER_DEG
            nx, ny = -math.sin(t[1]), math.cos(t[1])
            s_perp = nx * nx * sx * sx + 2.0 * nx * ny * sxy + ny * ny * sy * sy
            if s_perp <= 0:
                continue
            th.append(ang)
            vlen.append(L)
            spec.append(s_perp)
    return np.array(th), np.array(vlen), np.array(spec)


def fit(th, L, spec):
    """sigma_theta^2 = c * (spec / L^2) + b, by ordinary least squares.

    The regressor is n^T Sigma n / |v|^2, which is what the model says the covariance
    term is. An earlier fit used 1/|v|^2 alone, which is the same thing only if
    n^T Sigma n is constant -- it is not, it varies epoch to epoch with the reported
    covariance -- and that fit returned 0.36 where this one returns 0.254. The two are
    not two estimates of one quantity: the earlier one is biased by omitting the
    regressor. This is the definition the code and the paper use.

    Standard errors are returned because b is the parameter whose SIGNIFICANCE decides
    whether the floor is real, and a bare number cannot be checked.
    """
    x = spec / (L * L)
    y = th * th
    A = np.vstack([x, np.ones_like(x)]).T
    (ab, *_) = np.linalg.lstsq(A, y, rcond=None)
    a, b = ab
    resid = y - (a * x + b)
    dof = max(len(y) - 2, 1)
    s2 = float((resid ** 2).sum() / dof)
    cov = s2 * np.linalg.inv(A.T @ A)
    se_a, se_b = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    return a, b, float(np.sqrt(s2)), se_a, se_b


def report(label, th, L, spec):
    a, b, rms, se_a, se_b = fit(th, L, spec)
    rho = 1.0 - a / 2.0
    z_b = b / se_b if se_b > 0 else float('nan')
    # THE DIAGNOSTIC THAT DECIDES WHETHER THE TERM IS USABLE AT ALL. kappa is derived
    # from theta ~ dv_perp/|v|, which needs the inter-epoch displacement to be large
    # against the position uncertainty. When this ratio falls toward 1 the observed
    # displacement is mostly noise, its direction says nothing about the heading, and
    # a term that scores it injects noise rather than information. No value of c fixes
    # that, because c scales only the part of the variance that IS in Sigma.
    ratio = np.median(L) / np.sqrt(np.median(spec))
    print(f'{label:<22}{len(th):>7}{np.median(L):>9.1f}{np.sqrt(np.median(spec)):>11.2f}'
          f'{ratio:>9.1f}{a:>9.3f}{rho:>8.2f}{b:>11.4f}{np.degrees(np.sqrt(max(b, 0))):>11.1f}'
          f'{z_b:>8.1f}{rms:>9.4f}')


def main():
    geo, _ds = read_network()
    print('  sig_perp = sqrt(median n^T Sigma n), the position sigma ACROSS the heading.')
    print('  |v|/sig_perp is the heading signal-to-noise: >>1 means the displacement')
    print('  direction is measurable at all. b is the part of sigma_theta^2 that does NOT')
    print('  scale with Sigma; sig_b is its square root in degrees.')
    print()
    print(f'{"dataset":<22}{"n":>7}{"|v| m":>9}{"sig_perp":>11}{"|v|/sig_p":>10}'
          f'{"c (=a)":>9}{"rho":>8}{"b rad^2":>11}{"sig_b deg":>11}{"z(b)":>8}{"rms":>9}')
    print('-' * 100)

    # --- real data ---
    gt, _ = R.load_gt()
    gt_edges = {(k[0], k[1]): e for k, (e, _, _) in gt.items()}
    uni = {}
    for r in csv.DictReader(open(R.ALIGNED, newline=''), delimiter=';'):
        uni[(r['id'].strip(), v.norm_ts(r['timestamp']))] = r['uni_seq'].strip()
    obs_rows = list(csv.DictReader(open(REAL_OBS, newline=''), delimiter=';'))
    run_rows = list(csv.DictReader(open(R.OUTDIR / 'cmm.csv', newline=''), delimiter=';'))

    # CaMM's seq restarts per sub-segment, so both sides are keyed through aligned.csv's
    # uni_seq: real epoch index -> uni_seq. See real_final_tables.load_gt.
    idx = {}
    for r in csv.DictReader(open(R.ALIGNED, newline=''), delimiter=';'):
        idx[(r['id'].strip(), v.norm_ts(r['timestamp']))] = r['uni_seq'].strip()
    seq_ts = {}
    per = {}
    for r in obs_rows:
        per.setdefault(r['id'].strip(), []).append(v.norm_ts(r['timestamp']))
    for tid, st in per.items():
        for i, s in enumerate(sorted(st)):
            seq_ts[(tid, str(i))] = s

    def key_real(r):
        ts = v.norm_ts(r['timestamp'])
        # observations and the run both carry a per-trajectory seq; align by timestamp
        return None if ts is None else ('__ts__', ts)

    # simpler and correct: index both files by (id, timestamp) directly
    obs_by_ts, run_by_ts, gt_by_ts = {}, {}, {}
    for r in obs_rows:
        obs_by_ts[(r['id'].strip(), v.norm_ts(r['timestamp']))] = r
    for r in run_rows:
        run_by_ts[(r['id'].strip(), v.norm_ts(r['timestamp']))] = r
    for r in csv.DictReader(open(R.ALIGNED, newline=''), delimiter=';'):
        if r['gt_edge'].strip() not in ('0', '-1', ''):
            gt_by_ts[(r['id'].strip(), v.norm_ts(r['timestamp']))] = r['gt_edge'].strip()

    th, L, spec = [], [], []
    for (tid, ts), o in obs_by_ts.items():
        r = run_by_ts.get((tid, ts))
        if r is None or (tid, ts) not in gt_by_ts:
            continue
        prev = None
        # find the previous epoch of this trajectory by timestamp
        cand = [k for k in obs_by_ts if k[0] == tid and k[1] < ts]
        if not cand:
            continue
        prev = max(cand)
        if gt_by_ts.get(prev) != gt_by_ts[(tid, ts)]:
            continue
        e = r['cpath'].strip()
        if not e or not v.is_edge_match(e, gt_by_ts[(tid, ts)]) or e not in geo:
            continue
        pa, pb = obs_by_ts[prev], o
        lat = (float(pa['y']) + float(pb['y'])) / 2.0
        kx = M_PER_DEG * math.cos(math.radians(lat))
        dx = (float(pb['x']) - float(pa['x'])) * kx
        dy = (float(pb['y']) - float(pa['y'])) * M_PER_DEG
        LL = math.hypot(dx, dy)
        if LL < 1.0:
            continue
        t = local_tangent(geo[e], (float(pb['x']), float(pb['y'])))
        if t is None:
            continue
        ang = math.atan2(math.sin(math.atan2(dy, dx) - t[1]),
                         math.cos(math.atan2(dy, dx) - t[1]))
        sx, sy = float(pb['sde']) * kx, float(pb['sdn']) * M_PER_DEG
        sxy = float(pb['sdne']) * kx * M_PER_DEG
        nx, ny = -math.sin(t[1]), math.cos(t[1])
        sp = nx * nx * sx * sx + 2 * nx * ny * sxy + ny * ny * sy * sy
        if sp <= 0:
            continue
        th.append(ang); L.append(LL); spec.append(sp)
    report('Haikou (real, 1 Hz)', np.array(th), np.array(L), np.array(spec))

    # --- simulation ---
    import eval_sim_corpus as E
    for lvl in SIM_LEVELS:
        d = SIM / lvl / 'no_occlusion' / 'no_fault'
        if not (d / 'cmm_trajectory.csv').exists():
            continue
        src = [(r['id'].strip(), int(r['seq'])) for r in
               csv.DictReader(open(d / 'observations.csv', newline=''), delimiter=';')]
        gt_edges = E.load_gt_edges(d)
        obs_rows = list(csv.DictReader(open(d / 'observations.csv', newline=''), delimiter=';'))
        run_rows = list(csv.DictReader(open(R.OUTDIR / f'{lvl}_cmm.csv', newline=''), delimiter=';')) \
            if (R.OUTDIR / f'{lvl}_cmm.csv').exists() else None
        if run_rows is None:
            arch = BASE / 'data/archive_20260928/sim' / f'{lvl}_cmm.csv'
            if not arch.exists():
                print(f'{lvl:<22}   (no run; skipping)')
                continue
            run_rows = list(csv.DictReader(open(arch, newline=''), delimiter=';'))
        th, L, spec = samples(obs_rows, run_rows, gt_edges, geo,
                              lambda r: (r['id'].strip(), int(r['seq'])))
        report(f'sim {lvl}', th, L, spec)
    return 0


if __name__ == '__main__':
    sys.exit(main())
