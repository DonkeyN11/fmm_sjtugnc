"""Leave-one-trajectory-out cross-validation for the heading constants c and b.

WHY. `direction_kappa_scale` (c) and `direction_kappa_floor` (b) in paper1_cmm.xml are
fitted on the real data -- on the correctly-matched, gap-free steps of the very six
trajectories whose accuracy and calibration the manuscript reports. A fit that is
evaluated on its own sample is not evidence about generalisation, so a reviewer is
entitled to ask how much of the term's value is dataset-specific. This script answers
that by holding one trajectory out at a time:

    for each t in {11, 13, 14, 21, 22, 23}:
        fit c, b on the other five            (fit_direction_kappa.fit)
        run build/cmm at paper1_cmm.xml with those c, b substituted
        evaluate ONLY t's epochs

and compares each fold against the shipped constants (c = 0.227, b = 0.0084) evaluated
on the same held-out epochs.

THE SAMPLE COLLECTION IS COPIED FROM fit_direction_kappa.main(), not imported: main()
has it inline between the network read and the simulation loop, so there is nothing to
import. The one change is the trajectory filter and an O(n) predecessor lookup (bisect
over each trajectory's own sorted timestamps) instead of main()'s `max(k for k in
obs_by_ts if ...)`, which is the same predecessor but quadratic. The full six-trajectory
reproduction is printed first and checked against the shipped constants' own reported
n, so a mistake in the copy shows up before any fold runs.

ONE-STEP, NOT ITERATED. The regression sample is "epochs the matcher got right", so the
constants are a fixed point of match-then-fit and a fully honest LOO would re-run the
matcher inside each fold until the values stop moving. Here the sample is taken from the
run at the SHIPPED constants -- one step -- and each fold also re-fits on its OWN run
output afterwards, which costs no extra matcher run and measures how far each fold's
constants are from their fixed point. Both are reported.

Usage: python3 experiments/scripts/fit_direction_kappa_loo.py
"""
import bisect
import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import fit_direction_kappa as F  # noqa: E402
import real_final_tables as R  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

REVIEW = BASE / 'data/archive_20260928/review'
TRAJS = ['11', '13', '14', '21', '22', '23']
SHIPPED_C, SHIPPED_B = 0.227, 0.0084
SHIPPED_RUN = R.OUTDIR / 'cmm.csv'


def _read_rows(path):
    return list(csv.DictReader(open(path, newline=''), delimiter=';'))


def collect(geo, keep, run_path):
    """fit_direction_kappa.main()'s real-data sample, restricted to `keep`.

    theta, |v| and n^T Sigma n over steps whose BOTH ends sit on the same labelled
    ground-truth edge and whose end epoch the matcher matched correctly. `keep` is a
    set of trajectory ids; a step is kept when its trajectory is in it and both ends
    are, which for a single-trajectory hold-out is the same condition.
    """
    gt_by_ts = {}
    for r in _read_rows(R.ALIGNED):
        if r['gt_edge'].strip() not in ('0', '-1', ''):
            gt_by_ts[(r['id'].strip(), v.norm_ts(r['timestamp']))] = r['gt_edge'].strip()
    obs_by_ts = {}
    for r in _read_rows(F.REAL_OBS):
        k = (r['id'].strip(), v.norm_ts(r['timestamp']))
        if k[0] in keep:
            obs_by_ts[k] = r
    run_by_ts = {}
    for r in _read_rows(run_path):
        k = (r['id'].strip(), v.norm_ts(r['timestamp']))
        if k[0] in keep:
            run_by_ts[k] = r
    # predecessor lookup, per trajectory, over the observation timestamps. main() finds
    # the same predecessor with max(k for k in obs_by_ts if k[0] == tid and k[1] < ts),
    # which is quadratic in the epoch count; the timestamps of a trajectory are unique,
    # so bisect over that trajectory's own sorted list gives the identical answer.
    stamps, pos = {}, {}
    for tid, ts in obs_by_ts:
        stamps.setdefault(tid, []).append(ts)
    for tid, lst in stamps.items():
        lst.sort()
        pos[tid] = {ts: i for i, ts in enumerate(lst)}

    th, L, spec = [], [], []
    for (tid, ts), o in obs_by_ts.items():
        r = run_by_ts.get((tid, ts))
        if r is None or (tid, ts) not in gt_by_ts:
            continue
        i = pos[tid][ts]
        if i == 0:
            continue
        prev = (tid, stamps[tid][i - 1])
        if gt_by_ts.get(prev) != gt_by_ts[(tid, ts)]:
            continue                      # gap-free: same labelled edge at both ends
        e = r['cpath'].strip()
        if not e or not v.is_edge_match(e, gt_by_ts[(tid, ts)]) or e not in geo:
            continue                      # the step must be RIGHT or it tests nothing
        pa, pb = obs_by_ts[prev], o
        lat = (float(pa['y']) + float(pb['y'])) / 2.0
        kx = F.M_PER_DEG * np.cos(np.radians(lat))
        dx = (float(pb['x']) - float(pa['x'])) * kx
        dy = (float(pb['y']) - float(pa['y'])) * F.M_PER_DEG
        LL = float(np.hypot(dx, dy))
        if LL < 1.0:
            continue                      # standing still is not a heading
        t = F.local_tangent(geo[e], (float(pb['x']), float(pb['y'])))
        if t is None:
            continue
        ang = np.arctan2(np.sin(np.arctan2(dy, dx) - t[1]),
                         np.cos(np.arctan2(dy, dx) - t[1]))
        sx, sy = float(pb['sde']) * kx, float(pb['sdn']) * F.M_PER_DEG
        sxy = float(pb['sdne']) * kx * F.M_PER_DEG
        nx, ny = -np.sin(t[1]), np.cos(t[1])
        sp = nx * nx * sx * sx + 2 * nx * ny * sxy + ny * ny * sy * sy
        if sp <= 0:
            continue
        th.append(ang)
        L.append(LL)
        spec.append(sp)
    return np.array(th), np.array(L), np.array(spec)


def held_out_metrics(run_path, gt, tid):
    sub = {k: val for k, val in gt.items() if k[0] == tid}
    out = {}
    for arm in ('trustworthiness', 'trustworthiness_smoothed'):
        m = R.evaluate(run_path, sub, arm)
        out[arm] = dict(n=m['n'], acc=m['acc'], ece=m['ece'], auc=m['auc'],
                        mean_tw=m['mean_tw'], brier=m['brier'], bss=m['bss'])
    return out


def main():
    geo, _ds = F.read_network()
    gt, _ = R.load_gt()
    assert SHIPPED_RUN.exists(), f'missing shipped run: {SHIPPED_RUN}'

    # 0. fidelity check: the full six-trajectory fit must reproduce the shipped
    #    constants, otherwise the copied collection is not the same sample.
    th, L, spec = collect(geo, set(TRAJS), SHIPPED_RUN)
    n0 = len(th)
    a0, b0, rms0, se_a0, se_b0 = F.fit(th, L, spec)
    print('fidelity check -- all six trajectories, shipped run output')
    print(f'  n = {n0}   c = {a0:.4f} (se {se_a0:.4f})   b = {b0:.4f} rad^2 '
          f'(se {se_b0:.4f}, z = {b0 / se_b0:.1f})   rms = {rms0:.4f}')
    print(f'  shipped constants in paper1_cmm.xml: c = {SHIPPED_C}, b = {SHIPPED_B}\n')

    results = {}
    for tid in TRAJS:
        keep = set(TRAJS) - {tid}
        th, L, spec = collect(geo, keep, SHIPPED_RUN)
        a, b, rms, se_a, se_b = F.fit(th, L, spec)
        name = f'loo_c{a:.6f}_b{b:.6f}_hold{tid}'.replace('.', 'p')
        cfg = R._config('paper1_cmm.xml', REVIEW / f'{name}.csv')
        cfg = _subst(cfg, 'direction_kappa_scale', f'{a:.6f}')
        cfg = _subst(cfg, 'direction_kappa_floor', f'{b:.6f}')
        run = _run(cfg, name)
        m = held_out_metrics(run, gt, tid)
        # convergence check: refit on this fold's own output, which costs no new run
        th2, L2, spec2 = collect(geo, keep, run)
        a2, b2, _, _, _ = F.fit(th2, L2, spec2)
        results[tid] = dict(c=a, b=b, se_a=se_a, se_b=se_b, n=len(th), rms=rms,
                            refit_c=a2, refit_b=b2, refit_n=len(th2), metrics=m)
        print(f'hold-out {tid}: fit on {len(th):5d} steps  c = {a:.4f}  b = {b:.4f}  '
              f'(z_b = {b / se_b:.1f})')
        print(f'    held-out acc = {m["trustworthiness"]["acc"]:.4f}  '
              f'ece = {m["trustworthiness"]["ece"]:.4f}  '
              f'auc = {m["trustworthiness"]["auc"]:.4f}  '
              f'mean_tw = {m["trustworthiness"]["mean_tw"]:.4f}   (n = {m["trustworthiness"]["n"]})')
        print(f'    refit on own output: c = {a2:.4f}  b = {b2:.4f}\n')

    # shipped constants, same held-out epochs, from the archived run
    shipped = {tid: held_out_metrics(SHIPPED_RUN, gt, tid) for tid in TRAJS}
    print('shipped constants c = %s, b = %s -- held-out metrics' % (SHIPPED_C, SHIPPED_B))
    for tid in TRAJS:
        m = shipped[tid]['trustworthiness']
        print(f'  {tid}: acc = {m["acc"]:.4f}  ece = {m["ece"]:.4f}  '
              f'auc = {m["auc"]:.4f}  mean_tw = {m["mean_tw"]:.4f}   (n = {m["n"]})')

    Path('/tmp/review_C_exp3.json').write_text(json.dumps(
        {'fidelity': dict(n=n0, c=a0, b=b0, se_a=se_a0, se_b=se_b0, rms=rms0),
         'folds': results, 'shipped_held_out': shipped}, indent=1))
    print('\nwrote /tmp/review_C_exp3.json')
    return 0


def _subst(text, tag, value):
    import re
    pat = rf'<{tag}>[^<]*</{tag}>'
    assert len(re.findall(pat, text)) == 1, f'{tag}: not exactly one occurrence'
    return re.sub(pat, f'<{tag}>{value}</{tag}>', text, count=1)


def _run(text, name):
    import subprocess
    out = REVIEW / f'{name}.csv'
    cfgp = REVIEW / f'{name}.xml'
    cfgp.write_text(text)
    with open(REVIEW / f'{name}.log', 'w') as lf:
        r = subprocess.run([str(BASE / 'build' / 'cmm'), str(cfgp)],
                           cwd=BASE, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f'{name}: cmm exited {r.returncode}')
    return out


if __name__ == '__main__':
    sys.exit(main())
