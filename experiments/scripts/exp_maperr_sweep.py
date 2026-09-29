"""Sweep the map-registration term in the emission, at the manuscript's operating point.

WHY. The residual d = z_i - x_ij is not the GNSS error alone. It also carries the
map's own error -- true road position vs digitised centreline -- plus the vehicle's
lateral offset within the carriageway. The candidate SEARCH already allows for both:
that is what delta_road = 13.5 m is, and the manuscript's own decomposition of it is
3.5 m of lane width plus a 10 m map allowance. The EMISSION does not, so Sigma_i
currently claims the residual is the reported GNSS covariance when it is not.

The two are added in the VARIANCE, so for independent sources the physically
motivated value is sqrt(10^2 + 3.5^2) = 10.6 m, not the 13.5 m that delta_road adds
linearly (a radius allowance is a worst case; a variance is not). The sweep brackets
that: 0 (must reproduce the archived run), 3.5, 5, 10, 10.6, 13.5, 20.

WHAT IS HELD FIXED. Everything else, including the direction penalty -- which keeps
the reported covariance on purpose, since kappa = |v|^2/(sde*sdn) is a heading
uncertainty and a misregistered map moves the road, not the vehicle's heading. The
earlier iso* sweeps changed emission width and direction strength together and could
not attribute the effect; this one changes one mechanism.

PASS CRITERION. sigma_map = 0 must reproduce data/archive_20260922/real_final/
cmm_final.csv exactly on the filtering column. If it does not, the run is not at the
manuscript's operating point and no other row may be used.

Usage: python3 experiments/scripts/exp_maperr_sweep.py
"""
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

# Overridable, and following the same CMM_CONFIG switch as real_final_tables, so that a
# candidate operating point can be swept beside the delivered one.
OUTDIR = Path(os.environ.get('CMM_OUTDIR') or (BASE / 'data/archive_20260927/maperr'))
REFERENCE = R.OUTDIR / 'cmm.csv'
GT_CSV = BASE / 'data/real_vehicle/hainan_06/processed/aligned.csv'


def arm_config(out, ms):
    """The operating point, with the map term set to `ms` and the output redirected.

    DERIVED from real_final_tables.CMM_CONFIG rather than restated, because the template
    that used to sit here had drifted: it carried the HPL-adaptive disk (k_search = 1,
    delta_road = 13.5 m, phmi = 1e-5), i.e. ablation-ladder row 7, while the table it
    feeds is row 5 -- the same confusion memory cmm-operating-point-is-row5-fixed-300m
    records for the main table, in a second copy that the ruling did not reach. It also
    anchored its pass criterion to data/archive_20260922, an archive two model revisions
    old. Restating an operating point in a second file is what let both happen; the fix
    is to have one copy of it.
    """
    text = R._config(R.CMM_CONFIG, out)
    pat = re.compile(r'<map_error_sigma>[^<]*</map_error_sigma>')
    want = f'<map_error_sigma>{ms:g}</map_error_sigma>'
    if pat.search(text):
        text = pat.sub(want, text, count=1)
    else:
        anchor = '  </parameters>'
        assert anchor in text, 'no <parameters> block'
        text = text.replace(anchor, f'    {want}\n{anchor}', 1)
    n = len(re.findall(r'<map_error_sigma>', text))
    assert n == 1, f'map_error_sigma: {n} occurrences'
    return text

GPS = 'data/real_vehicle/hainan_06/cmm_input_points_k1.csv'
SIGMAS = [0.0, 3.5, 5.0, 10.0, 10.6, 13.5, 20.0]


def load_gt():
    gt = {}
    for r in csv.DictReader(open(GT_CSV, newline=''), delimiter=';'):
        if r['id'].strip() in v.EXCLUDE_TRAJ:
            continue
        e = r['gt_edge'].strip()
        if e in ('0', '-1', ''):
            continue
        gt[(r['id'].strip(), v.norm_ts(r['timestamp']))] = e
    return gt


def metrics(path, gt, col):
    rows = list(csv.DictReader(open(path, newline=''), delimiter=';'))
    by = {(r['id'].strip(), v.norm_ts(r['timestamp'])): r for r in rows}
    lab, sc = [], []
    for k, g in gt.items():
        r = by.get(k)
        if r is None:
            continue
        c = v.is_edge_match(r['cpath'].strip(), g)
        try:
            t = float(r[col])
        except (ValueError, KeyError):
            continue
        if t <= -900.0:
            continue
        lab.append(1.0 if c else 0.0)
        sc.append(t)
    lab, sc = np.array(lab), np.array(sc)
    p = lab.mean()
    bsc = p * (1 - p)
    bs = float(((sc - lab) ** 2).mean())
    return dict(n=len(lab), acc=100 * lab.mean(), ece=v.ece(sc, lab),
                bss=1 - bs / bsc, auc=v.auc(lab, sc),
                sep=float(sc[lab == 1].mean() - sc[lab == 0].mean()),
                one=100 * (sc >= 1.0).mean(), dist=len(np.unique(sc)),
                lo=100 * (sc < 0.1).mean(), mean=float(sc.mean()))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    gt = load_gt()
    print(f'GT: {len(gt)} evaluable epochs   output dir: {OUTDIR}\n')

    results = {}
    for ms in SIGMAS:
        out = OUTDIR / f'map{ms:g}.csv'
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='w') as fh:
            fh.write(arm_config(out, ms))
            cfg = fh.name
        log = OUTDIR / f'log_map{ms:g}.txt'
        with open(log, 'w') as lf:
            r = subprocess.run([str(BASE / 'build/cmm'), cfg], cwd=BASE,
                               stdout=lf, stderr=subprocess.STDOUT)
        Path(cfg).unlink()
        if r.returncode != 0:
            print(f'sigma_map={ms:5.1f}  RUN FAILED (exit {r.returncode}); see {log}')
            continue
        results[ms] = out
        print(f'sigma_map={ms:5.1f}  rows={sum(1 for _ in open(out))-1:7d}  -> {out.name}')

    # --- pass criterion -----------------------------------------------------
    # The row that must reproduce the operating point is the one AT the operating point's
    # own map_error_sigma, not sigma_map = 0. Anchoring to 0 was correct only while the
    # archived run it compared against was itself run with no map term; the manuscript's
    # operating point has carried 10.6 m since 2026-09-27, so a sigma_map = 0 row differs
    # from it BY CONSTRUCTION and the criterion would fail on a correct sweep.
    op_sigma = float(re.search(r'<map_error_sigma>([^<]*)</map_error_sigma>',
                               R._config(R.CMM_CONFIG, 'x')).group(1))
    if op_sigma not in results:
        print(f'\nTHE OPERATING POINT\'S map_error_sigma ({op_sigma:g} m) IS NOT IN THE '
              f'SWEEP {SIGMAS}.\nNo row can be checked against it. Stopping.')
        return 1
    print(f'\n=== pass criterion: sigma_map = {op_sigma:g} (the operating point) must '
          f'reproduce {R.OUTDIR / "cmm.csv"} ===')
    a = metrics(results[op_sigma], gt, 'trustworthiness')
    b = metrics(REFERENCE, gt, 'trustworthiness')
    keys = ('n', 'acc', 'ece', 'auc', 'sep')
    same = all(abs(a[k] - b[k]) < 1e-9 for k in keys)
    for k in keys:
        print(f'  {k:5s} sweep {a[k]:12.6f}   operating point {b[k]:12.6f}   '
              f'{"OK" if abs(a[k]-b[k]) < 1e-9 else "DIFF"}')
    if not same:
        print(f'\n  THE sigma_map = {op_sigma:g} ROW DOES NOT REPRODUCE THE OPERATING POINT.')
        print('  The sweep is not derived from the same configuration. Stopping.')
        return 1
    print('  reproduces exactly.')

    # --- the sweep ----------------------------------------------------------
    for col, label in (('trustworthiness', 'FILTERING  alpha_t'),
                       ('trustworthiness_smoothed', 'SMOOTHED   alpha_t*beta_t')):
        print(f'\n=== {label} ===')
        hdr = (f"{'sigma_map':>10}{'acc%':>8}{'ECE':>9}{'BSS':>9}{'AUC':>8}"
               f"{'sep':>8}{'=1.0':>8}{'<0.1':>7}{'distinct':>9}{'meanTW':>9}")
        print(hdr)
        for ms in SIGMAS:
            if ms not in results:
                continue
            m = metrics(results[ms], gt, col)
            print(f"{ms:10.1f}{m['acc']:8.2f}{m['ece']:9.4f}{m['bss']:+9.4f}"
                  f"{m['auc']:8.4f}{m['sep']:8.4f}{m['one']:7.1f}%{m['lo']:6.2f}%"
                  f"{m['dist']:9d}{m['mean']:9.4f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
