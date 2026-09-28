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
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import verify_paper_numbers as v  # noqa: E402

OUTDIR = BASE / 'data/archive_20260927/maperr'
REFERENCE = BASE / 'data/archive_20260922/real_final/cmm_final.csv'
GT_CSV = BASE / 'data/real_vehicle/hainan_06/processed/aligned.csv'

# The manuscript's operating point, verbatim from newdelta/real_d13p5.xml, with
# <trustworthiness_smoothed/> added so one run yields both arms.
TMPL = """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <input>
    <network><file>input/map/hainan/edges.shp</file><id>key</id><source>u</source><target>v</target></network>
    <ubodt><file>input/map/hainan_ubodt_indexed.bin</file></ubodt>
    <gps>
      <file>{gps}</file>
      <id>id</id><x>x</x><y>y</y><timestamp>timestamp</timestamp>
      <sde>sde</sde><sdn>sdn</sdn><sdu>sdu</sdu>
      <sdne>sdne</sdne><sdeu>sdeu</sdeu><sdun>sdun</sdun>
      <protection_level>protection_level</protection_level>
    </gps>
  </input>
  <output>
    <file>{out}</file>
    <fields><point_mode/><seq/><timestamp/><ogeom/><cpath/><tpath/><opath/><pgeom/>
      <ep/><tp/><trustworthiness/><trustworthiness_smoothed/><n_best_trustworthiness/>
      <candidates/><status/><delta_entropy/><posterior_entropy/><cumu_prob/></fields>
  </output>
  <parameters>
    <k>16</k>
    <min_candidates>1</min_candidates>
    <reverse_tolerance>0.1</reverse_tolerance>
    <use_mahalanobis>true</use_mahalanobis>
    <filtered>true</filtered>
    <max_interval>180.0</max_interval>
    <trustworthiness_threshold>0.0</trustworthiness_threshold>
    <phmi>0.00001</phmi>
    <cumulative_reverse_pct>0.03</cumulative_reverse_pct>
    <k_search>1.0</k_search><delta_road>0.000121272</delta_road>
    <map_error_sigma>{ms}</map_error_sigma>
  </parameters>
  <other>
    <log_level>2</log_level>
    <use_omp>true</use_omp>
    <input_epsg>4326</input_epsg>
  </other>
</config>
"""

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
            fh.write(TMPL.format(gps=str((BASE / GPS).resolve()), out=str(out), ms=ms))
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

    if 0.0 not in results:
        print('\nNO NEUTRAL RUN; cannot establish the operating point. Stopping.')
        return 1

    # --- pass criterion -----------------------------------------------------
    print('\n=== pass criterion: sigma_map = 0 must reproduce the archived run ===')
    a = metrics(results[0.0], gt, 'trustworthiness')
    b = metrics(REFERENCE, gt, 'trustworthiness')
    keys = ('n', 'acc', 'ece', 'auc', 'sep')
    same = all(abs(a[k] - b[k]) < 1e-9 for k in keys)
    for k in keys:
        print(f'  {k:5s} new {a[k]:12.6f}   archived {b[k]:12.6f}   '
              f'{"OK" if abs(a[k]-b[k]) < 1e-9 else "DIFF"}')
    if not same:
        print('\n  NEUTRAL RUN DOES NOT REPRODUCE. Not at the operating point.')
        print('  No other row may be used. Stopping.')
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
