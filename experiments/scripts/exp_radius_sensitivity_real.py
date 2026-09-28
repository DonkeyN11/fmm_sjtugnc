"""Search-radius sensitivity on the real data: both matchers at r in {100,200,300,500} m.

WHY. Section V searches a fixed 300 m disk, which is a choice and not a derivation, so
it has to be defended twice over: the proposed score's advantage must not be a property
of that particular radius, and the baseline must not be disadvantaged by it. Since both
matchers now search the same disk, varying it moves both arms together, which is what
makes this control cleaner than one that varies the radius for only one of them.

WHAT IS HELD FIXED. Everything else. CaMM keeps its covariance emission, its Mahalanobis
projection, its direction term and its map-registration term, and `k_search = 0` with
`delta_road = r`, so the radius is a constant and the only thing that changes is which
candidates enter the layer. The baseline keeps sigma = 100 m and its native Viterbi
score. The candidate cap k = 16 is unchanged, so at the large radius the cap rather than
the disk may bind --- which is reported, because it is the mechanism by which this
control could fail to be a clean one.

READ THE INTERVALS. Section V establishes that six trajectories cannot resolve most of
the between-matcher differences at r = 300 m, and that bound does not go away when the
radius moves. What this experiment can show is the SHAPE --- whether the accuracy and ECE
curves are flat in r, and whether the two matchers move together --- and that is what the
table is for. It cannot show that CaMM beats the baseline at any single radius.

Usage: python3 experiments/scripts/exp_radius_sensitivity_real.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

OUTDIR = BASE / 'data/archive_20260928/radius'
RADII = [100.0, 200.0, 300.0, 500.0]
M_PER_DEG = 111320.0


def run_cmm(r_m):
    """The paper's operating point with delta_road set to r (k_search = 0, so r IS the disk)."""
    xml = R._config('paper1_cmm.xml', OUTDIR / f'cmm_r{int(r_m)}.csv')
    before = xml
    xml = xml.replace('<delta_road>0.0026949335</delta_road>',
                      f'<delta_road>{r_m / M_PER_DEG!r}</delta_road>')
    assert xml != before, 'delta_road not substituted'
    return _run(xml, 'cmm')


def run_fmm(r_m):
    """The baseline, with its fixed radius set to r and nothing else touched."""
    xml = R._config('paper1_fmm_baseline.xml', OUTDIR / f'fmm_r{int(r_m)}.csv')
    before = xml
    xml = xml.replace('<r>0.0026949335</r>', f'<r>{r_m / M_PER_DEG!r}</r>')
    assert xml != before, 'r not substituted'
    return _run(xml, 'fmm')


def _run(xml, binary):
    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='w') as fh:
        fh.write(xml)
        cfg = fh.name
    r = subprocess.run([str(BASE / 'build' / binary), cfg], cwd=BASE,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    Path(cfg).unlink()
    if r.returncode != 0:
        raise RuntimeError(f'{binary} failed with exit {r.returncode}')


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    gt, ts_by_seq = R.load_gt()
    print(f'{len(gt)} evaluable epochs\n')

    rows = []
    for r_m in RADII:
        run_cmm(r_m)
        run_fmm(r_m)
        cmm = R.evaluate(OUTDIR / f'cmm_r{int(r_m)}.csv', gt, 'trustworthiness')
        cmm_s = R.evaluate(OUTDIR / f'cmm_r{int(r_m)}.csv', gt, 'trustworthiness_smoothed')
        fmm = R.evaluate(OUTDIR / f'fmm_r{int(r_m)}.csv', gt, 'trustworthiness', ts_by_seq)
        rows.append((r_m, cmm, cmm_s, fmm))
        print(f'r = {r_m:5.0f} m   done')

    def table(title, pick):
        print(f'\n=== {title} ===')
        hdr = (f"{'r (m)':>7}{'arm':>16}{'acc %':>9}{'ECE':>9}{'BSS':>9}"
               f"{'AUC':>8}{'sep':>8}{'cand':>7}{'rows':>8}")
        print(hdr)
        print('-' * len(hdr))
        for r_m, cmm, cmm_s, fmm in rows:
            for lab, m in (('CaMM filt', cmm), ('CaMM smooth', cmm_s), ('FMM native', fmm)):
                print(f'{r_m:7.0f}{lab:>16}{m["acc"]:9.2f}{m["ece"]:9.4f}{m["bss"]:+9.3f}'
                      f'{m["auc"]:8.4f}{m["sep"]:8.4f}{m["cand"]:7.2f}{m["n"]:8d}')

    table('as reported', None)

    # Does the ordering of the arms move with r? That is the question this control exists
    # to answer, so it gets its own summary rather than being read off the table.
    print('\n=== does the ECE advantage survive the radius change? ===')
    print(f"{'r (m)':>7}{'ECE CaMM filt':>16}{'ECE FMM native':>17}{'ratio':>9}{'AUC CaMM':>11}{'AUC FMM':>10}")
    for r_m, cmm, cmm_s, fmm in rows:
        print(f'{r_m:7.0f}{cmm["ece"]:16.4f}{fmm["ece"]:17.4f}'
              f'{fmm["ece"]/cmm["ece"]:9.2f}{cmm["auc"]:11.4f}{fmm["auc"]:10.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
