#!/usr/bin/env python3
"""D9 (manuscript R18): what "the bin contributes 63 % of it" is a share OF.

The sentence -- "95.8 % of the population occupies the bin [0.9,1], which carries 63 % of
it" -- never names the denominator, and a population share is not the same statistic as an
ECE share. This prints the three candidate readings side by side on the manuscript's own ten
equal-width bins.

Definition used for the ECE share (stated, not implied):
    ECE      = sum_b (n_b/N) * |conf_b - acc_b|
    share_b  = (n_b/N) * |conf_b - acc_b| / ECE

Usage: python3 experiments/scripts/diag_bin_contrib.py [score_col]
"""
import csv
import sys
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R      # noqa: E402
import verify_paper_numbers as v   # noqa: E402

DIAG = BASE / 'data/archive_20261019/diag'
RUN = BASE / 'data/archive_20260929/scalar/cmm.csv'
NB = 10
FIELDS = ['bin', 'lo', 'hi', 'n', 'share_pop', 'conf', 'acc', 'gap_signed', 'gap_abs',
          'weighted_gap', 'share_of_ece', 'n_errors', 'share_of_errors']


def main():
    col = sys.argv[1] if len(sys.argv) > 1 else 'trustworthiness_smoothed'
    out_name = sys.argv[2] if len(sys.argv) > 2 else 'bin_contrib.csv'
    gt, _ = R.load_gt()
    n, ok, y, s, errs, cands, per = R._pairs(RUN, gt, col)
    N = len(s)
    ece = v.ece(s, y, n=NB)
    tot_err = int((y == 0).sum())
    rows = []
    for b in range(NB):
        lo, hi = b / NB, (b + 1) / NB
        m = (s >= lo) & (s < hi) if b < NB - 1 else (s >= lo) & (s <= hi)
        nb = int(m.sum())
        row = {k: None for k in FIELDS}
        row.update(bin=b, lo=lo, hi=hi, n=nb)
        if nb:
            conf, acc = float(s[m].mean()), float(y[m].mean())
            w = (nb / N) * abs(conf - acc)
            ne = int((y[m] == 0).sum())
            row.update(share_pop=nb / N, conf=conf, acc=acc,
                       gap_signed=conf - acc, gap_abs=abs(conf - acc),
                       weighted_gap=w, share_of_ece=w / ece if ece else float('nan'),
                       n_errors=ne,
                       share_of_errors=ne / tot_err if tot_err else float('nan'))
        rows.append(row)
    DIAG.mkdir(parents=True, exist_ok=True)
    with open(DIAG / out_name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'score={col} n={N} ece10={ece:.8f} errors={tot_err}')
    print(f'{"bin":>3}{"lo":>6}{"hi":>5}{"n":>7}{"pop%":>8}{"conf":>9}{"acc":>9}'
          f'{"gap":>10}{"wtd":>10}{"ece%":>8}{"err":>6}{"err%":>8}')
    for r in rows:
        if not r['n']:
            print(f'{r["bin"]:>3}{r["lo"]:>6.1f}{r["hi"]:>5.1f}{0:>7}  (empty)')
            continue
        print(f'{r["bin"]:>3}{r["lo"]:>6.1f}{r["hi"]:>5.1f}{r["n"]:>7}'
              f'{100*r["share_pop"]:>8.2f}{r["conf"]:>9.5f}{r["acc"]:>9.5f}'
              f'{r["gap_signed"]:>10.5f}{r["weighted_gap"]:>10.6f}'
              f'{100*r["share_of_ece"]:>8.2f}{r["n_errors"]:>6}{100*r["share_of_errors"]:>8.2f}')
    top = [r for r in rows if r['lo'] == 0.9][0]
    print(f'bin[0.9,1]: pop={100*top["share_pop"]:.2f}%  ece_share={100*top["share_of_ece"]:.2f}%'
          f'  error_share={100*top["share_of_errors"]:.2f}%  gap={top["gap_signed"]:+.5f}')
    print(f'wrote {DIAG / out_name} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
