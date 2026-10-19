#!/usr/bin/env python3
"""D7 (manuscript R22): Table VI prints the proposed delta ECE as .041, Table II/III as
.042, on what is claimed to be the same archived run. Recompute through the paper's own
metric layer at full precision, and repeat at 15 and 20 bins, so the digit is decided by
arithmetic rather than by transcription.

Usage: python3 experiments/scripts/diag_rounding_check.py
"""
import sys
from pathlib import Path

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R   # noqa: E402
import verify_paper_numbers as v  # noqa: E402

DIAG = BASE / 'data/archive_20261019/diag'
TARGETS = [
    ('delivered delta (archive cmm_viterbi)', BASE / 'data/archive_20260929/scalar/cmm_viterbi.csv'),
    ('per-epoch raw transition delta (tpattr)', BASE / 'data/archive_20260930/tpattr/cm_raw_viterbi.csv'),
    ('delivered alpha (archive cmm)', BASE / 'data/archive_20260929/scalar/cmm.csv'),
]


def main():
    gt, _ = R.load_gt()
    lines = []
    for label, path in TARGETS:
        if not path.exists():
            lines.append(f'{label}\tMISSING\t{path}')
            continue
        col = 'trustworthiness'
        n, ok, lab, sc, errs, cands, per = R._pairs(path, gt, col)
        for nb in (10, 15, 20):
            e = v.ece(sc, lab, n=nb)
            lines.append(f'{label}\tbins={nb}\tn={len(sc)}\tacc={100.0*ok/n:.6f}'
                         f'\tECE={e:.8f}\t3dp={e:.3f}')
        # the exact bin-[0.9,1] population share, for the R18 companion check
        lines.append(f'{label}\tshare_s>=0.9={float((sc >= 0.9).mean()):.6f}'
                     f'\tshare_s>=1.0={float((sc >= 1.0).mean()):.6f}')
    DIAG.mkdir(parents=True, exist_ok=True)
    (DIAG / 'rounding_check.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
