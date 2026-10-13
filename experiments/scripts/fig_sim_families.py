#!/usr/bin/env python3
"""Section IV's stress-family figure: the degraded and mismatch cells, paired, plus one
representative ROC.

WHY THIS EXISTS. The corpus table lists eight stress cells (step fault, occlusion,
both, and five assumed-vs-true pseudorange-noise mismatches) as eight more rows of
numbers; the pattern across them -- that the smoothed posterior's advantage survives
the mismatch family, and what mismatch does to *detection* -- is not visible in a
table. Panels (a) and (b) place the three estimators cell by cell (ECE, AUC) --
delta, filtering alpha, smoothed alpha*beta; alpha tracks delta here as it does on
the whole 1 Hz corpus (within 0.004 ECE, Table I) -- and panel (c) draws the ROC of
all three in the worst case of the mismatch family, which is the measured motivation
for the RAIM-FDE sentence in Section IV.

DATA SOURCE. three_estimator_tables.sim_table(), the same function that prints
Table III, so figure and table cannot disagree. The ROC is computed from the same
per-cell CSVs through exp_sim_sigma_sweep.pairs and the tie-corrected rank AUC of
verify_paper_numbers.auc, the only AUC definition this project quotes.

Colour carries the estimator and matches the sensitivity figure, Fig. 3 (delta #2471A3,
alpha #1E8449, alpha*beta #CA6F1E). The float is single-column (`figure`, stacked
panels); the colour key sits in the caption rather than an in-figure legend.

Usage: python3 experiments/scripts/fig_sim_families.py
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import three_estimator_tables as T  # noqa: E402
import exp_sim_sigma_sweep as S     # noqa: E402
import verify_paper_numbers as v    # noqa: E402

FIGS = BASE / 'docs/calib/figs'
DPI = 300
C = {'delta': '#2471A3', 'alpha': '#1E8449', 'ab': '#CA6F1E'}
MK = {'delta': 'o', 'alpha': 's', 'ab': '^'}
GRAY = '#666666'


def family_label(tag):
    """`sigma_30/no_occlusion/with_fault` -> `fault`; `sigma_mismatch/pr25_wls20` ->
    `25 / 20` (true / assumed sigma_rho, m -- `pr` is the actual noise the generator
    injected, `wls20` the sigma the solver assumed; exp_sim_sigma_sweep.py:88)."""
    m = re.search(r'pr(\d+)_wls(\d+)', tag)
    if m:
        return f'{m.group(1)} / {m.group(2)}'
    fault, occl = 'with_fault' in tag, 'with_occlusion' in tag
    if fault and occl:
        return 'fault + occlusion'
    return 'fault' if fault else 'occlusion'


def family_rows():
    rows = [r for r in T.sim_table() if not re.fullmatch(r'sigma_\d+', r['cell'])]
    deg = [r for r in rows if 'mismatch' not in r['cell']]
    mis = sorted([r for r in rows if 'mismatch' in r['cell']],
                 key=lambda r: int(re.search(r'pr(\d+)', r['cell']).group(1)))
    return deg + mis


def main():
    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 6.5,
        'axes.linewidth': 0.7, 'savefig.dpi': DPI, 'savefig.bbox': 'tight',
    })
    rows = family_rows()
    labels = [family_label(r['cell']) for r in rows]
    y = np.arange(len(rows))[::-1]

    # Single-column width (the float is `figure`, not `figure*`): the three panels
    # stack vertically. The colour key lives in the caption, so no figure legend.
    fig, axs = plt.subplots(3, 1, figsize=(3.45, 4.5),
                            gridspec_kw={'height_ratios': [1.0, 1.0, 1.05]})
    ax_a, ax_b, ax_c = axs

    for ax, key, title in ((ax_a, 'ece', '(a) ECE per stress cell'),
                           (ax_b, 'auc', '(b) mismatch-detection AUC')):
        for i, r in zip(y, rows):
            vals = [r['arms'][k][key] for k in ('delta', 'alpha', 'alpha*beta')]
            ax.plot(vals, [i, i, i], '-', color=GRAY, lw=0.8, zorder=1)
        ax.plot([r['arms']['delta'][key] for r in rows], y, MK['delta'],
                color=C['delta'], ms=3.5)
        ax.plot([r['arms']['alpha'][key] for r in rows], y, MK['alpha'],
                color=C['alpha'], ms=3.2, mfc='white')
        ax.plot([r['arms']['alpha*beta'][key] for r in rows], y, MK['ab'],
                color=C['ab'], ms=3.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is ax_a else [])
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.5, axis='x')
    ax_a.set_xlabel('ECE')
    ax_b.set_xlabel('ROC AUC')
    ax_a.set_ylabel('stress cell')

    # (c) ROC of the worst mismatch cell
    worst = max([r for r in rows if 'mismatch' in r['cell']],
                key=lambda r: int(re.search(r'pr(\d+)', r['cell']).group(1)))
    tag = worst['cell']
    data_dir = dict((t, d) for t, _, d in S.cells())[tag]
    safe = tag.replace('/', '_')
    names = {'delta': r'$\delta$', 'alpha': r'$\alpha$', 'ab': r'$\alpha\beta$'}
    for key, path, col, colr in (
            ('delta', S.OUTDIR / f'{safe}_cmm_scalar-viterbi.csv', 'trustworthiness', 'delta'),
            ('alpha', S.OUTDIR / f'{safe}_cmm_scalar.csv', 'trustworthiness', 'alpha'),
            ('alpha*beta', S.OUTDIR / f'{safe}_cmm_scalar.csv', 'trustworthiness_smoothed', 'ab')):
        yy, ss, _ = S.pairs(data_dir, path, col)
        fpr, tpr, _ = v.roc_curve_and_auc(yy, ss)
        auc = v.auc(yy, ss)
        ax_c.plot(fpr, tpr, '-' + MK[colr], color=C[colr],
                  lw=1.3, ms=3, markevery=max(len(fpr) // 10, 1),
                  label=names[colr] + f'  (AUC {auc:.3f})')
    ax_c.plot([0, 1], [0, 1], ':', color=GRAY, lw=0.8)
    ax_c.set_xlabel('false-alarm rate')
    ax_c.set_ylabel('detection rate')
    ax_c.set_title(f'(c) ROC, mismatch {family_label(tag)}')
    ax_c.legend(loc='lower right', frameon=False)
    ax_c.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / 'sim_families.png'
    fig.savefig(out, facecolor='white')
    fig.savefig(FIGS / 'sim_families.svg', facecolor='white')
    print(f'saved {out} and .svg')
    for r, lab in zip(rows, labels):
        print(f"  {lab:<12} ECE d {r['arms']['delta']['ece']:.4f} "
              f"a {r['arms']['alpha']['ece']:.4f} "
              f"ab {r['arms']['alpha*beta']['ece']:.4f} | AUC "
              f"{r['arms']['delta']['auc']:.3f} / {r['arms']['alpha']['auc']:.3f} / "
              f"{r['arms']['alpha*beta']['auc']:.3f}")


if __name__ == '__main__':
    main()
