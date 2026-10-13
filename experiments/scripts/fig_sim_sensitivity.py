#!/usr/bin/env python3
"""Section IV's sensitivity figure: the sigma_rho sweep and the sampling-rate sweep.

WHY THIS EXISTS. Section IV carries the 15-cell corpus table and no figure; a reader
cannot see the estimator trade the table asserts (the smoothed posterior's ECE is 1.3
to 7.5 times smaller than the Viterbi score's) or where the claim stops holding. This
figure draws both sweeps on one page:

  (a) smoothed / Viterbi ECE against sigma_rho, seven levels, with the constant
      predictor as the zero reference line;
  (b) the same cells' rank AUC, with the chance line;
  (c), (d)  ECE against the sampling step (12/24/60 m = 1.0/0.5/0.2 Hz) at
      sigma_rho = 10 and 30 m, from data/archive_20261012/lowrate/lowrate.json --
      the corpus family regenerated at the delivered operating point on 2026-10-12.

WHY THE RATE PANELS CARRY THREE SERIES (was five until 2026-10-13). Donkey.Ning's
ruling: every figure in the paper draws the delivered configurations, and the heading
ablation is not one -- it is a diagnostic arm. The no-heading numbers that separate
"the term was removed" from "the rate coarsened" at 0.2 Hz (0.0088 against 0.127 at
1 Hz, sigma_rho = 10 m; 0.084 against 0.101 at 30 m) stay in Section IV-B as TEXT;
they remain reproducible from the same low-rate JSON via exp_sim_lowrate.py, whose
own prints still carry the ablated arms. The panels therefore plot the delivered
model's three estimators (delta, filtering alpha, smoothed alpha*beta) at each rate;
at 0.2 Hz the 2 s gap guard has disabled the heading term for every arm, which the
text states and the caption notes.

DATA SOURCES. Panels (a)/(b) come from three_estimator_tables.sim_table(), the same
function that prints Table I, so figure and table cannot disagree. Panels (c)/(d)
read the low-rate archive's own JSON. Nothing is recomputed here.

Colour carries the estimator (delta #2471A3, alpha #1E8449, alpha*beta #CA6F1E,
matching Fig. 4); linestyle carries the model (solid full, dashed no-heading).
Exit codes: 0 on write.

Usage: python3 experiments/scripts/fig_sim_sensitivity.py
"""
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import three_estimator_tables as T  # noqa: E402

LOWRATE = Path(os.environ.get('CMM_LOWRATE_JSON')
               or (BASE / 'data/archive_20261012/lowrate/lowrate.json'))
FIGS = BASE / 'docs/calib/figs'
DPI = 300

C = {'delta': '#2471A3', 'alpha': '#1E8449', 'ab': '#CA6F1E'}
MK = {'delta': 'o', 'alpha': 's', 'ab': '^'}
GRAY = '#666666'


def sigma_rows():
    """The seven pure sigma_rho levels, ordered by noise."""
    rows = [r for r in T.sim_table() if re.fullmatch(r'sigma_\d+', r['cell'])]
    rows.sort(key=lambda r: float(r['sigma_rho']))
    return rows


def style():
    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
        'axes.linewidth': 0.7, 'savefig.dpi': DPI, 'savefig.bbox': 'tight',
    })


def main():
    style()
    rows = sigma_rows()
    sig = [float(r['sigma_rho']) for r in rows]
    ece_d = [r['arms']['delta']['ece'] for r in rows]
    ece_a = [r['arms']['alpha']['ece'] for r in rows]
    ece_ab = [r['arms']['alpha*beta']['ece'] for r in rows]
    auc_d = [r['arms']['delta']['auc'] for r in rows]
    auc_a = [r['arms']['alpha']['auc'] for r in rows]
    auc_ab = [r['arms']['alpha*beta']['auc'] for r in rows]

    lowrate = json.load(open(LOWRATE))['cells']

    fig, axs = plt.subplots(2, 2, figsize=(7.16, 3.7))

    # (a) ECE vs sigma_rho
    ax = axs[0][0]
    ax.plot(sig, ece_d, '-' + MK['delta'], color=C['delta'], lw=1.4, ms=4,
            label=r'$\delta$ (Viterbi)')
    ax.plot(sig, ece_a, '--' + MK['alpha'], color=C['alpha'], lw=1.2, ms=3.5,
            label=r'$\alpha$ (filtering)')
    ax.plot(sig, ece_ab, '-' + MK['ab'], color=C['ab'], lw=1.4, ms=4,
            label=r'$\alpha\beta$ (smoothed)')
    ax.axhline(0.0, color=GRAY, lw=0.8, ls=':')
    ax.set_xscale('log')
    ax.set_xticks([1, 5, 10, 20, 30])
    ax.set_xticklabels(['1', '5', '10', '20', '30'])
    ax.set_xlabel(r'pseudorange noise $\sigma_\rho$ (m)')
    ax.set_ylabel('ECE')
    ax.set_title('(a) calibration against noise')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(alpha=0.25, lw=0.5)

    # (b) AUC vs sigma_rho
    ax = axs[0][1]
    ax.plot(sig, auc_d, '-' + MK['delta'], color=C['delta'], lw=1.4, ms=4,
            label=r'$\delta$ (Viterbi)')
    ax.plot(sig, auc_a, '--' + MK['alpha'], color=C['alpha'], lw=1.2, ms=3.5,
            label=r'$\alpha$ (filtering)')
    ax.plot(sig, auc_ab, '-' + MK['ab'], color=C['ab'], lw=1.4, ms=4,
            label=r'$\alpha\beta$ (smoothed)')
    ax.axhline(0.5, color=GRAY, lw=0.8, ls=':')
    ax.text(sig[-1], 0.5, ' chance', color=GRAY, fontsize=7.5, va='bottom',
            ha='right')
    ax.set_xscale('log')
    ax.set_xticks([1, 5, 10, 20, 30])
    ax.set_xticklabels(['1', '5', '10', '20', '30'])
    ax.set_xlabel(r'pseudorange noise $\sigma_\rho$ (m)')
    ax.set_ylabel('ROC AUC')
    ax.set_title('(b) mismatch detection against noise')
    ax.legend(loc='lower left', frameon=False)
    ax.grid(alpha=0.25, lw=0.5)

    # (c), (d) ECE vs sampling step -- delivered model only (three solid series);
    # direct end labels, no legend box, auto-stacked in value order
    for ax, sr in ((axs[1][0], '10.0'), (axs[1][1], '30.0')):
        cells = sorted([c for c in lowrate if str(c['sigma_rho']) == sr],
                       key=lambda c: c['step_m'])
        step = [c['step_m'] for c in cells]
        ends = {}
        for key, col, lab in (('delta', 'delta', r'$\delta$'),
                              ('alpha', 'alpha', r'$\alpha$'),
                              ('alpha*beta', 'ab', r'$\alpha\beta$')):
            ax.plot(step, [c['arms'][key]['ece'] for c in cells], '-' + MK[col],
                    color=C[col], lw=1.3, ms=4)
            ends[lab] = ([c['arms'][key]['ece'] for c in cells][-1], col)
        ax.set_xscale('log')
        ax.set_xlim(11, 110)
        ax.margins(y=0.20)
        ymin, ymax = ax.get_ylim()
        gap = 0.085 * (ymax - ymin)
        prev = -1e9
        for lab, (ye, col) in sorted(ends.items(), key=lambda kv: kv[1][0]):
            ytxt = max(ye, prev + gap)
            prev = ytxt
            ax.text(64, ytxt, lab, color=C[col], fontsize=7, va='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.75,
                              pad=0.15))
        ax.set_xticks([12, 24, 60])
        ax.set_xticklabels(['12\n1.0 Hz', '24\n0.5 Hz', '60\n0.2 Hz'])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('sampling step (m) and rate')
        ax.set_ylabel('ECE')
        ax.set_title(f'(c) sampling rate, $\\sigma_\\rho$ = {sr[:-2]} m'
                     if sr == '10.0' else
                     f'(d) sampling rate, $\\sigma_\\rho$ = {sr[:-2]} m')
        ax.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / 'sim_sensitivity.png'
    fig.savefig(out, facecolor='white')
    fig.savefig(FIGS / 'sim_sensitivity.svg', facecolor='white')
    print(f'saved {out} and .svg')
    # the numbers the caption may quote
    for r in rows:
        print(f"  sigma {r['sigma_rho']:>4}: ECE d {r['arms']['delta']['ece']:.4f} "
              f"a {r['arms']['alpha']['ece']:.4f} "
              f"ab {r['arms']['alpha*beta']['ece']:.4f} | AUC d "
              f"{r['arms']['delta']['auc']:.3f} a {r['arms']['alpha']['auc']:.3f} "
              f"ab {r['arms']['alpha*beta']['auc']:.3f}")
    for c in lowrate:
        if str(c['sigma_rho']) in ('10.0', '30.0'):
            print(f"  {c['tag']}: ECE d {c['arms']['delta']['ece']:.4f} "
                  f"a {c['arms']['alpha']['ece']:.4f} "
                  f"ab {c['arms']['alpha*beta']['ece']:.4f} | no-head d "
                  f"{c['arms']['no-heading delta']['ece']:.4f} ab "
                  f"{c['arms']['no-heading alpha*beta']['ece']:.4f}  "
                  f"(no-heading arms are NOT plotted; text-only, see docstring)")


if __name__ == '__main__':
    main()
