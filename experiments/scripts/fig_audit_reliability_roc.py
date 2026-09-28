"""The audit's own figure: reliability diagram and ROC curve, four arms, one page.

WHY THIS EXISTS. Section V of the manuscript is an audit of whether the reported
confidence is a probability, and it cites "reliability diagrams" in the abstract and in
the introduction. Until now the paper carried no reliability diagram and no ROC curve:
every audit result was a number in a table. A calibration paper whose central exhibit has
no calibration plot is the first thing a referee notices.

WHAT THE TWO PANELS SHOW.

  Left, reliability: the unit interval in ten equal-width bins, empirical frequency of
  correctness against the mean score in the bin, marker area proportional to the bin's
  population. The diagonal is a perfectly calibrated score. The large marker crowded
  against the right edge is the saturation this paper reports -- 68 to 80 per cent of
  epochs sit at exactly 1.000, so one bin carries most of the mass and the diagram can
  say very little about the interior. That is drawn, not hidden: the top-bin share is
  annotated for each arm.

  Right, ROC: mismatch detection, with the tie-corrected rank AUC of
  verify_paper_numbers.auc(). The curves cross and the AUCs are close, which is the
  honest picture and matches Section V's finding that six trajectories cannot rank the
  two matchers on discrimination.

COLOUR. Fixed order, never cycled: CaMM filtering, CaMM smoothed, FMM native, FMM
posterior. The palette is checked by the dataviz validator for the lightness band, chroma
floor, CVD separation, normal-vision floor and contrast; it passes all five. Identity is
never colour-alone -- every series is direct-labelled in both panels.

Inputs are the runs written by experiments/scripts/real_final_tables.py.
Usage: python3 experiments/scripts/fig_audit_reliability_roc.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

FIG = BASE / 'docs/calib/figs'

# (label, run, column, seq->ts map, colour, linestyle, marker)
ARMS = [
    ('CaMM filtering', R.OUTDIR / 'cmm.csv', 'trustworthiness', False,
     '#2471A3', '-', 'o'),
    ('CaMM smoothed', R.OUTDIR / 'cmm.csv', 'trustworthiness_smoothed', False,
     '#16A085', '-', 's'),
    ('FMM native', R.OUTDIR / 'fmm.csv', 'trustworthiness', True,
     '#C0392B', '--', '^'),
    ('FMM posterior', R.OUTDIR / 'fmm_post.csv', 'trustworthiness', True,
     '#8E44AD', '--', 'D'),
]

NBINS = 10


def pairs(path, col, ts_by_seq, gt):
    run = R.load_run(path, ts_by_seq)
    y, s = [], []
    for key, (gt_edge, _, _) in gt.items():
        r = run.get(key)
        if r is None:
            continue
        try:
            sc = float(r[col])
        except (ValueError, KeyError):
            continue
        if sc <= -900.0:
            continue
        y.append(1.0 if v.is_edge_match(r['cpath'].strip(), gt_edge) else 0.0)
        s.append(sc)
    return np.array(y), np.array(s)


def reliability(y, s, nbins=NBINS):
    """Per-bin (mean score, empirical frequency, count), empty bins dropped."""
    edges = np.linspace(0.0, 1.0, nbins + 1)
    out = []
    for b in range(nbins):
        lo, hi = edges[b], edges[b + 1]
        m = (s >= lo) & (s <= hi) if b == nbins - 1 else (s >= lo) & (s < hi)
        n = int(m.sum())
        if n:
            out.append((float(s[m].mean()), float(y[m].mean()), n))
    return out


def roc(y, s):
    """Empirical ROC step curve. The scalar AUC is the rank statistic, not its area."""
    order = np.argsort(-s, kind='mergesort')
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    npos, nneg = y.sum(), (1.0 - y).sum()
    tpr = np.concatenate([[0.0], tp / npos])
    fpr = np.concatenate([[0.0], fp / nneg])
    return fpr, tpr


def main():
    gt, ts_by_seq = R.load_gt()
    fig, (axr, axc) = plt.subplots(1, 2, figsize=(7.16, 3.05), dpi=300)

    # ---------------- reliability ----------------
    # Markers only, no connecting line. Each bin is an independent conditional
    # frequency, not a sample of a curve, and with six or seven non-empty bins a
    # polyline through them draws a shape the data does not contain. Overlapping
    # markers are dodged along x by a small per-arm offset so the saturated bin --
    # where four arms land on top of each other -- stays readable.
    axr.plot([0, 1], [0, 1], '-', color='#888888', lw=1.0, zorder=1)
    axr.text(0.76, 0.63, 'perfectly calibrated', rotation=38, fontsize=6.2,
             color='#666666', ha='center', va='center', rotation_mode='anchor')

    exact_one = {}
    for k, (lab, path, col, use_seq, c, ls, mk) in enumerate(ARMS):
        y, s = pairs(path, col, ts_by_seq if use_seq else None, gt)
        bins = reliability(y, s)
        n = np.array([b[2] for b in bins], float)
        dx = (k - 1.5) * 0.024
        # radius ∝ population (so area ∝ n, as a reliability diagram requires), with
        # a floor so a small bin is still visible and a cap so the saturated one does
        # not swallow the panel.
        size = 4.0 + 95.0 * np.sqrt(n / n.sum())
        axr.scatter([b[0] + dx for b in bins], [b[1] for b in bins], s=size, c=c,
                    marker=mk, zorder=4, edgecolors='white', linewidths=0.6,
                    label=lab)
        exact_one[lab] = 100.0 * float((s >= 1.0).mean())

    axr.set_xlim(-0.02, 1.02)
    axr.set_ylim(-0.02, 1.05)
    axr.set_xlabel('mean reported confidence in bin', fontsize=7.5)
    axr.set_ylabel('empirical frequency of correctness', fontsize=7.5)
    axr.set_title('Reliability', fontsize=8.5, pad=4)
    axr.tick_params(labelsize=6.5)
    axr.grid(alpha=0.25, lw=0.5)
    axr.legend(loc='lower right', fontsize=5.8, framealpha=0.92, handlelength=0.9,
               borderpad=0.35, labelspacing=0.3, markerscale=0.7,
               handletextpad=0.4)
    # The saturation that limits this panel, stated on the panel. This is the share at
    # EXACTLY 1.000, which is not the top bin's population ([0.9, 1.0]) -- the two are
    # different numbers and quoting the bin's for the point mass was a real error in
    # the first draft of this figure.
    share = '\n'.join(f'  {lab:<15}{v:5.1f} %' for lab, v in exact_one.items())
    axr.text(0.03, 0.33, 'epochs at exactly 1.000:\n' + share, fontsize=5.8,
             color='#333333', family='monospace', va='top', zorder=6,
             bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#cccccc', lw=0.6))

    # ---------------- ROC ----------------
    axc.plot([0, 1], [0, 1], '-', color='#888888', lw=1.0, zorder=1)
    for lab, path, col, use_seq, c, ls, mk in ARMS:
        y, s = pairs(path, col, ts_by_seq if use_seq else None, gt)
        fpr, tpr = roc(y, s)
        a = v.auc(y, s)
        axc.plot(fpr, tpr, ls, color=c, lw=1.4, zorder=3)
        # direct labels instead of a legend box: identity is never colour-alone, and
        # the AUC belongs beside the curve it describes.
        lo, hi = np.percentile(s, [2.5, 97.5])
        axc.annotate(f'{lab}  AUC {a:.3f}', xy=(0.98, 0.30), fontsize=5.8, color=c,
                     ha='right', va='center')
    # place the labels without overlap: recompute y positions in draw order
    ys = np.linspace(0.34, 0.14, len(ARMS))
    for txt, yy in zip(list(axc.texts), ys):
        if txt.get_text().startswith('perfect'):
            continue
        txt.set_position((0.98, yy))

    axc.set_xlim(-0.02, 1.02)
    axc.set_ylim(-0.02, 1.02)
    axc.set_xlabel('false-positive rate', fontsize=7.5)
    axc.set_ylabel('true-positive rate', fontsize=7.5)
    axc.set_title('ROC: trustworthiness as a mismatch detector', fontsize=8.5, pad=4)
    axc.tick_params(labelsize=6.5)
    axc.grid(alpha=0.25, lw=0.5)

    fig.tight_layout(pad=0.4)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        out = FIG / f'audit_reliability_roc.{ext}'
        fig.savefig(out, facecolor='white')
        print(f'Saved: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
