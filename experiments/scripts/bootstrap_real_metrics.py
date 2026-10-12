"""Trajectory-clustered bootstrap intervals for every headline scalar in §V.

WHY. Every number the manuscript reports is a point estimate quoted to three or four
decimals and then used to compare two matchers. No interval, standard error or
significance test appeared anywhere in §V. On this dataset that omission is not a
formality: the 13,069 epochs come from SIX trajectories, and epochs within a trajectory
share a route, a receiver state and a stretch of map, so they are not independent draws.
An epoch-level standard error would be too small by roughly the design effect, and the
comparisons the paper is built on are between arms evaluated on the SAME trajectories,
which is exactly the case where a clustered interval and an i.i.d. one disagree most.

THE RESAMPLING UNIT IS THE TRAJECTORY. All six are resampled with replacement, and every
epoch of a drawn trajectory comes along. This is the standard cluster bootstrap; the
alternative (resampling epochs) is the one that would produce the too-narrow interval
described above, and it is not offered here because it answers a question nobody asked --
the trajectories are the sampling unit the experiment actually has.

THE CONTRASTS ARE PAIRED. "Is the baseline's AUC higher than ours" is not answered by
comparing two marginal intervals for overlap; it is answered by the distribution of the
DIFFERENCE on the same resampled trajectory set, which removes the between-trajectory
variance that both arms share. Both are printed. Where they disagree, the paired
difference is the one to believe.

WHAT IS AND IS NOT ESTABLISHED BY THIS. A cluster bootstrap with six clusters is a small
-sample procedure: the percentile interval is coarse and slightly too narrow, since the
resampled sets contain on average only 1 - (1-1/6)^6 = 66.5 % of the distinct
trajectories. It is reported as the honest limit of what six trajectories can say, not as
a precise interval. Where an interval spans zero difference, the correct reading is "this
dataset cannot resolve the effect", which is a weaker and more accurate statement than
"there is no effect".

Usage: python3 experiments/scripts/bootstrap_real_metrics.py [--n 2000]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

SEED = 20260928

# ANCHOR. The manuscript reports the SCALAR operating point, whose runs live in
# `archive_20260929/scalar` (`use_mahalanobis=false`, `isotropic_emission_scale=1`). This
# script used to reach its inputs through `R.OUTDIR`, whose default is
# `archive_20260928/final` -- the ANISOTROPIC run. The table itself was generated with
# `CMM_OUTDIR` set, so its numbers are the scalar run's; what was wrong is that a bare
# `python3 bootstrap_real_metrics.py` silently reports the OTHER archive rather than
# failing. Measured 2026-10-09 by asserting all 42 cells of tab:intervals both ways: on
# the anisotropic archive 15 of 42 do not reproduce (BSS rows off by up to 0.024) and
# nothing said so before the assertion block below existed; on the scalar archive all 42
# agree. This is the failure `verify_paper_numbers.py` documents at its own `CMM_CSV` and
# was re-anchored for on 2026-09-30; the same fix was missed here.
# `CMM_OUTDIR` still overrides, for reading another archive deliberately.
OUTDIR = Path(os.environ.get('CMM_OUTDIR') or (BASE / 'data/archive_20260929/scalar'))

# (label, run file, score column, join through the seq->timestamp map?)
#
# Named by ESTIMATOR, not by matcher, because that is the axis the paper ablates: the
# three columns are the Viterbi score delta_t/sum delta_t, the filtering posterior
# alpha_t/sum alpha_t and the smoothed posterior alpha_t beta_t/sum alpha_t beta_t, all
# read off one decoded path. The proposed matcher's delta arm is a separate file because
# tw_use_viterbi is a config key rather than an output column; the baseline's is not,
# because fmm.csv is already run at tw_use_viterbi=true.
ARMS = [
    ('proposed delta', OUTDIR / 'cmm_viterbi.csv', 'trustworthiness', False),
    ('proposed alpha', OUTDIR / 'cmm.csv', 'trustworthiness', False),
    ('proposed ab', OUTDIR / 'cmm.csv', 'trustworthiness_smoothed', False),
    ('baseline delta', OUTDIR / 'fmm.csv', 'trustworthiness', True),
    ('baseline alpha', OUTDIR / 'fmm_post.csv', 'trustworthiness', True),
    ('baseline ab', OUTDIR / 'fmm.csv', 'trustworthiness_smoothed', True),
]


def records(path, col, ts_by_seq, gt):
    """Per-epoch (traj, label, score, position error) for one arm."""
    run = R.load_run(path, ts_by_seq)
    out = []
    for key, (gt_edge, gx, gy) in gt.items():
        r = run.get(key)
        if r is None:
            continue
        try:
            s = float(r[col])
        except (ValueError, KeyError):
            continue
        if s <= -900.0:
            continue
        p = R._pgeom_xy(r)
        err = v.haversine_m(p[0], p[1], gx, gy) if (p and gx == gx and gy == gy) else np.nan
        out.append((key[0], 1.0 if v.is_edge_match(r['cpath'].strip(), gt_edge) else 0.0, s, err))
    return out


def stats(rec):
    y = np.array([r[1] for r in rec])
    s = np.array([r[2] for r in rec])
    e = np.array([r[3] for r in rec])
    e = e[~np.isnan(e)]
    p = y.mean()
    brier = float(((s - y) ** 2).mean())
    return dict(acc=100.0 * y.mean(),
                ece=v.ece(s, y),
                auc=v.auc(y, s),
                sep=float(s[y == 1].mean() - s[y == 0].mean()),
                bss=1.0 - brier / (p * (1.0 - p)),
                err=float(e.mean()) if e.size else np.nan)


def resample(records_by_traj, trajs, rng):
    """Concatenate the epochs of the drawn trajectories."""
    out = []
    for t in rng.choice(trajs, size=len(trajs), replace=True):
        out.extend(records_by_traj[t])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000, help='bootstrap replicates')
    args = ap.parse_args()

    gt, ts_by_seq = R.load_gt()
    data = {}
    for lab, path, col, use_seq in ARMS:
        data[lab] = records(path, col, ts_by_seq if use_seq else None, gt)

    by_traj = {}
    for lab, rec in data.items():
        d = {}
        for r in rec:
            d.setdefault(r[0], []).append(r)
        by_traj[lab] = d
    trajs = sorted(by_traj[ARMS[0][0]])
    print(f'{len(gt)} evaluable epochs over {len(trajs)} trajectories: {trajs}')
    print(f'{args.n} bootstrap replicates, resampling unit = trajectory, seed {SEED}\n')

    rng = np.random.default_rng(SEED)
    # one set of drawn trajectory lists, shared by every arm, so that the arms are
    # compared on the same resamples -- which is what makes the differences paired.
    draws = [list(rng.choice(trajs, size=len(trajs), replace=True)) for _ in range(args.n)]

    boot = {lab: {k: [] for k in ('acc', 'ece', 'auc', 'sep', 'bss', 'err')} for lab in data}
    for drawn in draws:
        for lab in data:
            rec = []
            for t in drawn:
                rec.extend(by_traj[lab][t])
            if len(rec) < 10 or len({r[1] for r in rec}) < 2:
                continue
            st = stats(rec)
            for k, val in st.items():
                boot[lab][k].append(val)

    point = {lab: stats(rec) for lab, rec in data.items()}

    print('=== marginal point estimate and 95 % trajectory-clustered interval ===')
    hdr = f"{'arm':<16}" + ''.join(f'{k:>22}' for k in ('acc %', 'ECE', 'BSS', 'AUC', 'sep', 'err m'))
    print(hdr)
    print('-' * len(hdr))
    for lab in data:
        row = f'{lab:<16}'
        for k in ('acc', 'ece', 'bss', 'auc', 'sep', 'err'):
            lo, hi = np.percentile(boot[lab][k], [2.5, 97.5])
            row += f'{point[lab][k]:8.3f} [{lo:6.3f},{hi:6.3f}]'
        print(row)

    print('\n=== paired differences on the same resampled trajectories ===')
    print('(the interval is of the DIFFERENCE; it spanning 0 means this dataset cannot')
    print(' resolve the contrast, which is weaker than "the two are equal")\n')
    # Three groups, in the order the paper argues them.
    #
    # (1) WITHIN the proposed matcher: the ablation. Same path, same candidates, three
    #     evidence sets. This is the paper's own claim, so every metric is asked of it.
    # (2) AGAINST the baseline at the estimator the baseline actually reports: the
    #     comparison a reader will make.
    # (3) SAME estimator, different observation model: isolates the model from the
    #     estimator, so that a win cannot be attributed to the wrong axis.
    PAIRS = [
        # (1) the ablation
        ('proposed ab', 'proposed delta', 'ECE', 'ece'),
        ('proposed ab', 'proposed delta', 'BSS', 'bss'),
        ('proposed ab', 'proposed delta', 'AUC', 'auc'),
        ('proposed ab', 'proposed alpha', 'ECE', 'ece'),
        ('proposed ab', 'proposed alpha', 'BSS', 'bss'),
        ('proposed ab', 'proposed alpha', 'AUC', 'auc'),
        ('proposed alpha', 'proposed delta', 'ECE', 'ece'),
        ('proposed alpha', 'proposed delta', 'AUC', 'auc'),
        # (2) against the baseline's own reported score
        ('proposed ab', 'baseline delta', 'ECE', 'ece'),
        ('proposed ab', 'baseline delta', 'BSS', 'bss'),
        ('proposed ab', 'baseline delta', 'AUC', 'auc'),
        ('proposed ab', 'baseline delta', 'accuracy', 'acc'),
        ('proposed ab', 'baseline delta', 'pos. error', 'err'),
        # (3) estimator held fixed, model varied
        ('proposed delta', 'baseline delta', 'ECE', 'ece'),
        ('proposed delta', 'baseline delta', 'AUC', 'auc'),
        ('proposed ab', 'baseline ab', 'ECE', 'ece'),
        ('proposed ab', 'baseline ab', 'BSS', 'bss'),
        ('proposed alpha', 'baseline alpha', 'ECE', 'ece'),
    ]
    for a, b, name, key in PAIRS:
        d = np.array(boot[a][key]) - np.array(boot[b][key])
        lo, hi = np.percentile(d, [2.5, 97.5])
        pt = point[a][key] - point[b][key]
        crosses = 'SPANS 0' if lo <= 0.0 <= hi else 'excludes 0'
        print(f'  {name:<11} {a} - {b}: {pt:+8.4f}  [{lo:+8.4f}, {hi:+8.4f}]  {crosses}')

    # ---- tab:intervals, asserted -------------------------------------------------
    # The 24 marginal cells (table top) and the 18 paired-difference rows (table bottom).
    # Nothing checked any of these before 2026-10-09: verify_paper_numbers.py asserts the
    # point ECEs and the accuracies but not one interval endpoint, which is how this
    # script's wrong archive anchor (see ANCHOR above) moved the BSS rows by up to 0.024
    # in silence. The tolerance is the project's own, the one verify_paper_numbers.check
    # uses, so a cell that is off is marked the same way here as anywhere else.
    def _band(paper_v):
        return max(0.005, 0.001 * abs(paper_v)), max(0.02, 0.005 * abs(paper_v))

    def _status(paper_v, comp_v):
        d = abs(comp_v - paper_v)
        te, to = _band(paper_v)
        return '✅ EXACT' if d <= te else '🟡 OK' if d <= to else '⚠️  REVIEW'

    PAPER_MARGINAL = {
        ('proposed delta', 'ece'): (0.042, 0.022, 0.070),
        ('proposed delta', 'bss'): (-0.81, -1.45, -0.10),
        ('proposed delta', 'auc'): (0.840, 0.740, 0.956),
        ('proposed delta', 'sep'): (0.465, 0.297, 0.666),
        ('proposed alpha', 'ece'): (0.038, 0.022, 0.065),
        ('proposed alpha', 'bss'): (-0.75, -1.36, -0.09),
        ('proposed alpha', 'auc'): (0.838, 0.737, 0.955),
        ('proposed alpha', 'sep'): (0.429, 0.273, 0.622),
        ('proposed ab', 'ece'): (0.007, 0.004, 0.016),
        ('proposed ab', 'bss'): (0.27, 0.14, 0.45),
        ('proposed ab', 'auc'): (0.822, 0.710, 0.957),
        ('proposed ab', 'sep'): (0.261, 0.175, 0.367),
        ('baseline delta', 'ece'): (0.193, 0.158, 0.241),
        ('baseline delta', 'bss'): (-2.08, -11.99, -0.55),
        ('baseline delta', 'auc'): (0.865, 0.714, 0.936),
        ('baseline delta', 'sep'): (0.442, 0.109, 0.567),
        ('baseline alpha', 'ece'): (0.364, 0.280, 0.425),
        ('baseline alpha', 'bss'): (-4.78, -30.10, -1.04),
        ('baseline alpha', 'auc'): (0.789, 0.727, 0.842),
        ('baseline alpha', 'sep'): (0.333, 0.259, 0.397),
        ('baseline ab', 'ece'): (0.197, 0.089, 0.277),
        ('baseline ab', 'bss'): (-2.71, -20.23, -0.12),
        ('baseline ab', 'auc'): (0.788, 0.717, 0.899),
        ('baseline ab', 'sep'): (0.352, 0.222, 0.622),
    }
    PAPER_PAIRED = [  # same order as PAIRS above
        (-0.0348, -0.0550, -0.0167), (+1.0816, +0.5176, +1.6275), (-0.0186, -0.0384, +0.0075),
        (-0.0316, -0.0504, -0.0170), (+1.0226, +0.5295, +1.5387), (-0.0164, -0.0356, +0.0096),
        (-0.0032, -0.0051, +0.0004), (-0.0021, -0.0034, -0.0010),
        (-0.1864, -0.2299, -0.1524), (+2.3559, +0.7101, +12.3746), (-0.0434, -0.1767, +0.2073),
        (+2.6857, -0.6235, +8.4417), (-0.3359, -0.8849, -0.0307),
        (-0.1516, -0.1849, -0.1267), (-0.0248, -0.1446, +0.2022),
        (-0.1903, -0.2727, -0.0735), (+2.9807, +0.2761, +20.6315),
        (-0.3258, -0.3983, -0.2217),
    ]

    print('\n=== tab:intervals, asserted cell by cell ===')
    print(f'{"cell":<34}{"paper point":>12}{"comp":>10}{"Δ":>10}   '
          f'{"paper interval":>22}{"comp interval":>22}  status')
    print('-' * 130)
    n_bad = 0
    for (lab, k), (pp, plo, phi) in PAPER_MARGINAL.items():
        p = point[lab][k]
        clo, chi = np.percentile(boot[lab][k], [2.5, 97.5])
        st = max((_status(pp, p), _status(plo, clo), _status(phi, chi)),
                 key=lambda s: ('EXACT', 'OK', 'REVIEW').index(s.split()[-1]))
        n_bad += st.endswith('REVIEW')
        print(f'{lab + " " + k.upper():<34}{pp:>12.4f}{p:>10.4f}{p - pp:>+10.4f}   '
              f'[{plo:+.4f},{phi:+.4f}]    [{clo:+.4f},{chi:+.4f}]    {st}')
    print('-' * 130)
    for (a, b, name, k), (pp, plo, phi) in zip(PAIRS, PAPER_PAIRED):
        d = np.array(boot[a][k]) - np.array(boot[b][k])
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = point[a][k] - point[b][k]
        st = max((_status(pp, p), _status(plo, lo), _status(phi, hi)),
                 key=lambda s: ('EXACT', 'OK', 'REVIEW').index(s.split()[-1]))
        n_bad += st.endswith('REVIEW')
        print(f'{a + " - " + b + " " + name:<34}{pp:>12.4f}{p:>10.4f}{p - pp:>+10.4f}   '
              f'[{plo:+.4f},{phi:+.4f}]    [{lo:+.4f},{hi:+.4f}]    {st}')
    print(f'{len(PAPER_MARGINAL) + len(PAPER_PAIRED)} cells checked, {n_bad} need review')

    # Exact sign-flip test. Six clusters make the percentile interval coarse; a
    # sign-flip test on the per-trajectory differences needs no approximation at all,
    # because all 2^6 = 64 allocations can be enumerated. The statistic is the mean of
    # the six per-trajectory differences, and 1/64 is the smallest p the test returns.
    print('\n=== exact sign-flip test on per-trajectory differences ===')
    for a, b, name, key in (('proposed ab', 'baseline delta', 'ECE', 'ece'),
                            ('proposed delta', 'baseline delta', 'ECE', 'ece'),
                            ('proposed ab', 'baseline delta', 'accuracy', 'acc')):
        d = np.array([stats(by_traj[a][t])[key] - stats(by_traj[b][t])[key]
                      for t in trajs])
        obs = float(d.mean())
        signs = np.array([[1.0 if f >> i & 1 else -1.0 for i in range(len(d))]
                          for f in range(2 ** len(d))])
        flips = (signs * d).mean(axis=1)
        p = float((flips <= obs + 1e-12).mean() if obs < 0
                  else (flips >= obs - 1e-12).mean())
        print(f'  {name:<9} {a} - {b}: mean over trajectories {obs:+8.4f}  '
              f'p = {round(p * 64):.0f}/64')

    # The ECE is a B=10 binned estimate; the ORDERING of the arms must not be an
    # artefact of that choice, and the absolute values must be read as bin-dependent.
    print('\n=== ECE vs bin count ===')
    hdr = f"{'arm':<16}" + ''.join(f'{b:>10}' for b in (5, 10, 15, 20))
    print(hdr)
    print('-' * len(hdr))
    for lab in data:
        y = np.array([r[1] for r in data[lab]])
        s = np.array([r[2] for r in data[lab]])
        print(f'{lab:<16}' + ''.join(f'{v.ece(s, y, n=b):10.4f}' for b in (5, 10, 15, 20)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
