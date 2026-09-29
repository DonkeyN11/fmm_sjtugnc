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
import sys
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

SEED = 20260928

# (label, run file, score column, join through the seq->timestamp map?)
#
# Named by ESTIMATOR, not by matcher, because that is the axis the paper ablates: the
# three columns are the Viterbi score delta_t/sum delta_t, the filtering posterior
# alpha_t/sum alpha_t and the smoothed posterior alpha_t beta_t/sum alpha_t beta_t, all
# read off one decoded path. The proposed matcher's delta arm is a separate file because
# tw_use_viterbi is a config key rather than an output column; the baseline's is not,
# because fmm.csv is already run at tw_use_viterbi=true.
ARMS = [
    ('proposed delta', R.OUTDIR / 'cmm_viterbi.csv', 'trustworthiness', False),
    ('proposed alpha', R.OUTDIR / 'cmm.csv', 'trustworthiness', False),
    ('proposed ab', R.OUTDIR / 'cmm.csv', 'trustworthiness_smoothed', False),
    ('baseline delta', R.OUTDIR / 'fmm.csv', 'trustworthiness', True),
    ('baseline alpha', R.OUTDIR / 'fmm_post.csv', 'trustworthiness', True),
    ('baseline ab', R.OUTDIR / 'fmm.csv', 'trustworthiness_smoothed', True),
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
