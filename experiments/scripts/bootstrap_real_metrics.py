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
ARMS = [
    ('CaMM filtering', R.OUTDIR / 'cmm.csv', 'trustworthiness', False),
    ('CaMM smoothed', R.OUTDIR / 'cmm.csv', 'trustworthiness_smoothed', False),
    ('FMM native', R.OUTDIR / 'fmm.csv', 'trustworthiness', True),
    ('FMM posterior', R.OUTDIR / 'fmm_post.csv', 'trustworthiness', True),
    ('FMM smoothed', R.OUTDIR / 'fmm.csv', 'trustworthiness_smoothed', True),
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
    PAIRS = [
        ('FMM native', 'CaMM filtering', 'AUC', 'auc'),
        ('FMM native', 'CaMM filtering', 'separation', 'sep'),
        ('CaMM filtering', 'CaMM smoothed', 'AUC', 'auc'),
        ('CaMM filtering', 'CaMM smoothed', 'separation', 'sep'),
        ('CaMM filtering', 'CaMM smoothed', 'ECE', 'ece'),
        ('CaMM filtering', 'CaMM smoothed', 'BSS', 'bss'),
        ('FMM native', 'FMM posterior', 'ECE', 'ece'),
        ('FMM native', 'FMM posterior', 'AUC', 'auc'),
        ('CaMM filtering', 'FMM native', 'accuracy', 'acc'),
        ('CaMM filtering', 'FMM native', 'pos. error', 'err'),
        ('CaMM filtering', 'FMM native', 'ECE', 'ece'),
        ('CaMM filtering', 'FMM native', 'BSS', 'bss'),
        ('CaMM smoothed', 'FMM native', 'ECE', 'ece'),
        ('CaMM smoothed', 'FMM native', 'BSS', 'bss'),
        ('CaMM smoothed', 'FMM native', 'AUC', 'auc'),
        ('CaMM smoothed', 'FMM smoothed', 'ECE', 'ece'),
        ('CaMM smoothed', 'FMM smoothed', 'BSS', 'bss'),
        ('CaMM filtering', 'FMM smoothed', 'ECE', 'ece'),
    ]
    for a, b, name, key in PAIRS:
        d = np.array(boot[a][key]) - np.array(boot[b][key])
        lo, hi = np.percentile(d, [2.5, 97.5])
        pt = point[a][key] - point[b][key]
        crosses = 'SPANS 0' if lo <= 0.0 <= hi else 'excludes 0'
        print(f'  {name:<11} {a} - {b}: {pt:+8.4f}  [{lo:+8.4f}, {hi:+8.4f}]  {crosses}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
