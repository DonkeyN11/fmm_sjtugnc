"""Mark the missing-road stretches as non-evaluable, by setting gt_edge = 0.

WHY. On two stretches of Traj 11 the vehicle drives a road that is absent from the
network. The matcher cannot be right there -- the correct answer is not a candidate --
so the epochs cannot carry a ground-truth edge. They are marked the same way an
off-road epoch is (gt_edge = 0), which is the mechanism the evaluators already use to
exclude an epoch: `gt_edge in {'0','-1',''}` is dropped from the denominator in
verify_paper_numbers and in eval_result_csv.

Donkey.Ning's ruling (2026-09-28): this problem is not solved in paper 1. Detecting a
vehicle driving a missing road needs integrity risk and window-level candidate/trajectory
consistency, and belongs to paper 3. Marking the epochs 0 is the honest state for this
paper rather than reporting the matcher as wrong on a question the network cannot pose.

DETECTING A MISSING ROAD IS NOT POSSIBLE FROM THE PER-EPOCH GNSS SIGNAL, and that is a
measured result, not an assumption. On these epochs the fix is 11 m from the matched edge
with sigma_bar = 10.6 m -- a 1.1 sigma fit -- so a chi-squared test at p < 0.01 flags 0 of
the 61 epochs. The heading residual is 22.29 deg against a core scatter of 1.10 deg, but
its tail is heavier than any von Mises mixture that fits it, so a symmetric heading term
costs 1.4 pp of accuracy and doubles the false-alarm rate (see
tests/test_cmm_direction_penalty.cpp and docs/calib/paper1_work/tw_semantics_design.md).
An off-network state with a real likelihood removes the saturation defect (78.4 % -> 0.0 %
of epochs at exactly 1.0) but leaves these epochs' score unchanged at 0.95.

THE WINDOWS ARE KEYED ON TIMESTAMPS, NOT ON uni_seq. This file used (id, uni_seq) until
2026-09-28, and uni_seq is not stable: it is assigned by align_real_data.py over the union
of the four source tables in timestamp order, so adding one epoch anywhere in a trajectory
renumbers every later epoch of it. Regenerating aligned.csv on 2026-09-28 shifted these
two windows from 1653-1696 and 2502-2511 to 1656-1699 and 2505-2514 -- and since this
script only ever writes '0' and never restores a label, a re-run against the stale indices
would have silently marked 54 WRONG epochs while leaving the intended ones in the
denominator. Timestamps are the stable key; they are what ground_truth.csv is actually
built on and what align_real_data.py joins on.

THE WINDOWS BELOW WERE CONFIRMED BY INSPECTION, not by a threshold. A geometric test
(the fix closer to the matched edge than to the labelled one) found 61 candidate epochs;
54 of them sit in these two runs with a decisive margin (median 88.9 m and 40.6 m), and
the rest have margins of 0.2-3.5 m and are NOT marked here -- those are ordinary matcher
errors, not missing roads. The two windows were then read off satellite imagery and
confirmed by Donkey.Ning: one runs along a small road under the elevated highway before
joining the main road; the other leaves the main road onto a small road.

IDEMPOTENT, AND REVERSIBLE. Running twice changes nothing the second time and reports 0
edits. `--unmark` restores the labelled edge from the backup pair, which is the only way
back: gt_edge = 0 is indistinguishable from a genuine off-network epoch once written.
Usage: python3 experiments/scripts/mark_missing_road_gt.py [--unmark]
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

BASE = Path('/home/ncz/fmm_sjtugnc')
PROC = BASE / 'data/real_vehicle/hainan_06/processed'
GT = PROC / 'ground_truth.csv'
ALIGNED = PROC / 'aligned.csv'
BAK_SUFFIX = '.bak_before_missing_road_mark'

# (trajectory, first timestamp, last timestamp, expected epochs, what the imagery shows)
WINDOWS = [
    ('11', 1750308435, 1750308478, 44,
     'small road under the elevated highway, then joining the main road'),
    ('11', 1750309287, 1750309296, 10,
     'leaving the main road onto a small road'),
]

NON_EVALUABLE = '0'


def ts_of(row):
    return round(float(row['timestamp']))


def targets_by_ts():
    out = {}
    for tid, lo, hi, n, _ in WINDOWS:
        keys = [(tid, t) for t in range(lo, hi + 1)]
        assert len(keys) == n, f'window {tid} {lo}-{hi}: {len(keys)} != declared {n}'
        for k in keys:
            out[k] = True
    return out


def run(mark):
    targets = targets_by_ts()
    print(f'windows: {len(WINDOWS)}, epochs targeted: {len(targets)}')
    if not mark:
        print('MODE: --unmark (restoring from the backups)')

    for path in (GT, ALIGNED):
        if not path.exists():
            print(f'MISSING: {path}', file=sys.stderr)
            return 1

    # Backup once, not per run, so a second invocation cannot overwrite the original.
    for path in (GT, ALIGNED):
        bak = path.with_suffix(path.suffix + BAK_SUFFIX)
        if not bak.exists():
            if mark:
                shutil.copy2(path, bak)
                print(f'backup: {bak.name}')
            else:
                print(f'MISSING BACKUP for --unmark: {bak}', file=sys.stderr)
                return 1
        else:
            print(f'backup already present: {bak.name}')

    # --- rebuild both files from the backup when unmarking -------------------
    if not mark:
        for path in (GT, ALIGNED):
            shutil.copy2(path.with_suffix(path.suffix + BAK_SUFFIX), path)
        print('restored both files from the backups.')
        return report(ALIGNED)

    # --- ground_truth.csv: id;seq;timestamp;x;y;edge_id ----------------------
    with open(GT, newline='') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
        fields = list(rows[0].keys())
        assert 'timestamp' in fields, 'ground_truth.csv layout changed'
    edited, seen = 0, set()
    for r in rows:
        key = (r['id'].strip(), ts_of(r))
        if key in targets:
            seen.add(key)
            if r['edge_id'].strip() != NON_EVALUABLE:
                r['edge_id'] = NON_EVALUABLE
                edited += 1
    with open(GT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=';')
        w.writeheader()
        w.writerows(rows)
    missing = set(targets) - seen
    print(f'ground_truth.csv: {edited} rows set to {NON_EVALUABLE}'
          + (f'   WARNING: {len(missing)} targeted (id,ts) not found: '
             f'{sorted(missing)[:3]}' if missing else ''))

    # --- aligned.csv: same timestamps, the same counter ----------------------
    with open(ALIGNED, newline='') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
        fields = list(rows[0].keys())
    assert 'timestamp' in fields and 'gt_edge' in fields, 'aligned.csv layout changed'
    edited_al, seen = 0, set()
    for r in rows:
        key = (r['id'].strip(), ts_of(r))
        if key in targets:
            seen.add(key)
            if r['gt_edge'].strip() != NON_EVALUABLE:
                r['gt_edge'] = NON_EVALUABLE
                edited_al += 1
    with open(ALIGNED, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=';')
        w.writeheader()
        w.writerows(rows)
    missing = set(targets) - seen
    print(f'aligned.csv:      {edited_al} rows set to {NON_EVALUABLE}'
          + (f'   WARNING: {len(missing)} targeted (id,ts) not found: '
             f'{sorted(missing)[:3]}' if missing else ''))
    return report(ALIGNED)


def report(aligned):
    """What the evaluable population is now, under the evaluators' own rule."""
    sys.path.insert(0, str(BASE / 'experiments/scripts'))
    import verify_paper_numbers as v
    n = 0
    with open(aligned, newline='') as f:
        for r in csv.DictReader(f, delimiter=';'):
            if r['id'].strip() in v.EXCLUDE_TRAJ:
                continue
            if r['gt_edge'].strip() in ('0', '-1', ''):
                continue
            n += 1
    print(f'\nevaluable epochs now: {n}')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--unmark', action='store_true',
                    help='restore both files from the .bak_before_missing_road_mark copies')
    return run(not ap.parse_args().unmark)


if __name__ == '__main__':
    sys.exit(main())
