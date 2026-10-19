#!/usr/bin/env python3
"""The 67 epochs the smoothed posterior reports as exactly 1.000 while the match is wrong.

Author request: a per-epoch, checkable list, plus a graded suspicion of the label so the
"is my ground truth wrong?" hypothesis can be tested rather than argued.

THE SET is defined exactly as the manuscript defines the tail: the delivered scalar run's
`trustworthiness_smoothed == 1.0` and `is_edge_match(cpath, gt_edge) == False`, with the
label and the edge-match rule taken from real_final_tables.load_gt / verify_paper_numbers.
is_edge_match (no private rule). Counts are asserted, not assumed: 67 epochs, 40/14/13 on
trajectories 11/13/22.

OFFSET CLASS is a statement about geometry, not about the label: d_rtk_to_matched_m is how
far the RTK truth sits from the road the matcher chose.
    < 3 m      ambiguous  the two roads cannot be told apart by position alone
    3-10 m     probable   separable, but inside the noise of the fixes
    > 10 m     clear      separable by position; the label is not the explanation

CANDIDATE SET. candidate_membership.json reports n_gt_not_in_disk = 0 and
n_gt_in_disk_dropped_by_cap = 0 over all 13,069 evaluable epochs, so no truth edge can be
missing from the disk. This script re-checks it per epoch for the 67 anyway: gt_in_disk and
gt_rank_in_disk (rank by distance within the disk) are recomputed with the same
candidate_membership.load_segments / min_dist_per_fid functions.

SENSITIVITY. Reclassification of the ambiguous class as correct (or its removal from the
population) changes accuracy, and changes the smoothed posterior's ECE and Brier skill. Both
variants are recomputed through the paper's own metric functions and reported beside the
delivered values.

Usage: python3 experiments/scripts/online67_list.py
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R        # noqa: E402
import verify_paper_numbers as vp    # noqa: E402
import candidate_membership as CM    # noqa: E402

OUT = BASE / 'data/archive_20261019/online67'
RUN = BASE / 'data/archive_20260929/scalar/cmm.csv'
WRONG = BASE / 'data/archive_20261013/wrong_epoch_geometry.csv'
MEMBERSHIP = BASE / 'data/archive_20261013/candidate_membership.json'
TZ = timezone(timedelta(hours=8))
AMB_M, PROB_M = 3.0, 10.0
COLS = ['traj', 'seq', 'uni_seq', 'timestamp_utc', 'timestamp_local', 'gt_edge',
        'matched_edge', 'd_rtk_to_label_m', 'd_rtk_to_matched_m', 'd_spp_to_label_m',
        'd_spp_to_matched_m', 'tw_smoothed', 'label_suspect', 'n_candidates',
        'matched_edge_oneway', 'gt_edge_len_m', 'matched_edge_len_m', 'offset_class',
        'same_node_touch', 'gt_in_disk', 'gt_rank_in_disk', 'n_in_disk']


def edges():
    """FID -> (u, v, oneway, length) from the same shapefile both pipelines use."""
    from osgeo import ogr
    ds = ogr.Open(str(BASE / 'input/map/hainan/edges.shp'))
    ly = ds.GetLayer(0)
    out = {}
    for f in ly:
        out[f.GetFID()] = (f.GetField('u'), f.GetField('v'), f.GetField('oneway'),
                           float(f.GetField('length')))
    return out


def edge_attrs():
    """FID -> the shapefile attributes that say what kind of road it is."""
    from osgeo import ogr
    ds = ogr.Open(str(BASE / 'input/map/hainan/edges.shp'))
    ly = ds.GetLayer(0)
    out = {}
    for f in ly:
        out[f.GetFID()] = {k: f.GetField(k) for k in
                           ('highway', 'name', 'bridge', 'oneway', 'lanes', 'maxspeed')}
        out[f.GetFID()]['length'] = round(float(f.GetField('length')), 2)
    return out


def offset_class(d):
    if d is None:
        return ''
    if d < AMB_M:
        return 'ambiguous'
    if d <= PROB_M:
        return 'probable'
    return 'clear'


def main():
    gt, _ts = R.load_gt()
    run = R.load_run(RUN)
    wrong = {(r['traj'].strip(), int(float(r['timestamp']))): r
             for r in csv.DictReader(open(WRONG, newline=''))}
    print(f'wrong_epoch_geometry rows={len(wrong)}')

    # --- the population, for the sensitivity recomputation ---
    keys, labels, scores = [], [], []
    for key, (ge, _x, _y) in gt.items():
        r = run.get(key)
        if r is None:
            continue
        try:
            s = float(r['trustworthiness_smoothed'])
        except (KeyError, ValueError):
            continue
        if s <= -900.0:
            continue
        keys.append((key[0], vp.norm_ts(r['timestamp'])))
        labels.append(1.0 if vp.is_edge_match(r['cpath'].strip(), ge) else 0.0)
        scores.append(s)
    labels = np.array(labels)
    scores = np.array(scores)
    sevens = [(k, i) for i, k in enumerate(keys)
              if abs(scores[i] - 1.0) < 1e-12 and labels[i] == 0.0]
    print(f'population n={len(labels)}  errors={int((labels == 0).sum())}'
          f'  tw==1 and wrong={len(sevens)}')
    by_traj = Counter(k[0] for k, _ in sevens)
    assert len(sevens) == 67, f'expected 67, got {len(sevens)}'
    assert by_traj == Counter({'11': 40, '13': 14, '22': 13}), by_traj
    print(f'ASSERT ok: 67 epochs, by traj {dict(sorted(by_traj.items()))}')

    # --- per-epoch rows ---
    geo = edges()
    a, grid, _nf = CM.load_segments()
    obs = {}
    for r in csv.DictReader(open(R.ALIGNED, newline=''), delimiter=';'):
        k = (r['id'].strip(), vp.norm_ts(r['timestamp']))
        try:
            obs[k] = (float(r['obs_x']), float(r['obs_y']))
        except (ValueError, KeyError):
            obs[k] = None

    rows, row_keys, missing_wrong, missing_edge = [], [], 0, 0
    for (tid, ts_norm), idx in sevens:
        w = wrong.get((tid, int(float(ts_norm))))
        if w is None:
            missing_wrong += 1
            continue
        r = run[(tid, ts_norm)]
        ts = int(float(ts_norm))
        gtf, mf = w['gt_edge'], w['matched_edge']
        gu = geo.get(int(gtf)) if gtf.isdigit() else None
        mu = geo.get(int(mf)) if mf.isdigit() else None
        if gu is None or mu is None:
            missing_edge += 1
        d = float(w['d_rtk_to_matched_m'])
        shared = ''
        if gu and mu:
            shared = bool({gu[0], gu[1]} & {mu[0], mu[1]})
        gtd, grk, ndisk = '', '', ''
        o = obs.get((tid, ts_norm))
        if o is not None and gtf.isdigit():
            fids, dist = CM.min_dist_per_fid(a, grid, o[0], o[1])
            ndisk = int(len(fids))
            gtd = bool(int(gtf) in set(fids.tolist()))
            if gtd:
                order = np.argsort(dist, kind='stable')
                grk = int(np.where(fids[order] == int(gtf))[0][0]) + 1
        rows.append({
            'traj': tid, 'seq': r['seq'], 'uni_seq': w['uni_seq'],
            'timestamp_utc': datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp_local': datetime.fromtimestamp(ts, TZ).strftime('%Y-%m-%d %H:%M:%S'),
            'gt_edge': gtf, 'matched_edge': mf,
            'd_rtk_to_label_m': w['d_rtk_to_label_m'],
            'd_rtk_to_matched_m': d,
            'd_spp_to_label_m': w['d_spp_to_label_m'],
            'd_spp_to_matched_m': w['d_spp_to_matched_m'],
            'tw_smoothed': r['trustworthiness_smoothed'],
            'label_suspect': w['label_suspect'],
            'n_candidates': R._n_candidates(r),
            'matched_edge_oneway': mu[2] if mu else '',
            'gt_edge_len_m': round(gu[3], 2) if gu else '',
            'matched_edge_len_m': round(mu[3], 2) if mu else '',
            'offset_class': offset_class(d),
            'same_node_touch': shared,
            'gt_in_disk': gtd, 'gt_rank_in_disk': grk, 'n_in_disk': ndisk,
        })
        row_keys.append((tid, ts_norm))
    print(f'rows built={len(rows)}  missing_in_wrong_csv={missing_wrong}'
          f'  missing_edge_attrs={missing_edge}')
    assert len(rows) == 67

    # The 67 are not independent failures: group them by the (truth, matched) road pair,
    # because two of those pairs explain almost all of them, and the pair's identity is
    # what the author has to look at in the error maps.
    attrs = edge_attrs()
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['gt_edge'], r['matched_edge'])].append(r)
    episodes = []
    for (g, m), rs in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        episodes.append(dict(
            gt_edge=g, matched_edge=m, n=len(rs),
            trajs=sorted({x['traj'] for x in rs}),
            offset_min_m=min(float(x['d_rtk_to_matched_m']) for x in rs),
            offset_max_m=max(float(x['d_rtk_to_matched_m']) for x in rs),
            gt_attrs=attrs.get(int(g)) if g.isdigit() else None,
            matched_attrs=attrs.get(int(m)) if m.isdigit() else None))
    for e in episodes:
        print(f'  episode gt={e["gt_edge"]} matched={e["matched_edge"]} n={e["n"]}'
              f' traj={e["trajs"]} offset {e["offset_min_m"]}-{e["offset_max_m"]} m'
              f' | gt {e["gt_attrs"]} | matched {e["matched_attrs"]}')

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'epochs67.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    # --- grading ---
    cls = Counter(r['offset_class'] for r in rows)
    d_by = defaultdict(list)
    for r in rows:
        d_by[r['offset_class']].append(float(r['d_rtk_to_matched_m']))
    cross = defaultdict(Counter)
    for r in rows:
        cross[r['traj']][r['offset_class']] += 1
    amb_keys = {k for r, k in zip(rows, row_keys) if r['offset_class'] == 'ambiguous'}
    print(f'ambiguous keys resolved={len(amb_keys)} (expected {cls.get("ambiguous", 0)})')
    assert len(amb_keys) == cls.get('ambiguous', 0)
    pos = {k: i for i, k in enumerate(keys)}

    # --- sensitivity on the whole population ---
    def metrics(lab, sc):
        p = lab.mean()
        br = float(((sc - lab) ** 2).mean())
        return dict(n=int(len(lab)), acc=round(100.0 * lab.mean(), 6),
                    ece=round(float(vp.ece(sc, lab)), 8),
                    brier=round(br, 8),
                    bss=round(1 - br / (p * (1 - p)), 8) if 0 < p < 1 else None)
    base = metrics(labels, scores)
    flip_lab = labels.copy()
    for k in amb_keys:
        flip_lab[pos[k]] = 1.0
    flip = metrics(flip_lab, scores)
    keep = np.array([i for i, k in enumerate(keys) if k not in amb_keys])
    excl = metrics(labels[keep], scores[keep])
    n_amb = len(amb_keys)
    all_errors = int((labels == 0).sum())
    # A looser threshold, so the answer does not rest on where 3 m was drawn: everything
    # under 10 m reclassified as correct. Reported beside the 3 m variant.
    amb10 = {k for r, k in zip(rows, row_keys) if float(r['d_rtk_to_matched_m']) < 10.0}
    flip10_lab = labels.copy()
    for k in amb10:
        flip10_lab[pos[k]] = 1.0
    flip10 = metrics(flip10_lab, scores)
    # Upper bounds, so the answer to "could the label explain the tail?" is bounded rather
    # than only sampled: (i) the whole dual-carriageway trunk episode reclassified as
    # correct, (ii) all 67 reclassified, which no reading of the data supports but which
    # says what the tail is worth at most.
    ep2 = {k for r, k in zip(rows, row_keys)
           if (r['gt_edge'], r['matched_edge']) == ('149747', '149873')}
    flip_ep2 = labels.copy()
    for k in ep2:
        flip_ep2[pos[k]] = 1.0
    m_ep2 = metrics(flip_ep2, scores)
    flip_all = labels.copy()
    for _k, i in sevens:
        flip_all[i] = 1.0
    m_all = metrics(flip_all, scores)
    summary = dict(
        n_total=len(rows), by_traj=dict(sorted(by_traj.items())),
        episodes=episodes,
        offset_class={c: dict(n=cls.get(c, 0),
                              pct=round(100.0 * cls.get(c, 0) / len(rows), 3),
                              d_median_m=round(float(np.median(d_by[c])), 3) if d_by[c] else None,
                              d_min_m=min(d_by[c]) if d_by[c] else None,
                              d_max_m=max(d_by[c]) if d_by[c] else None)
                      for c in ('ambiguous', 'probable', 'clear')},
        label_suspect=dict(n=sum(1 for r in rows if r['label_suspect'] == '1'),
                           which=[(r['traj'], r['uni_seq']) for r in rows
                                  if r['label_suspect'] == '1']),
        crosstab={t: dict(cross[t]) for t in sorted(cross)},
        same_node_touch=dict(n=sum(1 for r in rows if r['same_node_touch'] is True)),
        candidate_set=dict(
            gt_in_disk_all=all(r['gt_in_disk'] for r in rows),
            gt_in_disk_n=sum(1 for r in rows if r['gt_in_disk'] is True),
            rank_min=min((r['gt_rank_in_disk'] for r in rows
                          if r['gt_rank_in_disk'] != ''), default=None),
            rank_max=max((r['gt_rank_in_disk'] for r in rows
                          if r['gt_rank_in_disk'] != ''), default=None),
            rank_ge_16=sum(1 for r in rows
                           if r['gt_rank_in_disk'] != '' and r['gt_rank_in_disk'] >= 16),
            n_in_disk_min=min((r['n_in_disk'] for r in rows if r['n_in_disk'] != ''),
                              default=None),
            n_in_disk_max=max((r['n_in_disk'] for r in rows if r['n_in_disk'] != ''),
                              default=None),
            membership_json=json.load(open(MEMBERSHIP))['n_gt_in_disk_dropped_by_cap']),
        sensitivity=dict(delivered=base, ambiguous_as_correct=flip,
                         ambiguous_excluded=excl,
                         ambiguous_lt10_as_correct=flip10,
                         trunk_episode_as_correct=m_ep2,
                         all67_as_correct=m_all,
                         n_trunk_episode=len(ep2),
                         n_ambiguous=n_amb,
                         n_under_10m=len(amb10),
                         tail_remaining_after_removal=67 - n_amb,
                         tail_remaining_pct_of_all_errors=round(
                             100.0 * (67 - n_amb) / all_errors, 3),
                         all_errors=all_errors))
    (OUT / 'epochs67_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # --- human list ---
    lines = ['# The 67 epochs reported as exactly 1.000 and matched wrong', '',
             'Offset = d_rtk_to_matched_m (RTK truth to the chosen road). '
             'ambiguous <3 m, probable 3-10 m, clear >10 m.', '']
    lines.append('## Road pairs behind the 67 (two pairs explain 66)')
    lines.append('')
    lines.append('| truth edge | matched edge | n | traj | offsets m | gt highway/len | matched highway/len |')
    lines.append('|---|---|---|---|---|---|---|')
    for e in episodes:
        ga, ma = e['gt_attrs'] or {}, e['matched_attrs'] or {}
        lines.append(f'| {e["gt_edge"]} | {e["matched_edge"]} | {e["n"]} | '
                     f'{",".join(e["trajs"])} | {e["offset_min_m"]:.1f}-{e["offset_max_m"]:.1f} | '
                     f'{ga.get("highway")}/{ga.get("length")} m | '
                     f'{ma.get("highway")}/{ma.get("length")} m |')
    lines.append('')
    for t in ('11', '13', '22'):
        sub = [r for r in rows if r['traj'] == t]
        lines.append(f'## Trajectory {t} ({len(sub)} epochs)')
        lines.append('')
        lines.append('| # | local time (+08) | edge | offset m | class | label_suspect |')
        lines.append('|---|---|---|---|---|---|')
        for i, r in enumerate(sub, 1):
            lines.append(f'| {i} | {r["timestamp_local"]} | {r["matched_edge"]} | '
                         f'{r["d_rtk_to_matched_m"]} | {r["offset_class"]} | '
                         f'{r["label_suspect"]} |')
        lines.append('')
    (OUT / 'epochs67.md').write_text('\n'.join(lines) + '\n')

    print(json.dumps(summary, indent=2, ensure_ascii=False)[:2600])
    print(f'wrote {OUT}/epochs67.csv, epochs67_summary.json, epochs67.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
