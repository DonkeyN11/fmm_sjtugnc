#!/usr/bin/env python3
"""
Deep dive: PHMI bin-level calibration and off-road trustworthiness analysis.

1. ECE bin distribution at phmi_pl_multiplier=5 vs baseline
2. Traj21 first 100 epochs: trustworthiness when candidates outside PL
3. Generic analysis: does PHMI flag epochs where all candidates exceed PL?

Usage:
  python3 tests/python/exp3_deep_dive.py
"""

import csv, math, re
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[2]
INPUT_CSV = BASE / 'dataset-hainan-06/cmm_input_points.csv'
THRESHOLD_M = 5.0
METERS_PER_DEG = 111_000.0

PP_RE = re.compile(r'POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)', re.I)

def parse_point(wkt):
    m = PP_RE.search(wkt or '')
    return (float(m.group(1)), float(m.group(2))) if m else None

def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    dlon, dlat = math.radians(lon2 - lon1), math.radians(lat2 - lat1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

def parse_candidates(raw):
    if not raw or raw.strip() in ('', '()'): return []
    cands = []
    for token in raw.strip().strip('()').split('),('):
        token = token.strip('()')
        parts = token.split(',')
        if len(parts) >= 2:
            try: cands.append((float(parts[0]), float(parts[1])))
            except ValueError: pass
    return cands

# ── Load PL data ──
pl_by_ts = {}
with open(INPUT_CSV, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter=';'):
        ts = row['timestamp'].strip().rstrip('.0').rstrip('.')
        pl_by_ts[ts] = float(row['protection_level'])

# ── Load CMM outputs ──
def load_output(path, tid_filter=None):
    data = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            tid = row.get('id','').strip()
            if tid_filter and tid != tid_filter: continue
            og = parse_point(row.get('ogeom',''))
            pg = parse_point(row.get('pgeom',''))
            if og is None or pg is None: continue
            ts = row['timestamp'].strip().rstrip('.0').rstrip('.')
            data.append({
                'tid': tid, 'ts': ts,
                'ogeom': og, 'pgeom': pg,
                'err': haversine_m(*og, *pg),
                'tw': float(row.get('trustworthiness', '0')),
                'ep': float(row.get('ep', '0')),
                'cpath': row.get('cpath', ''),
                'candidates': parse_candidates(row.get('candidates', '')),
            })
    return data

# Load mult=5 (PHMI active) and baseline (no PHMI)
print("Loading mult=5 data...")
phmi_data = load_output(BASE / 'dataset-hainan-06/mr/multiplier_sweep/cmm_mult05.csv')
print("Loading baseline (no PHMI) data...")
base_data = load_output(BASE / 'dataset-hainan-06/mr/multitraj_sweep/cmm_all_lag000.csv')

# Merge by (tid, ts)
base_by_key = {(r['tid'], r['ts']): r for r in base_data}

# Add PHMI PL info
phmi_mult = 5
for r in phmi_data:
    ts = r['ts']
    pl_deg = pl_by_ts.get(ts, 0)
    r['eff_pl_m'] = pl_deg * METERS_PER_DEG * phmi_mult
    cands = r['candidates']
    inside = sum(1 for cx, cy in cands if haversine_m(*r['ogeom'], cx, cy) <= r['eff_pl_m'])
    r['n_cand'] = len(cands)
    r['n_inside_pl'] = inside
    r['all_outside'] = (inside == 0 and len(cands) > 0)
    r['all_inside'] = (inside == len(cands) and len(cands) > 0)

# ══════════════════════════════════════════════════════════════════════════════
# 1. ECE Bin Distribution
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"  1. ECE BIN DISTRIBUTION — PHMI mult=5 vs Baseline (≤{THRESHOLD_M:.0f}m)")
print(f"{'=' * 80}")

for label, data, src_label in [('PHMI(mult=5)', phmi_data, 'phmi'), ('Baseline(no PHMI)', base_data, 'base')]:
    print(f"\n  {label}:")
    trusts = [r['tw'] for r in data]
    errors = [r['err'] for r in data]
    labels = [1 if e <= THRESHOLD_M else 0 for e in errors]
    n = len(data)

    ece_total = 0.0
    print(f"  {'Bin':>6s}  {'Range':>8s}  {'n':>7s}  {'mean_tw':>8s}  {'acc':>7s}  {'gap':>7s}")
    print(f"  {'-' * 6}  {'-' * 8}  {'-' * 7}  {'-' * 8}  {'-' * 7}  {'-' * 7}")
    for i in range(10):
        lo, hi = i/10, (i+1)/10
        if i == 9:
            idxs = [j for j in range(n) if lo <= trusts[j] <= hi]
        else:
            idxs = [j for j in range(n) if lo <= trusts[j] < hi]
        nb = len(idxs)
        if nb == 0: continue
        mc = sum(trusts[j] for j in idxs) / nb
        acc = sum(labels[j] for j in idxs) / nb
        gap = abs(mc - acc)
        ece_total += nb / n * gap
        bar = '▓' * min(50, int(nb / n * 100 // 2))
        print(f"  [{lo:.1f}-{hi:.1f})  {nb:7d}  {mc:8.4f}  {acc:7.4f}  {gap:7.4f}  {bar}")
    print(f"  ECE = {ece_total:.4f}  n = {n}  correct = {sum(labels)/n*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Traj21 First 100 Epochs
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"  2. TRAJ21 FIRST 100 EPOCHS — PHMI mult=5 trustworthiness detail")
print(f"{'=' * 80}")

traj21_phmi = [r for r in phmi_data if r['tid'] == '21']
traj21_base = [r for r in base_data if r['tid'] == '21']

print(f"\n  Traj21 total: {len(traj21_phmi)} epochs")
print(f"  All candidates outside PHMI-PL: {sum(1 for r in traj21_phmi if r['all_outside'])} epochs ({sum(1 for r in traj21_phmi if r['all_outside'])/len(traj21_phmi)*100:.1f}%)")
print(f"  PHMI-PL threshold: mean {sum(r['eff_pl_m'] for r in traj21_phmi)/len(traj21_phmi):.0f}m")

# Print first 100 epochs detail
print(f"\n  {'Seq':>4s}  {'ts':>12s}  {'err(m)':>7s}  {'TW(phmi)':>10s}  {'TW(base)':>10s}  {'PL(m)':>6s}  {'cands':>6s}  {'in_PL':>6s}  {'all_out':>8s}  {'cpath':>10s}")
print(f"  {'-' * 4}  {'-' * 12}  {'-' * 7}  {'-' * 10}  {'-' * 10}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 8}  {'-' * 10}")
for i, (r_phmi, r_base) in enumerate(zip(traj21_phmi[:100], traj21_base[:100])):
    print(f"  {i:4d}  {r_phmi['ts']:>12s}  {r_phmi['err']:7.1f}  {r_phmi['tw']:10.6f}  {r_base['tw']:10.6f}  "
          f"{r_phmi['eff_pl_m']:6.1f}  {r_phmi['n_cand']:6d}  {r_phmi['n_inside_pl']:6d}  "
          f"{str(r_phmi['all_outside']):>8s}  {r_phmi['cpath'][:10]:>10s}")

# Summary stats
tw_phmi_vals = [r['tw'] for r in traj21_phmi[:100]]
tw_base_vals = []  # need to match
for r in traj21_phmi[:100]:
    key = (r['tid'], r['ts'])
    if key in base_by_key:
        tw_base_vals.append(base_by_key[key]['tw'])

tw_diff = [p - b if b > 0 else 0 for p, b in zip(tw_phmi_vals, tw_base_vals)]
err_vals = [r['err'] for r in traj21_phmi[:100]]
all_out_epochs = [i for i, r in enumerate(traj21_phmi[:100]) if r['all_outside']]

print(f"\n  Summary (first 100 epochs):")
print(f"    Mean TW (PHMI):  {sum(tw_phmi_vals)/len(tw_phmi_vals):.4f}")
print(f"    Mean TW (base):  {sum(tw_base_vals)/len(tw_base_vals):.4f}")
print(f"    Mean ΔTW:        {sum(tw_diff)/len(tw_diff):.6f}")
print(f"    Epochs with all_outside: {len(all_out_epochs)} → {all_out_epochs}")
print(f"    Mean err:        {sum(err_vals)/len(err_vals):.1f}m")
print(f"    Correct ≤5m:     {sum(1 for e in err_vals if e<=5)/len(err_vals)*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 3. All-outside epoch analysis (all trajectories)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"  3. ALL-OUTSIDE-PL EPOCH ANALYSIS — can PHMI trigger warnings?")
print(f"{'=' * 80}")

all_out_phmi = [r for r in phmi_data if r['all_outside']]
all_in_phmi = [r for r in phmi_data if r['all_inside']]

print(f"\n  Total epochs: {len(phmi_data)}")
print(f"  All candidates OUTSIDE PHMI-PL: {len(all_out_phmi)} ({len(all_out_phmi)/len(phmi_data)*100:.1f}%)")
print(f"  All candidates INSIDE PHMI-PL:  {len(all_in_phmi)} ({len(all_in_phmi)/len(phmi_data)*100:.1f}%)")

if all_out_phmi:
    tw_out = [r['tw'] for r in all_out_phmi]
    err_out = [r['err'] for r in all_out_phmi]
    tw_in = [r['tw'] for r in all_in_phmi]
    err_in = [r['err'] for r in all_in_phmi]

    print(f"\n  Metric           |  All-OUTSIDE-PL  |  All-INSIDE-PL  |  Delta")
    print(f"  {'─' * 17}  |  {'─' * 15}  |  {'─' * 15}  |  {'─' * 10}")
    print(f"  Mean TW          |  {sum(tw_out)/len(tw_out):15.6f}  |  {sum(tw_in)/len(tw_in):15.6f}  |  {(sum(tw_out)/len(tw_out)-sum(tw_in)/len(tw_in)):+10.6f}")
    print(f"  Mean err(m)      |  {sum(err_out)/len(err_out):15.1f}  |  {sum(err_in)/len(err_in):15.1f}  |  {(sum(err_out)/len(err_out)-sum(err_in)/len(err_in)):+10.1f}")
    print(f"  Median err(m)    |  {sorted(err_out)[len(err_out)//2]:15.1f}  |  {sorted(err_in)[len(err_in)//2]:15.1f}  |  {(sorted(err_out)[len(err_out)//2]-sorted(err_in)[len(err_in)//2]):+10.1f}")
    print(f"  Correct ≤5m      |  {sum(1 for e in err_out if e<=5)/len(err_out)*100:14.1f}%  |  {sum(1 for e in err_in if e<=5)/len(err_in)*100:14.1f}%")

    # TW bin distribution for all-outside epochs
    print(f"\n  Trustworthiness distribution for ALL-OUTSIDE-PL epochs ({len(all_out_phmi)} epochs):")
    for i in range(10):
        lo, hi = i/10, (i+1)/10
        cnt = sum(1 for t in tw_out if (lo <= t <= hi if i == 9 else lo <= t < hi))
        if cnt > 0:
            bar = '▓' * min(60, cnt)
            print(f"    [{lo:.1f}-{hi:.1f}): {cnt:5d}  {bar}")

    # Per-trajectory breakdown
    print(f"\n  Per-trajectory all-outside epochs:")
    by_traj = defaultdict(list)
    for r in all_out_phmi:
        by_traj[r['tid']].append(r)
    for tid in sorted(by_traj, key=int):
        rs = by_traj[tid]
        mean_tw = sum(r['tw'] for r in rs) / len(rs)
        mean_err = sum(r['err'] for r in rs) / len(rs)
        all_err = [r['err'] for r in rs]
        print(f"    Traj {tid}: {len(rs):4d} epochs  "
              f"mean_TW={mean_tw:.4f}  mean_err={mean_err:.1f}m  "
              f"correct={sum(1 for e in all_err if e<=5)/len(all_err)*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Warning trigger effectiveness
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"  4. WARNING TRIGGER EFFECTIVENESS")
print(f"{'=' * 80}")

print(f"""
  Scenario: "all candidates outside PHMI-PL → trust should be low → trigger alarm"

  For real-time quality monitoring, set a trustworthiness threshold θ.
  When TW < θ, flag the match for manual inspection.

  Analysis: for all-outside-PL epochs, what fraction would be flagged at various θ?
""")

for theta in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    flagged = sum(1 for r in all_out_phmi if r['tw'] < theta)
    pct = flagged / len(all_out_phmi) * 100 if all_out_phmi else 0
    print(f"    θ={theta:.1f}: {flagged:5d}/{len(all_out_phmi)} all-outside epochs flagged ({pct:5.1f}%)")

# What threshold would flag 90% of all-outside epochs?
tw_sorted = sorted([r['tw'] for r in all_out_phmi])
idx_90 = max(0, int(len(tw_sorted) * 0.9) - 1)
print(f"\n  θ ≥ {tw_sorted[idx_90]:.4f} would flag 90% of all-outside-PL epochs")


if __name__ == '__main__':
    pass
