#!/usr/bin/env python3
"""
PHMI multiplier sweep: protection_level_multiplier ∈ [5, 15].
For each multiplier run CMM with PHMI=1e-5 on aggregated input.
Report PL coverage, ECE, and find optimal multiplier.

Usage:
  python3 tests/python/exp3_multiplier_sweep.py
"""

import csv, math, re, subprocess, tempfile, os
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import defaultdict


BASE = Path(__file__).resolve().parents[2]
INPUT_CSV = BASE / 'dataset-hainan-06/cmm_input_points.csv'
XML_TEMPLATE = BASE / 'input/config/cmm_config_omp.xml'
CMM_BIN = BASE / 'build/cmm'
OUT_DIR = BASE / 'dataset-hainan-06/mr/multiplier_sweep'

MULTIPLIERS = list(range(5, 16))  # 5, 6, ..., 15
THRESHOLD_M = 5.0
PHMI = 0.00001
METERS_PER_DEG = 111_000.0

PP_RE = re.compile(r'POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)', re.I)

def parse_point(wkt):
    m = PP_RE.search(wkt or '')
    return (float(m.group(1)), float(m.group(2))) if m else None

def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

def compute_ece(confidences, labels, n_bins=10):
    N = len(confidences); ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if i == n_bins - 1:
            idxs = [j for j in range(N) if lo <= confidences[j] <= hi]
        else:
            idxs = [j for j in range(N) if lo <= confidences[j] < hi]
        n_b = len(idxs)
        if n_b == 0: continue
        mc = sum(confidences[j] for j in idxs) / n_b
        acc = sum(labels[j] for j in idxs) / n_b
        ece += n_b / N * abs(mc - acc)
    return ece

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
def load_pl_data():
    pl_by_ts = {}
    with open(INPUT_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            ts = row['timestamp'].strip().rstrip('.0').rstrip('.')
            pl_by_ts[ts] = float(row.get('protection_level', '0'))
    return pl_by_ts

# ── Run CMM for one multiplier ──
def run_cmm_with_multiplier(mult, out_file):
    tree = ET.parse(XML_TEMPLATE)
    root = tree.getroot()
    root.find('output').find('file').text = out_file
    root.find('input').find('gps').find('file').text = str(INPUT_CSV)
    params = root.find('parameters')
    for tag, val in [('phmi_pl_multiplier', mult), ('phmi', PHMI),
                     ('lag_steps', 0), ('protection_level_multiplier', 10)]:
        el = params.find(tag)
        if el is None: el = ET.SubElement(params, tag)
        el.text = str(val)
    tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='wb')
    tree.write(tmp, encoding='utf-8', xml_declaration=True)
    tmp.close()
    try:
        subprocess.run([str(CMM_BIN), tmp.name], check=True,
                      capture_output=True, cwd=str(BASE), timeout=300)
    except Exception as e:
        print(f"    FAILED: {e}")
        os.unlink(tmp.name)
        return False
    os.unlink(tmp.name)
    return True

# ── Main ──
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl_by_ts = load_pl_data()

    print(f"{'=' * 90}")
    print(f"  PHMI Multiplier Sweep: protection_level_multiplier ∈ [{MULTIPLIERS[0]}, {MULTIPLIERS[-1]}]")
    print(f"  PHMI = {PHMI}  |  All 7 trajectories  |  Threshold ≤{THRESHOLD_M:.0f}m")
    print(f"{'=' * 90}")

    results = []  # list of dicts

    for mult in MULTIPLIERS:
        out_file = str(OUT_DIR / f'cmm_mult{mult:02d}.csv')
        print(f"\n  mult={mult:2d} (effPL≈{4e-5*mult*METERS_PER_DEG:.0f}m) ...", end='', flush=True)
        if not Path(out_file).exists():
            if not run_cmm_with_multiplier(mult, out_file):
                continue
        print(" ok", end='', flush=True)

        # ── Parse output ──
        traj = defaultdict(lambda: {
            'errors': [], 'trusts': [], 'pl_in': 0, 'pl_out': 0, 'pl_total': 0
        })
        with open(out_file, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter=';'):
                tid = row.get('id', '').strip()
                og = parse_point(row.get('ogeom', ''))
                pg = parse_point(row.get('pgeom', ''))
                if og is None or pg is None: continue
                err = haversine_m(*og, *pg)
                try:
                    tw = float(row.get('trustworthiness', '0'))
                except: continue
                traj[tid]['errors'].append(err)
                traj[tid]['trusts'].append(tw)

                ts = row['timestamp'].strip().rstrip('.0').rstrip('.')
                pl_deg = pl_by_ts.get(ts, 0)
                eff_pl_m = pl_deg * METERS_PER_DEG * mult
                cands = parse_candidates(row.get('candidates', ''))
                for cx, cy in cands:
                    d = haversine_m(*og, cx, cy)
                    if d <= eff_pl_m:
                        traj[tid]['pl_in'] += 1
                    else:
                        traj[tid]['pl_out'] += 1
                    traj[tid]['pl_total'] += 1

        # ── Compute per-trajectory stats ──
        row_data = {'mult': mult, 'eff_pl_m': round(4.7e-5 * mult * METERS_PER_DEG)}
        all_eces, all_fracs = [], []
        for tid in sorted(traj, key=int):
            d = traj[tid]
            labels = [1 if e <= THRESHOLD_M else 0 for e in d['errors']]
            ece = compute_ece(d['trusts'], labels)
            total = d['pl_total']
            frac_in = d['pl_in'] / total * 100 if total > 0 else 0
            pct_correct = sum(labels) / len(labels) * 100 if labels else 0
            row_data[f'traj{tid}_ece'] = ece
            row_data[f'traj{tid}_frac'] = frac_in
            row_data[f'traj{tid}_correct'] = pct_correct
            all_eces.append(ece)
            all_fracs.append(frac_in)
        row_data['mean_ece'] = sum(all_eces) / len(all_eces) if all_eces else 0
        row_data['mean_frac_in'] = sum(all_fracs) / len(all_fracs) if all_fracs else 0
        results.append(row_data)

        print(f"  frac_in={row_data['mean_frac_in']:.1f}%  ECE={row_data['mean_ece']:.4f}")

    # ── Summary table ──
    print(f"\n{'=' * 90}")
    print(f"  RESULTS: PL Coverage and ECE by multiplier")
    print(f"{'=' * 90}")
    traj_ids = ['11', '12', '13', '14', '21', '22', '23']
    header = f"  {'mult':>4s} {'effPL':>7s}"
    for tid in traj_ids: header += f"  {'T' + tid + ' frac_in':>12s}"
    header += f"  {'MEAN frac_in':>12s}  {'MEAN ECE':>9s}  {'ΔECE':>9s}"
    print(header)
    print(f"  {'-' * 4} {'-' * 7}" + f"  {'-' * 12}" * len(traj_ids) + f"  {'-' * 12}  {'-' * 9}  {'-' * 9}")

    base_ece = results[0]['mean_ece'] if results else 0.5  # mult=5 as baseline
    for r in results:
        line = f"  {r['mult']:4d} {r['eff_pl_m']:4.0f}m"
        for tid in traj_ids:
            line += f"  {r.get(f'traj{tid}_frac', 0):11.1f}%"
        line += f"  {r['mean_frac_in']:11.1f}%  {r['mean_ece']:9.4f}"
        delta = r['mean_ece'] - results[0]['mean_ece']
        line += f"  {delta:+9.4f}"
        print(line)

    # ── Find optimal ──
    best = min(results, key=lambda r: r['mean_ece'])
    print(f"\n  ★ Optimal multiplier = {best['mult']}")
    print(f"    Effective PL ≈ {best['eff_pl_m']:.0f}m")
    print(f"    Mean frac_in_PL = {best['mean_frac_in']:.1f}%")
    print(f"    Mean ECE = {best['mean_ece']:.4f}")
    print(f"    ECE reduction vs mult=5: {(results[0]['mean_ece'] - best['mean_ece']):.4f} "
          f"({(results[0]['mean_ece'] - best['mean_ece']) / results[0]['mean_ece'] * 100:.1f}%)")


if __name__ == '__main__':
    main()
