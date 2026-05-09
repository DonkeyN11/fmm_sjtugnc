#!/usr/bin/env python3
"""
Experiment 3: PHMI-enhanced emission + PL coverage analysis.

Pipeline:
  1. Run CMM with PHMI=1e-5 on all 7 trajectories (single aggregated input).
  2. For each epoch, extract the candidate list from CMM output and compute:
       - fraction of candidates within GNSS Protection Level (PL)
       - mean Euclidean distance from GPS to candidates
  3. Run lag sweep with PHMI enabled; compare ECE against baseline (no PHMI).
  4. Output: PL coverage statistics + ECE comparison table.

Usage:
  python3 tests/python/exp3_phmi_analysis.py
"""

import csv, math, re, subprocess, tempfile, os
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parents[2]
INPUT_CSV = BASE / 'dataset-hainan-06/cmm_input_points.csv'
XML_TEMPLATE = BASE / 'input/config/cmm_config_omp.xml'
CMM_BIN = BASE / 'build/cmm'
LAGS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]
THRESHOLD_M = 5.0
OUT_DIR = BASE / 'dataset-hainan-06/mr/phmi_analysis'


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry & Metrics
# ═══════════════════════════════════════════════════════════════════════════════

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
    N = len(confidences)
    ece = 0.0
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

def compute_brier_logloss(confidences, labels):
    brier = sum((c - l) ** 2 for c, l in zip(confidences, labels)) / len(confidences)
    eps = 1e-15
    logloss = -sum(l * math.log(max(c, eps)) + (1 - l) * math.log(max(1 - c, eps))
                   for c, l in zip(confidences, labels)) / len(confidences)
    return brier, logloss


# ═══════════════════════════════════════════════════════════════════════════════
# CMM Runner
# ═══════════════════════════════════════════════════════════════════════════════

def build_xml(xml_template, out_file, lag, phmi=0.00001):
    """Create temp XML with given lag_steps and PHMI, pointing at aggregated CSV."""
    tree = ET.parse(xml_template)
    root = tree.getroot()
    root.find('output').find('file').text = out_file
    params = root.find('parameters')
    for tag in ['lag_steps', 'phmi']:
        el = params.find(tag)
        if el is None:
            el = ET.SubElement(params, tag)
    params.find('lag_steps').text = str(lag)
    params.find('phmi').text = str(phmi)
    root.find('input').find('gps').find('file').text = str(INPUT_CSV)
    tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='wb')
    tree.write(tmp, encoding='utf-8', xml_declaration=True)
    tmp.close()
    return Path(tmp.name)


def run_cmm_lag_sweep(lags, phmi=0.00001):
    """Run CMM once per lag, return {lag: csv_path, ...}"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for lag in lags:
        out_file = str(OUT_DIR / f'cmm_phmi_lag{lag:03d}.csv')
        if Path(out_file).exists():
            outputs[lag] = out_file
            continue
        tmp_xml = build_xml(XML_TEMPLATE, out_file, lag, phmi)
        print(f"    lag={lag:3d} ...", end='', flush=True)
        try:
            subprocess.run([str(CMM_BIN), str(tmp_xml)], check=True,
                          capture_output=True, cwd=str(BASE), timeout=300)
        except subprocess.CalledProcessError as e:
            print(f" FAILED")
            os.unlink(tmp_xml)
            continue
        os.unlink(tmp_xml)
        outputs[lag] = out_file
        print(" ok")
    return outputs


# ═══════════════════════════════════════════════════════════════════════════════
# PL Coverage Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def parse_candidates(raw):
    """Parse the candidates column: (x,y,ep),(x,y,ep),..."""
    if not raw or raw.strip() == '()' or raw.strip() == '':
        return []
    cands = []
    for token in raw.strip().strip('()').split('),('):
        token = token.strip('()')
        parts = token.split(',')
        if len(parts) >= 3:
            try:
                cands.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                pass
    return cands


def analyze_pl_coverage(cmm_output_csv, all_pl_data):
    """
    Compute per-epoch PL coverage statistics.
    all_pl_data: {idx: protection_level_in_degrees} from input CSV.

    Returns dict: {traj_id: {
        'n_total': int,
        'n_all_inside_pl': int,     # epochs where ALL candidates within PL
        'n_none_inside_pl': int,    # epochs where NO candidate within PL
        'n_partial_inside_pl': int, # mixed
        'frac_inside_mean': float,  # mean fraction of candidates within PL
        'cand_dist_mean': float,    # mean distance from GPS to candidates (m)
    }}
    """
    traj_data = defaultdict(lambda: {
        'total': 0, 'all_inside': 0, 'none_inside': 0, 'partial': 0,
        'frac_inside': [], 'cand_dists': [], 'gps_pts': [], 'pl_vals': [],
    })

    with open(cmm_output_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            tid = row.get('id', '').strip()
            ogeom = parse_point(row.get('ogeom', ''))
            if ogeom is None:
                continue
            gx, gy = ogeom
            candidates = parse_candidates(row.get('candidates', ''))

            # Get protection level for this epoch
            # We need to match by timestamp to the input data
            # Use timestamp as key
            ts = row.get('timestamp', '').strip().rstrip('.0').rstrip('.')
            pl = all_pl_data.get(ts, 0.0)  # PL in meters (after conversion)

            if not candidates:
                continue

            n_inside = 0
            for cx, cy, _ in candidates:
                dist_m = haversine_m(gx, gy, cx, cy)
                traj_data[tid]['cand_dists'].append(dist_m)
                if dist_m <= pl:
                    n_inside += 1

            frac = n_inside / len(candidates)
            traj_data[tid]['frac_inside'].append(frac)
            traj_data[tid]['total'] += 1
            traj_data[tid]['gps_pts'].append((gx, gy))
            traj_data[tid]['pl_vals'].append(pl)

            if n_inside == len(candidates):
                traj_data[tid]['all_inside'] += 1
            elif n_inside == 0:
                traj_data[tid]['none_inside'] += 1
            else:
                traj_data[tid]['partial'] += 1

    summary = {}
    for tid, d in traj_data.items():
        n = d['total']
        summary[tid] = {
            'n_total': n,
            'all_inside': d['all_inside'],
            'none_inside': d['none_inside'],
            'partial': d['partial'],
            'frac_all_inside': d['all_inside'] / n * 100 if n > 0 else 0,
            'frac_none_inside': d['none_inside'] / n * 100 if n > 0 else 0,
            'frac_partial': d['partial'] / n * 100 if n > 0 else 0,
            'frac_inside_mean': sum(d['frac_inside']) / len(d['frac_inside']) * 100 if d['frac_inside'] else 0,
            'cand_dist_mean': sum(d['cand_dists']) / len(d['cand_dists']) if d['cand_dists'] else 0,
            'cand_dist_median': sorted(d['cand_dists'])[len(d['cand_dists']) // 2] if d['cand_dists'] else 0,
            'pl_mean_deg': sum(d['pl_vals']) / len(d['pl_vals']) if d['pl_vals'] else 0,
        }
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Load input data (for PL values indexed by timestamp)
# ═══════════════════════════════════════════════════════════════════════════════

def load_input_pl_data(input_csv, pl_multiplier=100):
    """Return dict: timestamp → effective_protection_level (in METERS).

    Input CSV has protection_level in degrees (e.g., 4e-5° ≈ 4.4m).
    Convert to meters and scale by protection_level_multiplier.
    """
    METERS_PER_DEG = 111_000.0  # rough, latitude-dependent; use this average
    pl_data = {}
    with open(input_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            # Normalize timestamp: input has "1750306259.0", CMM output has "1750306259"
            ts = row.get('timestamp', '').strip().rstrip('.0').rstrip('.')
            try:
                pl_deg = float(row.get('protection_level', '0'))
            except (ValueError, TypeError):
                pl_deg = 0.0
            # Convert to meters and apply multiplier (same as C++ effective_pl)
            pl_data[ts] = pl_deg * METERS_PER_DEG * pl_multiplier
    return pl_data


# ═══════════════════════════════════════════════════════════════════════════════
# ECE comparison with/without PHMI
# ═══════════════════════════════════════════════════════════════════════════════

def compare_ece(phmi_results, baseline_dir, lags):
    """
    phmi_results: {lag: csv_path}
    baseline_dir: path to multitraj_sweep directory (no PHMI)
    Returns: dict with per-trajectory, per-lag ECE comparison
    """
    baseline_dir = Path(baseline_dir)

    def load_traj_data(csv_path):
        """Return {tid: {'trusts': [], 'eps': [], 'errors': []}}"""
        data = defaultdict(lambda: {'trusts': [], 'eps': [], 'errors': []})
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter=';'):
                tid = row.get('id', '').strip()
                og = parse_point(row.get('ogeom', ''))
                pg = parse_point(row.get('pgeom', ''))
                if og is None or pg is None:
                    continue
                err = haversine_m(*og, *pg)
                try:
                    tw = float(row.get('trustworthiness', '0'))
                    ep = float(row.get('ep', '0'))
                except (ValueError, TypeError):
                    continue
                data[tid]['trusts'].append(tw)
                data[tid]['eps'].append(ep)
                data[tid]['errors'].append(err)
        return data

    comparison = {}  # {lag: {tid: {'phmi_ece': ..., 'base_ece': ...}}}

    for lag in lags:
        # PHMI data
        phmi_csv = phmi_results.get(lag)
        if not phmi_csv or not Path(phmi_csv).exists():
            continue
        phmi_data = load_traj_data(phmi_csv)

        # Baseline data (no PHMI, from previous sweep)
        base_csv = baseline_dir / f'cmm_all_lag{lag:03d}.csv'
        if not base_csv.exists():
            continue
        base_data = load_traj_data(base_csv)

        comparison[lag] = {}
        for tid in phmi_data:
            if tid not in base_data:
                continue
            labels_phmi = [1 if e <= THRESHOLD_M else 0 for e in phmi_data[tid]['errors']]
            labels_base = [1 if e <= THRESHOLD_M else 0 for e in base_data[tid]['errors']]
            n = len(phmi_data[tid]['errors'])

            ece_phmi = compute_ece(phmi_data[tid]['trusts'], labels_phmi)
            ece_base = compute_ece(base_data[tid]['trusts'], labels_base)
            brier_phmi, ll_phmi = compute_brier_logloss(phmi_data[tid]['trusts'], labels_phmi)
            brier_base, ll_base = compute_brier_logloss(base_data[tid]['trusts'], labels_base)

            n_correct = sum(labels_phmi)
            comparison[lag][tid] = {
                'ece_phmi': ece_phmi, 'ece_base': ece_base,
                'brier_phmi': brier_phmi, 'brier_base': brier_base,
                'll_phmi': ll_phmi, 'll_base': ll_base,
                'n': n, 'n_correct': n_correct,
            }

    return comparison


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 78}")
    print(f"  Experiment 3: PHMI Emission + PL Coverage Analysis")
    print(f"{'=' * 78}")

    # ── Step 1: Run CMM with PHMI=1e-5, lag sweep ──
    print(f"\n  [Step 1] Running CMM with PHMI=1e-5, {len(LAGS)} lags...")
    phmi_outputs = run_cmm_lag_sweep(LAGS, phmi=0.00001)

    # ── Step 2: PL coverage analysis ──
    print(f"\n  [Step 2] Analyzing GNSS Protection Level coverage...")
    pl_data = load_input_pl_data(INPUT_CSV, pl_multiplier=100)

    # Use lag=0 output for PL coverage analysis (coverage doesn't depend on lag)
    lag0_csv = phmi_outputs.get(0)
    if not lag0_csv:
        print("  ERROR: No lag=0 output. Cannot analyze.")
        return

    coverage = analyze_pl_coverage(lag0_csv, pl_data)

    traj_ids = sorted(coverage.keys(), key=int)
    print(f"\n  ── PL Coverage Statistics ──")
    print(f"  {'Traj':>5s}  {'Pts':>6s}  {'All in PL%':>10s}  "
          f"{'None in PL%':>11s}  {'Partial%':>9s}  "
          f"{'Mean frac in':>12s}  {'Mean cand dist':>14s}  {'Median dist':>11s}")
    print(f"  {'-' * 5}  {'-' * 6}  {'-' * 10}  {'-' * 11}  {'-' * 9}  "
          f"{'-' * 12}  {'-' * 14}  {'-' * 11}")

    totals = {'n': 0, 'all_in': 0, 'none_in': 0, 'partial': 0}
    for tid in traj_ids:
        c = coverage[tid]
        print(f"  {tid:>5s}  {c['n_total']:6d}  {c['frac_all_inside']:9.1f}%  "
              f"{c['frac_none_inside']:10.1f}%  {c['frac_partial']:8.1f}%  "
              f"{c['frac_inside_mean']:11.1f}%  {c['cand_dist_mean']:13.1f}m  {c['cand_dist_median']:10.1f}m")
        totals['n'] += c['n_total']
        totals['all_in'] += c['all_inside']
        totals['none_in'] += c['none_inside']
        totals['partial'] += c['partial']

    n_tot = totals['n']
    print(f"  {'─' * 78}")
    print(f"  ALL    {n_tot:6d}  {totals['all_in']/n_tot*100:9.1f}%  "
          f"{totals['none_in']/n_tot*100:10.1f}%  {totals['partial']/n_tot*100:8.1f}%")

    # ── Step 3: Scenario classification ──
    print(f"\n  ── Scenario Classification ──")
    print(f"  'All in PL'  = GNSS PL contains all road candidates → map seems correct")
    print(f"  'None in PL' = no candidate within PL → possible map error / off-road")
    print(f"  'Partial'   = mixed → PL may be too tight or road deviates locally")
    print()

    # Check correlation with matching quality
    print(f"  Relationship with matching quality:")
    for tid in traj_ids:
        c = coverage[tid]
        # Load ECE from phmi lag=0
        if 0 in phmi_outputs:
            trusts, errors = [], []
            with open(phmi_outputs[0], newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f, delimiter=';'):
                    if row.get('id', '').strip() != tid:
                        continue
                    og = parse_point(row.get('ogeom', ''))
                    pg = parse_point(row.get('pgeom', ''))
                    if og is None or pg is None: continue
                    errors.append(haversine_m(*og, *pg))
                    try: trusts.append(float(row.get('trustworthiness', '0')))
                    except: continue
            ece_val = compute_ece(trusts, [1 if e <= THRESHOLD_M else 0 for e in errors])
            pct_correct = sum(1 for e in errors if e <= THRESHOLD_M) / len(errors) * 100
            print(f"    Traj {tid}: all_in={c['frac_all_inside']:.0f}%  "
                  f"none_in={c['frac_none_inside']:.0f}%  "
                  f"ECE(phmi)={ece_val:.4f}  correct={pct_correct:.1f}%")

    # ── Step 4: ECE comparison PHMI vs no-PHMI ──
    print(f"\n  [Step 4] ECE comparison: PHMI=1e-5 vs PHMI=0")
    baseline_dir = BASE / 'dataset-hainan-06/mr/multitraj_sweep'
    comp = compare_ece(phmi_outputs, baseline_dir, LAGS)

    if comp:
        print(f"\n  ── Per-Trajectory ECE (lag=0) ──")
        print(f"  {'Traj':>5s}  {'ECE (no PHMI)':>14s}  {'ECE (PHMI=1e-5)':>16s}  {'Delta':>10s}  {'Improvement':>14s}  {'%Correct':>10s}")
        print(f"  {'-' * 5}  {'-' * 14}  {'-' * 16}  {'-' * 10}  {'-' * 14}  {'-' * 10}")
        for tid in traj_ids:
            if 0 in comp and tid in comp[0]:
                r = comp[0][tid]
                delta = r['ece_phmi'] - r['ece_base']
                impr = -delta / r['ece_base'] * 100 if r['ece_base'] > 0 else 0
                print(f"  {tid:>5s}  {r['ece_base']:14.4f}  {r['ece_phmi']:16.4f}  "
                      f"{delta:+10.4f}  {impr:+13.1f}%  {r['n_correct']/r['n']*100:9.1f}%")

        # Per-lag mean ECE comparison
        print(f"\n  ── Mean ECE across trajectories ──")
        print(f"  {'lag':>5s}  {'ECE no PHMI':>13s}  {'ECE PHMI':>13s}  {'Delta':>10s}  {'Impr.%':>8s}")
        print(f"  {'-' * 5}  {'-' * 13}  {'-' * 13}  {'-' * 10}  {'-' * 8}")
        for lag in LAGS:
            if lag not in comp:
                continue
            eces_no = [comp[lag][tid]['ece_base'] for tid in comp[lag]]
            eces_phmi = [comp[lag][tid]['ece_phmi'] for tid in comp[lag]]
            if not eces_no:
                continue
            mean_no = sum(eces_no) / len(eces_no)
            mean_phmi = sum(eces_phmi) / len(eces_phmi)
            delta = mean_phmi - mean_no
            impr = -delta / mean_no * 100 if mean_no > 0 else 0
            print(f"  {lag:5d}  {mean_no:13.4f}  {mean_phmi:13.4f}  {delta:+10.4f}  {impr:+7.1f}%")


if __name__ == '__main__':
    main()
