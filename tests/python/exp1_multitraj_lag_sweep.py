#!/usr/bin/env python3
"""
Multi-trajectory lag-steps sweep across trajectories 11,12,13,14,21,22,23.

Points the CMM config at the aggregated input CSV (cmm_input_points.csv)
and processes all trajectories in a single CMM run per lag value.
Output is grouped by trajectory ID for per-trajectory and aggregate analysis.

Usage:
  python3 tests/python/exp1_multitraj_lag_sweep.py
"""

import csv, math, re, subprocess, tempfile, os
from pathlib import Path
from xml.etree import ElementTree as ET


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parents[2]
INPUT_CSV = BASE / 'dataset-hainan-06/cmm_input_points.csv'
XML_TEMPLATE = BASE / 'input/config/cmm_config_omp.xml'
CMM_BIN = BASE / 'build/cmm'

LAGS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]
THRESHOLD_M = 5.0
OUT_DIR = BASE / 'dataset-hainan-06/mr/multitraj_sweep'


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
    per_bin = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = [(lo <= c <= hi) if i == n_bins - 1 else (lo <= c < hi)
                for c in confidences]
        idxs = [j for j, m in enumerate(mask) if m]
        n_b = len(idxs)
        if n_b == 0:
            per_bin.append({'n': 0, 'mean_conf': float('nan'), 'accuracy': float('nan')})
            continue
        per_bin.append({
            'n': n_b,
            'mean_conf': sum(confidences[j] for j in idxs) / n_b,
            'accuracy': sum(labels[j] for j in idxs) / n_b,
        })
    N = len(confidences)
    ece = sum(b['n'] / N * abs(b['mean_conf'] - b['accuracy'])
              for b in per_bin if b['n'] > 0)
    return ece, per_bin

def compute_metrics(confidences, labels):
    ece, bins = compute_ece(confidences, labels)
    brier = sum((c - l) ** 2 for c, l in zip(confidences, labels)) / len(confidences)
    eps = 1e-15
    logloss = -sum(l * math.log(max(c, eps)) + (1 - l) * math.log(max(1 - c, eps))
                   for c, l in zip(confidences, labels)) / len(confidences)
    return {'ece': ece, 'brier': brier, 'logloss': logloss, 'bins': bins}


# ═══════════════════════════════════════════════════════════════════════════════
# XML config builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_xml(xml_template, out_file, lag):
    """Create temp XML pointing at aggregated input CSV, with given lag_steps."""
    tree = ET.parse(xml_template)
    root = tree.getroot()
    root.find('output').find('file').text = out_file
    params = root.find('parameters')
    lag_el = params.find('lag_steps')
    if lag_el is None:
        lag_el = ET.SubElement(params, 'lag_steps')
    lag_el.text = str(lag)
    # Point GPS at aggregated input
    root.find('input').find('gps').find('file').text = str(INPUT_CSV)
    tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='wb')
    tree.write(tmp, encoding='utf-8', xml_declaration=True)
    tmp.close()
    return Path(tmp.name)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 78}")
    print(f"  Multi-Trajectory Lag-Steps Sweep")
    print(f"  Input: {INPUT_CSV}")
    print(f"  Lags: {LAGS}")
    print(f"  Threshold: ≤{THRESHOLD_M:.0f}m")
    print(f"{'=' * 78}")

    # ── Run CMM for each lag ──
    lag_outputs = {}  # lag → output CSV path

    for lag in LAGS:
        out_file = str(OUT_DIR / f'cmm_all_lag{lag:03d}.csv')
        tmp_xml = build_xml(XML_TEMPLATE, out_file, lag)

        print(f"\n  lag={lag:3d} ...", end='', flush=True)
        try:
            result = subprocess.run(
                [str(CMM_BIN), str(tmp_xml)],
                check=True, capture_output=True, text=True,
                cwd=str(CMM_BIN.parent.parent), timeout=120
            )
        except subprocess.CalledProcessError as e:
            print(f" FAILED: {e.stderr[:120] if e.stderr else str(e)}")
            os.unlink(tmp_xml)
            continue
        except subprocess.TimeoutExpired:
            print(f" TIMEOUT")
            os.unlink(tmp_xml)
            continue

        os.unlink(tmp_xml)
        lag_outputs[lag] = out_file
        # Extract timing info
        for line in result.stderr.split('\n'):
            if 'Time takes' in line or 'matched' in line or 'Trajectories' in line:
                pass  # info level goes to stderr with spdlog
        # Look in stdout for time info
        elapsed = ""
        for line in [result.stderr, result.stdout]:
            if isinstance(line, str):
                for l in line.split('\n'):
                    if 'Time takes' in l:
                        elapsed += l.strip() + " "
        print(f" done  {elapsed}")

    if not lag_outputs:
        print("\n  No successful runs.")
        return

    # ── Parse all outputs, group by trajectory ID ──
    # Structure: all_data[traj_id][lag] = {'trusts': [...], 'eps': [...], 'errors': [...]}
    all_data = {}
    traj_ids = set()
    max_lag = max(lag_outputs.keys()) if lag_outputs else 0

    print(f"\n  Parsing outputs ...", end='', flush=True)
    for lag, path in lag_outputs.items():
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter=';'):
                tid = row.get('id', '').strip()
                traj_ids.add(tid)
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

                if tid not in all_data:
                    all_data[tid] = {}
                if lag not in all_data[tid]:
                    all_data[tid][lag] = {'trusts': [], 'eps': [], 'errors': []}
                all_data[tid][lag]['trusts'].append(tw)
                all_data[tid][lag]['eps'].append(ep)
                all_data[tid][lag]['errors'].append(err)

    traj_ids = sorted(traj_ids, key=int)
    print(f" {len(traj_ids)} trajectories found: {traj_ids}")

    # ── Compute per-trajectory, per-lag metrics ──
    summary = {}  # summary[tid][lag] = metrics dict

    for tid in traj_ids:
        summary[tid] = {}
        for lag in LAGS:
            if lag not in all_data.get(tid, {}):
                continue
            d = all_data[tid][lag]
            labels = [1 if e <= THRESHOLD_M else 0 for e in d['errors']]
            n = len(d['errors'])
            m = compute_metrics(d['trusts'], labels)
            m['n'] = n
            m['n_correct'] = sum(labels)
            m['lag'] = lag
            err_sort = sorted(d['errors'])
            m['error_mean'] = sum(d['errors']) / n if n > 0 else 0
            m['error_median'] = err_sort[n // 2] if n > 0 else 0
            m['error_max'] = err_sort[-1] if n > 0 else 0
            summary[tid][lag] = m

    # ── 1. Per-trajectory, per-lag detail ──
    print(f"\n{'=' * 78}")
    print(f"  CALIBRATION TABLE (trustworthiness ECE, ≤{THRESHOLD_M:.0f}m error)")
    print(f"{'=' * 78}")

    header = f"  {'lag':>5s}"
    for tid in traj_ids:
        header += f"  {'traj ' + tid:>16s}"
    header += f"  {'MEAN':>10s}  {'STD':>8s}"
    print(header)
    print(f"  {'-' * 5}" + f"  {'-' * 16}" * len(traj_ids) + f"  {'-' * 10}  {'-' * 8}")

    for lag in LAGS:
        row = f"  {lag:5d}"
        eces = []
        for tid in traj_ids:
            m = summary[tid].get(lag)
            if m:
                row += f"  {m['ece']:14.4f} {m['n']:4d}"
                eces.append(m['ece'])
            else:
                row += f"  {'---':>17s}"
        if eces:
            mu = sum(eces) / len(eces)
            sd = (sum((x - mu) ** 2 for x in eces) / len(eces)) ** 0.5
            row += f"  {mu:10.4f}  {sd:8.4f}"
        print(row)
        # Mark best lag
        if eces:
            best_here = min(summary[tid].get(lag, {'ece': 99})['ece'] for tid in traj_ids if lag in summary[tid])
            # ... skip for now

    # ── 2. Best lag per trajectory ──
    print(f"\n  Best lag per trajectory:")
    per_traj_best = {}
    for tid in traj_ids:
        entries = summary[tid]
        if not entries:
            continue
        best_lag = min(entries, key=lambda l: entries[l]['ece'])
        best_m = entries[best_lag]
        base_m = entries.get(0, best_m)
        impr = (base_m['ece'] - best_m['ece']) / base_m['ece'] * 100 if base_m['ece'] > 0 else 0
        per_traj_best[tid] = best_lag
        print(f"    Traj {tid}: best lag={best_lag:3d}  "
              f"ECE {base_m['ece']:.4f}→{best_m['ece']:.4f} ({impr:+.1f}%)  "
              f"n={best_m['n']}  %correct={best_m['n_correct']/best_m['n']*100:.1f}%  "
              f"err_median={best_m['error_median']:.1f}m")

    # ── 3. Cross-trajectory mean ECE per lag ──
    print(f"\n  Cross-trajectory mean ECE:")
    mean_eces = {}
    for lag in LAGS:
        vals = [summary[tid][lag]['ece'] for tid in traj_ids if lag in summary[tid]]
        if vals:
            mean_eces[lag] = sum(vals) / len(vals)

    base_mean = mean_eces.get(0, 1.0)
    prev = None
    for lag in sorted(mean_eces):
        m = mean_eces[lag]
        d = f"{prev - m:+.6f}" if prev is not None else ""
        pct = f" ({(prev - m) / prev * 100:+.1f}%)" if prev is not None and prev > 0 else ""
        sat = " ← saturation" if prev is not None and abs(prev - m) < 0.0005 else ""
        print(f"    lag={lag:3d}:  mean ECE = {m:.4f}  Δ = {d}{pct}{sat}")
        prev = m

    # ── 4. Recommendation ──
    best_mean_lag = min(mean_eces, key=mean_eces.get)
    improvement = base_mean - mean_eces[best_mean_lag]
    rel_impr = improvement / base_mean * 100 if base_mean > 0 else 0

    print(f"\n{'=' * 78}")
    print(f"  RECOMMENDATION")
    print(f"{'=' * 78}")
    print(f"""
  Cross-trajectory optimal lag = {best_mean_lag}
    Mean ECE at lag={best_mean_lag}:  {mean_eces[best_mean_lag]:.4f}
    Mean ECE at lag=0:               {base_mean:.4f}
    Absolute improvement:            {improvement:.4f}
    Relative improvement:            {rel_impr:.1f}%

  Per-trajectory best lags: {per_traj_best}
""")

    # ── 5. Error statistics ──
    print(f"  ── Per-trajectory error statistics (baseline lag=0) ──")
    print(f"  {'Traj':>5s}  {'Points':>7s}  {'Mean err':>8s}  {'Median':>8s}  "
          f"{'Max':>8s}  {'%Correct ≤' + str(int(THRESHOLD_M)) + 'm':>15s}")
    print(f"  {'-' * 5}  {'-' * 7}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 18}")
    for tid in traj_ids:
        m = summary[tid].get(0)
        if m:
            print(f"  {tid:>5s}  {m['n']:7d}  {m['error_mean']:8.1f}  {m['error_median']:8.1f}  "
                  f"{m['error_max']:8.1f}  {m['n_correct']/m['n']*100:14.1f}%")


if __name__ == '__main__':
    main()
