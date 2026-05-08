#!/usr/bin/env python3
"""
Experiment 1: Reliability Diagram — Emission Probability Calibration

Compares the emission model calibration of two HMM map-matching approaches:

  - FMM  (isotropic) : EP = P(obs|state) ∝ exp(-d_eucl² / 2σ²)
  - CMM  (anisotropic): EP = P(obs|state) ∝ exp(-½·dᵀΣ⁻¹d)

Hypothesis: Anisotropic (Mahalanobis-based) EP is better calibrated than
            isotropic (Euclidean-based) EP, producing a reliability curve
            closer to the diagonal y=x.

Metrics per error threshold (3m, 5m, 10m):
  1. Reliability Diagram — binned EP vs actual accuracy
  2. ECE & MCE — Expected & Maximum Calibration Error
  3. Brier Score & Log-Loss — Strictly proper scoring rules
  4. ASCII & JSON output

Usage:
  python3 tests/python/exp1_reliability_diagram.py
"""

import bisect
import csv
import json
import math
import re
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry
# ═══════════════════════════════════════════════════════════════════════════════

def parse_point(wkt: str):
    m = re.search(r'POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)', wkt or '', re.I)
    return (float(m.group(1)), float(m.group(2))) if m else None


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ece_and_bins(confidences, labels, n_bins=10):
    """
    confidences: EP values ∈ [0,1], higher = model more confident in observation
    labels     : 1 = correct match (error ≤ threshold), 0 = wrong
    Returns dict with ECE, MCE, per_bin data.
    """
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    per_bin = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = [(lo <= c < hi) for c in confidences]
        if i == n_bins - 1:
            mask = [(lo <= c <= hi) for c in confidences]

        idxs = [j for j, m in enumerate(mask) if m]
        if not idxs:
            per_bin.append({
                'bin_center': round((lo + hi) / 2, 2),
                'mean_conf': float('nan'), 'accuracy': float('nan'),
                'count': 0, 'lo': lo, 'hi': hi
            })
            continue

        n_b = len(idxs)
        mean_c = sum(confidences[j] for j in idxs) / n_b
        acc = sum(labels[j] for j in idxs) / n_b

        per_bin.append({
            'bin_center': round((lo + hi) / 2, 2),
            'mean_conf': round(mean_c, 4),
            'accuracy': round(acc, 4),
            'count': n_b,
            'lo': lo, 'hi': hi
        })

    N = len(confidences)
    ece = sum(b['count'] / N * abs(b['mean_conf'] - b['accuracy'])
              for b in per_bin if b['count'] > 0)
    mce = max((abs(b['mean_conf'] - b['accuracy']) for b in per_bin if b['count'] > 0),
              default=float('nan'))

    return {'ece': round(ece, 4), 'mce': round(mce, 4),
            'n_total': N, 'per_bin': per_bin}


def compute_brier_logloss(confidences, labels):
    brier = sum((c - l) ** 2 for c, l in zip(confidences, labels)) / len(confidences)
    eps = 1e-15
    logloss = -sum(l * math.log(max(c, eps)) + (1 - l) * math.log(max(1 - c, eps))
                   for c, l in zip(confidences, labels)) / len(confidences)
    return round(brier, 6), round(logloss, 6)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_cmm(path: Path):
    """Returns list: [{ogeom, pgeom, error_m, ts, ep, tp, ...}]"""
    records = []
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            og = parse_point(row.get('ogeom', ''))
            pg = parse_point(row.get('pgeom', ''))
            if og is None or pg is None:
                continue
            err = haversine_m(*og, *pg)
            def _f(k):
                try: return float(row.get(k, ''))
                except: return float('nan')
            records.append({
                'ogeom': og, 'pgeom': pg, 'error_m': err,
                'ts': int(row.get('timestamp', '0')),
                'ep': _f('ep'), 'tp': _f('tp'),
                'trustworthiness': _f('trustworthiness'),
                'cumu_prob': _f('cumu_prob'),
                'delta_entropy': _f('delta_entropy'),
            })
    return records


def load_fmm(path: Path):
    """Returns list: [{pgeom, ts, ep, tp, trustworthiness}]"""
    records = []
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            pg = parse_point(row.get('pgeom', ''))
            if pg is None:
                continue
            def _f(k):
                try: return float(row.get(k, ''))
                except: return float('nan')
            records.append({
                'pgeom': pg,
                'ts': int(row.get('timestamp', '0')),
                'ep': _f('ep'), 'tp': _f('tp'),
                'trustworthiness': _f('trustworthiness'),
            })
    return records


def match_fmm_to_cmm_ogeom(fmm_records, cmm_records):
    """Align FMM rows to CMM rows by nearest timestamp (tolerance 500ms).
    Compute FMM error = haversine(CMM_ogeom, FMM_pgeom)."""
    cmm_ts_sorted = sorted(r['ts'] for r in cmm_records)
    cmm_by_ts = {r['ts']: r for r in cmm_records}

    matched = []
    for fmm_r in fmm_records:
        ts = fmm_r['ts']
        idx = bisect.bisect_left(cmm_ts_sorted, ts)
        best_ts = cmm_ts_sorted[max(0, min(len(cmm_ts_sorted) - 1, idx))]
        best_dist = abs(best_ts - ts)
        for j in range(max(0, idx - 2), min(len(cmm_ts_sorted), idx + 2)):
            d = abs(cmm_ts_sorted[j] - ts)
            if d < best_dist:
                best_dist, best_ts = d, cmm_ts_sorted[j]

        if best_dist > 500:
            continue
        cmm_r = cmm_by_ts[best_ts]
        fmm_r['error_m'] = haversine_m(*cmm_r['ogeom'], *fmm_r['pgeom'])
        matched.append(fmm_r)

    return matched


# ═══════════════════════════════════════════════════════════════════════════════
# ASCII Plot
# ═══════════════════════════════════════════════════════════════════════════════

def ascii_reliability_plot(cmm_bins, fmm_bins, title, ece_c, ece_f):
    """Draw ASCII reliability diagram."""
    rows = 21
    cols = 11
    grid = [[' ' for _ in range(cols)] for _ in range(rows)]

    # diagonal dots
    for r in range(rows):
        c = r * (cols - 1) // (rows - 1)
        grid[r][c] = '·'

    # CMM markers (C) and FMM markers (F)
    for bins_data, ch in [(cmm_bins, 'C'), (fmm_bins, 'F')]:
        for b in bins_data:
            if b['count'] == 0:
                continue
            row = rows - 1 - min(rows - 1, int(round(b['accuracy'] * (rows - 1))))
            col = min(cols - 1, int(round(b['mean_conf'] * (cols - 1))))
            existing = grid[row][col]
            if existing == '·':
                grid[row][col] = ch
            elif existing != ch:
                grid[row][col] = 'B'  # Both

    print("\n  " + title + (f"  (CMM ECE={ece_c:.4f}, FMM ECE={ece_f:.4f})"))
    print("        0    0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9   1.0")
    print("        +----+-----+-----+-----+-----+-----+-----+-----+-----+-----+")
    for r in range(rows):
        acc = (rows - 1 - r) / (rows - 1)
        line = f"  {acc:.2f}  |"
        for c in range(cols):
            line += f"  {grid[r][c]}  "
        line += '|'
        print(line)
    print("        +----+-----+-----+-----+-----+-----+-----+-----+-----+-----+")
    print("  Legend: C=CMM(aniso)  F=FMM(iso)  B=Both  ·=perfect calibration")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    base = Path(__file__).resolve().parents[2]
    cmm_path = base / 'dataset-hainan-06/mr/cmm_0508_traj11_delta_entropy.csv'
    fmm_path = base / 'dataset-hainan-06/mr/fmm_traj11_0508.csv'

    # ── Load ──
    print("Loading CMM (anisotropic) data...")
    cmm = load_cmm(cmm_path)
    print(f"  CMM: {len(cmm)} points  |  EP ∈ [{min(r['ep'] for r in cmm):.4f}, "
          f"{max(r['ep'] for r in cmm):.4f}]")

    print("Loading FMM (isotropic) data + matching to CMM ogeom...")
    fmm_raw = load_fmm(fmm_path)
    fmm = match_fmm_to_cmm_ogeom(fmm_raw, cmm)
    print(f"  FMM: {len(fmm)}/{len(fmm_raw)} matched  |  EP ∈ [{min(r['ep'] for r in fmm):.4f}, "
          f"{max(r['ep'] for r in fmm):.4f}]")

    thresholds_m = [3, 5, 10]

    print("\n" + "=" * 78)
    print("  Experiment 1: Reliability Diagram")
    print("  Metric: EP (emission probability P(obs|state))")
    print("  Hypothesis: Anisotropic Mahalanobis EP is better calibrated")
    print("              than isotropic Euclidean EP.")
    print("=" * 78)

    all_results = {}

    for thresh in thresholds_m:
        print(f"\n{'─' * 78}")
        print(f"  Error threshold ≤ {thresh}m")

        # CMM
        cmm_ep_conf = [r['ep'] for r in cmm]
        cmm_labels = [1 if r['error_m'] <= thresh else 0 for r in cmm]
        cmm_cal = compute_ece_and_bins(cmm_ep_conf, cmm_labels)
        brier_c, ll_c = compute_brier_logloss(cmm_ep_conf, cmm_labels)

        # FMM
        fmm_ep_conf = [r['ep'] for r in fmm]
        fmm_labels = [1 if r['error_m'] <= thresh else 0 for r in fmm]
        fmm_cal = compute_ece_and_bins(fmm_ep_conf, fmm_labels)
        brier_f, ll_f = compute_brier_logloss(fmm_ep_conf, fmm_labels)

        all_results[thresh] = {
            'cmm': {'cal': cmm_cal, 'brier': brier_c, 'logloss': ll_c, 'n_correct': sum(cmm_labels)},
            'fmm': {'cal': fmm_cal, 'brier': brier_f, 'logloss': ll_f, 'n_correct': sum(fmm_labels)},
        }

        # ── Bin table ──
        print(f"\n  {'Bin':>4s}  {'Range':>8s}  {'CMM n':>7s}  {'CMM EP':>7s}  {'CMM acc':>8s}  "
              f"{'FMM n':>7s}  {'FMM EP':>7s}  {'FMM acc':>8s}")
        print("  " + "-" * 68)
        for i in range(10):
            cb = cmm_cal['per_bin'][i]
            fb = fmm_cal['per_bin'][i]
            c_ep_str = f'{cb["mean_conf"]:.4f}' if cb['count'] > 0 else '    nan'
            c_ac_str = f'{cb["accuracy"]:.4f}' if cb['count'] > 0 else '    nan'
            f_ep_str = f'{fb["mean_conf"]:.4f}' if fb['count'] > 0 else '    nan'
            f_ac_str = f'{fb["accuracy"]:.4f}' if fb['count'] > 0 else '    nan'
            print(f"  {i:4d}  {cb['lo']:.1f}-{cb['hi']:.1f}  {cb['count']:7d}  {c_ep_str:>7s}  {c_ac_str:>8s}  "
                  f"{fb['count']:7d}  {f_ep_str:>7s}  {f_ac_str:>8s}")

        # ── Summary ──
        nc = all_results[thresh]['cmm']['n_correct']
        nf = all_results[thresh]['fmm']['n_correct']
        print(f"\n  CMM: {nc}/{len(cmm)} correct ({nc/len(cmm)*100:.1f}%) | "
              f"ECE={cmm_cal['ece']:.4f}  MCE={cmm_cal['mce']:.4f}  "
              f"Brier={brier_c:.4f}  LogLoss={ll_c:.4f}")
        print(f"  FMM: {nf}/{len(fmm)} correct ({nf/len(fmm)*100:.1f}%) | "
              f"ECE={fmm_cal['ece']:.4f}  MCE={fmm_cal['mce']:.4f}  "
              f"Brier={brier_f:.4f}  LogLoss={ll_f:.4f}")

        # ── ASCII plot ──
        ascii_reliability_plot(
            cmm_cal['per_bin'], fmm_cal['per_bin'],
            f"EP Calibration at ≤{thresh}m",
            cmm_cal['ece'], fmm_cal['ece']
        )

    # ── Final comparison table ──
    print("\n" + "=" * 78)
    print("  Final Comparison")
    print("=" * 78)
    header = f"  {'Metric':<16s}  "
    for t in thresholds_m:
        header += f"{'≤' + str(t) + 'm':>28s}  "
    print(header)
    sub = f"  {'':16s}  "
    for t in thresholds_m:
        sub += f"{'CMM':>13s}  {'FMM':>13s}  "
    print(sub)
    print("  " + "-" * 76)

    for mkey, label, d in [('ece', 'ECE ↓', True), ('mce', 'MCE ↓', True),
                             ('brier', 'Brier ↓', True), ('logloss', 'LogLoss ↓', True)]:
        row = f"  {label:<16s}  "
        for t in thresholds_m:
            cv = all_results[t]['cmm']['cal'][mkey] if mkey in ('ece','mce') else all_results[t]['cmm'][mkey]
            fv = all_results[t]['fmm']['cal'][mkey] if mkey in ('ece','mce') else all_results[t]['fmm'][mkey]
            # Mark winner
            c_better = (cv < fv) if d else (cv > fv)
            cm = ' ★' if c_better else ''
            fm = ' ★' if not c_better else ''
            row += f"  {cv:>11.4f}{cm}  {fv:>11.4f}{fm}  "
        print(row)

    # ── Export JSON ──
    output = {
        'experiment': 'exp1_reliability_diagram',
        'metric': 'EP (emission probability)',
        'methods': {'cmm': 'anisotropic Mahalanobis EP', 'fmm': 'isotropic Euclidean EP'},
        'thresholds_m': thresholds_m,
        'per_threshold': {}
    }
    for t in thresholds_m:
        output['per_threshold'][str(t)] = {
            'cmm': {
                'per_bin': all_results[t]['cmm']['cal']['per_bin'],
                'ece': all_results[t]['cmm']['cal']['ece'],
                'mce': all_results[t]['cmm']['cal']['mce'],
                'brier': all_results[t]['cmm']['brier'],
                'logloss': all_results[t]['cmm']['logloss'],
                'n_correct': all_results[t]['cmm']['n_correct'],
                'n_total': len(cmm),
            },
            'fmm': {
                'per_bin': all_results[t]['fmm']['cal']['per_bin'],
                'ece': all_results[t]['fmm']['cal']['ece'],
                'mce': all_results[t]['fmm']['cal']['mce'],
                'brier': all_results[t]['fmm']['brier'],
                'logloss': all_results[t]['fmm']['logloss'],
                'n_correct': all_results[t]['fmm']['n_correct'],
                'n_total': len(fmm),
            }
        }

    out_path = base / 'dataset-hainan-06/mr/exp1_reliability.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Calibration data exported → {out_path}")

    # ── Interpretation ──
    print("\n" + "=" * 78)
    print("  Interpretation")
    print("=" * 78)
    best_t = thresholds_m[0]
    ece_c = all_results[best_t]['cmm']['cal']['ece']
    ece_f = all_results[best_t]['fmm']['cal']['ece']
    print(f"""
  At ≤{best_t}m threshold:
    CMM (aniso) ECE = {ece_c:.4f}    FMM (iso) ECE = {ece_f:.4f}

  {('★ CMM anisotropic EP IS better calibrated than isotropic' if ece_c < ece_f else
     'FMM isotropic EP is better calibrated')}

  {('  ECE < 0.05 → excellent calibration, EP usable as probability' if ece_c < 0.05 else
     '  ECE 0.05–0.10 → moderate calibration' if ece_c < 0.10 else
     '  ECE > 0.10 → poor calibration')}
  """)

    # Add hypothesis test summary
    print("  Hypothesis test:")
    for t in thresholds_m:
        cd = all_results[t]['cmm']['cal']
        fd = all_results[t]['fmm']['cal']
        winner = 'CMM ★' if cd['ece'] < fd['ece'] else 'FMM ★'
        print(f"    ≤{t}m:  CMM ECE {cd['ece']:.4f}  vs  FMM ECE {fd['ece']:.4f}  →  {winner}")


if __name__ == '__main__':
    main()
