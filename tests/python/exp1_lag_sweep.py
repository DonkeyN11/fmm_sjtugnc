#!/usr/bin/env python3
"""
Lag-steps sweep: measure calibration vs delay trade-off.

Runs CMM with lag_steps ∈ [0, 10, 20, ..., max_lag] on traj11,
computes calibration metrics (ECE, Brier, LogLoss) at each level,
and recommends the optimal balance point.

Usage:
  python3 tests/python/exp1_lag_sweep.py [--max 100] [--step 10]
"""

import csv, math, re, sys, subprocess, tempfile, shutil, os, argparse
from pathlib import Path
from xml.etree import ElementTree as ET


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry & Metrics
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

def compute_ece(confidences, labels, n_bins=10):
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    per_bin = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = [(lo <= c <= hi) if i == n_bins - 1 else (lo <= c < hi)
                for c in confidences]
        idxs = [j for j, m in enumerate(mask) if m]
        if not idxs:
            per_bin.append({'n': 0, 'mean_conf': float('nan'), 'accuracy': float('nan')})
            continue
        n_b = len(idxs)
        per_bin.append({
            'n': n_b,
            'mean_conf': sum(confidences[j] for j in idxs) / n_b,
            'accuracy': sum(labels[j] for j in idxs) / n_b,
        })
    N = len(confidences)
    ece = sum(b['n'] / N * abs(b['mean_conf'] - b['accuracy'])
              for b in per_bin if b['n'] > 0)
    return ece, per_bin

def compute_brier_logloss(confidences, labels):
    brier = sum((c - l) ** 2 for c, l in zip(confidences, labels)) / len(confidences)
    eps = 1e-15
    logloss = -sum(l * math.log(max(c, eps)) + (1 - l) * math.log(max(1 - c, eps))
                   for c, l in zip(confidences, labels)) / len(confidences)
    return brier, logloss


# ═══════════════════════════════════════════════════════════════════════════════
# CMM Runner
# ═══════════════════════════════════════════════════════════════════════════════

def set_lag_in_xml(xml_path: Path, lag: int, out_file: str) -> Path:
    """Write a temp XML with updated lag_steps and output file. Returns path to temp file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    params = root.find('parameters')
    if params is None:
        params = ET.SubElement(root, 'parameters')
    lag_el = params.find('lag_steps')
    if lag_el is None:
        lag_el = ET.SubElement(params, 'lag_steps')
    lag_el.text = str(lag)

    # Also update output file to avoid overwrites between runs
    out_el = root.find('output').find('file')
    out_el.text = out_file

    tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='wb')
    tree.write(tmp, encoding='utf-8', xml_declaration=True)
    tmp.close()
    return Path(tmp.name)

def run_cmm(config_path: Path, cmm_bin: Path):
    """Run CMM, capture output path from config."""
    subprocess.run([str(cmm_bin), str(config_path)],
                   check=True, capture_output=True, cwd=str(cmm_bin.parent.parent))
    # Read output path from config
    tree = ET.parse(config_path)
    out_file = tree.getroot().find('output').find('file').text
    return out_file

def analyze_output(out_path: Path, threshold_m=5.0):
    """Compute calibration metrics for trustworthiness and EP."""
    trusts, eps, errors = [], [], []
    with open(out_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            og = parse_point(row.get('ogeom', ''))
            pg = parse_point(row.get('pgeom', ''))
            if og is None or pg is None:
                continue
            err = haversine_m(*og, *pg)
            errors.append(err)
            def _f(k):
                try: return float(row.get(k, ''))
                except: return float('nan')
            trusts.append(_f('trustworthiness'))
            eps.append(_f('ep'))

    labels = [1 if e <= threshold_m else 0 for e in errors]
    n = len(errors)
    n_correct = sum(labels)

    # Trustworthiness calibration
    ece_tw, bins_tw = compute_ece(trusts, labels)
    brier_tw, ll_tw = compute_brier_logloss(trusts, labels)

    # EP calibration
    ece_ep, bins_ep = compute_ece(eps, labels)
    brier_ep, ll_ep = compute_brier_logloss(eps, labels)

    return {
        'n': n, 'n_correct': n_correct,
        'ece_tw': round(ece_tw, 6), 'brier_tw': round(brier_tw, 6), 'logloss_tw': round(ll_tw, 6),
        'ece_ep': round(ece_ep, 6), 'brier_ep': round(brier_ep, 6), 'logloss_ep': round(ll_ep, 6),
        'bins_tw': bins_tw, 'bins_ep': bins_ep,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=100, help='Maximum lag steps')
    parser.add_argument('--step', type=int, default=10, help='Step increment')
    parser.add_argument('--threshold', type=float, default=5.0, help='Error threshold (m)')
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    xml_path = base / 'input/config/cmm_config_omp.xml'
    cmm_bin = base / 'build/cmm'

    lags = list(range(0, args.max + 1, args.step))
    results = []

    print(f"{'=' * 72}")
    print(f"  Lag-steps sweep — calibration vs delay trade-off")
    print(f"  Trajectory: traj11  |  Threshold: ≤{args.threshold:.0f}m")
    print(f"{'=' * 72}")

    backup_dir = base / 'dataset-hainan-06/mr/lag_sweep'
    backup_dir.mkdir(parents=True, exist_ok=True)

    for lag in lags:
        sys.stdout.write(f"\n  lag={lag:3d}  ...")
        sys.stdout.flush()

        # Unique output file per lag run
        out_name = str(backup_dir / f"cmm_0508_traj11_lag{lag:03d}.csv")
        tmp_xml = set_lag_in_xml(xml_path, lag, out_name)

        # Run CMM
        try:
            _ = run_cmm(tmp_xml, cmm_bin)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {e}")
            os.unlink(tmp_xml)
            continue

        # Analyze
        r = analyze_output(Path(out_name), args.threshold)
        r['lag'] = lag
        results.append(r)
        os.unlink(tmp_xml)

        print(f"  done  |  ECE(tw)={r['ece_tw']:.4f}  ECE(ep)={r['ece_ep']:.4f}  "
              f"Brier={r['brier_tw']:.4f}  LogLoss={r['logloss_tw']:.4f}")

    # ── Summary table ──
    if not results:
        print("\n  No results.")
        return

    print(f"\n{'=' * 72}")
    print(f"  Calibration Summary")
    print(f"{'=' * 72}")
    print(f"  {'lag':>5s}  {'ECE(tw)':>9s}  {'ECE(ep)':>9s}  {'Brier(tw)':>10s}  {'LogLoss':>10s}  {'%correct':>9s}  {'ΔECE/step':>10s}")
    print(f"  {'-' * 5}  {'-' * 9}  {'-' * 9}  {'-' * 10}  {'-' * 10}  {'-' * 9}  {'-' * 10}")

    prev_ece = None
    for i, r in enumerate(results):
        delta = "" if prev_ece is None else f"{prev_ece - r['ece_tw']:+.6f}"
        print(f"  {r['lag']:5d}  {r['ece_tw']:9.6f}  {r['ece_ep']:9.6f}  {r['brier_tw']:10.6f}  "
              f"{r['logloss_tw']:10.6f}  {r['n_correct']/r['n']*100:8.1f}%  {delta:>10s}")
        prev_ece = r['ece_tw']

    # ── Best trade-off point ──
    print(f"\n{'=' * 72}")
    print(f"  Trade-off Analysis")
    print(f"{'=' * 72}")

    # Find point of diminishing returns: where ΔECE between consecutive lags
    # falls below some threshold
    deltas = []
    for i in range(1, len(results)):
        d = results[i - 1]['ece_tw'] - results[i]['ece_tw']
        l = results[i]['lag'] - results[i - 1]['lag']
        deltas.append(d / l if l > 0 else 0)

    print(f"  ECE improvement per step:")
    for i in range(len(deltas)):
        lag_from = results[i]['lag']
        lag_to = results[i + 1]['lag']
        print(f"    lag {lag_from:3d} → {lag_to:3d}:  ΔECE/step = {deltas[i]:+.6f}")

    # Recommended: first lag where improvement < 0.1% ECE per step
    rec_lag = None
    for i in range(len(deltas)):
        if abs(deltas[i]) < 0.0005:  # less than 0.05 percentage points
            rec_lag = results[i + 1]['lag']
            break
    if rec_lag is None and results:
        # Fallback: use the inflection point (max curvature)
        best_improvement = max(range(len(deltas)), key=lambda i: deltas[i])
        rec_lag = results[min(best_improvement + 2, len(results) - 1)]['lag']

    rec = next((r for r in results if r['lag'] == rec_lag), results[-1])
    base_r = results[0]

    print(f"\n  ★ Recommended lag_steps = {rec_lag}")
    print(f"    ECE improvement:  {base_r['ece_tw']:.6f} → {rec['ece_tw']:.6f}  "
          f"({(base_r['ece_tw']-rec['ece_tw'])/base_r['ece_tw']*100:+.1f}%)")
    print(f"    Brier improvement: {base_r['brier_tw']:.6f} → {rec['brier_tw']:.6f}")
    print(f"    Cost: {rec_lag} step delay (~{rec_lag}s at 1 Hz GPS)")

    # ── Bin-level detail for best lag ──
    print(f"\n  Bin detail at recommended lag={rec_lag} (≤{args.threshold:.0f}m):")
    for b in rec['bins_tw']:
        if b['n'] == 0:
            continue
        print(f"    n={b['n']:5d}  mean_trust={b['mean_conf']:.4f}  acc={b['accuracy']:.4f}  "
              f"gap={abs(b['mean_conf']-b['accuracy']):.4f}")


if __name__ == '__main__':
    main()
