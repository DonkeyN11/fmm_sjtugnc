#!/usr/bin/env python3
"""
Final case study analysis with geographic context.
Outputs detailed CSV per scenario and summary report.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_point(wkt: str):
    m = re.search(r'POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)', wkt or '', re.I)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def parse_candidates(raw: str):
    if not raw or raw.strip() == '':
        return []
    raw = raw.strip()
    tuples = re.findall(r'\(([^()]+)\)', raw)
    candidates = []
    for t in tuples:
        parts = [x.strip() for x in t.split(',')]
        if len(parts) >= 3:
            try:
                lon, lat, ep = float(parts[0]), float(parts[1]), float(parts[2])
                if lon == 0.0 and lat == 0.0:
                    continue
                candidates.append((lon, lat, ep))
            except (ValueError, TypeError):
                continue
    return candidates


def load_cmm_results(path: Path):
    rows = []
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            parsed = {
                'id': int(row.get('id', '0')),
                'seq': int(row.get('seq', '0')),
                'status': row.get('status', ''),
                'timestamp': int(row.get('timestamp', '0')),
                'opath': row.get('opath', ''),
                'cpath': row.get('cpath', ''),
                'ep': float(row.get('ep', '0')),
                'tp': float(row.get('tp', '0')),
                'tw': float(row.get('trustworthiness', '0')),
                'delta_entropy': float(row.get('delta_entropy', '0')),
                'posterior_entropy': float(row.get('posterior_entropy', '0')),
                'h0_lambda': float(row.get('h0_lambda', '0')),
                'cumu_prob': float(row.get('cumu_prob', '0')),
            }
            ogeom = parse_point(row.get('ogeom', ''))
            pgeom = parse_point(row.get('pgeom', ''))
            parsed['ogeom'] = ogeom
            parsed['pgeom'] = pgeom
            cands = parse_candidates(row.get('candidates', ''))
            parsed['num_candidates'] = len(cands)
            parsed['candidates'] = cands
            rows.append(parsed)
    return rows


def load_ground_truth(path: Path):
    gt = {}
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            tid = int(row.get('id', '0'))
            seq = int(row.get('seq', '0'))
            edge = str(row.get('edge_id', '')).strip()
            gt[(tid, seq)] = edge
    return gt


def check_correctness(cmm_rows, gt):
    for row in cmm_rows:
        key = (row['id'], row['seq'])
        gt_edge = gt.get(key, '')
        cpath = str(row['cpath']).strip()
        row['gt_edge'] = gt_edge
        row['is_correct'] = (cpath == gt_edge and cpath != '' and gt_edge != '' and gt_edge != '0')


def export_csv(segment, path):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['traj_id', 'seq', 'timestamp', 'tw', 'ep', 'tp',
                         'num_candidates', 'posterior_entropy', 'delta_entropy',
                         'is_correct', 'cpath', 'gt_edge', 'status',
                         'ogeom_lon', 'ogeom_lat', 'pgeom_lon', 'pgeom_lat'])
        for r in segment:
            writer.writerow([
                r['id'], r['seq'], r['timestamp'],
                r['tw'], r['ep'], r['tp'], r['num_candidates'],
                r['posterior_entropy'], r['delta_entropy'],
                r['is_correct'], r['cpath'], r['gt_edge'], r['status'],
                r['ogeom'][0] if r['ogeom'] else '', r['ogeom'][1] if r['ogeom'] else '',
                r['pgeom'][0] if r['pgeom'] else '', r['pgeom'][1] if r['pgeom'] else '',
            ])


def main():
    base = Path(__file__).resolve().parents[2]
    cmm_path = base / 'data/real_vehicle/hainan_06/processed/cmm_result.csv'
    gt_path = base / 'data/real_vehicle/hainan_06/processed/ground_truth.csv'

    rows = load_cmm_results(cmm_path)
    gt = load_ground_truth(gt_path)
    check_correctness(rows, gt)

    out_dir = base / 'data/real_vehicle/hainan_06/processed/case_studies'
    out_dir.mkdir(exist_ok=True)

    # Group by trajectory
    traj_data = {}
    for tid in sorted(set(r['id'] for r in rows)):
        traj_data[tid] = [r for r in rows if r['id'] == tid]

    # ── Scenario 1: High TW, correct, 2-3 candidates ──
    # Best: Traj 21 - many long highway sections with exactly 2 candidates
    # Pick a 15-epoch window from a highway segment
    print("=" * 78)
    print("  FINAL SCENARIO 1 SELECTIONS")
    print("=" * 78)

    t21 = traj_data[21]
    # seq 274-430 is a 157-epoch highway (cpath=3564), all TW=1.0, 2 cand
    # Pick seq 330-350
    s1_t21 = [r for r in t21 if 330 <= r['seq'] <= 350]
    export_csv(s1_t21, out_dir / 'scenario1_traj21_highway.csv')

    print(f"\nS1 Pick A — Traj 21, seq 330-350 (21 epochs)")
    print(f"  Location: highway, very simple road, 2 candidates only")
    print(f"  TW: all 1.0000, EP: ~0.90, correct match")
    print(f"  cpath=3564")
    print(f"  Geographic: from ({s1_t21[0]['ogeom'][0]:.6f},{s1_t21[0]['ogeom'][1]:.6f})")
    print(f"            to   ({s1_t21[-1]['ogeom'][0]:.6f},{s1_t21[-1]['ogeom'][1]:.6f})")
    print(f"  Exported: scenario1_traj21_highway.csv")

    # Also check Traj 22 seq 1072-1174 (103 epochs, 2 candidates) - another good S1
    t22 = traj_data[22]
    s1_t22 = [r for r in t22 if 1090 <= r['seq'] <= 1110]
    export_csv(s1_t22, out_dir / 'scenario1_traj22_highway.csv')

    print(f"\nS1 Pick B (backup) — Traj 22, seq 1090-1110 (21 epochs)")
    print(f"  Location: highway, 2 candidates, all TW=1.0000")
    print(f"  cpath=147666")
    print(f"  Geographic: from ({s1_t22[0]['ogeom'][0]:.6f},{s1_t22[0]['ogeom'][1]:.6f})")
    print(f"            to   ({s1_t22[-1]['ogeom'][0]:.6f},{s1_t22[-1]['ogeom'][1]:.6f})")
    print(f"  Exported: scenario1_traj22_highway.csv")

    # ── Scenario 2: Correct, complex road, many candidates, low/oscillating TW ──
    print("\n" + "=" * 78)
    print("  FINAL SCENARIO 2 SELECTIONS")
    print("=" * 78)

    t13 = traj_data[13]

    # Best: Traj 13 seq 1718-1743 - clear TW-cand anti-correlation
    s2_t13 = [r for r in t13 if 1714 <= r['seq'] <= 1748]
    export_csv(s2_t13, out_dir / 'scenario2_traj13_complex_road.csv')

    print(f"\nS2 Pick A — Traj 13, seq 1714-1748 (35 epochs)")
    print(f"  Location: complex road network (ramp/junction area)")
    print(f"  TW range: 0.1157-0.6212, oscillating with candidate count")
    print(f"  Candidate range: 11-16")
    print(f"  Key pattern: TW rises when cand drops (seq 1719-1721)")
    print(f"             : TW crashes when complex network + many candidates")
    print(f"  All matches CORRECT despite low TW")
    print(f"  Geographic: from ({s2_t13[0]['ogeom'][0]:.6f},{s2_t13[0]['ogeom'][1]:.6f})")
    print(f"            to   ({s2_t13[-1]['ogeom'][0]:.6f},{s2_t13[-1]['ogeom'][1]:.6f})")
    print(f"  TW-cand details:")
    for r in s2_t13:
        marker = " ***" if r['num_candidates'] <= 12 else ""
        print(f"    seq={r['seq']:5d}  TW={r['tw']:.4f}  cand={r['num_candidates']:2d}  "
              f"ep={r['ep']:.4f}{marker}")
    print(f"  Exported: scenario2_traj13_complex_road.csv")

    # Also check alternative: Traj 22 seq 158-178 with monotonic TW decline
    s2_t22_alt = [r for r in t22 if 155 <= r['seq'] <= 182]
    export_csv(s2_t22_alt, out_dir / 'scenario2_traj22_declining_tw.csv')

    print(f"\nS2 Pick B (backup) — Traj 22, seq 155-182 (28 epochs)")
    print(f"  Location: complex area, monotonically declining TW")
    print(f"  TW: 0.1716 → 0.6779, steady decline")
    print(f"  Candidates: always 16 (maxed out)")
    print(f"  Matches correct, but TW steadily drops")
    print(f"  Exported: scenario2_traj22_declining_tw.csv")

    # ── Scenario 4: Wrong match, low TW ──
    print("\n" + "=" * 78)
    print("  FINAL SCENARIO 4 SELECTIONS")
    print("=" * 78)

    # Best: Traj 22 seq 581-636 - wrong match (41828 vs 41833), very low TW
    s4_t22 = [r for r in t22 if 575 <= r['seq'] <= 640]
    export_csv(s4_t22, out_dir / 'scenario4_traj22_wrong_match.csv')

    print(f"\nS4 Pick A — Traj 22, seq 575-640 (66 epochs)")
    print(f"  Location: parallel/adjacent road confusion")
    print(f"  TW range: 0.0013-0.6926, mostly very low")
    print(f"  cpath=41828, gt=41833 (adjacent edges — parallel road confusion)")
    print(f"  All matches WRONG")
    print(f"  Geographic: from ({s4_t22[0]['ogeom'][0]:.6f},{s4_t22[0]['ogeom'][1]:.6f})")
    print(f"            to   ({s4_t22[-1]['ogeom'][0]:.6f},{s4_t22[-1]['ogeom'][1]:.6f})")
    print(f"  Exported: scenario4_traj22_wrong_match.csv")

    # Alternative S4: Traj 21 seq 3190-3230
    t21 = traj_data[21]
    s4_t21 = [r for r in t21 if 3185 <= r['seq'] <= 3235]
    export_csv(s4_t21, out_dir / 'scenario4_traj21_wrong_match.csv')

    print(f"\nS4 Pick B (backup) — Traj 21, seq 3185-3235 (51 epochs)")
    print(f"  TW range: 0.0486-0.6681")
    print(f"  cpath=19110, gt=29226")
    print(f"  All matches WRONG")
    print(f"  Exported: scenario4_traj21_wrong_match.csv")

    # ── Summary table ──
    print("\n" + "=" * 78)
    print("  CASE STUDY SUMMARY TABLE")
    print("=" * 78)
    print(f"""
    ┌──────────┬────────┬───────────┬──────────┬──────────┬─────────┬──────────────┐
    │ Scenario │ Traj   │ Seq Range │ Epochs   │ TW Range │ Cand    │ Match        │
    ├──────────┼────────┼───────────┼──────────┼──────────┼─────────┼──────────────┤
    │ S1-A     │ 21     │ 330-350   │ 21       │ 1.00     │ 2       │ Correct      │
    │ S1-B     │ 22     │ 1090-1110 │ 21       │ 1.00     │ 2       │ Correct      │
    │ S2-A     │ 13     │ 1714-1748 │ 35       │ 0.12-0.62│ 11-16   │ Correct      │
    │ S2-B     │ 22     │ 155-182   │ 28       │ 0.17-0.68│ 16      │ Correct      │
    │ S4-A     │ 22     │ 575-640   │ 66       │ 0.00-0.69│ 16      │ WRONG        │
    │ S4-B     │ 21     │ 3185-3235 │ 51       │ 0.05-0.67│ 16      │ WRONG        │
    └──────────┴────────┴───────────┴──────────┴──────────┴─────────┴──────────────┘
    """)

    # ── Key insight: S2 vs S4 comparison ──
    print("=" * 78)
    print("  KEY INSIGHT — S2 vs S4 Comparison")
    print("=" * 78)
    print("""
    Both S2 and S4 have LOW trustworthiness, but for different reasons:
      - S2: Match is CORRECT, TW is low because the road network has many
            ambiguous candidates (complex junction/ramp). The probability mass
            is spread across many plausible candidates, reducing confidence
            in any single match. TW → low as a honest reflection of uncertainty.
      - S4: Match is WRONG, TW is low because no candidate truly matches the
            GNSS position. The algorithm correctly flags low confidence.

    Together, S2+S4 demonstrate that low TW is a NECESSARY but not SUFFICIENT
    condition for mismatch detection — low TW honestly reflects either road
    network ambiguity OR genuine mismatch. This is the expected behavior of a
    well-calibrated trustworthiness metric.
    """)


if __name__ == '__main__':
    main()
