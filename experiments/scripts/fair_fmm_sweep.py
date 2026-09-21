#!/usr/bin/env python3
"""Score FMM against CaMM on CaMM's own terms: same input, same GT, same knobs.

The paper compares CaMM with FMM, but the two runs are not comparable as
archived: CaMM searches a per-epoch disk of radius HPL_i (median ~21 m on
Haikou, doubled when empty) while the archived FMM run searches a fixed
0.03 deg disk (~3.3 km). Everything the comparison is supposed to show --
adaptive geometry, covariance-aware emission -- is entangled with a 150x
difference in search radius.

This script sweeps FMM over radius and over the two knobs FMM shares with CaMM,
so the comparison can be made at matched search geometry:

  reverse_tolerance   the archive ran FMM at 0.0 while the CaMM real-data
                      config runs at 0.1. That is not a minor difference: at
                      0.0 a fix that jitters *backwards* along its own edge has
                      no admissible transition, and because update_tg() aborts
                      the whole trajectory at the first such pair, a 1.9 m
                      jitter killed all 2696 epochs of trajectory 11 at r=0.003.
                      Both values are swept so the effect is visible and not
                      silently chosen.
  gps_error           FMM's isotropic sigma, in network units (degrees here).
                      Swept rather than fixed, so FMM is reported at its best
                      sigma for each radius rather than at one we picked.

Metric definitions come from inventory_real_metrics.evaluate(), which imports
them from verify_paper_numbers.py -- the same code path that scores CaMM, so
the two families of numbers are directly comparable.

Usage:
    python experiments/scripts/fair_fmm_sweep.py                # full matrix
    python experiments/scripts/fair_fmm_sweep.py --paper-check  # only the
        # archived-config reproduction (r=0.03, sigma=0.001) vs both rts
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import inventory_real_metrics as inv  # noqa: E402

BASE = SCRIPT_DIR.parents[1]
FMM_BIN = BASE / "build/fmm"
GT = BASE / "data/real_vehicle/hainan_06/processed/aligned_0729.csv"
GPS_INPUT = BASE / "data/real_vehicle/hainan_06/cmm_input_points.csv"

XML = """<?xml version='1.0' encoding='utf-8'?>
<config>
  <input>
    <ubodt><file>input/map/hainan_ubodt_indexed.bin</file></ubodt>
    <network>
      <file>input/map/hainan/edges.shp</file>
      <id>key</id><source>u</source><target>v</target>
    </network>
    <gps>
      <file>{gps}</file>
      <gps_point />
      <id>id</id><x>x</x><y>y</y><timestamp>timestamp</timestamp>
    </gps>
  </input>
  <parameters>
    <k>16</k>
    <r>{r}</r>
    <gps_error>{gps_error}</gps_error>
    <reverse_tolerance>{rt}</reverse_tolerance>
  </parameters>
  <output>
    <fields><cpath /><pgeom /><trustworthiness /><timestamp /></fields>
    <file>{out}</file>
  </output>
  <other><use_omp /><input_epsg>4326</input_epsg></other>
</config>
"""


def run_fmm(work: Path, r: float, gps_error: float, rt: float,
            gps_input: Path = GPS_INPUT, tag: str = "") -> Path:
    name = tag or f"r{r}_s{gps_error}_rt{rt}"
    xml, out = work / f"{name}.xml", work / f"{name}.csv"
    xml.write_text(XML.format(gps=gps_input.resolve(), r=r,
                              gps_error=gps_error, rt=rt, out=out))
    with open(work / f"{name}.log", "w") as log:
        subprocess.run([str(FMM_BIN), str(xml)], stdout=log,
                       stderr=subprocess.STDOUT, cwd=BASE, check=False)
    return out


def score(path: Path, gt):
    """Same metrics as the CaMM rows; absent-on-disk rows are reported, not dropped."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    m = inv.evaluate(path, None, gt)
    m["absent"] = len(gt[0]) - m["n"]
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=Path("/tmp/fmm_fair_sweep"))
    ap.add_argument("--paper-check", action="store_true")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    gt = inv.load_gt(GT)

    if args.paper_check:
        # The archived artifact (fmm_result.csv, hence the manuscript's 88.9 %)
        # was produced at r=0.03, sigma=0.001. Reproduce it at both rts and
        # compare cell-by-cell on (id, seq) to learn which one the archive used.
        for rt in (0.0, 0.1):
            p = run_fmm(args.work, 0.03, 0.001, rt, tag=f"papercheck_rt{rt}")
            print(f"rt={rt}: {p.name} rows={sum(1 for _ in p.open()) - 1}")
        print()
        for other in ("fmm_result.csv", "fmm_original_paper.csv"):
            q = BASE / "data/real_vehicle/hainan_06/processed" / other
            if not q.exists():
                q = Path("/tmp/fmm_fair") / other
            if not q.exists():
                print(f"  {other}: not found")
                continue
            a = Path(args.work) / "papercheck_rt0.0.csv"
            same = diff_cells(a, q)
            print(f"  rt=0.0 vs {other}: {same} differing cells")
            b = Path(args.work) / "papercheck_rt0.1.csv"
            print(f"  rt=0.1 vs {other}: {diff_cells(b, q)} differing cells")
        return 0

    radii = [0.0002, 0.0005, 0.001, 0.003, 0.03]
    sigmas = [0.00002, 0.00005, 0.0001, 0.0005]
    rows = []
    for rt in (0.0, 0.1):
        for r in radii:
            for s in sigmas:
                p = run_fmm(args.work, r, s, rt)
                m = score(p, gt)
                if m is None:
                    print(f"r={r} s={s} rt={rt}: no output")
                    continue
                rows.append((rt, r, s, m))

    hdr = (f"{'rt':>4}{'r_deg':>9}{'~m':>6}{'sigma':>9}{'~m':>6}"
           f"{'n':>7}{'acc%':>9}{'ECE':>8}{'AUC':>8}{'TW_cor':>9}{'TW_wrg':>9}"
           f"{'sep':>8}{'err_m':>8}{'p95':>8}{'absent':>8}")
    print(hdr)
    print("-" * len(hdr))
    for rt, r, s, m in rows:
        print(f"{rt:>4}{r:>9}{r*111320:>6.0f}{s:>9}{s*111320:>6.0f}"
              f"{m['n']:>7}{m['acc']:>9.4f}{m['ece']:>8.4f}{m['auc']:>8.4f}"
              f"{m['tw_c']:>9.4f}{m['tw_w']:>9.4f}{m['sep']:>8.4f}"
              f"{m['perr']:>8.2f}{m['perr95']:>8.2f}{m['absent']:>8}")
    return 0


def diff_cells(a: Path, b: Path) -> int:
    """Count (id, seq) cells whose cpath differs between two result files."""
    def load(p):
        import csv
        d = {}
        with open(p, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                d[(r["id"].strip(), int(r["seq"]))] = r["cpath"].strip()
        return d
    da, db = load(a), load(b)
    keys = set(da) | set(db)
    return sum(1 for k in keys if da.get(k) != db.get(k))


if __name__ == "__main__":
    sys.exit(main())
