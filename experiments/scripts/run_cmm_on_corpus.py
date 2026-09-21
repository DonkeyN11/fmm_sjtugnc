#!/usr/bin/env python3
"""Run the CMM matcher over every dataset in a corpus tree, in parallel.

The XML is byte-for-byte the one `rerun_cmm_fixed_en.py` builds (k=16,
min_candidates=1, reverse_tolerance=0.0, use_mahalanobis=true, filtered=false,
max_interval=180.0, trustworthiness_threshold=0.0, phmi=1e-5,
cumulative_reverse_pct=0.03, direction_penalty=true, enable_gap_bridging=true),
so a result produced here is comparable to one produced there. `rerun_cmm_fixed_en`
hardcodes its dataset list and runs serially; this discovers the tree instead and
runs several matchers at once, which matters because the sweep dataset alone is
70k epochs.

Usage
-----
    python experiments/scripts/run_cmm_on_corpus.py \
        --root data/simulation_plfix --suffix _plfix --jobs 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
CMM_BIN = BASE / "build/cmm"
NETWORK = BASE / "input/map/hainan/edges.shp"
UBODT = BASE / "input/map/hainan_ubodt_indexed.bin"


def build_cmm_xml(gps_csv: str, mr_out: str) -> str:
    """Identical to rerun_cmm_fixed_en.build_cmm_xml (and exp3's build_cmm_xml)."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<config>
  <input>
    <network><file>{NETWORK}</file><id>key</id><source>u</source><target>v</target></network>
    <ubodt><file>{UBODT}</file></ubodt>
    <gps>
      <file>{gps_csv}</file><id>id</id><x>x</x><y>y</y><timestamp>timestamp</timestamp>
      <sde>sde</sde><sdn>sdn</sdn><sdu>sdu</sdu>
      <sdne>sdne</sdne><sdeu>sdeu</sdeu><sdun>sdun</sdun>
      <protection_level>protection_level</protection_level>
    </gps>
  </input>
  <output>
    <file>{mr_out}</file>
    <fields><point_mode/><seq/><timestamp/><ogeom/><cpath/><tpath/><opath/><pgeom/>
      <ep/><tp/><trustworthiness/><n_best_trustworthiness/><candidates/>
      <status/><delta_entropy/><posterior_entropy/><cumu_prob/></fields>
  </output>
  <parameters>
    <k>16</k><min_candidates>1</min_candidates>
    <reverse_tolerance>0.0</reverse_tolerance>
    <use_mahalanobis>true</use_mahalanobis><filtered>false</filtered>
    <max_interval>180.0</max_interval><trustworthiness_threshold>0.0</trustworthiness_threshold>
    <phmi>0.00001</phmi><cumulative_reverse_pct>0.03</cumulative_reverse_pct>
    <direction_penalty>true</direction_penalty><enable_gap_bridging>true</enable_gap_bridging>
  </parameters>
  <other><log_level>2</log_level><use_omp>true</use_omp></other>
</config>"""


def run_one(data_dir: Path, suffix: str, force: bool) -> tuple[str, bool, str, float]:
    obs = data_dir / "observations.csv"
    out = data_dir / f"cmm_result{suffix}.csv"
    if out.exists() and not force:
        return str(data_dir), True, "skip", 0.0

    tmp = Path(tempfile.mkstemp(suffix=".xml")[1])
    tmp.write_text(build_cmm_xml(str(obs.resolve()), str(out.resolve())))
    t0 = time.time()
    try:
        r = subprocess.run([str(CMM_BIN), str(tmp)], capture_output=True, text=True)
    finally:
        tmp.unlink(missing_ok=True)
    dt = time.time() - t0
    if r.returncode != 0:
        return str(data_dir), False, (r.stderr or r.stdout or "")[-400:], dt
    return str(data_dir), True, "", dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/simulation_plfix"))
    ap.add_argument("--suffix", default="_plfix")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = (BASE / args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"corpus root not found: {root}")
    if not CMM_BIN.exists():
        raise SystemExit(f"matcher not built: {CMM_BIN}")

    dirs = sorted(p.parent for p in root.rglob("observations.csv"))
    print(f"corpus : {root}")
    print(f"suffix : cmm_result{args.suffix}.csv")
    print(f"n      : {len(dirs)} datasets, {args.jobs} at a time", flush=True)

    t0 = time.time()
    fails = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(run_one, d, args.suffix, args.force): d for d in dirs}
        for f in as_completed(futs):
            name, ok, msg, dt = f.result()
            rel = Path(name).relative_to(root)
            if not ok:
                fails.append(str(rel))
                print(f"  [FAIL] {rel}  {dt:6.1f}s\n{msg}", flush=True)
            elif msg != "skip":
                print(f"  [ok  ] {rel}  {dt:6.1f}s", flush=True)

    print(f"\ndone in {time.time() - t0:.1f}s; {len(fails)} failures", flush=True)
    if fails:
        print("failed:", fails)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
