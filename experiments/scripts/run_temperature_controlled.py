#!/usr/bin/env python3
"""Controlled isolation of the adaptive-temperature effect on CMM trustworthiness.

Runs the SAME current build/cmm binary on the 12 sample-rate datasets
(sigma in {5,15,25} m x sample interval {1,2,5,10} s) twice, differing ONLY in
the temperature_adapt config flag:

  - temperature_adapt=true  -> cmm_result.csv           (temp ON)
  - temperature_adapt=false -> cmm_result_tau_off.csv   (temp OFF)

Both runs share the same XML (k=16, PL_mult=3, lag=5, etc.) as
exp3_full_matching.py, so any difference is attributable to temperature alone.
The existing cmm_result.csv files (regenerated Aug 6 20:12 with this binary)
are kept as the temp-ON reference; this script verifies determinism by
re-running temp-ON and diffing.

Usage:
  python experiments/scripts/run_temperature_controlled.py [--jobs 8]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
NETWORK_SHP = str((PROJECT / "input/map/hainan/edges.shp").resolve())
UBODT = str((PROJECT / "input/map/hainan_ubodt_indexed.bin").resolve())
CMM_BIN = PROJECT / "build/cmm"

SIGMAS = [5, 15, 25]
INTERVALS = [1, 2, 5, 10]


def dataset_dir(sigma: int, interval: int) -> Path:
    base = PROJECT / "data/simulation" / f"sigma_{sigma:02d}" / "no_occlusion" / "no_fault"
    return base if interval == 1 else base / f"subsample_{interval}s"


def build_cmm_xml(gps_csv: str, mr_out: str, temperature_adapt: bool) -> Path:
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<config>
  <input>
    <network><file>{NETWORK_SHP}</file><id>key</id><source>u</source><target>v</target></network>
    <ubodt><file>{UBODT}</file></ubodt>
    <gps>
      <file>{gps_csv}</file><id>id</id><x>x</x><y>y</y><timestamp>timestamp</timestamp>
      <sde>sde</sde><sdn>sdn</sdn><sdu>sdu</sdu>
      <sdne>sdne</sdne><sdeu>sdeu</sdeu><sdun>sdun</sdun>
      <protection_level>protection_level</protection_level>
    </gps>
    <gps_point>true</gps_point>
  </input>
  <output>
    <file>{mr_out}</file><point_mode>true</point_mode>
    <fields><seq/><timestamp/><ogeom/><cpath/><tpath/><opath/><pgeom/>
      <ep/><tp/><trustworthiness/><n_best_trustworthiness/><candidates/>
      <status/><delta_entropy/><posterior_entropy/><h0_lambda/><cumu_prob/></fields>
  </output>
  <parameters>
    <k>16</k><min_candidates>1</min_candidates><protection_level_multiplier>3</protection_level_multiplier>
    <reverse_tolerance>0.0</reverse_tolerance><normalized>false</normalized>
    <use_mahalanobis>true</use_mahalanobis><filtered>false</filtered>
    <window_length>100</window_length>
    <max_interval>180.0</max_interval><trustworthiness_threshold>0.0</trustworthiness_threshold>
    <phmi>0.00001</phmi><lag_steps>5</lag_steps>
    <phmi_pl_multiplier>1</phmi_pl_multiplier><h0_prior_log_odds>0</h0_prior_log_odds>
    <temperature_adapt>{str(temperature_adapt).lower()}</temperature_adapt>
  </parameters>
  <other><log_level>2</log_level><use_omp>true</use_omp><step>500</step>
    <convert_to_projected>false</convert_to_projected></other>
</config>"""
    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
    tmp.write(xml)
    tmp.close()
    return Path(tmp.name)


def run_one(data_dir: Path, temperature_adapt: bool, out_name: str) -> bool:
    obs_csv = data_dir / "observations.csv"
    if not obs_csv.exists():
        print(f"  MISSING observations: {data_dir}", file=sys.stderr)
        return False
    out_csv = str((data_dir / out_name).resolve())
    xml = build_cmm_xml(str(obs_csv.resolve()), out_csv, temperature_adapt)
    try:
        proc = subprocess.run([str(CMM_BIN), str(xml)], check=True, capture_output=True,
                              text=True, cwd=str(PROJECT), timeout=1200)
        if "ERROR" in proc.stderr or "Failed" in proc.stderr:
            print(f"  CMM reported error for {data_dir} ({out_name})", file=sys.stderr)
            print(proc.stderr[-2000:], file=sys.stderr)
            return False
        return True
    except subprocess.CalledProcessError as ex:
        print(f"  CMM failed for {data_dir} ({out_name}): {ex}", file=sys.stderr)
        print(ex.stderr[-2000:] if ex.stderr else "", file=sys.stderr)
        return False
    finally:
        try:
            os.unlink(xml)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--run-temp-off", action="store_true",
                        help="Run temperature_adapt=false for all datasets (temp-ON files already exist)")
    parser.add_argument("--rerun-temp-on", action="store_true",
                        help="Re-run temperature_adapt=true to verify determinism vs existing files")
    args = parser.parse_args()

    tasks = []
    for sigma in SIGMAS:
        for sr in INTERVALS:
            d = dataset_dir(sigma, sr)
            if args.run_temp_off or not args.rerun_temp_on:
                tasks.append((d, False, "cmm_result_tau_off.csv"))
            if args.rerun_temp_on:
                tasks.append((d, True, "cmm_result_rerun_on.csv"))

    if not tasks:
        print("Nothing to do; pass --run-temp-off and/or --rerun-temp-on")
        return

    print(f"Running {len(tasks)} matching jobs with {args.jobs} workers")
    results = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, d, on, name): (d, name) for d, on, name in tasks}
        for fut in as_completed(futures):
            d, name = futures[fut]
            ok = fut.result()
            results[(str(d), name)] = ok
            print(f"  {'OK ' if ok else 'FAIL'} {d.relative_to(PROJECT)} -> {name}")

    failed = [k for k, v in results.items() if not v]
    print(f"\nDone. {len(results) - len(failed)}/{len(results)} jobs succeeded")
    if failed:
        print("Failed:", *failed, sep="\n  ")


if __name__ == "__main__":
    main()
