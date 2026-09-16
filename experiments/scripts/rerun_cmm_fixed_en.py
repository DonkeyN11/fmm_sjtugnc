#!/usr/bin/env python3
"""Batch re-run CMM with the fixed E/N covariance transposition.

The E/N transposition bug in the point-based CSV parser
(cmm_algorithm.cpp, CovarianceMatrix{*sdn_opt, *sde_opt, ...}) was fixed.
This script re-runs all affected point-based datasets with the fixed binary,
writing results with the suffix `_fixed_en` to distinguish them from the
old (buggy) results.

Coverage:
  - sigma sweep:    data/simulation/sigma_XX/no_occlusion/no_fault/
  - subsamples:     data/simulation/sigma_XX/.../subsample_NNs/
  - degraded:       data/simulation/sigma_30/{no_occlusion,with_occlusion}/{no_fault,with_fault}/
  - sigma mismatch: data/simulation/sigma_mismatch/prXX_wls20/

Config: replicates exp3_full_matching.py build_cmm_xml EXACTLY
(k=16, min_candidates=1, protection_level_multiplier=3, lag_steps=5,
 phmi=1e-5, phmi_pl_multiplier=1, normalized=false, use_mahalanobis=true,
 filtered=false). Unspecified params use XML defaults (background_prob=0.1,
 map_error_std=5.0e-5, cumulative_reverse_pct=0.03, temperature_adapt=true,
 direction_penalty=true).

Usage: python experiments/scripts/rerun_cmm_fixed_en.py [--force]
"""

import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
CMM_BIN = BASE / "build/cmm"
NETWORK = BASE / "input/map/hainan/edges.shp"
UBODT = BASE / "input/map/hainan_ubodt_indexed.bin"
SUFFIX = "_fixed_en"

# All affected point-based datasets (relative to BASE)
DATASETS = [
    # sigma sweep (7)
    "data/simulation/sigma_01/no_occlusion/no_fault",
    "data/simulation/sigma_05/no_occlusion/no_fault",
    "data/simulation/sigma_10/no_occlusion/no_fault",
    "data/simulation/sigma_15/no_occlusion/no_fault",
    "data/simulation/sigma_20/no_occlusion/no_fault",
    "data/simulation/sigma_25/no_occlusion/no_fault",
    "data/simulation/sigma_30/no_occlusion/no_fault",
    # subsamples (6 rates x 3 sigma levels)
    "data/simulation/sigma_05/no_occlusion/no_fault/subsample_2s",
    "data/simulation/sigma_05/no_occlusion/no_fault/subsample_5s",
    "data/simulation/sigma_05/no_occlusion/no_fault/subsample_10s",
    "data/simulation/sigma_05/no_occlusion/no_fault/subsample_20s",
    "data/simulation/sigma_05/no_occlusion/no_fault/subsample_30s",
    "data/simulation/sigma_05/no_occlusion/no_fault/subsample_60s",
    "data/simulation/sigma_15/no_occlusion/no_fault/subsample_2s",
    "data/simulation/sigma_15/no_occlusion/no_fault/subsample_5s",
    "data/simulation/sigma_15/no_occlusion/no_fault/subsample_10s",
    "data/simulation/sigma_15/no_occlusion/no_fault/subsample_20s",
    "data/simulation/sigma_15/no_occlusion/no_fault/subsample_30s",
    "data/simulation/sigma_15/no_occlusion/no_fault/subsample_60s",
    "data/simulation/sigma_25/no_occlusion/no_fault/subsample_2s",
    "data/simulation/sigma_25/no_occlusion/no_fault/subsample_5s",
    "data/simulation/sigma_25/no_occlusion/no_fault/subsample_10s",
    "data/simulation/sigma_25/no_occlusion/no_fault/subsample_20s",
    "data/simulation/sigma_25/no_occlusion/no_fault/subsample_30s",
    "data/simulation/sigma_25/no_occlusion/no_fault/subsample_60s",
    # degraded (3 more; clean already in sigma_30/no_occlusion/no_fault)
    "data/simulation/sigma_30/no_occlusion/with_fault",
    "data/simulation/sigma_30/with_occlusion/no_fault",
    "data/simulation/sigma_30/with_occlusion/with_fault",
    # sigma mismatch (5)
    "data/simulation/sigma_mismatch/pr10_wls20",
    "data/simulation/sigma_mismatch/pr15_wls20",
    "data/simulation/sigma_mismatch/pr20_wls20",
    "data/simulation/sigma_mismatch/pr25_wls20",
    "data/simulation/sigma_mismatch/pr30_wls20",
]


def build_cmm_xml(gps_csv: str, mr_out: str) -> str:
    """Replicate exp3_full_matching.py build_cmm_xml exactly."""
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
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
  </parameters>
  <other><log_level>2</log_level><use_omp>true</use_omp><step>500</step>
    <convert_to_projected>false</convert_to_projected></other>
</config>"""
    return xml


def main():
    force = "--force" in sys.argv
    n_run, n_skip, n_fail, n_missing = 0, 0, 0, 0

    for rel in DATASETS:
        data_dir = BASE / rel
        obs_csv = data_dir / "observations.csv"
        out_file = data_dir / f"cmm_result{SUFFIX}.csv"

        if not obs_csv.exists():
            print(f"[MISSING] {rel}: observations.csv not found")
            n_missing += 1
            continue
        if out_file.exists() and not force:
            print(f"[SKIP]    {rel}: {out_file.name} exists")
            n_skip += 1
            continue

        xml_text = build_cmm_xml(str(obs_csv.resolve()), str(out_file.resolve()))
        tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
        tmp.write(xml_text)
        tmp.close()

        try:
            r = subprocess.run([str(CMM_BIN), tmp.name], capture_output=True,
                               text=True, cwd=str(BASE), timeout=600)
            if r.returncode == 0 and out_file.exists():
                n_lines = sum(1 for _ in open(out_file))
                print(f"[OK]      {rel}: {n_lines} lines")
                n_run += 1
            else:
                print(f"[FAIL]    {rel}: rc={r.returncode} {r.stderr[-200:]}")
                n_fail += 1
        except subprocess.TimeoutExpired:
            print(f"[FAIL]    {rel}: timeout")
            n_fail += 1
        finally:
            try:
                tmp.close()
                Path(tmp.name).unlink()
            except OSError:
                pass

    print(f"\nDone: {n_run} run, {n_skip} skipped, {n_fail} failed, {n_missing} missing input")


if __name__ == "__main__":
    main()
