#!/usr/bin/env python3
"""
Build ground truth points CSV for real_data from RTK NMEA solutions.

Aligns by first-matching-timestamp offset, then pairs SPP and RTK by seq position.
Both are 1 Hz on the same vehicle — only the start times differ.

Output: experiments/data/real_data/ground_truth_points.csv
Format: id; seq; timestamp; x; y

Usage:
  python experiments/scripts/build_real_data_gt.py
"""

import csv
from pathlib import Path
from datetime import date, datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/real_data"
RTK_BASE = PROJECT_ROOT.parent / "dataset-hainan-06"

TRAJ_MAP = {11: "1.1", 12: "1.2", 13: "1.3", 14: "1.4",
             21: "2.1", 22: "2.2", 23: "2.3"}


def dm_to_dd(dm_str):
    if not dm_str: return None
    try:
        val = float(dm_str); d = int(val / 100); m = val - d * 100
        return d + m / 60.0
    except: return None


def parse_rtk_nmea_ordered(filepath):
    """Two-pass RTK NMEA parser.

    In the RTK file, $GNGGA (position) appears BEFORE $GNRMC (date) at the same
    epoch (e.g., GGA at line 7, RMC at line 8, both at 041632). A single-pass
    parser skips the first GGA because current_date is still None.

    Fix: Pass 1 collects (time_str, date) from RMC. Pass 2 re-reads GGA and
    pairs each position with its date by matching time_str.
    """
    # ── Pass 1: collect dates from RMC (also records RMC's own time_str) ──
    time_to_date = {}
    current_date = None
    with open(filepath, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if line.startswith("#"): continue
            idx = line.find("$")
            if idx < 0: continue
            content = line[idx:].split("*")[0]; fields = content.split(",")
            if len(fields) < 2: continue
            mt = fields[0][3:]
            if mt == "RMC" and len(fields) >= 10:
                try:
                    d = date(int(fields[9][4:6]) + 2000,
                             int(fields[9][2:4]), int(fields[9][0:2]))
                    current_date = d
                except: pass
                # RMC carries time + date — self-register
                if len(fields) >= 2 and current_date:
                    time_to_date[fields[1]] = current_date
            elif mt == "GGA" and len(fields) >= 2:
                time_str = fields[1]
                if time_str and current_date:
                    time_to_date[time_str] = current_date

    # ── Pass 2: extract positions from GGA ──
    timestamps, lons, lats = [], [], []
    with open(filepath, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if line.startswith("#"): continue
            idx = line.find("$")
            if idx < 0: continue
            content = line[idx:].split("*")[0]; fields = content.split(",")
            if len(fields) < 2: continue
            mt = fields[0][3:]
            if mt != "GGA" or len(fields) < 10:
                continue
            time_str = fields[1]
            if time_str not in time_to_date:
                continue
            lat = dm_to_dd(fields[2]); ns = fields[3]
            lon = dm_to_dd(fields[4]); ew = fields[5]
            if not time_str or lat is None or lon is None:
                continue
            if ns == "S": lat = -lat
            if ew == "W": lon = -lon
            d_obj = time_to_date[time_str]
            try:
                h, m = int(time_str[0:2]), int(time_str[2:4]); s = float(time_str[4:])
                dt = datetime(d_obj.year, d_obj.month, d_obj.day,
                              h, m, int(s), int((s - int(s)) * 1e6), tzinfo=timezone.utc)
                timestamps.append(round(dt.timestamp()))
                lons.append(lon)
                lats.append(lat)
            except: pass
    return timestamps, lons, lats


def main():
    # ── Load RTK as ordered lists ──
    rtk_data = {}
    for tid, tdir in TRAJ_MAP.items():
        f = RTK_BASE / tdir / "实时定位结果" / "rtk_solution_clean.txt"
        if f.exists():
            rtk_data[tid] = parse_rtk_nmea_ordered(str(f))
            print(f"Traj {tid}: {len(rtk_data[tid][0])} RTK epochs")

    # ── Load SPP as ordered lists per trajectory ──
    spp_data = {}
    obs_file = DATA_DIR / "cmm_input_points.csv"
    with open(obs_file, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = int(row["id"])
            spp_data.setdefault(tid, []).append({
                "ts": round(float(row["timestamp"])),
                "x": row["x"], "y": row["y"],
            })

    # ── Align by seq offset ──
    gt_rows = []
    for tid in sorted(rtk_data.keys()):
        if tid not in spp_data:
            continue
        rtk_ts, rtk_lons, rtk_lats = rtk_data[tid]
        rtk_set = set(rtk_ts)
        spp_list = spp_data[tid]

        # Find offset: first SPP timestamp that exists in RTK
        offset = None
        for i, sp in enumerate(spp_list):
            if sp["ts"] in rtk_set:
                offset = i
                break

        if offset is None:
            print(f"Traj {tid}: no time overlap, skipping")
            continue

        # Pair by seq: SPP[offset + k] ↔ RTK[k]
        matched = 0
        for k in range(len(rtk_ts)):
            spp_idx = offset + k
            if spp_idx >= len(spp_list):
                break
            gt_rows.append({
                "id": str(tid),
                "seq": str(k),
                "timestamp": spp_list[spp_idx]["ts"],
                "x": str(rtk_lons[k]),
                "y": str(rtk_lats[k]),
            })
            matched += 1

        print(f"Traj {tid}: offset={offset}, matched={matched} "
              f"(SPP range [{offset}, {offset+matched}), RTK range [0, {matched}))")

    # ── Write ──
    out = DATA_DIR / "ground_truth_points.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "seq", "timestamp", "x", "y"],
                           delimiter=";")
        w.writeheader()
        w.writerows(gt_rows)
    print(f"\nSaved: {out} ({len(gt_rows)} rows)")


if __name__ == "__main__":
    main()
