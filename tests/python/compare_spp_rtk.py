#!/usr/bin/env python3
"""Compare extracted SPP positions (cmm_input_points.csv) against RTK ground truth."""
import csv, math, sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "dataset-hainan-06"

def dm_to_dd(dm_str):
    if not dm_str: return None
    try:
        val = float(dm_str); d = int(val / 100); m = val - d * 100
        return d + m / 60.0
    except: return None

def parse_rtk_nmea(filepath):
    """Return dict[unix_ts_int] = (lon, lat)."""
    result = {}
    current_date = None
    with open(filepath, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if line.startswith("#"): continue
            idx = line.find("$")
            if idx < 0: continue
            content = line[idx:].split("*")[0]
            fields = content.split(",")
            if len(fields) < 2: continue
            mt = fields[0][3:]
            if mt == "RMC" and len(fields) >= 10:
                try: current_date = date(int(fields[9][4:6])+2000, int(fields[9][2:4]), int(fields[9][0:2]))
                except: pass
            if mt == "GGA" and len(fields) >= 10:
                ts = fields[1]; lat_s = fields[2]; ns = fields[3]; lon_s = fields[4]; ew = fields[5]
                if not ts: continue
                lat = dm_to_dd(lat_s); lon = dm_to_dd(lon_s)
                if lat is None or lon is None or current_date is None: continue
                if ns == "S": lat = -lat
                if ew == "W": lon = -lon
                try:
                    h,m = int(ts[0:2]), int(ts[2:4]); s = float(ts[4:])
                    dt = datetime(current_date.year, current_date.month, current_date.day,
                                  h, m, int(s), int((s-int(s))*1e6), tzinfo=timezone.utc)
                    result[int(dt.timestamp())] = (lon, lat)
                except: pass
    return result

def haversine_m(lon1, lat1, lon2, lat2):
    mlat = math.radians((lat1+lat2)/2)
    dx = (lon1-lon2)*111320*math.cos(mlat)
    dy = (lat1-lat2)*111320
    return math.sqrt(dx*dx+dy*dy)

# Load RTK per trajectory
traj_rtk = {}
for d in sorted(DATA.glob("?.?")):
    tid = int(d.name.replace(".", ""))
    f = d / "实时定位结果" / "rtk_solution_clean.txt"
    if f.exists():
        rtk = parse_rtk_nmea(str(f))
        traj_rtk[tid] = rtk
        print(f"Traj {tid}: {len(rtk)} RTK epochs")

# Load SPP
spp_by_traj = defaultdict(list)
spp_path = DATA / "cmm_input_points.csv"
with open(spp_path, newline="") as f:
    for row in csv.DictReader(f, delimiter=";"):
        tid = int(row["id"])
        ts = float(row["timestamp"])
        lon = float(row["x"]); lat = float(row["y"])
        sde = float(row["sde"]); sdn = float(row["sdn"]); sdne = float(row["sdne"])
        spp_by_traj[tid].append({"ts": ts, "lon": lon, "lat": lat,
                                  "sde": sde, "sdn": sdn, "sdne": sdne})

# Match and compute errors
all_stats = []
all_errors = []

for tid in sorted(spp_by_traj.keys()):
    if tid not in traj_rtk:
        print(f"Traj {tid}: no RTK data")
        continue
    spp = spp_by_traj[tid]
    rtk = traj_rtk[tid]
    matched = 0
    errs = []
    for sp in spp:
        ts_int = int(sp["ts"])
        if ts_int not in rtk:
            continue
        rtk_lon, rtk_lat = rtk[ts_int]
        err = haversine_m(sp["lon"], sp["lat"], rtk_lon, rtk_lat)
        errs.append(err)
        all_errors.append({"traj": tid, "err_m": err, "sde_m": sp["sde"]*111320,
                           "sdn_m": sp["sdn"]*111320, "spp_lon": sp["lon"],
                           "spp_lat": sp["lat"], "rtk_lon": rtk_lon, "rtk_lat": rtk_lat})
        matched += 1
    if errs:
        e = np.array(errs)
        s = {"traj": tid, "n_spp": len(spp), "n_rtk": len(rtk), "n_match": matched,
             "mean": float(np.mean(e)), "median": float(np.median(e)),
             "rmse": float(np.sqrt(np.mean(e**2))),
             "p68": float(np.percentile(e, 68)), "p95": float(np.percentile(e, 95)),
             "max": float(np.max(e))}
        all_stats.append(s)
        print(f"Traj {tid}: {matched}/{len(spp)} matched, mean={s['mean']:.1f}m, "
              f"median={s['median']:.1f}m, P68={s['p68']:.1f}m, P95={s['p95']:.1f}m")

# Combined
combined = np.array([e["err_m"] for e in all_errors])
print(f"\nCombined ({len(combined)} epochs):")
print(f"  Mean={np.mean(combined):.1f}m, Median={np.median(combined):.1f}m, "
      f"RMSE={np.sqrt(np.mean(combined**2)):.1f}m")
print(f"  P68={np.percentile(combined,68):.1f}m, P95={np.percentile(combined,95):.1f}m, "
      f"Max={np.max(combined):.1f}m")

# Save CSVs
out = PROJECT_ROOT / "experiments/output/spp_error"
out.mkdir(parents=True, exist_ok=True)

with open(out / "spp_vs_rtk_stats.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["traj","n_spp","n_rtk","n_match","mean","median","rmse","p68","p95","max"])
    w.writeheader(); w.writerows(all_stats)

with open(out / "spp_vs_rtk_all.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["traj","err_m","sde_m","sdn_m","spp_lon","spp_lat","rtk_lon","rtk_lat"])
    w.writeheader(); w.writerows(all_errors)

print(f"\nOutput: {out}/spp_vs_rtk_stats.csv, {out}/spp_vs_rtk_all.csv")
# Restore backup
import shutil
shutil.copy(DATA / "cmm_input_points_rtk.bak", DATA / "cmm_input_points.csv")
print("Restored cmm_input_points.csv from RTK backup")
