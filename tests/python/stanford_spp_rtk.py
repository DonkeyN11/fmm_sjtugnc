#!/usr/bin/env python3
"""
Stanford plot: SPP horizontal error vs RAIM Protection Level from cmm_input_points.csv.

The SPP error is computed against RTK ground truth (rtk_solution_clean.txt).
The PL is from cmm_input_points.csv (RAIM-computed, merged by merge_raim_pl.py).
PL is in degrees → converted to meters.

Usage:
  python tests/python/stanford_spp_rtk.py
"""

import csv, math
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 300
plt.rcParams.update({"font.size":7,"axes.labelsize":8,"axes.titlesize":9,
    "legend.fontsize":6,"xtick.labelsize":6,"ytick.labelsize":6,
    "figure.dpi":DPI,"savefig.dpi":DPI,"savefig.bbox":"tight"})

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "dataset-hainan-06"
PAPER_FIGS = PROJECT_ROOT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
OUT_DIR = PROJECT_ROOT / "experiments/output/spp_error"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAJECTORIES = {"1.1": 11, "1.2": 12, "1.3": 13, "1.4": 14, "2.1": 21, "2.2": 22, "2.3": 23}

def dm_to_dd(dm_str):
    if not dm_str: return None
    try: val=float(dm_str); d=int(val/100); m=val-d*100; return d+m/60.0
    except: return None

def parse_rtk_nmea(filepath):
    result = {}
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
                try: current_date = date(int(fields[9][4:6])+2000, int(fields[9][2:4]), int(fields[9][0:2]))
                except: pass
            if mt == "GGA" and len(fields) >= 10 and current_date:
                ts = fields[1]; lat=dm_to_dd(fields[2]); ns=fields[3]; lon=dm_to_dd(fields[4]); ew=fields[5]
                if not ts or lat is None or lon is None: continue
                if ns=="S": lat=-lat
                if ew=="W": lon=-lon
                try:
                    h,m=int(ts[0:2]),int(ts[2:4]); s=float(ts[4:])
                    dt=datetime(current_date.year,current_date.month,current_date.day,
                                h,m,int(s),int((s-int(s))*1e6),tzinfo=timezone.utc)
                    result[round(dt.timestamp())]=(lon,lat)
                except: pass
    return result

def haversine_m(lon1, lat1, lon2, lat2):
    mlat=math.radians((lat1+lat2)/2)
    dx=(lon1-lon2)*111320*math.cos(mlat)
    dy=(lat1-lat2)*111320
    return math.sqrt(dx*dx+dy*dy)

# Load SPP data with PL
spp_data = defaultdict(list)  # traj_id -> [(ts, lon, lat, pl_deg)]
with open(DATA / "cmm_input_points.csv", newline="") as f:
    for row in csv.DictReader(f, delimiter=";"):
        tid = int(row["id"])
        ts = float(row["timestamp"])
        lon = float(row["x"]); lat = float(row["y"])
        pl_deg = float(row["protection_level"])
        spp_data[tid].append((ts, lon, lat, pl_deg))

print(f"Loaded SPP+PL for {len(spp_data)} trajectories")

# Load RTK and compute errors
all_errs = []
all_pls = []
per_traj = defaultdict(lambda: {"errs": [], "pls": []})

for traj_dir, tid in TRAJECTORIES.items():
    rtk_file = DATA / traj_dir / "实时定位结果" / "rtk_solution_clean.txt"
    if not rtk_file.exists(): continue
    rtk = parse_rtk_nmea(str(rtk_file))
    spp_list = spp_data[tid]
    matched = 0
    for ts, lon, lat, pl_deg in spp_list:
        ts_int = int(ts)
        if ts_int not in rtk: continue
        rtk_lon, rtk_lat = rtk[ts_int]
        err = haversine_m(lon, lat, rtk_lon, rtk_lat)
        # Convert PL from degrees to meters (use mean lat)
        mlat = math.radians((lat + rtk_lat) / 2)
        pl_m = pl_deg * 111320.0 * math.cos(mlat)
        all_errs.append(err); all_pls.append(pl_m)
        per_traj[tid]["errs"].append(err); per_traj[tid]["pls"].append(pl_m)
        matched += 1
    print(f"Traj {tid}: {matched} matched")

all_errs = np.array(all_errs); all_pls = np.array(all_pls)
hmi = all_errs > all_pls
n_hmi = np.sum(hmi); n_total = len(all_errs)
p_md = n_hmi / n_total

# ── Combined Stanford plot ──
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ok = ~hmi
ax.scatter(all_pls[ok], all_errs[ok], s=0.5, alpha=0.3, color="#2166ac",
           label=f"OK (error ≤ PL, n={n_total-n_hmi})")
ax.scatter(all_pls[hmi], all_errs[hmi], s=8, alpha=0.8, color="red", marker="x",
           label=f"HMI (error > PL, n={n_hmi})")
mx = max(np.percentile(all_pls, 99.5), np.percentile(all_errs, 99.5)) * 1.05
ax.plot([0, mx], [0, mx], "k--", lw=0.8, alpha=0.5)
ax.fill_between([0, mx], [0, mx], mx, alpha=0.06, color="red")
ax.set_xlim(0, mx); ax.set_ylim(0, mx)

# Stats
stats_text = (f"n = {n_total} epochs (7 trajectories)\n"
              f"P_md = {p_md:.4f} ({n_hmi} HMI)\n"
              f"PL: mean={np.mean(all_pls):.1f}m, median={np.median(all_pls):.1f}m\n"
              f"Error: mean={np.mean(all_errs):.1f}m, P95={np.percentile(all_errs,95):.1f}m\n"
              f"Error > PL: {p_md*100:.2f}%")
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=7,
        fontfamily="monospace", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
ax.set_xlabel("Protection Level (m)"); ax.set_ylabel("Horizontal Error (m)")
ax.set_title("Stanford Diagram: SPP Error vs RAIM Protection Level\n(7 trajectories, Hainan-06 dataset)", fontweight="bold")
ax.legend(fontsize=6, loc="lower right"); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "stanford_spp_rtk.png", dpi=DPI)
fig.savefig(PAPER_FIGS / "stanford_spp_rtk.png", dpi=DPI)
plt.close(fig)
print(f"\nSaved: {OUT_DIR}/stanford_spp_rtk.png")
print(f"Saved: {PAPER_FIGS}/stanford_spp_rtk.png")

# ── Per-trajectory subplots ──
order = [11,12,13,14,21,22,23]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for idx, tid in enumerate(order):
    ax = axes.flat[idx]
    e = np.array(per_traj[tid]["errs"]); p = np.array(per_traj[tid]["pls"])
    if len(e) == 0: ax.axis("off"); continue
    h = e > p; n_h = np.sum(h)
    ax.scatter(p[~h], e[~h], s=1, alpha=0.3, color="#2166ac")
    ax.scatter(p[h], e[h], s=10, alpha=0.8, color="red", marker="x")
    mx_t = max(np.percentile(p,99), np.percentile(e,99))*1.05
    ax.plot([0,mx_t],[0,mx_t],"k--",lw=0.8,alpha=0.5)
    ax.fill_between([0,mx_t],[0,mx_t],mx_t,alpha=0.06,color="red")
    ax.set_xlim(0,mx_t); ax.set_ylim(0,mx_t)
    ax.set_title(f"Traj {tid} (n={len(e)}, P_md={n_h/len(e):.4f})")
    ax.set_xlabel("PL (m)"); ax.set_ylabel("Error (m)"); ax.grid(alpha=0.3)
axes.flat[7].axis("off")
fig.suptitle("Stanford Diagrams per Trajectory: SPP Error vs RAIM PL", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "stanford_spp_rtk_per_traj.png", dpi=DPI)
plt.close(fig)
print(f"Saved: {OUT_DIR}/stanford_spp_rtk_per_traj.png")

# Print per-trajectory table
print("\nPer-Trajectory Summary:")
print(f"{'Traj':>5} {'n':>6} {'P_md':>8} {'PL_mean':>8} {'PL_med':>8} {'Err_mean':>9} {'Err_P95':>8}")
for tid in order:
    e=np.array(per_traj[tid]["errs"]); p=np.array(per_traj[tid]["pls"])
    if len(e)==0: continue
    print(f"{tid:>5} {len(e):>6} {np.sum(e>p)/len(e):>8.4f} {np.mean(p):>8.1f} "
          f"{np.median(p):>8.1f} {np.mean(e):>9.1f} {np.percentile(e,95):>8.1f}")
