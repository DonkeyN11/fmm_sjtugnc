#!/usr/bin/env python3
"""
Analyze SPP positioning error against RTK ground truth.

For each trajectory (1.1 -- 2.3):
  1. Parses RTK NMEA (rtk_solution_clean.txt) as ground truth
  2. Computes SPP WLS positions from RINEX 3.04 observations
  3. Time-aligns SPP and RTK by GPST (nearest-second matching)
  4. Computes horizontal error: mean, median, RMSE, P95, max, CEP50/CEP95
  5. Generates error histogram, CDF, time-series, scatter plots

Usage:
  python tests/python/analyze_spp_error.py --traj 1.1
  python tests/python/analyze_spp_error.py --all
  python tests/python/analyze_spp_error.py --all --output-dir experiments/output/spp_error
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 200
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})

# Add the script directory to path for compute_raim_pl imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compute_raim_pl import (
    parse_rinex3_obs, parse_rinex3_nav, compute_sat_position, compute_sat_clock,
    compute_geometry_matrix, compute_elevation_weights, wls_solve,
    lla_to_ecef, C, OMEGA_E, ELEV_MASK_DEG,
)

PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_BASE = PROJECT_ROOT / "dataset-hainan-06"

# Trajectory names
TRAJECTORIES = ["1.1", "1.2", "1.3", "1.4", "2.1", "2.2", "2.3"]


# ══════════════════════════════════════════════════════════════════════════════
# RTK NMEA parser
# ══════════════════════════════════════════════════════════════════════════════

def dm_to_dd(dm_str):
    if not dm_str: return None
    try:
        val = float(dm_str)
        degrees = int(val / 100)
        minutes = val - degrees * 100
        return degrees + minutes / 60.0
    except (ValueError, TypeError):
        return None


def parse_rtk_nmea(filepath: str) -> Dict[int, Tuple[float, float]]:
    """Parse RTK NMEA file.

    RTK uses Unix timestamps; RINEX uses GPS seconds-of-week (SOW).
    Both are converted to GPS SOW for matching.

    Returns dict[gps_sow_int] = (lon_deg, lat_deg).
    """
    result = {}
    current_date = None
    GPS_EPOCH_UNIX = 315964800  # 1980-01-06 00:00:00 UTC
    LEAP_SECONDS = 18  # as of June 2025

    with open(filepath, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if line.startswith("#"): continue
            idx = line.find("$")
            if idx < 0: continue
            content = line[idx:].split("*")[0]
            fields = content.split(",")
            if len(fields) < 2: continue

            msg_type = fields[0][3:]

            if msg_type == "RMC" and len(fields) >= 10:
                try:
                    d = int(fields[9][0:2])
                    m = int(fields[9][2:4])
                    y = int(fields[9][4:6]) + 2000
                    current_date = date(y, m, d)
                except (ValueError, IndexError):
                    pass

            if msg_type == "GGA" and len(fields) >= 10:
                time_str = fields[1]
                lat_dm = fields[2]; lat_ns = fields[3]
                lon_dm = fields[4]; lon_ew = fields[5]
                if not time_str: continue

                lat_dd = dm_to_dd(lat_dm)
                lon_dd = dm_to_dd(lon_dm)
                if lat_dd is None or lon_dd is None: continue
                if lat_ns == "S": lat_dd = -lat_dd
                if lon_ew == "W": lon_dd = -lon_dd

                if current_date is None: continue
                try:
                    hh = int(time_str[0:2])
                    mm = int(time_str[2:4])
                    ss = float(time_str[4:])
                    dt = datetime(current_date.year, current_date.month, current_date.day,
                                  hh, mm, int(ss), int((ss - int(ss)) * 1e6),
                                  tzinfo=timezone.utc)
                    unix_ts = dt.timestamp()
                    # Convert Unix ts to GPS seconds of week
                    gps_abs = unix_ts - GPS_EPOCH_UNIX + LEAP_SECONDS
                    sow_int = round(gps_abs % 604800)
                    result[sow_int] = (lon_dd, lat_dd)
                except (ValueError, IndexError):
                    pass

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SPP WLS positioning from RINEX
# ══════════════════════════════════════════════════════════════════════════════

PR_CODES = {
    "G": ["C1C", "C1W"],
    "C": ["C2I", "C2Q", "C1X"],
    "R": ["C1C", "C1P"],
    "E": ["C1C", "C1X"],
    "J": ["C1C", "C1X"],
}

_WGS84_A = 6378137.0
_WGS84_E2 = 0.00669437999014


def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """ECEF to WGS84 geodetic (lat_deg, lon_deg, alt_m)."""
    lon = math.degrees(math.atan2(y, x))
    p = math.sqrt(x * x + y * y)
    lat = math.degrees(math.atan2(z, p * (1.0 - _WGS84_E2)))
    for _ in range(5):
        sin_lat = math.sin(math.radians(lat))
        N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        h = p / math.cos(math.radians(lat)) - N
        lat = math.degrees(math.atan2(z, p * (1.0 - _WGS84_E2 * N / (N + h))))
    return lat, lon, 0.0


def compute_spp_positions(traj_name: str) -> List[Dict]:
    """Compute SPP WLS positions from RINEX data.

    Returns list of dicts with: gpst, lon, lat, sde, sdn, sdne, n_sats, sigma0, hpl_m
    """
    spath = DATA_BASE / traj_name / "GNSS原始观测" / "SPP"
    if not spath.is_dir():
        print(f"  SPP dir not found: {spath}")
        return []

    obs_files = [f for f in os.listdir(spath) if f.endswith(".25O") or f.endswith(".obs")]
    nav_g = [f for f in os.listdir(spath) if f.endswith(".25N") or f.endswith(".25P")]
    nav_c = [f for f in os.listdir(spath) if f.endswith(".25C")]
    nav_g_gps = [f for f in os.listdir(spath) if f.endswith(".25G")]

    if not obs_files:
        print(f"  No RINEX obs file in {spath}")
        return []

    obs_file = str(spath / obs_files[0])
    nav_files = []
    for fl in [nav_g, nav_c, nav_g_gps]:
        for fn in fl:
            nav_files.append(str(spath / fn))

    if not nav_files:
        print(f"  No nav files in {spath}")
        return []

    # Parse RINEX
    header, epochs = parse_rinex3_obs(obs_file)
    all_ephs = []
    for nf in nav_files:
        all_ephs.extend(parse_rinex3_nav(nf))

    eph_by_prn: Dict[str, list] = defaultdict(list)
    for eph in all_ephs:
        eph_by_prn[eph.prn].append(eph)

    # Approximate position
    if header.approx_pos is not None:
        approx_ecef = header.approx_pos
    else:
        approx_ecef = lla_to_ecef(19.96, 110.48, 10.0)

    x, y, z = approx_ecef[0], approx_ecef[1], approx_ecef[2]
    lon = math.degrees(math.atan2(y, x))
    p = math.sqrt(x * x + y * y)
    lat = math.degrees(math.atan2(z, p * (1.0 - _WGS84_E2)))
    ref_lla = (lat, lon, 10.0)

    results = []
    for epoch in epochs:
        sat_positions, sat_prng, sys_chars = [], [], []

        for prn, obs_data in epoch.sat_data.items():
            sys = prn[0]
            codes = PR_CODES.get(sys, ["C1C"])
            pr_val = None
            for code in codes:
                if code in obs_data:
                    pr_val = obs_data[code]
                    break
            if pr_val is None: continue

            eph_list = eph_by_prn.get(prn, [])
            if not eph_list: continue
            best_eph = min(eph_list, key=lambda e: abs(epoch.gpst - e.toe_gpst))
            max_age = 7200 if sys == "G" else 3600
            if abs(epoch.gpst - best_eph.toe_gpst) > max_age: continue

            sat_pos = compute_sat_position(best_eph, epoch.gpst)
            if sat_pos is None: continue
            dt_sv, d_rel = compute_sat_clock(best_eph, epoch.gpst)
            pr_corrected = pr_val - (dt_sv + d_rel) * C

            # Earth rotation
            rot = OMEGA_E * np.linalg.norm(approx_ecef - sat_pos) / C
            cr, sr = math.cos(rot), math.sin(rot)
            sat_rot = np.array([sat_pos[0] * cr + sat_pos[1] * sr,
                                -sat_pos[0] * sr + sat_pos[1] * cr,
                                sat_pos[2]])
            sat_positions.append(sat_rot)
            sat_prng.append(pr_corrected)
            sys_chars.append(sys)

        n_sats = len(sat_positions)
        if n_sats < 5: continue

        H = compute_geometry_matrix(sat_positions, approx_ecef, sys_chars)
        W, valid_idx = compute_elevation_weights(sat_positions, approx_ecef, ref_lla)
        if len(valid_idx) < 5: continue

        predicted = np.array([np.linalg.norm(sat_positions[i] - approx_ecef)
                              for i in range(n_sats)])
        pr_residuals = np.array(sat_prng) - predicted

        dx, cov, sigma0 = wls_solve(H, W, pr_residuals, valid_idx)
        if dx is None:
            continue

        # ECEF position
        pos_ecef = approx_ecef + dx[:3]
        lat_deg, lon_deg, _ = ecef_to_lla(pos_ecef[0], pos_ecef[1], pos_ecef[2])

        # Covariance in ENU
        if cov is not None and cov.shape[0] >= 3:
            # Convert ECEF covariance to ENU
            sin_lat = math.sin(math.radians(lat_deg))
            cos_lat = math.cos(math.radians(lat_deg))
            sin_lon = math.sin(math.radians(lon_deg))
            cos_lon = math.cos(math.radians(lon_deg))
            R = np.array([
                [-sin_lon, cos_lon, 0],
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
            ])
            cov_enu = R @ cov[:3, :3] @ R.T
            sde = math.sqrt(max(cov_enu[0, 0], 0))
            sdn = math.sqrt(max(cov_enu[1, 1], 0))
            sdne = cov_enu[0, 1]
        else:
            sde = sdn = sdne = 5.0

        # HPL (simplified: K * sigma_major)
        tr = cov_enu[0, 0] + cov_enu[1, 1]
        det = cov_enu[0, 0] * cov_enu[1, 1] - cov_enu[0, 1] ** 2
        sigma_major = math.sqrt(max((tr + math.sqrt(max(tr**2 - 4*det, 0))) / 2.0, 0))
        hpl = 6.0 * sigma_major

        gpst_int = round(epoch.gpst)
        results.append({
            "sow": gpst_int,
            "lon": lon_deg,
            "lat": lat_deg,
            "sde": sde,
            "sdn": sdn,
            "sdne": sdne,
            "n_sats": len(valid_idx),
            "sigma0": sigma0,
            "hpl_m": hpl,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Error computation
# ══════════════════════════════════════════════════════════════════════════════

def haversine_m(lon1, lat1, lon2, lat2):
    mlat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon1 - lon2) * 111320.0 * math.cos(mlat)
    dy = (lat1 - lat2) * 111320.0
    return math.sqrt(dx * dx + dy * dy)


def analyze_trajectory(traj_name: str) -> Optional[Dict]:
    print(f"\n{'='*60}")
    print(f"  Trajectory: {traj_name}")
    print(f"{'='*60}")

    # Load RTK ground truth
    rtk_path = DATA_BASE / traj_name / "实时定位结果" / "rtk_solution_clean.txt"
    if not rtk_path.exists():
        print(f"  RTK file not found: {rtk_path}")
        return None
    rtk = parse_rtk_nmea(str(rtk_path))
    print(f"  RTK epochs: {len(rtk)}")

    # Compute SPP positions
    spp = compute_spp_positions(traj_name)
    print(f"  SPP epochs: {len(spp)}")

    # Time-align by GPS seconds of week
    errors = []
    matched = 0
    unmatched = 0
    for sp in spp:
        sow = sp["sow"]
        if sow not in rtk:
            unmatched += 1
            continue
        rtk_lon, rtk_lat = rtk[sow]
        err_m = haversine_m(sp["lon"], sp["lat"], rtk_lon, rtk_lat)
        errors.append({
            "sow": sow,
            "err_m": err_m,
            "spp_lon": sp["lon"],
            "spp_lat": sp["lat"],
            "rtk_lon": rtk_lon,
            "rtk_lat": rtk_lat,
            "sde": sp["sde"],
            "sdn": sp["sdn"],
            "sdne": sp["sdne"],
            "n_sats": sp["n_sats"],
            "hpl_m": sp["hpl_m"],
        })
        matched += 1

    print(f"  Matched: {matched}, Unmatched SPP: {unmatched}")

    if matched == 0:
        print(f"  WARNING: No matched epochs!")
        return None

    err_vals = np.array([e["err_m"] for e in errors])
    stats = {
        "traj": traj_name,
        "n_rtk": len(rtk),
        "n_spp": len(spp),
        "n_matched": matched,
        "mean": float(np.mean(err_vals)),
        "median": float(np.median(err_vals)),
        "rmse": float(np.sqrt(np.mean(err_vals ** 2))),
        "p68": float(np.percentile(err_vals, 68)),
        "p95": float(np.percentile(err_vals, 95)),
        "max": float(np.max(err_vals)),
        "std": float(np.std(err_vals)),
    }
    print(f"  Error: mean={stats['mean']:.1f}m, median={stats['median']:.1f}m, "
          f"RMSE={stats['rmse']:.1f}m, P95={stats['p95']:.1f}m, max={stats['max']:.1f}m")

    return {"stats": stats, "errors": errors}


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(all_results: List[Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-trajectory figures ──
    for res in all_results:
        traj = res["stats"]["traj"]
        errs = np.array([e["err_m"] for e in res["errors"]])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Error time-series
        ax = axes[0]
        gpsts = [e["sow"] - res["errors"][0]["sow"] for e in res["errors"]]
        ax.plot(gpsts, errs, lw=0.3, color="#2166ac", alpha=0.8)
        ax.axhline(res["stats"]["mean"], color="red", lw=0.8, ls="--", label=f"mean={res['stats']['mean']:.1f}m")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Horizontal error (m)")
        ax.set_title(f"Traj {traj}: Error Time Series"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_ylim(0, max(100, res["stats"]["p95"] * 1.5))

        # Error histogram + CDF
        ax = axes[1]
        ax.hist(errs, bins=50, density=True, alpha=0.6, color="#2166ac", edgecolor="white", lw=0.3)
        ax.axvline(res["stats"]["median"], color="red", lw=1, ls="--", label=f"median={res['stats']['median']:.1f}")
        ax.axvline(res["stats"]["p95"], color="orange", lw=1, ls=":", label=f"P95={res['stats']['p95']:.1f}")
        ax.set_xlabel("Error (m)"); ax.set_ylabel("Density")
        ax.set_title(f"Traj {traj}: Error Distribution"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlim(0, max(100, res["stats"]["p95"] * 1.2))

        # Error vs SPP std (sde)
        ax = axes[2]
        sdes = [e["sde"] * 111320 for e in res["errors"]]  # deg to m approx
        ax.scatter(sdes, errs, s=2, alpha=0.3, color="#2166ac")
        ax.plot([0, max(sdes)*1.1], [0, max(sdes)*1.1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("SPP sde (m)"); ax.set_ylabel("Horiz. error (m)")
        ax.set_title(f"Traj {traj}: Error vs σ_E"); ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"spp_error_traj{traj.replace('.','_')}.png", dpi=DPI)
        plt.close(fig)

    # ── Combined summary figure ──
    all_errs = np.concatenate([np.array([e["err_m"] for e in r["errors"]]) for r in all_results])
    traj_labels = [r["stats"]["traj"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Combined error histogram
    ax = axes[0, 0]
    ax.hist(all_errs, bins=80, density=True, alpha=0.6, color="#2166ac", edgecolor="white", lw=0.3)
    ax.axvline(np.median(all_errs), color="red", lw=1.5, ls="--", label=f"median={np.median(all_errs):.1f}m")
    ax.axvline(np.percentile(all_errs, 95), color="orange", lw=1.5, ls=":",
               label=f"P95={np.percentile(all_errs,95):.1f}m")
    ax.set_xlabel("Horizontal error (m)"); ax.set_ylabel("Density")
    ax.set_title(f"All Trajectories: Error Distribution (n={len(all_errs)})")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, np.percentile(all_errs, 99))

    # CDF
    ax = axes[0, 1]
    sorted_err = np.sort(all_errs)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax.semilogx(sorted_err, cdf, lw=1.5, color="#2166ac")
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.axhline(0.95, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Error (m)"); ax.set_ylabel("CDF")
    ax.set_title("Cumulative Distribution"); ax.grid(alpha=0.3, which="both")
    ax.set_xlim(0.5, sorted_err[-1])

    # Per-trajectory box plot
    ax = axes[1, 0]
    traj_data = [np.array([e["err_m"] for e in r["errors"]]) for r in all_results]
    bp = ax.boxplot(traj_data, labels=traj_labels, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#92c5de")
        patch.set_alpha(0.7)
    ax.set_xlabel("Trajectory"); ax.set_ylabel("Error (m)")
    ax.set_title("Per-Trajectory Error Distribution"); ax.grid(alpha=0.3, axis="y")

    # Mean error bar chart
    ax = axes[1, 1]
    means = [r["stats"]["mean"] for r in all_results]
    medians = [r["stats"]["median"] for r in all_results]
    x = np.arange(len(traj_labels))
    w = 0.35
    ax.bar(x - w/2, means, w, color="#2166ac", label="Mean", edgecolor="white", lw=0.5)
    ax.bar(x + w/2, medians, w, color="#b2182b", label="Median", edgecolor="white", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(traj_labels)
    ax.set_ylabel("Error (m)"); ax.set_title("Mean & Median Error per Trajectory")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("SPP vs RTK Positioning Error Analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "spp_error_summary.png", dpi=DPI)
    plt.close(fig)

    # ── Write summary CSV ──
    csv_path = output_dir / "spp_error_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["traj", "n_rtk", "n_spp", "n_matched",
                                          "mean", "median", "rmse", "p68", "p95", "max", "std"])
        w.writeheader()
        for r in all_results:
            w.writerow(r["stats"])
    print(f"\n  Summary CSV: {csv_path}")

    # ── Combined CMM-format result ──
    # Write all SPP points + RTK coords for future CMM input
    all_points = []
    for res in all_results:
        traj_id = int(res["stats"]["traj"].replace(".", ""))
        for e in res["errors"]:
            all_points.append({
                "id": traj_id,
                "timestamp": e["sow"],
                "x": e["spp_lon"],
                "y": e["spp_lat"],
                "rtk_x": e["rtk_lon"],
                "rtk_y": e["rtk_lat"],
                "sde": e.get("sde", 0),
                "sdn": e.get("sdn", 0),
                "sdne": e.get("sdne", 0),
                "n_sats": e.get("n_sats", 0),
                "hpl_m": e.get("hpl_m", 0),
            })

    all_path = output_dir / "spp_positions_all.csv"
    with open(all_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "timestamp", "x", "y", "rtk_x", "rtk_y",
                                          "sde", "sdn", "sdne", "n_sats", "hpl_m"])
        w.writeheader()
        w.writerows(all_points)
    print(f"  All SPP positions: {all_path} ({len(all_points)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze SPP positioning error vs RTK")
    parser.add_argument("--traj", type=str, default=None, help="Single trajectory (e.g. 1.1)")
    parser.add_argument("--all", action="store_true", help="Process all trajectories")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/output/spp_error"),
                        help="Output directory for plots and CSVs")
    args = parser.parse_args()

    project_root = SCRIPT_DIR.parents[1]
    args.output_dir = project_root / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.traj:
        trajs = [args.traj]
    elif args.all:
        trajs = TRAJECTORIES
    else:
        trajs = ["1.1"]

    all_results = []
    for traj in trajs:
        res = analyze_trajectory(traj)
        if res:
            all_results.append(res)

    if not all_results:
        print("No valid results.")
        return

    # Combined stats
    all_errs = np.concatenate([np.array([e["err_m"] for e in r["errors"]]) for r in all_results])
    print(f"\n{'='*60}")
    print(f"  Combined ({len(all_results)} trajectories, {len(all_errs)} epochs)")
    print(f"  Mean={np.mean(all_errs):.1f}m, Median={np.median(all_errs):.1f}m, "
          f"RMSE={np.sqrt(np.mean(all_errs**2)):.1f}m, "
          f"P68={np.percentile(all_errs,68):.1f}m, P95={np.percentile(all_errs,95):.1f}m")

    plot_all(all_results, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
