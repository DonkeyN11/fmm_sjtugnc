#!/usr/bin/env python3
"""
Analyze SPP positioning error against RTK ground truth.

For each trajectory (1.1 -- 2.3):
  1. Parse RTK NMEA (rtk_solution_clean.txt) as ground truth (lon, lat)
  2. Compute SPP WLS positions from RINEX 3.04 observations
     (reuses the exact WLS pipeline from compute_raim_pl.py _compute_epoch_pl)
  3. Time-align SPP and RTK by GPS seconds-of-week
  4. Compute horizontal error: mean, median, RMSE, P68, P95, max
  5. Generate per-trajectory + combined summary plots

Usage:
  python tests/python/analyze_spp_error.py --traj 1.1
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
from scipy.stats import chi2, ncx2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 200
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})

# ── Import the battle-tested RINEX/WLS/RAIM functions ─────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compute_raim_pl import (
    parse_rinex3_obs, parse_rinex3_nav,
    compute_sat_position, compute_sat_clock,
    compute_geometry_matrix, compute_elevation_weights, wls_solve,
    lla_to_ecef, ecef_to_enu,
    C, OMEGA_E,
    P_FA, P_MD, ELEV_MASK_DEG,
    GPS_PSEUDORANGE_CODES, BDS_PSEUDORANGE_CODES,
    GLO_PSEUDORANGE_CODES, GAL_PSEUDORANGE_CODES, QZS_PSEUDORANGE_CODES,
    ObsHeader, EpochObs, Ephemeris,
)

PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_BASE = PROJECT_ROOT / "dataset-hainan-06"
TRAJECTORIES = ["1.1", "1.2", "1.3", "1.4", "2.1", "2.2", "2.3"]

_WGS84_A = 6378137.0
_WGS84_E2 = 0.00669437999014

PR_CODES = {
    "G": GPS_PSEUDORANGE_CODES,
    "C": BDS_PSEUDORANGE_CODES,
    "R": GLO_PSEUDORANGE_CODES,
    "E": GAL_PSEUDORANGE_CODES,
    "J": QZS_PSEUDORANGE_CODES,
}


# ══════════════════════════════════════════════════════════════════════════════
# RTK NMEA Parser  (GPS seconds-of-week keys)
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
    """Parse RTK NMEA to dict[gps_sow_int] = (lon_deg, lat_deg)."""
    result = {}
    current_date = None
    GPS_EPOCH_UNIX = 315964800
    LEAP_SECONDS = 18

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
                    d_obj = date(int(fields[9][4:6]) + 2000, int(fields[9][2:4]),
                                 int(fields[9][0:2]))
                    current_date = d_obj
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
                    hh = int(time_str[0:2]); mm = int(time_str[2:4])
                    ss = float(time_str[4:])
                    dt = datetime(current_date.year, current_date.month,
                                  current_date.day, hh, mm, int(ss),
                                  int((ss - int(ss)) * 1e6), tzinfo=timezone.utc)
                    unix_ts = dt.timestamp()
                    gps_abs = unix_ts - GPS_EPOCH_UNIX + LEAP_SECONDS
                    sow_int = round(gps_abs % 604800)
                    result[sow_int] = (lon_dd, lat_dd)
                except (ValueError, IndexError):
                    pass
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SPP position computation  (exact replica of _compute_epoch_pl + position)
# ══════════════════════════════════════════════════════════════════════════════

def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """ECEF (m) → WGS84 geodetic (lat_deg, lon_deg, alt_m)."""
    lon = math.degrees(math.atan2(y, x))
    p = math.sqrt(x * x + y * y)
    lat = math.degrees(math.atan2(z, p * (1.0 - _WGS84_E2)))
    for _ in range(5):
        sin_lat = math.sin(math.radians(lat))
        N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        h = p / math.cos(math.radians(lat)) - N
        lat = math.degrees(math.atan2(z, p * (1.0 - _WGS84_E2 * N / (N + h))))
    return lat, lon, 0.0


def cov_ecef_to_enu(cov_3x3: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rotate 3×3 ECEF covariance to ENU."""
    sin_lat = math.sin(math.radians(lat_deg))
    cos_lat = math.cos(math.radians(lat_deg))
    sin_lon = math.sin(math.radians(lon_deg))
    cos_lon = math.cos(math.radians(lon_deg))
    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    ])
    return R @ cov_3x3 @ R.T


def compute_epoch_spp(epoch: EpochObs,
                      eph_by_prn: Dict[str, List[Ephemeris]],
                      approx_ecef: np.ndarray,
                      ref_lla: Tuple[float, float, float],
                      ) -> Optional[Dict]:
    """Identical satellite selection + WLS as _compute_epoch_pl, but returns position.

    Returns dict with: sow, lon, lat, sde_m, sdn_m, sdne_m, n_sats, sigma0, hpl_m
    """
    sat_positions: List[np.ndarray] = []
    sat_prng: List[float] = []
    sys_chars: List[str] = []

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
        max_age_s = 7200 if sys == 'G' else 3600
        if abs(epoch.gpst - best_eph.toe_gpst) > max_age_s: continue

        sat_pos = compute_sat_position(best_eph, epoch.gpst)
        if sat_pos is None: continue
        dt_sv, d_rel = compute_sat_clock(best_eph, epoch.gpst)
        pr_corrected = pr_val - (dt_sv + d_rel) * C

        rot_angle = OMEGA_E * np.linalg.norm(approx_ecef - sat_pos) / C
        cos_ra, sin_ra = math.cos(rot_angle), math.sin(rot_angle)
        sat_rot = np.array([
            sat_pos[0] * cos_ra + sat_pos[1] * sin_ra,
            -sat_pos[0] * sin_ra + sat_pos[1] * cos_ra,
            sat_pos[2]
        ])
        sat_positions.append(sat_rot)
        sat_prng.append(pr_corrected)
        sys_chars.append(sys)

    n_sats = len(sat_positions)
    n_state = 4 + sum(1 for sc in set(sys_chars) if sc != 'G')
    if n_sats < n_state + 1:
        return None

    H = compute_geometry_matrix(sat_positions, approx_ecef, sys_chars)
    W, valid_idx = compute_elevation_weights(sat_positions, approx_ecef, ref_lla)
    if len(valid_idx) < 5:
        return None

    predicted_ranges = np.array([
        np.linalg.norm(sat_positions[i] - approx_ecef) for i in range(n_sats)
    ])
    prange_residuals = np.array(sat_prng) - predicted_ranges

    dx, cov, sigma0 = wls_solve(H, W, prange_residuals, valid_idx)

    NOMINAL_SIGMA = 3.0
    if dx is None or sigma0 < 0.1 or sigma0 > 500.0:
        sigma0 = NOMINAL_SIGMA
        Hv = H[valid_idx]
        Wv = np.diag(np.diag(W)[valid_idx])
        HWH = Hv.T @ Wv @ Hv
        try:
            cov = NOMINAL_SIGMA**2 * np.linalg.inv(HWH)
        except np.linalg.LinAlgError:
            return None

    # ── Position ──
    pos_ecef = approx_ecef + dx[:3]
    lat_deg, lon_deg, _ = ecef_to_lla(pos_ecef[0], pos_ecef[1], pos_ecef[2])

    # ── ENU covariance ──
    cov_enu = cov_ecef_to_enu(cov[:3, :3], lat_deg, lon_deg)
    sde = math.sqrt(max(cov_enu[0, 0], 0))
    sdn = math.sqrt(max(cov_enu[1, 1], 0))
    sdne = cov_enu[0, 1]

    # ── HPL (same as _compute_epoch_pl) ──
    cov_h = cov[:2, :2]
    eigvals = np.linalg.eigvalsh(cov_h)
    sigma_major = math.sqrt(max(float(v) for v in eigvals))

    dof = len(valid_idx) - n_state
    if dof > 0:
        t_d = chi2.ppf(1.0 - P_FA, dof)
        lam = 1.0
        for _ in range(50):
            prob = ncx2.cdf(t_d, dof, lam)
            if abs(prob - P_MD) < 1e-4: break
            lam = lam * (1.2 if prob > P_MD else 0.8)
        k_h = math.sqrt(lam / dof) if dof > 0 else 6.0
    else:
        k_h = 6.0
    hpl = k_h * sigma_major

    return {
        "sow": round(epoch.gpst),
        "lon": lon_deg, "lat": lat_deg,
        "sde_m": sde, "sdn_m": sdn, "sdne_m": sdne,
        "n_sats": len(valid_idx), "sigma0": sigma0, "hpl_m": hpl,
    }


def compute_spp_positions(traj_name: str) -> List[Dict]:
    """Run SPP WLS on all RINEX epochs for one trajectory."""
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

    header, epochs = parse_rinex3_obs(obs_file)

    all_ephs = []
    for nf in nav_files:
        all_ephs.extend(parse_rinex3_nav(nf))
    eph_by_prn: Dict[str, list] = defaultdict(list)
    for eph in all_ephs:
        eph_by_prn[eph.prn].append(eph)

    # Approx position
    if header.approx_pos is not None:
        approx_ecef = header.approx_pos
    else:
        approx_ecef = lla_to_ecef(19.96, 110.48, 10.0)

    # ref_lla from approx_ecef
    lat_tmp, lon_tmp, _ = ecef_to_lla(approx_ecef[0], approx_ecef[1], approx_ecef[2])
    ref_lla = (lat_tmp, lon_tmp, 10.0)

    results = []
    skip = 0
    for epoch in epochs:
        r = compute_epoch_spp(epoch, eph_by_prn, approx_ecef, ref_lla)
        if r is not None:
            results.append(r)
        else:
            skip += 1

    print(f"    {len(results)} SPP epochs computed ({skip} skipped)")
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
    print(f"\n{'=' * 60}")
    print(f"  Trajectory: {traj_name}")
    print(f"{'=' * 60}")

    rtk_path = DATA_BASE / traj_name / "实时定位结果" / "rtk_solution_clean.txt"
    if not rtk_path.exists():
        print(f"  RTK file not found: {rtk_path}")
        return None

    rtk = parse_rtk_nmea(str(rtk_path))
    print(f"  RTK epochs: {len(rtk)}")

    spp = compute_spp_positions(traj_name)
    if not spp:
        print(f"  No SPP positions computed")
        return None

    # Time-align
    errors = []
    matched = 0
    for sp in spp:
        sow = sp["sow"]
        if sow not in rtk:
            continue
        rtk_lon, rtk_lat = rtk[sow]
        err_m = haversine_m(sp["lon"], sp["lat"], rtk_lon, rtk_lat)
        errors.append({
            "sow": sow,
            "err_m": err_m,
            "spp_lon": sp["lon"], "spp_lat": sp["lat"],
            "rtk_lon": rtk_lon, "rtk_lat": rtk_lat,
            "sde_m": sp["sde_m"], "sdn_m": sp["sdn_m"], "sdne_m": sp["sdne_m"],
            "n_sats": sp["n_sats"], "sigma0": sp["sigma0"], "hpl_m": sp["hpl_m"],
        })
        matched += 1

    print(f"  Matched: {matched}/{len(spp)}")

    if matched < 10:
        print(f"  WARNING: Too few matched epochs!")
        return None

    err_vals = np.array([e["err_m"] for e in errors])
    stats = {
        "traj": traj_name,
        "n_rtk": len(rtk), "n_spp": len(spp), "n_matched": matched,
        "mean": float(np.mean(err_vals)),
        "median": float(np.median(err_vals)),
        "rmse": float(np.sqrt(np.mean(err_vals ** 2))),
        "p68": float(np.percentile(err_vals, 68)),
        "p95": float(np.percentile(err_vals, 95)),
        "max": float(np.max(err_vals)),
        "std": float(np.std(err_vals)),
    }
    print(f"  Error: mean={stats['mean']:.1f}m, median={stats['median']:.1f}m, "
          f"RMSE={stats['rmse']:.1f}m, P68={stats['p68']:.1f}m, P95={stats['p95']:.1f}m")

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
        sows = [e["sow"] - res["errors"][0]["sow"] for e in res["errors"]]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        ax.plot(sows, errs, lw=0.3, color="#2166ac", alpha=0.8)
        ax.axhline(res["stats"]["mean"], color="red", lw=0.8, ls="--",
                   label=f"mean={res['stats']['mean']:.1f}m")
        ax.axhline(res["stats"]["p95"], color="orange", lw=0.8, ls=":",
                   label=f"P95={res['stats']['p95']:.1f}m")
        ax.set_xlabel("Time (s rel.)"); ax.set_ylabel("Horiz. error (m)")
        ax.set_title(f"Traj {traj}: Error Time Series"); ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
        ymax = max(res["stats"]["p95"] * 1.5, 50)
        ax.set_ylim(0, ymax)

        ax = axes[1]
        ax.hist(errs, bins=50, density=True, alpha=0.6, color="#2166ac",
                edgecolor="white", lw=0.3)
        for val, lbl, c in [(res["stats"]["median"], "median", "red"),
                             (res["stats"]["p95"], "P95", "orange")]:
            ax.axvline(val, color=c, lw=1, ls="--", label=f"{lbl}={val:.1f}m")
        ax.set_xlabel("Error (m)"); ax.set_ylabel("Density")
        ax.set_title(f"Traj {traj}: Error Distribution"); ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, res["stats"]["p95"] * 1.3)

        ax = axes[2]
        sdes = [e["sde_m"] for e in res["errors"]]
        ax.scatter(sdes, errs, s=2, alpha=0.3, color="#2166ac")
        ax.set_xlabel("SPP σ_E (m)"); ax.set_ylabel("Horiz. error (m)")
        ax.set_title(f"Traj {traj}: Error vs σ_E"); ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"spp_error_traj{traj.replace('.', '_')}.png",
                    dpi=DPI)
        plt.close(fig)

    # ── Combined summary ──
    all_errs = np.concatenate([np.array([e["err_m"] for e in r["errors"]])
                                for r in all_results])
    traj_labels = [r["stats"]["traj"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.hist(all_errs, bins=80, density=True, alpha=0.6, color="#2166ac",
            edgecolor="white", lw=0.3)
    for val, lbl, c in [(np.median(all_errs), "median", "red"),
                         (np.percentile(all_errs, 95), "P95", "orange")]:
        ax.axvline(val, color=c, lw=1.5, ls="--", label=f"{lbl}={val:.1f}m")
    ax.set_xlabel("Horizontal error (m)"); ax.set_ylabel("Density")
    ax.set_title(f"All: Error Distribution (n={len(all_errs)})")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, np.percentile(all_errs, 99))

    ax = axes[0, 1]
    sorted_err = np.sort(all_errs)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax.semilogx(sorted_err, cdf, lw=1.5, color="#2166ac")
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.axhline(0.95, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Error (m)"); ax.set_ylabel("CDF")
    ax.set_title("Cumulative Distribution"); ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    traj_data = [np.array([e["err_m"] for e in r["errors"]]) for r in all_results]
    bp = ax.boxplot(traj_data, labels=traj_labels, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#92c5de"); patch.set_alpha(0.7)
    ax.set_xlabel("Trajectory"); ax.set_ylabel("Error (m)")
    ax.set_title("Per-Trajectory Error Distribution")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    means = [r["stats"]["mean"] for r in all_results]
    medians = [r["stats"]["median"] for r in all_results]
    x = np.arange(len(traj_labels)); w = 0.35
    ax.bar(x - w/2, means, w, color="#2166ac", label="Mean", edgecolor="white", lw=0.5)
    ax.bar(x + w/2, medians, w, color="#b2182b", label="Median",
           edgecolor="white", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(traj_labels)
    ax.set_ylabel("Error (m)"); ax.set_title("Mean & Median per Trajectory")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("SPP vs RTK Positioning Error Analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "spp_error_summary.png", dpi=DPI)
    plt.close(fig)

    # ── Write CSVs ──
    csv_path = output_dir / "spp_error_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["traj", "n_rtk", "n_spp", "n_matched",
                                          "mean", "median", "rmse", "p68", "p95",
                                          "max", "std"])
        w.writeheader()
        for r in all_results:
            w.writerow(r["stats"])
    print(f"\n  Summary CSV: {csv_path}")

    all_path = output_dir / "spp_positions_all.csv"
    with open(all_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "sow", "spp_lon", "spp_lat",
                                          "rtk_lon", "rtk_lat", "err_m",
                                          "sde_m", "sdn_m", "sdne_m",
                                          "n_sats", "sigma0", "hpl_m"])
        w.writeheader()
        for res in all_results:
            tid = int(res["stats"]["traj"].replace(".", ""))
            for e in res["errors"]:
                w.writerow({"id": tid, "sow": e["sow"],
                            "spp_lon": e["spp_lon"], "spp_lat": e["spp_lat"],
                            "rtk_lon": e["rtk_lon"], "rtk_lat": e["rtk_lat"],
                            "err_m": e["err_m"],
                            "sde_m": e["sde_m"], "sdn_m": e["sdn_m"],
                            "sdne_m": e["sdne_m"],
                            "n_sats": e["n_sats"], "sigma0": e["sigma0"],
                            "hpl_m": e["hpl_m"]})
    print(f"  All SPP positions: {all_path} ({len(all_errs)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/output/spp_error"))
    args = parser.parse_args()

    args.output_dir = PROJECT_ROOT / args.output_dir

    if args.all:
        trajs = TRAJECTORIES
    elif args.traj:
        trajs = [args.traj]
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

    all_errs = np.concatenate([np.array([e["err_m"] for e in r["errors"]])
                                for r in all_results])
    print(f"\n{'=' * 60}")
    print(f"  Combined ({len(all_results)} trajectories, {len(all_errs)} epochs)")
    print(f"  Mean={np.mean(all_errs):.1f}m, Median={np.median(all_errs):.1f}m, "
          f"RMSE={np.sqrt(np.mean(all_errs**2)):.1f}m, "
          f"P68={np.percentile(all_errs, 68):.1f}m, "
          f"P95={np.percentile(all_errs, 95):.1f}m")

    plot_all(all_results, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
