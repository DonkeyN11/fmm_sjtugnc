#!/usr/bin/env python3
"""
Monte Carlo GNSS/CMM data setup script.

Features:
 - Multi-satellite DOP simulation per point (random sky geometry).
 - Sigma ladder 1..10 (10 trajectories each) with fixed sigma per trajectory.
 - Covariance trace-based binning linked to DOP.
 - Protection Level (PL) computed per point (chi-square multiplier).
 - Stratified calibration/test split preserving sigma distribution.
 - Outputs CSVs for trajectories and split indices plus quick summaries.
"""
from __future__ import annotations

import argparse
import math
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------- GNSS / DOP utilities --------------------------- #

def random_satellite_unit_vector(min_el_deg: float = 45.0) -> np.ndarray:
    """Generate a random unit vector representing satellite line-of-sight."""
    az = random.uniform(0, 2 * math.pi)
    el = random.uniform(math.radians(min_el_deg), math.radians(80))  # avoid horizon
    x = math.cos(el) * math.cos(az)
    y = math.cos(el) * math.sin(az)
    z = math.sin(el)
    return np.array([x, y, z])


def compute_dop(num_sats: int, min_el_deg: float = 45.0) -> Tuple[float, float, np.ndarray]:
    """
    Compute HDOP/PDOP from random satellite geometry.
    Returns (hdop, pdop, satellites[nsat,3]). If geometry is singular, fall back to a high DOP.
    """
    sats = np.stack([random_satellite_unit_vector(min_el_deg) for _ in range(num_sats)], axis=0)
    # Design matrix for least squares position (x, y, z, clock)
    G = np.hstack((sats, np.ones((num_sats, 1))))
    try:
        q = np.linalg.inv(G.T @ G)
        hdop = math.sqrt(q[0, 0] + q[1, 1])
        pdop = math.sqrt(q[0, 0] + q[1, 1] + q[2, 2])
    except np.linalg.LinAlgError:
        hdop = pdop = 20.0  # very poor geometry
    return hdop, pdop, sats


# --------------------------- Data model helpers ---------------------------- #

@dataclass
class TrajectorySpec:
    traj_id: int
    sigma: float  # pseudorange noise std (meters)

@dataclass
class SatelliteGeometry:
    traj_id: int
    hdop: float
    pdop: float
    azimuth_deg: List[float]
    elevation_deg: List[float]


def generate_trajectory(num_points: int, step_std: float = 5.0) -> np.ndarray:
    """Simple 2D random walk in meters."""
    steps = np.random.normal(scale=step_std, size=(num_points, 2))
    coords = np.cumsum(steps, axis=0)
    return coords


def wls_position(truth_pos: np.ndarray, sat_positions: np.ndarray, sigma: float, max_iter: int = 8) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Weighted LS position estimation from pseudoranges.
    Returns (estimated_pos[3], clock_bias, covariance_3x3).
    """
    num_sats = sat_positions.shape[0]
    if num_sats < 4:
        raise ValueError("Need at least 4 satellites for WLS")

    # Generate pseudoranges with noise
    true_ranges = np.linalg.norm(sat_positions - truth_pos, axis=1)
    noise = np.random.normal(scale=sigma, size=num_sats)
    rho = true_ranges + noise  # clock bias set to 0 in truth

    # Initialize at truth
    x = truth_pos.copy()
    b = 0.0
    for _ in range(max_iter):
        ranges = np.linalg.norm(sat_positions - x, axis=1)
        los = (sat_positions - x) / ranges[:, None]
        H = np.hstack((-los, np.ones((num_sats, 1))))
        r = rho - ranges - b

        try:
            Q = np.linalg.inv(H.T @ H)
        except np.linalg.LinAlgError:
            break
        dx = Q @ H.T @ r
        x += dx[:3]
        b += dx[3]
        if np.linalg.norm(dx[:3]) < 1e-4:
            break

    # Final covariance (assume sigma^2 per range)
    cov_full = (sigma ** 2) * Q if 'Q' in locals() else np.full((4, 4), np.nan)
    cov_xyz = cov_full[:3, :3]
    return x, b, cov_xyz


def protection_level_from_cov(cov: np.ndarray, integrity_risk: float = 1e-5) -> float:
    """
    Compute horizontal Protection Level assuming Gaussian error.
    Use chi-square factor for 2D at given integrity risk.
    """
    # Approximate chi-square inverse via scipy-free lookup for common risks
    chi_lookup = {1e-3: 13.82, 1e-4: 18.42, 1e-5: 23.03, 1e-6: 27.63}
    k = chi_lookup.get(integrity_risk, 23.03)
    sigma_major = math.sqrt(max(np.linalg.eigvalsh(cov)))
    return math.sqrt(k) * sigma_major


def bin_by_trace(trace_val: float, edges: List[float]) -> str:
    """Assign a bin label based on covariance trace."""
    for i, edge in enumerate(edges):
        if trace_val <= edge:
            return f"bin_{i}"
    return f"bin_{len(edges)}"


def az_el_from_vector(v: np.ndarray) -> Tuple[float, float]:
    """Return azimuth (deg 0-360) and elevation (deg) from a unit vector."""
    x, y, z = v
    az = math.degrees(math.atan2(y, x)) % 360.0
    el = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    return az, el


# ------------------------------ Main routine ------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate Monte Carlo GNSS/CMM setup data")
    parser.add_argument("--output_dir", default="output/monte_carlo_setup", help="Directory for generated CSVs")
    parser.add_argument("--num_points", type=int, default=100, help="Points per trajectory")
    parser.add_argument("--num_sats", type=int, default=8, help="Satellites used for DOP simulation")
    parser.add_argument("--calib_ratio", type=float, default=0.2, help="Calibration split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--plot_sky", action="store_true", help="Save a sky plot for the first trajectory's satellites")
    parser.add_argument("--plot_prefix", default="sky_plot", help="Filename prefix for sky plot (PNG)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare sigma ladder: 10 trajectories per sigma 1..10
    specs: List[TrajectorySpec] = []
    traj_id = 0
    for sigma in range(1, 11):
        for _ in range(10):
            specs.append(TrajectorySpec(traj_id=traj_id, sigma=float(sigma)))
            traj_id += 1

    rows = []
    truth_rows = []
    sat_rows: List[SatelliteGeometry] = []
    trace_values = []
    for spec in specs:
        coords = generate_trajectory(args.num_points)
        hdop, pdop, sats_unit = compute_dop(args.num_sats)
        orbit_radius = 20200000.0  # meters
        sat_positions = sats_unit * orbit_radius
        trace_val = None
        for idx, (x, y) in enumerate(coords):
            truth_pos = np.array([x, y, 0.0])
            est_pos, clock_bias, cov_xyz = wls_position(truth_pos, sat_positions, spec.sigma)
            cov_h = cov_xyz[:2, :2]
            trace_val = float(np.trace(cov_h))
            pl = protection_level_from_cov(cov_h)
            rows.append(
                {
                    "traj_id": spec.traj_id,
                    "point_idx": idx,
                    "x_m": est_pos[0],
                    "y_m": est_pos[1],
                    "z_m": est_pos[2],
                    "clock_bias_m": clock_bias,
                    "sigma_m": spec.sigma,
                    "hdop": hdop,
                    "pdop": pdop,
                    "cov_xx": cov_h[0, 0],
                    "cov_yy": cov_h[1, 1],
                    "cov_xy": cov_h[0, 1],
                    "cov_trace": trace_val,
                    "protection_level_m": pl,
                }
            )
            truth_rows.append({"traj_id": spec.traj_id, "point_idx": idx, "x_m": x, "y_m": y, "z_m": 0.0})
        az_list, el_list = zip(*[az_el_from_vector(v) for v in sats_unit])
        sat_rows.append(
            SatelliteGeometry(
                traj_id=spec.traj_id,
                hdop=hdop,
                pdop=pdop,
                azimuth_deg=list(az_list),
                elevation_deg=list(el_list),
            )
        )
        trace_values.append(trace_val)

    df = pd.DataFrame(rows)
    truth_df = pd.DataFrame(truth_rows)
    sat_df = pd.DataFrame(
        [
            {
                "traj_id": g.traj_id,
                "hdop": g.hdop,
                "pdop": g.pdop,
                "azimuth_deg": "|".join(f"{a:.3f}" for a in g.azimuth_deg),
                "elevation_deg": "|".join(f"{e:.3f}" for e in g.elevation_deg),
            }
            for g in sat_rows
        ]
    )

    # Bin by covariance trace (edges at quantiles)
    edges = list(np.quantile(trace_values, [0.25, 0.5, 0.75, 0.9]))
    df["cov_bin"] = df["cov_trace"].apply(lambda v: bin_by_trace(v, edges))

    # Stratified calibration/test split by sigma
    calib_ids = []
    test_ids = []
    for sigma, group in df.groupby("sigma_m")["traj_id"].unique().items():
        trajs = list(group)
        random.shuffle(trajs)
        split = max(1, int(len(trajs) * args.calib_ratio))
        calib_ids.extend(trajs[:split])
        test_ids.extend(trajs[split:])
    calib_ids = sorted(set(calib_ids))
    test_ids = sorted(set(test_ids))

    df["split"] = df["traj_id"].apply(lambda tid: "calibration" if tid in calib_ids else "test")

    # Save outputs
    df.to_csv(output_dir / "monte_carlo_points.csv", index=False)
    truth_df.to_csv(output_dir / "monte_carlo_truth.csv", index=False)
    sat_df.to_csv(output_dir / "satellite_geometry.csv", index=False)
    pd.DataFrame({"traj_id": calib_ids}).to_csv(output_dir / "calibration_trajs.csv", index=False)
    pd.DataFrame({"traj_id": test_ids}).to_csv(output_dir / "test_trajs.csv", index=False)
    # Aggregated noisy trajectories in cmm_trajectory.csv-style
    agg_rows = []
    for tid, g in df.sort_values("point_idx").groupby("traj_id"):
        coords = list(zip(g["x_m"].values, g["y_m"].values))
        wkt = "LINESTRING (" + ", ".join(f"{x:.8f} {y:.8f}" for x, y in coords) + ")"
        timestamps = g["point_idx"].astype(float).tolist()
        covs = []
        for _, row in g.iterrows():
            sde = math.sqrt(max(row["cov_xx"], 0.0))
            sdn = math.sqrt(max(row["cov_yy"], 0.0))
            sdne = row["cov_xy"]
            covs.append([sde, sdn, 0.0, sdne, 0.0, 0.0])
        protection_levels = g["protection_level_m"].astype(float).tolist()
        agg_rows.append(
            {
                "id": int(tid),
                "geom": wkt,
                "timestamps": json.dumps(timestamps),
                "covariances": json.dumps(covs),
                "protection_levels": json.dumps(protection_levels),
            }
        )
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(output_dir / "cmm_trajectory.csv", sep=";", index=False)

    if args.plot_sky and sat_rows:
        first = sat_rows[0]
        az = np.radians(first.azimuth_deg)
        el = np.radians(first.elevation_deg)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.scatter(az, (math.pi / 2) - el, c="tab:blue", edgecolors="white", s=80, alpha=0.85)
        ax.set_title(f"Sky Plot (traj {first.traj_id}, HDOP={first.hdop:.2f})")
        sky_path = output_dir / f"{args.plot_prefix}.png"
        plt.tight_layout()
        plt.savefig(sky_path, dpi=150)
        plt.close(fig)

    # Quick summaries
    summary = (
        df.groupby("sigma_m")["traj_id"]
        .nunique()
        .reset_index()
        .rename(columns={"traj_id": "num_trajs"})
    )
    summary.to_csv(output_dir / "sigma_counts.csv", index=False)

    print("Saved data to", output_dir)
    print("Sigma summary:\n", summary)
    print("Covariance trace bin edges:", edges)


if __name__ == "__main__":
    main()
